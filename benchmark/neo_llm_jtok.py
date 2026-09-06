"""Multi-layer NeoLLM/JTok benchmark using the user's existing model source.

This harness is intentionally separate from ``modeling_neollm.py``. It loads
the configuration and model files supplied by the caller and creates a small
in-memory configuration for a reproducible training/inference run. The model
source owns the JTok backend dispatch; the harness only selects
``config.jtok_kernel_backend`` for each comparison.
The legacy Leviathan generator, tokenizer, checkpoint and persistent
configuration are never rewritten.

The Torch path calls the source model's native JTok/JTok-M implementation.
The Triton path calls the same model source with its strict CCE adapter
dispatch. Thus the comparison includes attention, MLP, residuals, Leviathan
coordinate creation, and the selected loss rather than comparing an isolated
surface operation.

The benchmark uses ``torch.compile(mode="max-autotune")`` only when
``--compiled`` is requested.  It does not change a process-global compiler
policy, optimizer configuration, MXFP8 state, tokenizer, or checkpoint.

Example in the existing remote environment::

    /venv/main/bin/python benchmark/neo_llm_jtok.py \
        --modeling-file /modeling_neollm.py \
        --configuration-file /configuration_neollm.py \
        --variant jtokm --backend both --compiled --optimizer-step

There is no VRAM limit by default.  ``--memory-limit-gib`` is an optional
process-level benchmark guard and is never read by the kernel.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import torch
import transformers
from torch import nn

SEED = 1729
COMPILE_MODE = "max-autotune"


def _seed_everything(seed: int) -> None:
    """Reset model and input RNGs before every backend copy."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mark_cudagraph_step_begin() -> None:
    """Mark one compiled invocation without changing Inductor policy."""

    marker = getattr(
        getattr(torch, "compiler", None),
        "cudagraph_mark_step_begin",
        None,
    )
    if marker is not None:
        marker()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _load_neo_modules(
    configuration_file: Path,
    modeling_file: Path,
) -> tuple[ModuleType, ModuleType]:
    """Load the two source files under their expected import names."""

    source_dir = str(configuration_file.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    # The supplied modeling file imports ``configuration_neollm`` by name.
    # Reloading is avoided because it can leave classes from two source files
    # mixed in one process; this executable is intended to run in a fresh
    # process for each source revision.
    configuration_spec = importlib.util.spec_from_file_location(
        "configuration_neollm", configuration_file
    )
    if configuration_spec is None or configuration_spec.loader is None:
        raise ImportError(f"Cannot load configuration source: {configuration_file}")
    configuration_module = importlib.util.module_from_spec(configuration_spec)
    sys.modules["configuration_neollm"] = configuration_module
    configuration_spec.loader.exec_module(configuration_module)

    modeling_spec = importlib.util.spec_from_file_location("modeling_neollm", modeling_file)
    if modeling_spec is None or modeling_spec.loader is None:
        raise ImportError(f"Cannot load modeling source: {modeling_file}")
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    sys.modules["modeling_neollm"] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)
    return configuration_module, modeling_module


def _make_config(
    config_class: type,
    args: argparse.Namespace,
    *,
    backend: str,
) -> Any:
    """Build a small in-memory NeoLLM config without touching disk state."""

    use_jtok = args.variant in {"jtok", "jtokm"}
    config = config_class(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_hidden_layers=args.layers,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.key_value_heads,
        head_dim=args.hidden // args.attention_heads,
        max_position_embeddings=max(args.sequence, 64),
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        ntp_loss_backend="cce",
        use_liger_kernel=False,
        cce_loss_impl="cce_kahan_full_c",
        use_mile_loss=False,
        use_mu_loss=False,
        use_meap=False,
        attention_bias=False,
        attention_dropout=0.0,
        dropout_rate=0.0,
        use_hola_memory=False,
        use_momentum_attention=False,
        use_mea_attention=False,
        use_lucid_attention=False,
        use_affine_scaled_attention=False,
        use_xsa=False,
        use_directional_routing=False,
        use_attn_res=False,
        use_stack_memory=False,
        use_fan_residual=False,
        use_learnable_multipliers=False,
        use_embedding_multipliers=False,
        use_lns=False,
        use_gpas=False,
        use_siamesenorm=False,
        use_embedding_input_norm=True,
        use_token_generator=True,
        generator_d_seed=args.d_seed,
        generator_num_modes=args.generator_modes,
        generator_num_knots=args.generator_knots,
        generator_spline_degree=2,
        generator_k=args.generator_k,
        generator_krank=args.generator_rank,
        use_jtok=use_jtok,
        use_jtokm=args.variant == "jtokm",
        jtok_num_modes=args.jtok_modes,
        jtok_num_experts=args.experts,
        jtok_top_k=args.top_k,
        jtok_norm_eps=1e-6,
        jtok_aux_loss_weight=args.aux_weight,
        jtok_kernel_backend=backend,
        use_hadamard_o_proj=False,
        polynorm_exclusive=False,
        use_spelling_bee_embeddings=False,
        use_repo=False,
        use_repo_grape=False,
        use_laurel=False,
        use_laurel_rw=False,
        use_laurel_lr=False,
        use_tweo=False,
        use_nitp=False,
        use_nitp_temporal=False,
        use_nextlat=False,
        use_iha=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        return_dict=True,
    )
    # This is an in-memory choice for the benchmark only.  It avoids relying
    # on an attention backend selected by a checkpoint's config.json.
    config._attn_implementation = "eager"
    # The comparison is only valid when both JTok backends share the same
    # Triton Leviathan producer.  This is runtime-only benchmark metadata; it
    # is not serialized into NeoLLM's config.json.
    config._require_leviathan_triton = True
    return config


class _LossOnly(nn.Module):
    """Keep the compiled boundary tensor-only while executing the full model."""

    def __init__(self, model: nn.Module, *, training_loss: bool):
        super().__init__()
        self.model = model
        self.training_loss = training_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.training_loss:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            if outputs.loss is None:
                raise RuntimeError("NeoLLM did not return a training loss")
            return outputs.loss
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if outputs.logits is None:
            raise RuntimeError("NeoLLM did not return inference logits")
        return outputs.logits.float().square().mean()


def _make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    input_ids = torch.randint(
        3,
        args.vocab_size,
        (args.batch, args.sequence),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )
    attention_mask = torch.ones(
        args.batch, args.sequence, device="cuda", dtype=torch.long
    )
    if not 0.0 < args.valid_ratio <= 1.0:
        raise ValueError("--valid-ratio must be in (0, 1]")
    if args.valid_ratio < 1.0:
        base_length = max(1, round(args.sequence * args.valid_ratio))
        row_offsets = torch.arange(args.batch, device="cuda") % 5 - 2
        lengths = (base_length + row_offsets).clamp(1, args.sequence)
        positions = torch.arange(args.sequence, device="cuda").unsqueeze(0)
        attention_mask = (positions < lengths.unsqueeze(1)).long()
        input_ids = torch.where(
            attention_mask.bool(),
            input_ids,
            torch.zeros_like(input_ids),
        )
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    return input_ids, attention_mask, labels


def _dynamo_counters() -> dict[str, dict[str, int]]:
    counters = getattr(torch._dynamo.utils, "counters", {})
    return {
        str(name): {str(key): int(value) for key, value in values.items()}
        for name, values in counters.items()
        if values
    }


def _set_memory_guard(limit_gib: float | None) -> None:
    if limit_gib is None:
        return
    if limit_gib <= 0.0:
        raise ValueError("--memory-limit-gib must be positive")
    properties = torch.cuda.get_device_properties(0)
    fraction = min(limit_gib * 1024**3 / properties.total_memory, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)


def _zero_grad(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _one_step(
    callable_model: Callable[..., torch.Tensor],
    model: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    training: bool,
    optimizer: torch.optim.Optimizer | None,
    mark_cudagraph_step: bool = False,
) -> torch.Tensor:
    input_ids, attention_mask, labels = inputs
    if mark_cudagraph_step:
        _mark_cudagraph_step_begin()
    if training:
        _zero_grad(model)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("neo_llm.training.forward"):
            loss = callable_model(input_ids, attention_mask, labels)
        with torch.profiler.record_function("neo_llm.training.backward"):
            loss.backward()
        if optimizer is not None:
            with torch.profiler.record_function("neo_llm.training.optimizer_step"):
                optimizer.step()
        # CUDA Graph Trees can recycle a user-visible scalar on the next
        # invocation.  Clone only the benchmark's retained report value,
        # outside the compiled model, instead of inserting a graph-boundary
        # marker into the model's training path.
        return loss.detach().clone()
    with torch.no_grad():
        with torch.profiler.record_function("neo_llm.inference.prefill"):
            return callable_model(input_ids, attention_mask, labels).detach()


def _measure_step(
    callable_model: Callable[..., torch.Tensor],
    model: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    mark_cudagraph_step: bool = False,
) -> tuple[float, float, float, torch.Tensor]:
    if mark_cudagraph_step:
        _mark_cudagraph_step_begin()
    start = torch.cuda.Event(enable_timing=True)
    forward_done = torch.cuda.Event(enable_timing=True)
    backward_done = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    input_ids, attention_mask, labels = inputs
    if args.mode == "inference":
        with torch.no_grad():
            with torch.profiler.record_function("neo_llm.inference.prefill"):
                loss = callable_model(input_ids, attention_mask, labels)
        forward_done.record()
        backward_done.record()
    else:
        _zero_grad(model)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("neo_llm.training.forward"):
            loss = callable_model(input_ids, attention_mask, labels)
        forward_done.record()
        with torch.profiler.record_function("neo_llm.training.backward"):
            loss.backward()
        backward_done.record()
        if optimizer is not None:
            with torch.profiler.record_function("neo_llm.training.optimizer_step"):
                optimizer.step()
    end.record()
    end.synchronize()
    return (
        start.elapsed_time(forward_done),
        forward_done.elapsed_time(backward_done),
        start.elapsed_time(end),
        loss.detach(),
    )


def _profile_once(
    callable_model: Callable[..., torch.Tensor],
    model: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    output_path: Path,
) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        _measure_step(
            callable_model,
            model,
            inputs,
            args=args,
            optimizer=optimizer,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(output_path))
    cuda_ops: list[dict[str, Any]] = []
    for event in profile.key_averages():
        device_name = str(getattr(event, "device_type", "")).lower()
        if "cuda" in device_name:
            cuda_ops.append(
                {
                    "name": event.key,
                    "calls": int(event.count),
                    "cuda_time_us": float(event.device_time_total),
                    "cuda_memory_bytes": int(getattr(event, "self_device_memory_usage", 0)),
                }
            )
    return {
        "trace": str(output_path),
        "cuda_operator_count": len(cuda_ops),
        "cuda_operators": sorted(
            cuda_ops,
            key=lambda item: item["cuda_time_us"],
            reverse=True,
        ),
    }


def _run_backend(
    args: argparse.Namespace,
    *,
    backend: str,
    configuration_module: ModuleType,
    modeling_module: ModuleType,
) -> dict[str, Any]:
    _seed_everything(args.seed)
    torch._dynamo.reset()
    if hasattr(torch._dynamo.utils, "counters"):
        torch._dynamo.utils.counters.clear()

    config = _make_config(
        configuration_module.NeoLLMConfig,
        args,
        backend=backend,
    )
    model = modeling_module.NeoLLMForCausalLM(config).cuda().to(dtype=args.dtype)
    model.train(args.mode == "training")
    token_generator = getattr(getattr(model, "model", model), "token_generator", None)
    leviathan_ready = bool(
        token_generator is not None
        and getattr(token_generator, "use_leviathan_triton", False)
        and getattr(modeling_module, "_LEV_KERNEL_AVAILABLE", False)
        and getattr(modeling_module, "_LEV_GEOMETRY_KERNEL_AVAILABLE", False)
    )
    if not leviathan_ready:
        raise RuntimeError(
            "The real NeoLLM comparison requires the Triton Leviathan kernel "
            "and its geometry adapter; refusing to benchmark a reference "
            "Leviathan fallback."
        )
    wrapper = _LossOnly(model, training_loss=args.mode == "training").cuda()
    inputs = _make_inputs(args)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=0.0)
        if args.optimizer_step and args.mode == "training"
        else None
    )
    callable_model: Callable[..., torch.Tensor] = wrapper
    if args.compiled:
        callable_model = torch.compile(wrapper, mode=COMPILE_MODE)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cold_start = time.perf_counter()
    cold_loss = _one_step(
        callable_model,
        model,
        inputs,
        training=args.mode == "training",
        optimizer=optimizer,
        mark_cudagraph_step=args.compiled,
    )
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1000.0
    cold_peak_allocated = torch.cuda.max_memory_allocated()
    cold_peak_reserved = torch.cuda.max_memory_reserved()

    for _ in range(args.warmup):
        _one_step(
            callable_model,
            model,
            inputs,
        training=args.mode == "training",
        optimizer=optimizer,
        mark_cudagraph_step=args.compiled,
    )
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    forward_ms: list[float] = []
    backward_ms: list[float] = []
    total_ms: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        forward, backward, total, loss = _measure_step(
            callable_model,
            model,
            inputs,
            args=args,
            optimizer=optimizer,
            mark_cudagraph_step=args.compiled,
        )
        forward_ms.append(forward)
        backward_ms.append(backward)
        total_ms.append(total)
        losses.append(float(loss.float().cpu()))

    trace = None
    if args.profile:
        trace = _profile_once(
            callable_model,
            model,
            inputs,
            args=args,
            optimizer=optimizer,
            output_path=args.profile_dir / f"neo_{args.variant}_{backend}_{args.mode}.json.gz",
        )

    jtok_module = next(
        (module for module in model.modules() if hasattr(module, "jtok_kernel_backend")),
        None,
    )
    return {
        "backend": backend,
        "model_jtok_kernel_backend": getattr(
            jtok_module,
            "jtok_kernel_backend",
            None,
        ),
        "leviathan": {
            "required": True,
            "kernel_ready": leviathan_ready,
            "reference_fallback_forbidden": True,
            "geometry_adapter_available": bool(
                getattr(modeling_module, "_LEV_GEOMETRY_KERNEL_AVAILABLE", False)
            ),
        },
        "compiled": args.compiled,
        "compile_mode": COMPILE_MODE if args.compiled else None,
        "mode": args.mode,
        "variant": args.variant,
        "shape": [args.batch, args.sequence, args.hidden],
        "layers": args.layers,
        "attention_heads": args.attention_heads,
        "d_seed": args.d_seed,
        "experts": args.experts,
        "top_k": args.top_k,
        "valid_ratio": args.valid_ratio,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "cold_ms": cold_ms,
        "forward_ms": _summary(forward_ms),
        "backward_ms": _summary(backward_ms),
        "total_ms": _summary(total_ms),
        "loss": {
            "cold": float(cold_loss.float().cpu()),
            "stable_first": losses[0],
            "stable_last": losses[-1],
            "all_finite": all(math.isfinite(value) for value in losses),
        },
        "memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "incremental_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() - baseline_allocated
            ),
            "incremental_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() - baseline_reserved
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cold_peak_allocated_bytes": cold_peak_allocated,
            "cold_peak_reserved_bytes": cold_peak_reserved,
        },
        "profile": trace,
        "dynamo_counters": _dynamo_counters(),
        "cuda_device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-file", type=Path, default=Path("/modeling_neollm.py"))
    parser.add_argument(
        "--configuration-file",
        type=Path,
        default=Path("/configuration_neollm.py"),
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "jtok", "jtokm"),
        default="jtokm",
        help="baseline=Leviathan only; jtok/jtokm enable the extension",
    )
    parser.add_argument("--backend", choices=("torch", "triton", "both"), default="both")
    parser.add_argument("--mode", choices=("inference", "training"), default="training")
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimizer-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--key-value-heads", type=int, default=2)
    parser.add_argument("--d-seed", type=int, default=8)
    parser.add_argument("--generator-modes", type=int, default=2)
    parser.add_argument("--generator-knots", type=int, default=5)
    parser.add_argument("--generator-k", type=int, default=2)
    parser.add_argument("--generator-rank", type=int, default=4)
    parser.add_argument("--jtok-modes", type=int, default=3)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--valid-ratio", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--aux-weight", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("benchmark/results/neo_jtok_profiles"),
    )
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.batch < 1 or args.sequence < 1 or args.layers < 1:
        raise ValueError("batch, sequence, and layers must be positive")
    if args.hidden < 1 or args.intermediate < 1 or args.vocab_size < 8:
        raise ValueError("hidden/intermediate/vocab-size must be positive")
    if args.attention_heads < 1 or args.hidden % args.attention_heads:
        raise ValueError("hidden must be divisible by attention-heads")
    if args.key_value_heads < 1 or args.attention_heads % args.key_value_heads:
        raise ValueError("key-value-heads must divide attention-heads")
    if args.d_seed < 1 or args.generator_modes < 1 or args.generator_knots < 1:
        raise ValueError("generator dimensions must be positive")
    if args.generator_k < 1 or args.generator_rank < 1 or args.jtok_modes < 1:
        raise ValueError("generator-k/rank and jtok-modes must be positive")
    if args.experts < 1 or not 1 <= args.top_k <= args.experts:
        raise ValueError("top-k must be in [1, experts]")
    if not 0.0 < args.valid_ratio <= 1.0:
        raise ValueError("valid-ratio must be in (0, 1]")
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("warmup must be non-negative and steps must be positive")
    args.dtype = _dtype(args.dtype)
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA; local CPU execution is not supported")
    _set_memory_guard(args.memory_limit_gib)
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    configuration_module, modeling_module = _load_neo_modules(
        args.configuration_file,
        args.modeling_file,
    )
    backends = ("torch", "triton") if args.backend == "both" else (args.backend,)
    results = []
    for backend in backends:
        results.append(
            _run_backend(
                args,
                backend=backend,
                configuration_module=configuration_module,
                modeling_module=modeling_module,
            )
        )
    report: dict[str, Any] = {
        "seed": args.seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "compile_mode": COMPILE_MODE,
        "memory_limit_gib_benchmark_only": args.memory_limit_gib,
        "source_files": {
            "modeling": str(args.modeling_file),
            "configuration": str(args.configuration_file),
        },
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
