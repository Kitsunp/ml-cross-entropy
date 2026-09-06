"""Reproducible JTok/JTok-M kernel benchmark and profiler.

The benchmark compares the same multi-layer module and weights through the
dense Torch reference and the external Triton kernel.  ``torch.compile`` is
local to the selected process and uses ``mode="max-autotune"``; no global
Inductor policy, optimizer setting, or model configuration is changed.

Examples (run in the existing CUDA environment)::

    python benchmark/leviathan_jtok.py --variant jtokm --backend torch --compiled
    python benchmark/leviathan_jtok.py --variant jtokm --backend triton --compiled
    python benchmark/leviathan_jtok.py --variant jtokm --backend both --mode training

The optional ``--memory-limit-gib`` is a benchmark-process guard only.  It is
``None`` by default and is never read by the kernel.  The default seed is
recorded in the JSON output so runs can be reproduced without relying on the
process-global RNG state.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

from cut_cross_entropy.leviathan import jtok_apply, jtokm_apply

SEED = 1729
COMPILE_MODE = "max-autotune"


def _seed_everything(seed: int) -> None:
    """Reset all RNGs used by a backend run so copies share exact weights."""

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


class JTokBlock(nn.Module):
    """One realistic MLP-to-JTok/JTok-M training block.

    The linear projection is included so the benchmark exercises gradient
    flow into the upstream layer, not only a detached standalone kernel.  The
    surface parameters are ordinary module parameters and are identical for
    the Torch and Triton copies.
    """

    def __init__(
        self,
        hidden: int,
        d_seed: int,
        knots: int,
        modes: int,
        experts: int,
        top_k: int,
        *,
        variant: str,
        backend: str,
        num_layers: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.backend = backend
        self.top_k = top_k
        self.d_seed = d_seed
        self.hidden = hidden
        self.mixture = variant == "jtokm"
        self.proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        surface_experts = experts if self.mixture else 1
        self.spline_coeff = nn.Parameter(
            torch.randn(surface_experts, modes, d_seed, knots, dtype=dtype) * 0.15
        )
        self.W_out = nn.Parameter(
            torch.randn(surface_experts, modes, hidden, dtype=dtype) * 0.2
        )
        self.W_res = nn.Parameter(
            torch.randn(surface_experts, d_seed, hidden, dtype=dtype) * 0.2
        )
        self.scaler = nn.Parameter(torch.randn(hidden, dtype=dtype) * 0.1)
        if self.mixture:
            self.router = nn.Linear(hidden, experts, bias=False, dtype=dtype)
        else:
            self.router = None
        self.register_buffer(
            "knot_grid",
            torch.linspace(0.0, 1.0, knots, dtype=torch.float32),
            persistent=False,
        )
        self.norm_eps = 1e-6
        self.jtokm_residual_scale = 1.0 / math.sqrt(2.0 * num_layers)

    def forward(
        self,
        x: torch.Tensor,
        z_tilde: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        with torch.profiler.record_function("jtok.mlp"):
            delta = self.proj(x)
        if not self.mixture:
            with torch.profiler.record_function("jtok.surface"):
                update = jtok_apply(
                    delta,
                    z_tilde,
                    self.spline_coeff,
                    self.W_out,
                    self.W_res,
                    self.scaler,
                    self.knot_grid,
                    valid_mask=valid_mask,
                    backend=self.backend,  # type: ignore[arg-type]
                )
            return x + update, None

        with torch.profiler.record_function("jtokm.router"):
            update, stats = jtokm_apply(
                delta,
                z_tilde,
                x,
                self.spline_coeff,
                self.W_out,
                self.W_res,
                self.scaler,
                self.router.weight,
                self.knot_grid,
                top_k=self.top_k,
                valid_mask=valid_mask,
                compute_aux=self.training,
                backend=self.backend,  # type: ignore[arg-type]
                residual_scale=self.jtokm_residual_scale,
            )
        return x + update, stats


class JTokStack(nn.Module):
    """Several coupled layers used by the real-training benchmark."""

    def __init__(
        self,
        *,
        layers: int,
        hidden: int,
        d_seed: int,
        knots: int,
        modes: int,
        experts: int,
        top_k: int,
        variant: str,
        backend: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            JTokBlock(
                hidden,
                d_seed,
                knots,
                modes,
                experts,
                top_k,
                variant=variant,
                backend=backend,
                num_layers=layers,
                dtype=dtype,
            )
            for _ in range(layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        z_tilde: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]]:
        stats: list[dict[str, torch.Tensor]] = []
        for layer_index, layer in enumerate(self.layers):
            with torch.profiler.record_function(f"jtok.layer.{layer_index}"):
                x, layer_stats = layer(x, z_tilde, valid_mask)
            if layer_stats is not None:
                stats.append(layer_stats)
        return x, tuple(stats)


def _make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    shape = (args.batch, args.sequence, args.hidden)
    x = torch.randn(shape, device="cuda", dtype=args.dtype, generator=generator) * 0.02
    z = torch.sigmoid(
        torch.randn(
            args.batch,
            args.sequence,
            args.d_seed,
            device="cuda",
            dtype=args.dtype,
            generator=generator,
        )
    )
    if not 0.0 < args.valid_ratio <= 1.0:
        raise ValueError("--valid-ratio must be in (0, 1]")
    if args.valid_ratio == 1.0:
        valid = torch.ones(args.batch, args.sequence, device="cuda", dtype=torch.bool)
    else:
        lengths = max(1, round(args.sequence * args.valid_ratio))
        row_offsets = torch.arange(args.batch, device="cuda") % 5 - 2
        lengths = (lengths + row_offsets).clamp(0, args.sequence)
        positions = torch.arange(args.sequence, device="cuda").unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
    return x, z, valid


def _make_model(args: argparse.Namespace, backend: str) -> JTokStack:
    return JTokStack(
        layers=args.layers,
        hidden=args.hidden,
        d_seed=args.d_seed,
        knots=args.knots,
        modes=args.modes,
        experts=args.experts,
        top_k=args.top_k,
        variant=args.variant,
        backend=backend,
        dtype=args.dtype,
    ).cuda().train(args.mode == "training")


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
    if limit_gib <= 0:
        raise ValueError("--memory-limit-gib must be positive")
    properties = torch.cuda.get_device_properties(0)
    fraction = min(limit_gib * 1024**3 / properties.total_memory, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)


def _zero_grad(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _auxiliary_loss(
    stats: tuple[dict[str, torch.Tensor], ...],
    *,
    experts: int,
    top_k: int,
    weight: float,
) -> torch.Tensor:
    if not stats:
        return torch.zeros((), device="cuda", dtype=torch.float32)
    losses = []
    for item in stats:
        p_sum = item["p_sum"]
        f_sum = item["f_sum"].to(device=p_sum.device, dtype=torch.float32)
        tokens = item["valid_tokens"].clamp_min(1.0)
        losses.append(float(experts) * (p_sum.float() / tokens * f_sum / (tokens * top_k)).sum())
    return torch.as_tensor(weight, device=losses[0].device, dtype=torch.float32) * torch.stack(losses).mean()


def _step(
    callable_model: Callable[..., tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]]],
    model: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    training: bool,
    aux_weight: float,
    experts: int,
    top_k: int,
    optimizer: torch.optim.Optimizer | None,
    mark_cudagraph_step: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    x, z, valid = inputs
    if mark_cudagraph_step:
        _mark_cudagraph_step_begin()
    if training:
        _zero_grad(model)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    with torch.profiler.record_function("training.forward" if training else "inference.prefill"):
        output, stats = callable_model(x, z, valid)
        loss = output.float().square().mean()
        if training and stats:
            loss = loss + _auxiliary_loss(
                stats,
                experts=experts,
                top_k=top_k,
                weight=aux_weight,
            )
    if training:
        with torch.profiler.record_function("training.backward"):
            loss.backward()
        if optimizer is not None:
            with torch.profiler.record_function("training.optimizer_step"):
                optimizer.step()
    # A compiled CUDA-Graph Tree may recycle the output storage on the next
    # invocation.  This snapshot is outside the compiled callable and keeps
    # the cold loss safe for the final report without adding a graph-boundary
    # marker or changing production training behavior.
    return loss.detach().clone(), stats[-1] if stats else None


def _measure_step(
    callable_model: Callable[..., tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]]],
    model: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    mark_cudagraph_step: bool = False,
) -> tuple[float, float, float, torch.Tensor, dict[str, torch.Tensor] | None]:
    if mark_cudagraph_step:
        _mark_cudagraph_step_begin()
    start = torch.cuda.Event(enable_timing=True)
    forward_done = torch.cuda.Event(enable_timing=True)
    backward_done = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if args.mode == "inference":
        with torch.no_grad():
            output, stats = callable_model(*inputs)
            loss = output.float().square().mean()
        forward_done.record()
        backward_done.record()
    else:
        _zero_grad(model)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        output, stats = callable_model(*inputs)
        loss = output.float().square().mean()
        if stats:
            loss = loss + _auxiliary_loss(
                stats,
                experts=args.experts,
                top_k=args.top_k,
                weight=args.aux_weight,
            )
        forward_done.record()
        loss.backward()
        backward_done.record()
        if optimizer is not None:
            optimizer.step()
    end.record()
    end.synchronize()
    return (
        start.elapsed_time(forward_done),
        forward_done.elapsed_time(backward_done),
        start.elapsed_time(end),
        loss.detach(),
        stats[-1] if stats else None,
    )


def _profile_once(
    callable_model: Callable[..., tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]]],
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
        _measure_step(callable_model, model, inputs, args=args, optimizer=optimizer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(output_path))
    cuda_ops: list[dict[str, Any]] = []
    for event in profile.key_averages():
        try:
            device_name = str(event.device_type).lower()
        except Exception:
            device_name = ""
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
        "cuda_operators": sorted(cuda_ops, key=lambda item: item["cuda_time_us"], reverse=True),
    }


def _run_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    # ``--backend both`` executes sequentially in one process.  Resetting the
    # RNG before each copy prevents initialization order from biasing the
    # Torch-vs-Triton comparison.
    _seed_everything(args.seed)
    torch._dynamo.reset()
    if hasattr(torch._dynamo.utils, "counters"):
        torch._dynamo.utils.counters.clear()
    model = _make_model(args, backend)
    inputs = _make_inputs(args)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=0.0)
        if args.optimizer_step and args.mode == "training"
        else None
    )
    callable_model: Callable[..., tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]]] = model
    compile_cold_ms: float | None = None
    if args.compiled:
        callable_model = torch.compile(model, mode=COMPILE_MODE, fullgraph=True)

    # The first invocation is kept separate: it includes Triton and/or
    # Inductor compilation, while stable timings below exclude cold start.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cold_start = time.perf_counter()
    if args.mode == "inference":
        with torch.no_grad():
            cold_loss, cold_stats = _step(
                callable_model,
                model,
                inputs,
                training=False,
                aux_weight=args.aux_weight,
                experts=args.experts,
                top_k=args.top_k,
                optimizer=None,
                mark_cudagraph_step=args.compiled,
            )
    else:
        cold_loss, cold_stats = _step(
            callable_model,
            model,
            inputs,
            training=True,
            aux_weight=args.aux_weight,
            experts=args.experts,
            top_k=args.top_k,
            optimizer=optimizer,
            mark_cudagraph_step=args.compiled,
        )
    torch.cuda.synchronize()
    compile_cold_ms = (time.perf_counter() - cold_start) * 1000.0
    cold_peak_allocated = torch.cuda.max_memory_allocated()
    cold_peak_reserved = torch.cuda.max_memory_reserved()

    for _ in range(args.warmup):
        if args.mode == "inference":
            with torch.no_grad():
                _step(
                    callable_model,
                    model,
                    inputs,
                    training=False,
                    aux_weight=args.aux_weight,
                    experts=args.experts,
                    top_k=args.top_k,
                    optimizer=None,
                    mark_cudagraph_step=args.compiled,
                )
        else:
            _step(
                callable_model,
                model,
                inputs,
                training=True,
                aux_weight=args.aux_weight,
                experts=args.experts,
                top_k=args.top_k,
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
    last_stats: dict[str, torch.Tensor] | None = cold_stats
    for _ in range(args.steps):
        fwd, bwd, total, loss, last_stats = _measure_step(
            callable_model,
            model,
            inputs,
            args=args,
            optimizer=optimizer,
            mark_cudagraph_step=args.compiled,
        )
        forward_ms.append(fwd)
        backward_ms.append(bwd)
        total_ms.append(total)
        losses.append(float(loss.cpu()))

    trace = None
    if args.profile:
        trace = _profile_once(
            callable_model,
            model,
            inputs,
            args=args,
            optimizer=optimizer,
            output_path=args.profile_dir / f"{args.variant}_{backend}_{args.mode}.json.gz",
        )

    def scalar(name: str) -> float | None:
        if last_stats is None or name not in last_stats:
            return None
        return float(last_stats[name].detach().float().cpu())

    return {
        "backend": backend,
        "compiled": args.compiled,
        "compile_mode": COMPILE_MODE if args.compiled else None,
        "mode": args.mode,
        "variant": args.variant,
        "shape": [args.batch, args.sequence, args.hidden],
        "layers": args.layers,
        "d_seed": args.d_seed,
        "knots": args.knots,
        "modes": args.modes,
        "experts": args.experts,
        "top_k": args.top_k,
        "valid_ratio": args.valid_ratio,
        "compile_cold_ms": compile_cold_ms,
        "forward_ms": _summary(forward_ms),
        "backward_ms": _summary(backward_ms),
        "total_ms": _summary(total_ms),
        "loss": {
            "cold": float(cold_loss.cpu()),
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
        "jtokm_metrics": {
            "valid_tokens": scalar("valid_tokens"),
            "active_experts": scalar("active_experts"),
            "load_cv": scalar("load_cv"),
            "load_entropy": scalar("load_entropy"),
            "router_entropy": scalar("router_entropy"),
            "max_load": scalar("max_load"),
            "min_load": scalar("min_load"),
            "invalid_fraction": scalar("invalid_fraction"),
        },
        "profile": trace,
        "dynamo_counters": _dynamo_counters(),
        "cuda_device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("jtok", "jtokm"), default="jtokm")
    parser.add_argument("--backend", choices=("torch", "triton", "both"), default="both")
    parser.add_argument("--mode", choices=("inference", "training"), default="training")
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--d-seed", type=int, default=32)
    parser.add_argument("--knots", type=int, default=9)
    parser.add_argument("--modes", type=int, default=5)
    parser.add_argument("--experts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--valid-ratio", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--aux-weight", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--optimizer-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-dir", type=Path, default=Path("benchmark/results/jtok_profiles"))
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.batch < 1 or args.sequence < 1 or args.layers < 1:
        raise ValueError("batch, sequence, and layers must be positive")
    if args.hidden < 1 or args.d_seed < 1 or args.knots < 1 or args.modes < 1:
        raise ValueError("hidden/d-seed/knots/modes must be positive")
    if args.experts < 1 or not 1 <= args.top_k <= args.experts:
        raise ValueError("top-k must be in [1, experts]")
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

    backends = ("torch", "triton") if args.backend == "both" else (args.backend,)
    results = []
    for backend in backends:
        results.append(_run_backend(args, backend))
    report: dict[str, Any] = {
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": getattr(__import__("triton"), "__version__", None),
        "compile_mode": COMPILE_MODE,
        "memory_limit_gib_benchmark_only": args.memory_limit_gib,
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
