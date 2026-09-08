"""Model-free Leviathan -> JTok/JTok-M integration reproducer.

This benchmark deliberately does *not* import ``modeling_neollm.py``.  It
uses only the repository-owned Leviathan generator, the compiler-safe seed
bridge, and the public JTok dispatch.  That makes compiler and CUDA-Graph
failures reproducible even when the downstream model source is in another
repository or is not available in the test checkout.

The harness exercises a real training-shaped path::

    Leviathan baseline:
        token ids -> Triton Leviathan embedding -> loss -> backward -> AdamW
    JTok/JTok-M:
        token ids -> Triton Leviathan embedding + differentiable seed
                  -> Linear/LayerNorm seed coordinate
                  -> several JTok/JTok-M layers -> loss -> backward -> AdamW

Examples (run in the CUDA environment)::

    # Validate the model-free partitioned-compile path.
    python benchmark/leviathan_jtok_integration.py \
        --compiled --variant jtok --expected-status pass

    # The same graph as one full Dynamo graph.
    python benchmark/leviathan_jtok_integration.py \
        --compiled --fullgraph --variant jtok --expected-status pass

    # Keep partitioned compilation but disable CUDA graphs for this process.
    python benchmark/leviathan_jtok_integration.py \
        --compiled --disable-cudagraphs --variant jtok --expected-status pass

The optional ``benchmark/neo_llm_jtok.py`` runner is the one to use when the
downstream model files are available and the full CCE training graph must be
audited.  It records the model-level CUDA-Graph error without making those
files a dependency of this benchmark.

``--expected-status`` is useful for regression tracking: ``error`` returns
success only when the selected configuration fails, while ``pass`` returns
success only when it completes with finite outputs and gradients.  No
``modeling_neollm.py``, tokenizer, checkpoint, or global Inductor setting is
required or modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F

# ``python benchmark/<file>.py`` puts ``benchmark/`` rather than the
# repository root on ``sys.path``.  Make the runner self-contained instead of
# requiring callers to remember ``PYTHONPATH=.``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cut_cross_entropy.leviathan import (
    LeviathanConfig,
    LeviathanGenerator,
    jtok_apply,
    jtokm_apply,
    jtokm_auxiliary_loss,
    leviathan_embedding_compiler_safe,
    leviathan_embedding_with_seed_compiler_safe,
)

SEED = 1729
COMPILE_MODE = "max-autotune"


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mark_cudagraph_step_begin() -> None:
    marker = getattr(
        getattr(torch, "compiler", None),
        "cudagraph_mark_step_begin",
        None,
    )
    if marker is not None:
        marker()


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _set_memory_guard(limit_gib: float | None) -> None:
    """Limit only this benchmark process; never read the value in a kernel."""

    if limit_gib is None:
        return
    if limit_gib <= 0.0:
        raise ValueError("--memory-limit-gib must be positive")
    properties = torch.cuda.get_device_properties(0)
    fraction = min(limit_gib * 1024**3 / properties.total_memory, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)


class SeedCoordinate(nn.Module):
    """The model-independent Linear -> LayerNorm -> sigmoid seed bridge."""

    def __init__(self, d_seed: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_seed, d_seed, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_seed, dtype=dtype))
        self.norm_weight = nn.Parameter(torch.ones(d_seed, dtype=dtype))
        self.norm_bias = nn.Parameter(torch.zeros(d_seed, dtype=dtype))
        self.eps = 1e-6
        nn.init.eye_(self.weight)

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        shape = seed.shape[:-1]
        flat = seed.reshape(-1, seed.shape[-1])
        projected = F.linear(flat.to(self.weight.dtype), self.weight, self.bias).float()
        mean = projected.mean(dim=-1, keepdim=True)
        variance = projected.var(dim=-1, keepdim=True, unbiased=False)
        projected = (projected - mean) * torch.rsqrt(variance + self.eps)
        projected = projected * self.norm_weight.float() + self.norm_bias.float()
        projected = torch.sigmoid(0.5 * projected).clamp(0.0, 1.0)
        return projected.to(seed.dtype).reshape(*shape, seed.shape[-1])


class IntegrationLayer(nn.Module):
    """A decoder-like residual junction around one public JTok call."""

    def __init__(
        self,
        *,
        hidden: int,
        d_seed: int,
        knots: int,
        modes: int,
        experts: int,
        top_k: int,
        variant: str,
        backend: str,
        dtype: torch.dtype,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.backend = backend
        self.top_k = top_k
        self.hidden = hidden
        self.d_seed = d_seed
        self.experts = experts if variant == "jtokm" else 1
        self.num_modes = modes
        self.num_knots = knots
        self.residual_scale = residual_scale
        self.proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.norm = nn.LayerNorm(hidden, dtype=dtype)
        self.spline_coeff = nn.Parameter(
            torch.randn(self.experts, modes, d_seed, knots, dtype=dtype) * 0.15
        )
        self.W_out = nn.Parameter(
            torch.randn(self.experts, modes, hidden, dtype=dtype) * 0.2
        )
        self.W_res = nn.Parameter(
            torch.randn(self.experts, d_seed, hidden, dtype=dtype) * 0.2
        )
        self.scaler = nn.Parameter(torch.randn(hidden, dtype=dtype) * 0.1)
        self.router = (
            nn.Linear(hidden, self.experts, bias=False, dtype=dtype)
            if variant == "jtokm"
            else None
        )
        self.register_buffer(
            "knot_grid",
            torch.linspace(0.0, 1.0, knots, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        z_tilde: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        compute_aux: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        delta = self.proj(hidden)
        if self.variant == "jtok":
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
            return self.norm(hidden + update), None

        update, stats = jtokm_apply(
            delta,
            z_tilde,
            hidden,
            self.spline_coeff,
            self.W_out,
            self.W_res,
            self.scaler,
            self.router.weight if self.router is not None else self.scaler.new_empty(0),
            self.knot_grid,
            top_k=self.top_k,
            valid_mask=valid_mask,
            compute_aux=compute_aux,
            backend=self.backend,  # type: ignore[arg-type]
            residual_scale=self.residual_scale,
        )
        return self.norm(hidden + update), stats


class LeviathanJTokIntegration(nn.Module):
    """Self-contained multi-layer training graph for compiler regression tests."""

    def __init__(self, args: argparse.Namespace, backend: str) -> None:
        super().__init__()
        self.variant = args.variant
        self.backend = backend
        has_jtok = args.variant != "leviathan"
        self.experts = args.experts if args.variant == "jtokm" else 1
        self.top_k = args.top_k if args.variant == "jtokm" else 1
        self.aux_weight = args.aux_weight
        self.lev_config = LeviathanConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden,
            generator_d_seed=args.d_seed,
            generator_num_modes=args.generator_modes,
            generator_num_knots=args.generator_knots,
            generator_spline_degree=2,
            generator_k=args.generator_k,
            generator_krank=args.generator_rank,
            dtype=args.dtype,
        )
        self.generator = LeviathanGenerator(self.lev_config).cuda()
        self.coordinate = (
            SeedCoordinate(args.d_seed, args.dtype).cuda() if has_jtok else None
        )
        residual_scale = 1.0 / math.sqrt(2.0 * args.layers)
        self.layers = nn.ModuleList()
        if has_jtok:
            self.layers.extend(
                IntegrationLayer(
                    hidden=args.hidden,
                    d_seed=args.d_seed,
                    knots=args.knots,
                    modes=args.modes,
                    experts=args.experts,
                    top_k=args.top_k,
                    variant=args.variant,
                    backend=backend,
                    dtype=args.dtype,
                    residual_scale=residual_scale,
                ).cuda()
                for _ in range(args.layers)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...]]:
        params = {
            "codebooks": self.generator.codebooks,
            "head_proj_weight": self.generator.head_proj_weight,
            "head_norm_weight": self.generator.head_norm_weight,
            "head_norm_bias": self.generator.head_norm_bias,
            "head_spline_delta": self.generator.head_spline_delta,
            "head_out_weight": self.generator.head_out_weight,
        }
        if self.variant == "leviathan":
            hidden = leviathan_embedding_compiler_safe(
                input_ids,
                params,
                self.lev_config,
                self.generator.knot_grid,
            )
            return hidden, ()

        hidden, seed = leviathan_embedding_with_seed_compiler_safe(
            input_ids,
            params,
            self.lev_config,
            self.generator.knot_grid,
        )
        if self.coordinate is None:  # pragma: no cover - guarded by variant
            raise RuntimeError("JTok variants require a seed coordinate bridge")
        z_tilde = self.coordinate(seed)
        stats: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            hidden, layer_stats = layer(
                hidden,
                z_tilde,
                valid_mask,
                compute_aux=self.training and self.variant == "jtokm",
            )
            if layer_stats is not None:
                stats.append(layer_stats)
        return hidden, tuple(stats)


class LossWrapper(nn.Module):
    """Keep the compiler boundary tensor-only while retaining JTok-M loss."""

    def __init__(self, model: LeviathanJTokIntegration) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden, stats = self.model(input_ids, valid_mask)
        loss = hidden.float().square().mean()
        if self.model.variant == "jtokm":
            for item in stats:
                loss = loss + jtokm_auxiliary_loss(
                    item,
                    num_experts=self.model.experts,
                    top_k=self.model.top_k,
                    weight=self.model.aux_weight,
                )
        return loss


def _make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch, args.sequence),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )
    if not 0.0 < args.valid_ratio <= 1.0:
        raise ValueError("--valid-ratio must be in (0, 1]")
    if args.valid_ratio == 1.0:
        valid = torch.ones(
            args.batch,
            args.sequence,
            device="cuda",
            dtype=torch.bool,
        )
    else:
        base_length = max(1, round(args.sequence * args.valid_ratio))
        offsets = torch.arange(args.batch, device="cuda") % 5 - 2
        lengths = (base_length + offsets).clamp(1, args.sequence)
        positions = torch.arange(args.sequence, device="cuda").unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        input_ids = torch.where(valid, input_ids, torch.zeros_like(input_ids))
    return input_ids, valid


def _optimizer_step(
    callable_model: Callable[..., torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    compiled: bool,
) -> torch.Tensor:
    if compiled:
        _mark_cudagraph_step_begin()
    for parameter in model.parameters():
        parameter.grad = None
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    loss = callable_model(*inputs)
    loss.backward()
    if optimizer is not None:
        optimizer.step()
    return loss.detach().clone()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _seed_everything(args.seed)
    torch._dynamo.reset()
    if hasattr(torch._dynamo.utils, "counters"):
        torch._dynamo.utils.counters.clear()
    model = LeviathanJTokIntegration(args, args.backend).train()
    wrapper = LossWrapper(model).cuda().train()
    inputs = _make_inputs(args)
    optimizer = (
        torch.optim.AdamW(wrapper.parameters(), lr=0.0)
        if args.optimizer_step
        else None
    )
    callable_model: Callable[..., torch.Tensor] = wrapper
    compile_options: dict[str, object] | None = None
    if args.disable_cudagraphs:
        # Torch 2.14 rejects passing ``mode`` and ``options`` together.  The
        # mode's relevant choice is expressed explicitly only for this local
        # call; no Inductor config is changed process-wide.
        compile_options = {
            "max_autotune": True,
            "triton.cudagraphs": False,
        }
    if args.compiled:
        kwargs: dict[str, Any] = {"fullgraph": args.fullgraph}
        if compile_options is None:
            kwargs["mode"] = COMPILE_MODE
        else:
            kwargs["options"] = compile_options
        callable_model = torch.compile(wrapper, **kwargs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cold_started = time.perf_counter()
    cold_loss = _optimizer_step(
        callable_model,
        wrapper,
        optimizer,
        inputs,
        compiled=args.compiled,
    )
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_started) * 1000.0
    cold_peak_allocated = torch.cuda.max_memory_allocated()
    cold_peak_reserved = torch.cuda.max_memory_reserved()

    for _ in range(args.warmup):
        _optimizer_step(
            callable_model,
            wrapper,
            optimizer,
            inputs,
            compiled=args.compiled,
        )
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    elapsed_ms: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = _optimizer_step(
            callable_model,
            wrapper,
            optimizer,
            inputs,
            compiled=args.compiled,
        )
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
        losses.append(float(loss.float().cpu()))

    finite = all(
        torch.isfinite(parameter).all().item()
        for parameter in wrapper.parameters()
        if parameter.requires_grad
    ) and all(math.isfinite(value) for value in losses)
    return {
        "status": "pass" if finite else "nonfinite",
        "seed": args.seed,
        "backend": args.backend,
        "variant": args.variant,
        "compiled": args.compiled,
        "fullgraph": args.fullgraph if args.compiled else None,
        "disable_cudagraphs": args.disable_cudagraphs,
        "inductor_cache_dir": (
            str(args.inductor_cache_dir) if args.inductor_cache_dir is not None else None
        ),
        "compile_mode": COMPILE_MODE if args.compiled else None,
        "optimizer_step": args.optimizer_step,
        "shape": [args.batch, args.sequence, args.hidden],
        "layers": args.layers,
        "leviathan_geometry": {
            "vocab_size": args.vocab_size,
            "d_seed": args.d_seed,
            "modes": args.generator_modes,
            "knots": args.generator_knots,
            "k": args.generator_k,
            "rank": args.generator_rank,
        },
        "jtok_geometry": {
            "modes": args.modes,
            "knots": args.knots,
            "experts": args.experts if args.variant == "jtokm" else 1,
            "top_k": args.top_k if args.variant == "jtokm" else 1,
        },
        "cold_ms": cold_ms,
        "stable_ms": _summary(elapsed_ms),
        "loss": {
            "cold": float(cold_loss.float().cpu()),
            "first": losses[0],
            "last": losses[-1],
            "all_finite": all(math.isfinite(value) for value in losses),
        },
        "memory": {
            "cold_peak_allocated_bytes": cold_peak_allocated,
            "cold_peak_reserved_bytes": cold_peak_reserved,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "stable_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "stable_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "incremental_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() - baseline_allocated
            ),
            "incremental_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() - baseline_reserved
            ),
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dynamo_counters": {
            str(name): {str(key): int(value) for key, value in values.items()}
            for name, values in getattr(torch._dynamo.utils, "counters", {}).items()
            if values
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("leviathan", "jtok", "jtokm"),
        default="jtok",
        help="Baseline Leviathan-only path or the opt-in JTok/JTok-M extension.",
    )
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--disable-cudagraphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass triton.cudagraphs=False only to this torch.compile call.",
    )
    parser.add_argument(
        "--expected-status",
        choices=("any", "pass", "error"),
        default="any",
        help="Regression expectation; does not change runtime behavior.",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-seed", type=int, default=128)
    parser.add_argument("--generator-modes", type=int, default=8)
    parser.add_argument("--generator-knots", type=int, default=16)
    parser.add_argument("--generator-k", type=int, default=3)
    parser.add_argument("--generator-rank", type=int, default=64)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--knots", type=int, default=16)
    parser.add_argument("--experts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--valid-ratio", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--aux-weight", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--optimizer-step",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    parser.add_argument(
        "--inductor-cache-dir",
        type=Path,
        default=None,
        help="Optional per-run TORCHINDUCTOR_CACHE_DIR for cold-cache reproduction.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.batch < 1 or args.sequence < 1 or args.hidden < 64 or args.layers < 1:
        raise ValueError("batch/sequence/layers must be positive and hidden >= 64")
    if args.hidden % 16 or args.vocab_size < 1:
        raise ValueError("hidden must be divisible by 16 and vocab-size positive")
    if args.d_seed < 32 or args.d_seed & (args.d_seed - 1):
        raise ValueError("d-seed must be a power of two >= 32")
    if args.generator_rank < 16 or args.generator_rank & (args.generator_rank - 1):
        raise ValueError("generator-rank must be a power of two >= 16")
    if args.generator_knots < 8 or args.generator_knots & (args.generator_knots - 1):
        raise ValueError("generator-knots must be a power of two >= 8")
    if args.modes < 1 or args.knots < 1:
        raise ValueError("JTok modes and knots must be positive")
    if args.experts < 1 or not 1 <= args.top_k <= args.experts:
        raise ValueError("top-k must be in [1, experts]")
    if not 0.0 < args.valid_ratio <= 1.0:
        raise ValueError("valid-ratio must be in (0, 1]")
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("warmup must be non-negative and steps positive")
    args.dtype = _dtype(args.dtype)
    if args.backend == "triton" and args.dtype == torch.float32:
        raise ValueError("Triton JTok/Leviathan requires bf16 or fp16")
    if args.fullgraph and not args.compiled:
        raise ValueError("--fullgraph requires --compiled")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    _set_memory_guard(args.memory_limit_gib)
    if args.inductor_cache_dir is not None:
        args.inductor_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.inductor_cache_dir)
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    try:
        report = _run(args)
    except Exception as exc:
        report = {
            "status": "error",
            "seed": args.seed,
            "backend": args.backend,
            "variant": args.variant,
            "compiled": args.compiled,
            "fullgraph": args.fullgraph if args.compiled else None,
            "disable_cudagraphs": args.disable_cudagraphs,
            "inductor_cache_dir": (
                str(args.inductor_cache_dir)
                if args.inductor_cache_dir is not None
                else None
            ),
            "expected_status": args.expected_status,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-24:],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    status = report["status"]
    if args.expected_status != "any":
        expected = args.expected_status
        if status != expected:
            raise SystemExit(
                f"Expected status {expected!r}, received {status!r}."
            )
    elif status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
