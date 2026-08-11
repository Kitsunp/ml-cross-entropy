"""Reproduce the direct CCE forward/backward latency measurements.

The memory limit applies only to this benchmark process. It does not change the
CCE implementation or impose a runtime limit on training.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=64_402)
    parser.add_argument("--valid-ratio", type=float, default=1.0)
    parser.add_argument("--objective", choices=("ce", "mile_mu"), default="mile_mu")
    parser.add_argument(
        "--matmul-precision", choices=("high", "highest"), default="high"
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        default=None,
        help="Optional limit for this benchmark process only",
    )
    parser.add_argument("--seed", type=int, default=20_260_811)
    return parser.parse_args()


ARGS = _parse_args()
sys.path.insert(0, str(ARGS.root.resolve()))

import torch
import triton

from cut_cross_entropy import linear_cross_entropy


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not 0.0 < ARGS.valid_ratio <= 1.0:
        raise ValueError("--valid-ratio must be in (0, 1]")

    generator = torch.Generator(device="cuda").manual_seed(ARGS.seed)
    e = torch.randn(
        ARGS.batch,
        ARGS.seq,
        ARGS.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    c = torch.randn(
        ARGS.vocab,
        ARGS.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    targets = torch.randint(
        ARGS.vocab,
        (ARGS.batch, ARGS.seq),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )

    if ARGS.valid_ratio < 1.0:
        predictions = ARGS.seq - 1
        usable = max(1, round(predictions * ARGS.valid_ratio))
        row_offsets = torch.arange(ARGS.batch, device="cuda") % 5 - 2
        lengths = (usable + 1 + row_offsets).clamp(1, ARGS.seq)
        positions = torch.arange(ARGS.seq, device="cuda").unsqueeze(0)
        targets = targets.masked_fill(positions >= lengths.unsqueeze(1), -100)

    e.requires_grad_(True)
    c.requires_grad_(True)
    valid = int((targets[:, 1:] != -100).sum().cpu())
    return e, c, targets, valid


def _loss(e: torch.Tensor, c: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    use_extensions = ARGS.objective == "mile_mu"
    return linear_cross_entropy(
        e,
        c,
        targets,
        shift=1,
        reduction="mean",
        impl="cce_kahan_full_c",
        mile_enabled=use_extensions,
        mile_gamma=1.0,
        mu_loss_enabled=use_extensions,
        mu_loss_lambda=1.0e-4,
    )


def _step(e: torch.Tensor, c: torch.Tensor, targets: torch.Tensor) -> None:
    e.grad = None
    c.grad = None
    _loss(e, c, targets).backward()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if ARGS.warmup < 0 or ARGS.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats must be positive")

    props = torch.cuda.get_device_properties(0)
    if ARGS.memory_limit_gib is not None:
        if ARGS.memory_limit_gib <= 0:
            raise ValueError("--memory-limit-gib must be positive")
        memory_fraction = min(ARGS.memory_limit_gib * 1024**3 / props.total_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    torch.set_float32_matmul_precision(ARGS.matmul_precision)
    e, c, targets, valid = _make_inputs()

    _step(e, c, targets)
    for _ in range(ARGS.warmup):
        _step(e, c, targets)
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    forward_ms: list[float] = []
    backward_ms: list[float] = []

    for _ in range(ARGS.repeats):
        e.grad = None
        c.grad = None
        start = torch.cuda.Event(enable_timing=True)
        forward_done = torch.cuda.Event(enable_timing=True)
        backward_done = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = _loss(e, c, targets)
        forward_done.record()
        loss.backward()
        backward_done.record()
        torch.cuda.synchronize()
        forward_ms.append(start.elapsed_time(forward_done))
        backward_ms.append(forward_done.elapsed_time(backward_done))

    total_ms = [fwd + bwd for fwd, bwd in zip(forward_ms, backward_ms, strict=True)]
    result = {
        "root": str(ARGS.root.resolve()),
        "device": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "shape": {
            "batch": ARGS.batch,
            "seq": ARGS.seq,
            "hidden": ARGS.hidden,
            "vocab": ARGS.vocab,
            "valid_predictions": valid,
        },
        "objective": ARGS.objective,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "forward_ms": _summary(forward_ms),
        "backward_ms": _summary(backward_ms),
        "total_ms": _summary(total_ms),
        "baseline_allocated_bytes": baseline_allocated,
        "incremental_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "test_memory_limit_gib": ARGS.memory_limit_gib,
        "finite": {
            "loss": bool(torch.isfinite(loss).cpu()),
            "e_grad": bool(torch.isfinite(e.grad).all().cpu()),
            "c_grad": bool(torch.isfinite(c.grad).all().cpu()),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
