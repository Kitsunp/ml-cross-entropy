"""Profile a fixed-shape compiled CCE training step on a production-like geometry."""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from cut_cross_entropy import linear_cross_entropy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rows", type=int, default=32_705)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20_260_814)
    parser.add_argument("--memory-limit-gib", type=float)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.steps < 1 or args.warmup < 0:
        raise ValueError("steps must be positive and warmup non-negative")
    if args.memory_limit_gib is not None:
        if args.memory_limit_gib <= 0:
            raise ValueError("memory-limit-gib must be positive")
        properties = torch.cuda.get_device_properties(0)
        fraction = min(args.memory_limit_gib * 1024**3 / properties.total_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    e = (
        torch.randn(
            args.rows,
            args.dim,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / math.sqrt(args.dim)
    ).requires_grad_(True)
    c = torch.randn(
        args.vocab,
        args.dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    targets = torch.randint(
        0, args.vocab, (args.rows,), generator=generator, device="cuda"
    )

    def forward(
        embeddings: torch.Tensor,
        classifier: torch.Tensor,
        labels: torch.Tensor,
    ):
        return linear_cross_entropy(
            embeddings,
            classifier,
            labels,
            shift=1,
            reduction="mean",
            impl="cce_kahan_full_c",
            mile_enabled=True,
            mile_gamma=1.0,
            return_loss_metrics=True,
        )

    compiled = torch.compile(forward, fullgraph=True, mode="max-autotune")
    for _ in range(args.warmup):
        e.grad = None
        c.grad = None
        warmup_loss, warmup_metrics = compiled(e, c, targets)
        warmup_loss.backward()
    torch.cuda.synchronize()
    del warmup_loss, warmup_metrics
    e.grad = None
    c.grad = None
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    times_ms: list[float] = []
    loss = metrics = None
    for _ in range(args.steps):
        e.grad = None
        c.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss, metrics = compiled(e, c, targets)
        loss.backward()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    assert loss is not None and metrics is not None
    assert e.grad is not None and c.grad is not None
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compile_mode": "max-autotune",
        "steps": args.steps,
        "warmup": args.warmup,
        "seed": args.seed,
        "test_memory_limit_gib": args.memory_limit_gib,
        "rows": args.rows,
        "effective_tokens": args.rows - 1,
        "vocab": args.vocab,
        "dim": args.dim,
        "loss": float(loss.detach()),
        "unweighted_loss": float(metrics["ntp_ce_unweighted"]),
        "mile_delta": float(metrics["mile_reweighting_delta"]),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_e_grad": bool(torch.isfinite(e.grad).all()),
        "finite_c_grad": bool(torch.isfinite(c.grad).all()),
        "latency_ms_mean": statistics.mean(times_ms),
        "latency_ms_median": statistics.median(times_ms),
        "latency_ms_min": min(times_ms),
        "latency_ms_max": max(times_ms),
        "incremental_peak_bytes": torch.cuda.max_memory_allocated() - baseline_allocated,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "incremental_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() - baseline_reserved
        ),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
