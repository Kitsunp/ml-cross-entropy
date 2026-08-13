"""Reproducible latency and incremental-memory benchmark for patch CCE.

This benchmark limits only its own CUDA process.  The CCE implementation does
not impose a VRAM limit on training applications.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch
import triton

from cut_cross_entropy import linear_cross_entropy

IGNORE_INDEX = -100


def _measure(
    function: Callable[[], torch.Tensor],
    clear_gradients: Callable[[], None],
    warmup: int,
    repetitions: int,
) -> tuple[float, int]:
    for _ in range(warmup):
        clear_gradients()
        function()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends, strict=True):
        clear_gradients()
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    latency_ms = float(
        statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    )

    # Warmup gradients are persistent allocations and must not be part of the
    # baseline. Clear them and synchronize before resetting peak statistics.
    clear_gradients()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    clear_gradients()
    return latency_ms, peak - baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--max-test-vram-gib", type=float, default=10.0)
    parser.add_argument("--mile", action="store_true")
    parser.add_argument("--mu-loss", action="store_true")
    args = parser.parse_args()

    if min(args.rows, args.dim, args.vocab, args.patch_size, args.repetitions) < 1:
        parser.error("rows, dim, vocab, patch-size, and repetitions must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.max_test_vram_gib <= 0:
        parser.error("max-test-vram-gib must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_test_vram_gib / total_gib))
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    generator = torch.Generator(device="cuda").manual_seed(123)
    e = torch.randn(
        args.rows,
        args.dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
        requires_grad=args.training,
    )
    c = torch.randn(
        args.vocab,
        args.dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
        requires_grad=args.training,
    )
    bias = torch.randn(
        args.vocab,
        device="cuda",
        dtype=dtype,
        generator=generator,
        requires_grad=args.training,
    )
    patch_targets = torch.randint(
        args.vocab,
        (args.rows, args.patch_size),
        device="cuda",
        generator=generator,
    )
    token_targets = torch.full_like(patch_targets, IGNORE_INDEX)
    token_targets[:, 0] = patch_targets[:, 0]
    options = {
        "bias": bias,
        "impl": "cce_exact",
        "filter_eps": None,
        "mile_enabled": args.mile,
        "mu_loss_enabled": args.mu_loss,
    }

    def finish(loss: torch.Tensor) -> torch.Tensor:
        if args.training:
            loss.backward()
        return loss

    cases = {
        "base_one_target": lambda: finish(
            linear_cross_entropy(e, c, patch_targets[:, 0], **options)
        ),
        "patch_k_targets": lambda: finish(
            linear_cross_entropy(
                e,
                c,
                patch_targets,
                patch_training_enabled=True,
                **options,
            )
        ),
        "flag_on_token_phase": lambda: finish(
            linear_cross_entropy(
                e,
                c,
                token_targets,
                patch_training_enabled=True,
                **options,
            )
        ),
        "repeat_embedding_reference": lambda: finish(
            linear_cross_entropy(
                e.repeat_interleave(args.patch_size, dim=0),
                c,
                patch_targets.flatten(),
                **options,
            )
        ),
    }

    def clear_gradients() -> None:
        e.grad = None
        c.grad = None
        bias.grad = None

    results = {}
    for name, function in cases.items():
        latency_ms, incremental_peak_bytes = _measure(
            function, clear_gradients, args.warmup, args.repetitions
        )
        results[name] = {
            "latency_ms": latency_ms,
            "incremental_peak_bytes": incremental_peak_bytes,
        }

    base_ms = results["base_one_target"]["latency_ms"]
    base_memory = results["base_one_target"]["incremental_peak_bytes"]
    for result in results.values():
        result["latency_delta_vs_base_ms"] = result["latency_ms"] - base_ms
        result["memory_delta_vs_base_bytes"] = result["incremental_peak_bytes"] - base_memory

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "rows": args.rows,
                "dim": args.dim,
                "vocab": args.vocab,
                "patch_size": args.patch_size,
                "dtype": args.dtype,
                "training": args.training,
                "mile": args.mile,
                "mu_loss": args.mu_loss,
                "max_test_vram_gib": args.max_test_vram_gib,
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
