"""Run repeated PolyNorm forward/backward checks without changing production policy."""

from __future__ import annotations

import argparse
import json
import time

import torch

from cut_cross_entropy.polynorm import polynorm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-limit-gib", type=float, default=10.0)
    args = parser.parse_args()
    if args.steps < 1 or args.check_every < 1:
        raise ValueError("steps and check-every must be positive")

    properties = torch.cuda.get_device_properties(0)
    if args.memory_limit_gib is not None:
        fraction = min(args.memory_limit_gib * 1024**3 / properties.total_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)

    torch.manual_seed(123)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(
        (args.rows, args.hidden), device="cuda", dtype=dtype, requires_grad=True
    )
    weight = torch.full(
        (3,), 1.0 / 3.0, device="cuda", dtype=dtype, requires_grad=True
    )
    bias = torch.zeros((1,), device="cuda", dtype=dtype, requires_grad=True)
    grad_output = torch.randn_like(x)

    def function(
        input_: torch.Tensor,
        weight_: torch.Tensor,
        bias_: torch.Tensor,
    ) -> torch.Tensor:
        return polynorm(
            input_, weight_, bias_, dropout_p=args.dropout_p
        )

    eager_output = function(x, weight, bias)
    eager_gradients = torch.autograd.grad(
        eager_output, (x, weight, bias), grad_output
    )
    del eager_output, eager_gradients

    callable_ = function
    if args.compiled:
        callable_ = torch.compile(function, mode="max-autotune", fullgraph=True)

    if args.compiled:
        torch.compiler.cudagraph_mark_step_begin()
    output = callable_(x, weight, bias)
    gradients = torch.autograd.grad(output, (x, weight, bias), grad_output)
    del output, gradients
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    checks: list[dict[str, float | int | bool]] = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        if args.compiled:
            torch.compiler.cudagraph_mark_step_begin()
        output = callable_(x, weight, bias)
        gradients = torch.autograd.grad(output, (x, weight, bias), grad_output)
        if step == 1 or step == args.steps or step % args.check_every == 0:
            finite = bool(torch.isfinite(output).all()) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            checks.append(
                {
                    "step": step,
                    "finite": finite,
                    "checksum": output.float().sum().item(),
                    "zero_fraction": (output == 0).float().mean().item(),
                }
            )
        del output, gradients
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    counters = {
        name: dict(values)
        for name, values in torch._dynamo.utils.counters.items()
        if values
    }
    print(
        json.dumps(
            {
                "device": properties.name,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "compiled": args.compiled,
                "shape": [args.rows, args.hidden],
                "dtype": args.dtype,
                "dropout_p": args.dropout_p,
                "steps": args.steps,
                "checks_performed": len(checks),
                "all_checks_finite": all(bool(check["finite"]) for check in checks),
                "distinct_checked_outputs": len(
                    {float(check["checksum"]) for check in checks}
                ),
                "zero_fraction_min": min(
                    float(check["zero_fraction"]) for check in checks
                ),
                "zero_fraction_max": max(
                    float(check["zero_fraction"]) for check in checks
                ),
                "elapsed_seconds_including_checks": elapsed,
                "incremental_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() - baseline_allocated
                ),
                "checks": checks,
                "dynamo_counters": counters,
                "test_memory_limit_gib": args.memory_limit_gib,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
