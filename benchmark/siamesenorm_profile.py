"""Reproducible paired-boundary audit; not a real-training success claim."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from cut_cross_entropy.siamesenorm import siamesenorm, siamesenorm_reference


def errors(actual, expected):
    difference = actual.float() - expected.float()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_rms": (
            difference.square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-12)
        ).item(),
    }


def numerical(
    backend,
    dtype,
    width,
    outer=True,
    scale=True,
    magnitude=1.0,
    rows=17,
    affine_dtype=None,
    scale_dtype=None,
):
    torch.manual_seed(318)
    affine_dtype = dtype if affine_dtype is None else affine_dtype
    scale_dtype = affine_dtype if scale_dtype is None else scale_dtype
    values = [
        torch.randn(rows, width, device="cuda", dtype=dtype) * magnitude,
        torch.randn(rows, width, device="cuda", dtype=dtype) * magnitude,
        torch.randn(width, device="cuda", dtype=affine_dtype) * 0.2 + 1,
        torch.randn(width, device="cuda", dtype=affine_dtype) * 0.2 + 1,
        torch.randn(width, device="cuda", dtype=affine_dtype) * 0.2 + 1 if outer else None,
        torch.randn(width, device="cuda", dtype=scale_dtype) * 0.2 + 1 if scale else None,
    ]
    values = [v.requires_grad_() if v is not None else None for v in values]
    copies = [v.detach().clone().requires_grad_() if v is not None else None for v in values]
    a = siamesenorm(*values, backend=backend)
    b = siamesenorm_reference(*copies)
    g = torch.randn_like(b)
    a.backward(g)
    b.backward(g)
    return {
        "dtype": str(dtype),
        "affine_dtype": str(affine_dtype),
        "scale_dtype": str(scale_dtype),
        "width": width,
        "rows": rows,
        "outer": outer,
        "scale": scale,
        "magnitude": magnitude,
        "output": errors(a, b),
        "grads": [errors(v.grad, c.grad) for v, c in zip(values, copies) if v is not None],
        "finite": bool(
            torch.isfinite(a).all()
            and all(torch.isfinite(v.grad).all() for v in values if v is not None)
        ),
    }


def measure(backend, rows, width, steps, output):
    from torch._dynamo.utils import counters

    torch.manual_seed(318)
    inputs = [
        torch.randn(rows, width, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(2)
    ]
    inputs += [
        (torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).requires_grad_()
        for _ in range(4)
    ]
    grad = torch.randn_like(inputs[0])
    counters.clear()
    # Match train.py's global cache policy without dropping any max-autotune
    # mode options (including CUDA Graphs and coordinate-descent tuning).
    torch._inductor.config.force_disable_caches = True
    compiled = torch.compile(
        lambda *args: siamesenorm(*args, backend=backend),
        fullgraph=True,
        mode="max-autotune",
    )

    def step():
        torch.compiler.cudagraph_mark_step_begin()
        for x in inputs:
            x.grad = None
        compiled(*inputs).backward(grad)

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    warmed = {
        "graphs": counters["stats"]["unique_graphs"],
        "breaks": sum(counters["graph_break"].values()),
    }
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    timers = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(steps)
    ]
    for begin, end in timers:
        begin.record()
        step()
        end.record()
    torch.cuda.synchronize()
    latencies = [a.elapsed_time(b) for a, b in timers]
    result = {
        "backend": backend,
        "scope": "paired_norm_operator_forward_backward",
        "rows": rows,
        "width": width,
        "steps": steps,
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.mean(latencies),
        "allocated_baseline": baseline,
        "allocated_peak": torch.cuda.max_memory_allocated(),
        "reserved_peak": torch.cuda.max_memory_reserved(),
        "warmed": warmed,
        "after": {
            "graphs": counters["stats"]["unique_graphs"],
            "breaks": sum(counters["graph_break"].values()),
        },
    }
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(3):
            step()
    prof.export_chrome_trace(str(output / f"{backend}.trace.json"))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["reference", "triton", "cute"], required=True)
    p.add_argument("--rows", type=int, default=32768)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--numerical-only", action="store_true")
    p.add_argument(
        "--skip-numerical",
        action="store_true",
        help="Reuse the recorded numerical audit for unchanged source",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for dtype in () if args.skip_numerical else (torch.float32, torch.bfloat16):
            for outer, scale in ((True, True), (False, True), (True, False), (False, False)):
                checks.append(numerical(args.backend, dtype, args.width, outer, scale))
            checks.append(numerical(args.backend, dtype, args.width, magnitude=0.0))
            checks.append(numerical(args.backend, dtype, args.width, magnitude=1e-5))
            checks.append(numerical(args.backend, dtype, args.width, rows=1025))
        if not args.skip_numerical:
            for outer, scale in ((True, True), (True, False), (False, True)):
                checks.append(
                    numerical(
                        args.backend,
                        torch.float32,
                        args.width,
                        outer,
                        scale,
                        rows=1025,
                        affine_dtype=torch.bfloat16,
                    )
                )
                checks.append(
                    numerical(
                        args.backend,
                        torch.bfloat16,
                        args.width,
                        outer,
                        scale,
                        rows=1025,
                        affine_dtype=torch.float32,
                    )
                )
    if checks:
        prof.export_chrome_trace(str(args.output / f"{args.backend}.numerical.trace.json"))
    result = {"checks": checks}
    # Record the failed audit before raising, so failures remain reproducible.
    (args.output / f"{args.backend}.json").write_text(json.dumps(result, indent=2))
    for check in checks:
        tolerance = (
            2e-5
            if all(check[k] == str(torch.float32) for k in ("dtype", "affine_dtype", "scale_dtype"))
            else 3e-3
        )
        if not check["finite"] or any(
            item["relative_rms"] >= tolerance for item in [check["output"], *check["grads"]]
        ):
            raise RuntimeError(f"Numerical gate failed: {check}")
    if not args.numerical_only:
        result["measurement"] = measure(
            args.backend, args.rows, args.width, args.steps, args.output
        )
    (args.output / f"{args.backend}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "backend": args.backend,
                "numerical_cases": len(checks),
                "all_finite": all(c["finite"] for c in checks),
                "worst_output_relative_rms": max(
                    (c["output"]["relative_rms"] for c in checks), default=None
                ),
                "measurement": result.get("measurement"),
            }
        )
    )


if __name__ == "__main__":
    main()
