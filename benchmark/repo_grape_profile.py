"""Compare fused REPO-GRAPE with its ``torch.compile(max-autotune)`` reference.

The optional memory cap applies only to this benchmark process. It never
changes dispatch or memory policy in the installed library.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import pathlib
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import cut_cross_entropy.repo_grape.triton as implementation  # noqa: E402
from cut_cross_entropy.repo_grape import repo_grape  # noqa: E402

TEST_PATH = ROOT / "tests" / "test_repo_grape.py"
SPEC = importlib.util.spec_from_file_location("repo_grape_test_reference", TEST_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TEST_PATH}")
reference_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference_module
SPEC.loader.exec_module(reference_module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--key-heads", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--rot-half", type=int, required=True)
    parser.add_argument("--sequence-pseudo-factor", type=int, choices=(1, 2), default=1)
    parser.add_argument("--input-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--fuse-norm", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--momentum-gamma", type=float, default=0.1)
    parser.add_argument("--output-dtype", choices=("bf16", "same"), default="bf16")
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--event-repeats", type=int, default=30)
    parser.add_argument("--memory-limit-gib", type=float, default=10.0)
    parser.add_argument("--forward-block", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--forward-warps", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--backward-block", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--backward-warps", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    properties = torch.cuda.get_device_properties(0)
    if args.memory_limit_gib is not None:
        fraction = min(args.memory_limit_gib * 1024**3 / properties.total_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)

    # Private selector overrides are benchmark-only geometry probes. Production
    # dispatch remains automatic and exposes no corresponding public flags.
    if args.forward_block is not None:
        implementation._select_forward_geometry = (
            lambda _sequence, _head_dim, *, has_rms_norm, supports_stream=False: args.forward_block
        )
    if args.forward_warps is not None:
        implementation._select_forward_num_warps = lambda _block, _head_dim, *, has_rms_norm: (
            args.forward_warps
        )
    if args.backward_block is not None:
        implementation._select_backward_geometry = lambda _batch, _sequence, _head_dim: (
            args.backward_block
        )
    if args.backward_warps is not None:
        implementation._select_backward_num_warps = (
            lambda _block, _head_dim, *, has_rms_norm=False: args.backward_warps
        )

    case = reference_module.Case(
        args.batch,
        args.query_heads,
        args.key_heads,
        args.sequence,
        args.head_dim,
        args.rot_half,
        args.sequence_pseudo_factor,
    )
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.input_dtype]
    inputs = reference_module._inputs(case, dtype=dtype)
    q, k, z, position_ids, inv_freq, alpha, log_scale, q_weight, k_weight = inputs
    differentiable = [q, k, z, alpha, log_scale]
    if args.fuse_norm:
        differentiable.extend((q_weight, k_weight))
    if not args.forward_only:
        for tensor in differentiable:
            tensor.requires_grad_(True)
    output_dtype = torch.bfloat16 if args.output_dtype == "bf16" else dtype
    torch.manual_seed(19)
    grad_outputs = (
        torch.randn_like(q, dtype=output_dtype),
        torch.randn_like(k, dtype=output_dtype),
    )

    def reference(q, k, z, position_ids, inv_freq, alpha, log_scale, qw, kw):
        return reference_module._reference(
            q,
            k,
            z,
            position_ids,
            inv_freq,
            alpha,
            log_scale,
            qw if args.fuse_norm else None,
            kw if args.fuse_norm else None,
            attention_scaling=1.0,
            momentum_gamma=args.momentum_gamma,
            rms_norm_eps=1.0e-6,
            sequence_pseudo_factor=case.sequence_pseudo_factor,
            output_dtype=output_dtype,
        )

    def kernel(q, k, z, position_ids, inv_freq, alpha, log_scale, qw, kw):
        return repo_grape(
            q,
            k,
            z,
            position_ids,
            inv_freq,
            alpha,
            log_scale,
            1.0,
            sequence_pseudo_factor=case.sequence_pseudo_factor,
            momentum_gamma=args.momentum_gamma,
            output_dtype=output_dtype,
            q_norm_weight=qw if args.fuse_norm else None,
            k_norm_weight=kw if args.fuse_norm else None,
            rms_norm_eps=1.0e-6,
        )

    compiled = {
        "reference": torch.compile(reference, mode="max-autotune", fullgraph=True),
        "kernel": torch.compile(kernel, mode="max-autotune", fullgraph=True),
    }

    def step(variant: str):
        if args.forward_only:
            with torch.no_grad():
                return compiled[variant](*inputs)
        outputs = compiled[variant](*inputs)
        return torch.autograd.grad(outputs, differentiable, grad_outputs)

    expected = tuple(value.detach().clone() for value in step("reference"))
    actual = tuple(value.detach().clone() for value in step("kernel"))
    torch.cuda.synchronize()
    max_abs = [
        (got.float() - want.float()).abs().max().item() for got, want in zip(actual, expected)
    ]
    relative_l2 = [
        ((got.float() - want.float()).norm() / want.float().norm().clamp_min(1.0e-12)).item()
        for got, want in zip(actual, expected)
    ]
    del expected, actual

    results: dict[str, dict[str, object]] = {}
    for variant in ("reference", "kernel"):
        for _ in range(10):
            values = step(variant)
        del values
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
        ) as profile:
            for _ in range(args.profile_steps):
                values = step(variant)
        del values
        torch.cuda.synchronize()
        averages = profile.key_averages()
        kernel_us = (
            sum(
                event.self_device_time_total
                for event in averages
                if event.key.startswith("## Call CompiledFxGraph")
            )
            / args.profile_steps
        )
        breakdown_us = {
            event.key: event.self_device_time_total / args.profile_steps
            for event in averages
            if event.self_device_time_total > 0
            and (
                event.key.startswith("## Call CompiledFxGraph")
                or event.key.startswith("_repo_grape_")
            )
        }

        samples = []
        for _ in range(args.event_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            values = step(variant)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)
            del values

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        values = step(variant)
        torch.cuda.synchronize()
        peak_increment = torch.cuda.max_memory_allocated() - baseline
        del values
        results[variant] = {
            "kernel_us": kernel_us,
            "event_us": statistics.median(samples),
            "peak_increment_bytes": peak_increment,
            "breakdown_us": breakdown_us,
        }

    payload: dict[str, object] = {
        "device": properties.name,
        "torch": torch.__version__,
        "triton": getattr(__import__("triton"), "__version__", "unknown"),
        "case": vars(args),
        "reference": results["reference"],
        "kernel": results["kernel"],
        "kernel_speedup": (
            float(results["reference"]["kernel_us"]) / float(results["kernel"]["kernel_us"])
        ),
        "event_speedup": (
            float(results["reference"]["event_us"]) / float(results["kernel"]["event_us"])
        ),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
    }
    if args.compact:
        payload = {
            "device": payload["device"],
            "torch": payload["torch"],
            "triton": payload["triton"],
            "case": payload["case"],
            "kernel_speedup": payload["kernel_speedup"],
            "event_speedup": payload["event_speedup"],
            "reference_kernel_us": results["reference"]["kernel_us"],
            "kernel_kernel_us": results["kernel"]["kernel_us"],
            "reference_peak_bytes": results["reference"]["peak_increment_bytes"],
            "kernel_peak_bytes": results["kernel"]["peak_increment_bytes"],
            "max_abs": payload["max_abs"],
            "relative_l2": payload["relative_l2"],
        }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
