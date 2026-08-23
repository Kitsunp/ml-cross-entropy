"""Compare Leviathan backward implementations on a CUDA device.

This benchmark intentionally keeps the model precision policy used by the
training integration: BF16 parameters/activations, FP32 reductions, and
``torch.set_float32_matmul_precision("high")``.  It reports numerical error
for the output and every trainable Leviathan tensor before reporting timing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict
from typing import Any

import torch

from cut_cross_entropy.leviathan import (
    LeviathanConfig,
    LeviathanGenerator,
    leviathan_embedding,
    leviathan_forward_ref,
)

_PARAMETER_NAMES = (
    "codebooks",
    "head_proj_weight",
    "head_norm_weight",
    "head_norm_bias",
    "head_spline_delta",
    "head_out_weight",
)


def _set_mode(mode: str) -> None:
    if mode == "exact":
        os.environ["LEV_DOT"] = "0"
        os.environ.pop("LEV_PREMUL_DMM", None)
    elif mode == "dot":
        os.environ["LEV_DOT"] = "1"
        os.environ["LEV_PREMUL_DMM"] = "0"
    elif mode == "dot-premul-bf16":
        os.environ["LEV_DOT"] = "1"
        os.environ["LEV_PREMUL_DMM"] = "1"
    elif mode == "reference":
        os.environ["LEV_DOT"] = "0"
        os.environ.pop("LEV_PREMUL_DMM", None)
    else:
        raise ValueError(f"unknown mode: {mode}")


def _clone_params(
    source: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: source[name].detach().to(device).clone().requires_grad_()
        for name in _PARAMETER_NAMES
    }


def _run_once(
    mode: str,
    source: dict[str, torch.Tensor],
    cfg: LeviathanConfig,
    ids: torch.Tensor,
    grad_output: torch.Tensor,
) -> dict[str, Any]:
    _set_mode(mode)
    params = _clone_params(source, ids.device)
    params["knot_grid"] = source["knot_grid"].to(ids.device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    if mode == "reference":
        output, _ = leviathan_forward_ref(
            ids, params, cfg, save_intermediates=False
        )
    else:
        output = leviathan_embedding(ids, params, cfg)
    output.backward(grad_output)
    torch.cuda.synchronize()

    return {
        "elapsed_ms": (time.perf_counter() - start) * 1_000.0,
        "output": output.detach().float().cpu(),
        "grads": {
            name: params[name].grad.detach().float().cpu()
            for name in _PARAMETER_NAMES
        },
    }


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    actual64 = actual.double().reshape(-1)
    expected64 = expected.double().reshape(-1)
    delta = actual64 - expected64
    expected_norm = torch.linalg.vector_norm(expected64)
    actual_norm = torch.linalg.vector_norm(actual64)
    denom = expected_norm.clamp_min(torch.finfo(torch.float64).tiny)
    cosine_denom = (actual_norm * expected_norm).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / denom),
        "cosine": float(torch.dot(actual64, expected64) / cosine_denom),
        "actual_nonfinite": int((~torch.isfinite(actual64)).sum()),
        "expected_nonfinite": int((~torch.isfinite(expected64)).sum()),
    }


def _time_mode(
    mode: str,
    source: dict[str, torch.Tensor],
    cfg: LeviathanConfig,
    ids: torch.Tensor,
    grad_output: torch.Tensor,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    for _ in range(warmup):
        _run_once(mode, source, cfg, ids, grad_output)
    samples = [
        _run_once(mode, source, cfg, ids, grad_output)["elapsed_ms"]
        for _ in range(steps)
    ]
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[31, 257, 4096])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    cfg = LeviathanConfig(
        vocab_size=50_304,
        hidden_size=512,
        generator_d_seed=128,
        generator_num_modes=8,
        generator_num_knots=16,
        generator_k=3,
        generator_krank=64,
        dtype=torch.bfloat16,
    )
    generator = LeviathanGenerator(cfg)
    source = {
        name: getattr(generator, name).detach().clone()
        for name in _PARAMETER_NAMES
    }
    source["knot_grid"] = generator.knot_grid.detach().clone()

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "config": {
            key: str(value) if key == "dtype" else value
            for key, value in asdict(cfg).items()
        },
        "cases": [],
    }
    for tokens in args.tokens:
        ids = torch.randint(cfg.vocab_size, (tokens,), device=device)
        grad_output = (
            torch.randn(tokens, cfg.hidden_size, device=device, dtype=torch.float32)
            / cfg.hidden_size**0.5
        ).to(torch.bfloat16)

        exact = _run_once("exact", source, cfg, ids, grad_output)
        candidates = {
            "dot": _run_once("dot", source, cfg, ids, grad_output),
            "dot-premul-bf16": _run_once(
                "dot-premul-bf16", source, cfg, ids, grad_output
            ),
        }
        if args.include_reference and tokens <= 257:
            candidates["reference"] = _run_once(
                "reference", source, cfg, ids, grad_output
            )

        case: dict[str, Any] = {"tokens": tokens, "error_vs_exact": {}}
        for mode, result in candidates.items():
            errors = {"output": _error(result["output"], exact["output"])}
            errors.update(
                {
                    name: _error(result["grads"][name], exact["grads"][name])
                    for name in _PARAMETER_NAMES
                }
            )
            case["error_vs_exact"][mode] = errors

        if tokens >= 257:
            case["timing"] = {
                mode: _time_mode(
                    mode,
                    source,
                    cfg,
                    ids,
                    grad_output,
                    args.warmup,
                    args.steps,
                )
                for mode in ("exact", "dot", "dot-premul-bf16")
            }
        report["cases"].append(case)

    print("LEVIATHAN_BACKWARD_COMPARE=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
