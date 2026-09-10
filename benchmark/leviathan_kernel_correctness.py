"""Check split-N Leviathan backward correctness on CUDA.

This probe intentionally uses synthetic tensors only for functionality checks:
it compares the current S=1 kernel with split-N variants and never treats its
timings as a performance result.  Performance acceptance is performed by the
real 10-train/10-validation runner.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch

from cut_cross_entropy.leviathan import LeviathanConfig, LeviathanGenerator, autograd_fn

_PARAMETER_NAMES = (
    "codebooks",
    "head_proj_weight",
    "head_norm_weight",
    "head_norm_bias",
    "head_spline_delta",
    "head_out_weight",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[257, 513])
    parser.add_argument("--splits", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    delta = (actual.float() - expected.float()).double().reshape(-1)
    expected_flat = expected.float().double().reshape(-1)
    denom = torch.linalg.vector_norm(expected_flat).clamp_min(1e-30)
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / denom),
        "actual_nonfinite": int((~torch.isfinite(actual)).sum()),
    }


def _clone_params(
    source: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    params = {
        name: source[name].detach().to(device).clone().requires_grad_()
        for name in _PARAMETER_NAMES
    }
    params["knot_grid"] = source["knot_grid"].detach().to(device).clone()
    return params


def _run(
    source: dict[str, torch.Tensor],
    cfg: LeviathanConfig,
    ids: torch.Tensor,
    grad_output: torch.Tensor,
    splits: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    os.environ["LEV_DOT"] = "1"
    os.environ["LEV_FUSE_CHAIN_DDELTA"] = "1"
    os.environ["LEV_DDELTA_BM"] = "128"
    os.environ["LEV_DDELTA_BD"] = "1"
    os.environ["LEV_DDELTA_BR"] = "64"
    os.environ["LEV_DDELTA_SPLITS"] = str(splits)
    params = _clone_params(source, ids.device)
    output = autograd_fn.leviathan_apply(ids, params, cfg)
    output.backward(grad_output)
    return output.detach(), {
        name: params[name].grad.detach()
        for name in _PARAMETER_NAMES
    }


def run_probe(tokens: list[int], splits: list[int], seed: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Leviathan kernel probe")
    if any(value < 1 for value in tokens):
        raise ValueError("token counts must be positive")
    if any(value not in (2, 4, 8) for value in splits):
        raise ValueError("split counts must be one of 2, 4, 8")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = LeviathanConfig(
        vocab_size=50_304,
        hidden_size=128,
        generator_d_seed=128,
        generator_num_modes=8,
        generator_num_knots=16,
        generator_k=3,
        generator_krank=64,
        dtype=torch.bfloat16,
    )
    generator = LeviathanGenerator(cfg).cuda()
    source = {
        name: getattr(generator, name).detach().clone()
        for name in _PARAMETER_NAMES
    }
    source["knot_grid"] = generator.knot_grid.detach().clone()

    # Bypass the public fallback dispatcher for this functionality probe.  A
    # fallback here would make a false-positive split result look valid.
    if autograd_fn._leviathan_forward is None:
        raise RuntimeError("Leviathan Triton forward is unavailable")
    if autograd_fn._leviathan_backward_triton is None:
        raise RuntimeError("Leviathan Triton backward is unavailable")
    original_run_forward = autograd_fn._run_forward
    original_run_backward = autograd_fn._run_backward

    def strict_forward(ids, params, config, variant):
        return autograd_fn._leviathan_forward(
            ids,
            params,
            config,
            save_intermediates=True,
            variant=variant,
        )

    def strict_backward(ctx, grad_out, params, ids):
        grads = autograd_fn._leviathan_backward_triton(
            grad_out,
            params,
            ctx.cfg,
            ctx.saved_intermediates,
            ids,
        )
        if grads is None:
            raise RuntimeError("Leviathan Triton backward rejected the probe")
        return grads

    autograd_fn._run_forward = strict_forward
    autograd_fn._run_backward = strict_backward
    try:
        report: dict[str, Any] = {
            "schema": "leviathan-kernel-correctness-v1",
            "synthetic": True,
            "performance_gate": False,
            "config": {
                "hidden_size": cfg.hidden_size,
                "d_seed": cfg.generator_d_seed,
                "num_modes": cfg.generator_num_modes,
                "num_knots": cfg.generator_num_knots,
                "krank": cfg.generator_krank,
            },
            "cases": [],
        }
        device = torch.device("cuda")
        for token_count in tokens:
            ids = torch.randint(cfg.vocab_size, (token_count,), device=device)
            grad_output = torch.randn(
                token_count,
                cfg.hidden_size,
                device=device,
                dtype=torch.float32,
            ).to(torch.bfloat16)
            baseline_output, baseline_grads = _run(
                source, cfg, ids, grad_output, splits=1
            )
            case: dict[str, Any] = {
                "tokens": token_count,
                "baseline": "S=1",
                "variants": {},
            }
            for split_count in splits:
                output, grads = _run(
                    source, cfg, ids, grad_output, splits=split_count
                )
                case["variants"][f"S={split_count}"] = {
                    "output": _error(output, baseline_output),
                    "grads": {
                        name: _error(grads[name], baseline_grads[name])
                        for name in _PARAMETER_NAMES
                    },
                }
            report["cases"].append(case)
        return report
    finally:
        autograd_fn._run_forward = original_run_forward
        autograd_fn._run_backward = original_run_backward


def main() -> int:
    args = _parse_args()
    print(json.dumps(run_probe(args.tokens, args.splits, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
