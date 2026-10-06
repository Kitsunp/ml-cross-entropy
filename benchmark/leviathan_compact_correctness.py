"""Synthetic correctness probe for the compact Leviathan spline candidate.

This is a functionality test only.  It compares the compact and dense
Leviathan Triton paths, and checks compact S=1 against the latest split-N
reduction.  It never represents pretraining validation or a throughput gate;
those remain the real 10-train/10-validation profiled runs.

The script requires a source manifest and a coherent package checkout. It
rejects a stale installed package before launching the diagnostic kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

_STAGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STAGING_ROOT))
from cut_cross_entropy.leviathan import forward_impl
from cut_cross_entropy.leviathan.backward_kernels import leviathan_backward_triton
from leviathan_candidate_provenance import verify_candidate


@dataclass
class ProbeConfig:
    vocab_size: int = 50_304
    hidden_size: int = 128
    generator_d_seed: int = 128
    generator_num_modes: int = 8
    generator_num_knots: int = 16
    generator_k: int = 3
    generator_krank: int = 64
    generator_spline_degree: int = 2
    dtype: torch.dtype = torch.bfloat16

    @property
    def b(self) -> int:
        return math.ceil(self.vocab_size ** (1.0 / self.generator_k))


_PARAMETER_NAMES = (
    "codebooks",
    "head_proj_weight",
    "head_norm_weight",
    "head_norm_bias",
    "head_spline_delta",
    "head_out_weight",
)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    delta = (actual.float() - expected.float()).double().reshape(-1)
    expected_flat = expected.float().double().reshape(-1)
    denom = torch.linalg.vector_norm(expected_flat).clamp_min(1e-30)
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / denom),
        "actual_nonfinite": int((~torch.isfinite(actual)).sum()),
    }


def _make_params(cfg: ProbeConfig, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    b = cfg.b
    d = cfg.generator_d_seed
    h = cfg.generator_num_modes
    kappa = cfg.generator_num_knots
    krank = cfg.generator_krank
    params = {
        "codebooks": torch.randn(
            cfg.generator_k, b, d, device=device, dtype=cfg.dtype, generator=generator
        ) * 0.02,
        "head_proj_weight": torch.randn(
            h, d, d, device=device, dtype=cfg.dtype, generator=generator
        ) * 0.02,
        "head_norm_weight": torch.ones(h, d, device=device, dtype=cfg.dtype),
        "head_norm_bias": torch.zeros(h, d, device=device, dtype=cfg.dtype),
        "head_spline_delta": torch.randn(
            h, d, kappa, krank, device=device, dtype=cfg.dtype, generator=generator
        ) * 0.1,
        "head_out_weight": torch.randn(
            h, krank, cfg.hidden_size, device=device, dtype=cfg.dtype, generator=generator
        ) * (0.02 / math.sqrt(h)),
        "knot_grid": torch.linspace(0.0, 1.0, kappa, device=device),
    }
    return params


def _clone_params(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in source.items()
    }


def _run(
    source: dict[str, torch.Tensor],
    cfg: ProbeConfig,
    ids: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    compact: bool,
    splits: int,
    fuse_chain: bool = True,
    seed_grad: torch.Tensor | None = None,
    block_m: int = 128,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    os.environ["LEV_DOT"] = "1"
    os.environ["LEV_COMPACT_SPLINE"] = "1" if compact else "0"
    os.environ["LEV_FUSE_CHAIN_DDELTA"] = "1" if fuse_chain else "0"
    os.environ["LEV_DDELTA_BM"] = str(block_m)
    os.environ["LEV_DDELTA_BD"] = "1"
    os.environ["LEV_DDELTA_BR"] = "64"
    os.environ["LEV_DDELTA_SPLITS"] = str(splits)
    params = _clone_params(source)
    output, saved = forward_impl.leviathan_forward(
        ids,
        params,
        cfg,
        save_intermediates=True,
    )
    if saved is None or bool(saved.get("compact_spline", False)) != compact:
        raise RuntimeError("correctness probe did not execute the requested candidate")
    grads = leviathan_backward_triton(
        grad_output,
        params,
        cfg,
        saved,
        ids,
        seed_grad=seed_grad,
    )
    if grads is None:
        raise RuntimeError("Leviathan Triton backward rejected the correctness probe")
    torch.cuda.synchronize(ids.device)
    return output.detach(), grads, saved


def _assert_error(metrics: dict[str, float | int], tolerance: float, name: str) -> None:
    if metrics["actual_nonfinite"] != 0 or not math.isfinite(metrics["relative_l2"]):
        raise AssertionError(f"non-finite candidate result: {name}")
    if metrics["relative_l2"] > tolerance:
        raise AssertionError(f"candidate parity failed: {name}: {metrics['relative_l2']}")


def _compare(
    output: torch.Tensor, grads: dict[str, torch.Tensor],
    expected: torch.Tensor, expected_grads: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result = {
        "output": _error(output, expected),
        "grads": {name: _error(grads[name], expected_grads[name]) for name in _PARAMETER_NAMES},
    }
    _assert_error(result["output"], 3e-4, "output")
    for name, error in result["grads"].items():
        _assert_error(error, 1e-3, name)
    return result


def run_probe(
    token_counts: list[int],
    seed: int,
    profile_path: Path | None = None,
    block_m: int = 128,
    checks: tuple[str, ...] = ("parity", "splits", "unfused", "seed"),
    output_path: Path | None = None,
    grid_storage: str = "fp32",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the compact correctness probe")
    if any(count <= 0 for count in token_counts):
        raise ValueError("token counts must be positive")
    cfg = ProbeConfig()
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    source = _make_params(cfg, device, seed)
    grid_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
                  "fp16": torch.float16}[grid_storage]
    source["knot_grid"] = source["knot_grid"].to(grid_dtype)
    report: dict[str, Any] = {
        "schema": "leviathan-compact-correctness-v2",
        "synthetic": True,
        "performance_gate": False,
        "latest_ddelta_split_reducer": True,
        "compact_policy": "LEV_COMPACT_SPLINE=1",
        "compact_block_m": block_m,
        "grid_storage": grid_storage,
        "seed": seed,
        "requested_token_counts": token_counts,
        "profile": {"requested": profile_path is not None, "diagnostic_only": True},
        "cases": [],
    }
    profile_context = nullcontext()
    if profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=True,
        )
    try:
        with profile_context as profiler:
            for token_count in token_counts:
                ids = torch.randint(cfg.vocab_size, (token_count,), device=device)
                grad_output = torch.randn(
                    token_count, cfg.hidden_size, device=device, dtype=torch.float32
                ).to(cfg.dtype)
                dense_output, dense_grads, _ = _run(
                    source, cfg, ids, grad_output, compact=False, splits=4
                )
                compact_output, compact_grads, _ = _run(
                    source, cfg, ids, grad_output, compact=True, splits=4, block_m=block_m,
                )
                case = {
                    "tokens": token_count, "compact_saved_flag": True,
                    "dense_vs_compact": _compare(compact_output, compact_grads,
                                                  dense_output, dense_grads),
                }
                report["cases"].append(case)
                for name, splits, fuse_chain in (
                    ("compact_s4_vs_s1", 1, True),
                    ("compact_fused_vs_unfused", 4, False),
                ):
                    if ("splits" if splits == 1 else "unfused") not in checks:
                        continue
                    output, grads, _ = _run(
                        source, cfg, ids, grad_output, compact=True, splits=splits,
                        fuse_chain=fuse_chain,
                        block_m=block_m,
                    )
                    case[name] = _compare(output, grads, compact_output, compact_grads)
                if "seed" not in checks:
                    continue
                # Isolate the shared seed contribution: zero embedding VJP
                # makes the expected codebook gradient an independent gather
                # VJP, which checks the future JTok/JTok-M gradient route.
                seed_grad = torch.randn(token_count, cfg.generator_d_seed, device=device)
                _, seed_grads, _ = _run(
                    source, cfg, ids, torch.zeros_like(grad_output), compact=True,
                    splits=4, seed_grad=seed_grad,
                    block_m=block_m,
                )
                from cut_cross_entropy.leviathan.compiler import _leviathan_seed_from_codebooks
                # Match production's FP32 scatter accumulation, followed by
                # one cast to parameter dtype; a BF16 gather VJP rounds each
                # contribution earlier and is a different numerical policy.
                codebooks = source["codebooks"].detach().float().requires_grad_()
                seed = _leviathan_seed_from_codebooks(ids, codebooks)
                expected_seed_grad = torch.autograd.grad((seed * seed_grad).sum(), codebooks)[0]
                expected_seed_grad = expected_seed_grad.to(source["codebooks"].dtype)
                case["shared_seed_gradient"] = _error(seed_grads["codebooks"], expected_seed_grad)
                _assert_error(case["shared_seed_gradient"], 1e-3, "shared seed gradient")
        report["status"] = "ok"
    except BaseException as error:
        report["status"] = "error"
        report["error_type"] = type(error).__name__
        raise
    finally:
        if profile_path is not None:
            profiler.export_chrome_trace(str(profile_path))
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[257, 513])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--block-m", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--grid-storage", choices=("fp32", "bf16", "fp16"), default="fp32",
                        help="Match the stored grid dtype, without regenerating its values")
    parser.add_argument("--checks", nargs="+", choices=("parity", "splits", "unfused", "seed"),
                        default=["parity", "splits", "unfused", "seed"])
    args = parser.parse_args()
    package_root = Path(forward_impl.__file__).resolve().parent.parent
    expected_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    verify_candidate(package_root, expected_manifest)
    report = run_probe(args.tokens, args.seed, args.profile, args.block_m,
                       tuple(args.checks), args.output, args.grid_storage)
    verify_candidate(package_root, expected_manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
