"""Reproducible numerical and performance probes for the fused MiLe path."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics

import torch

from cut_cross_entropy.cce_mile import cce_mile_forward_kernel


def _call_mile(
    lse: torch.Tensor,
    mean_logit: torch.Tensor,
    nll: torch.Tensor,
    gamma: float,
    vocab_size: int,
):
    kwargs = {"return_unweighted_nll_sum": True}
    if "max_entropy" in inspect.signature(cce_mile_forward_kernel).parameters:
        kwargs["max_entropy"] = math.log(vocab_size)
    return cce_mile_forward_kernel(lse, mean_logit, nll, gamma, **kwargs)


def _reference(
    lse: torch.Tensor,
    mean_logit: torch.Tensor,
    gamma: float,
    vocab_size: int,
) -> torch.Tensor:
    entropy = (lse.double() - mean_logit.double()).clamp(0.0, math.log(vocab_size))
    if gamma == 0.0:
        return torch.ones_like(entropy, dtype=torch.float32)
    log_weight = gamma * torch.log1p(entropy)
    scaled = torch.exp(log_weight - log_weight.max())
    return (scaled / scaled.mean()).float()


def _measure_case(
    size: int,
    gamma: float,
    scenario: str,
    steps: int,
    vocab_size: int,
) -> dict[str, object]:
    if scenario == "equal_fp32_extreme":
        lse = torch.full((size,), 3.0e38, device="cuda", dtype=torch.float32)
        mean_logit = torch.full_like(lse, -3.0e38)
    elif scenario == "wide_dynamic_range":
        lse = torch.logspace(-20, 38, size, device="cuda", dtype=torch.float32)
        mean_logit = torch.zeros_like(lse)
    else:
        raise ValueError(scenario)
    nll = torch.linspace(0.0, 32.0, size, device="cuda", dtype=torch.float32)
    expected = _reference(lse, mean_logit, gamma, vocab_size)

    for _ in range(2):
        outputs = _call_mile(lse, mean_logit, nll, gamma, vocab_size)
    torch.cuda.synchronize()
    del outputs
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    times_ms: list[float] = []
    actual_weight = actual_loss = nll_sum = None
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        actual_weight, actual_loss, nll_sum = _call_mile(
            lse, mean_logit, nll, gamma, vocab_size
        )
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    assert actual_weight is not None and actual_loss is not None and nll_sum is not None
    finite_weight = bool(torch.isfinite(actual_weight).all())
    finite_loss = bool(torch.isfinite(actual_loss).all())
    max_abs_error = (
        float((actual_weight - expected).abs().max()) if finite_weight else float("inf")
    )
    return {
        "size": size,
        "gamma": gamma,
        "scenario": scenario,
        "finite_weight": finite_weight,
        "finite_loss": finite_loss,
        "nan_weights": int(torch.isnan(actual_weight).sum()),
        "inf_weights": int(torch.isinf(actual_weight).sum()),
        "weight_mean": float(actual_weight.mean()),
        "max_abs_weight_error": max_abs_error,
        "nll_sum_finite": bool(torch.isfinite(nll_sum)),
        "latency_ms_mean": statistics.mean(times_ms),
        "latency_ms_min": min(times_ms),
        "latency_ms_max": max(times_ms),
        "incremental_peak_bytes": torch.cuda.max_memory_allocated() - baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=151_936)
    args = parser.parse_args()
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
    }
    for size in (16_384, 16_385, 32_704):
        for gamma in (1.0, 2.0, 8.0, 32.0):
            for scenario in ("equal_fp32_extreme", "wide_dynamic_range"):
                report["cases"].append(
                    _measure_case(size, gamma, scenario, args.steps, args.vocab_size)
                )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
