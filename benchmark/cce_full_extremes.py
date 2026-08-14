"""Full CCE forward/backward stress and step-cost probe."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from cut_cross_entropy import linear_cross_entropy


def _run_once(
    e_data: torch.Tensor,
    c_data: torch.Tensor,
    targets: torch.Tensor,
    *,
    mile: bool,
):
    e = e_data.detach().clone().requires_grad_(True)
    c = c_data.detach().clone().requires_grad_(True)
    loss, metrics = linear_cross_entropy(
        e,
        c,
        targets,
        shift=1,
        reduction="mean",
        impl="cce_kahan_full_c",
        mile_enabled=mile,
        mile_gamma=1.0,
        return_loss_metrics=True,
    )
    loss.backward()
    return loss, metrics, e.grad, c.grad


def _case(dtype: torch.dtype, scale: float, steps: int) -> dict[str, object]:
    rows, vocab, dim = 513, 32_768, 128
    generator = torch.Generator(device="cuda").manual_seed(20_260_814)
    e_data = torch.randn(rows, dim, generator=generator, device="cuda", dtype=dtype) * scale
    c_data = torch.randn(vocab, dim, generator=generator, device="cuda", dtype=dtype) * scale
    targets = torch.randint(0, vocab, (rows,), generator=generator, device="cuda")

    # Reference uses the exact stored BF16/FP16 values but accumulates in FP64.
    ref_logits = e_data[:-1].double() @ c_data.double().T
    ref_nll = F.cross_entropy(ref_logits, targets[1:], reduction="none")
    ref_mean = ref_nll.mean()
    del ref_logits, ref_nll

    for _ in range(2):
        outputs = _run_once(e_data, c_data, targets, mile=True)
    torch.cuda.synchronize()
    del outputs
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    times_ms: list[float] = []
    result = None
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _run_once(e_data, c_data, targets, mile=True)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    assert result is not None
    loss, metrics, de, dc = result
    return {
        "dtype": str(dtype),
        "scale": scale,
        "loss": float(loss.detach()),
        "reference_unweighted_loss": float(ref_mean),
        "unweighted_loss": float(metrics["ntp_ce_unweighted"]),
        "mile_delta": float(metrics["mile_reweighting_delta"]),
        "finite_loss": bool(torch.isfinite(loss)),
        "nonnegative_loss": bool(loss >= 0),
        "finite_de": bool(torch.isfinite(de).all()),
        "finite_dc": bool(torch.isfinite(dc).all()),
        "latency_ms_mean": statistics.mean(times_ms),
        "latency_ms_min": min(times_ms),
        "latency_ms_max": max(times_ms),
        "incremental_peak_bytes": torch.cuda.max_memory_allocated() - baseline,
    }


def _one_class(dtype: torch.dtype, scale: float) -> dict[str, object]:
    rows, dim = 257, 128
    e = torch.full((rows, dim), scale, device="cuda", dtype=dtype, requires_grad=True)
    c = torch.full((1, dim), scale, device="cuda", dtype=dtype, requires_grad=True)
    targets = torch.zeros(rows, device="cuda", dtype=torch.long)
    loss = linear_cross_entropy(
        e,
        c,
        targets,
        shift=1,
        reduction="none",
        impl="cce_exact",
    )
    loss.mean().backward()
    return {
        "dtype": str(dtype),
        "scale": scale,
        "loss_min": float(loss.min()),
        "loss_max": float(loss.max()),
        "finite": bool(torch.isfinite(loss).all()),
        "nonnegative": bool((loss >= 0).all()),
        "finite_de": bool(torch.isfinite(e.grad).all()),
        "finite_dc": bool(torch.isfinite(c.grad).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    report: dict[str, object] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
        "one_class": [],
    }
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        dtype_scales = (0.125, 1.0, 32.0) if dtype == torch.float16 else (0.125, 1.0, 1.0e4)
        for scale in dtype_scales:
            report["cases"].append(_case(dtype, scale, args.steps))
        for scale in dtype_scales:
            report["one_class"].append(_one_class(dtype, scale))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
