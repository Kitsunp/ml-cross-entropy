from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from torchao.float8.float8_training_tensor import GemmInputRole

from cut_cross_entropy.mlp.fp8 import _CONFIG, _cast, _dual_forward


def _concatenated_projection(
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fan_fp8 = _cast(
        fan,
        dtype=_CONFIG.cast_config_input.target_dtype,
        role=GemmInputRole.INPUT,
    )
    gate_fp8 = _cast(
        gate_weight.t(),
        dtype=_CONFIG.cast_config_weight.target_dtype,
        role=GemmInputRole.WEIGHT,
    )
    up_fp8 = _cast(
        up_weight.t(),
        dtype=_CONFIG.cast_config_weight.target_dtype,
        role=GemmInputRole.WEIGHT,
    )
    weights = torch.cat((gate_fp8._data, up_fp8._data), dim=1)
    weights = weights.t().contiguous().t()
    scale_a = fan_fp8._scale.reciprocal().repeat(fan.shape[0]).reshape(-1, 1)
    width = gate_weight.shape[0]
    scale_b = torch.cat(
        (
            gate_fp8._scale.reciprocal().repeat(width),
            up_fp8._scale.reciprocal().repeat(width),
        )
    ).reshape(1, 2 * width)
    output = torch._scaled_mm(
        fan_fp8._data,
        weights,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    return output.split(width, dim=1)


def _mark_step() -> None:
    marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def _time(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        _mark_step()
        outputs = function()
        outputs[0].sum().item()
        del outputs
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        _mark_step()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del outputs
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two TorchAO tensorwise FP8 GEMMs with a concatenated "
            "mixed-scale performance oracle for the future CuTe dual kernel."
        )
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--fan-features", type=int, default=544)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-test-vram-gib", type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(args.max_test_vram_gib * 1024**3 / total, 1.0),
        device,
    )
    torch.manual_seed(2026)
    fan = torch.randn(
        args.rows,
        args.fan_features,
        device=device,
        dtype=torch.bfloat16,
    )
    gate_weight = torch.randn(
        args.intermediate,
        args.fan_features,
        device=device,
        dtype=torch.bfloat16,
    )
    up_weight = torch.randn_like(gate_weight)

    separate = torch.compile(
        _dual_forward,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
    )
    concatenated = torch.compile(
        _concatenated_projection,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
    )
    _mark_step()
    expected = tuple(
        value.clone()
        for value in separate(fan, gate_weight, up_weight)
    )
    _mark_step()
    actual = concatenated(fan, gate_weight, up_weight)
    relative_errors = [
        float(
            (actual_value.float() - expected_value.float()).norm()
            / expected_value.float().norm().clamp_min(1.0)
        )
        for actual_value, expected_value in zip(actual, expected, strict=True)
    ]
    del actual, expected
    results = {
        "shape": [args.rows, args.fan_features, args.intermediate],
        "separate": _time(
            lambda: separate(fan, gate_weight, up_weight),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "concatenated_oracle": _time(
            lambda: concatenated(fan, gate_weight, up_weight),
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "relative_l2_errors": relative_errors,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    baseline = results["separate"]["median_ms"]
    candidate = results["concatenated_oracle"]["median_ms"]
    results["concatenated_oracle"]["latency_reduction_percent"] = (
        (baseline - candidate) / baseline * 100.0
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
