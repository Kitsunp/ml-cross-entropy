from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from cut_cross_entropy.mlp import _cute_gupn
from cut_cross_entropy.polynorm import polynorm_reference


def _reference(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    logits: torch.Tensor,
    column: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    gate = gate0 * gate_row
    activation = polynorm_reference(
        gate,
        weight,
        bias,
        exclusive_logits=logits,
    )
    if dropout_p:
        activation = torch.nn.functional.dropout(
            activation,
            p=dropout_p,
            training=True,
        )
    return activation * up * column


def _mark_step() -> None:
    marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def _time(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        _mark_step()
        output = function()
        output.sum().item()
        del output
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    samples: list[float] = []
    output_bytes = 0
    for _ in range(iterations):
        _mark_step()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        output_bytes = output.numel() * output.element_size()
        del output
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    incremental_allocated = max(peak_allocated - baseline_allocated, 0)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "output_bytes": output_bytes,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": incremental_allocated,
        "incremental_peak_excluding_output_bytes": max(
            incremental_allocated - output_bytes,
            0,
        ),
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_reserved_bytes": max(
            peak_reserved - baseline_reserved,
            0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the exclusive CuTe GUPN stage with max-autotune."
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--layout",
        choices=("plain", "xor", "both"),
        default="both",
    )
    parser.add_argument("--max-test-vram-gib", type=float, default=10.0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not _cute_gupn.is_available():
        raise RuntimeError("CUDA and nvidia-cutlass-dsl are required")
    if args.rows <= 0 or args.hidden <= 0 or args.hidden % 4:
        parser.error("rows must be positive and hidden must be positive/divisible by four")
    if not 0.0 <= args.dropout_p < 1.0:
        parser.error("dropout-p must be in [0, 1)")

    device = torch.device("cuda", torch.cuda.current_device())
    total_memory = torch.cuda.get_device_properties(device).total_memory
    if args.max_test_vram_gib > 0:
        torch.cuda.set_per_process_memory_fraction(
            min(args.max_test_vram_gib * 1024**3 / total_memory, 1.0),
            device,
        )
    torch.manual_seed(2026)
    gate0 = torch.randn(
        (args.rows, args.hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    up = torch.randn_like(gate0)
    gate_row = torch.randn(args.hidden, device=device, dtype=torch.bfloat16)
    weight = torch.randn(3, device=device, dtype=torch.bfloat16)
    bias = torch.randn(1, device=device, dtype=torch.bfloat16)
    logits = torch.randn(2, device=device, dtype=torch.bfloat16)
    column = torch.randn(args.hidden, device=device, dtype=torch.bfloat16)
    seeds = torch.tensor(
        [123456789, 987654321, 1122334455, 556677889],
        device=device,
        dtype=torch.int64,
    )

    def reference() -> torch.Tensor:
        return _reference(
            gate0,
            up,
            gate_row,
            weight,
            bias,
            logits,
            column,
            args.dropout_p,
        )

    compiled_reference = torch.compile(
        reference,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
    )
    results: dict[str, object] = {
        "shape": [args.rows, args.hidden],
        "dropout_p": args.dropout_p,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(device)
        ),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutlass_dsl": __import__("importlib.metadata").metadata.version(
            "nvidia-cutlass-dsl"
        ),
        "max_autotune": _time(
            compiled_reference,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }

    layouts = (
        (False, True)
        if args.layout == "both"
        else (args.layout == "xor",)
    )
    for use_xor in layouts:
        name = "cute_xor" if use_xor else "cute_plain"

        def cute_call(use_xor: bool = use_xor) -> torch.Tensor:
            return _cute_gupn.forward(
                gate0,
                up,
                gate_row,
                weight,
                bias,
                logits,
                column,
                seeds,
                dropout_p=args.dropout_p,
                use_xor=use_xor,
            )

        results[name] = _time(
            cute_call,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    baseline_ms = results["max_autotune"]["median_ms"]
    for name in ("cute_plain", "cute_xor"):
        if name in results:
            cute_ms = results[name]["median_ms"]
            results[name]["latency_reduction_percent"] = (
                baseline_ms - cute_ms
            ) / baseline_ms * 100.0
            results[name]["speedup_x"] = baseline_ms / cute_ms

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
