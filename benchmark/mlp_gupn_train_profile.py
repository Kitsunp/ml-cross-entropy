from __future__ import annotations

import argparse
import json
import statistics

import torch

from cut_cross_entropy.mlp import gupn
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one isolated exclusive-GUPN training route. Run each "
            "implementation in a fresh process so CUDA Graph pools do not overlap."
        )
    )
    parser.add_argument(
        "--implementation",
        choices=("max_autotune", "cute"),
        required=True,
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-test-vram-gib", type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    total_memory = torch.cuda.get_device_properties(device).total_memory
    if args.max_test_vram_gib > 0:
        torch.cuda.set_per_process_memory_fraction(
            min(args.max_test_vram_gib * 1024**3 / total_memory, 1.0),
            device,
        )
    torch.manual_seed(2026)
    shape = (args.rows, args.hidden)
    gate0 = torch.randn(
        shape,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    up = torch.randn_like(gate0, requires_grad=True)
    gate_row = torch.randn(
        args.hidden,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        3,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    bias = torch.randn(
        1,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    logits = torch.randn(
        2,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    column = torch.randn(
        args.hidden,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    tensors = (gate0, up, gate_row, weight, bias, logits, column)

    if args.implementation == "cute":
        def function(
            gate: torch.Tensor,
            up_value: torch.Tensor,
            row: torch.Tensor,
            poly_weight: torch.Tensor,
            poly_bias: torch.Tensor,
            exclusive: torch.Tensor,
            down_column: torch.Tensor,
        ) -> torch.Tensor:
            return gupn(
                gate,
                up_value,
                row,
                poly_weight,
                poly_bias,
                exclusive,
                down_column,
                dropout_p=args.dropout_p,
            ).float().square().mean()
    else:
        def function(
            gate: torch.Tensor,
            up_value: torch.Tensor,
            row: torch.Tensor,
            poly_weight: torch.Tensor,
            poly_bias: torch.Tensor,
            exclusive: torch.Tensor,
            down_column: torch.Tensor,
        ) -> torch.Tensor:
            return _reference(
                gate,
                up_value,
                row,
                poly_weight,
                poly_bias,
                exclusive,
                down_column,
                args.dropout_p,
            ).float().square().mean()

    compiled = torch.compile(
        function,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
    )
    for _ in range(args.warmup):
        _mark_step()
        for tensor in tensors:
            tensor.grad = None
        loss = compiled(*tensors)
        loss.backward()
        del loss
    for tensor in tensors:
        tensor.grad = None
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    samples: list[float] = []
    losses: list[float] = []
    for _ in range(args.iterations):
        _mark_step()
        for tensor in tensors:
            tensor.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = compiled(*tensors)
        loss.backward()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        losses.append(float(loss.detach()))
        del loss

    result = {
        "implementation": args.implementation,
        "shape": list(shape),
        "dropout_p": args.dropout_p,
        "steps": args.iterations,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "incremental_peak_allocated_bytes": max(
            torch.cuda.max_memory_allocated(device) - baseline_allocated,
            0,
        ),
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "incremental_peak_reserved_bytes": max(
            torch.cuda.max_memory_reserved(device) - baseline_reserved,
            0,
        ),
        "final_loss": losses[-1],
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
