"""Exercise PolyNorm inside a training MLP with FP32, BF16, or TorchAO FP8 linears."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from torch import nn

from cut_cross_entropy.polynorm import _cute, polynorm, polynorm_reference


class TrainingMLP(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        dtype: torch.dtype,
        polynorm_backend: str,
    ) -> None:
        super().__init__()
        self.polynorm_backend = polynorm_backend
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.polynorm_weight = nn.Parameter(
            torch.full((3,), 1.0 / 3.0, dtype=dtype)
        )
        self.polynorm_bias = nn.Parameter(torch.zeros((1,), dtype=dtype))
        self.column = nn.Parameter(torch.ones((intermediate,), dtype=dtype))

    def forward(self, x: torch.Tensor, dropout_p: float) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.polynorm_backend == "cute":
            activation = polynorm(
                gate,
                self.polynorm_weight,
                self.polynorm_bias,
                dropout_p=dropout_p,
            )
        else:
            activation = polynorm_reference(
                gate,
                self.polynorm_weight,
                self.polynorm_bias,
            )
            activation = torch.nn.functional.dropout(
                activation, p=dropout_p, training=True
            )
        return self.down_proj((activation * up) * self.column)


class TrainingStack(nn.Module):
    def __init__(
        self,
        layers: int,
        hidden: int,
        intermediate: int,
        dtype: torch.dtype,
        polynorm_backend: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TrainingMLP(hidden, intermediate, dtype, polynorm_backend)
            for _ in range(layers)
        )

    def forward(self, x: torch.Tensor, dropout_p: float) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x, dropout_p)
        return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fp32", "bf16", "fp8"), required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument(
        "--polynorm-backend", choices=("cute", "reference"), default="cute"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-limit-gib", type=float, default=10.0)
    args = parser.parse_args()

    properties = torch.cuda.get_device_properties(0)
    fraction = min(args.memory_limit_gib * 1024**3 / properties.total_memory, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.manual_seed(123)
    torch.set_float32_matmul_precision("high")

    model_dtype = torch.float32 if args.mode == "fp32" else torch.bfloat16
    if args.layers < 1:
        raise ValueError("layers must be positive")
    model = TrainingStack(
        args.layers,
        args.hidden,
        args.intermediate,
        model_dtype,
        args.polynorm_backend,
    ).cuda().train()
    if args.mode == "fp8":
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        model = convert_to_float8_training(
            model,
            module_filter_fn=lambda module, _fqn: isinstance(module, nn.Linear),
            config=Float8LinearConfig(),
        )

    input_ = torch.randn(
        (args.batch, args.sequence, args.hidden),
        device="cuda",
        dtype=model_dtype,
    )
    learning_rate = 1.0e-4
    parameters = list(model.parameters())
    masters = [parameter.detach().float().clone() for parameter in parameters]
    first_parameter = next(model.parameters())
    initial_parameter = first_parameter.detach().clone()
    initial_master = masters[0].clone()

    callable_: nn.Module = model
    if args.compiled:
        callable_ = torch.compile(model, mode="max-autotune", fullgraph=True)

    autocast_enabled = args.mode in ("bf16", "fp8")

    def step() -> tuple[torch.Tensor, bool]:
        for parameter in parameters:
            parameter.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            output = callable_(input_, args.dropout_p)
            loss = output.float().square().mean()
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        with torch.no_grad():
            for parameter, master in zip(parameters, masters, strict=True):
                if parameter.grad is None:
                    continue
                master.add_(parameter.grad.float(), alpha=-learning_rate)
                parameter.copy_(master.to(parameter.dtype))
        return loss.detach(), gradients_finite

    for _ in range(args.warmup):
        if args.compiled:
            torch.compiler.cudagraph_mark_step_begin()
        loss, gradients_finite = step()
        if not bool(torch.isfinite(loss)) or not gradients_finite:
            raise RuntimeError("non-finite warmup training step")
    for parameter in parameters:
        parameter.grad = None
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    losses: list[float] = []
    finite_steps: list[bool] = []
    for _ in range(args.steps):
        if args.compiled:
            torch.compiler.cudagraph_mark_step_begin()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss, gradients_finite = step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        losses.append(loss.item())
        finite_steps.append(bool(torch.isfinite(loss)) and gradients_finite)

    parameter_delta = (first_parameter.detach() - initial_parameter).float().norm().item()
    master_delta = (masters[0] - initial_master).norm().item()
    float8_linear_count = sum(
        type(module).__name__ == "Float8Linear" for module in model.modules()
    )
    cute_kernel_cache_entries = len(getattr(_cute, "_CACHE", ()))
    counters = {
        name: dict(values)
        for name, values in torch._dynamo.utils.counters.items()
        if values
    }
    print(
        json.dumps(
            {
                "device": properties.name,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "mode": args.mode,
                "compiled": args.compiled,
                "shape": [args.batch, args.sequence, args.hidden],
                "intermediate": args.intermediate,
                "layers": args.layers,
                "dropout_p": args.dropout_p,
                "polynorm_backend": args.polynorm_backend,
                "steps": args.steps,
                "median_step_ms": statistics.median(times),
                "mean_step_ms": statistics.fmean(times),
                "min_step_ms": min(times),
                "max_step_ms": max(times),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "all_steps_finite": all(finite_steps),
                "parameter_delta_norm": parameter_delta,
                "fp32_master_delta_norm": master_delta,
                "float8_linear_count": float8_linear_count,
                "cute_kernel_cache_entries": cute_kernel_cache_entries,
                "incremental_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() - baseline_allocated
                ),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "baseline_reserved_bytes": baseline_reserved,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "incremental_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved() - baseline_reserved
                ),
                "test_memory_limit_gib": args.memory_limit_gib,
                "dynamo_counters": counters,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
