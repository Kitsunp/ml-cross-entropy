from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from dataclasses import asdict, dataclass

import torch
from torch import nn

from cut_cross_entropy.mlp import fp8_gupn, gupn
from cut_cross_entropy.polynorm import polynorm_reference


@dataclass(frozen=True)
class Measurement:
    device: str
    compute_capability: str
    torch_version: str
    torch_cuda: str
    precision: str
    gupn_route: str
    compile_mode: str
    batch: int
    sequence_length: int
    hidden_size: int
    intermediate_size: int
    layers: int
    dropout_p: float
    steps: int
    median_step_ms: float
    mean_step_ms: float
    min_step_ms: float
    max_step_ms: float
    tokens_per_second: float
    baseline_allocated_bytes: int
    peak_allocated_bytes: int
    incremental_peak_allocated_bytes: int
    baseline_reserved_bytes: int
    peak_reserved_bytes: int
    incremental_peak_reserved_bytes: int


class ReferenceLayer(nn.Module):
    """Model-shaped oracle matching the current NeoLLM MLP module graph."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        fan_ratio: float,
        dropout_p: float,
        exclusive: bool,
        gupn_route: str,
    ) -> None:
        super().__init__()
        fan_size = hidden_size + int(hidden_size * fan_ratio)
        self.periodic_dim = int(fan_size * fan_ratio)
        fan_projection_size = fan_size - self.periodic_dim

        self.fan = nn.Linear(hidden_size, fan_projection_size, bias=True)
        self.gate = nn.Linear(fan_size, intermediate_size, bias=False)
        self.up = nn.Linear(fan_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_row_multiplier = nn.Parameter(torch.ones(intermediate_size))
        self.down_column_multiplier = nn.Parameter(torch.ones(intermediate_size))
        self.down_row_multiplier = nn.Parameter(torch.ones(hidden_size))
        self.polynorm_weight = nn.Parameter(torch.ones(3) / 3)
        self.polynorm_bias = nn.Parameter(torch.zeros(1))
        self.exclusive_logits = (
            nn.Parameter(torch.logit(torch.full((2,), 1.0e-4)))
            if exclusive
            else None
        )
        self.dropout_p = float(dropout_p)
        self.gupn_route = gupn_route

        nn.init.normal_(self.fan.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fan.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.fan(x)
        periodic = projected[..., : self.periodic_dim]
        passthrough = projected[..., self.periodic_dim :]
        fan = torch.cat(
            (torch.cos(periodic), torch.sin(periodic), passthrough),
            dim=-1,
        )

        if self.gupn_route == "fp8-cute":
            if self.exclusive_logits is None:
                raise RuntimeError("FP8 CuTe GUPN requires exclusive PolyNorm")
            down_input = fp8_gupn(
                fan,
                self.gate.weight,
                self.up.weight,
                self.gate_row_multiplier,
                self.polynorm_weight,
                self.polynorm_bias,
                self.exclusive_logits,
                self.down_column_multiplier,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            gate0 = self.gate(fan)
            up = self.up(fan)
        if self.gupn_route == "cute":
            if self.exclusive_logits is None:
                raise RuntimeError("CuTe GUPN requires exclusive PolyNorm")
            down_input = gupn(
                gate0,
                up,
                self.gate_row_multiplier,
                self.polynorm_weight,
                self.polynorm_bias,
                self.exclusive_logits,
                self.down_column_multiplier,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        elif self.gupn_route == "reference":
            gate = gate0 * self.gate_row_multiplier
            activation = polynorm_reference(
                gate,
                self.polynorm_weight,
                self.polynorm_bias,
                exclusive_logits=self.exclusive_logits,
            )
            if self.training and self.dropout_p:
                activation = torch.nn.functional.dropout(
                    activation,
                    p=self.dropout_p,
                    training=True,
                )
            down_input = activation * up * self.down_column_multiplier
        return self.down(down_input) * self.down_row_multiplier


class ReferenceStack(nn.Module):
    def __init__(self, layers: list[ReferenceLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    return torch.bfloat16


def _enable_torchao_fp8(model: nn.Module) -> None:
    from torchao.float8 import convert_to_float8_training

    def compatible_linear(module: nn.Module, _fqn: str) -> bool:
        return bool(
            isinstance(module, nn.Linear)
            and module.in_features % 16 == 0
            and module.out_features % 16 == 0
        )

    convert_to_float8_training(model, module_filter_fn=compatible_linear)


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    dtype = _dtype_for_precision(args.precision)
    layers = [
        ReferenceLayer(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            fan_ratio=args.fan_ratio,
            dropout_p=args.dropout_p,
            exclusive=not args.nonexclusive,
            gupn_route=args.gupn_route,
        )
        for _ in range(args.layers)
    ]
    model = ReferenceStack(layers).to(device=device, dtype=dtype).train()
    if args.precision == "fp8":
        _enable_torchao_fp8(model)
    return model


def _mark_step_begin() -> None:
    marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def _clear_gradients(model: nn.Module, x: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    x.grad = None


def _step(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _mark_step_begin()
    output = model(x)
    loss = output.float().square().mean()
    loss.backward()
    return output, loss


def _set_test_memory_cap(device: torch.device, gib: float) -> None:
    if gib <= 0:
        return
    total = torch.cuda.get_device_properties(device).total_memory
    requested = int(gib * 1024**3)
    torch.cuda.set_per_process_memory_fraction(min(requested / total, 1.0), device)


def run(args: argparse.Namespace) -> Measurement:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    _set_test_memory_cap(device, args.max_test_vram_gib)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = _build_model(args, device)
    dtype = _dtype_for_precision(args.precision)
    x = torch.randn(
        args.batch,
        args.sequence_length,
        args.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    if args.compile_mode != "none":
        model = torch.compile(
            model,
            backend="inductor",
            mode=args.compile_mode,
            fullgraph=True,
        )

    for _ in range(args.warmup):
        _clear_gradients(model, x)
        output, loss = _step(model, x)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite warmup loss")
        if not torch.isfinite(output).all():
            raise RuntimeError("non-finite warmup output")
    _clear_gradients(model, x)
    del output, loss
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    fingerprints: list[float] = []
    for _ in range(args.steps):
        _clear_gradients(model, x)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output, loss = _step(model, x)
        end.record()
        end.synchronize()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite measured loss")
        timings.append(start.elapsed_time(end))
        fingerprints.append(float(output.float().sum().item()))

    if args.dropout_p and args.steps > 1 and len(set(fingerprints)) == 1:
        raise RuntimeError("dropout output repeated across every measured step")

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    median_ms = statistics.median(timings)
    tokens = args.batch * args.sequence_length * args.layers
    properties = torch.cuda.get_device_properties(device)
    return Measurement(
        device=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        torch_version=torch.__version__,
        torch_cuda=str(torch.version.cuda),
        precision=args.precision,
        gupn_route=args.gupn_route,
        compile_mode=args.compile_mode,
        batch=args.batch,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        layers=args.layers,
        dropout_p=args.dropout_p,
        steps=args.steps,
        median_step_ms=median_ms,
        mean_step_ms=statistics.fmean(timings),
        min_step_ms=min(timings),
        max_step_ms=max(timings),
        tokens_per_second=tokens / (median_ms / 1000.0),
        baseline_allocated_bytes=baseline_allocated,
        peak_allocated_bytes=peak_allocated,
        incremental_peak_allocated_bytes=max(peak_allocated - baseline_allocated, 0),
        baseline_reserved_bytes=baseline_reserved,
        peak_reserved_bytes=peak_reserved,
        incremental_peak_reserved_bytes=max(peak_reserved - baseline_reserved, 0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the exact NeoLLM MLP training graph on CUDA."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--fan-ratio", type=float, default=0.0625)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--nonexclusive", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp8"), default="fp8")
    parser.add_argument(
        "--gupn-route",
        choices=("reference", "cute", "fp8-cute"),
        default="reference",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--max-test-vram-gib",
        type=float,
        default=10.0,
        help="Per-process test cap; this is never applied by production code.",
    )
    args = parser.parse_args()
    if args.batch <= 0 or args.sequence_length <= 0 or args.layers <= 0:
        parser.error("batch, sequence-length and layers must be positive")
    if args.steps <= 0 or args.warmup < 0:
        parser.error("steps must be positive and warmup non-negative")
    if not 0.0 <= args.dropout_p < 1.0:
        parser.error("dropout-p must be in [0, 1)")
    if args.gupn_route != "reference" and args.nonexclusive:
        parser.error("CuTe GUPN routes require exclusive PolyNorm")
    if args.gupn_route == "fp8-cute" and args.precision != "fp8":
        parser.error("--gupn-route fp8-cute requires --precision fp8")
    if not math.isfinite(args.max_test_vram_gib):
        parser.error("max-test-vram-gib must be finite")
    return args


if __name__ == "__main__":
    print(json.dumps(asdict(run(parse_args())), indent=2, sort_keys=True))
