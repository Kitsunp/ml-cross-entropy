"""Profile the PolyNorm reference and compiler-generated CUDA kernels.

Run one backend per process so compiler caches, CUDA graph pools, and allocator
state do not contaminate comparisons.  The optional memory limit applies only
to this benchmark process; it never changes production behavior.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from collections.abc import Callable

import torch


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("torch", "torch_compile", "cute"), required=True
    )
    parser.add_argument("--compile-mode", choices=("default", "max-autotune"), default="default")
    parser.add_argument(
        "--compile-cute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the CuTe custom-op boundary as it is used inside a compiled model.",
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--output-dtype",
        choices=("native", "input"),
        default="native",
        help="Keep autocast output dtype or cast at the next BF16 consumer boundary.",
    )
    parser.add_argument("--autocast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multiply-up", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--multiply-column", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--proj-eps", type=float, default=1.0e-6)
    parser.add_argument("--exclusive-init", type=float, default=1.0e-4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    return parser.parse_args()


def polynorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    *,
    eps: float,
    proj_eps: float,
    exclusive: bool,
) -> torch.Tensor:
    """Reference PolyNorm semantics, kept independent of production code."""
    x_sq = x.pow(2)
    x_cu = x * x_sq

    x1 = x * x_sq.mean(-1, keepdim=True).add(eps).rsqrt()
    x2 = x_sq * (x_sq * x_sq).mean(-1, keepdim=True).add(eps).rsqrt()
    x3 = x_cu * (x_cu * x_cu).mean(-1, keepdim=True).add(eps).rsqrt()

    if exclusive:
        alpha2, alpha3 = torch.sigmoid(exclusive_logits).unbind()
        x1_f = x1.float()
        ref_norm_sq = x1_f.pow(2).sum(-1, keepdim=True).clamp_min(proj_eps)

        def exclusive_branch(branch: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
            dot = (branch.float() * x1_f).sum(dim=-1, keepdim=True)
            proj_coeff = (dot / ref_norm_sq).to(branch.dtype)
            out = branch - alpha.to(branch.dtype) * proj_coeff * x1
            return out * out.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()

        x2 = exclusive_branch(x2, alpha2)
        x3 = exclusive_branch(x3, alpha3)

    return weight[0] * x3 + weight[1] * x2 + weight[2] * x1 + bias


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[round(0.95 * (len(ordered) - 1))]
    return {
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "min": ordered[0],
        "p95": p95,
        "max": ordered[-1],
    }


def _mark_step() -> None:
    marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.batch < 1 or args.sequence < 1 or args.hidden < 1:
        raise ValueError("batch, sequence, and hidden must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if not 0.0 < args.exclusive_init < 1.0:
        raise ValueError("exclusive-init must be in (0, 1)")
    if not 0.0 <= args.dropout_p < 1.0:
        raise ValueError("dropout-p must be in [0, 1)")

    dtype = getattr(torch, args.dtype)
    properties = torch.cuda.get_device_properties(0)
    if args.memory_limit_gib is not None:
        if args.memory_limit_gib <= 0:
            raise ValueError("memory-limit-gib must be positive")
        fraction = min(args.memory_limit_gib * 1024**3 / properties.total_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    shape = (args.batch, args.sequence, args.hidden)
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator, requires_grad=True)
    up = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        generator=generator,
        requires_grad=args.multiply_up,
    )
    column = torch.ones(
        (args.hidden,),
        device="cuda",
        dtype=torch.float32,
        requires_grad=args.multiply_column,
    )
    grad_output = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    weight = torch.full((3,), 1.0 / 3.0, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.zeros(1, device="cuda", dtype=dtype, requires_grad=True)
    logit = math.log(args.exclusive_init / (1.0 - args.exclusive_init))
    exclusive_logits = torch.full((2,), logit, device="cuda", dtype=dtype, requires_grad=True)

    def eager_fn(
        input_: torch.Tensor,
        weight_: torch.Tensor,
        bias_: torch.Tensor,
        logits_: torch.Tensor,
        up_: torch.Tensor,
        column_: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(
            "cuda",
            dtype=dtype,
            enabled=args.autocast and dtype != torch.float32,
        ):
            output = polynorm_reference(
                input_,
                weight_,
                bias_,
                logits_,
                eps=args.eps,
                proj_eps=args.proj_eps,
                exclusive=args.exclusive,
            )
            if args.multiply_up:
                output = output * up_
            if args.dropout_p:
                output = torch.nn.functional.dropout(
                    output, p=args.dropout_p, training=True
                )
            if args.multiply_column:
                output = output * column_
            return output.to(dtype) if args.output_dtype == "input" else output

    function: Callable[..., torch.Tensor] = eager_fn
    if args.backend == "torch_compile":
        compile_kwargs: dict[str, object] = {"fullgraph": True}
        if args.compile_mode != "default":
            compile_kwargs["mode"] = args.compile_mode
        function = torch.compile(eager_fn, **compile_kwargs)
    elif args.backend == "cute":
        from cut_cross_entropy.polynorm import _cute, polynorm

        if not _cute.is_available():
            raise RuntimeError("nvidia-cutlass-dsl is required for --backend cute")

        def cute_fn(
            input_: torch.Tensor,
            weight_: torch.Tensor,
            bias_: torch.Tensor,
            logits_: torch.Tensor,
            up_: torch.Tensor,
            column_: torch.Tensor,
        ) -> torch.Tensor:
            output = polynorm(
                input_,
                weight_,
                bias_,
                eps=args.eps,
                proj_eps=args.proj_eps,
                exclusive_logits=logits_ if args.exclusive else None,
                dropout_p=args.dropout_p,
            )
            if args.multiply_up:
                output = output * up_
            if args.multiply_column:
                output = output * column_
            return output.to(dtype) if args.output_dtype == "input" else output

        function = cute_fn
        if args.compile_cute:
            compile_kwargs = {"fullgraph": True}
            if args.compile_mode != "default":
                compile_kwargs["mode"] = args.compile_mode
            function = torch.compile(cute_fn, **compile_kwargs)

    differentiated_list = [x, weight, bias]
    if args.exclusive:
        differentiated_list.append(exclusive_logits)
    if args.multiply_up:
        differentiated_list.append(up)
    if args.multiply_column:
        differentiated_list.append(column)
    differentiated = tuple(differentiated_list)

    def run_step() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if args.backend == "torch_compile" or args.compile_cute:
            _mark_step()
        output = function(x, weight, bias, exclusive_logits, up, column)
        gradients = torch.autograd.grad(output, differentiated, grad_output)
        return output, gradients

    input_allocated = torch.cuda.memory_allocated()
    for _ in range(args.warmup + 1):
        output, gradients = run_step()
        del output, gradients
    output, gradients = run_step()
    finite = bool(torch.isfinite(output).all()) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    del output, gradients
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.synchronize()

    resident_allocated = torch.cuda.memory_allocated()
    resident_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    forward_ms: list[float] = []
    backward_ms: list[float] = []
    for _ in range(args.iterations):
        if args.backend == "torch_compile" or args.compile_cute:
            _mark_step()
        start = torch.cuda.Event(enable_timing=True)
        forward_done = torch.cuda.Event(enable_timing=True)
        backward_done = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function(x, weight, bias, exclusive_logits, up, column)
        forward_done.record()
        gradients = torch.autograd.grad(output, differentiated, grad_output)
        backward_done.record()
        torch.cuda.synchronize()
        forward_ms.append(start.elapsed_time(forward_done))
        backward_ms.append(forward_done.elapsed_time(backward_done))
        del output, gradients, start, forward_done, backward_done

    total_ms = [forward + backward for forward, backward in zip(forward_ms, backward_ms, strict=True)]
    elements = args.batch * args.sequence * args.hidden
    total_median = statistics.median(total_ms)
    result = {
        "backend": args.backend,
        "compile_mode": (
            args.compile_mode
            if args.backend == "torch_compile" or args.compile_cute
            else None
        ),
        "compile_cute": args.compile_cute,
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "shape": list(shape),
        "dtype": str(dtype),
        "output_dtype_policy": args.output_dtype,
        "autocast": args.autocast and dtype != torch.float32,
        "exclusive": args.exclusive,
        "multiply_up": args.multiply_up,
        "multiply_column": args.multiply_column,
        "dropout_p": args.dropout_p,
        "forward_ms": _summary(forward_ms),
        "backward_ms": _summary(backward_ms),
        "total_ms": _summary(total_ms),
        "billion_elements_per_second": elements / total_median / 1.0e6,
        "memory": {
            "input_allocated_bytes": input_allocated,
            "resident_allocated_after_warmup_bytes": resident_allocated,
            "resident_reserved_after_warmup_bytes": resident_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - resident_allocated,
            "incremental_peak_reserved_bytes": torch.cuda.max_memory_reserved() - resident_reserved,
            "test_memory_limit_gib": args.memory_limit_gib,
        },
        "finite": finite,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
