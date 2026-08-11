from __future__ import annotations

import torch

from . import _cute
from .reference import polynorm_reference

# Below this size, Inductor can fuse the reference expression more cheaply than
# entering an opaque custom op.  Keep eager execution on CuTe: the reference is
# only the faster path when it is part of a larger compiled graph.
_COMPILED_CUTE_MIN_ELEMENTS = 8 * 1024 * 1024


def _prefer_compiler_fusion(x: torch.Tensor) -> bool:
    return bool(
        torch.compiler.is_compiling()
        and x.numel() < _COMPILED_CUTE_MIN_ELEMENTS
    )


def _cute_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    exclusive_logits: torch.Tensor | None,
    dropout_p: float,
) -> bool:
    return bool(
        _cute.is_available()
        and exclusive_logits is None
        and eps == 1.0e-6
        and x.is_cuda
        and weight.is_cuda
        and bias.is_cuda
        and x.dtype in (torch.bfloat16, torch.float32)
        and weight.dtype == x.dtype
        and bias.dtype == x.dtype
        and x.ndim >= 2
        and x.shape[-1] > 0
        and x.shape[-1] % _cute.VECTOR_WIDTH == 0
        and x.is_contiguous()
        and weight.shape == (3,)
        and bias.shape == (1,)
        and 0.0 <= dropout_p < 1.0
    )


@torch.library.custom_op(
    "cut_cross_entropy::polynorm_forward",
    mutates_args=(),
    device_types="cuda",
)
def _polynorm_forward_op(
    x: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _cute.forward(
        x,
        seeds,
        weight,
        bias,
        dropout_p=dropout_p,
        save_stats=True,
    )


@_polynorm_forward_op.register_fake
def _polynorm_forward_fake(
    x: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del seeds, weight, bias, dropout_p
    return (
        torch.empty_like(x),
        torch.empty((x.shape[0], 3), device=x.device, dtype=torch.float32),
    )


@torch.library.custom_op(
    "cut_cross_entropy::polynorm_inference",
    mutates_args=(),
    device_types="cuda",
)
def _polynorm_inference_op(
    x: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    output, _stats = _cute.forward(
        x,
        seeds,
        weight,
        bias,
        dropout_p=dropout_p,
        save_stats=False,
    )
    return output


@_polynorm_inference_op.register_fake
def _polynorm_inference_fake(
    x: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    del seeds, weight, bias, dropout_p
    return torch.empty_like(x)


@torch.library.custom_op(
    "cut_cross_entropy::polynorm_backward",
    mutates_args=(),
    device_types="cuda",
)
def _polynorm_backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    stats: torch.Tensor,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _cute.backward(
        grad_output,
        x,
        seeds,
        weight,
        stats,
        dropout_p=dropout_p,
    )


@_polynorm_backward_op.register_fake
def _polynorm_backward_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    stats: torch.Tensor,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_output, seeds, stats, dropout_p
    return torch.empty_like(x), torch.empty_like(weight), torch.empty_like(weight[:1])


def _setup_context(ctx, inputs, output) -> None:
    x, seeds, weight, _bias, dropout_p = inputs
    _result, stats = output
    ctx.save_for_backward(x, seeds, weight, stats)
    ctx.dropout_p = dropout_p
    ctx.mark_non_differentiable(stats)


def _backward(ctx, grad_output, _grad_stats):
    x, seeds, weight, stats = ctx.saved_tensors
    grad_x, grad_weight, grad_bias = _polynorm_backward_op(
        grad_output,
        x,
        seeds,
        weight,
        stats,
        ctx.dropout_p,
    )
    return grad_x, None, grad_weight, grad_bias, None


torch.library.register_autograd(
    _polynorm_forward_op,
    _backward,
    setup_context=_setup_context,
)


def polynorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    proj_eps: float = 1.0e-6,
    exclusive_logits: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Run PolyNorm and optional dropout through CuTe when supported."""
    dropout_p = float(dropout_p)
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if _prefer_compiler_fusion(x) or not _cute_supported(
        x, weight, bias, eps, exclusive_logits, dropout_p
    ):
        output = polynorm_reference(
            x,
            weight,
            bias,
            eps=eps,
            proj_eps=proj_eps,
            exclusive_logits=exclusive_logits,
        )
        if dropout_p:
            output = torch.nn.functional.dropout(
                output, p=dropout_p, training=True
            )
        return output

    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    seeds = (
        torch.randint(
            0,
            1 << 32,
            (4,),
            device=x.device,
            dtype=torch.int64,
        )
        if dropout_p
        else torch.empty((4,), device=x.device, dtype=torch.int64)
    )
    needs_backward = torch.is_grad_enabled() and (
        x.requires_grad or weight.requires_grad or bias.requires_grad
    )
    if needs_backward:
        output, _stats = _polynorm_forward_op(
            x_2d, seeds, weight, bias, dropout_p
        )
    else:
        output = _polynorm_inference_op(
            x_2d, seeds, weight, bias, dropout_p
        )
    return output.reshape(original_shape)


__all__ = ["polynorm"]
