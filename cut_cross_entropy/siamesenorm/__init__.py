"""Fused paired RMSNorm; explicit candidate backends until performance gates pass."""

from __future__ import annotations

import math

import torch

from cut_cross_entropy.torch_2_14 import mark_warmup_incomplete_once

from . import _cute, _triton
from .reference import output_dtype, siamesenorm_reference

try:
    from torch.library import triton_op
except ImportError:  # Compatibility: the original custom_op remains available.
    triton_op = None


@torch.library.custom_op("cut_cross_entropy::siamesenorm_forward", mutates_args=())
def _forward(
    x: torch.Tensor,
    y: torch.Tensor,
    wx: torch.Tensor,
    wy: torch.Tensor,
    wo: torch.Tensor | None,
    scale: torch.Tensor | None,
    eps: float,
    backend: str,
    post_scale: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    implementation = _cute if backend == "cute" else _triton
    result = implementation.forward(x, y, wx, wy, wo, scale, eps, post_scale, out_dtype)
    mark_warmup_incomplete_once(
        "siamesenorm.forward",
        (
            backend,
            x.device.index,
            tuple(x.shape),
            tuple(t.dtype if t is not None else None for t in (x, y, wx, wy, wo, scale)),
            eps,
            post_scale is not None,
            out_dtype,
        ),
    )
    return result


@_forward.register_fake
def _forward_fake(x, y, wx, wy, wo, scale, eps, backend, post_scale, out_dtype):
    return torch.empty_like(
        x, dtype=output_dtype(x, y, scale) if out_dtype is None else out_dtype
    ), torch.empty((x.shape[0], 3), device=x.device, dtype=torch.float32)


@torch.library.custom_op("cut_cross_entropy::siamesenorm_backward", mutates_args=())
def _backward_op(
    g: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    wx: torch.Tensor,
    wy: torch.Tensor,
    wo: torch.Tensor | None,
    scale: torch.Tensor | None,
    stats: torch.Tensor,
    backend: str,
    post_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    implementation = _cute if backend == "cute" else _triton
    result = implementation.backward(g, x, y, wx, wy, wo, scale, stats, post_scale)
    mark_warmup_incomplete_once(
        "siamesenorm.backward",
        (
            backend,
            x.device.index,
            tuple(x.shape),
            tuple(t.dtype if t is not None else None for t in (x, y, wx, wy, wo, scale)),
            g.dtype,
            post_scale is not None,
        ),
    )
    # Dispatcher return schemas do not accept optional tensors here. Empty
    # fresh tensors stand for absent parameter gradients; autograd drops them.
    return (
        *result[:4],
        result[4] if wo is not None else x.new_empty((0,)),
        result[5] if scale is not None else x.new_empty((0,)),
    )


@_backward_op.register_fake
def _backward_fake(g, x, y, wx, wy, wo, scale, stats, backend, post_scale):
    return (
        torch.empty_like(x),
        torch.empty_like(y),
        torch.empty_like(wx),
        torch.empty_like(wy),
        torch.empty_like(wo) if wo is not None else x.new_empty((0,)),
        torch.empty_like(scale) if scale is not None else x.new_empty((0,)),
    )


def _setup(ctx, inputs, output):
    x, y, wx, wy, wo, scale, _eps, backend, post_scale, _out_dtype = inputs
    ctx.save_for_backward(x, y, wx, wy, wo, scale, output[1], post_scale)
    ctx.backend = backend
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, g, _stats):
    grads = _backward_op(g, *ctx.saved_tensors[:7], ctx.backend, ctx.saved_tensors[7])
    return (
        *grads[:4],
        grads[4] if ctx.saved_tensors[4] is not None else None,
        grads[5] if ctx.saved_tensors[5] is not None else None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(_forward, _backward, setup_context=_setup)

if triton_op is not None:

    @triton_op("cut_cross_entropy::siamesenorm_triton_forward", mutates_args={})
    def _triton_forward_op(
        x: torch.Tensor,
        y: torch.Tensor,
        wx: torch.Tensor,
        wy: torch.Tensor,
        wo: torch.Tensor | None,
        scale: torch.Tensor | None,
        eps: float,
        post_scale: torch.Tensor | None,
        out_dtype: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _triton.forward(x, y, wx, wy, wo, scale, eps, post_scale, out_dtype)

    @triton_op("cut_cross_entropy::siamesenorm_triton_backward", mutates_args={})
    def _triton_backward_op(
        g: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        wx: torch.Tensor,
        wy: torch.Tensor,
        wo: torch.Tensor | None,
        scale: torch.Tensor | None,
        stats: torch.Tensor,
        post_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grads = _triton.backward(g, x, y, wx, wy, wo, scale, stats, post_scale)
        return (
            *grads[:4],
            grads[4] if wo is not None else x.new_empty((0,)),
            grads[5] if scale is not None else x.new_empty((0,)),
        )

    def _triton_setup(ctx, inputs, output):
        _setup(ctx, (*inputs[:7], "triton", *inputs[7:]), output)

    def _triton_backward(ctx, g, _stats):
        grads = _triton_backward_op(g, *ctx.saved_tensors)
        return (
            *grads[:4],
            grads[4] if ctx.saved_tensors[4] is not None else None,
            grads[5] if ctx.saved_tensors[5] is not None else None,
            None,
            None,
            None,
        )

    _triton_forward_op.register_autograd(_triton_backward, setup_context=_triton_setup)


def siamesenorm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight_x: torch.Tensor,
    weight_y: torch.Tensor,
    weight_out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
    backend: str = "reference",
    post_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Paired RMSNorm candidate with static backend selection.

    ``cute`` and ``triton`` each implement both forward and backward. ``auto``
    prefers CuTe when eligible, otherwise Triton on supported CUDA tensors. The
    default stays ``reference`` until the full-model acceptance gate passes.
    Unsupported layouts/dtypes use Torch; an explicitly forced backend raises
    instead of producing a benchmark falsely labeled as a custom kernel.
    ``post_scale`` optionally fuses a non-trainable native-dtype scalar multiply;
    ``out_dtype`` optionally fuses a final FP32/BF16 consumer cast. Neither
    changes the native norm rounding or gradients. Move only an existing cast.
    """
    if backend not in ("reference", "auto", "triton", "cute"):
        raise ValueError("backend must be reference, auto, triton or cute")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if x.shape != y.shape or x.ndim < 1:
        raise ValueError("streams must have identical non-scalar shapes")
    params = (weight_x, weight_y, weight_out, scale)
    if any(p is not None and p.shape != (x.shape[-1],) for p in params):
        raise ValueError("affine and scale parameters must be hidden-width vectors")
    if out_dtype not in (None, torch.float32, torch.bfloat16):
        raise ValueError("out_dtype must be FP32, BF16 or None")
    if post_scale is not None and (
        post_scale.ndim != 0
        or post_scale.requires_grad
        or post_scale.dtype != output_dtype(x, y, scale)
        or post_scale.device != x.device
    ):
        raise ValueError("post_scale must be a fixed scalar in the native norm dtype/device")
    tensors = (x, y, *(p for p in (*params, post_scale) if p is not None))
    supported = (
        x.is_cuda
        and x.numel() > 0
        and 0 < x.shape[-1] <= 4096
        and all(
            t.device == x.device
            and t.dtype in (torch.float32, torch.bfloat16)
            and t.is_contiguous()
            for t in tensors
        )
    )
    if backend == "reference" or (backend == "auto" and not supported):
        return siamesenorm_reference(
            x,
            y,
            weight_x,
            weight_y,
            weight_out,
            scale,
            eps=eps,
            post_scale=post_scale,
            out_dtype=out_dtype,
        )
    if not supported:
        raise ValueError("forced backend requires contiguous CUDA FP32/BF16 tensors on one device")
    cute_supported = _cute.is_available() and x.shape[-1] % 32 == 0 and x.shape[-1] <= 1024
    if backend == "auto":
        backend = "cute" if cute_supported else "triton"
    if backend == "cute" and not cute_supported:
        raise ValueError("CuTe requires its optional package and width divisible by 32 up to 1024")
    args = (
        x.reshape(-1, x.shape[-1]),
        y.reshape(-1, y.shape[-1]),
        weight_x,
        weight_y,
        weight_out,
        scale,
        eps,
    )
    if backend == "triton" and triton_op is not None:
        output, _stats = _triton_forward_op(*args, post_scale, out_dtype)
    else:
        output, _stats = _forward(*args, backend, post_scale, out_dtype)
    return output.reshape(x.shape)


__all__ = ["siamesenorm", "siamesenorm_reference"]
