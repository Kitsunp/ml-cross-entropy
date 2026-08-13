from __future__ import annotations

import torch

from . import _cute_gupn, _cute_gupn_backward
from .reference import gupn_reference


def _needs_backward(tensors: tuple[torch.Tensor, ...]) -> bool:
    return bool(torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors))


def _supported(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    eps: float,
    proj_eps: float,
    dropout_p: float,
) -> bool:
    tensors = (
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
    )
    return bool(
        _cute_gupn.is_available()
        and (
            not _needs_backward(tensors)
            or _cute_gupn_backward.is_available()
        )
        and gate0.is_cuda
        and gate0.dtype == torch.bfloat16
        and gate0.ndim >= 2
        and gate0.numel() > 0
        and gate0.shape[-1] % _cute_gupn.VECTOR_WIDTH == 0
        and gate0.is_contiguous()
        and up.shape == gate0.shape
        and up.device == gate0.device
        and up.dtype == gate0.dtype
        and up.is_contiguous()
        and gate_row_multiplier.shape == (gate0.shape[-1],)
        and polynorm_weight.shape == (3,)
        and polynorm_bias.shape == (1,)
        and exclusive_logits.shape == (2,)
        and down_column_multiplier.shape == (gate0.shape[-1],)
        and all(tensor.device == gate0.device for tensor in tensors[2:])
        and all(tensor.dtype == gate0.dtype for tensor in tensors[2:])
        and eps == 1.0e-6
        and proj_eps == 1.0e-6
        and 0.0 <= dropout_p <= _cute_gupn.MAX_DROPOUT_P
    )


def gupn_uses_cute(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    proj_eps: float = 1.0e-6,
    dropout_p: float = 0.0,
) -> bool:
    return _supported(
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        float(eps),
        float(proj_eps),
        float(dropout_p),
    )


@torch.library.custom_op(
    "cut_cross_entropy::gupn_exclusive_inference",
    mutates_args=(),
    device_types="cuda",
)
def _gupn_inference_op(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    return _cute_gupn.forward(
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p=dropout_p,
        use_xor=True,
    )


@_gupn_inference_op.register_fake
def _gupn_inference_fake(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    del (
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p,
    )
    return torch.empty_like(gate0)


@torch.library.custom_op(
    "cut_cross_entropy::gupn_exclusive_forward",
    mutates_args=(),
    device_types="cuda",
)
def _gupn_forward_op(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    return _cute_gupn.forward(
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p=dropout_p,
        use_xor=True,
    )


@_gupn_forward_op.register_fake
def _gupn_forward_fake(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    del (
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p,
    )
    return torch.empty_like(gate0)


@torch.library.custom_op(
    "cut_cross_entropy::gupn_exclusive_backward",
    mutates_args=(),
    device_types="cuda",
)
def _gupn_backward_op(
    grad_output: torch.Tensor,
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _cute_gupn_backward.backward(
        grad_output,
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p=dropout_p,
        use_xor=True,
    )


@_gupn_backward_op.register_fake
def _gupn_backward_fake(
    grad_output: torch.Tensor,
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del grad_output, seeds, dropout_p
    return (
        torch.empty_like(gate0),
        torch.empty_like(up),
        torch.empty_like(gate_row_multiplier),
        torch.empty_like(polynorm_weight),
        torch.empty_like(polynorm_bias),
        torch.empty_like(exclusive_logits),
        torch.empty_like(down_column_multiplier),
    )


def _gupn_setup_context(ctx, inputs, output) -> None:
    del output
    (
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p,
    ) = inputs
    ctx.save_for_backward(
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
    )
    ctx.dropout_p = dropout_p


def _gupn_backward(ctx, grad_output):
    (
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
    ) = ctx.saved_tensors
    gradients = _gupn_backward_op(
        grad_output,
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        ctx.dropout_p,
    )
    return (*gradients, None, None)


torch.library.register_autograd(
    _gupn_forward_op,
    _gupn_backward,
    setup_context=_gupn_setup_context,
)


def gupn(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    proj_eps: float = 1.0e-6,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Run exclusive GUPN through CuTe with a recomputed custom backward."""
    dropout_p = float(dropout_p)
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if not gupn_uses_cute(
        gate0,
        up,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        eps=eps,
        proj_eps=proj_eps,
        dropout_p=dropout_p,
    ):
        return gupn_reference(
            gate0,
            up,
            gate_row_multiplier,
            polynorm_weight,
            polynorm_bias,
            exclusive_logits,
            down_column_multiplier,
            eps=eps,
            proj_eps=proj_eps,
            dropout_p=dropout_p,
        )

    original_shape = gate0.shape
    gate0_2d = gate0.reshape(-1, gate0.shape[-1])
    up_2d = up.reshape(-1, up.shape[-1])
    seeds = (
        torch.randint(0, 1 << 32, (4,), device=gate0.device, dtype=torch.int64)
        if dropout_p
        else torch.empty((4,), device=gate0.device, dtype=torch.int64)
    )
    operator = (
        _gupn_forward_op
        if _needs_backward(
            (
                gate0,
                up,
                gate_row_multiplier,
                polynorm_weight,
                polynorm_bias,
                exclusive_logits,
                down_column_multiplier,
            )
        )
        else _gupn_inference_op
    )
    return operator(
        gate0_2d,
        up_2d,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p,
    ).reshape(original_shape)


__all__ = ["gupn", "gupn_uses_cute"]
