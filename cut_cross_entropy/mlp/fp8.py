from __future__ import annotations

import torch

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_training_tensor import (
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)

from . import _cute_gupn, _cute_gupn_backward


_CONFIG = Float8LinearConfig()
_MM_CONFIG = LinearMMConfig(
    ScaledMMConfig(
        _CONFIG.emulate,
        _CONFIG.gemm_config_output.use_fast_accum,
        False,
        _CONFIG.pad_inner_dim,
    ),
    ScaledMMConfig(
        _CONFIG.emulate,
        _CONFIG.gemm_config_grad_input.use_fast_accum,
        False,
        _CONFIG.pad_inner_dim,
    ),
    ScaledMMConfig(
        _CONFIG.emulate,
        _CONFIG.gemm_config_grad_weight.use_fast_accum,
        False,
        _CONFIG.pad_inner_dim,
    ),
)


def _cast(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    role: GemmInputRole,
):
    return hp_tensor_to_float8_dynamic(
        tensor,
        dtype,
        _MM_CONFIG,
        gemm_input_role=role,
        scaling_granularity=(
            _CONFIG.cast_config_input.scaling_granularity
            if role is GemmInputRole.INPUT
            else _CONFIG.cast_config_weight.scaling_granularity
            if role is GemmInputRole.WEIGHT
            else _CONFIG.cast_config_grad_output.scaling_granularity
        ),
        round_scales_to_power_of_2=_CONFIG.round_scales_to_power_of_2,
    )


def _dual_forward(
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TorchAO-exact staged dual projection with one dynamic input cast."""
    fan_fp8 = _cast(
        fan,
        dtype=_CONFIG.cast_config_input.target_dtype,
        role=GemmInputRole.INPUT,
    )
    gate_weight_fp8 = _cast(
        gate_weight.t(),
        dtype=_CONFIG.cast_config_weight.target_dtype,
        role=GemmInputRole.WEIGHT,
    )
    up_weight_fp8 = _cast(
        up_weight.t(),
        dtype=_CONFIG.cast_config_weight.target_dtype,
        role=GemmInputRole.WEIGHT,
    )
    return (
        torch.mm(fan_fp8, gate_weight_fp8),
        torch.mm(fan_fp8, up_weight_fp8),
    )


def _dual_backward(
    grad_gate: torch.Tensor,
    grad_up: torch.Tensor,
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Default TorchAO FP8 backward with the common FAN cast shared."""
    original_shape = fan.shape
    fan_2d = fan.reshape(-1, fan.shape[-1])
    grad_gate_2d = grad_gate.reshape(-1, grad_gate.shape[-1])
    grad_up_2d = grad_up.reshape(-1, grad_up.shape[-1])

    grad_gate_input_fp8 = _cast(
        grad_gate_2d,
        dtype=_CONFIG.cast_config_grad_output.target_dtype,
        role=GemmInputRole.GRAD_OUTPUT,
    )
    grad_up_input_fp8 = _cast(
        grad_up_2d,
        dtype=_CONFIG.cast_config_grad_output.target_dtype,
        role=GemmInputRole.GRAD_OUTPUT,
    )
    gate_weight_input_fp8 = _cast(
        gate_weight.t(),
        dtype=_CONFIG.cast_config_weight_for_grad_input.target_dtype,
        role=GemmInputRole.WEIGHT,
    )
    up_weight_input_fp8 = _cast(
        up_weight.t(),
        dtype=_CONFIG.cast_config_weight_for_grad_input.target_dtype,
        role=GemmInputRole.WEIGHT,
    )
    grad_fan = (
        torch.mm(grad_gate_input_fp8, gate_weight_input_fp8.t())
        + torch.mm(grad_up_input_fp8, up_weight_input_fp8.t())
    ).reshape(original_shape)

    fan_weight_fp8 = _cast(
        fan_2d,
        dtype=_CONFIG.cast_config_input_for_grad_weight.target_dtype,
        role=GemmInputRole.INPUT,
    )
    grad_gate_weight_fp8 = _cast(
        grad_gate_2d,
        dtype=_CONFIG.cast_config_grad_output_for_grad_weight.target_dtype,
        role=GemmInputRole.GRAD_OUTPUT,
    )
    grad_up_weight_fp8 = _cast(
        grad_up_2d,
        dtype=_CONFIG.cast_config_grad_output_for_grad_weight.target_dtype,
        role=GemmInputRole.GRAD_OUTPUT,
    )
    grad_gate_weight = torch.mm(
        grad_gate_weight_fp8.t(),
        fan_weight_fp8,
    )
    grad_up_weight = torch.mm(
        grad_up_weight_fp8.t(),
        fan_weight_fp8,
    )
    return grad_fan, grad_gate_weight, grad_up_weight


def _validate(
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    dropout_p: float,
) -> None:
    if not fan.is_cuda or fan.dtype != torch.bfloat16 or fan.ndim < 2:
        raise ValueError("fan must be a CUDA BF16 tensor with at least two dimensions")
    fan_features = fan.shape[-1]
    if gate_weight.ndim != 2 or gate_weight.shape[1] != fan_features:
        raise ValueError("gate_weight must have shape (intermediate, fan_features)")
    if up_weight.shape != gate_weight.shape:
        raise ValueError("up_weight must match gate_weight")
    intermediate = gate_weight.shape[0]
    expected = (
        ("gate_row_multiplier", gate_row_multiplier, (intermediate,)),
        ("polynorm_weight", polynorm_weight, (3,)),
        ("polynorm_bias", polynorm_bias, (1,)),
        ("exclusive_logits", exclusive_logits, (2,)),
        ("down_column_multiplier", down_column_multiplier, (intermediate,)),
    )
    tensors = (gate_weight, up_weight) + tuple(item[1] for item in expected)
    if any(tensor.device != fan.device for tensor in tensors):
        raise ValueError("all tensors must be on the FAN device")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("all tensors must be BF16")
    for name, tensor, shape in expected:
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if fan_features % 16 or intermediate % 16:
        raise ValueError("FP8 gate/up dimensions must be divisible by 16")
    if fan.numel() // fan_features % 16:
        raise ValueError(
            "the flattened row count must be divisible by 16 for FP8 grad-weight"
        )
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")


@torch.library.custom_op(
    "cut_cross_entropy::fp8_gupn_forward",
    mutates_args=(),
    device_types="cuda",
)
def _fp8_gupn_forward_op(
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    original_shape = fan.shape[:-1] + (gate_weight.shape[0],)
    fan_2d = fan.reshape(-1, fan.shape[-1])
    gate0, up = _dual_forward(fan_2d, gate_weight, up_weight)
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
    ).reshape(original_shape)


@_fp8_gupn_forward_op.register_fake
def _fp8_gupn_forward_fake(
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    del (
        up_weight,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p,
    )
    return fan.new_empty(fan.shape[:-1] + (gate_weight.shape[0],))


@torch.library.custom_op(
    "cut_cross_entropy::fp8_gupn_backward",
    mutates_args=(),
    device_types="cuda",
)
def _fp8_gupn_backward_op(
    grad_output: torch.Tensor,
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
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
    torch.Tensor,
]:
    fan_2d = fan.reshape(-1, fan.shape[-1])
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    gate0, up = _dual_forward(fan_2d, gate_weight, up_weight)
    (
        grad_gate,
        grad_up,
        grad_gate_row,
        grad_polynorm_weight,
        grad_polynorm_bias,
        grad_exclusive_logits,
        grad_down_column,
    ) = _cute_gupn_backward.backward(
        grad_output_2d,
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
    grad_fan, grad_gate_weight, grad_up_weight = _dual_backward(
        grad_gate,
        grad_up,
        fan_2d,
        gate_weight,
        up_weight,
    )
    return (
        grad_fan.reshape_as(fan),
        grad_gate_weight,
        grad_up_weight,
        grad_gate_row,
        grad_polynorm_weight,
        grad_polynorm_bias,
        grad_exclusive_logits,
        grad_down_column,
    )


@_fp8_gupn_backward_op.register_fake
def _fp8_gupn_backward_fake(
    grad_output: torch.Tensor,
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
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
    torch.Tensor,
]:
    del grad_output, seeds, dropout_p
    return tuple(
        torch.empty_like(tensor)
        for tensor in (
            fan,
            gate_weight,
            up_weight,
            gate_row_multiplier,
            polynorm_weight,
            polynorm_bias,
            exclusive_logits,
            down_column_multiplier,
        )
    )


def _setup_context(ctx, inputs, output) -> None:
    del output
    (
        fan,
        gate_weight,
        up_weight,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p,
    ) = inputs
    ctx.save_for_backward(
        fan,
        gate_weight,
        up_weight,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
    )
    ctx.dropout_p = dropout_p


def _backward(ctx, grad_output):
    gradients = _fp8_gupn_backward_op(
        grad_output,
        *ctx.saved_tensors,
        ctx.dropout_p,
    )
    return (*gradients, None, None)


torch.library.register_autograd(
    _fp8_gupn_forward_op,
    _backward,
    setup_context=_setup_context,
)


def fp8_gupn(
    fan: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    *,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Staged FP8 gate/up plus CuTe GUPN with recomputation in backward."""
    dropout_p = float(dropout_p)
    _validate(
        fan,
        gate_weight,
        up_weight,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        dropout_p,
    )
    seeds = (
        torch.randint(0, 1 << 32, (4,), device=fan.device, dtype=torch.int64)
        if dropout_p
        else torch.empty((4,), device=fan.device, dtype=torch.int64)
    )
    return _fp8_gupn_forward_op(
        fan,
        gate_weight,
        up_weight,
        gate_row_multiplier,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column_multiplier,
        seeds,
        dropout_p,
    )


__all__ = ["fp8_gupn"]
