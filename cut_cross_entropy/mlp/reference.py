from __future__ import annotations

import torch
import torch.nn.functional as F

from cut_cross_entropy.polynorm.reference import polynorm_reference


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def fan_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    periodic_dim: int,
) -> torch.Tensor:
    """Evaluate the FAN projection used by NeoLLM without changing its rounding points."""
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.ndim}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be a matrix, got {weight.ndim} dimensions")
    if weight.shape[1] != x.shape[-1]:
        raise ValueError(
            "FAN input dimension mismatch: "
            f"x has {x.shape[-1]} features but weight expects {weight.shape[1]}"
        )
    _require_shape("bias", bias, (weight.shape[0],))
    if not 0 <= periodic_dim <= weight.shape[0]:
        raise ValueError(
            "periodic_dim must be between zero and the FAN projection width, "
            f"got {periodic_dim} for width {weight.shape[0]}"
        )

    projected = F.linear(x, weight, bias)
    periodic = projected[..., :periodic_dim]
    passthrough = projected[..., periodic_dim:]
    return torch.cat((torch.cos(periodic), torch.sin(periodic), passthrough), dim=-1)


def _validate_mlp_contract(
    x: torch.Tensor,
    fan_weight: torch.Tensor,
    fan_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    gate_row_multiplier: torch.Tensor | None,
    down_column_multiplier: torch.Tensor | None,
    down_row_multiplier: torch.Tensor | None,
    exclusive_logits: torch.Tensor | None,
    fan_periodic_dim: int,
    dropout_p: float,
) -> tuple[int, int, int]:
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.ndim}")
    if fan_weight.ndim != 2:
        raise ValueError("fan_weight must be a matrix")
    hidden_size = x.shape[-1]
    if fan_weight.shape[1] != hidden_size:
        raise ValueError(
            f"fan_weight expects {fan_weight.shape[1]} features, x has {hidden_size}"
        )
    _require_shape("fan_bias", fan_bias, (fan_weight.shape[0],))
    if not 0 <= fan_periodic_dim <= fan_weight.shape[0]:
        raise ValueError(
            "fan_periodic_dim must be between zero and the FAN projection width"
        )

    fan_size = fan_weight.shape[0] + fan_periodic_dim
    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("gate_weight, up_weight and down_weight must be matrices")
    if gate_weight.shape[1] != fan_size or up_weight.shape[1] != fan_size:
        raise ValueError(
            "gate/up input width must equal the FAN output width "
            f"({fan_size}), got {gate_weight.shape[1]} and {up_weight.shape[1]}"
        )
    intermediate_size = gate_weight.shape[0]
    if up_weight.shape[0] != intermediate_size:
        raise ValueError("gate_weight and up_weight must have the same output width")
    if down_weight.shape != (hidden_size, intermediate_size):
        raise ValueError(
            "down_weight must restore the input hidden width; expected "
            f"{(hidden_size, intermediate_size)}, got {tuple(down_weight.shape)}"
        )

    _require_shape("polynorm_weight", polynorm_weight, (3,))
    _require_shape("polynorm_bias", polynorm_bias, (1,))
    if gate_row_multiplier is not None:
        _require_shape("gate_row_multiplier", gate_row_multiplier, (intermediate_size,))
    if down_column_multiplier is not None:
        _require_shape(
            "down_column_multiplier", down_column_multiplier, (intermediate_size,)
        )
    if down_row_multiplier is not None:
        _require_shape("down_row_multiplier", down_row_multiplier, (hidden_size,))
    if exclusive_logits is not None:
        _require_shape("exclusive_logits", exclusive_logits, (2,))
    return hidden_size, fan_size, intermediate_size


def gupn_reference(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor | None,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor | None,
    down_column_multiplier: torch.Tensor | None,
    *,
    eps: float = 1.0e-6,
    proj_eps: float = 1.0e-6,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """Reference for gate multiplier through the down-column boundary."""
    dropout_p = float(dropout_p)
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
    if gate0.shape != up.shape:
        raise ValueError("gate0 and up must have identical shapes")
    hidden = gate0.shape[-1]
    if gate_row_multiplier is not None:
        _require_shape("gate_row_multiplier", gate_row_multiplier, (hidden,))
    _require_shape("polynorm_weight", polynorm_weight, (3,))
    _require_shape("polynorm_bias", polynorm_bias, (1,))
    if exclusive_logits is not None:
        _require_shape("exclusive_logits", exclusive_logits, (2,))
    if down_column_multiplier is not None:
        _require_shape("down_column_multiplier", down_column_multiplier, (hidden,))

    gate = (
        gate0 * gate_row_multiplier
        if gate_row_multiplier is not None
        else gate0
    )
    activation = polynorm_reference(
        gate,
        polynorm_weight,
        polynorm_bias,
        eps=eps,
        proj_eps=proj_eps,
        exclusive_logits=exclusive_logits,
    )
    if training and dropout_p:
        activation = F.dropout(activation, p=dropout_p, training=True)
    down_input = activation * up
    if down_column_multiplier is not None:
        down_input = down_input * down_column_multiplier
    return down_input


def _rms_normalize_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    width = x.shape[-1]
    scale = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt().float()
    x_float = x.float()
    grad_float = grad_output.float()
    dot = (grad_float * x_float).sum(-1, keepdim=True)
    return grad_float * scale - x_float * scale.pow(3) * dot / width


def _exclusive_backward(
    grad_output: torch.Tensor,
    branch: torch.Tensor,
    reference: torch.Tensor,
    alpha: torch.Tensor,
    projection: torch.Tensor,
    reference_norm_sq: torch.Tensor,
    reference_norm_sq_unclamped: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_float = grad_output.float()
    branch_float = branch.float()
    reference_float = reference.float()
    alpha_float = alpha.float()
    projection_float = projection.float()
    dot_grad_reference = (grad_float * reference_float).sum(-1, keepdim=True)

    grad_branch = grad_float - (
        alpha_float
        * dot_grad_reference
        / reference_norm_sq
        * reference_float
    )
    grad_reference = -alpha_float * projection_float * grad_float
    grad_reference = grad_reference - (
        alpha_float
        * dot_grad_reference
        / reference_norm_sq
        * branch_float
    )
    projection_numerator = (branch_float * reference_float).sum(-1, keepdim=True)
    denominator_is_live = (
        reference_norm_sq_unclamped >= reference_norm_sq
    ).to(reference_float.dtype)
    grad_reference = grad_reference + (
        alpha_float
        * dot_grad_reference
        * (2.0 * projection_numerator)
        / reference_norm_sq.pow(2)
        * reference_float
        * denominator_is_live
    )
    grad_alpha = -(projection_float * dot_grad_reference)
    return grad_branch, grad_reference, grad_alpha


def _polynomial_branch_backward(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    normalizer: torch.Tensor,
    order: int,
) -> torch.Tensor:
    width = gate.shape[-1]
    gate_float = gate.float()
    grad_float = grad_output.float()
    normalizer_float = normalizer.float()
    power = gate_float.pow(order)
    dot = (grad_float * power).sum(-1, keepdim=True)
    direct = (
        float(order)
        * normalizer_float
        * grad_float
        * gate_float.pow(order - 1)
    )
    correction = (
        float(order)
        / width
        * normalizer_float.pow(3)
        * dot
        * gate_float.pow(2 * order - 1)
    )
    return direct - correction


def gupn_backward_reference(
    grad_output: torch.Tensor,
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row_multiplier: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column_multiplier: torch.Tensor,
    dropout_multiplier: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    proj_eps: float = 1.0e-6,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Analytical GUPN backward used as the device-kernel correctness oracle.

    ``dropout_multiplier`` contains the replayed mask divided by ``1-p``.  Keeping
    RNG outside this function lets the CuTe backward generate Philox values in
    registers without storing a full mask between forward and backward.
    """
    if gate0.shape != up.shape or gate0.shape != grad_output.shape:
        raise ValueError("gate0, up and grad_output must have identical shapes")
    if dropout_multiplier.shape != gate0.shape:
        raise ValueError("dropout_multiplier must match gate0")
    hidden = gate0.shape[-1]
    _require_shape("gate_row_multiplier", gate_row_multiplier, (hidden,))
    _require_shape("polynorm_weight", polynorm_weight, (3,))
    _require_shape("polynorm_bias", polynorm_bias, (1,))
    _require_shape("exclusive_logits", exclusive_logits, (2,))
    _require_shape("down_column_multiplier", down_column_multiplier, (hidden,))

    gate = gate0 * gate_row_multiplier
    gate_sq = gate.pow(2)
    gate_cu = gate * gate_sq
    inv1 = gate_sq.mean(-1, keepdim=True).add(eps).rsqrt()
    inv2 = (gate_sq * gate_sq).mean(-1, keepdim=True).add(eps).rsqrt()
    inv3 = (gate_cu * gate_cu).mean(-1, keepdim=True).add(eps).rsqrt()
    x1 = gate * inv1
    x2 = gate_sq * inv2
    x3 = gate_cu * inv3

    alpha2, alpha3 = torch.sigmoid(exclusive_logits).unbind()
    x1_float = x1.float()
    reference_norm_sq_unclamped = x1_float.pow(2).sum(-1, keepdim=True)
    reference_norm_sq = reference_norm_sq_unclamped.clamp_min(proj_eps)
    projection2 = (
        (x2.float() * x1_float).sum(-1, keepdim=True) / reference_norm_sq
    ).to(gate.dtype)
    projection3 = (
        (x3.float() * x1_float).sum(-1, keepdim=True) / reference_norm_sq
    ).to(gate.dtype)
    residual2 = x2 - alpha2.to(gate.dtype) * projection2 * x1
    residual3 = x3 - alpha3.to(gate.dtype) * projection3 * x1
    exclusive2 = residual2 * residual2.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    exclusive3 = residual3 * residual3.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()

    activation = (
        polynorm_weight[0] * exclusive3
        + polynorm_weight[1] * exclusive2
        + polynorm_weight[2] * x1
        + polynorm_bias
    )
    dropped_activation = activation * dropout_multiplier
    grad_output_float = grad_output.float()
    up_float = up.float()
    column_float = down_column_multiplier.float()
    dropout_float = dropout_multiplier.float()
    grad_activation = (
        grad_output_float * column_float * up_float * dropout_float
    )
    grad_up = (
        grad_output_float * column_float * dropped_activation.float()
    )
    grad_column = (
        grad_output_float * (dropped_activation * up).float()
    ).reshape(-1, hidden).sum(0)

    grad_weight = torch.stack(
        (
            (grad_activation * exclusive3.float()).sum(),
            (grad_activation * exclusive2.float()).sum(),
            (grad_activation * x1.float()).sum(),
        )
    )
    grad_bias = grad_activation.sum().reshape_as(polynorm_bias)

    grad_exclusive3 = grad_activation * polynorm_weight[0].float()
    grad_exclusive2 = grad_activation * polynorm_weight[1].float()
    grad_x1 = grad_activation * polynorm_weight[2].float()
    grad_residual2 = _rms_normalize_backward(grad_exclusive2, residual2, eps)
    grad_residual3 = _rms_normalize_backward(grad_exclusive3, residual3, eps)

    grad_x2, grad_x1_from2, grad_alpha2 = _exclusive_backward(
        grad_residual2,
        x2,
        x1,
        alpha2,
        projection2,
        reference_norm_sq,
        reference_norm_sq_unclamped,
    )
    grad_x3, grad_x1_from3, grad_alpha3 = _exclusive_backward(
        grad_residual3,
        x3,
        x1,
        alpha3,
        projection3,
        reference_norm_sq,
        reference_norm_sq_unclamped,
    )
    grad_x1 = grad_x1 + grad_x1_from2 + grad_x1_from3

    grad_gate = (
        _polynomial_branch_backward(grad_x1, gate, inv1, 1)
        + _polynomial_branch_backward(grad_x2, gate, inv2, 2)
        + _polynomial_branch_backward(grad_x3, gate, inv3, 3)
    )
    grad_gate0 = grad_gate * gate_row_multiplier.float()
    grad_gate_row = (
        grad_gate * gate0.float()
    ).reshape(-1, hidden).sum(0)

    grad_alpha = torch.stack((grad_alpha2.sum(), grad_alpha3.sum()))
    alpha = torch.stack((alpha2, alpha3)).float()
    grad_logits = grad_alpha * alpha * (1.0 - alpha)
    return (
        grad_gate0.to(gate0.dtype),
        grad_up.to(up.dtype),
        grad_gate_row.to(gate_row_multiplier.dtype),
        grad_weight.to(polynorm_weight.dtype),
        grad_bias.to(polynorm_bias.dtype),
        grad_logits.to(exclusive_logits.dtype),
        grad_column.to(down_column_multiplier.dtype),
    )


def neollm_mlp_reference(
    x: torch.Tensor,
    *,
    fan_weight: torch.Tensor,
    fan_bias: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    fan_periodic_dim: int,
    gate_row_multiplier: torch.Tensor | None = None,
    down_column_multiplier: torch.Tensor | None = None,
    down_row_multiplier: torch.Tensor | None = None,
    exclusive_logits: torch.Tensor | None = None,
    eps: float = 1.0e-6,
    proj_eps: float = 1.0e-6,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """Evaluate the exact NeoLLM MLP graph used as the CuTe correctness oracle.

    The function returns only ``Y``.  In particular it does not expose FAN, gate,
    up or PolyNorm intermediates as auxiliary outputs that AOTAutograd could retain.
    All multipliers remain explicit parameters and every reduction in exclusive
    PolyNorm follows :func:`polynorm_reference`.
    """
    dropout_p = float(dropout_p)
    _validate_mlp_contract(
        x,
        fan_weight,
        fan_bias,
        gate_weight,
        up_weight,
        down_weight,
        polynorm_weight,
        polynorm_bias,
        gate_row_multiplier,
        down_column_multiplier,
        down_row_multiplier,
        exclusive_logits,
        fan_periodic_dim,
        dropout_p,
    )

    fan = fan_reference(
        x,
        fan_weight,
        fan_bias,
        periodic_dim=fan_periodic_dim,
    )
    gate0 = F.linear(fan, gate_weight)
    up = F.linear(fan, up_weight)
    down_input = gupn_reference(
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
        training=training,
    )
    output = F.linear(down_input, down_weight)
    if down_row_multiplier is not None:
        output = output * down_row_multiplier
    return output


__all__ = [
    "fan_reference",
    "gupn_backward_reference",
    "gupn_reference",
    "neollm_mlp_reference",
]
