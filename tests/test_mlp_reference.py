from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy.mlp import fan_reference, neollm_mlp_reference
from cut_cross_entropy.polynorm import polynorm_reference


def _parameters(
    *,
    hidden_size: int = 8,
    fan_projection_size: int = 9,
    periodic_dim: int = 2,
    intermediate_size: int = 12,
) -> dict[str, torch.Tensor | int]:
    fan_size = fan_projection_size + periodic_dim
    tensors = {
        "fan_weight": torch.randn(fan_projection_size, hidden_size),
        "fan_bias": torch.randn(fan_projection_size),
        "gate_weight": torch.randn(intermediate_size, fan_size),
        "up_weight": torch.randn(intermediate_size, fan_size),
        "down_weight": torch.randn(hidden_size, intermediate_size),
        "polynorm_weight": torch.randn(3),
        "polynorm_bias": torch.randn(1),
        "gate_row_multiplier": torch.randn(intermediate_size),
        "down_column_multiplier": torch.randn(intermediate_size),
        "down_row_multiplier": torch.randn(hidden_size),
        "exclusive_logits": torch.randn(2),
    }
    for tensor in tensors.values():
        tensor.requires_grad_()
    return {**tensors, "fan_periodic_dim": periodic_dim}


def _literal_graph(
    x: torch.Tensor,
    params: dict[str, torch.Tensor | int],
    *,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    fan_weight = params["fan_weight"]
    fan_bias = params["fan_bias"]
    periodic_dim = params["fan_periodic_dim"]
    assert isinstance(fan_weight, torch.Tensor)
    assert isinstance(fan_bias, torch.Tensor)
    assert isinstance(periodic_dim, int)

    projected = F.linear(x, fan_weight, fan_bias)
    periodic, passthrough = torch.split(
        projected,
        [periodic_dim, projected.shape[-1] - periodic_dim],
        dim=-1,
    )
    fan = torch.cat((torch.cos(periodic), torch.sin(periodic), passthrough), dim=-1)
    gate = F.linear(fan, params["gate_weight"]) * params["gate_row_multiplier"]
    up = F.linear(fan, params["up_weight"])
    activation = polynorm_reference(
        gate,
        params["polynorm_weight"],
        params["polynorm_bias"],
        exclusive_logits=params["exclusive_logits"],
    )
    if training and dropout_p:
        activation = F.dropout(activation, p=dropout_p, training=True)
    down_input = activation * up * params["down_column_multiplier"]
    return F.linear(down_input, params["down_weight"]) * params["down_row_multiplier"]


def _tensor_params(params: dict[str, torch.Tensor | int]) -> list[torch.Tensor]:
    return [value for value in params.values() if isinstance(value, torch.Tensor)]


def test_fan_reference_matches_model_operation_order() -> None:
    x = torch.randn(2, 3, 8)
    weight = torch.randn(9, 8)
    bias = torch.randn(9)

    projected = F.linear(x, weight, bias)
    expected = torch.cat(
        (
            torch.cos(projected[..., :2]),
            torch.sin(projected[..., :2]),
            projected[..., 2:],
        ),
        dim=-1,
    )

    torch.testing.assert_close(
        fan_reference(x, weight, bias, periodic_dim=2),
        expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("exclusive", [False, True])
def test_mlp_reference_matches_literal_forward_and_backward(exclusive: bool) -> None:
    torch.manual_seed(11)
    x_actual = torch.randn(2, 3, 8, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_()
    actual_params = _parameters()
    expected_params = {
        name: value.detach().clone().requires_grad_()
        if isinstance(value, torch.Tensor)
        else value
        for name, value in actual_params.items()
    }
    if not exclusive:
        actual_params["exclusive_logits"] = None
        expected_params["exclusive_logits"] = None

    actual = neollm_mlp_reference(x_actual, **actual_params)
    expected = _literal_graph(x_expected, expected_params)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    grad = torch.randn_like(actual)
    actual_inputs = [x_actual, *_tensor_params(actual_params)]
    expected_inputs = [x_expected, *_tensor_params(expected_params)]
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad)
    expected_grads = torch.autograd.grad(expected, expected_inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0.0, atol=0.0)


def test_exclusive_logits_are_trainable_in_full_mlp() -> None:
    torch.manual_seed(17)
    x = torch.randn(4, 5, 8, requires_grad=True)
    params = _parameters()

    output = neollm_mlp_reference(x, **params)
    output.float().square().mean().backward()

    exclusive_logits = params["exclusive_logits"]
    assert isinstance(exclusive_logits, torch.Tensor)
    assert exclusive_logits.grad is not None
    assert torch.isfinite(exclusive_logits.grad).all()
    assert exclusive_logits.grad.abs().sum() > 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in _tensor_params(params)
    )


def test_dropout_is_seed_reproducible_and_disabled_for_inference() -> None:
    torch.manual_seed(23)
    x = torch.randn(8, 4, 8)
    params = _parameters()

    torch.manual_seed(101)
    first = neollm_mlp_reference(x, **params, dropout_p=0.25)
    torch.manual_seed(101)
    second = neollm_mlp_reference(x, **params, dropout_p=0.25)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    torch.manual_seed(101)
    inference_with_p = neollm_mlp_reference(
        x,
        **params,
        dropout_p=0.25,
        training=False,
    )
    inference_without_p = neollm_mlp_reference(
        x,
        **params,
        dropout_p=0.0,
        training=False,
    )
    torch.testing.assert_close(inference_with_p, inference_without_p, rtol=0.0, atol=0.0)


def test_contract_rejects_a_down_weight_that_does_not_restore_hidden_size() -> None:
    x = torch.randn(2, 8)
    params = _parameters()
    params["down_weight"] = torch.randn(9, 12)

    with pytest.raises(ValueError, match="down_weight must restore"):
        neollm_mlp_reference(x, **params)
