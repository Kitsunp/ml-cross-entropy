from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchao")

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_linear import (
    Float8Linear,
    matmul_with_hp_or_float8_args,
)

from cut_cross_entropy.mlp import fp8_gupn, gupn
from cut_cross_entropy.mlp.fp8 import (
    _CONFIG,
    _MM_CONFIG,
    _dual_backward,
    _dual_forward,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def _inputs(
    rows: int = 16,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(101)
    device = torch.device("cuda")
    fan_features = 544
    intermediate = 1536
    return (
        torch.randn(
            rows,
            fan_features,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            intermediate,
            fan_features,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            intermediate,
            fan_features,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            intermediate,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            3,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            1,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            2,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            intermediate,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
    )


def test_dual_fp8_projection_is_bit_exact_to_two_torchao_linears() -> None:
    fan, gate_weight, up_weight, *_ = _inputs(rows=512)
    config = Float8LinearConfig()
    gate = Float8Linear(
        fan.shape[-1],
        gate_weight.shape[0],
        bias=False,
        device=fan.device,
        dtype=fan.dtype,
        config=config,
    )
    up = Float8Linear(
        fan.shape[-1],
        up_weight.shape[0],
        bias=False,
        device=fan.device,
        dtype=fan.dtype,
        config=config,
    )
    gate.weight = torch.nn.Parameter(gate_weight.detach().clone())
    up.weight = torch.nn.Parameter(up_weight.detach().clone())

    expected_gate = gate(fan)
    expected_up = up(fan)
    actual_gate, actual_up = _dual_forward(
        fan.detach(),
        gate_weight.detach(),
        up_weight.detach(),
    )
    assert torch.equal(actual_gate, expected_gate)
    assert torch.equal(actual_up, expected_up)

    grad_gate = torch.randn_like(expected_gate)
    grad_up = torch.randn_like(expected_up)
    (expected_gate * grad_gate + expected_up * grad_up).sum().backward()
    actual_gradients = _dual_backward(
        grad_gate,
        grad_up,
        fan.detach(),
        gate_weight.detach(),
        up_weight.detach(),
    )
    expected_gradients = (fan.grad, gate.weight.grad, up.weight.grad)
    for actual, expected in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert torch.equal(actual, expected)


def test_fp8_gupn_matches_staged_torchao_route() -> None:
    tensors = _inputs()
    fan, gate_weight, up_weight, *gupn_parameters = tensors
    clones = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)
    (
        fan_ref,
        gate_weight_ref,
        up_weight_ref,
        *gupn_parameters_ref,
    ) = clones

    actual = fp8_gupn(*tensors, dropout_p=0.0)
    gate_ref = matmul_with_hp_or_float8_args.apply(
        fan_ref,
        gate_weight_ref.t(),
        _MM_CONFIG,
        _CONFIG,
    )
    up_ref = matmul_with_hp_or_float8_args.apply(
        fan_ref,
        up_weight_ref.t(),
        _MM_CONFIG,
        _CONFIG,
    )
    expected = gupn(
        gate_ref,
        up_ref,
        *gupn_parameters_ref,
        dropout_p=0.0,
    )
    assert torch.equal(actual, expected)

    grad_output = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, tensors, grad_output)
    expected_gradients = torch.autograd.grad(expected, clones, grad_output)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=0.0,
            atol=0.0,
        )


def test_fp8_gupn_does_not_save_gate_or_up_for_backward() -> None:
    tensors = _inputs(rows=32)
    saved_shapes: list[tuple[int, ...]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = fp8_gupn(*tensors, dropout_p=0.1)
    output.float().square().mean().backward()

    forbidden = (tensors[0].shape[0], tensors[1].shape[0])
    assert forbidden not in saved_shapes
    assert tuple(tensors[0].shape) in saved_shapes
    assert tuple(tensors[1].shape) in saved_shapes


def test_fp8_gupn_runs_five_compiled_training_steps() -> None:
    tensors = _inputs(rows=32)

    def function(*values: torch.Tensor) -> torch.Tensor:
        return fp8_gupn(*values, dropout_p=0.1).float().square().mean()

    compiled = torch.compile(
        function,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
    )
    losses: list[float] = []
    for _ in range(5):
        marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if marker is not None:
            marker()
        for tensor in tensors:
            tensor.grad = None
        loss = compiled(*tensors)
        loss.backward()
        losses.append(float(loss.detach()))
        assert all(tensor.grad is not None for tensor in tensors)
        assert all(torch.isfinite(tensor.grad).all() for tensor in tensors)
    assert len(losses) == 5


def test_fp8_gupn_rejects_invalid_grad_weight_geometry() -> None:
    tensors = _inputs(rows=16)
    invalid = (tensors[0][:8],) + tensors[1:]
    with pytest.raises(ValueError, match="row count"):
        fp8_gupn(*invalid)
