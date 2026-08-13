from __future__ import annotations

import math

import pytest
import torch

from cut_cross_entropy.mlp import (
    _cute_gupn,
    gupn,
    gupn_backward_reference,
    gupn_reference,
    gupn_uses_cute,
)
from cut_cross_entropy.mlp import _cute_gupn_backward
from cut_cross_entropy.polynorm import polynorm_reference
from cut_cross_entropy.mlp import compiler as gupn_compiler


def _limit_test_vram(device: torch.device, gib: float = 10.0) -> None:
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(gib * 1024**3 / total, 1.0),
        device,
    )


def _philox_mask(
    count: int,
    seeds: torch.Tensor,
    dropout_p: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask32 = (1 << 32) - 1
    threshold = math.ceil(dropout_p * (1 << 32))
    seed0, seed1, seed2, seed3 = [int(value) for value in seeds]
    keep: list[bool] = []
    for counter in range((count + 3) // 4):
        c0 = counter & mask32
        c1 = (counter >> 32) & mask32
        c2, c3 = seed2, seed3
        k0, k1 = seed0, seed1
        for _ in range(10):
            product0 = c0 * 0xD2511F53
            product1 = c2 * 0xCD9E8D57
            lo0, hi0 = product0 & mask32, (product0 >> 32) & mask32
            lo1, hi1 = product1 & mask32, (product1 >> 32) & mask32
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
            k0 = (k0 + 0x9E3779B9) & mask32
            k1 = (k1 + 0xBB67AE85) & mask32
        keep.extend(value >= threshold for value in (c0, c1, c2, c3))
    return torch.tensor(keep[:count], device=device, dtype=torch.bool)


def _reference(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    column: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    gate = gate0 * gate_row
    activation = polynorm_reference(
        gate,
        weight,
        bias,
        exclusive_logits=exclusive_logits,
    )
    if dropout_p:
        keep = _philox_mask(
            activation.numel(),
            seeds,
            dropout_p,
            device=activation.device,
        ).view_as(activation)
        activation = activation * keep * (1.0 / (1.0 - dropout_p))
    return activation * up * column


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute_gupn.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.1])
def test_exclusive_gupn_forward_matches_reference(dropout_p: float) -> None:
    torch.manual_seed(71)
    device = torch.device("cuda")
    shape = (8, 1536)
    gate0 = torch.randn(shape, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate0)
    gate_row = torch.randn(shape[1], device=device, dtype=torch.bfloat16)
    weight = torch.randn(3, device=device, dtype=torch.bfloat16)
    bias = torch.randn(1, device=device, dtype=torch.bfloat16)
    exclusive_logits = torch.randn(2, device=device, dtype=torch.bfloat16)
    column = torch.randn(shape[1], device=device, dtype=torch.bfloat16)
    seeds = torch.tensor(
        [123456789, 987654321, 1122334455, 556677889],
        device=device,
        dtype=torch.int64,
    )

    actual = _cute_gupn.forward(
        gate0,
        up,
        gate_row,
        weight,
        bias,
        exclusive_logits,
        column,
        seeds,
        dropout_p=dropout_p,
        use_xor=True,
    )
    expected = _reference(
        gate0,
        up,
        gate_row,
        weight,
        bias,
        exclusive_logits,
        column,
        seeds,
        dropout_p,
    )

    torch.testing.assert_close(actual, expected, rtol=5.0e-2, atol=5.0e-2)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute_gupn.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_exclusive_gupn_dropout_replays_from_seeds() -> None:
    device = torch.device("cuda")
    gate0 = torch.randn((4, 1536), device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate0)
    gate_row = torch.ones(1536, device=device, dtype=torch.bfloat16)
    weight = torch.ones(3, device=device, dtype=torch.bfloat16) / 3
    bias = torch.zeros(1, device=device, dtype=torch.bfloat16)
    logits = torch.zeros(2, device=device, dtype=torch.bfloat16)
    column = torch.ones(1536, device=device, dtype=torch.bfloat16)
    seeds = torch.tensor([7, 11, 13, 17], device=device, dtype=torch.int64)

    first = _cute_gupn.forward(
        gate0, up, gate_row, weight, bias, logits, column, seeds, dropout_p=0.25
    )
    second = _cute_gupn.forward(
        gate0, up, gate_row, weight, bias, logits, column, seeds, dropout_p=0.25
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    changed_seeds = seeds.clone()
    changed_seeds[0] += 1
    changed = _cute_gupn.forward(
        gate0,
        up,
        gate_row,
        weight,
        bias,
        logits,
        column,
        changed_seeds,
        dropout_p=0.25,
    )
    assert not torch.equal(first, changed)


def test_public_gupn_cpu_fallback_preserves_all_gradients() -> None:
    torch.manual_seed(79)
    gate0 = torch.randn(4, 12, requires_grad=True)
    up = torch.randn(4, 12, requires_grad=True)
    gate_row = torch.randn(12, requires_grad=True)
    weight = torch.randn(3, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)
    logits = torch.randn(2, requires_grad=True)
    column = torch.randn(12, requires_grad=True)
    tensors = (gate0, up, gate_row, weight, bias, logits, column)
    clones = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)

    actual = gupn(*tensors)
    expected = gupn_reference(*clones)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    grad = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, tensors, grad)
    expected_gradients = torch.autograd.grad(expected, clones, grad)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute_gupn.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_public_gupn_training_uses_recomputed_cute_backward() -> None:
    device = torch.device("cuda")
    gate0 = torch.randn((4, 1536), device=device, dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn_like(gate0, requires_grad=True)
    gate_row = torch.ones(1536, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.ones(3, device=device, dtype=torch.bfloat16) / 3).requires_grad_()
    bias = torch.zeros(1, device=device, dtype=torch.bfloat16, requires_grad=True)
    logits = torch.zeros(2, device=device, dtype=torch.bfloat16, requires_grad=True)
    column = torch.ones(1536, device=device, dtype=torch.bfloat16, requires_grad=True)

    assert gupn_uses_cute(gate0, up, gate_row, weight, bias, logits, column)
    output = gupn(gate0, up, gate_row, weight, bias, logits, column)
    output.float().square().mean().backward()
    assert all(
        tensor.grad is not None
        for tensor in (gate0, up, gate_row, weight, bias, logits, column)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute_gupn_backward.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.1])
@pytest.mark.parametrize("use_xor", [False, True])
def test_cute_gupn_backward_matches_analytical_oracle(
    dropout_p: float,
    use_xor: bool,
) -> None:
    torch.manual_seed(89)
    device = torch.device("cuda")
    shape = (5, 1536)
    gate0 = torch.randn(shape, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate0)
    grad_output = torch.randn_like(gate0)
    gate_row = torch.randn(shape[1], device=device, dtype=torch.bfloat16)
    weight = torch.randn(3, device=device, dtype=torch.bfloat16)
    bias = torch.randn(1, device=device, dtype=torch.bfloat16)
    logits = torch.randn(2, device=device, dtype=torch.bfloat16)
    column = torch.randn(shape[1], device=device, dtype=torch.bfloat16)
    seeds = torch.tensor([7, 11, 13, 17], device=device, dtype=torch.int64)
    keep = _philox_mask(
        gate0.numel(),
        seeds,
        dropout_p,
        device=device,
    ).view_as(gate0)
    dropout_multiplier = (
        keep.to(gate0.dtype) / (1.0 - dropout_p)
        if dropout_p
        else torch.ones_like(gate0)
    )

    actual = _cute_gupn_backward.backward(
        grad_output,
        gate0,
        up,
        gate_row,
        weight,
        bias,
        logits,
        column,
        seeds,
        dropout_p=dropout_p,
        use_xor=use_xor,
    )
    expected = gupn_backward_reference(
        grad_output,
        gate0,
        up,
        gate_row,
        weight,
        bias,
        logits,
        column,
        dropout_multiplier,
    )
    names = (
        "gate0",
        "up",
        "gate_row",
        "weight",
        "bias",
        "logits",
        "column",
    )
    for name, actual_gradient, expected_gradient in zip(
        names,
        actual,
        expected,
        strict=True,
    ):
        difference = (
            actual_gradient.float() - expected_gradient.float()
        ).norm()
        scale = expected_gradient.float().norm().clamp_min(1.0)
        assert float(difference / scale) < 3.0e-2, name
        assert torch.isfinite(actual_gradient).all(), name


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute_gupn.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_public_gupn_is_a_fullgraph_inference_custom_op() -> None:
    device = torch.device("cuda")
    gate0 = torch.randn((8, 1536), device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate0)
    gate_row = torch.ones(1536, device=device, dtype=torch.bfloat16)
    weight = torch.ones(3, device=device, dtype=torch.bfloat16) / 3
    bias = torch.zeros(1, device=device, dtype=torch.bfloat16)
    logits = torch.zeros(2, device=device, dtype=torch.bfloat16)
    column = torch.ones(1536, device=device, dtype=torch.bfloat16)

    assert gupn_uses_cute(gate0, up, gate_row, weight, bias, logits, column)

    def function(gate: torch.Tensor, up_value: torch.Tensor) -> torch.Tensor:
        return gupn(
            gate,
            up_value,
            gate_row,
            weight,
            bias,
            logits,
            column,
            dropout_p=0.1,
        )

    compiled = torch.compile(
        function,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
    )
    outputs = []
    for _ in range(5):
        marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if marker is not None:
            marker()
        outputs.append(compiled(gate0, up).clone())
    assert len({float(output.float().sum()) for output in outputs}) == 5
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute_gupn_backward.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_public_gupn_runs_five_compiled_training_steps() -> None:
    torch.manual_seed(97)
    device = torch.device("cuda")
    _limit_test_vram(device)
    shape = (8, 1536)
    gate0 = torch.randn(
        shape,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    up = torch.randn_like(gate0, requires_grad=True)
    gate_row = torch.ones(
        shape[1],
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = (
        torch.ones(3, device=device, dtype=torch.bfloat16) / 3
    ).requires_grad_()
    bias = torch.zeros(
        1,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    logits = torch.zeros(
        2,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    column = torch.ones(
        shape[1],
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    parameters = (gate0, up, gate_row, weight, bias, logits, column)

    def function(
        gate: torch.Tensor,
        up_value: torch.Tensor,
        gate_multiplier: torch.Tensor,
        poly_weight: torch.Tensor,
        poly_bias: torch.Tensor,
        exclusive: torch.Tensor,
        down_multiplier: torch.Tensor,
    ) -> torch.Tensor:
        output = gupn(
            gate,
            up_value,
            gate_multiplier,
            poly_weight,
            poly_bias,
            exclusive,
            down_multiplier,
            dropout_p=0.1,
        )
        return output.float().square().mean()

    compiled = torch.compile(
        function,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
    )
    losses = []
    for _ in range(5):
        marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if marker is not None:
            marker()
        for parameter in parameters:
            parameter.grad = None
        loss = compiled(*parameters)
        loss.backward()
        losses.append(float(loss.detach()))
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(
            torch.isfinite(parameter.grad).all() for parameter in parameters
        )
    assert len(losses) == 5


@pytest.mark.parametrize(
    ("device", "dtype", "rtol", "atol"),
    [
        ("cpu", torch.float32, 2.0e-5, 2.0e-5),
        pytest.param(
            "cuda",
            torch.bfloat16,
            8.0e-2,
            8.0e-2,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is required",
            ),
        ),
    ],
)
def test_analytical_gupn_backward_matches_autograd(
    device: str,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    torch.manual_seed(83)
    shape = (4, 16)
    tensors = (
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape[1], device=device, dtype=dtype, requires_grad=True),
        torch.randn(3, device=device, dtype=dtype, requires_grad=True),
        torch.randn(1, device=device, dtype=dtype, requires_grad=True),
        torch.randn(2, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape[1], device=device, dtype=dtype, requires_grad=True),
    )
    gate0, up, gate_row, weight, bias, logits, column = tensors
    dropout_multiplier = (
        (torch.rand(shape, device=device) >= 0.25).to(dtype) / 0.75
    )
    gate = gate0 * gate_row
    activation = polynorm_reference(
        gate,
        weight,
        bias,
        exclusive_logits=logits,
    )
    expected_output = activation * dropout_multiplier * up * column
    grad_output = torch.randn_like(expected_output)
    expected_gradients = torch.autograd.grad(expected_output, tensors, grad_output)

    actual_gradients = gupn_backward_reference(
        grad_output,
        gate0,
        up,
        gate_row,
        weight,
        bias,
        logits,
        column,
        dropout_multiplier,
    )
    names = (
        "gate0",
        "up",
        "gate_row",
        "weight",
        "bias",
        "logits",
        "column",
    )
    for name, actual, expected in zip(
        names,
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            msg=lambda message, name=name: f"{name}: {message}",
        )
