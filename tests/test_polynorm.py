from __future__ import annotations

import pytest
import torch

from cut_cross_entropy.polynorm import (
    _cute,
    polynorm,
    polynorm_reference,
    polynorm_uses_cute,
)
from cut_cross_entropy.polynorm import compiler as polynorm_compiler


def test_polynorm_cpu_fallback_without_dropout_matches_reference() -> None:
    x = torch.randn(4, 12)
    weight = torch.randn(3)
    bias = torch.randn(1)

    actual = polynorm(x, weight, bias)
    expected = polynorm_reference(x, weight, bias)

    torch.testing.assert_close(actual, expected)


def test_polynorm_cpu_fallback_applies_requested_dropout() -> None:
    x = torch.randn(32, 64)
    weight = torch.ones(3) / 3
    bias = torch.zeros(1)

    torch.manual_seed(123)
    actual = polynorm(x, weight, bias, dropout_p=0.25)

    zero_fraction = (actual == 0).float().mean()
    assert 0.20 < zero_fraction < 0.30


def test_compiled_size_dispatch_keeps_small_expression_visible(monkeypatch) -> None:
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    below = torch.empty(
        polynorm_compiler._COMPILED_CUTE_MIN_ELEMENTS - 1,
        device="meta",
    )
    at_threshold = torch.empty(
        polynorm_compiler._COMPILED_CUTE_MIN_ELEMENTS,
        device="meta",
    )

    assert polynorm_compiler._prefer_compiler_fusion(below)
    assert not polynorm_compiler._prefer_compiler_fusion(at_threshold)


def test_public_dispatch_query_matches_reference_fallback(monkeypatch) -> None:
    x = torch.randn(4, 12, requires_grad=True)
    weight = torch.randn(3, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)
    monkeypatch.setattr(
        polynorm_compiler,
        "_cute_supported",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        polynorm_compiler,
        "_prefer_compiler_fusion",
        lambda _x: True,
    )
    assert not polynorm_uses_cute(x, weight, bias)

    monkeypatch.setattr(
        polynorm_compiler,
        "_prefer_compiler_fusion",
        lambda _x: False,
    )
    assert polynorm_uses_cute(x, weight, bias)

    monkeypatch.setattr(
        polynorm_compiler,
        "_cute_supported",
        lambda *args, **kwargs: False,
    )
    assert not polynorm_uses_cute(x, weight, bias)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_cuda_empty_rows_use_reference_fallback() -> None:
    x = torch.empty((0, 1536), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(3, device="cuda", dtype=torch.bfloat16) / 3
    bias = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
    cache_entries = len(_cute._CACHE)

    output = polynorm(x, weight, bias)

    assert output.shape == x.shape
    assert len(_cute._CACHE) == cache_entries


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_cuda_misaligned_input_and_strided_weight_are_normalized() -> None:
    base_x = torch.randn((5, 12), device="cuda", dtype=torch.bfloat16)
    x = base_x[1:].detach().requires_grad_()
    base_weight = torch.randn(6, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = base_weight[::2]
    bias = torch.randn(1, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert x.is_contiguous() and x.data_ptr() % _cute.DESCRIPTOR_ALIGNMENT != 0
    assert weight.stride() == (2,)

    actual = polynorm(x, weight, bias)
    expected = polynorm_reference(x, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)

    actual_gradients = torch.autograd.grad(
        actual.float().sum(), (x, base_weight, bias)
    )
    expected_gradients = torch.autograd.grad(
        expected.float().sum(), (x, base_weight, bias)
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=5.0e-2,
            atol=5.0e-2,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_cuda_extreme_dropout_and_wide_hidden_fall_back() -> None:
    weight = torch.ones(3, device="cuda", dtype=torch.bfloat16) / 3
    bias = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
    cache_entries = len(_cute._CACHE)

    narrow = torch.full((2, 16), 0.1, device="cuda", dtype=torch.bfloat16)
    extreme = polynorm(narrow, weight, bias, dropout_p=0.9999999999)
    wide = torch.full(
        (1, 65_536),
        0.1,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    wide_output = polynorm(wide, weight, bias)

    assert extreme.shape == narrow.shape
    assert wide_output.shape == wide.shape
    assert len(_cute._CACHE) == cache_entries


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_cuda_inference_uses_constant_size_dummy_stats() -> None:
    x = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16)
    seeds = torch.empty(4, device="cuda", dtype=torch.int64)
    weight = torch.ones(3, device="cuda", dtype=torch.bfloat16) / 3
    bias = torch.zeros(1, device="cuda", dtype=torch.bfloat16)

    output, stats = _cute.forward(
        x,
        seeds,
        weight,
        bias,
        dropout_p=0.0,
        save_stats=False,
    )

    assert stats.shape == (1, 4)
    torch.testing.assert_close(
        output,
        polynorm_reference(x, weight, bias),
        rtol=2.0e-2,
        atol=2.0e-2,
    )


def _polynorm_float64_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x64 = x.double()
    square = x64.square()
    cube = x64 * square
    branches = (
        cube * (cube.square().mean(-1, keepdim=True) + 1.0e-6).rsqrt(),
        square * (square.square().mean(-1, keepdim=True) + 1.0e-6).rsqrt(),
        x64 * (square.mean(-1, keepdim=True) + 1.0e-6).rsqrt(),
    )
    return sum(
        weight[index].double() * branch
        for index, branch in enumerate(branches)
    ) + bias.double()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
@pytest.mark.parametrize(
    "scale",
    (65_536.0, 1_048_576.0, 1.0e13, torch.finfo(torch.float32).max),
)
def test_cuda_scaled_rows_match_float64_forward_and_backward(scale: float) -> None:
    pattern = torch.linspace(
        -1.0,
        1.0,
        1536,
        device="cuda",
        dtype=torch.float32,
    ).reshape(1, -1)
    x = (pattern * scale).detach().requires_grad_()
    weight = torch.tensor(
        (0.37, -0.21, 0.58),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.tensor(
        (0.13,), device="cuda", dtype=torch.float32, requires_grad=True
    )
    grad_output = torch.cos(pattern * 3.0)

    assert polynorm_uses_cute(x, weight, bias)
    actual = polynorm(x, weight, bias)
    assert torch.isfinite(actual).all()
    expected = _polynorm_float64_oracle(x, weight, bias)
    actual_gradients = torch.autograd.grad(
        actual, (x, weight, bias), grad_output
    )
    expected_gradients = torch.autograd.grad(
        expected, (x, weight, bias), grad_output.double()
    )

    assert all(torch.isfinite(gradient).all() for gradient in actual_gradients)
    torch.testing.assert_close(
        actual.double(), expected, rtol=2.0e-5, atol=2.0e-5
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient.double(),
            expected_gradient.double(),
            rtol=2.0e-4,
            atol=2.0e-5,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cute.is_available(),
    reason="CUDA and nvidia-cutlass-dsl are required",
)
def test_cuda_rescaling_boundary_selects_expected_path() -> None:
    seeds = torch.empty(4, device="cuda", dtype=torch.int64)
    weight = torch.ones(3, device="cuda", dtype=torch.float32) / 3
    bias = torch.zeros(1, device="cuda", dtype=torch.float32)
    inputs = torch.tensor(
        ((524_288.0,) * 4, (1_048_576.0,) * 4),
        device="cuda",
        dtype=torch.float32,
    )

    output, stats = _cute.forward(
        inputs,
        seeds,
        weight,
        bias,
        dropout_p=0.0,
        save_stats=True,
    )

    assert torch.isfinite(output).all()
    assert stats[0, 0].item() == 1.0
    assert stats[1, 0].item() == 1.0 / 1_048_576.0
    torch.testing.assert_close(
        output[0], output[1], rtol=2.0e-5, atol=2.0e-5
    )
