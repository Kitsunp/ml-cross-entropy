from __future__ import annotations

import torch

from cut_cross_entropy.polynorm import compiler as polynorm_compiler
from cut_cross_entropy.polynorm import polynorm, polynorm_reference


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
