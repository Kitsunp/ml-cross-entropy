from __future__ import annotations

import pytest
import torch

from cut_cross_entropy.leviathan import (
    LeviathanConfig,
    LeviathanGenerator,
    leviathan_embedding,
)
from cut_cross_entropy.leviathan.runtime_policy import use_dot_specialization

_PARAMETER_NAMES = (
    "codebooks",
    "head_proj_weight",
    "head_norm_weight",
    "head_norm_bias",
    "head_spline_delta",
    "head_out_weight",
)


def test_dot_specialization_policy_is_narrow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEV_DOT", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    device = torch.device("cuda")

    assert use_dot_specialization(
        device, d_seed=128, num_knots=16, krank=64
    )
    assert not use_dot_specialization(
        device, d_seed=256, num_knots=16, krank=64
    )
    assert not use_dot_specialization(
        device, d_seed=128, num_knots=8, krank=64
    )
    assert not use_dot_specialization(
        device, d_seed=128, num_knots=16, krank=32
    )
    assert not use_dot_specialization(
        torch.device("cpu"), d_seed=128, num_knots=16, krank=64
    )


def test_dot_specialization_respects_diagnostic_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEV_DOT", "0")
    assert not use_dot_specialization(
        torch.device("cuda"), d_seed=128, num_knots=16, krank=64
    )
    monkeypatch.setenv("LEV_DOT", "1")
    assert use_dot_specialization(
        torch.device("cpu"), d_seed=128, num_knots=16, krank=64
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm120_auto_dot_backward_matches_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("automatic dot specialization is restricted to SM120+")

    torch.manual_seed(20260823)
    torch.cuda.manual_seed_all(20260823)
    torch.set_float32_matmul_precision("high")
    cfg = LeviathanConfig(
        vocab_size=50_304,
        hidden_size=512,
        generator_d_seed=128,
        generator_num_modes=8,
        generator_num_knots=16,
        generator_k=3,
        generator_krank=64,
        dtype=torch.bfloat16,
    )
    generator = LeviathanGenerator(cfg)
    source = {
        name: getattr(generator, name).detach().cuda()
        for name in _PARAMETER_NAMES
    }
    knot_grid = generator.knot_grid.detach().cuda()
    ids = torch.randint(cfg.vocab_size, (257,), device="cuda")
    grad_output = (
        torch.randn(257, cfg.hidden_size, device="cuda")
        / cfg.hidden_size**0.5
    ).to(torch.bfloat16)

    def run(*, force_exact: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if force_exact:
            monkeypatch.setenv("LEV_DOT", "0")
        else:
            monkeypatch.delenv("LEV_DOT", raising=False)
        monkeypatch.delenv("LEV_PREMUL_DMM", raising=False)
        params = {
            name: value.detach().clone().requires_grad_()
            for name, value in source.items()
        }
        params["knot_grid"] = knot_grid
        output = leviathan_embedding(ids, params, cfg)
        output.backward(grad_output)
        return output.detach().float(), {
            name: params[name].grad.detach().float()
            for name in _PARAMETER_NAMES
        }

    exact_output, exact_grads = run(force_exact=True)
    auto_output, auto_grads = run(force_exact=False)

    def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
        delta = torch.linalg.vector_norm((actual - expected).double())
        denom = torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
        return float(delta / denom)

    assert relative_l2(auto_output, exact_output) < 3e-4
    for name in _PARAMETER_NAMES:
        assert torch.isfinite(auto_grads[name]).all()
        assert relative_l2(auto_grads[name], exact_grads[name]) < 1e-3, name
