"""Contract and numerical tests for the opt-in Leviathan-JTok extension.

CUDA-only tests are intentionally skipped on the local Windows checkout.  The
same file runs in the remote CUDA environment, where it additionally verifies
the Triton custom-op, autograd, ``torch.compile`` and ``opcheck`` boundaries.
The tests keep MEAP validity separate from CCE/MiLe masks; the model-level
integration owns that distinction, while this package only consumes the
already-computed ``valid_mask``.
"""

from __future__ import annotations

import math

import pytest
import torch

from cut_cross_entropy.leviathan import (
    jtok_apply,
    jtokm_apply,
    jtokm_auxiliary_loss,
    jtokm_routing_stats,
)


def _inputs(
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    n: int = 19,
    hidden: int = 13,
    d_seed: int = 5,
    knots: int = 7,
    modes: int = 3,
    experts: int = 4,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(1729)
    delta = torch.randn(n, hidden, device=device, dtype=dtype, generator=generator)
    z = torch.sigmoid(
        torch.randn(n, d_seed, device=device, dtype=dtype, generator=generator)
    )
    coeff = torch.randn(
        experts, modes, d_seed, knots, device=device, dtype=dtype, generator=generator
    ) * 0.15
    spline_out = torch.randn(
        experts, modes, hidden, device=device, dtype=dtype, generator=generator
    ) * 0.2
    residual_out = torch.randn(
        experts, d_seed, hidden, device=device, dtype=dtype, generator=generator
    ) * 0.2
    scaler = torch.randn(hidden, device=device, dtype=dtype, generator=generator) * 0.1
    router_state = torch.randn(n, hidden, device=device, dtype=dtype, generator=generator)
    router_weight = torch.randn(
        experts, hidden, device=device, dtype=dtype, generator=generator
    ) * 0.2
    grid = torch.linspace(0.0, 1.0, knots, device=device, dtype=torch.float32)
    return {
        "delta": delta,
        "z": z,
        "coeff": coeff,
        "spline_out": spline_out,
        "residual_out": residual_out,
        "scaler": scaler,
        "router_state": router_state,
        "router_weight": router_weight,
        "grid": grid,
    }


def test_jtok_torch_path_handles_noncontiguous_inputs_and_mask() -> None:
    values = _inputs(n=18)
    delta = values["delta"].view(2, 9, -1).transpose(0, 1)
    z = values["z"].view(2, 9, -1).transpose(0, 1)
    assert not delta.is_contiguous()
    assert not z.is_contiguous()
    valid = torch.ones(delta.shape[:-1], dtype=torch.bool)
    valid[1, 0] = False

    output = jtok_apply(
        delta,
        z,
        values["coeff"][:1],
        values["spline_out"][:1],
        values["residual_out"][:1],
        values["scaler"],
        values["grid"],
        valid_mask=valid,
        backend="torch",
    )

    assert output.shape == delta.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[1, 0], delta[1, 0], rtol=0, atol=0)
    assert not torch.equal(output[0, 0], delta[0, 0])


def test_jtokm_torch_path_is_dense_reference_and_returns_metrics() -> None:
    values = _inputs()
    valid = torch.ones(values["delta"].shape[0], dtype=torch.bool)
    valid[[2, 11]] = False
    output, stats = jtokm_apply(
        values["delta"],
        values["z"],
        values["router_state"],
        values["coeff"],
        values["spline_out"],
        values["residual_out"],
        values["scaler"],
        values["router_weight"],
        values["grid"],
        top_k=2,
        valid_mask=valid,
        compute_aux=True,
        backend="torch",
    )

    assert stats is not None
    assert output.shape == values["delta"].shape
    assert torch.isfinite(output).all()
    assert stats["valid_tokens"].item() == 17
    assert stats["expert_probability"].shape == (4,)
    assert stats["expert_fraction"].shape == (4,)
    assert not stats["p_sum"].requires_grad


def test_jtokm_auxiliary_loss_matches_definition() -> None:
    logits = torch.tensor(
        [[2.0, -1.0, 0.5], [0.0, 1.0, -2.0], [1.0, -3.0, 0.25]],
        dtype=torch.float32,
        requires_grad=True,
    )
    expert_idx = torch.tensor([[0, 2], [1, 0], [2, 1]])
    valid = torch.tensor([True, False, True])
    stats = jtokm_routing_stats(logits, expert_idx, valid)
    loss = jtokm_auxiliary_loss(
        stats,
        num_experts=3,
        top_k=2,
        weight=0.07,
    )

    probabilities = torch.sigmoid(logits).detach()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    p_i = probabilities[valid].sum(dim=0) / 2.0
    hard = torch.zeros_like(probabilities)
    hard.scatter_(1, expert_idx, 1.0)
    f_i = hard[valid].sum(dim=0) / (2.0 * 2.0)
    expected = 0.07 * 3.0 * (p_i * f_i).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_jtokm_all_invalid_batch_is_neutral_and_finite() -> None:
    values = _inputs(n=11)
    valid = torch.zeros(11, dtype=torch.bool)
    output, stats = jtokm_apply(
        values["delta"],
        values["z"],
        values["router_state"],
        values["coeff"],
        values["spline_out"],
        values["residual_out"],
        values["scaler"],
        values["router_weight"],
        values["grid"],
        top_k=2,
        valid_mask=valid,
        compute_aux=True,
        backend="torch",
    )

    assert stats is not None
    torch.testing.assert_close(output, values["delta"], rtol=0, atol=0)
    assert stats["valid_tokens"].item() == 0
    assert stats["active_experts"].item() == 0
    assert stats["load_entropy"].item() == 0
    aux = jtokm_auxiliary_loss(stats, num_experts=4, top_k=2, weight=1.0)
    assert aux.item() == 0.0
    assert torch.isfinite(torch.stack([stats["load_cv"], stats["router_entropy"], aux])).all()


def test_jtokm_backward_reaches_router_and_surface_parameters() -> None:
    values = _inputs()
    trainable = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in values.items()
        if key != "grid"
    }
    output, stats = jtokm_apply(
        trainable["delta"],
        trainable["z"],
        trainable["router_state"],
        trainable["coeff"],
        trainable["spline_out"],
        trainable["residual_out"],
        trainable["scaler"],
        trainable["router_weight"],
        values["grid"],
        top_k=2,
        compute_aux=True,
        backend="torch",
    )
    assert stats is not None
    loss = output.float().square().mean() + jtokm_auxiliary_loss(
        stats,
        num_experts=4,
        top_k=2,
        weight=1e-2,
    )
    loss.backward()

    for key in (
        "delta",
        "z",
        "router_state",
        "coeff",
        "spline_out",
        "residual_out",
        "scaler",
        "router_weight",
    ):
        gradient = trainable[key].grad
        assert gradient is not None, key
        assert torch.isfinite(gradient).all(), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_jtokm_triton_matches_torch_reference_on_odd_geometry(dtype: torch.dtype) -> None:
    values = _inputs(device="cuda", dtype=dtype, n=37, hidden=29, d_seed=6, knots=9, modes=5)
    valid = torch.ones(37, device="cuda", dtype=torch.bool)
    valid[::7] = False
    reference, _ = jtokm_apply(
        values["delta"],
        values["z"],
        values["router_state"],
        values["coeff"],
        values["spline_out"],
        values["residual_out"],
        values["scaler"],
        values["router_weight"],
        values["grid"],
        top_k=2,
        valid_mask=valid,
        backend="torch",
    )
    actual, _ = jtokm_apply(
        values["delta"],
        values["z"],
        values["router_state"],
        values["coeff"],
        values["spline_out"],
        values["residual_out"],
        values["scaler"],
        values["router_weight"],
        values["grid"],
        top_k=2,
        valid_mask=valid,
        backend="triton",
    )
    torch.cuda.synchronize()
    tolerance = 8e-2 if dtype == torch.bfloat16 else 3e-2
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtokm_triton_general_path_handles_hidden_above_single_tile() -> None:
    """Keep the two-stage reduction correct when hidden needs two tiles."""
    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=9,
        hidden=257,
        d_seed=4,
        knots=5,
        modes=2,
    )
    valid = torch.ones(9, device="cuda", dtype=torch.bool)
    valid[::3] = False
    reference, _ = jtokm_apply(
        values["delta"],
        values["z"],
        values["router_state"],
        values["coeff"],
        values["spline_out"],
        values["residual_out"],
        values["scaler"],
        values["router_weight"],
        values["grid"],
        top_k=2,
        valid_mask=valid,
        backend="torch",
    )
    actual, _ = jtokm_apply(
        values["delta"],
        values["z"],
        values["router_state"],
        values["coeff"],
        values["spline_out"],
        values["residual_out"],
        values["scaler"],
        values["router_weight"],
        values["grid"],
        top_k=2,
        valid_mask=valid,
        backend="triton",
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=8e-2, atol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtokm_triton_backward_reaches_all_trainable_inputs() -> None:
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=17, hidden=13, d_seed=5)
    trainable = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in values.items()
        if key != "grid"
    }
    output, _ = jtokm_apply(
        trainable["delta"],
        trainable["z"],
        trainable["router_state"],
        trainable["coeff"],
        trainable["spline_out"],
        trainable["residual_out"],
        trainable["scaler"],
        trainable["router_weight"],
        values["grid"],
        top_k=2,
        backend="triton",
    )
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    for key in (
        "delta",
        "z",
        "router_state",
        "coeff",
        "spline_out",
        "residual_out",
        "scaler",
        "router_weight",
    ):
        gradient = trainable[key].grad
        assert gradient is not None, key
        assert torch.isfinite(gradient).all(), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtokm_triton_backward_matches_torch_reference() -> None:
    """Compare the compact Triton backward, not only gradient finiteness."""
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=11, hidden=13, d_seed=5)
    valid = torch.ones(11, device="cuda", dtype=torch.bool)
    valid[[2, 7]] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)

    def run(backend: str) -> dict[str, torch.Tensor]:
        trainable = {
            key: value.detach().clone().requires_grad_(True)
            for key, value in values.items()
            if key != "grid"
        }
        output, _ = jtokm_apply(
            trainable["delta"],
            trainable["z"],
            trainable["router_state"],
            trainable["coeff"],
            trainable["spline_out"],
            trainable["residual_out"],
            trainable["scaler"],
            trainable["router_weight"],
            values["grid"],
            top_k=2,
            valid_mask=valid,
            backend=backend,  # type: ignore[arg-type]
        )
        (output.float() * probe).sum().backward()
        return {key: trainable[key].grad.detach().clone() for key in trainable}

    reference = run("torch")
    actual = run("triton")
    for key in reference:
        assert torch.isfinite(actual[key]).all(), key
        torch.testing.assert_close(
            actual[key].float(),
            reference[key].float(),
            rtol=0.18,
            atol=0.12,
            msg=f"gradient mismatch for {key}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtok_custom_opcheck_and_compiles() -> None:
    from cut_cross_entropy.leviathan.jtok import _jtok_forward_op

    values = _inputs(device="cuda", dtype=torch.bfloat16, n=17, hidden=13, d_seed=5)
    delta = values["delta"].requires_grad_(True)
    args = (
        delta,
        values["z"],
        values["coeff"][:1],
        values["spline_out"][:1],
        values["residual_out"][:1],
        values["scaler"],
        values["grid"],
        torch.zeros(17, 1, device="cuda", dtype=torch.long),
        torch.ones(17, 1, device="cuda", dtype=torch.float32),
        torch.empty(0, device="cuda", dtype=torch.bool),
        1e-6,
    )
    result = torch.library.opcheck(
        _jtok_forward_op,
        args,
        test_utils=("test_schema", "test_autograd_registration", "test_faketensor"),
        rtol=8e-2,
        atol=8e-2,
    )
    assert set(result.values()) == {"SUCCESS"}

    def run(x: torch.Tensor) -> torch.Tensor:
        return jtok_apply(
            x,
            values["z"],
            values["coeff"][:1],
            values["spline_out"][:1],
            values["residual_out"][:1],
            values["scaler"],
            values["grid"],
            backend="triton",
        )

    compiled = torch.compile(run, fullgraph=True, mode="max-autotune")
    output = compiled(delta)
    torch.cuda.synchronize()
    assert output.shape == delta.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_jtokm_routing_metrics_support_all_top_k(top_k: int) -> None:
    values = _inputs(n=23)
    output, stats = jtokm_apply(
        values["delta"],
        values["z"],
        values["router_state"],
        values["coeff"],
        values["spline_out"],
        values["residual_out"],
        values["scaler"],
        values["router_weight"],
        values["grid"],
        top_k=top_k,
        compute_aux=True,
        backend="torch",
    )
    assert stats is not None
    assert output.shape == values["delta"].shape
    assert math.isfinite(float(stats["load_cv"]))
    assert 0 <= float(stats["active_experts"]) <= 4
