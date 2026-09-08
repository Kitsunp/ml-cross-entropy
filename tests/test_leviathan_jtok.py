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
from torch import nn
import cut_cross_entropy.leviathan.jtok as jtok_impl

from cut_cross_entropy.leviathan import (
    jtok_apply,
    jtokm_apply,
    jtokm_auxiliary_loss,
    jtokm_routing_stats,
)
from cut_cross_entropy.leviathan.jtok import (
    _backward_hidden_tile_plan,
    _can_use_route_vectorized_mode_evaluation,
    _can_use_vectorized_mode_evaluation,
    _can_use_vectorized_token_projection,
)
from cut_cross_entropy.linear_cross_entropy import linear_cross_entropy


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


@pytest.mark.parametrize(
    ("d_seed", "knots", "modes", "expected"),
    [
        (128, 16, 4, True),
        (64, 32, 4, True),
        (128, 32, 4, False),
        (256, 16, 4, False),
    ],
)
def test_vectorized_token_projection_uses_geometry_budget(
    d_seed: int,
    knots: int,
    modes: int,
    expected: bool,
) -> None:
    """The dispatch guard is shape-driven, not tied to batch/hidden constants."""
    assert _can_use_vectorized_token_projection(d_seed, knots, modes) is expected


@pytest.mark.parametrize(
    ("d_seed", "knots", "modes", "expected"),
    [
        (128, 16, 4, True),
        (32, 16, 4, True),
        (128, 16, 8, False),
        (128, 32, 4, False),
        (128, 16, 33, False),
    ],
)
def test_vectorized_mode_evaluation_uses_geometry_budget(
    d_seed: int,
    knots: int,
    modes: int,
    expected: bool,
) -> None:
    """The mode fusion guard remains shape-driven for unusual geometries."""
    assert _can_use_vectorized_mode_evaluation(d_seed, knots, modes) is expected


@pytest.mark.parametrize(
    ("d_seed", "knots", "modes", "top_k", "expected"),
    [
        (128, 16, 4, 2, True),
        (128, 16, 4, 1, False),
        (128, 16, 4, 3, False),
        (128, 32, 4, 2, False),
        (128, 16, 8, 2, False),
    ],
)
def test_route_vectorized_modes_share_only_supported_top_k_geometry(
    d_seed: int,
    knots: int,
    modes: int,
    top_k: int,
    expected: bool,
) -> None:
    assert (
        _can_use_route_vectorized_mode_evaluation(d_seed, knots, modes, top_k)
        is expected
    )


@pytest.mark.parametrize(
    ("hidden", "d_seed", "modes", "top_k", "expected_block", "single"),
    [
        (256, 128, 4, 2, 256, True),
        (257, 4, 2, 2, 512, True),
        (512, 128, 4, 2, 512, True),
        (512, 128, 8, 4, 512, True),
        # The padded row is still 512, but the seed/mode work exceeds the
        # single-program budget and must retain the multi-tile reduction.
        (512, 256, 8, 4, 256, False),
        (1024, 128, 4, 2, 256, False),
        (0, 128, 4, 2, 0, False),
    ],
)
def test_backward_hidden_tile_plan_is_geometry_driven(
    hidden: int,
    d_seed: int,
    modes: int,
    top_k: int,
    expected_block: int,
    single: bool,
) -> None:
    plan = _backward_hidden_tile_plan(hidden, d_seed, modes, top_k)
    assert plan.block_h == expected_block
    assert plan.direct_store is single
    assert plan.num_tiles == (0 if hidden == 0 else (hidden + expected_block - 1) // expected_block)
    assert plan.work_items >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_complete_row_wide_backward_matches_tiled_partition() -> None:
    """The 512-lane complete-row path must preserve the tiled backward."""
    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=7,
        hidden=512,
        d_seed=128,
        knots=16,
        modes=4,
        experts=1,
    )
    valid = torch.ones(7, device="cuda", dtype=torch.bool)
    valid[[2, 6]] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)
    original_limit = jtok_impl._BACKWARD_SINGLE_TILE_MAX_H

    def run(limit: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        jtok_impl._BACKWARD_SINGLE_TILE_MAX_H = limit
        trainable = {
            key: value.detach().clone().requires_grad_(True)
            for key, value in values.items()
            if key != "grid"
        }
        output = jtok_apply(
            trainable["delta"],
            trainable["z"],
            trainable["coeff"],
            trainable["spline_out"],
            trainable["residual_out"],
            trainable["scaler"],
            values["grid"],
            valid_mask=valid,
            backend="triton",
        )
        (output.float() * probe).sum().backward()
        torch.cuda.synchronize()
        return output.detach(), {
            key: value.grad.detach().clone()
            for key, value in trainable.items()
            if value.grad is not None
        }

    try:
        jtok_impl._BACKWARD_SINGLE_TILE_MAX_H = 256
        tiled_plan = _backward_hidden_tile_plan(512, 128, 4, 1)
        assert tiled_plan.block_h == 256
        assert not tiled_plan.direct_store
        tiled, tiled_grads = run(256)

        jtok_impl._BACKWARD_SINGLE_TILE_MAX_H = 512
        complete_plan = _backward_hidden_tile_plan(512, 128, 4, 1)
        assert complete_plan.block_h == 512
        assert complete_plan.num_tiles == 1
        assert complete_plan.direct_store
        complete, complete_grads = run(512)
    finally:
        jtok_impl._BACKWARD_SINGLE_TILE_MAX_H = original_limit

    torch.testing.assert_close(complete, tiled, rtol=0, atol=0)
    for key in tiled_grads:
        torch.testing.assert_close(
            complete_grads[key].float(),
            tiled_grads[key].float(),
            rtol=0.03,
            atol=0.02,
            msg=f"complete-row/tiled gradient mismatch for {key}",
        )


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


def test_jtokm_router_validation_does_not_reflatten_seed(monkeypatch) -> None:
    """Router validation must not prepare Leviathan's seed a second time."""
    values = _inputs()
    original = jtok_impl._flatten_inputs
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(jtok_impl, "_flatten_inputs", counted)
    output, _ = jtokm_apply(
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
        backend="torch",
    )

    assert calls == 1
    assert output.shape == values["delta"].shape


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
def test_jtokm_route_vectorized_full_geometry_matches_reference() -> None:
    """Exercise the new shared-basis Top-K=2 route kernel directly.

    The ordinary odd-geometry test intentionally stays on the conservative
    path.  This case matches the full-model seed/mode geometry so the new
    route-vector dispatch is actually executed in both forward and backward.
    """
    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=13,
        hidden=512,
        d_seed=128,
        knots=16,
        modes=4,
        experts=8,
    )
    valid = torch.ones(13, device="cuda", dtype=torch.bool)
    valid[[2, 9]] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)

    def run(backend: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        return output.detach(), {
            key: value.grad.detach().clone() for key, value in trainable.items()
        }

    reference, reference_grads = run("torch")
    actual, actual_grads = run("triton")
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0.12, atol=0.12)
    for key in reference_grads:
        assert torch.isfinite(actual_grads[key]).all(), key
        torch.testing.assert_close(
            actual_grads[key].float(),
            reference_grads[key].float(),
            rtol=0.25,
            atol=0.2,
            msg=f"full-geometry route gradient mismatch for {key}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtokm_triton_general_path_handles_hidden_above_single_tile() -> None:
    """Check the global normalization reduction on a genuinely wide row.

    The derivative of ``surface / ||surface||`` contains one dot product over
    the complete hidden axis.  A per-tile dot product is numerically wrong,
    even though the forward result can still look correct.  ``hidden=513``
    forces the compact multi-tile path and makes this regression observable.
    """
    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=5,
        hidden=513,
        d_seed=4,
        knots=5,
        modes=2,
    )
    valid = torch.ones(5, device="cuda", dtype=torch.bool)
    valid[::3] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)

    def run(backend: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        return output.detach(), {
            key: value.grad.detach().clone() for key, value in trainable.items()
        }

    reference, reference_grads = run("torch")
    actual, actual_grads = run("triton")
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=8e-2, atol=8e-2)
    for key in reference_grads:
        assert torch.isfinite(actual_grads[key]).all(), key
        torch.testing.assert_close(
            actual_grads[key].float(),
            reference_grads[key].float(),
            rtol=0.25,
            atol=0.2,
            msg=f"wide normalization gradient mismatch for {key}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtokm_multi_tile_normalization_uses_global_dot() -> None:
    """Reject a per-tile dot product in the normalization derivative.

    The second hidden tile has a nonzero surface but zero upstream
    ``delta``.  Its gradient therefore depends entirely on the global
    normalization dot from the first tile; a tile-local reduction silently
    returns zero there.
    """
    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=3,
        hidden=513,
        d_seed=4,
        knots=5,
        modes=2,
        experts=4,
    )
    values["delta"] = torch.zeros_like(values["delta"])
    values["delta"][:, :256] = 1
    values["z"].fill_(0.5)
    values["coeff"].fill_(1)
    values["spline_out"].fill_(1)
    values["residual_out"].zero_()
    values["scaler"].fill_(1)
    values["router_state"].zero_()
    values["router_weight"].zero_()
    valid = torch.ones(3, device="cuda", dtype=torch.bool)
    probe = torch.zeros_like(values["delta"], dtype=torch.float32)
    probe[:, :256] = 1

    def run(backend: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        return output.detach(), {
            key: value.grad.detach().clone() for key, value in trainable.items()
        }

    reference, reference_grads = run("torch")
    actual, actual_grads = run("triton")
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=8e-2, atol=8e-2)
    reference_tail = reference_grads["spline_out"][..., 256:]
    assert reference_tail.abs().max().item() > 1e-3
    torch.testing.assert_close(
        actual_grads["spline_out"].float(),
        reference_grads["spline_out"].float(),
        rtol=1e-2,
        atol=1e-3,
        msg="multi-tile normalization used a tile-local dot product",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtok_multi_tile_normalization_uses_global_dot() -> None:
    """Exercise the same global reduction through the plain JTok boundary."""
    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=3,
        hidden=513,
        d_seed=4,
        knots=5,
        modes=2,
        experts=1,
    )
    values["delta"] = torch.zeros_like(values["delta"])
    values["delta"][:, :256] = 1
    values["z"].fill_(0.5)
    values["coeff"].fill_(1)
    values["spline_out"].fill_(1)
    values["residual_out"].zero_()
    values["scaler"].fill_(1)
    valid = torch.ones(3, device="cuda", dtype=torch.bool)
    probe = torch.zeros_like(values["delta"], dtype=torch.float32)
    probe[:, :256] = 1

    def run(backend: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trainable = {
            key: values[key].detach().clone().requires_grad_(True)
            for key in (
                "delta",
                "z",
                "coeff",
                "spline_out",
                "residual_out",
                "scaler",
            )
        }
        output = jtok_apply(
            trainable["delta"],
            trainable["z"],
            trainable["coeff"],
            trainable["spline_out"],
            trainable["residual_out"],
            trainable["scaler"],
            values["grid"],
            valid_mask=valid,
            backend=backend,  # type: ignore[arg-type]
        )
        (output.float() * probe).sum().backward()
        return output.detach(), {
            key: value.grad.detach().clone() for key, value in trainable.items()
        }

    reference, reference_grads = run("torch")
    actual, actual_grads = run("triton")
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=8e-2, atol=8e-2)
    reference_tail = reference_grads["spline_out"][..., 256:]
    assert reference_tail.abs().max().item() > 1e-3
    torch.testing.assert_close(
        actual_grads["spline_out"].float(),
        reference_grads["spline_out"].float(),
        rtol=1e-2,
        atol=1e-3,
        msg="plain JTok multi-tile normalization used a tile-local dot product",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtokm_triton_boundary_hidden_256_uses_compact_path() -> None:
    """Keep the measured single-/multi-tile dispatch boundary covered."""
    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=7,
        hidden=256,
        d_seed=4,
        knots=5,
        modes=2,
    )
    valid = torch.ones(7, device="cuda", dtype=torch.bool)
    valid[[1, 5]] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)

    def run(backend: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        gradients = {
            key: trainable[key].grad.detach().clone() for key in trainable
        }
        return output.detach(), gradients

    reference, reference_grads = run("torch")
    actual, actual_grads = run("triton")
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=8e-2, atol=8e-2)
    for key in reference_grads:
        assert torch.isfinite(actual_grads[key]).all(), key
        torch.testing.assert_close(
            actual_grads[key].float(),
            reference_grads[key].float(),
            rtol=0.18,
            atol=0.12,
            msg=f"boundary gradient mismatch for {key}",
        )


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
        gradients = {}
        for key, value in trainable.items():
            assert value.grad is not None, f"missing gradient for {key}"
            gradients[key] = value.grad.detach().clone()
        return gradients

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
def test_jtokm_triton_wide_backward_avoids_reference_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the tiled external backward with a hidden row above 256."""
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=7, hidden=300, d_seed=5)
    valid = torch.ones(7, device="cuda", dtype=torch.bool)
    valid[3] = False
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
    import cut_cross_entropy.leviathan.jtok as jtok_module

    monkeypatch.setattr(
        jtok_module,
        "_selected_surface_reference",
        lambda *args, **kwargs: pytest.fail(
            "the external JTok-M backward entered the Torch reference path"
        ),
    )
    actual = run("triton")
    for key in reference:
        assert torch.isfinite(actual[key]).all(), key
        torch.testing.assert_close(
            actual[key].float(),
            reference[key].float(),
            rtol=0.22,
            atol=0.18,
            msg=f"wide gradient mismatch for {key}",
        )


@pytest.mark.manual
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.skipif(
    not hasattr(torch, "compiler")
    or not hasattr(torch.compiler, "cudagraph_mark_warmup_incomplete"),
    reason="Test requires the Torch 2.14 CUDA-Graph warmup hook",
)
def test_jtokm_aux_graph_capture_warmup_is_package_only() -> None:
    """Keep the real JTok-M/Inductor capture regression without NeoLLM imports.

    The failure reproduced on Torch 2.14 when the route-shared mode kernel,
    ``compute_aux=True``, AOTAutograd backward, and CUDA Graph Trees met in
    the same max-autotune step.  This probe intentionally mirrors only the
    package contract: twelve JTok-M calls, the full d_seed/knots/modes/top-k
    geometry, an MEAP-like validity mask, CCE, and an optimizer update.
    It is manual because max-autotune compilation is expensive; it must remain
    independent of modeling_neollm.py and train.py.
    """
    from cut_cross_entropy.leviathan import jtokm_auxiliary_loss

    batch, sequence, hidden = 64, 512, 512
    d_seed, modes, knots, experts, vocab = 128, 4, 16, 5, 4096
    layers = 12
    device = torch.device("cuda")

    class _JTokMGraphProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.coeff = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(
                            experts,
                            modes,
                            d_seed,
                            knots,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        * 0.15
                    )
                    for _ in range(layers)
                ]
            )
            self.spline_out = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(
                            experts, modes, hidden,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        * 0.2
                    )
                    for _ in range(layers)
                ]
            )
            self.residual_out = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(
                            experts, d_seed, hidden,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        * 0.2
                    )
                    for _ in range(layers)
                ]
            )
            self.scaler = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(hidden, device=device, dtype=torch.bfloat16)
                        * 0.1
                    )
                    for _ in range(layers)
                ]
            )
            self.router_weight = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(
                            experts, hidden,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        * 0.2
                    )
                    for _ in range(layers)
                ]
            )
            self.lm_head = nn.Parameter(
                torch.randn(vocab, hidden, device=device, dtype=torch.bfloat16)
                * 0.02
            )
            self.register_buffer(
                "grid",
                torch.linspace(0.0, 1.0, knots, device=device, dtype=torch.float32),
            )

        def forward(
            self,
            delta: torch.Tensor,
            z: torch.Tensor,
            router_state: torch.Tensor,
            valid_mask: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            aux_loss = delta.new_zeros(())
            for coeff, out_weight, residual, scaler, router_weight in zip(
                self.coeff,
                self.spline_out,
                self.residual_out,
                self.scaler,
                self.router_weight,
            ):
                delta, stats = jtokm_apply(
                    delta,
                    z,
                    router_state,
                    coeff,
                    out_weight,
                    residual,
                    scaler,
                    router_weight,
                    self.grid,
                    top_k=2,
                    valid_mask=valid_mask,
                    compute_aux=True,
                    backend="triton",
                )
                aux_loss = aux_loss + jtokm_auxiliary_loss(
                    stats,
                    num_experts=experts,
                    top_k=2,
                    weight=1e-2,
                )
            loss = linear_cross_entropy(
                delta,
                self.lm_head,
                targets,
                impl="cce",
                shift=1,
            )
            return loss + aux_loss

    torch.manual_seed(1729)
    model = _JTokMGraphProbe()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    compiled = torch.compile(model, mode="max-autotune")
    delta = torch.randn(
        batch, sequence, hidden, device=device, dtype=torch.bfloat16
    )
    z = torch.sigmoid(
        torch.randn(batch, sequence, d_seed, device=device, dtype=torch.bfloat16)
    )
    router_state = torch.randn(
        batch, sequence, hidden, device=device, dtype=torch.bfloat16
    )
    valid_mask = torch.ones(batch, sequence, device=device, dtype=torch.bool)
    valid_mask.reshape(-1)[::7] = False
    targets = torch.randint(8, vocab, (batch, sequence), device=device)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = compiled(delta, z, router_state, valid_mask, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtok_triton_wide_d_seed_128_matches_reference() -> None:
    """Cover the full-model d_seed=128 vectorized token-gradient geometry."""
    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=7,
        hidden=512,
        d_seed=128,
        knots=16,
        modes=4,
        experts=1,
    )
    valid = torch.ones(7, device="cuda", dtype=torch.bool)
    valid[3] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)

    def run(backend: str) -> dict[str, torch.Tensor]:
        jtok_keys = (
            "delta",
            "z",
            "coeff",
            "spline_out",
            "residual_out",
            "scaler",
        )
        trainable = {
            key: values[key].detach().clone().requires_grad_(True)
            for key in jtok_keys
        }
        output = jtok_apply(
            trainable["delta"],
            trainable["z"],
            trainable["coeff"],
            trainable["spline_out"],
            trainable["residual_out"],
            trainable["scaler"],
            values["grid"],
            valid_mask=valid,
            backend=backend,  # type: ignore[arg-type]
        )
        (output.float() * probe).sum().backward()
        gradients = {}
        for key, value in trainable.items():
            assert value.grad is not None, f"missing gradient for {key}"
            gradients[key] = value.grad.detach().clone()
        return gradients

    reference = run("torch")
    actual = run("triton")
    for key in reference:
        assert torch.isfinite(actual[key]).all(), key
        torch.testing.assert_close(
            actual[key].float(),
            reference[key].float(),
            rtol=0.22,
            atol=0.18,
            msg=f"d_seed=128 gradient mismatch for {key}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_jtok_wide_forward_exposes_compact_mode_cache() -> None:
    """Keep the forward/backward compact-cache contract model-free."""
    from cut_cross_entropy.leviathan.jtok import _jtok_forward_op

    values = _inputs(
        device="cuda",
        dtype=torch.bfloat16,
        n=7,
        hidden=512,
        d_seed=128,
        knots=16,
        modes=4,
        experts=1,
    )
    expert_idx = torch.zeros(7, 1, device="cuda", dtype=torch.long)
    selected_weights = torch.ones(7, 1, device="cuda", dtype=torch.float32)
    valid_mask = torch.ones(7, device="cuda", dtype=torch.bool)
    valid_mask[3] = False

    output, modes = _jtok_forward_op(
        values["delta"],
        values["z"],
        values["coeff"][:1],
        values["spline_out"][:1],
        values["residual_out"][:1],
        values["scaler"],
        values["grid"],
        expert_idx,
        selected_weights,
        valid_mask,
        1e-6,
    )
    torch.cuda.synchronize()
    assert output.shape == values["delta"].shape
    assert modes.shape == (7, 1, 4)
    assert modes.dtype == values["spline_out"].dtype
    assert modes.device == values["delta"].device
    assert torch.isfinite(modes).all()


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
