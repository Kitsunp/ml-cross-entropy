"""New split-N regressions; synthetic inputs only establish functionality."""
from __future__ import annotations

import pytest
import torch

from cut_cross_entropy.leviathan import jtok_apply, jtokm_apply
from cut_cross_entropy.leviathan import jtok as implementation
from cut_cross_entropy.leviathan import jtok_projection as kernels
from test_leviathan_jtok import _inputs
from test_jtok_sparse_updates import _assert_finite_relative


def test_projection_split_is_explicit_and_disabled_by_default(monkeypatch):
    monkeypatch.delenv("JTOK_PROJECTION_SPLIT", raising=False)
    assert not implementation._projection_split_requested()
    monkeypatch.setenv("JTOK_PROJECTION_SPLIT", "1")
    assert implementation._projection_split_requested()
    monkeypatch.setenv("JTOK_PROJECTION_SPLIT", "0")
    assert not implementation._projection_split_requested()


def test_projection_split_plan_bounds_parameter_scratch():
    plan = kernels.projection_grad_plan(32768, 5, 4, 128, 512)
    assert plan == kernels.ProjectionGradPlan(8, 16, 64, 32, 10813440)
    assert kernels.projection_grad_plan(1, 1, 3, 7, 257).splits == 1
    assert kernels.projection_grad_plan(0, 1, 3, 7, 257).splits == 1
    with pytest.raises(ValueError, match="dimensions must be positive"):
        kernels.projection_grad_plan(-1, 1, 3, 7, 257)
    with pytest.raises(ValueError, match="128 MiB"):
        kernels.projection_grad_plan(32768, 512, 4, 128, 512)


def _launch_projection(g, z, modes, routes, weights, experts):
    import triton
    n, hidden = g.shape
    d, num_modes, top_k = z.shape[1], modes.shape[-1], routes.shape[-1]
    plan = kernels.projection_grad_plan(n, experts, num_modes, d, hidden)
    partial = torch.empty(experts, plan.splits, num_modes + d, hidden, device="cuda")
    grad_spline = torch.empty(experts, num_modes, hidden, device="cuda")
    grad_residual = torch.empty(experts, d, hidden, device="cuda")
    tiles = triton.cdiv(num_modes + d, plan.block_f) * triton.cdiv(hidden, plan.block_h)
    kernels._jtok_projection_split_kernel[(experts, plan.splits, tiles)](
        g, z, modes, routes, weights, partial, n,
        D_SEED=d, NUM_MODES=num_modes, HIDDEN=hidden, TOP_K=top_k,
        SPLITS=plan.splits, BLOCK_F=plan.block_f, BLOCK_H=plan.block_h,
        BLOCK_N=plan.block_n, num_warps=4, num_stages=1)
    kernels._jtok_projection_split_reduce_kernel[
        (experts, triton.cdiv((num_modes + d) * hidden, 1024))
    ](partial, grad_spline, grad_residual, D_SEED=d, NUM_MODES=num_modes,
      HIDDEN=hidden, SPLITS=plan.splits, BLOCK=1024, num_warps=4, num_stages=1)
    return grad_spline, grad_residual


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("experts,top_k,n,d,m,hidden", [
    (1, 1, 33, 7, 3, 257),
    (5, 2, 67, 128, 4, 513),
    (4, 4, 65, 5, 7, 259),
], ids=["plain_tail", "mixture_tail", "repeated_empty_experts"])
def test_projection_split_matches_double_oracle_and_fixed_order(experts, top_k, n, d, m, hidden):
    generator = torch.Generator(device="cuda").manual_seed(7301)
    g = torch.randn(n, hidden, device="cuda", generator=generator)
    z = torch.randn(n, d, device="cuda", dtype=torch.bfloat16, generator=generator)
    modes = torch.randn(n, top_k, m, device="cuda", dtype=torch.bfloat16, generator=generator)
    weights = torch.softmax(torch.randn(n, top_k, device="cuda", generator=generator), -1)
    routes = (torch.arange(n * top_k, device="cuda") % experts).reshape(n, top_k)
    routes[1] = 0  # Duplicate expert slots must add, not overwrite.
    if top_k == 4:
        routes.zero_()  # Other experts are completely empty.
    g[[2, n - 1]] = 0  # The existing predecessor owns the invalid-row mask.
    actual = _launch_projection(g, z, modes, routes, weights, experts)
    repeated = _launch_projection(g, z, modes, routes, weights, experts)
    expected_spline, expected_residual = [], []
    for expert in range(experts):
        selected = torch.where(routes == expert, weights.double(), 0.0)
        alpha = selected.sum(1)
        beta = (selected[:, :, None] * modes.double()).sum(1)
        expected_spline.append(beta.T @ g.double())
        expected_residual.append((alpha[:, None] * z.double()).T @ g.double())
    for got, same, expected in zip(actual, repeated,
                                   (torch.stack(expected_spline), torch.stack(expected_residual))):
        _assert_finite_relative(got, expected.float(), 3e-5)
        assert torch.equal(got, same)  # Reduction ownership is deterministic here.
        if top_k == 4:
            assert torch.count_nonzero(got[1:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mixture", [False, True], ids=["plain", "mixture"])
def test_projection_split_registered_backward_preserves_all_vjps(monkeypatch, mixture):
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=13, hidden=512,
                     d_seed=128, knots=16, modes=4, experts=5 if mixture else 1)
    values["coeff"] = (values["coeff"].float() * .1 + 1).to(torch.bfloat16)
    valid = torch.ones(13, device="cuda", dtype=torch.bool)
    valid[[1, 10]] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)
    results = []
    for enabled in (False, True):
        monkeypatch.setenv("JTOK_PROJECTION_SPLIT", "1" if enabled else "0")
        trainable = {key: value.detach().clone().requires_grad_()
                     for key, value in values.items() if key != "grid"}
        common = (trainable["coeff"], trainable["spline_out"], trainable["residual_out"],
                  trainable["scaler"])
        if mixture:
            output, _ = jtokm_apply(trainable["delta"], trainable["z"],
                                    trainable["router_state"], *common,
                                    trainable["router_weight"], values["grid"], top_k=2,
                                    valid_mask=valid, backend="triton")
        else:
            output = jtok_apply(trainable["delta"], trainable["z"], *common,
                                values["grid"], valid_mask=valid, backend="triton")
        (output.float() * probe).sum().backward()
        results.append((output.detach(), {key: value.grad for key, value in trainable.items()
                                          if value.grad is not None}))
    assert torch.equal(results[0][0], results[1][0])
    assert set(results[0][1]) == set(results[1][1])
    if mixture:
        assert {"router_state", "router_weight"} <= set(results[1][1])
    for key in results[0][1]:
        _assert_finite_relative(results[1][1][key], results[0][1][key], 1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens,hidden", [(2, 13), (0, 512)], ids=["unsupported", "empty"])
def test_projection_split_rejects_unsupported_and_preserves_empty_gradients(monkeypatch, tokens, hidden):
    monkeypatch.setenv("JTOK_PROJECTION_SPLIT", "1")
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=tokens, hidden=hidden,
                     d_seed=5, knots=7, modes=3, experts=1)
    args = (torch.ones_like(values["delta"]), values["delta"], values["z"], values["coeff"],
            values["spline_out"], values["residual_out"], values["scaler"],
            torch.zeros(tokens, 1, device="cuda", dtype=torch.long),
            torch.ones(tokens, 1, device="cuda"), values["grid"],
            torch.ones(tokens, device="cuda", dtype=torch.bool),
            torch.ones(tokens, 1, 3, device="cuda", dtype=torch.bfloat16))
    if tokens:
        with pytest.raises(ValueError, match="requires the wide projection"):
            implementation._run_jtok_backward_triton(*args, norm_eps=1e-6,
                                                     residual_scale=.1, mixture=False)
    else:
        gradients = implementation._run_jtok_backward_triton(*args, norm_eps=1e-6,
                                                             residual_scale=.1, mixture=False)
        for value in gradients:
            assert torch.count_nonzero(value) == 0
