"""New route-VJP factorization contracts; synthetic cases are not speed gates."""
from __future__ import annotations

import pytest
import torch

from cut_cross_entropy.leviathan import jtok_apply, jtokm_apply
from cut_cross_entropy.leviathan import jtok as implementation
from test_leviathan_jtok import _inputs
from test_jtok_sparse_updates import _assert_finite_relative


def test_route_vjp_factor_is_explicit_and_rejects_narrow_hidden(monkeypatch):
    monkeypatch.delenv("JTOK_ROUTE_VJP_FACTOR", raising=False)
    assert not implementation._route_vjp_factor_requested()
    monkeypatch.setenv("JTOK_ROUTE_VJP_FACTOR", "1")
    assert implementation._route_vjp_factor_requested()
    implementation._validate_route_vjp_factor(256)
    with pytest.raises(ValueError, match="requires the wide projection VJP"):
        implementation._validate_route_vjp_factor(255)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mixture,experts,top_k,hidden,block_h", [
    (False, 1, 1, 257, 256),
    (True, 4, 4, 259, 256),
    (True, 5, 2, 512, 512),
], ids=["plain_multitile_tail", "mixture_repeated_empty_tail", "mixture_single_tile"])
def test_route_vjp_factor_matches_double_with_zero_tiny_weights(
    mixture, experts, top_k, hidden, block_h,
):
    import triton
    n, d, m, eps, scale = 5, 7, 3, 1e-6, .1
    values = _inputs(device="cuda", dtype=torch.float32, n=n, hidden=hidden,
                     d_seed=d, modes=m, knots=7, experts=experts)
    generator = torch.Generator(device="cuda").manual_seed(9631)
    modes = torch.randn(n, top_k, m, device="cuda", generator=generator)
    routes = (torch.arange(n * top_k, device="cuda") % experts).reshape(n, top_k)
    if top_k == 4:
        routes.zero_()  # Repeated routes and experts with no selected tokens.
    weights = torch.rand(n, top_k, device="cuda", generator=generator)
    weights[0] = 0
    weights[1] = 2. ** -30
    valid = torch.ones(n, device="cuda", dtype=torch.bool)
    valid[-1] = False
    g = torch.randn(n, hidden, device="cuda", generator=generator)
    ws, wr = values["spline_out"][routes].double(), values["residual_out"][routes].double()
    surfaces = (torch.einsum("nrm,nrmh->nrh", modes.double(), ws)
                + torch.einsum("nd,nrdh->nrh", values["z"].double(), wr))
    mixed = (surfaces * weights.double()[..., None]).sum(1).float()
    mixed[~valid] = 0
    radius = mixed.double().norm(dim=1, keepdim=True)
    unit = mixed.double() / torch.where(radius > 0, radius, torch.ones_like(radius))
    denominator = radius + eps
    # Independent Double contractions at the supplied FP32 surface and the
    # exact additive-epsilon normalization derivative (independent of Triton).
    factor_ref = (scale * values["scaler"].double() * g.double() if mixture
                  else g.double() * values["delta"].double() * values["scaler"].double())
    radial = unit * (factor_ref * unit).sum(1, keepdim=True)
    gs = (factor_ref - radial) / denominator + radial * (eps / denominator.square())
    gs = torch.where(valid[:, None], gs, 0)
    a = torch.einsum("nh,nrmh->nrm", gs, ws)
    b = torch.einsum("nh,nrdh->nrd", gs, wr)
    expected = (weights.double()[..., None] * a,
                weights.double()[..., None] * b,
                (a * modes.double()).sum(2) + (b * values["z"].double()[:, None]).sum(2))
    # A one-route normalization is almost scale-invariant: bar_p can be
    # O(EPS) while the dot-product terms are much larger. A pure relative
    # error divided by bar_p is ill-conditioned in that regime. Keep the
    # same 3e-5 error budget, but normalize each cancelling scalar by the
    # expanded absolute contraction terms; zero/tiny-p rows stay relative.
    weight_condition_scale = (
        torch.einsum("nh,nrmh,nrm->nr", gs.abs(), ws.abs(), modes.double().abs())
        + torch.einsum("nh,nrdh,nd->nr", gs.abs(), wr.abs(), values["z"].double().abs()))

    def assert_weight_contraction(got, want):
        assert torch.isfinite(got).all()
        bound = 3e-5 * weight_condition_scale.clamp_min(1e-12)
        assert torch.all((got.double() - want.double()).abs() <= bound)
        for row in (0, 1):
            _assert_finite_relative(got[row], want[row].float(), 3e-5)

    answers = []
    norm_stats = torch.empty(2 * n, device="cuda")
    norm_dot = torch.empty(n, device="cuda")
    if hidden > block_h:
        implementation._jtok_backward_norm_dot_kernel[(n,)](
            mixed, values["delta"], values["scaler"], g, valid, norm_dot, norm_stats, n,
            HIDDEN=hidden, BLOCK_H=block_h, NUM_TILES=triton.cdiv(hidden, block_h),
            NORM_EPS=eps, RESIDUAL_SCALE=scale, MIXTURE=mixture, HAS_MASK=True,
            num_warps=4, num_stages=1)
    for enabled in (False, True):
        grad_mode, grad_residual = torch.zeros_like(modes), torch.zeros(n, top_k, d, device="cuda")
        grad_delta, grad_scaler = torch.empty_like(g), torch.zeros(hidden, device="cuda")
        grad_weights = torch.zeros_like(weights)
        surface_workspace = mixed.clone()
        # Fixed launch checks both ownership cases without autotune repetition.
        implementation._jtok_backward_multi_tile_fast_kernel.fn[
            (n, triton.cdiv(hidden, block_h))
        ](values["delta"], values["z"], values["spline_out"], values["residual_out"],
          routes, weights, modes, grad_mode, grad_residual, values["scaler"],
          surface_workspace, norm_stats, norm_dot, g, valid,
          grad_delta, grad_scaler, grad_weights, n, D_SEED=d, NUM_MODES=m,
          TOP_K=top_k, HIDDEN=hidden, BLOCK_H=block_h, NORM_EPS=eps,
          RESIDUAL_SCALE=scale, MIXTURE=mixture, HAS_MASK=True,
          SINGLE_HIDDEN_TILE=hidden <= block_h, ROUTE_VJP_FACTOR=enabled,
          num_warps=4, num_stages=1)
        actual = (grad_mode, grad_residual, grad_weights)
        for got, want in zip(actual[:2], expected[:2]):
            _assert_finite_relative(got, want.float(), 3e-5)
            for row in range(n - 1):
                _assert_finite_relative(got[row], want[row].float(), 3e-5)
            assert torch.count_nonzero(got[~valid]) == 0
        assert_weight_contraction(grad_weights, expected[2])
        assert torch.count_nonzero(grad_weights[~valid]) == 0
        assert torch.count_nonzero(grad_mode[0]) == 0
        assert torch.count_nonzero(grad_residual[0]) == 0
        assert torch.count_nonzero(grad_weights[0]) > 0  # bar_p is not p*bar_p.
        answers.append((actual, surface_workspace, grad_delta, grad_scaler))
    for previous, current in zip(answers[0][0][:2], answers[1][0][:2]):
        _assert_finite_relative(current, previous, 3e-5)
    assert_weight_contraction(answers[1][0][2], answers[0][0][2])
    for previous, current in zip(answers[0][1:], answers[1][1:]):
        _assert_finite_relative(current, previous, 3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mixture", [False, True], ids=["plain", "mixture"])
def test_route_vjp_factor_registered_backward_stacks_with_compact_and_split(monkeypatch, mixture):
    monkeypatch.setenv("JTOK_COMPACT_SPLINE_VJP", "1")
    monkeypatch.setenv("JTOK_PROJECTION_SPLIT", "1")
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=9, hidden=512,
                     d_seed=7, knots=7, modes=3, experts=3 if mixture else 1)
    values["coeff"] = (1 + .01 * values["coeff"]).to(torch.bfloat16)
    valid = torch.ones(9, device="cuda", dtype=torch.bool)
    valid[[1, 7]] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)
    answers = []
    for enabled in (False, True):
        monkeypatch.setenv("JTOK_ROUTE_VJP_FACTOR", "1" if enabled else "0")
        trainable = {key: value.detach().clone().requires_grad_()
                     for key, value in values.items() if key != "grid"}
        common = (trainable["coeff"], trainable["spline_out"], trainable["residual_out"],
                  trainable["scaler"])
        if mixture:
            output, _ = jtokm_apply(trainable["delta"], trainable["z"], trainable["router_state"],
                                    *common, trainable["router_weight"], values["grid"],
                                    top_k=2, valid_mask=valid, backend="triton")
        else:
            output = jtok_apply(trainable["delta"], trainable["z"], *common,
                                values["grid"], valid_mask=valid, backend="triton")
        (output.float() * probe).sum().backward()
        answers.append((output.detach(), {key: value.grad for key, value in trainable.items()
                                          if value.grad is not None}))
    assert torch.equal(answers[0][0], answers[1][0])
    assert set(answers[0][1]) == set(answers[1][1])
    if mixture:
        assert {"router_state", "router_weight"} <= set(answers[1][1])
    for key in answers[0][1]:
        _assert_finite_relative(answers[1][1][key], answers[0][1][key], 1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_route_vjp_factor_fullgraph_compiled_mixture_backward(monkeypatch):
    for name in ("JTOK_ROUTE_VJP_FACTOR", "JTOK_COMPACT_SPLINE_VJP", "JTOK_PROJECTION_SPLIT"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=3, hidden=512,
                     d_seed=7, knots=7, modes=3, experts=3)
    values["coeff"] = (1 + .01 * values["coeff"]).requires_grad_()
    delta, z = values["delta"].requires_grad_(), values["z"].requires_grad_()
    implementation._validate_compact_spline_vjp(512, 7, 7, 3, values["grid"])

    def run(x):
        output, _ = jtokm_apply(x, z, values["router_state"], values["coeff"],
                                values["spline_out"], values["residual_out"], values["scaler"],
                                values["router_weight"], values["grid"], top_k=2, backend="triton")
        return output.float().square().mean()

    loss = torch.compile(run, fullgraph=True, mode="max-autotune")(delta)
    loss.backward()
    for tensor in (loss, delta.grad, z.grad, values["coeff"].grad):
        assert tensor is not None and torch.isfinite(tensor).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_route_vjp_factor_empty_backward_retains_zero_parameter_gradients(monkeypatch):
    for name in ("JTOK_ROUTE_VJP_FACTOR", "JTOK_COMPACT_SPLINE_VJP", "JTOK_PROJECTION_SPLIT"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=0, hidden=512,
                     d_seed=7, knots=7, modes=3, experts=1)
    gradients = implementation._run_jtok_backward_triton(
        torch.ones_like(values["delta"]), values["delta"], values["z"], values["coeff"],
        values["spline_out"], values["residual_out"], values["scaler"],
        torch.zeros(0, 1, device="cuda", dtype=torch.long), torch.ones(0, 1, device="cuda"),
        values["grid"], torch.ones(0, device="cuda", dtype=torch.bool),
        torch.ones(0, 1, 3, device="cuda", dtype=torch.bfloat16),
        norm_eps=1e-6, residual_scale=.1, mixture=False)
    for tensor in gradients:
        assert torch.count_nonzero(tensor) == 0
