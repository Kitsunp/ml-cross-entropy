"""New zero-support VJP regressions; synthetic cases are not a throughput gate."""
from __future__ import annotations

import pytest
import torch

from cut_cross_entropy.leviathan import jtok_apply, jtokm_apply
from cut_cross_entropy.leviathan import jtok as implementation
from test_leviathan_jtok import _inputs


def test_sparse_updates_remain_explicit_and_disabled_by_default(monkeypatch):
    monkeypatch.delenv("JTOK_SPARSE_COEFF_UPDATES", raising=False)
    assert not implementation._sparse_coefficient_updates_requested()
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "1")
    assert implementation._sparse_coefficient_updates_requested()
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    assert not implementation._sparse_coefficient_updates_requested()


def _assert_finite_relative(actual, expected, tolerance):
    assert torch.isfinite(actual).all()
    error = torch.linalg.vector_norm((actual.float() - expected.float()).double())
    norm = torch.linalg.vector_norm(expected.float().double()).clamp_min(1e-12)
    assert (error / norm).item() <= tolerance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mixture", [False, True], ids=["plain", "mixture"])
@pytest.mark.parametrize("grid_storage", [torch.float32, torch.bfloat16],
                         ids=["fp32", "rounded_bf16"])
def test_sparse_vjp_matches_dense_and_normalized_oracle_at_edges(mixture, grid_storage):
    import triton
    n, d, knots, modes = 11, 128, 16, 4
    experts, top_k = (5, 2) if mixture else (1, 1)
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=n, hidden=512,
                     d_seed=d, knots=knots, modes=modes, experts=experts)
    grid = values["grid"].to(grid_storage).float()
    # Actual rounded knots, exact endpoints, half points and their neighbours.
    edges = torch.cat((grid, ((torch.arange(15, device="cuda") + 0.5) / 15),
                       torch.tensor([0., 1.], device="cuda")))
    edges = torch.cat((edges, torch.nextafter(edges, torch.zeros_like(edges)),
                      torch.nextafter(edges, torch.ones_like(edges))))
    z = edges[torch.arange(n * d, device="cuda") % edges.numel()].reshape(n, d)
    coeff = (values["coeff"].float() * .1 + 1).to(torch.bfloat16)
    coeff[:, :, 0].neg_()  # Signed products, without collapsing every mode.
    coeff[0, 1].zero_()  # Exact-zero Phi and the existing EPS/zero policy.
    routes = (torch.arange(n * top_k, device="cuda") % experts).reshape(n, top_k)
    if mixture:
        routes[4] = 0  # Repeated selected expert: FP32 accumulation still required.
    valid = torch.ones(n, device="cuda", dtype=torch.bool)
    valid[[2, 9]] = False
    basis = implementation._basis(z, grid)
    phi = torch.einsum("ndg,nrmdg->nrmd", basis, coeff[routes].float())
    products = ((1 - 2 * (phi < 0).sum(-1).remainder(2)).float()
                * torch.exp(torch.log(phi.abs() + 1e-9).sum(-1)))
    grad_modes = torch.randn_like(products)
    residual = torch.randn(n, top_k, d, device="cuda")
    outputs = []
    for sparse in (False, True):
        grad_z = torch.zeros(n, d, device="cuda")
        grad_coeff = torch.zeros_like(coeff, dtype=torch.float32)
        implementation._jtok_backward_token_projection_grad_block_kernel[(n, top_k)](
            z, coeff, grid, routes, products, grad_modes, residual, valid,
            grad_z, grad_coeff, n, D_SEED=d, NUM_KNOTS=knots, NUM_MODES=modes,
            TOP_K=top_k, KNOT_PAD=triton.next_power_of_2(knots), BLOCK_D=d,
            HAS_MASK=True, SPARSE_COEFF_UPDATES=sparse)
        outputs.append((grad_z, grad_coeff))
    z_ref = z.detach().clone().requires_grad_()
    coeff_ref = coeff.float().detach().requires_grad_()
    phi_ref = torch.einsum("ndg,nrmdg->nrmd", implementation._basis(z_ref, grid),
                           coeff_ref[routes])
    upstream = (grad_modes[..., None] * products[..., None]
                * torch.where(phi_ref < 0, -1., 1.) / (phi_ref.abs() + 1e-9))
    upstream = torch.where(valid[:, None, None, None], upstream, 0).detach()
    z_expected, coeff_expected = torch.autograd.grad((phi_ref * upstream).sum(),
                                                     (z_ref, coeff_ref))
    z_expected += torch.where(valid[:, None, None], residual, 0).sum(1)
    for grad_z, grad_coeff in outputs:
        _assert_finite_relative(grad_z, z_expected, 3e-5)
        _assert_finite_relative(grad_coeff, coeff_expected, 3e-5)
    _assert_finite_relative(outputs[1][0], outputs[0][0], 3e-5)
    _assert_finite_relative(outputs[1][1], outputs[0][1], 3e-5)
    assert torch.count_nonzero(outputs[1][0][~valid]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mixture", [False, True], ids=["plain", "mixture"])
def test_sparse_registered_backward_preserves_routing_and_all_vjps(monkeypatch, mixture):
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=13, hidden=512,
                     d_seed=128, knots=16, modes=4, experts=5 if mixture else 1)
    values["coeff"] = (values["coeff"].float() * .1 + 1).to(torch.bfloat16)
    valid = torch.ones(13, device="cuda", dtype=torch.bool)
    valid[[1, 10]] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)
    results = []
    for sparse in (False, True):
        monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "1" if sparse else "0")
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
def test_sparse_request_rejects_unsupported_vjp_instead_of_silent_noop(monkeypatch):
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "1")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=2, hidden=13,
                     d_seed=5, knots=7, modes=3, experts=1)
    with pytest.raises(ValueError, match="requires the wide vectorized"):
        implementation._run_jtok_backward_triton(
            torch.ones_like(values["delta"]), values["delta"], values["z"], values["coeff"],
            values["spline_out"], values["residual_out"], values["scaler"],
            torch.zeros(2, 1, device="cuda", dtype=torch.long),
            torch.ones(2, 1, device="cuda"), values["grid"],
            torch.ones(2, device="cuda", dtype=torch.bool),
            torch.ones(2, 1, 3, device="cuda", dtype=torch.bfloat16),
            norm_eps=1e-6, residual_scale=.1, mixture=False)
