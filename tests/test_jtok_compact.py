"""New compact spline VJP cases; synthetic functionality, never a speed gate."""
from __future__ import annotations

import pytest
import torch

from cut_cross_entropy.leviathan import jtok_apply, jtokm_apply
from cut_cross_entropy.leviathan import jtok as implementation
from cut_cross_entropy.leviathan import jtok_compact as kernels
from test_leviathan_jtok import _inputs
from test_jtok_sparse_updates import _assert_finite_relative


def test_compact_jtok_vjp_is_explicit_and_rejects_unsupported_contracts(monkeypatch):
    monkeypatch.delenv("JTOK_COMPACT_SPLINE_VJP", raising=False)
    assert not implementation._compact_spline_vjp_requested()
    monkeypatch.setenv("JTOK_COMPACT_SPLINE_VJP", "1")
    assert implementation._compact_spline_vjp_requested()
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    grid = torch.linspace(0, 1, 16)
    implementation._validate_compact_spline_vjp(512, 128, 16, 4, grid)
    for hidden, d, knots, modes in ((13, 128, 16, 4), (512, 5, 2, 3), (512, 256, 16, 4)):
        with pytest.raises(ValueError, match="wide vectorized spline VJP"):
            implementation._validate_compact_spline_vjp(hidden, d, knots, modes, grid)
    changed = grid.clone()
    changed[4] += .001
    with pytest.raises(ValueError, match="canonical stored unit knot grid"):
        implementation._validate_compact_spline_vjp(512, 128, 16, 4, changed)
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "1")
    with pytest.raises(ValueError, match="disable sparse"):
        implementation._validate_compact_spline_vjp(512, 128, 16, 4, grid)


def test_compact_jtok_prepare_grids_rejects_mutation_without_altering_model():
    model = torch.nn.Module()
    model.register_parameter("spline_coeff", torch.nn.Parameter(torch.ones(1, 3, 7, 7)))
    model.register_buffer("knot_grid", torch.linspace(0, 1, 7))
    original = model.knot_grid.clone()
    pointer = model.knot_grid.data_ptr()
    assert kernels.prepare_compact_jtok_grids(model) == 1
    assert torch.equal(model.knot_grid, original) and model.knot_grid.data_ptr() == pointer
    assert set(dict(model.named_buffers())) == {"knot_grid"}
    model.knot_grid[2] += .001
    with pytest.raises(ValueError, match="canonical stored unit knot grid"):
        kernels.prepare_compact_jtok_grids(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("experts,top_k,d,knots,modes,storage", [
    (1, 1, 128, 16, 4, torch.float32),
    (5, 2, 128, 16, 4, torch.bfloat16),
    (4, 4, 7, 7, 3, torch.float16),
    (5, 2, 7, 256, 1, torch.bfloat16),
], ids=["plain", "rounded_mixture", "tails_repeated_empty", "rounded_limit"])
def test_compact_jtok_vjp_matches_dense_oracle_and_basis_floor(experts, top_k, d, knots, modes, storage):
    import triton
    n = 11
    generator = torch.Generator(device="cuda").manual_seed(8819)
    grid = torch.linspace(0, 1, knots, device="cuda").to(storage).float()
    special = torch.tensor([-.5, 0., 1., 1.5, -1.5 / (knots - 1),
                            1 + 1.5 / (knots - 1)], device="cuda")
    special = torch.cat((special, torch.nextafter(special, torch.zeros_like(special))))
    if d == 128:
        # The old dense quotient is ill-conditioned near the outer support
        # boundary. Compare both implementations and Double strictly on the
        # production unit domain; the separate floor regression below has an
        # explicit FP32 oracle outside it, not the unstable dense derivative.
        special = torch.tensor([0., 1., .5, .5 / (knots - 1),
                                1 - .5 / (knots - 1)], device="cuda")
        special = torch.cat((special, torch.nextafter(special, torch.zeros_like(special))))
    edges = torch.cat((grid, (grid[:-1] + grid[1:]) / 2, special))
    edges = torch.cat((edges, torch.nextafter(edges, torch.zeros_like(edges)),
                      torch.nextafter(edges, torch.ones_like(edges))))
    indices = torch.arange(n * d, device="cuda") * edges.numel() // (n * d)
    z = edges[indices].reshape(n, d)
    z.flatten()[:special.numel()] = special
    coeff = (1 + .01 * torch.randn(experts, modes, d, knots, device="cuda", generator=generator))
    coeff = coeff.to(torch.bfloat16)
    coeff[:, :, 0].neg_()
    if modes > 1:
        coeff[:, 1].zero_()  # Exact-zero Phi and the unchanged product EPS policy.
    if modes > 2:
        coeff[:, 2].fill_(1)  # Constant coefficients: anchored derivative cancels.
    routes = (torch.arange(n * top_k, device="cuda") % experts).reshape(n, top_k)
    if top_k == 4:
        routes.zero_()  # Duplicate routes and empty experts.
    valid = torch.ones(n, device="cuda", dtype=torch.bool)
    valid[[2, 9]] = False
    # Nonzero synthetic predecessor values exercise every contraction, even
    # when an out-of-domain coordinate would annihilate a physical product.
    # The registered tests below separately use the actual forward product.
    products = .9 + .1 * torch.rand(n, top_k, modes, device="cuda", generator=generator)
    if modes > 1:
        products[:, :, 1] = 1e-9
    grad_modes = torch.randn(n, top_k, modes, device="cuda", generator=generator)
    residual = torch.randn(n, top_k, d, device="cuda", generator=generator)
    actual = (torch.zeros(n, d, device="cuda"), torch.zeros_like(coeff, dtype=torch.float32))
    dense = tuple(torch.zeros_like(tensor) for tensor in actual)
    common = (z, coeff, grid, routes, products, grad_modes, residual, valid)
    kernels._jtok_compact_spline_vjp_kernel[(n, top_k)](
        *common, *actual, n, D_SEED=d, NUM_KNOTS=knots, NUM_MODES=modes,
        TOP_K=top_k, BLOCK_D=triton.next_power_of_2(d), HAS_MASK=True,
        num_warps=4, num_stages=1)
    implementation._jtok_backward_token_projection_grad_block_kernel[(n, top_k)](
        *common, *dense, n, D_SEED=d, NUM_KNOTS=knots, NUM_MODES=modes,
        TOP_K=top_k, KNOT_PAD=triton.next_power_of_2(knots),
        BLOCK_D=triton.next_power_of_2(d), HAS_MASK=True, SPARSE_COEFF_UPDATES=False)
    # Independent normalized Double oracle, with the existing floor policy.
    z_ref = z.double().detach().requires_grad_()
    coeff_ref = coeff.double().detach().requires_grad_()
    distance = (z_ref[:, :, None] - grid.double()).abs() * (knots - 1)
    raw = torch.where(distance < .5, .75 - distance.square(),
                      torch.where(distance < 1.5, .5 * (1.5 - distance).square(), 0.))
    raw_sum = raw.sum(-1)
    basis = raw / raw_sum.clamp_min(1e-12)[:, :, None]
    phi_ref = torch.einsum("ndg,nrmdg->nrmd", basis, coeff_ref[routes])
    upstream = (grad_modes.double()[..., None] * products.double()[..., None]
                * torch.where(phi_ref < 0, -1., 1.) / (phi_ref.abs() + 1e-9))
    upstream = torch.where(valid[:, None, None, None], upstream, 0).detach()
    expected_z, expected_coeff = torch.autograd.grad((phi_ref * upstream).sum(), (z_ref, coeff_ref))
    expected_z = torch.where(raw_sum > 1e-12, expected_z, 0)
    expected_z += torch.where(valid[:, None, None], residual, 0).double().sum(1)
    for got, previous, expected in zip(actual, dense, (expected_z, expected_coeff)):
        _assert_finite_relative(got, previous, 6e-5)
        _assert_finite_relative(got, expected.float(), 6e-5)
    assert torch.count_nonzero(actual[0][~valid]) == 0
    if top_k == 4:
        assert torch.count_nonzero(actual[1][1:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mixture", [False, True], ids=["plain", "mixture"])
def test_compact_jtok_registered_backward_preserves_all_vjps(monkeypatch, mixture):
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    monkeypatch.setenv("JTOK_PROJECTION_SPLIT", "1")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=13, hidden=512,
                     d_seed=128, knots=16, modes=4, experts=5 if mixture else 1)
    values["grid"] = values["grid"].to(torch.bfloat16).float()
    values["coeff"] = (values["coeff"].float() * .1 + 1).to(torch.bfloat16)
    valid = torch.ones(13, device="cuda", dtype=torch.bool)
    valid[[1, 10]] = False
    probe = torch.randn_like(values["delta"], dtype=torch.float32)
    results = []
    for enabled in (False, True):
        monkeypatch.setenv("JTOK_COMPACT_SPLINE_VJP", "1" if enabled else "0")
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
        results.append((output.detach(), {key: value.grad for key, value in trainable.items()
                                          if value.grad is not None}))
    assert torch.equal(results[0][0], results[1][0])
    assert set(results[0][1]) == set(results[1][1])
    if mixture:
        assert {"router_state", "router_weight"} <= set(results[1][1])
    for key in results[0][1]:
        _assert_finite_relative(results[1][1][key], results[0][1][key], 1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_jtok_empty_backward_retains_zero_parameter_gradients(monkeypatch):
    monkeypatch.setenv("JTOK_COMPACT_SPLINE_VJP", "1")
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    monkeypatch.setenv("JTOK_PROJECTION_SPLIT", "1")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=0, hidden=512,
                     d_seed=7, knots=7, modes=3, experts=1)
    gradients = implementation._run_jtok_backward_triton(
        torch.ones_like(values["delta"]), values["delta"], values["z"], values["coeff"],
        values["spline_out"], values["residual_out"], values["scaler"],
        torch.zeros(0, 1, device="cuda", dtype=torch.long), torch.ones(0, 1, device="cuda"),
        values["grid"], torch.ones(0, device="cuda", dtype=torch.bool),
        torch.ones(0, 1, 3, device="cuda", dtype=torch.bfloat16),
        norm_eps=1e-6, residual_scale=.1, mixture=False)
    for value in gradients:
        assert torch.count_nonzero(value) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_jtok_outside_support_preserves_floor_and_constant_derivative():
    n, d, knots = 6, 7, 16
    grid = torch.linspace(0, 1, knots, device="cuda")
    edge = torch.tensor([-.5, -.1, -.09, 1.09, 1.1, 1.5], device="cuda")
    edge = torch.cat((edge, torch.nextafter(edge, torch.zeros_like(edge))))
    z = edge[torch.arange(n * d, device="cuda") % edge.numel()].reshape(n, d)
    coefficients = torch.ones(1, 1, d, knots, device="cuda", dtype=torch.bfloat16)
    modes = torch.ones(n, 1, 1, device="cuda")
    residual = torch.arange(n * d, device="cuda").reshape(n, 1, d).float() / 64
    grad_z, grad_coeff = torch.zeros_like(z), torch.zeros_like(coefficients, dtype=torch.float32)
    kernels._jtok_compact_spline_vjp_kernel[(n, 1)](
        z, coefficients, grid, torch.zeros(n, 1, device="cuda", dtype=torch.long),
        modes, modes, residual, torch.ones(n, device="cuda", dtype=torch.bool),
        grad_z, grad_coeff, n, D_SEED=d, NUM_KNOTS=knots, NUM_MODES=1, TOP_K=1,
        BLOCK_D=8, HAS_MASK=True, num_warps=4, num_stages=1)
    # Target stored FP32 basis, including branches whose FP64 support differs
    # by one ULP; constant coefficients have zero normalized derivative above
    # the floor, and the existing policy forces zero below the floor too.
    basis = implementation._basis(z, grid)
    phi = basis.sum(-1)
    expected_coeff = (basis / (phi.abs() + 1e-9)[:, :, None]).sum(0)[None, None]
    assert torch.equal(grad_z, residual[:, 0])
    _assert_finite_relative(grad_coeff, expected_coeff, 6e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("canonical", [True, False], ids=["valid_grid", "reject_changed_grid"])
def test_compact_jtok_fullgraph_compiled_mixture_backward(monkeypatch, canonical):
    monkeypatch.setenv("JTOK_COMPACT_SPLINE_VJP", "1")
    monkeypatch.setenv("JTOK_SPARSE_COEFF_UPDATES", "0")
    monkeypatch.setenv("JTOK_PROJECTION_SPLIT", "1")
    values = _inputs(device="cuda", dtype=torch.bfloat16, n=3, hidden=512,
                     d_seed=7, knots=7, modes=3, experts=3)
    values["coeff"] = (1 + .01 * values["coeff"]).requires_grad_()
    delta = values["delta"].requires_grad_()
    values["z"].requires_grad_()
    implementation._validate_compact_spline_vjp(512, 7, 7, 3, values["grid"])
    if not canonical:
        values["grid"][2] += .001  # A stale host cache must not bypass validation.

    def run(x):
        output, _ = jtokm_apply(x, values["z"], values["router_state"], values["coeff"],
                                values["spline_out"], values["residual_out"], values["scaler"],
                                values["router_weight"], values["grid"], top_k=2, backend="triton")
        return output.float().square().mean()

    compiled = torch.compile(run, fullgraph=True, mode="max-autotune")
    if not canonical:
        with pytest.raises((ValueError, RuntimeError), match=(
            "canonical stored unit knot grid|validate the compact spline grid before CUDA graph capture"
        )):
            compiled(delta).backward()
        return
    loss = compiled(delta)
    loss.backward()
    for tensor in (loss, delta.grad, values["z"].grad, values["coeff"].grad):
        assert tensor is not None and torch.isfinite(tensor).all()
