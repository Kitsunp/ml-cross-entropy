"""Compact spline regressions, independent of private training code/data."""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from cut_cross_entropy.leviathan import compiler
from cut_cross_entropy.leviathan import LeviathanConfig, LeviathanGenerator
from cut_cross_entropy.leviathan.spline_support_kernels import (
    _compact_quadratic_geometry,
    _lev_compact_contract,
)


@triton.jit
def _probe_contract(t_ptr, grid_ptr, coeff_ptr, out_ptr, dt_ptr, N,
                    BLOCK_M: tl.constexpr, KRANK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < N
    t = tl.load(t_ptr + rows, mask=mask, other=0.0)
    left, b0, b1, b2, b3, db0, _, db2, db3 = _compact_quadratic_geometry(
        t, grid_ptr, 16, True,
    )
    rcols = tl.arange(0, KRANK)
    phi, phi_dt = _lev_compact_contract(
        coeff_ptr, rcols, mask, left, b0, b1, b2, b3, db0, db2, db3, KRANK, True,
    )
    offset = rows[:, None] * KRANK + rcols[None, :]
    tl.store(out_ptr + offset, phi, mask=mask[:, None])
    tl.store(dt_ptr + offset, phi_dt, mask=mask[:, None])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", ["random", "constant", "signed"])
@pytest.mark.parametrize("grid_dtype", [torch.float32, torch.bfloat16, torch.float16],
                         ids=["canonical", "bf16_quantized", "fp16_quantized"])
def test_compact_contract_matches_dense_autograd_with_boundary_tails(case: str, grid_dtype) -> None:
    grid = torch.linspace(0.0, 1.0, 16, device="cuda").to(grid_dtype).float()
    half = (torch.arange(15, device="cuda", dtype=torch.float32) + 0.5) / 15.0
    boundaries = (grid[:, None] + torch.tensor([-1.5, -0.5, 0.5, 1.5], device="cuda") / 15).reshape(-1)
    boundaries = boundaries[(boundaries >= 0) & (boundaries <= 1)]
    half = torch.cat((half, boundaries))
    t = torch.cat((torch.tensor([0.0, 1.0], device="cuda"), half,
                   torch.nextafter(half, torch.zeros_like(half)),
                   torch.nextafter(half, torch.ones_like(half)))).requires_grad_()
    rng = torch.Generator(device="cuda").manual_seed(1729)
    coeff = torch.randn(16, 64, device="cuda", generator=rng).to(torch.bfloat16)
    if case == "constant":
        coeff[:] = 0.125
    elif case == "signed":
        coeff[:] = torch.linspace(-2.0, 0.0, 16, device="cuda")[:, None]
    actual = torch.empty(t.numel(), 64, device="cuda")
    actual_dt = torch.empty_like(actual)
    _probe_contract[(triton.cdiv(t.numel(), 32),)](
        t, grid, coeff, actual, actual_dt, t.numel(), 32, 64,
        num_warps=4,
    )
    distance = (t[:, None] - grid[None, :]).abs() * 15.0
    raw = torch.where(distance < 0.5, 0.75 - distance.square(),
                      torch.where(distance < 1.5, 0.5 * (1.5 - distance).square(), 0.0))
    basis = raw / raw.sum(-1, keepdim=True).clamp_min(1e-12)
    expected = basis @ (1.0 + coeff.float())
    # Check coordinate derivatives as a VJP rather than duplicating the
    # implementation's derivative formula in the expected value.
    upstream = torch.randn(expected.shape, device="cuda", generator=rng)
    expected_dt = torch.autograd.grad((expected * upstream).sum(), t)[0]
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(
        (actual_dt * upstream).sum(-1), expected_dt, rtol=5e-5, atol=5e-5,
    )
    if case == "constant":
        assert torch.count_nonzero(actual_dt) == 0
    assert torch.isfinite(actual).all() and torch.isfinite(actual_dt).all()


def test_compact_compiler_forward_rejection_cannot_use_reference(monkeypatch) -> None:
    monkeypatch.setenv("LEV_COMPACT_SPLINE", "1")

    def rejected(*args, **kwargs):
        raise ValueError("unsupported compact grid")

    def forbidden(*args, **kwargs):
        pytest.fail("compact forward entered the reference fallback")

    monkeypatch.setattr(compiler, "_leviathan_forward", rejected)
    monkeypatch.setattr(compiler, "leviathan_forward_ref", forbidden)
    with pytest.raises(ValueError, match="unsupported compact grid"):
        compiler._saved_or_reference(torch.tensor([0]), {}, object())


def test_compact_compiler_backward_rejection_cannot_use_reference(monkeypatch) -> None:
    monkeypatch.setattr(compiler, "_leviathan_backward_triton", lambda *a, **kw: None)

    def forbidden(*args, **kwargs):
        pytest.fail("compact backward entered the reference fallback")

    monkeypatch.setattr(compiler, "leviathan_backward", forbidden)
    with pytest.raises(RuntimeError, match="compact Leviathan backward requires Triton"):
        compiler._compute_leviathan_grads(
            torch.tensor([0.0]), {}, object(), {"compact_spline": True},
            torch.tensor([0]), True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("grid_dtype", [torch.float32, torch.bfloat16],
                         ids=["canonical", "bf16_quantized"])
def test_compact_compiled_seed_backward_keeps_forward_choice_and_fp32_scatter(monkeypatch, grid_dtype) -> None:
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("the compact candidate is restricted to SM120")
    torch.manual_seed(1729)
    monkeypatch.setenv("LEV_DOT", "1")
    monkeypatch.setenv("LEV_COMPACT_SPLINE", "1")
    monkeypatch.setenv("LEV_DDELTA_SPLITS", "4")
    cfg = LeviathanConfig(
        vocab_size=4096, hidden_size=128, generator_d_seed=128,
        generator_num_modes=2, generator_num_knots=16,
        generator_k=3, generator_krank=64, dtype=torch.bfloat16,
    )
    generator = LeviathanGenerator(cfg).cuda()
    generator.knot_grid = generator.knot_grid.to(grid_dtype)
    params = {name: value.detach().clone().requires_grad_()
              for name, value in generator.named_parameters()}
    ids = torch.arange(129, device="cuda") % cfg.vocab_size
    run = torch.compile(compiler.leviathan_embedding_with_seed_compiler_safe,
                        mode="max-autotune")
    _, seed = run(ids, params, cfg, generator.knot_grid)
    upstream = torch.linspace(-0.5, 0.5, seed.numel(), device="cuda").reshape_as(seed)
    visited = []
    original = compiler._compute_leviathan_grads

    def checked(grad_out, params, cfg, saved, ids, has_modes, seed_grad=None):
        assert saved["compact_spline"] is True
        visited.append(True)
        return original(grad_out, params, cfg, saved, ids, has_modes, seed_grad)

    monkeypatch.setattr(compiler, "_compute_leviathan_grads", checked)
    # The backward must use the forward's mode flag, even when the process
    # environment has changed since that forward ran.
    monkeypatch.setenv("LEV_COMPACT_SPLINE", "0")
    (seed.float() * upstream).sum().backward()
    oracle = params["codebooks"].detach().float().requires_grad_()
    expected_seed = compiler._leviathan_seed_from_codebooks(ids, oracle)
    expected = torch.autograd.grad(
        (expected_seed * upstream.to(seed.dtype).float()).sum(), oracle,
    )[0].to(params["codebooks"].dtype)
    torch.testing.assert_close(params["codebooks"].grad, expected, rtol=0.0, atol=0.0)
    assert visited == [True]
