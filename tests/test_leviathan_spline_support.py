from __future__ import annotations

import pytest
import torch

from cut_cross_entropy.leviathan import LeviathanConfig, LeviathanGenerator
from cut_cross_entropy.leviathan.backward_impl import _bspline
from cut_cross_entropy.leviathan.spline_support import (
    compact_quadratic_support,
    is_uniform_unit_grid,
)


def _scatter_compact(
    indices: torch.Tensor,
    values: torch.Tensor,
    num_knots: int,
) -> torch.Tensor:
    dense = torch.zeros(
        *values.shape[:-1],
        num_knots,
        dtype=values.dtype,
        device=values.device,
    )
    return dense.scatter_add(-1, indices, values)


def test_compact_support_matches_dense_reference_including_boundaries() -> None:
    cfg = LeviathanConfig(generator_num_knots=16)
    generator = LeviathanGenerator(cfg)
    scale = cfg.generator_num_knots - 1
    x = torch.tensor(
        [
            0.0,
            0.5 / scale,
            1.5 / scale,
            7.5 / scale,
            14.5 / scale,
            1.0,
        ],
        dtype=torch.float32,
    ).reshape(2, 3)

    indices, compact, compact_dt = compact_quadratic_support(
        x, cfg.generator_num_knots, generator.knot_grid
    )
    dense = generator._bspline_basis(x)
    dense, _, total, raw_dt = _bspline(x, cfg, generator.knot_grid)
    dense_dt = (raw_dt - dense * raw_dt.sum(dim=-1, keepdim=True)) / total

    torch.testing.assert_close(
        _scatter_compact(indices, compact, cfg.generator_num_knots),
        dense,
        rtol=0.0,
        atol=2e-7,
    )
    torch.testing.assert_close(
        _scatter_compact(indices, compact_dt, cfg.generator_num_knots),
        dense_dt,
        rtol=0.0,
        atol=4e-6,
    )
    assert torch.all(indices >= 0)
    assert torch.all(indices < cfg.generator_num_knots)
    assert torch.allclose(compact.sum(dim=-1), torch.ones_like(x))


def test_compact_support_matches_dense_reference_for_random_unit_values() -> None:
    cfg = LeviathanConfig(generator_num_knots=16)
    generator = LeviathanGenerator(cfg)
    x = torch.rand(11, 17, dtype=torch.float32)

    indices, compact, compact_dt = compact_quadratic_support(
        x, cfg.generator_num_knots, generator.knot_grid
    )
    dense, _, total, raw_dt = _bspline(x, cfg, generator.knot_grid)
    dense_dt = (raw_dt - dense * raw_dt.sum(dim=-1, keepdim=True)) / total

    torch.testing.assert_close(
        _scatter_compact(indices, compact, cfg.generator_num_knots),
        dense,
        rtol=0.0,
        atol=2e-6,
    )
    torch.testing.assert_close(
        _scatter_compact(indices, compact_dt, cfg.generator_num_knots),
        dense_dt,
        rtol=0.0,
        atol=3e-5,
    )


def test_compact_support_rejects_nonuniform_grid() -> None:
    grid = torch.linspace(0.0, 1.0, 16)
    custom = grid.clone()
    custom[5] += 0.001

    assert is_uniform_unit_grid(grid, 16)
    assert not is_uniform_unit_grid(custom, 16)
    with pytest.raises(ValueError, match="uniform unit knot grid"):
        compact_quadratic_support(torch.rand(4), 16, custom)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_compact_quantized_grid_matches_dense_stored_values(dtype) -> None:
    # This is the production representation: FP32 buffer -> model dtype ->
    # FP32 kernel work tensor. It is not a newly generated FP32 linspace.
    stored = torch.linspace(0.0, 1.0, 16).to(dtype)
    grid = stored.float()
    assert is_uniform_unit_grid(stored, 16)
    assert is_uniform_unit_grid(grid, 16)
    boundaries = (grid[:, None] + torch.tensor([-1.5, -0.5, 0.5, 1.5]) / 15.0).reshape(-1)
    boundaries = boundaries[(boundaries >= 0.0) & (boundaries <= 1.0)]
    x = torch.cat((torch.linspace(0, 1, 257), boundaries,
                   torch.nextafter(boundaries, torch.zeros_like(boundaries)),
                   torch.nextafter(boundaries, torch.ones_like(boundaries))))
    indices, weights, derivative = compact_quadratic_support(x, 16, stored)
    cfg = LeviathanConfig(generator_num_knots=16)
    dense, _, total, raw_dt = _bspline(x.reshape(1, -1), cfg, grid)
    dense = dense.reshape(-1, 16)
    raw_dt = raw_dt.reshape(-1, 16)
    dense_dt = (raw_dt - dense * raw_dt.sum(-1, keepdim=True)) / total.reshape(-1, 1)
    torch.testing.assert_close(_scatter_compact(indices, weights, 16), dense,
                               rtol=0, atol=3e-7)
    torch.testing.assert_close(_scatter_compact(indices, derivative, 16), dense_dt,
                               rtol=0, atol=6e-6)
    altered = stored.clone()
    altered[5] = altered[5] + 0.01
    assert not is_uniform_unit_grid(altered, 16)


def test_compact_grid_check_reuses_detached_storage_and_rechecks_mutation(monkeypatch) -> None:
    grid = torch.linspace(0.0, 1.0, 16)
    calls = []
    original = torch.equal

    def equal(actual, expected):
        calls.append(True)
        return original(actual, expected)

    monkeypatch.setattr(torch, "equal", equal)
    assert is_uniform_unit_grid(grid, 16)
    initial_calls = len(calls)
    assert initial_calls > 0
    assert is_uniform_unit_grid(grid.detach(), 16)
    assert len(calls) == initial_calls
    grid[5] += 0.01
    assert not is_uniform_unit_grid(grid.detach(), 16)
    assert len(calls) > initial_calls
    mutation_calls = len(calls)
    assert not is_uniform_unit_grid(grid.detach(), 16)
    assert len(calls) == mutation_calls


@pytest.mark.parametrize("dtype,num_knots", [(torch.bfloat16, 258), (torch.float16, 2050)])
def test_compact_quantized_grid_rejects_unproven_support_width(dtype, num_knots) -> None:
    grid = torch.linspace(0.0, 1.0, num_knots).to(dtype)
    assert not is_uniform_unit_grid(grid, num_knots)
    assert not is_uniform_unit_grid(grid.float(), num_knots)


def test_compact_derivative_matches_autograd_at_edges_and_half_cells() -> None:
    grid = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)
    half = (torch.arange(15, dtype=torch.float64) + 0.5) / 15.0
    x = torch.cat((torch.tensor([0.0, 1.0], dtype=torch.float64), half,
                   torch.nextafter(half, torch.zeros_like(half)),
                   torch.nextafter(half, torch.ones_like(half)))).requires_grad_()
    indices, weights, derivative = compact_quadratic_support(x, 16, grid)
    coefficients = torch.sin(torch.arange(16, dtype=torch.float64))
    cfg = LeviathanConfig(generator_num_knots=16)
    dense = _bspline(x.reshape(1, -1), cfg, grid)[0].reshape(x.numel(), 16)
    expected = (dense * coefficients).sum(-1)
    actual = (weights * coefficients[indices]).sum(-1)
    expected_derivative = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        (derivative * coefficients[indices]).sum(-1), expected_derivative,
        rtol=1e-10, atol=1e-10,
    )
