"""Opt-in local-support spline VJP shared by JTok and JTok-M.

The routing/mode VJP predecessor retains its ownership. This consumer only
contracts normalized local basis derivatives and scatters coefficient VJPs.
"""
from __future__ import annotations

import torch
from .spline_support import is_uniform_unit_grid


def prepare_compact_jtok_grids(model: torch.nn.Module) -> int:
    """Validate frozen JTok grids after the final device/dtype move, before capture.

    Structural module matching reuses the existing NeoLLM adapter contract;
    it does not add weights, buffers, hooks, or training work.
    """
    count = 0
    for module in model.modules():
        coefficients = getattr(module, "spline_coeff", None)
        grid = getattr(module, "knot_grid", None)
        if not isinstance(coefficients, torch.Tensor) or coefficients.ndim != 4:
            continue
        if not isinstance(grid, torch.Tensor):
            raise ValueError("compact JTok module must expose its frozen knot grid")
        if not is_uniform_unit_grid(grid, coefficients.shape[-1]):
            raise ValueError("compact spline VJP requires a canonical stored unit knot grid")
        count += 1
    return count


@torch.library.custom_op("cut_cross_entropy::jtok_compact_grid", mutates_args=())
def validated_compact_grid(grid: torch.Tensor, knots: int) -> torch.Tensor:
    """Keep value validation opaque, not the numerical Triton consumer.

    A new tiny grid tensor is required by the no-alias custom-op contract.
    The consumer actually uses it, so validation cannot be dead-code removed.
    Its device copy and lifetime are included in real profiling/timing.
    """
    if not is_uniform_unit_grid(grid, knots):
        raise ValueError("compact spline VJP requires a canonical stored unit knot grid")
    return grid.clone()


@validated_compact_grid.register_fake
def _validated_compact_grid_fake(grid: torch.Tensor, knots: int) -> torch.Tensor:
    return torch.empty_like(grid)


try:
    import triton
    import triton.language as tl
    from .spline_support_kernels import _compact_quadratic_geometry
except (ImportError, ModuleNotFoundError):
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _jtok_compact_spline_vjp_kernel(
        z_ptr, spline_coeff_ptr, grid_ptr, expert_idx_ptr, modes_ptr,
        grad_mode_ptr, grad_residual_ptr, valid_ptr, grad_z_ptr, grad_coeff_ptr,
        N, D_SEED: tl.constexpr, NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr, TOP_K: tl.constexpr,
        BLOCK_D: tl.constexpr, HAS_MASK: tl.constexpr,
    ):
        row = tl.program_id(0)
        slot = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        q = tl.arange(0, 4)
        row_mask = row < N
        coordinate_mask = row_mask & (d < D_SEED)
        if HAS_MASK:
            row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
        else:
            row_valid = row_mask
        expert = tl.load(expert_idx_ptr + row * TOP_K + slot,
                         mask=row_mask, other=0).to(tl.int32)
        x = tl.load(z_ptr + row * D_SEED + d,
                    mask=coordinate_mask, other=0.0).to(tl.float32)
        left, b0, b1, b2, b3, db0, db1, db2, db3, raw_sum = (
            _compact_quadratic_geometry(x, grid_ptr, NUM_KNOTS, True, True)
        )
        # Keep neighbours contiguous within each coordinate's four-lane tile.
        knots = left[:, None] + q[None, :]
        basis = tl.where(q[None, :] == 0, b0[:, None],
                         tl.where(q[None, :] == 1, b1[:, None],
                                  tl.where(q[None, :] == 2, b2[:, None], b3[:, None])))
        derivative = tl.where(q[None, :] == 0, db0[:, None],
                              tl.where(q[None, :] == 1, db1[:, None],
                                       tl.where(q[None, :] == 2, db2[:, None], db3[:, None])))
        # Match the established zero-derivative policy below the basis floor.
        derivative = tl.where((raw_sum > 1e-12)[:, None], derivative, 0.0)
        grad_z = tl.load(
            grad_residual_ptr + (row * TOP_K + slot) * D_SEED + d,
            mask=coordinate_mask, other=0.0,
        ).to(tl.float32)
        support_mask = coordinate_mask[:, None] & (knots < NUM_KNOTS)
        for mode in tl.range(0, NUM_MODES):
            coefficients = tl.load(
                spline_coeff_ptr + ((expert * NUM_MODES + mode) * D_SEED + d[:, None])
                * NUM_KNOTS + knots,
                mask=support_mask, other=0.0,
            ).to(tl.float32)
            phi = tl.sum(basis * coefficients, axis=1)
            middle = tl.sum(tl.where(q[None, :] == 1, coefficients, 0.0), axis=1)
            # sum(db)=0: cancel a coordinate-independent coefficient offset.
            dphi = tl.sum(derivative * (coefficients - middle[:, None]), axis=1)
            mode_value = tl.load(modes_ptr + (row * TOP_K + slot) * NUM_MODES + mode,
                                 mask=row_mask, other=0.0).to(tl.float32)
            grad_mode = tl.load(grad_mode_ptr + (row * TOP_K + slot) * NUM_MODES + mode,
                                mask=row_mask, other=0.0).to(tl.float32)
            grad_phi = grad_mode * mode_value * tl.where(phi < 0.0, -1.0, 1.0) / (tl.abs(phi) + 1e-9)
            grad_phi = tl.where(row_valid, grad_phi, 0.0)
            grad_z += grad_phi * dphi
            tl.atomic_add(
                grad_coeff_ptr + ((expert * NUM_MODES + mode) * D_SEED + d[:, None])
                * NUM_KNOTS + tl.minimum(knots, NUM_KNOTS - 1),
                grad_phi[:, None] * basis,
                mask=support_mask & row_valid & (basis != 0.0),
            )
        # Different selected routes still share z, so these atomics remain.
        tl.atomic_add(grad_z_ptr + row * D_SEED + d, grad_z,
                      mask=coordinate_mask & row_valid)
