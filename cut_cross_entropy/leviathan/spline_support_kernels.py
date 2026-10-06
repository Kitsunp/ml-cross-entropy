"""Device primitives shared by compact quadratic-spline consumers.

Geometry is independent of coefficient layout and expert routing. Leviathan
uses it immediately in registers; a JTok consumer can reuse the same weights
and normalized derivatives with its own coefficient strides.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _quadratic_weight(t, knot, SCALE: tl.constexpr, DERIVATIVE: tl.constexpr):
    distance = tl.abs(t - knot) * SCALE
    tail = 1.5 - distance
    weight = tl.where(
        distance < 0.5,
        0.75 - distance * distance,
        tl.where(distance < 1.5, 0.5 * tail * tail, 0.0),
    )
    derivative = tl.full(t.shape, 0.0, tl.float32)
    if DERIVATIVE:
        derivative = tl.where(
            distance < 0.5,
            -2.0 * distance,
            tl.where(distance < 1.5, -(1.5 - distance), 0.0),
        ) * tl.where(t >= knot, 1.0, -1.0) * SCALE
    return weight, derivative


@triton.jit
def _compact_quadratic_geometry(
    t, grid_ptr, KAPPA: tl.constexpr, DERIVATIVE: tl.constexpr,
    RETURN_RAW_SUM: tl.constexpr = False,
):
    """Return local support, normalized weights and d(weight)/dt.

    Requires the validated uniform unit grid and coordinates in [0, 1]. Actual
    grid values are loaded to preserve the dense path's floating-point knots.
    Three knots suffice over exact real numbers. Rounded grids may have a
    fourth tiny boundary contribution, which must also enter normalization.
    The middle derivative is determined by partition of unity, so consumers
    can cancel coordinate-independent coefficient offsets before contraction.
    RETURN_RAW_SUM exposes the unclamped sum for consumers that preserve an
    explicit zero-derivative floor policy outside the unit interval. The
    default return values and Leviathan arithmetic are unchanged.
    """
    left = tl.floor(t * (KAPPA - 1.0) - 0.5).to(tl.int32)
    left = tl.minimum(tl.maximum(left, 0), KAPPA - 3)
    previous = tl.load(grid_ptr + left - 1, mask=left > 0, other=-1.0)
    previous_active = (left > 0) & (tl.abs(t - previous) * (KAPPA - 1.0) < 1.5)
    left = tl.where(previous_active, left - 1, left)
    g0 = tl.load(grid_ptr + left)
    g1 = tl.load(grid_ptr + left + 1)
    g2 = tl.load(grid_ptr + left + 2)
    g3_valid = left + 3 < KAPPA
    g3 = tl.load(grid_ptr + left + 3, mask=g3_valid, other=0.0)
    a0, da0 = _quadratic_weight(t, g0, KAPPA - 1.0, DERIVATIVE)
    a1, da1 = _quadratic_weight(t, g1, KAPPA - 1.0, DERIVATIVE)
    a2, da2 = _quadratic_weight(t, g2, KAPPA - 1.0, DERIVATIVE)
    a3, da3 = _quadratic_weight(t, g3, KAPPA - 1.0, DERIVATIVE)
    a3 = tl.where(g3_valid, a3, 0.0)
    da3 = tl.where(g3_valid, da3, 0.0)
    raw_total = (a0 + a1) + (a2 + a3)
    total = tl.maximum(raw_total, 1e-12)
    b0, b1, b2 = a0 / total, a1 / total, a2 / total
    b3 = a3 / total
    db0 = tl.full(t.shape, 0.0, tl.float32)
    db1 = tl.full(t.shape, 0.0, tl.float32)
    db2 = tl.full(t.shape, 0.0, tl.float32)
    db3 = tl.full(t.shape, 0.0, tl.float32)
    if DERIVATIVE:
        dtotal = (da0 + da1) + (da2 + da3)
        db0 = (da0 - b0 * dtotal) / total
        db2 = (da2 - b2 * dtotal) / total
        db3 = (da3 - b3 * dtotal) / total
        db1 = -((db0 + db2) + db3)
    if RETURN_RAW_SUM:
        return left, b0, b1, b2, b3, db0, db1, db2, db3, raw_total
    else:
        return left, b0, b1, b2, b3, db0, db1, db2, db3


@triton.jit
def _lev_compact_contract(
    coeff_ptr, rcols, row_mask, left, b0, b1, b2, b3, db0, db2, db3,
    KRANK: tl.constexpr, DERIVATIVE: tl.constexpr,
):
    """Interpolate 1+Delta using two-dimensional coefficient loads.

    The middle row is the derivative anchor. Only it and one neighbour need
    be live as coefficient tiles; no [tokens, support, rank] tile is built.
    Since sum(db)=0, phi' = sum(q!=1) dbq*(Sq-S1). The fourth coefficient
    load is skipped unless this tile contains a floating-point boundary halo.
    """
    middle = tl.load(
        coeff_ptr + (left[:, None] + 1) * KRANK + rcols[None, :],
        mask=row_mask[:, None], other=0.0,
    ).to(tl.float32)
    phi = b1[:, None] * (1.0 + middle)
    phi_dt = tl.full(phi.shape, 0.0, tl.float32)
    # A runtime loop with unroll factor one keeps the two neighbour loads in
    # separate iterations instead of expanding both coefficient tiles.
    for q in tl.range(0, 2, loop_unroll_factor=1):
        offset = 2 * q
        weight = tl.where(q == 0, b0, b2)
        derivative = tl.where(q == 0, db0, db2)
        neighbour = tl.load(
            coeff_ptr + (left[:, None] + offset) * KRANK + rcols[None, :],
            mask=row_mask[:, None], other=0.0,
        ).to(tl.float32)
        phi += weight[:, None] * (1.0 + neighbour)
        if DERIVATIVE:
            # Delta differences cancel the unit coefficient offset exactly.
            phi_dt += derivative[:, None] * (neighbour - middle)
    halo_mask = row_mask & ((b3 != 0.0) | (db3 != 0.0))
    if tl.sum(halo_mask.to(tl.int32), axis=0) > 0:
        neighbour = tl.load(
            coeff_ptr + (left[:, None] + 3) * KRANK + rcols[None, :],
            mask=halo_mask[:, None], other=0.0,
        ).to(tl.float32)
        phi += b3[:, None] * (1.0 + neighbour)
        if DERIVATIVE:
            phi_dt += db3[:, None] * (neighbour - middle)
    return phi, phi_dt
