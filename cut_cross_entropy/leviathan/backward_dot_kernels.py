"""Dot-based backward kernels (fast path, LEV_DOT=1).

The per-d tl.dot construction of phi / dB replaces the 16-iteration g-loops
(tensor cores).  Numerics: phi via dot with input_precision=ieee reproduces
the fp32 elementwise chain (L0-validated: identical cos with the dot
forward); dB is a gradient intermediate (fp32 accumulate).
Memory: unchanged (no phi save) — same lean buffers as the g-loop kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS_LOG = 1e-9
NORM_EPS = 1e-5


@triton.jit
def _lev_bwd_chain_dot_kernel(
    xhat_ptr, t_ptr, rsqrt_ptr, modes_ptr, dm_ptr,
    gamma_ptr, beta_ptr, delta_ptr, knot_grid_ptr,
    dxhat_ptr, dzh_ptr, dgamma_partial_ptr, dbeta_partial_ptr, N, NUM_BLOCKS,
    D_SEED: tl.constexpr, KAPPA: tl.constexpr, KAPPA_P: tl.constexpr,
    KRANK: tl.constexpr, H_: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr, LOG_EPS: tl.constexpr, DOT_IEEE: tl.constexpr,
    PREMUL_DMM: tl.constexpr, USE_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    m = pid_m
    rows = pid_b * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < N

    dcols = tl.arange(0, D_SEED)
    rcols = tl.arange(0, KRANK)
    gcols_p = tl.arange(0, KAPPA_P)
    gmask = gcols_p < KAPPA
    scale = KAPPA - 1.0
    hp = H_ * KRANK

    grid_v = tl.load(knot_grid_ptr + gcols_p, mask=gmask, other=0.0)  # [KAPPA_P]

    gamma = tl.load(gamma_ptr + m * D_SEED + dcols)
    beta = tl.load(beta_ptr + m * D_SEED + dcols)
    xhat = tl.load(xhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                   + dcols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + m * N + rows, mask=mask, other=0.0)
    dM = tl.load(dm_ptr + rows[:, None] * hp + m * KRANK + rcols[None, :],
                 mask=mask[:, None], other=0.0)
    M = tl.load(modes_ptr + rows[:, None] * hp + m * KRANK + rcols[None, :],
                mask=mask[:, None], other=0.0).to(tl.float32)

    # ---- per-d loop: phi (dot) -> dphi -> dB (dot) -> dt -> dy -> LN ----
    for dc in tl.range(0, D_SEED, loop_unroll_factor=1):
        xc = tl.load(xhat_ptr + m * (N * D_SEED) + rows * D_SEED + dc,
                     mask=mask, other=0.0).to(tl.float32)     # [BM]
        gc = tl.load(gamma_ptr + m * D_SEED + dc)
        bc = tl.load(beta_ptr + m * D_SEED + dc)
        if USE_T:
            tc = tl.load(t_ptr + m * (N * D_SEED) + rows * D_SEED + dc,
                         mask=mask, other=0.0).to(tl.float32)
        else:
            tc = 1.0 / (1.0 + tl.exp(-0.5 * (xc * gc + bc)))
        tc = tl.minimum(tl.maximum(tc, 0.0), 1.0)             # [BM]
        # basis (vectorized over g) + normalizer
        d3 = tl.abs(tc[:, None] - grid_v[None, :]) * scale    # [BM, KAPPA_P]
        w3 = tl.where(d3 < 0.5, 0.75 - d3 * d3,
                      tl.where(d3 < 1.5,
                               0.5 * (1.5 - d3) * (1.5 - d3), 0.0))
        w3 = tl.where(gmask[None, :], w3, 0.0)                # pad -> exact 0
        denom = tl.maximum(tl.sum(w3, 1, keep_dims=True), 1e-12)
        bgn = w3 / denom                                      # [BM, KAPPA_P]
        # phi via dot (ieee = exact fp32 chain); st padded with 1*0 = 0
        st = tl.load(delta_ptr + m * (D_SEED * KAPPA * KRANK)
                     + dc * (KAPPA * KRANK)
                     + (gcols_p % KAPPA)[:, None] * KRANK
                     + rcols[None, :],
                     mask=gmask[:, None], other=0.0)          # [KAPPA_P, KRANK] bf16
        st = 1.0 + st.to(tl.float32)
        st_t = tl.trans(st)                                   # [KRANK, KAPPA_P]
        if DOT_IEEE:
            phi = tl.dot(bgn, st, input_precision="ieee")     # [BM, KRANK]
        else:
            phi = tl.dot(bgn, st, input_precision="tf32x3")
        dphi = ((dM.to(tl.float32) if PREMUL_DMM else dM * M)
                * tl.where(phi >= 0, 1.0, -1.0)
                / (tl.abs(phi) + LOG_EPS))                    # [BM, KRANK]
        # dB via dot over r: [BM, KRANK] @ [KRANK, KAPPA]
        if DOT_IEEE:
            dB = tl.dot(dphi, st_t, input_precision="ieee")   # [BM, KAPPA_P]
        else:
            dB = tl.dot(dphi, st_t, input_precision="tf32x3")
        # spline backward, vectorized over g (exact split):
        #   dt = scale/D^2 * (D*A - wsum*C)
        dwd3 = tl.where(d3 < 0.5, -2.0 * d3,
                        tl.where(d3 < 1.5, -(1.5 - d3), 0.0))
        sgn3 = tl.where(tc[:, None] - grid_v[None, :] >= 0, 1.0, -1.0)
        wsum = tl.sum(dB * w3, 1)                             # [BM]
        A = tl.sum(dB * dwd3 * sgn3, 1)
        C = tl.sum(dwd3 * sgn3, 1)
        denom1 = tl.reshape(denom, [BLOCK_M])
        dt = (denom1 * A - wsum * C) * (scale / (denom1 * denom1))
        # sigmoid(x/2) + clamp backward; LN backward (per-d part)
        tmask = ((tc > 0.0) & (tc < 1.0)).to(tl.float32)
        dy = dt * (0.5 * tc * (1.0 - tc)) * tmask             # [BM]
        # Consume dy while it is in registers.  The following stats buffer is
        # indexed by (head, token block, d), so every element has one writer.
        partial_base = (m * NUM_BLOCKS + pid_b) * D_SEED
        tl.store(dgamma_partial_ptr + partial_base + dc,
                 tl.sum(dy * xc), mask=dc < D_SEED)
        tl.store(dbeta_partial_ptr + partial_base + dc,
                 tl.sum(dy), mask=dc < D_SEED)
        dxc = dy * gc
        tl.store(dxhat_ptr + m * (N * D_SEED) + rows * D_SEED + dc,
                 dxc, mask=mask)

    # dxhat full tile (barrier-ordered store/load roundtrip)
    tl.debug_barrier()
    dxhat = tl.load(dxhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                    + dcols[None, :], mask=mask[:, None], other=0.0)

    # LayerNorm backward (full-d): dzh = rsqrt * (dxhat - m1 - xhat*m2)
    m1 = tl.sum(dxhat, 1, keep_dims=True) * (1.0 / D_SEED)
    m2 = tl.sum(dxhat * xhat, 1, keep_dims=True) * (1.0 / D_SEED)
    dzh = rsqrt[:, None] * (dxhat - m1 - xhat * m2)
    tl.store(dzh_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
             + dcols[None, :], dzh, mask=mask[:, None])


@triton.jit
def _lev_bwd_ddelta_dot_kernel(
    xhat_ptr, t_ptr, rsqrt_ptr, modes_ptr, dm_ptr,
    gamma_ptr, beta_ptr, delta_ptr, knot_grid_ptr, ddelta_ptr,
    dzh_ptr, dgamma_partial_ptr, dbeta_partial_ptr, N, NUM_BLOCKS,
    D_SEED: tl.constexpr, KAPPA: tl.constexpr, KAPPA_P: tl.constexpr,
    KRANK: tl.constexpr, H_: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr,
    EPS: tl.constexpr, LOG_EPS: tl.constexpr, DOT_IEEE: tl.constexpr,
    FUSE_CHAIN: tl.constexpr, PREMUL_DMM: tl.constexpr,
    USE_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_r = tl.program_id(2)
    m = pid_m
    dc = pid_d * BLOCK_D
    rc = pid_r * BLOCK_R
    ccols = dc + tl.arange(0, BLOCK_D)
    rcols = rc + tl.arange(0, BLOCK_R)
    gcols = tl.arange(0, KAPPA)
    gcols_p = tl.arange(0, KAPPA_P)
    gmask = gcols_p < KAPPA
    scale = KAPPA - 1.0
    hp = H_ * KRANK

    grid_v = tl.load(knot_grid_ptr + gcols_p, mask=gmask, other=0.0)

    # Keep the padded dot dimension in the accumulator.  For KAPPA < 16,
    # tl.dot uses KAPPA_P=16; reducing into a KAPPA-wide buffer makes the
    # [KAPPA_P, BLOCK_R] result shape incompatible during compilation.
    acc = tl.zeros([BLOCK_D, KAPPA_P, BLOCK_R], tl.float32)
    for nb in tl.range(0, N, BLOCK_M, loop_unroll_factor=1):
        rows = nb + tl.arange(0, BLOCK_M)
        mask = rows < N
        # dM and M are invariant across the seed dimensions in this tile.
        # Hoist their loads so the compiler can overlap them with the basis
        # construction and avoid repeating the traffic when BLOCK_D > 1.
        dM = tl.load(dm_ptr + rows[:, None] * hp + m * KRANK
                     + rcols[None, :], mask=mask[:, None], other=0.0)
        M = tl.load(modes_ptr + rows[:, None] * hp + m * KRANK
                    + rcols[None, :], mask=mask[:, None],
                    other=0.0).to(tl.float32)
        # per-d of this tile: basis + phi + dphi, then acc += B^T @ dphi
        for di in tl.static_range(BLOCK_D):
            gcv = tl.load(gamma_ptr + m * D_SEED + dc + di)
            bcv = tl.load(beta_ptr + m * D_SEED + dc + di)
            xc = tl.load(xhat_ptr + m * (N * D_SEED) + rows * D_SEED
                         + (dc + di), mask=mask, other=0.0).to(tl.float32)
            if USE_T:
                tcv = tl.load(t_ptr + m * (N * D_SEED) + rows * D_SEED
                              + (dc + di), mask=mask, other=0.0).to(tl.float32)
            else:
                tcv = 1.0 / (1.0 + tl.exp(-0.5 * (xc * gcv + bcv)))
            tcv = tl.minimum(tl.maximum(tcv, 0.0), 1.0)
            d3 = tl.abs(tcv[:, None] - grid_v[None, :]) * scale
            w3 = tl.where(d3 < 0.5, 0.75 - d3 * d3,
                          tl.where(d3 < 1.5,
                                   0.5 * (1.5 - d3) * (1.5 - d3), 0.0))
            w3 = tl.where(gmask[None, :], w3, 0.0)
            denom = tl.maximum(tl.sum(w3, 1, keep_dims=True), 1e-12)
            bgn = w3 / denom                                  # [BM, KAPPA_P]
            st = tl.load(delta_ptr + m * (D_SEED * KAPPA * KRANK)
                         + (dc + di) * (KAPPA * KRANK)
                         + (gcols_p % KAPPA)[:, None] * KRANK
                         + rcols[None, :],
                         mask=gmask[:, None], other=0.0)
            st = 1.0 + st.to(tl.float32)                      # [KAPPA_P, BLOCK_R]
            if DOT_IEEE:
                phi = tl.dot(bgn, st, input_precision="ieee")  # [BM, BLOCK_R]
            else:
                phi = tl.dot(bgn, st, input_precision="tf32x3")
            dphi = ((dM.to(tl.float32) if PREMUL_DMM else dM * M)
                    * tl.where(phi >= 0, 1.0, -1.0)
                    / (tl.abs(phi) + LOG_EPS))
            if FUSE_CHAIN:
                # The old chain kernel performed this same dB -> dt -> dy
                # work after dphi had already been computed once here.  Keep
                # it in registers so this pass also produces the LN inputs.
                st_t = tl.trans(st)
                if DOT_IEEE:
                    dB = tl.dot(dphi, st_t, input_precision="ieee")
                else:
                    dB = tl.dot(dphi, st_t, input_precision="tf32x3")
                dwd3 = tl.where(d3 < 0.5, -2.0 * d3,
                                tl.where(d3 < 1.5, -(1.5 - d3), 0.0))
                sgn3 = tl.where(tcv[:, None] - grid_v[None, :] >= 0,
                                1.0, -1.0)
                wsum = tl.sum(dB * w3, 1)
                A = tl.sum(dB * dwd3 * sgn3, 1)
                C = tl.sum(dwd3 * sgn3, 1)
                denom1 = tl.reshape(denom, [BLOCK_M])
                dt = ((denom1 * A - wsum * C)
                      * (scale / (denom1 * denom1)))
                tmask = ((tcv > 0.0) & (tcv < 1.0)).to(tl.float32)
                dy = dt * (0.5 * tcv * (1.0 - tcv)) * tmask
                partial_base = (m * NUM_BLOCKS + nb // BLOCK_M) * D_SEED
                d_index = dc + di
                tl.store(dgamma_partial_ptr + partial_base + d_index,
                         tl.sum(dy * xc), mask=d_index < D_SEED)
                tl.store(dbeta_partial_ptr + partial_base + d_index,
                         tl.sum(dy), mask=d_index < D_SEED)
                tl.store(dzh_ptr + m * (N * D_SEED) + rows * D_SEED
                         + d_index, dy * gcv, mask=mask)
            # acc[d, g, r] += sum_n B[n,g] * dphi[n,r]  =  B^T @ dphi
            if DOT_IEEE:
                part = tl.dot(tl.trans(bgn), dphi, input_precision="ieee")
            else:
                part = tl.dot(tl.trans(bgn), dphi, input_precision="tf32x3")
            acc = tl.where(
                tl.arange(0, BLOCK_D)[:, None, None] == di,
                acc + part[None, :, :], acc)
    # store only the real g columns (padded columns are zero and masked out)
    tl.store(ddelta_ptr + m * (D_SEED * KAPPA * KRANK)
             + ccols[:, None, None] * (KAPPA * KRANK)
             + gcols_p[None, :, None] * KRANK
             + rcols[None, None, :],
             acc, mask=gmask[None, :, None])


@triton.jit
def _lev_bwd_ln_kernel(
    xhat_ptr, rsqrt_ptr, dzh_ptr, N,
    D_SEED: tl.constexpr, H_: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """LayerNorm reduction after fused ddelta writes dxhat into dzh."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid_b * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < N
    dcols = tl.arange(0, D_SEED)
    xhat = tl.load(xhat_ptr + pid_m * (N * D_SEED)
                   + rows[:, None] * D_SEED + dcols[None, :],
                   mask=mask[:, None], other=0.0).to(tl.float32)
    dxhat = tl.load(dzh_ptr + pid_m * (N * D_SEED)
                    + rows[:, None] * D_SEED + dcols[None, :],
                    mask=mask[:, None], other=0.0)
    rsqrt = tl.load(rsqrt_ptr + pid_m * N + rows, mask=mask, other=0.0)
    m1 = tl.sum(dxhat, 1, keep_dims=True) * (1.0 / D_SEED)
    m2 = tl.sum(dxhat * xhat, 1, keep_dims=True) * (1.0 / D_SEED)
    dzh = rsqrt[:, None] * (dxhat - m1 - xhat * m2)
    tl.store(dzh_ptr + pid_m * (N * D_SEED)
             + rows[:, None] * D_SEED + dcols[None, :],
             dzh, mask=mask[:, None])
