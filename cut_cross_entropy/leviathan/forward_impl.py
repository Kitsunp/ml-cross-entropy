"""Triton forward kernels for the LeviathanGenerator (LEV layer).

Reference math (leviathan_core.LeviathanGenerator, semantic oracle):
    b        = ceil(V^(1/k))                       (derived, never hardcoded)
    i -> (i_1..i_k)                                base-b decomposition
    z(i)     = sum_r C_r[i_r]                      codebook sum (bf16)
    per head l (h heads):
        zh      = sigmoid(1/2 * LN(W_seed,l z))    LN manual (eps=1e-5)
        B[n,d,g]  = quadratic B-spline of zh on kappa knots in [0,1],
                    normalized over g (clamp_min 1e-12)
        phi[n,d,r]= sum_g B[n,d,g] * (1 + delta_l[d,g,r])
        M_l[n,r]  = sign-parity product over d of phi   (= sign*exp(sum log|phi|))
    E[n,:]   = sum_l M_l[n,:] @ W_out,l

Architecture (2 kernels, NOTHING materialized for B/phi in HBM):
  * kernel A (_lev_fused_auto): one program per BLOCK_M-token block; inside it
    all h heads.  ids -> base-b digits -> codebook gather (bf16) -> z (stored,
    small).  Per head: K-chunked bf16 dot with W_seed^T (register-bounded),
    LayerNorm (two-pass, fp32), sigmoid(x/2) clamp [0,1] (fp32, stored to a
    per-token-block scratch x that each program owns -> no cross-program race),
    then the spline chain: B-spline basis tiles are computed per d-chunk of
    D_CHUNK columns and per knot g (registers only), phi accumulates in a
    3-D register tile [BLOCK_M, D_CHUNK, KRANK], log|phi|+1e-9 and sign parity
    accumulate over d, and M = (1 - 2*parity) * exp(log) is stored as bf16
    modes [N, h*krank].
  * kernel B (_lev_gemm_auto): modes [N, h*krank] @ W_out_cat [h*krank, D]
    -> embeds [N, D] bf16 (fp32 accumulation; reference accumulates in bf16
    per head -- see README "numerics deviations").

Memory (N=65536, D=512, d=128, h=8, krank=64, kappa=16, k=3, bf16):
    z     [N, 128]    bf16     16 MB     (also the saved intermediate)
    x     [N, 128]    fp32     32 MB     scratch (per-program rows)
    modes [N, 512]    bf16     64 MB     (saved intermediate)
    embeds[N, 512]    bf16     64 MB
    weights: codebooks 45 KB + W_seed 256 KB + delta 1 MB + W_out 512 KB (L2)
    total ~180 MB + allocator slack  < 10 GB budget at N=65536.
    The reference OOMs at N>=16384 because it materializes B [N,128,16] and
    phi [N,128,64] fp32 per head (~0.6 MB/token).

Autotune: ONLY hardware/tiling params -- BLOCK_M, BLOCK_N, BLOCK_K,
num_warps, num_stages and the vector-width hint VEC.  D_CHUNK (=4) is a fixed
design constant (numerics-neutral: the log-sum over d keeps the reference's
sequential order).  BLOCK_K = min(64, d_seed) is derived at launch.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# design constants (not autotuned; numerics-neutral)
# ---------------------------------------------------------------------------
D_CHUNK = 4          # d_seed columns processed per spline/phi register tile
LOG_EPS = 1e-9       # reference: torch.log(phi.abs() + 1e-9)
NORM_EPS = 1e-5      # reference: head_norm_eps
ROUND_ZH = False     # RECO policy (numerics §7.1): no bf16 rounding of the
                     # seed-projection output (fp32 chain). The reference
                     # rounds (F.linear bf16 output), but the fp32-oracle
                     # acceptance criterion (cos > 0.9999) is met only
                     # without the extra rounding; deviation <= 1 ulp fp32
                     # vs the reference's .float() of the bf16 dot.

# ---------------------------------------------------------------------------
# autotune configs -- hardware/tiling parameters ONLY
# ---------------------------------------------------------------------------
_A_CONFIGS = [
    # (BLOCK_M, num_warps, num_stages, VEC)
    triton.Config({"BLOCK_M": 16, "VEC": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 16, "VEC": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16, "VEC": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 16, "VEC": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "VEC": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32, "VEC": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "VEC": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 32, "VEC": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "VEC": 4}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "VEC": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "VEC": 4}, num_warps=16, num_stages=1),
]

_B_CONFIGS = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
]


# ---------------------------------------------------------------------------
# kernel A0: ids -> z (base-b decomposition + codebook gather/sum), in its
# OWN launch.  A separate launch serializes with kernel A on the stream, so
# the z store can never be reordered after kernel A's z loads (a shared-kernel
# store->load across the unrolled head loop raced nondeterministically).
# ---------------------------------------------------------------------------
@triton.jit
def _lev_gather_kernel(ids_ptr, codebooks_ptr, z_ptr, N,
                       B_: tl.constexpr, K_: tl.constexpr, KP2: tl.constexpr,
                       D_SEED: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < N
    # Digit order AND addition order mirror the reference bit-exactly
    # (least significant digit -> codebook k-1; sum C[0] first).
    t = tl.load(ids_ptr + rows, mask=mask, other=0).to(tl.int64)
    dg_all = tl.zeros([BLOCK_M, KP2], tl.int32)
    for j in tl.static_range(K_):
        dg = (t % B_).to(tl.int32)
        t = t // B_
        dg_all = tl.where(tl.arange(0, KP2)[None, :] == (K_ - 1 - j),
                          dg[:, None], dg_all)
    z = tl.zeros([BLOCK_M, D_SEED], tl.bfloat16)
    for r in tl.static_range(K_):
        digit_r = tl.sum(tl.where(tl.arange(0, KP2)[None, :] == r,
                                  dg_all, 0), axis=1)
        cb = tl.load(codebooks_ptr + r * (B_ * D_SEED)
                     + digit_r[:, None] * D_SEED
                     + tl.arange(0, D_SEED)[None, :])
        z += cb
    tl.store(z_ptr + rows[:, None] * D_SEED + tl.arange(0, D_SEED)[None, :],
             z, mask=mask[:, None])


# ---------------------------------------------------------------------------
# kernel A: fused ids -> modes  (spline chain fully in registers)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_A_CONFIGS, key=["N"])
@triton.jit
def _lev_fused_auto(
    ids_ptr, codebooks_ptr, w_seed_t_ptr, norm_w_ptr, norm_b_ptr,
    delta_ptr, knot_grid_ptr, z_ptr, x_ptr, t_ptr, xhat_ptr, mean_ptr, rsqrt_ptr,
    modes_ptr, N,
    B_: tl.constexpr, K_: tl.constexpr, KP2: tl.constexpr,
    D_SEED: tl.constexpr,
    KAPPA: tl.constexpr, KRANK: tl.constexpr, H_: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, D_CHUNK: tl.constexpr,
    VEC: tl.constexpr, SPLINE_BF16: tl.constexpr,
    ROUND_ZH: tl.constexpr, SAVE_XH: tl.constexpr, SAVE_T: tl.constexpr,
    EPS: tl.constexpr, LOG_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < N

    dcols = tl.arange(0, D_SEED)
    kcols_all = tl.arange(0, BLOCK_K)
    rcols_all = tl.arange(0, KRANK)

    for m in tl.range(0, H_, loop_unroll_factor=1):
        # ---- phase A: zh = z @ W_seed^T (bf16 dot, fp32 acc), LN, sigmoid ----
        zh = tl.zeros([BLOCK_M, D_SEED], tl.float32)
        for kc in tl.range(0, D_SEED, BLOCK_K, loop_unroll_factor=1):
            kcols = kc + kcols_all
            kmask = kcols < D_SEED
            zc = tl.load(z_ptr + rows[:, None] * D_SEED + kcols[None, :],
                         mask=mask[:, None] & kmask[None, :], other=0.0)
            wt = tl.load(w_seed_t_ptr + m * (D_SEED * D_SEED)
                         + kcols[:, None] * D_SEED + dcols[None, :],
                         mask=kmask[:, None], other=0.0)
            zh += tl.dot(zc, wt)
        if ROUND_ZH:
            zh = zh.to(tl.bfloat16).to(tl.float32)
        # LayerNorm (manual, two-pass, fp32; same formulation as reference)
        mean = tl.sum(zh, 1, keep_dims=True) * (1.0 / D_SEED)
        dev = zh - mean
        var = tl.sum(dev * dev, 1, keep_dims=True) * (1.0 / D_SEED)
        nw = tl.load(norm_w_ptr + m * D_SEED + dcols)
        nb = tl.load(norm_b_ptr + m * D_SEED + dcols)
        xhat = dev / tl.sqrt(var + EPS)
        x = xhat * nw[None, :] + nb[None, :]
        x = 1.0 / (1.0 + tl.exp(-0.5 * x))          # sigmoid(x/2), torch formula
        x = tl.minimum(tl.maximum(x, 0.0), 1.0)     # reference clamp(0, 1)
        # x roundtrip through per-head slots, ordered by a CTA barrier: without
        # the barrier the compiler could sink the store / hoist the load across
        # the unrolled head iterations and read stale torch.empty garbage
        # (nondeterministic head-0 corruption observed on the 5070 Ti).
        tl.store(x_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                 + dcols[None, :], x, mask=mask[:, None])
        if SAVE_T:
            tl.store(t_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                     + dcols[None, :], x.to(tl.float16), mask=mask[:, None])
        if SAVE_XH:  # lean saved intermediates (exact backward, no recompute)
            tl.store(xhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                     + dcols[None, :], xhat, mask=mask[:, None])
            tl.store(mean_ptr + m * N + rows[:, None]
                     + tl.zeros([BLOCK_M, 1], tl.int32),
                     mean, mask=mask[:, None])
            tl.store(rsqrt_ptr + m * N + rows[:, None]
                     + tl.zeros([BLOCK_M, 1], tl.int32),
                     1.0 / tl.sqrt(var + EPS), mask=mask[:, None])
        tl.debug_barrier()

        # ---- phase B: B-spline -> phi (registers) -> log|.| + sign parity ----
        log_acc = tl.zeros([BLOCK_M, KRANK], tl.float32)
        sign_acc = tl.zeros([BLOCK_M, KRANK], tl.int32)
        scale = KAPPA - 1.0
        for dc in tl.range(0, D_SEED, D_CHUNK, loop_unroll_factor=1):
            ccols = dc + tl.arange(0, D_CHUNK)
            xc = tl.load(x_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                         + ccols[None, :],
                         mask=mask[:, None], other=0.0)
            # pass 1: normalizer sum_g B (recompute of B is ~10 FLOP/elem)
            denom = tl.zeros([BLOCK_M, D_CHUNK], tl.float32)
            for g in tl.range(0, KAPPA, loop_unroll_factor=1):
                gv = tl.load(knot_grid_ptr + g + tl.arange(0, 1))
                grid_g = tl.reshape(gv, [1, 1])
                d_ = tl.abs(xc - grid_g) * scale
                bg = tl.where(d_ < 0.5, 0.75 - d_ * d_,
                              tl.where(d_ < 1.5,
                                       0.5 * (1.5 - d_) * (1.5 - d_), 0.0))
                denom += bg
            denom = tl.maximum(denom, 1e-12)
            # pass 2: phi register tile + log/parity accumulation over d
            phi = tl.zeros([BLOCK_M, D_CHUNK, KRANK], tl.float32)
            for g in tl.range(0, KAPPA, loop_unroll_factor=1):
                gv = tl.load(knot_grid_ptr + g + tl.arange(0, 1))
                grid_g = tl.reshape(gv, [1, 1])
                d_ = tl.abs(xc - grid_g) * scale
                bg = tl.where(d_ < 0.5, 0.75 - d_ * d_,
                              tl.where(d_ < 1.5,
                                       0.5 * (1.5 - d_) * (1.5 - d_), 0.0))
                bg = bg / denom
                s = tl.load(delta_ptr + m * (D_SEED * KAPPA * KRANK)
                            + ccols[:, None] * (KAPPA * KRANK) + g * KRANK
                            + rcols_all[None, :])               # bf16
                s = 1.0 + s.to(tl.float32)                      # 1 + delta
                if SPLINE_BF16:      # EXPERIMENTAL approximate variant
                    phi += (bg.to(tl.bfloat16)[:, :, None]
                            * s.to(tl.bfloat16)[None, :, :]).to(tl.float32)
                else:                # reference-exact fp32 chain
                    phi += bg[:, :, None] * s[None, :, :]
            log_acc += tl.sum(tl.log(tl.abs(phi) + LOG_EPS), axis=1)
            sign_acc ^= tl.sum((phi < 0).to(tl.int32), axis=1) & 1

        sgn = 1.0 - 2.0 * (sign_acc & 1).to(tl.float32)
        modes = (sgn * tl.exp(log_acc)).to(tl.bfloat16)
        tl.store(modes_ptr + rows[:, None] * (H_ * KRANK) + m * KRANK
                 + rcols_all[None, :], modes, mask=mask[:, None])


# ---------------------------------------------------------------------------
# kernel A-dot: same phase A; phase B built with per-d tl.dot (tensor cores).
# No phi save (memory stays lean); the dot replaces the 16-iteration g-loop.
# ---------------------------------------------------------------------------
@triton.jit
def _lev_fused_dot(
    ids_ptr, codebooks_ptr, w_seed_t_ptr, norm_w_ptr, norm_b_ptr,
    delta_ptr, knot_grid_ptr, z_ptr, x_ptr, t_ptr, xhat_ptr, mean_ptr, rsqrt_ptr,
    modes_ptr, N,
    B_: tl.constexpr, K_: tl.constexpr, KP2: tl.constexpr,
    D_SEED: tl.constexpr,
    KAPPA: tl.constexpr, KAPPA_P: tl.constexpr,
    KRANK: tl.constexpr, H_: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    VEC: tl.constexpr, SPLINE_BF16: tl.constexpr,
    ROUND_ZH: tl.constexpr, SAVE_XH: tl.constexpr, SAVE_T: tl.constexpr,
    DOT_IEEE: tl.constexpr, FUSE_GATHER: tl.constexpr, SAVE_Z: tl.constexpr,
    EPS: tl.constexpr, LOG_EPS: tl.constexpr,
    SPLIT_HEAD: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_head = tl.program_id(1)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < N

    dcols = tl.arange(0, D_SEED)
    kcols_all = tl.arange(0, BLOCK_K)
    rcols_all = tl.arange(0, KRANK)
    gcols = tl.arange(0, KAPPA)
    gcols_p = tl.arange(0, KAPPA_P)
    gmask = gcols_p < KAPPA
    grid_v = tl.load(knot_grid_ptr + gcols_p, mask=gmask, other=0.0)  # [KAPPA_P]
    scale = KAPPA - 1.0

    # Fused A0: derive z once in registers, preserving the reference's digit
    # order and bf16 codebook accumulation.  Training still stores z when the
    # backward needs it, but the separate gather launch and the following HBM
    # read are removed.  In inference SAVE_Z=False avoids the store entirely.
    if FUSE_GATHER:
        token = tl.load(ids_ptr + rows, mask=mask, other=0).to(tl.int64)
        digits = tl.zeros([BLOCK_M, KP2], tl.int32)
        for j in tl.static_range(K_):
            digit = (token % B_).to(tl.int32)
            token = token // B_
            digits = tl.where(
                tl.arange(0, KP2)[None, :] == (K_ - 1 - j),
                digit[:, None], digits)
        z_reg = tl.zeros([BLOCK_M, D_SEED], tl.bfloat16)
        for r in tl.static_range(K_):
            digit_r = tl.sum(
                tl.where(tl.arange(0, KP2)[None, :] == r, digits, 0), axis=1)
            cb = tl.load(codebooks_ptr + r * (B_ * D_SEED)
                         + digit_r[:, None] * D_SEED
                         + dcols[None, :])
            z_reg += cb
        if SAVE_Z:
            tl.store(z_ptr + rows[:, None] * D_SEED + dcols[None, :],
                     z_reg, mask=mask[:, None])

    head_count = 1 if SPLIT_HEAD else H_
    for mh in tl.range(0, head_count, loop_unroll_factor=1):
        m = pid_head + mh
        # ---- phase A: zh = z @ W_seed^T (bf16 dot, fp32 acc), LN, sigmoid ----
        zh = tl.zeros([BLOCK_M, D_SEED], tl.float32)
        for kc in tl.range(0, D_SEED, BLOCK_K, loop_unroll_factor=1):
            kcols = kc + kcols_all
            kmask = kcols < D_SEED
            if FUSE_GATHER:
                # FUSE_GATHER is launched with BLOCK_K=D_SEED, so the full
                # register tile is the dot operand (Triton does not support
                # dynamic tensor slicing of a register tile).
                zc = z_reg
            else:
                zc = tl.load(z_ptr + rows[:, None] * D_SEED + kcols[None, :],
                             mask=mask[:, None] & kmask[None, :], other=0.0)
            wt = tl.load(w_seed_t_ptr + m * (D_SEED * D_SEED)
                         + kcols[:, None] * D_SEED + dcols[None, :],
                         mask=kmask[:, None], other=0.0)
            zh += tl.dot(zc, wt)
        if ROUND_ZH:
            zh = zh.to(tl.bfloat16).to(tl.float32)
        mean = tl.sum(zh, 1, keep_dims=True) * (1.0 / D_SEED)
        dev = zh - mean
        var = tl.sum(dev * dev, 1, keep_dims=True) * (1.0 / D_SEED)
        nw = tl.load(norm_w_ptr + m * D_SEED + dcols)
        nb = tl.load(norm_b_ptr + m * D_SEED + dcols)
        xhat = dev / tl.sqrt(var + EPS)
        x = xhat * nw[None, :] + nb[None, :]
        x = 1.0 / (1.0 + tl.exp(-0.5 * x))
        x = tl.minimum(tl.maximum(x, 0.0), 1.0)
        tl.store(x_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                 + dcols[None, :], x, mask=mask[:, None])
        if SAVE_T:
            tl.store(t_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                     + dcols[None, :], x.to(tl.float16), mask=mask[:, None])
        if SAVE_XH:
            tl.store(xhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                     + dcols[None, :], xhat, mask=mask[:, None])
            tl.store(mean_ptr + m * N + rows[:, None]
                     + tl.zeros([BLOCK_M, 1], tl.int32),
                     mean, mask=mask[:, None])
            tl.store(rsqrt_ptr + m * N + rows[:, None]
                     + tl.zeros([BLOCK_M, 1], tl.int32),
                     1.0 / tl.sqrt(var + EPS), mask=mask[:, None])
        tl.debug_barrier()

        # ---- phase B: per-d basis (vectorized over g) + phi via tl.dot ----
        log_acc = tl.zeros([BLOCK_M, KRANK], tl.float32)
        sign_acc = tl.zeros([BLOCK_M, KRANK], tl.int32)
        for dc in tl.range(0, D_SEED, loop_unroll_factor=4):
            xc = tl.load(x_ptr + m * (N * D_SEED) + rows * D_SEED + dc,
                         mask=mask, other=0.0)                    # [BM]
            d3 = tl.abs(xc[:, None] - grid_v[None, :]) * scale    # [BM, KAPPA_P]
            w3 = tl.where(d3 < 0.5, 0.75 - d3 * d3,
                          tl.where(d3 < 1.5,
                                   0.5 * (1.5 - d3) * (1.5 - d3), 0.0))
            w3 = tl.where(gmask[None, :], w3, 0.0)                # pad -> 0
            denom = tl.maximum(tl.sum(w3, 1, keep_dims=True), 1e-12)
            bgn = w3 / denom                                      # [BM, KAPPA_P]
            st = tl.load(delta_ptr + m * (D_SEED * KAPPA * KRANK)
                         + dc * (KAPPA * KRANK)
                         + (gcols_p % KAPPA)[:, None] * KRANK
                         + rcols_all[None, :],
                         mask=gmask[:, None], other=0.0)          # bf16
            st = 1.0 + st.to(tl.float32)                          # [KAPPA_P, KRANK]
            if DOT_IEEE:
                phi = tl.dot(bgn, st, input_precision="ieee")
            else:
                phi = tl.dot(bgn, st, input_precision="tf32x3")
            log_acc += tl.log(tl.abs(phi) + LOG_EPS)
            sign_acc ^= (phi < 0).to(tl.int32) & 1

        sgn = 1.0 - 2.0 * (sign_acc & 1).to(tl.float32)
        modes = (sgn * tl.exp(log_acc)).to(tl.bfloat16)
        tl.store(modes_ptr + rows[:, None] * (H_ * KRANK) + m * KRANK
                 + rcols_all[None, :], modes, mask=mask[:, None])


# ---------------------------------------------------------------------------
# kernel B: modes @ W_out_cat -> embeds  (bf16 GEMM, fp32 acc)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_B_CONFIGS, key=["N", "D_OUT"])
@triton.jit
def _lev_gemm_auto(ids_ptr, modes_ptr, w_out_ptr, mask_embedding_ptr, e_ptr,
                   N, HK, D_OUT, MASK_TOKEN_ID: tl.constexpr,
                   HAS_MEAP: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in tl.range(0, HK, BLOCK_K):
        kmask = (k0 + rk) < HK
        a = tl.load(modes_ptr + rm[:, None] * HK + (k0 + rk)[None, :],
                    mask=(rm[:, None] < N) & kmask[None, :], other=0.0)
        b = tl.load(w_out_ptr + (k0 + rk)[:, None] * D_OUT + rn[None, :],
                    mask=kmask[:, None] & (rn[None, :] < D_OUT), other=0.0)
        acc += tl.dot(a, b)
    e = acc.to(tl.bfloat16)
    if HAS_MEAP:
        is_meap = tl.load(ids_ptr + rm, mask=rm < N, other=-1) == MASK_TOKEN_ID
        mask_embedding = tl.load(
            mask_embedding_ptr + rn,
            mask=rn < D_OUT,
            other=0.0,
        ).to(tl.bfloat16)
        e = tl.where(is_meap[:, None], mask_embedding[None, :], e)
    tl.store(e_ptr + rm[:, None] * D_OUT + rn[None, :], e,
             mask=(rm[:, None] < N) & (rn[None, :] < D_OUT))


# ---------------------------------------------------------------------------
# host-side plumbing
# ---------------------------------------------------------------------------
_PARAM_KEYS = ("codebooks", "head_proj_weight", "head_norm_weight",
               "head_norm_bias", "head_spline_delta", "head_out_weight")


def params_from_generator(gen) -> dict:
    """Extract the params dict from a leviathan_core.LeviathanGenerator."""
    p = {k: getattr(gen, k) for k in _PARAM_KEYS}
    p["knot_grid"] = gen.knot_grid
    return p


def _base_k_decompose(ids: torch.Tensor, cfg) -> torch.Tensor:
    """Bitwise mirror of the reference decomposition (for tests/backward)."""
    b = math.ceil(cfg.vocab_size ** (1.0 / cfg.generator_k))
    k = cfg.generator_k
    t = ids.long().clone()
    coords = torch.empty(*ids.shape, k, dtype=torch.long, device=ids.device)
    for r in range(k - 1, -1, -1):
        coords[..., r] = t % b
        t = t // b
    return coords


def _prepared(params: dict, cfg, device) -> dict:
    """Validate + layout-prepare the params (cheap; exact values preserved)."""
    k = cfg.generator_k
    b = math.ceil(cfg.vocab_size ** (1.0 / k))
    if b ** k < cfg.vocab_size:
        raise ValueError(f"base-b cannot represent vocab: b={b}, k={k}")
    d = cfg.generator_d_seed
    krank = cfg.generator_krank
    kappa = cfg.generator_num_knots
    h = cfg.generator_num_modes
    D = cfg.hidden_size

    def _chk(name, t, shape):
        if t is None or tuple(t.shape) != tuple(shape):
            raise ValueError(
                f"param {name}: expected shape {tuple(shape)}, got "
                f"{None if t is None else tuple(t.shape)}")
        if t.dtype != torch.bfloat16:
            raise TypeError(
                f"param {name}: kernel path requires bf16, got {t.dtype}")

    cb = params["codebooks"]; _chk("codebooks", cb, (k, b, d))
    wp = params["head_proj_weight"]; _chk("head_proj_weight", wp, (h, d, d))
    nw = params["head_norm_weight"]; _chk("head_norm_weight", nw, (h, d))
    nb = params["head_norm_bias"]; _chk("head_norm_bias", nb, (h, d))
    dl = params["head_spline_delta"]; _chk("head_spline_delta", dl, (h, d, kappa, krank))
    wo = params["head_out_weight"]; _chk("head_out_weight", wo, (h, krank, D))
    for name, t in (("codebooks", cb), ("head_proj_weight", wp),
                    ("head_norm_weight", nw), ("head_norm_bias", nb),
                    ("head_spline_delta", dl), ("head_out_weight", wo)):
        if t.device != device:
            raise RuntimeError(f"param {name} on {t.device}, expected {device}")
    grid = params.get("knot_grid")
    if grid is None:
        grid = torch.linspace(0.0, 1.0, kappa, dtype=torch.float32, device=device)
    else:
        grid = grid.to(device=device, dtype=torch.float32).contiguous()
    if grid.numel() != kappa:
        raise ValueError(f"knot_grid: expected {kappa} elements")

    # W_seed transposed (K-major for the dot) + fp32 LN params + W_out view
    return {
        "codebooks": cb.contiguous(),
        "w_seed_t": wp.mT.contiguous(),                 # [h, d, d]
        "norm_w": nw.float().contiguous(),              # [h, d] fp32
        "norm_b": nb.float().contiguous(),              # [h, d] fp32
        "delta": dl.contiguous(),                       # [h, d, kappa, krank] bf16
        "knot_grid": grid,                              # [kappa] fp32
        "w_out_cat": wo.reshape(h * krank, D).contiguous(),  # [h*krank, D] bf16
    }


def _use_raw_launch() -> bool:
    """Always raw-launch kernel A/B.

    The @triton.autotune wrapper is DISABLED (measured 2026-08-07): its
    benchmarking introduces a nondeterministic race across specializations
    in multi-config processes (NaN counts varied run to run in the harness
    correctness suite with autotune ON; clean with the raw path).  The
    autotuner's gain is nil anyway: the per-config timing at N=4096 shows
    config[0] (BLOCK_M=16, w=4, s=1, VEC=8) at 4.73ms == the best of all 11
    configs (8-warp variants are ~1.8x slower).
    """
    return True


def _auto_split_head(device, num_heads: int) -> bool:
    """Expose independent heads to the scheduler on Blackwell SM120+."""
    override = os.environ.get("LEV_SPLIT_HEAD")
    if override is not None:
        return override != "0"
    if getattr(device, "type", None) != "cuda" or num_heads < 4:
        return False
    try:
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor) >= (12, 0)
    except (RuntimeError, TypeError, AttributeError):
        return False


def leviathan_forward(ids, params, cfg, save_intermediates=False,
                      variant="exact", mask_embedding=None,
                      mask_token_id=None):
    """Kernel forward: (embeds, saved).

    ids:       int tensor [*shape] (any integral dtype)
    params:    dict(codebooks [k,b,d], head_proj_weight [h,d,d],
                    head_norm_weight [h,d], head_norm_bias [h,d],
                    head_spline_delta [h,d,kappa,krank],
                    head_out_weight [h,krank,D], knot_grid [kappa] optional)
    cfg:       leviathan_core.LeviathanConfig (or duck-typed)
    variant:   "exact" (reference math, fp32 spline chain) | "fast"
               (EXPERIMENTAL: bf16 spline products, approximate)
    save_intermediates: if True, saved = {"z", "B_por_head": None,
               "phi_por_head": None, "modes_por_head"}  (B/phi are NOT saved:
               the backward recomputes them from z -- see README)
    Returns (embeds [*shape, D], saved).
    Requires CUDA, or TRITON_INTERPRET=1 for CPU debugging (N <= 64).
    """
    if variant not in ("exact", "fast"):
        raise ValueError(f"variant must be 'exact' or 'fast', got {variant!r}")
    if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError(
            "leviathan_forward needs CUDA (or TRITON_INTERPRET=1 for CPU "
            "sanity checks); use dispatch.leviathan_embedding for the "
            "CPU reference fallback.")

    orig_shape = ids.shape
    N = ids.numel()
    ids = ids.to(device=params["codebooks"].device, dtype=torch.long).contiguous()
    device = ids.device
    prep = _prepared(params, cfg, device)

    d = cfg.generator_d_seed
    krank = cfg.generator_krank
    h = cfg.generator_num_modes
    D = cfg.hidden_size
    kappa = cfg.generator_num_knots
    k = cfg.generator_k
    b = math.ceil(cfg.vocab_size ** (1.0 / k))
    hk = h * krank
    has_meap = mask_embedding is not None or mask_token_id is not None
    if has_meap:
        if mask_embedding is None or mask_token_id is None:
            raise ValueError(
                "mask_embedding and mask_token_id must be provided together"
            )
        if mask_embedding.ndim != 1 or mask_embedding.numel() != D:
            raise ValueError(
                f"mask_embedding must have shape ({D},), got "
                f"{tuple(mask_embedding.shape)}"
            )
        if mask_embedding.device != device:
            raise ValueError("mask_embedding must be on the Leviathan device")
        if not mask_embedding.is_floating_point():
            raise TypeError("mask_embedding must be floating point")
        mask_embedding_kernel = mask_embedding.contiguous()
        mask_token_id_kernel = int(mask_token_id)
    else:
        # Zero-length view: no allocation and no load in the HAS_MEAP=False
        # specialization.
        mask_embedding_kernel = prep["knot_grid"].reshape(-1)[:0]
        mask_token_id_kernel = -1

    # Every element is written by A0/A before any consumer reads it; avoid
    # device-wide memset work and leave the launch ordering to the kernels.
    z = torch.empty(N, d, dtype=torch.bfloat16, device=device)
    x = torch.empty(h, N, d, dtype=torch.float32, device=device)  # per-head slots
    modes = torch.empty(N, hk, dtype=torch.bfloat16, device=device)
    embeds = torch.empty(N, D, dtype=torch.bfloat16, device=device)
    save_xh = bool(save_intermediates)
    # xhat is consumed in fp32 by backward.  An FP16 checkpoint halves the
    # persistent saved-intermediate footprint and passes the gates, but adds a
    # small backward conversion cost at large N; keep FP32 as the latency
    # default and expose the memory-first variant explicitly.
    save_xh_fp16 = save_xh and os.environ.get("LEV_SAVE_XH_FP16", "0") != "0"
    # Optional t checkpoint: numerically valid, but its fp16 store/load did
    # not amortize on the target GPU (A/B: +4.2% peak, +3.2% backward), so the
    # production default keeps the cheaper recomputation path.
    save_t = save_xh and os.environ.get("LEV_SAVE_T", "0") != "0"
    t_saved = (torch.empty(h, N, d, dtype=torch.float16, device=device)
               if save_t else torch.empty(1, 1, 1, dtype=torch.float16,
                                           device=device))
    if save_xh:
        xhat = torch.empty(
            h, N, d,
            dtype=torch.float16 if save_xh_fp16 else torch.float32,
            device=device)
        mean = torch.empty(h, N, dtype=torch.float32, device=device)
        rsqrt = torch.empty(h, N, dtype=torch.float32, device=device)
    else:
        xhat = torch.empty(1, 1, 1, dtype=torch.float32, device=device)
        mean = torch.empty(1, 1, dtype=torch.float32, device=device)
        rsqrt = torch.empty(1, 1, dtype=torch.float32, device=device)

    spline_bf16 = variant == "fast"
    block_k = min(64, d)

    # ---- launch kernel A0/A ----
    # The dot variant pads KAPPA<16 to a 16-wide dot.  Keep the exact chain
    # for those small knot grids until the padded path has a matching
    # numerical validation (the chain is already fast for these configs).
    dot_phi = os.environ.get("LEV_DOT") == "1" and kappa >= 16
    dot_ieee = os.environ.get("LEV_DOT_IEEE", "1") != "0"
    # A full z register tile is safe for the paper default d_seed=128.  Keep
    # the proven two-pass path for d_seed=256 to avoid register spills.
    fuse_gather = (dot_phi and d <= 128
                   and os.environ.get("LEV_FUSE_GATHER", "0") != "0")
    if fuse_gather:
        block_k = d
    if not fuse_gather:
        grid_g = (triton.cdiv(N, 256),)
        _lev_gather_kernel[grid_g](
            ids, prep["codebooks"], z, N,
            B_=b, K_=k, KP2=triton.next_power_of_2(k), D_SEED=d, BLOCK_M=256)
    grid_a = (triton.cdiv(N, _A_CONFIGS[0].kwargs["BLOCK_M"]),)
    if _use_raw_launch():
        c = _A_CONFIGS[0]
        if dot_phi:
            # Keep the measured default conservative, while exposing the
            # launch knobs for architecture-specific probes.  The selector is
            # deterministic and does not autotune at runtime.
            split_head = _auto_split_head(device, h) and not fuse_gather
            default_warps = 4 if split_head else 8
            cd = triton.Config(
                {
                    "BLOCK_M": int(os.environ.get("LEV_FWD_BM", "32")),
                    "VEC": int(os.environ.get("LEV_FWD_VEC", "8")),
                },
                num_warps=int(os.environ.get("LEV_FWD_WARPS",
                                              str(default_warps))),
                num_stages=int(os.environ.get("LEV_FWD_STAGES", "1")),
            )
            grid_ad = (
                triton.cdiv(N, cd.kwargs["BLOCK_M"]),
                h if split_head else 1,
            )
            _lev_fused_dot[grid_ad](
                ids, prep["codebooks"], prep["w_seed_t"], prep["norm_w"],
                prep["norm_b"], prep["delta"], prep["knot_grid"],
                z, x, t_saved, xhat, mean, rsqrt, modes, N,
                B_=b, K_=k, KP2=triton.next_power_of_2(k), D_SEED=d,
                KAPPA=kappa, KAPPA_P=max(kappa, 16), KRANK=krank, H_=h,
                BLOCK_M=cd.kwargs["BLOCK_M"], BLOCK_K=block_k,
                VEC=cd.kwargs["VEC"], SPLINE_BF16=spline_bf16,
                ROUND_ZH=ROUND_ZH, SAVE_XH=save_xh, SAVE_T=save_t,
                DOT_IEEE=dot_ieee, FUSE_GATHER=fuse_gather,
                SAVE_Z=save_xh, EPS=NORM_EPS, LOG_EPS=LOG_EPS,
                SPLIT_HEAD=split_head,
                num_warps=cd.num_warps, num_stages=cd.num_stages)
        else:
            _lev_fused_auto.fn[grid_a](
                ids, prep["codebooks"], prep["w_seed_t"], prep["norm_w"],
                prep["norm_b"], prep["delta"], prep["knot_grid"],
                z, x, t_saved, xhat, mean, rsqrt, modes, N,
                B_=b, K_=k, KP2=triton.next_power_of_2(k), D_SEED=d, KAPPA=kappa, KRANK=krank, H_=h,
                BLOCK_M=c.kwargs["BLOCK_M"], BLOCK_K=block_k, D_CHUNK=D_CHUNK,
                VEC=c.kwargs["VEC"], SPLINE_BF16=spline_bf16,
                ROUND_ZH=ROUND_ZH, SAVE_XH=save_xh, SAVE_T=save_t,
                EPS=NORM_EPS, LOG_EPS=LOG_EPS,
                num_warps=c.num_warps, num_stages=c.num_stages)
    else:
        _lev_fused_auto[grid_a](
            ids, prep["codebooks"], prep["w_seed_t"], prep["norm_w"],
            prep["norm_b"], prep["delta"], prep["knot_grid"],
            z, x, t_saved, xhat, mean, rsqrt, modes, N,
            B_=b, K_=k, KP2=triton.next_power_of_2(k), D_SEED=d, KAPPA=kappa, KRANK=krank, H_=h,
            BLOCK_K=block_k, D_CHUNK=D_CHUNK,
            SPLINE_BF16=spline_bf16, ROUND_ZH=ROUND_ZH, SAVE_XH=save_xh,
            SAVE_T=save_t,
            EPS=NORM_EPS, LOG_EPS=LOG_EPS)

    # ---- launch kernel B ----
    grid_b = (triton.cdiv(N, _B_CONFIGS[0].kwargs["BLOCK_M"]),
              triton.cdiv(D, _B_CONFIGS[0].kwargs["BLOCK_N"]))
    if _use_raw_launch():
        c = _B_CONFIGS[0]
        _lev_gemm_auto.fn[grid_b](
            ids, modes, prep["w_out_cat"], mask_embedding_kernel, embeds,
            N, hk, D, MASK_TOKEN_ID=mask_token_id_kernel,
            HAS_MEAP=has_meap,
            BLOCK_M=c.kwargs["BLOCK_M"], BLOCK_N=c.kwargs["BLOCK_N"],
            BLOCK_K=c.kwargs["BLOCK_K"],
            num_warps=c.num_warps, num_stages=c.num_stages)
    else:
        _lev_gemm_auto[grid_b](
            ids, modes, prep["w_out_cat"], mask_embedding_kernel, embeds,
            N, hk, D, MASK_TOKEN_ID=mask_token_id_kernel,
            HAS_MEAP=has_meap,
        )

    saved = None
    if save_intermediates:
        saved = {
            "z": z,
            "knot_grid": prep["knot_grid"],
            "x_hat_por_head": xhat,          # [h, N, d] fp32 — lean backward
            "t_por_head": t_saved if save_t else None,  # [h, N, d] fp16; avoids sigmoid recompute
            "mean_por_head": mean.unsqueeze(-1),    # [h, N, 1] fp32
            "rsqrt_por_head": rsqrt.unsqueeze(-1),  # [h, N, 1] fp32
            "B_por_head": None,      # never materialized (memory: see README)
            "phi_por_head": None,    # never materialized (memory: see README)
            "modes_por_head": modes.view(N, h, krank),
        }
    return embeds.view(*orig_shape, D), saved
