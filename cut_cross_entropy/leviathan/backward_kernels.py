"""Triton backward kernels for the LEV layer (deterministic, no atomics).

Design (main agent, 2026-08-07):
- Kernel 1 (_lev_bwd_chain_kernel): the expensive per-token chain — t,
  B-spline, phi, dphi, dB, dt, dy, LayerNorm backward, dzh — fused per
  (head, token block).  It stores only dzh (first as dxhat scratch, then
  overwritten with the final LayerNorm gradient) and block-local partials for
  dgamma/dbeta.  NO atomics: reduced-tile tl.atomic_add on cross-thread
  tl.sum results is redundant/nondeterministic in Triton 3.7 (measured ~350x
  over-accumulation that varies run to run).
- Kernel 2 (_lev_bwd_stats_kernel): reduces the block-local dgamma/dbeta
  partials.  The old per-token dy materialization is eliminated; the reduction
  is deterministic and reads only [heads, token_blocks, d_seed].
- Kernel 3 (_lev_bwd_ddelta_kernel): ddelta[d,g,r] = sum_n B*dphi, computed
  by RECOMPUTING the t/B/phi/dphi chain for each (d,r) tile while looping
  token blocks (grid h x d-tiles x r-tiles, register accumulation, single
  write).  Deterministic; total work = one extra pass over the phi chain.
- GEMM reductions (dM, dW_out, dW_l, dz, codebook scatter) run as cuBLAS/
  torch ops on the stored intermediates.
- Exact formulas mirror backward_impl.py (verified with gradcheck 23/23).

Gradients (given G = dE/dE_out [N, D]; per head l):
  dM     = G @ W_out^T
  dphi   = dM * M * sign(phi) / (|phi| + eps_log)
  ddelta = sum_n B * dphi
  dB     = sum_r dphi * (1 + delta)
  dw     = (D * dB - sum_g' dB_g' * w_g') / D^2
  dt     = sum_g dw_g * dw_dd_g * sign(t - g) * (kappa - 1)
  dy     = dt * 0.5 * t * (1 - t) * mask(t in (0,1))
  dxhat  = dy * gamma
  dzh    = rsqrt * (dxhat - mean_d(dxhat) - xhat * mean_d(dxhat * xhat))
  dgamma = sum_n dy * xhat ; dbeta = sum_n dy
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .runtime_policy import use_dot_specialization

EPS_LOG = 1e-9
NORM_EPS = 1e-5

_BWD_CONFIGS = [
    triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=1),
]


def _auto_fuse_chain_ddelta(device, kappa: int, krank: int) -> bool:
    """Select the fused chain+dDelta path only when the GPU can support it.

    The fused kernel is measured for Blackwell SM120 with one software stage;
    its generated shared-memory footprint is just under the 100 KiB consumer
    limit.  Query the runtime properties so GB202/RTX 5090 can use the same
    fused path without hard-coding its SM count, while older or constrained
    devices keep the deterministic two-pass implementation.  The current
    Blackwell profile uses BM=128 for long token blocks and keeps BM=32 for
    very short inputs.  The shared-memory guard and numerical suites still
    gate the selector; no compile-time global option is changed here.
    """
    if kappa != 16 or krank != 64 or getattr(device, "type", None) != "cuda":
        return False
    override = os.environ.get("LEV_FUSE_CHAIN_DDELTA")
    if override is not None:
        return override != "0"
    try:
        major, minor = torch.cuda.get_device_capability(device)
        props = torch.cuda.get_device_properties(device)
        shared_per_sm = int(getattr(props, "shared_memory_per_multiprocessor", 0))
        return (major, minor) >= (12, 0) and shared_per_sm >= 100 * 1024
    except (RuntimeError, TypeError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# kernel 1: per-(head, token-block) chain -> dxhat, dzh, dy (HBM)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_BWD_CONFIGS, key=["N"])
@triton.jit
def _lev_bwd_chain_kernel(
    xhat_ptr, t_ptr, rsqrt_ptr, modes_ptr, dm_ptr,
    gamma_ptr, beta_ptr, delta_ptr, knot_grid_ptr,
    dxhat_ptr, dzh_ptr, dgamma_partial_ptr, dbeta_partial_ptr, N, NUM_BLOCKS,
    D_SEED: tl.constexpr, KAPPA: tl.constexpr, KRANK: tl.constexpr,
    H_: tl.constexpr, BLOCK_M: tl.constexpr, D_CHUNK: tl.constexpr,
    EPS: tl.constexpr, LOG_EPS: tl.constexpr, USE_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    m = pid_m
    rows = pid_b * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < N

    dcols = tl.arange(0, D_SEED)
    rcols = tl.arange(0, KRANK)
    gcols = tl.arange(0, KAPPA)
    scale = KAPPA - 1.0
    hp = H_ * KRANK

    # per-head params + saved lean intermediates
    gamma = tl.load(gamma_ptr + m * D_SEED + dcols)
    beta = tl.load(beta_ptr + m * D_SEED + dcols)
    xhat = tl.load(xhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                   + dcols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + m * N + rows, mask=mask, other=0.0)
    dM = tl.load(dm_ptr + rows[:, None] * hp + m * KRANK + rcols[None, :],
                 mask=mask[:, None], other=0.0)
    M = tl.load(modes_ptr + rows[:, None] * hp + m * KRANK + rcols[None, :],
                mask=mask[:, None], other=0.0).to(tl.float32)

    # ---- chunk loop over d (mirror of the forward kernel's phase B) ----
    for dc in tl.range(0, D_SEED, D_CHUNK, loop_unroll_factor=1):
        ccols = dc + tl.arange(0, D_CHUNK)
        xc = tl.load(xhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                     + ccols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        gc = tl.load(gamma_ptr + m * D_SEED + ccols)
        bc = tl.load(beta_ptr + m * D_SEED + ccols)
        if USE_T:
            tc = tl.load(t_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                         + ccols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        else:
            tc = 1.0 / (1.0 + tl.exp(-0.5 * (xc * gc[None, :] + bc[None, :])))
        tc = tl.minimum(tl.maximum(tc, 0.0), 1.0)

        # vectorized basis over g: d3/w3 (one shot), denom from w3
        grid_v = tl.load(knot_grid_ptr + gcols)                 # [KAPPA]
        d3 = tl.abs(tc[:, :, None] - grid_v[None, None, :]) * scale
        w3 = tl.where(d3 < 0.5, 0.75 - d3 * d3,
                      tl.where(d3 < 1.5, 0.5 * (1.5 - d3) * (1.5 - d3),
                               0.0))                            # [BM, BD, KAPPA]
        denom = tl.maximum(tl.sum(w3, axis=2), 1e-12)

        # pass 2: phi tile
        phi = tl.zeros([BLOCK_M, D_CHUNK, KRANK], tl.float32)
        for g in tl.range(0, KAPPA, loop_unroll_factor=1):
            s = tl.load(delta_ptr + m * (D_SEED * KAPPA * KRANK)
                        + ccols[:, None] * (KAPPA * KRANK) + g * KRANK
                        + rcols[None, :])
            s = 1.0 + s.to(tl.float32)
            bgn = tl.sum(tl.where(gcols[None, None, :] == g, w3, 0.0),
                         axis=2) / denom
            phi += bgn[:, :, None] * s[None, :, :]

        # dphi (exact derivative of the log|.|+sign implementation)
        dphi = (dM[:, None, :] * M[:, None, :]
                * tl.where(phi >= 0, 1.0, -1.0) / (tl.abs(phi) + LOG_EPS))

        # pass 3: dB tile (per-g where-accumulate over r)
        dB = tl.zeros([BLOCK_M, D_CHUNK, KAPPA], tl.float32)
        for g in tl.range(0, KAPPA, loop_unroll_factor=1):
            s = tl.load(delta_ptr + m * (D_SEED * KAPPA * KRANK)
                        + ccols[:, None] * (KAPPA * KRANK) + g * KRANK
                        + rcols[None, :])
            s = 1.0 + s.to(tl.float32)
            dB = tl.where(gcols[None, None, :] == g,
                          tl.sum(dphi * s[None, :, :], axis=2)[:, :, None], dB)

        # spline backward, vectorized over g (exact split of the g-sum):
        #   dt = sum_g dw_g*dwd_g*sgn_g*scale,  dw_g = (D*dB_g - wsum)/D^2
        #   => dt = scale/D^2 * (D*A - wsum*C), A = sum_g dB*dwd*sgn,
        #      C = sum_g dwd*sgn        (denom D and wsum are g-independent)
        dwd3 = tl.where(d3 < 0.5, -2.0 * d3,
                        tl.where(d3 < 1.5, -(1.5 - d3), 0.0))
        sgn3 = tl.where(tc[:, :, None] - grid_v[None, None, :] >= 0,
                        1.0, -1.0)
        wsum = tl.sum(dB * w3, axis=2)
        A = tl.sum(dB * dwd3 * sgn3, axis=2)
        C = tl.sum(dwd3 * sgn3, axis=2)
        dt = (denom * A - wsum * C) * (scale / (denom * denom))

        # sigmoid(x/2) + clamp backward (per chunk)
        tmask = ((tc > 0.0) & (tc < 1.0)).to(tl.float32)
        dy = dt * (0.5 * tc * (1.0 - tc)) * tmask
        # Reduce the statistics while dy is still in registers.  Every
        # (head, token-block, d) element is written exactly once, so no atomic
        # operation or zero-initialized accumulation buffer is needed.
        partial_base = (m * NUM_BLOCKS + pid_b) * D_SEED
        tl.store(dgamma_partial_ptr + partial_base + ccols,
                 tl.sum(dy * xc, axis=0), mask=ccols < D_SEED)
        tl.store(dbeta_partial_ptr + partial_base + ccols,
                 tl.sum(dy, axis=0), mask=ccols < D_SEED)
        dxc = dy * gc[None, :]
        tl.store(dxhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                 + ccols[None, :], dxc, mask=mask[:, None])

    # dxhat full tile: store/load ordered by a CTA barrier (the forward
    # kernel proved this pattern removes the compiler store/load race).
    tl.debug_barrier()
    dxhat = tl.load(dxhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                    + dcols[None, :], mask=mask[:, None], other=0.0)

    # LayerNorm backward (full-d): dzh = rsqrt * (dxhat - m1 - xhat*m2)
    m1 = tl.sum(dxhat, 1, keep_dims=True) * (1.0 / D_SEED)
    m2 = tl.sum(dxhat * xhat, 1, keep_dims=True) * (1.0 / D_SEED)
    dzh = rsqrt[:, None] * (dxhat - m1 - xhat * m2)
    tl.store(dzh_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
             + dcols[None, :], dzh, mask=mask[:, None])


# ---------------------------------------------------------------------------
# kernel 2: dgamma/dbeta = sum_n dy*xhat / sum_n dy  (deterministic reduce)
# grid: (h, d-tiles); loops token blocks; register accumulation; one write.
# ---------------------------------------------------------------------------
@triton.jit
def _lev_bwd_stats_kernel(
    dgamma_partial_ptr, dbeta_partial_ptr, dgamma_ptr, dbeta_ptr, NUM_BLOCKS,
    D_SEED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    m = pid_m
    dc = pid_d * BLOCK_D
    ccols = dc + tl.arange(0, BLOCK_D)

    acc_g = tl.zeros([BLOCK_D], tl.float32)
    acc_b = tl.zeros([BLOCK_D], tl.float32)
    for nb in tl.range(0, NUM_BLOCKS, loop_unroll_factor=1):
        partial_base = (m * NUM_BLOCKS + nb) * D_SEED
        acc_g += tl.load(dgamma_partial_ptr + partial_base + ccols,
                         mask=ccols < D_SEED, other=0.0)
        acc_b += tl.load(dbeta_partial_ptr + partial_base + ccols,
                         mask=ccols < D_SEED, other=0.0)
    tl.store(dgamma_ptr + m * D_SEED + ccols, acc_g)
    tl.store(dbeta_ptr + m * D_SEED + ccols, acc_b)


# ---------------------------------------------------------------------------
# kernel 3: ddelta[d,g,r] = sum_n B*dphi  (deterministic recompute-reduce)
# grid: (h, d-tiles, r-tiles); loops token blocks; register accumulation.
# ---------------------------------------------------------------------------
@triton.jit
def _lev_bwd_ddelta_kernel(
    xhat_ptr, t_ptr, rsqrt_ptr, modes_ptr, dm_ptr,
    gamma_ptr, beta_ptr, delta_ptr, knot_grid_ptr, ddelta_ptr, N,
    D_SEED: tl.constexpr, KAPPA: tl.constexpr, KRANK: tl.constexpr,
    H_: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr, EPS: tl.constexpr, LOG_EPS: tl.constexpr,
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
    scale = KAPPA - 1.0
    hp = H_ * KRANK

    gc = tl.load(gamma_ptr + m * D_SEED + ccols)
    bc = tl.load(beta_ptr + m * D_SEED + ccols)

    acc = tl.zeros([BLOCK_D, KAPPA, BLOCK_R], tl.float32)
    for nb in tl.range(0, N, BLOCK_M, loop_unroll_factor=1):
        rows = nb + tl.arange(0, BLOCK_M)
        mask = rows < N
        xc = tl.load(xhat_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                     + ccols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        dM = tl.load(dm_ptr + rows[:, None] * hp + m * KRANK + rcols[None, :],
                     mask=mask[:, None], other=0.0)
        M = tl.load(modes_ptr + rows[:, None] * hp + m * KRANK
                    + rcols[None, :], mask=mask[:, None],
                    other=0.0).to(tl.float32)
        if USE_T:
            tc = tl.load(t_ptr + m * (N * D_SEED) + rows[:, None] * D_SEED
                         + ccols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        else:
            tc = 1.0 / (1.0 + tl.exp(-0.5 * (xc * gc[None, :] + bc[None, :])))
        tc = tl.minimum(tl.maximum(tc, 0.0), 1.0)
        # pass 1: normalizer D = sum_g w_g
        denom = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        for g in tl.range(0, KAPPA, loop_unroll_factor=1):
            gv = tl.load(knot_grid_ptr + g + tl.arange(0, 1))
            grid_g = tl.reshape(gv, [1, 1])
            d_ = tl.abs(tc - grid_g) * scale
            bg = tl.where(d_ < 0.5, 0.75 - d_ * d_,
                          tl.where(d_ < 1.5,
                                   0.5 * (1.5 - d_) * (1.5 - d_), 0.0))
            denom += bg
        denom = tl.maximum(denom, 1e-12)
        # pass 2: phi tile for (BLOCK_D, BLOCK_R)
        phi = tl.zeros([BLOCK_M, BLOCK_D, BLOCK_R], tl.float32)
        for g in tl.range(0, KAPPA, loop_unroll_factor=1):
            gv = tl.load(knot_grid_ptr + g + tl.arange(0, 1))
            grid_g = tl.reshape(gv, [1, 1])
            d_ = tl.abs(tc - grid_g) * scale
            bg = tl.where(d_ < 0.5, 0.75 - d_ * d_,
                          tl.where(d_ < 1.5,
                                   0.5 * (1.5 - d_) * (1.5 - d_), 0.0))
            bgn = bg / denom
            s = tl.load(delta_ptr + m * (D_SEED * KAPPA * KRANK)
                        + ccols[:, None] * (KAPPA * KRANK) + g * KRANK
                        + rcols[None, :])
            s = 1.0 + s.to(tl.float32)
            phi += bgn[:, :, None] * s[None, :, :]
        dphi = (dM[:, None, :] * M[:, None, :]
                * tl.where(phi >= 0, 1.0, -1.0) / (tl.abs(phi) + LOG_EPS))
        # pass 3: accumulate acc[d,g,r] += sum_n B[n,d,g]*dphi (bgn recompute)
        for g in tl.range(0, KAPPA, loop_unroll_factor=1):
            gv = tl.load(knot_grid_ptr + g + tl.arange(0, 1))
            grid_g = tl.reshape(gv, [1, 1])
            d_ = tl.abs(tc - grid_g) * scale
            bg = tl.where(d_ < 0.5, 0.75 - d_ * d_,
                          tl.where(d_ < 1.5,
                                   0.5 * (1.5 - d_) * (1.5 - d_), 0.0))
            bgn = bg / denom
            term = tl.sum(bgn[:, :, None] * dphi, axis=0)[:, None, :]
            acc = tl.where(gcols[None, :, None] == g, acc + term, acc)
    tl.store(ddelta_ptr + m * (D_SEED * KAPPA * KRANK)
             + ccols[:, None, None] * (KAPPA * KRANK)
             + gcols[None, :, None] * KRANK + rcols[None, None, :], acc)


def supports(cfg, saved) -> bool:
    """True when the Triton backward implements this configuration."""
    try:
        d = cfg.generator_d_seed
        kappa = cfg.generator_num_knots
        krank = cfg.generator_krank
        dtype = getattr(cfg, "dtype", torch.bfloat16)
        # pow2 requirements for register tiles (tl.arange)
        return (dtype == torch.bfloat16
                and d & (d - 1) == 0 and d >= 32
                and kappa & (kappa - 1) == 0 and 8 <= kappa <= 32
                and krank & (krank - 1) == 0 and 16 <= krank <= 128
                and isinstance(saved, dict)
                and saved.get("x_hat_por_head") is not None
                and saved.get("rsqrt_por_head") is not None
                and saved.get("z") is not None
                and saved.get("modes_por_head") is not None)
    except (AttributeError, TypeError):
        return False


def leviathan_backward_triton(grad_out, params, cfg, saved, ids):
    """Grads dict with the same keys as params (Triton chain + cuBLAS GEMMs).

    Returns None when the config is unsupported (caller falls back to the
    torch implementation).
    """
    if not supports(cfg, saved):
        return None

    dev = grad_out.device
    G_bf16 = grad_out.reshape(-1, cfg.hidden_size)
    use_dm_bf16 = os.environ.get("LEV_DM_BF16", "1") != "0"
    use_dwout_bf16 = os.environ.get("LEV_DWOUT_BF16", "1") != "0"
    # When both GEMMs consume the native BF16 training tensors, avoid a full
    # [N,D] FP32 grad-out materialization.  The diagnostic FP32 path remains
    # available through LEV_DM_BF16=0 or LEV_DWOUT_BF16=0.
    G = (None if use_dm_bf16 and use_dwout_bf16
         else G_bf16.float())
    N = G_bf16.shape[0]
    d = cfg.generator_d_seed
    kappa = cfg.generator_num_knots
    krank = cfg.generator_krank
    h = cfg.generator_num_modes
    D = cfg.hidden_size

    C = params["codebooks"]
    Wp = params["head_proj_weight"]
    gamma = params["head_norm_weight"]
    beta = params["head_norm_bias"]
    delta = params["head_spline_delta"]
    Wout = params["head_out_weight"]
    z = saved["z"]
    M = saved["modes_por_head"]            # [N, h, krank] bf16 (kernel fwd)
    if M.ndim == 3 and M.shape[0] == h and M.shape[1] == N:
        M = M.permute(1, 0, 2).contiguous()  # tolerate torch-bwd normalization
    xhat = saved["x_hat_por_head"]         # [h, N, d] fp32
    rsqrt = saved["rsqrt_por_head"]        # [h, N, 1] fp32
    t_saved = saved.get("t_por_head")
    use_t = t_saved is not None
    k = cfg.generator_k
    b = cfg.b

    # outputs (fp32 accumulation; cast to param dtype at the end)
    ddelta = torch.empty(h, d, kappa, krank, dtype=torch.float32, device=dev)
    dgamma = torch.empty(h, d, dtype=torch.float32, device=dev)
    dbeta = torch.empty(h, d, dtype=torch.float32, device=dev)
    dzh = torch.empty(h, N, d, dtype=torch.float32, device=dev)

    # dM = G @ W_out^T (cuBLAS GEMM).  Flattening the independent head/rank
    # axes gives cuBLAS a regular 2-D GEMM instead of an einsum dispatch, with
    # the same fp32 operands and no additional persistent workspace.
    if use_dm_bf16:
        # dM is an intermediate consumed by the Triton chain in fp32
        # arithmetic after load.  BF16 tensor-core accumulation is numerically
        # within the validated gate and saves both bandwidth and peak memory;
        # LEV_DM_BF16=0 retains the FP32 diagnostic path.
        Wout_bf16 = Wout.reshape(h * krank, D).contiguous()
        dM = G_bf16.matmul(Wout_bf16.t()).reshape(N, h, krank)
    else:
        Wout_flat = Wout.reshape(h * krank, D).float()
        dM = G.matmul(Wout_flat.t()).reshape(N, h, krank)

    knot_grid = saved.get("knot_grid")
    if knot_grid is None:
        grid_t = params.get("knot_grid")
        if grid_t is None:
            grid_t = torch.linspace(0.0, 1.0, kappa, dtype=torch.float32,
                                    device=dev)
        knot_grid = grid_t.to(device=dev, dtype=torch.float32).contiguous()

    xhat_c = xhat.reshape(h, N, d).contiguous()
    if use_t:
        t_c = t_saved.reshape(h, N, d).contiguous()
    else:
        # Keep compatibility with saved dictionaries created before t was
        # persisted; the constexpr branch does not touch this dummy pointer.
        t_c = torch.empty(1, 1, 1, dtype=torch.float16, device=dev)
    rsqrt_c = rsqrt.reshape(h, N).contiguous()
    M_c = M.reshape(N, h, krank).contiguous()

    # The chain writes per-block stats.  The same dzh allocation is passed as
    # both dxhat scratch and final dzh output; the kernel's barrier orders the
    # scratch load before the overwrite.  This removes two full [h,N,d] HBM
    # materializations (dy and a separate dxhat buffer).
    dot_chain_block_m = 128
    exact_chain_block_m = _BWD_CONFIGS[0].kwargs["BLOCK_M"]

    # ---- fast path: per-d dot kernels (SM120 tensor cores) ----
    # Match the forward dispatch: the padded dot path is not used for
    # KAPPA<16, where the exact chain is the validated implementation.
    if use_dot_specialization(
        dev,
        d_seed=d,
        num_knots=kappa,
        krank=krank,
    ):
        from . import backward_dot_kernels as bdk

        dot_ieee = os.environ.get("LEV_DOT_IEEE", "1") != "0"
        # Keep dM and M independent by default.  Premultiplying them in the
        # BF16 dM buffer was not measurably faster end-to-end on SM120 and
        # raised relative gradient error by roughly one order of magnitude.
        premul_dmm = os.environ.get("LEV_PREMUL_DMM", "0") != "0"
        if premul_dmm:
            # dM*M is independent of the seed dimension.  Keep the product in
            # the existing intermediate so the Triton kernels only load the
            # premultiplied value and promote it to FP32 for division.
            dM.mul_(M_c)
        # Measured defaults: on Blackwell SM120, BM=128/w4/s1 gives the
        # best dDelta throughput once there are enough token rows to fill the
        # larger tile. Short N<256 training blocks keep BM=32 so launch
        # occupancy and the L3 numerical gate remain stable.
        auto_fuse_chain = _auto_fuse_chain_ddelta(dev, kappa, krank)
        if kappa == 16 and krank >= 64:
            default_bm, default_bd, default_br = (
                ((32 if N < 256 else 128), 1, 64)
                if auto_fuse_chain else (32, 1, 64))
        else:
            default_bm, default_bd, default_br = 64, 2, 32
        block_m = int(os.environ.get("LEV_DDELTA_BM", default_bm))
        block_d = int(os.environ.get("LEV_DDELTA_BD", default_bd))
        block_r = min(int(os.environ.get("LEV_DDELTA_BR", default_br)), krank)
        if d % block_d or krank % block_r:
            raise ValueError(
                f"invalid ddelta tiles for d={d}, krank={krank}: "
                f"BM={block_m}, BD={block_d}, BR={block_r}"
            )
        fuse_chain_ddelta = (
            kappa == 16 and krank == 64 and block_d == 1
            and block_r == krank and block_m in (32, 64, 128)
            and (auto_fuse_chain or
                 os.environ.get("LEV_FUSE_CHAIN_DDELTA") not in (None, "0")))
        ddelta_launch_kwargs = (
            {"num_warps": 4, "num_stages": 1}
            if fuse_chain_ddelta and block_m >= 64 else {})
        if fuse_chain_ddelta:
            # One ddelta pass now also produces dxhat and block statistics;
            # no chain pass and no N-scaled ddelta partial workspace.
            num_blocks = triton.cdiv(N, block_m)
            dgamma_partial = torch.empty(
                h, num_blocks, d, dtype=torch.float32, device=dev)
            dbeta_partial = torch.empty_like(dgamma_partial)
            grid3 = (h, d // block_d, krank // block_r)
            bdk._lev_bwd_ddelta_dot_kernel[grid3](
                xhat_c, t_c, rsqrt_c, M_c, dM,
                gamma.contiguous(), beta.contiguous(), delta.contiguous(),
                knot_grid, ddelta, dzh, dgamma_partial, dbeta_partial,
                N, num_blocks,
                D_SEED=d, KAPPA=kappa, KAPPA_P=max(kappa, 16), KRANK=krank, H_=h,
                BLOCK_M=block_m, BLOCK_D=block_d, BLOCK_R=block_r,
                EPS=NORM_EPS, LOG_EPS=EPS_LOG, DOT_IEEE=dot_ieee,
                FUSE_CHAIN=True, PREMUL_DMM=premul_dmm, USE_T=use_t,
                **ddelta_launch_kwargs)
            grid_ln = (h, num_blocks)
            bdk._lev_bwd_ln_kernel[grid_ln](
                xhat_c, rsqrt_c, dzh, N,
                D_SEED=d, H_=h, BLOCK_M=block_m,
                num_warps=4, num_stages=1)
            grid2 = (h, d // 32)
            _lev_bwd_stats_kernel[grid2](
                dgamma_partial, dbeta_partial, dgamma, dbeta, num_blocks,
                D_SEED=d, BLOCK_M=1, BLOCK_D=32)
        else:
            num_blocks = triton.cdiv(N, dot_chain_block_m)
            dgamma_partial = torch.empty(
                h, num_blocks, d, dtype=torch.float32, device=dev)
            dbeta_partial = torch.empty_like(dgamma_partial)
            grid1 = (h, num_blocks)
            bdk._lev_bwd_chain_dot_kernel[grid1](
                xhat_c, t_c, rsqrt_c, M_c, dM,
                gamma.contiguous(), beta.contiguous(), delta.contiguous(),
                knot_grid, dzh, dzh, dgamma_partial, dbeta_partial,
                N, num_blocks,
                D_SEED=d, KAPPA=kappa, KAPPA_P=max(kappa, 16), KRANK=krank, H_=h,
                BLOCK_M=dot_chain_block_m, BLOCK_K=min(64, d),
                EPS=NORM_EPS, LOG_EPS=EPS_LOG, DOT_IEEE=dot_ieee, USE_T=use_t,
                PREMUL_DMM=premul_dmm,
                num_warps=4, num_stages=1)
            grid3 = (h, d // block_d, krank // block_r)
            bdk._lev_bwd_ddelta_dot_kernel[grid3](
                xhat_c, t_c, rsqrt_c, M_c, dM,
                gamma.contiguous(), beta.contiguous(), delta.contiguous(),
                knot_grid, ddelta, dzh, dgamma_partial, dbeta_partial,
                N, num_blocks,
                D_SEED=d, KAPPA=kappa, KAPPA_P=max(kappa, 16), KRANK=krank, H_=h,
                BLOCK_M=block_m, BLOCK_D=block_d, BLOCK_R=block_r,
                EPS=NORM_EPS, LOG_EPS=EPS_LOG, DOT_IEEE=dot_ieee,
                FUSE_CHAIN=False, PREMUL_DMM=premul_dmm, USE_T=use_t,
                **ddelta_launch_kwargs)
            grid2 = (h, d // 32)
            _lev_bwd_stats_kernel[grid2](
                dgamma_partial, dbeta_partial, dgamma, dbeta, num_blocks,
                D_SEED=d, BLOCK_M=1, BLOCK_D=32)
    else:
        # ---- kernel 1: per-(head, block) chain (raw launch — autotune wrapper
        # disabled for the same reason as the forward: nondeterministic race
        # across specializations in multi-config processes) ----
        c1 = _BWD_CONFIGS[0]
        num_blocks = triton.cdiv(N, exact_chain_block_m)
        dgamma_partial = torch.empty(
            h, num_blocks, d, dtype=torch.float32, device=dev)
        dbeta_partial = torch.empty_like(dgamma_partial)
        grid1 = (h, num_blocks)
        _lev_bwd_chain_kernel.fn[grid1](
            xhat_c, t_c, rsqrt_c, M_c, dM,
            gamma.contiguous(), beta.contiguous(), delta.contiguous(), knot_grid,
            dzh, dzh, dgamma_partial, dbeta_partial, N, num_blocks,
            D_SEED=d, KAPPA=kappa, KRANK=krank, H_=h,
            D_CHUNK=8, EPS=NORM_EPS, LOG_EPS=EPS_LOG, USE_T=use_t,
            BLOCK_M=c1.kwargs["BLOCK_M"],
            num_warps=c1.num_warps, num_stages=c1.num_stages)

        # ---- kernel 2: dgamma/dbeta (deterministic reduce) ----
        grid2 = (h, d // 32)
        _lev_bwd_stats_kernel[grid2](
            dgamma_partial, dbeta_partial, dgamma, dbeta, num_blocks,
            D_SEED=d, BLOCK_M=1, BLOCK_D=32)

        # ---- kernel 3: ddelta (deterministic recompute-reduce) ----
        # BLOCK_R adapts to krank (min 32; krank < 32 -> exact tile, pow2)
        block_r = min(32, krank)
        grid3 = (h, d // 8, krank // block_r)
        _lev_bwd_ddelta_kernel[grid3](
            xhat_c, t_c, rsqrt_c, M_c, dM,
            gamma.contiguous(), beta.contiguous(), delta.contiguous(), knot_grid,
            ddelta, N,
            D_SEED=d, KAPPA=kappa, KRANK=krank, H_=h,
            BLOCK_M=16, BLOCK_D=8, BLOCK_R=block_r,
            EPS=NORM_EPS, LOG_EPS=EPS_LOG, USE_T=use_t)

    # The partial statistics are fully reduced now; release them before the
    # remaining parameter reductions so they cannot inflate the live peak.
    del dgamma_partial, dbeta_partial

    # ---- GEMM reductions (cuBLAS) ----
    Wf = Wp.float()                         # [h, d, d]
    zf = z.float()                          # [N, d]
    dWp = torch.einsum("hnd,ni->hdi", dzh, zf)      # [h, d, d]
    dz = torch.einsum("hnd,hdi->ni", dzh, Wf)       # [N, d]
    if use_dwout_bf16:
        dWout = (M.reshape(N, h * krank).to(torch.bfloat16).t()
                 .matmul(G_bf16).reshape(h, krank, D))
    else:
        Mf = M.float()                      # [N, h, r]
        dWout = Mf.reshape(N, h * krank).t().matmul(G).reshape(h, krank, D)

    # codebook scatter (torch index_add — standard embedding backward)
    ids_l = ids.long().reshape(-1)
    coords = torch.empty(N, k, dtype=torch.long, device=dev)
    t_ids = ids_l.clone()
    for r in range(k - 1, -1, -1):
        coords[:, r] = t_ids % b
        t_ids = t_ids // b
    dC = torch.zeros_like(C, dtype=torch.float32)
    for r in range(k):
        dC[r].index_add_(0, coords[:, r], dz)

    # ---- cast to param dtypes (autograd convention) ----
    out = {
        "codebooks": dC.to(C.dtype),
        "head_proj_weight": dWp.to(Wp.dtype),
        "head_norm_weight": dgamma.to(gamma.dtype),
        "head_norm_bias": dbeta.to(beta.dtype),
        "head_spline_delta": ddelta.to(delta.dtype),
        "head_out_weight": dWout.to(Wout.dtype),
    }
    return out
