# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import math
import os

import torch
import triton
import triton.language as tl

from cut_cross_entropy.cce_patch import patch_target_backward
from cut_cross_entropy.mu_loss import add_mu_loss_gradient_kernel
from cut_cross_entropy.tl_autotune import (
    CCE_LOCK_BLOCK_B,
    CCE_LOCK_BLOCK_V,
    cce_backward_autotune,
)
from cut_cross_entropy.tl_utils import (
    b_bin_fn,
    tl_and_reduce_fn,
    tl_lock_add,
    tl_softcapping,
    tl_softcapping_grad,
)
from cut_cross_entropy.utils import TensorInfo
from cut_cross_entropy.vocab_parallel.utils import vp_reduce_e_grad


@triton.jit
def _scaled_gradient_cast_kernel(
    Input,
    Output,
    n_elements,
    inverse_scale,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(Input + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(Output + offsets, values * inverse_scale, mask=mask)


@triton.jit
def _scaled_gradient_cast_mu_kernel(
    Input,
    Output,
    Mu,
    VocabSize,
    dOut,
    n_elements,
    D,
    inverse_scale,
    mu_loss_lambda,
    BLOCK: tl.constexpr,
):
    """Cast the accumulated dC while adding μ-loss in the same memory pass."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(Input + offsets, mask=mask, other=0.0).to(tl.float32)
    dim = offsets % D
    mu = tl.load(Mu + dim, mask=mask, other=0.0).to(tl.float32)
    vocab_size = tl.load(VocabSize)
    d_out = tl.load(dOut)
    mu_gradient = d_out * (2.0 * mu_loss_lambda / vocab_size) * mu
    tl.store(Output + offsets, values * inverse_scale + mu_gradient, mask=mask)


def _finalize_scaled_gradient(
    gradient: torch.Tensor,
    output_dtype: torch.dtype,
    scale: float,
) -> torch.Tensor:
    assert gradient.is_contiguous()
    output = torch.empty_like(gradient, dtype=output_dtype)
    _scaled_gradient_cast_kernel[(triton.cdiv(gradient.numel(), 256),)](
        gradient,
        output,
        gradient.numel(),
        1.0 / scale,
        BLOCK=256,
        num_warps=4,
    )
    return output


def _finalize_scaled_gradient_with_mu(
    gradient: torch.Tensor,
    output_dtype: torch.dtype,
    scale: float,
    mu: torch.Tensor,
    mu_vocab_size: torch.Tensor,
    d_out: torch.Tensor,
    mu_loss_lambda: float,
) -> torch.Tensor:
    """Cast dC and add μ-loss without a second read/write over the FP32 buffer."""
    assert gradient.is_contiguous()
    assert mu.ndim == 1 and mu.numel() == gradient.size(1)
    assert mu_vocab_size.numel() == 1
    assert d_out.numel() == 1
    output = torch.empty_like(gradient, dtype=output_dtype)
    _scaled_gradient_cast_mu_kernel[(triton.cdiv(gradient.numel(), 256),)](
        gradient,
        output,
        mu,
        mu_vocab_size,
        d_out,
        gradient.numel(),
        gradient.size(1),
        1.0 / scale,
        mu_loss_lambda,
        BLOCK=256,
        num_warps=4,
    )
    return output


def _fp16_accum_scale(grad_scale: float, max_weight: float = 1.0) -> float:
    """Undo mean reduction with a conservative weighted-gradient bound."""
    if not math.isfinite(grad_scale) or grad_scale == 0.0:
        return 1.0
    if not math.isfinite(max_weight) or max_weight <= 0.0:
        return 1.0
    available = min(1024.0, 1.0 / abs(grad_scale))
    if max_weight > 1.0:
        # MiLe multiplies (P-Y) before the mean-reduction scale. Keep the
        # scaled target contribution below the finite FP16 range.
        available = min(
            available,
            torch.finfo(torch.float16).max / (abs(grad_scale) * max_weight),
        )
    if available < 2.0:
        return 1.0
    return float(2 ** math.floor(math.log2(available)))


_AUTO_FP16_OUTPUT_ELEMENTS = 8 * 1024 * 1024
_AUTO_FP16_MIN_SURFACE_ELEMENTS = 1024 * 1024
_AUTO_FP16_SUPPORTED_CC_MAJORS = frozenset({10, 12})


@functools.lru_cache(maxsize=None)
def _device_supports_auto_fp16_accumulation(device_index: int) -> bool:
    """Allow the supported Blackwell CC10.x/CC12.x device families."""
    major, _ = torch.cuda.get_device_capability(device_index)
    return major in _AUTO_FP16_SUPPORTED_CC_MAJORS


def _mile_weight_bound(vocab: int, gamma: float | None) -> float:
    """Return a device-free upper bound for normalized MiLe weights."""
    if gamma is None or gamma == 0.0:
        return 1.0
    if not math.isfinite(gamma) or gamma < 0.0:
        return math.inf
    if vocab <= 1:
        return 1.0
    try:
        # Entropy <= log(V), while the normalization denominator is >= 1.
        return float((1.0 + math.log(float(vocab))) ** gamma)
    except OverflowError:
        return math.inf


def _auto_fp16_accumulation_dtypes(
    e: torch.Tensor,
    e_info: TensorInfo,
    c: torch.Tensor,
    c_info: TensorInfo,
    effective_b: int,
    *,
    accum_e_fp32: bool,
    accum_c_fp32: bool,
    dlse: torch.Tensor | None,
    mile_weight: torch.Tensor | None,
    mu: torch.Tensor | None,
    mile_gamma: float | None,
    reduce_e_grad: bool,
    pg: torch.distributed.ProcessGroup | None,
) -> tuple[bool, bool]:
    """Conservative Blackwell CC10.x/CC12.x dispatch for the supported shape zone."""
    assert e.device.index is not None
    if not (
        accum_e_fp32
        and accum_c_fp32
        and e_info.requires_grad
        and c_info.requires_grad
        and e.is_contiguous()
        and c.is_contiguous()
        and e_info.dtype == torch.bfloat16
        and c_info.dtype == torch.bfloat16
        and dlse is None
        and not reduce_e_grad
        and pg is None
        and _device_supports_auto_fp16_accumulation(e.device.index)
    ):
        return False, False

    if mile_weight is not None:
        # Internal callers pass MiLe's exponent explicitly.  A direct kernel
        # caller that omits it must remain on the conservative FP32 path rather
        # than silently using an unbounded FP16 scale estimate.
        if mile_gamma is None:
            return False, False
        if not math.isfinite(_mile_weight_bound(c.size(0), mile_gamma)):
            return False, False

    dim = e.size(1)
    vocab = c.size(0)
    eligible = (dim >= 256 and (effective_b + vocab) * dim >= _AUTO_FP16_OUTPUT_ELEMENTS) or min(
        effective_b, vocab
    ) * dim >= _AUTO_FP16_MIN_SURFACE_ELEMENTS
    if not eligible:
        return False, False

    # With the fused μ cast, μ is added after the scaled CCE accumulation has
    # been converted back to the output dtype. This keeps the same guarded
    # FP16 CCE path as the no-μ case without mixing the regularizer into the
    # scaled accumulator. The opt-out keeps the historical FP32 destination.
    return True, mu is None or os.getenv("CCE_MU_FUSED_CAST", "1") != "0"


@triton.jit
def _cce_target_backward_kernel(
    E,
    C,
    MiLeWeight,
    dOut,
    Valids,
    Targets,
    dE,
    dC,
    dBias,
    grad_scale,
    de_accum_scale,
    dc_accum_scale,
    B,
    D,
    V,
    BMax,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_vb,
    shift,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_MILE: tl.constexpr,
    ITEM_DO: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    COMPUTE_DE: tl.constexpr,
    COMPUTE_DC: tl.constexpr,
    COMPUTE_DBIAS: tl.constexpr,
):
    direct_rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = direct_rows < B

    if HAS_VALIDS:
        rows = tl.load(Valids + direct_rows * stride_vb, mask=row_mask, other=BMax).to(tl.int64)
    else:
        rows = direct_rows.to(tl.int64)

    target_rows = rows + shift if HAS_SHIFT else rows
    targets = tl.load(
        Targets + target_rows,
        mask=row_mask & (target_rows < BMax),
        other=V,
    ).to(tl.int64)
    valid_target = row_mask & (targets >= 0) & (targets < V)

    if ITEM_DO:
        coefficient = -grad_scale * tl.load(dOut)
    else:
        coefficient = -grad_scale * tl.load(
            dOut + target_rows,
            mask=row_mask & (target_rows < BMax),
            other=0.0,
        )
    if HAS_MILE:
        coefficient *= tl.load(MiLeWeight + direct_rows, mask=row_mask, other=0.0)

    tile_mask = valid_target[:, None] & (offs_d[None, :] < D)
    if COMPUTE_DE:
        c = tl.load(
            C + targets[:, None] * stride_cv + offs_d[None, :] * stride_cd,
            mask=tile_mask,
            other=0.0,
        )
        de_ptrs = dE + rows[:, None] * stride_eb + offs_d[None, :] * stride_ed
        old_de = tl.load(de_ptrs, mask=tile_mask, other=0.0)
        tl.store(
            de_ptrs,
            old_de + coefficient[:, None] * de_accum_scale * c,
            mask=tile_mask,
        )

    if COMPUTE_DC:
        e = tl.load(
            E + rows[:, None] * stride_eb + offs_d[None, :] * stride_ed,
            mask=tile_mask,
            other=0.0,
        )
        dc_ptrs = dC + targets[:, None] * stride_cv + offs_d[None, :] * stride_cd
        tl.atomic_add(
            dc_ptrs,
            coefficient[:, None] * dc_accum_scale * e,
            mask=tile_mask,
            sem="relaxed",
            scope="gpu",
        )

    if COMPUTE_DBIAS:
        tl.atomic_add(
            dBias + targets,
            coefficient,
            mask=valid_target & (tl.program_id(1) == 0),
            sem="relaxed",
            scope="gpu",
        )


@triton.jit
def _mm_backward(
    do,
    da_ptrs,
    partial_mask_a,
    da_lock_ptr,
    n_locks,
    b_ptrs,
    partial_mask_b,
    stride_ad,
    stride_bd,
    D,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    d_inds = tl.arange(0, BLOCK_D)[None, :].to(tl.int64)

    b_ptrs = b_ptrs + d_inds * stride_bd
    da_ptrs = da_ptrs + d_inds * stride_ad

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            mask = partial_mask_b
        else:
            mask = partial_mask_b & (d_inds < (D - d * BLOCK_D))

        b = tl.load(b_ptrs, mask=mask, other=0.0)

        da_i = tl.dot(do, b, input_precision=DOT_PRECISION).to(da_ptrs.dtype.element_ty)

        if EVEN_D:
            mask = partial_mask_a
        else:
            mask = partial_mask_a & (d_inds < (D - d * BLOCK_D))

        if USE_ATOMIC:
            tl.atomic_add(da_ptrs, da_i, mask=mask, sem="relaxed", scope="gpu")
        else:
            lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
            this_da_lock_ptr = da_lock_ptr + lock_offset
            tl_lock_add(da_ptrs, da_i, mask, this_da_lock_ptr)

        b_ptrs += BLOCK_D * stride_bd
        da_ptrs += BLOCK_D * stride_ad


@triton.jit
def _block_is_filtered(check_val: tl.tensor, filter_eps: tl.tensor) -> tl.tensor:
    return tl.reduce(check_val < filter_eps, None, tl_and_reduce_fn)


@triton.jit
def _maybe_scale_gradient_tile(
    value,
    scale,
    output_dtype: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
):
    if APPLY_SCALE:
        return (value * scale).cast(output_dtype, fp_downcast_rounding="rtne")
    return value


def _cce_backward_kernel(
    E,
    C,
    Bias,
    LSE,
    MiLeWeight,
    dOut,
    grad_scale,
    dLSE,
    Valids,
    VocabOrdering,
    softcap,
    Targets,
    dE,
    dELocks,
    dC,
    dCLocks,
    dBias,
    B,
    D,
    V,
    BMax,
    n_de_locks_1,
    n_dc_locks_1,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_biasv,
    stride_vb,
    filter_eps,
    de_accum_scale,
    dc_accum_scale,
    shift,
    MODE,
    B_BIN,
    LOCK_BLOCK_B: tl.constexpr,
    LOCK_BLOCK_V: tl.constexpr,
    USE_ATOMIC_E: tl.constexpr,
    USE_ATOMIC_C: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MM_BACK_BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    EVEN_V: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    ITEM_DO: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_VOCAB_ORDERING: tl.constexpr,
    FILTER_E_GRAD: tl.constexpr,
    FILTER_C_GRAD: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_DLSE: tl.constexpr,
    HAS_MILE: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    COMPUTE_DC: tl.constexpr,
    COMPUTE_DE: tl.constexpr,
    COMPUTE_DBIAS: tl.constexpr,
    SCALE_DE: tl.constexpr,
    SCALE_DC: tl.constexpr,
    SHARED_ACCUM_SCALE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_v_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_v_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(tl.int64)

    direct_offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    v_mask = True if EVEN_V else direct_offs_v < V
    offs_v = direct_offs_v
    if HAS_VOCAB_ORDERING:
        offs_v = tl.load(VocabOrdering + direct_offs_v, mask=v_mask, other=0).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = v_mask[None, :] if not EVEN_V else True
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    accum = accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
    if HAS_BIAS:
        if EVEN_V:
            bias = tl.load(Bias + offs_v * stride_biasv)
        else:
            bias = tl.load(
                Bias + offs_v * stride_biasv,
                mask=v_mask,
                other=0.0,
            )
        accum += bias[None, :]

    if HAS_SOFTCAP:
        accum = tl_softcapping(accum, softcap)

    if HAS_VALIDS:
        direct_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
        lse = tl.load(LSE + direct_offs_b, mask=direct_offs_b < B, other=float("inf"))
    else:
        lse = tl.load(LSE + offs_b, mask=offs_b < B, other=float("inf"))

    accum = accum.cast(tl.float32)
    probabilities = tl.exp(accum - lse[:, None])
    if not EVEN_V:
        probabilities = tl.where(v_mask[None, :], probabilities, 0.0)
    d_accum = probabilities

    if HAS_TARGETS:
        if HAS_SHIFT:
            target_offs_b = offs_b + shift
        else:
            target_offs_b = offs_b

        targets = tl.load(Targets + target_offs_b, mask=target_offs_b < BMax, other=V + 1)
        is_target = targets[:, None] == offs_v[None, :]
        d_accum += tl.where(is_target, -1.0, 0.0)
    else:
        is_target = None

    if HAS_MILE:
        if HAS_VALIDS:
            mile_offs_b = direct_offs_b
        else:
            mile_offs_b = offs_b
        mile_row_mask = mile_offs_b < B
        mile_weight = tl.load(MiLeWeight + mile_offs_b, mask=mile_row_mask, other=0.0)[:, None]
        d_accum = mile_weight * d_accum

    should_skip = False
    if not (HAS_MILE and HAS_DLSE):
        if (FILTER_E_GRAD and COMPUTE_DE) and (FILTER_C_GRAD and COMPUTE_DC):
            if _block_is_filtered(tl.abs(d_accum), filter_eps):
                return
        elif (FILTER_E_GRAD and COMPUTE_DE) or (FILTER_C_GRAD and COMPUTE_DC):
            should_skip = _block_is_filtered(tl.abs(d_accum), filter_eps)

    if ITEM_DO:
        d_out = tl.load(dOut)
    else:
        if HAS_SHIFT:
            d_out_offs_b = offs_b + shift
        else:
            d_out_offs_b = offs_b

        d_out = tl.load(dOut + d_out_offs_b, mask=d_out_offs_b < BMax, other=0.0)[:, None]

    d_out = grad_scale * d_out

    if HAS_DLSE:
        if HAS_SHIFT:
            d_lse_offs_b = offs_b + shift
        else:
            d_lse_offs_b = offs_b

        d_lse = tl.load(dLSE + d_lse_offs_b, mask=d_lse_offs_b < BMax, other=0.0)[:, None]

        if HAS_MILE:
            d_accum = d_accum * d_out + probabilities * d_lse
        else:
            d_accum *= d_out + d_lse

        if HAS_TARGETS and not HAS_MILE:
            # We have d_accum = d_mm - is_target
            # We then want to get d_accum = d_mm * (d_out + d_lse) - is_target * d_out
            # If we do d_accum * (d_out + d_lse), we get d_mm * (d_out + d_lse) - is_target * (d_out + d_lse)
            # So we need to do d_accum += is_target * d_lse

            d_accum += tl.where(is_target, d_lse, 0.0)
    else:
        d_accum = d_accum * d_out

    if HAS_SOFTCAP:
        d_accum = tl_softcapping_grad(d_accum, accum, softcap)

    if COMPUTE_DBIAS:
        tl.atomic_add(
            dBias + offs_v * stride_biasv,
            tl.sum(d_accum, 0),
            mask=None if EVEN_V else v_mask,
        )

    if SHARED_ACCUM_SCALE:
        d_accum = (d_accum * de_accum_scale).cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        d_accum = d_accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")

    if COMPUTE_DE:
        if FILTER_E_GRAD:
            should_skip_e = should_skip
        else:
            should_skip_e = False

        if not should_skip_e:
            if USE_ATOMIC_E:
                de_locks = dELocks
            else:
                lock_offset = (pid_b * BLOCK_B // LOCK_BLOCK_B) * n_de_locks_1
                de_locks = dELocks + lock_offset

            _mm_backward(
                (
                    d_accum
                    if SHARED_ACCUM_SCALE
                    else _maybe_scale_gradient_tile(
                        d_accum, de_accum_scale, E.dtype.element_ty, SCALE_DE
                    )
                ),
                dE + (offs_b[:, None] * stride_eb),
                offs_b[:, None] < BMax,
                de_locks,
                n_de_locks_1,
                C + offs_v[:, None] * stride_cv,
                True if EVEN_V else v_mask[:, None],
                stride_ed,
                stride_cd,
                D,
                MM_BACK_BLOCK_D,
                MM_BACK_EVEN_D,
                USE_ATOMIC_E,
                DOT_PRECISION,
            )

    if COMPUTE_DC:
        if FILTER_C_GRAD:
            should_skip_c = should_skip
        else:
            should_skip_c = False

        if not should_skip_c:
            if USE_ATOMIC_C:
                dc_locks = dCLocks
            else:
                lock_offset = (pid_v * BLOCK_V // LOCK_BLOCK_V) * n_dc_locks_1
                dc_locks = dCLocks + lock_offset

            _mm_backward(
                tl.trans(
                    d_accum
                    if SHARED_ACCUM_SCALE
                    else _maybe_scale_gradient_tile(
                        d_accum, dc_accum_scale, E.dtype.element_ty, SCALE_DC
                    )
                ),
                dC + (offs_v[:, None] * stride_cv),
                True if EVEN_V else v_mask[:, None],
                dc_locks,
                n_dc_locks_1,
                E + (offs_b[:, None] * stride_eb),
                offs_b[:, None] < BMax,
                stride_cd,
                stride_ed,
                D,
                MM_BACK_BLOCK_D,
                MM_BACK_EVEN_D,
                USE_ATOMIC_C,
                DOT_PRECISION,
            )


def _cce_back_block_d(args) -> int:
    block_d = args["BLOCK_D"]
    return 2 * block_d


_cce_backward_kernel = triton.jit(_cce_backward_kernel, do_not_specialize=["MODE", "B_BIN"])
_cce_backward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "EVEN_V": lambda args: (args["V"] % args["BLOCK_V"]) == 0,
        "MM_BACK_BLOCK_D": lambda args: _cce_back_block_d(args),
        "MM_BACK_EVEN_D": lambda args: (args["D"] % _cce_back_block_d(args)) == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VOCAB_ORDERING": lambda args: args["VocabOrdering"] is not None,
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_DLSE": lambda args: args["dLSE"] is not None,
        "HAS_MILE": lambda args: args["MiLeWeight"] is not None,
        "HAS_SHIFT": lambda args: args["shift"] != 0,
        "ITEM_DO": lambda args: args["dOut"].numel() == 1,
        "GROUP_B": lambda args: 16,
        "COMPUTE_DC": lambda args: args["dC"] is not None,
        "COMPUTE_DE": lambda args: args["dE"] is not None,
        "COMPUTE_DBIAS": lambda args: args["dBias"] is not None,
        # MiLe forward computes its weighted-logit moment with IEEE products;
        # reconstruct logits with the same precision so backward differentiates
        # the loss that was actually evaluated.
        "DOT_PRECISION": lambda args: (
            "ieee"
            if args["MiLeWeight"] is not None
            else ("tf32" if torch.get_float32_matmul_precision() == "high" else "ieee")
        ),
    }
)(_cce_backward_kernel)
_cce_backward_kernel = cce_backward_autotune()(_cce_backward_kernel)  # type: ignore


def cce_backward_kernel(
    do: torch.Tensor,
    dlse: torch.Tensor | None,
    e: torch.Tensor,
    e_info: TensorInfo,
    c: torch.Tensor,
    c_info: TensorInfo,
    bias: torch.Tensor | None,
    bias_info: TensorInfo | None,
    lse: torch.Tensor,
    mile_weight: torch.Tensor | None,
    patch_target_weight: torch.Tensor | None,
    mu: torch.Tensor | None,
    mu_vocab_size: torch.Tensor | None,
    mu_loss_lambda: float | None,
    valids: torch.Tensor | None,
    softcap: float | None,
    filter_eps: float | None,
    targets: torch.Tensor | None = None,
    shift: int = 0,
    vocab_ordering: torch.Tensor | None = None,
    grad_scale: float = 1.0,
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    auto_mixed_grad_accum: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    reduce_e_grad: bool = False,
    pg: torch.distributed.ProcessGroup | None = None,
    mile_gamma: float | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    assert do.numel() in (e.size(0), 1)
    assert c.size(1) == e.size(1)
    assert lse.size(0) == e.size(0) or (valids is not None and lse.size(0) == valids.size(0))

    do = do.contiguous()
    lse = lse.contiguous()
    if mile_weight is not None:
        mile_weight = mile_weight.contiguous()
    if patch_target_weight is not None:
        assert targets is not None and targets.ndim == 2
        patch_target_weight = patch_target_weight.contiguous()
    if mu is not None:
        assert mu_vocab_size is not None
        assert mu_loss_lambda is not None
        mu = mu.contiguous()
        mu_vocab_size = mu_vocab_size.contiguous()

    default_accum_dtype = "auto" if auto_mixed_grad_accum else "fp32"
    de_accum_dtype = os.getenv("CCE_DE_ACCUM_DTYPE", default_accum_dtype)
    dc_accum_dtype = os.getenv("CCE_DC_ACCUM_DTYPE", default_accum_dtype)
    valid_accum_dtypes = {"auto", "fp32", "fp16", "bf16"}
    if de_accum_dtype not in valid_accum_dtypes:
        raise ValueError("CCE_DE_ACCUM_DTYPE must be 'auto', 'fp32', 'fp16', or 'bf16'")
    if dc_accum_dtype not in valid_accum_dtypes:
        raise ValueError("CCE_DC_ACCUM_DTYPE must be 'auto', 'fp32', 'fp16', or 'bf16'")
    if (de_accum_dtype == "auto") != (dc_accum_dtype == "auto"):
        raise ValueError(
            "CCE_DE_ACCUM_DTYPE and CCE_DC_ACCUM_DTYPE must both be 'auto' "
            "when automatic mixed accumulation is requested"
        )
    effective_b = valids.size(0) if valids is not None else e.size(0)
    if de_accum_dtype == "auto":
        use_fp16_e, use_fp16_c = _auto_fp16_accumulation_dtypes(
            e,
            e_info,
            c,
            c_info,
            effective_b,
            accum_e_fp32=accum_e_fp32,
            accum_c_fp32=accum_c_fp32,
            dlse=dlse,
            mile_weight=mile_weight,
            mu=mu,
            mile_gamma=mile_gamma,
            reduce_e_grad=reduce_e_grad,
            pg=pg,
        )
        de_accum_dtype = "fp16" if use_fp16_e else "fp32"
        dc_accum_dtype = "fp16" if use_fp16_c else "fp32"

    # The fused path can keep the CCE accumulation in the same conservative
    # scaled-FP16 mode used by cce_kahan_full_c. μ is added after the CCE
    # value is unscaled, so it does not need to share the FP32 accumulator.
    mu_fp16_fastpath = (
        mu is not None
        and os.getenv("CCE_MU_FUSED_CAST", "1") != "0"
        and auto_mixed_grad_accum
        and dc_accum_dtype == "fp16"
    )
    # Outside that explicitly guarded path, keep the original FP32 guarantee
    # for the independent μ-loss dC term.
    if mu is not None and dc_accum_dtype != "fp32" and not mu_fp16_fastpath:
        dc_accum_dtype = "fp32"

    if accum_e_fp32:
        de_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[de_accum_dtype]
    else:
        de_dtype = None
    de = torch.zeros_like(e, dtype=de_dtype) if e_info.requires_grad else None

    if accum_c_fp32:
        dc_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[dc_accum_dtype]
    else:
        dc_dtype = None
    dc = torch.zeros_like(c, dtype=dc_dtype) if c_info.requires_grad else None

    accum_e_fp32 = accum_e_fp32 and de is not None
    accum_c_fp32 = accum_c_fp32 and dc is not None

    backward_reduction = os.getenv("CCE_BACKWARD_REDUCTION", "auto")
    if backward_reduction not in {"auto", "lock", "atomic"}:
        raise ValueError("CCE_BACKWARD_REDUCTION must be 'auto', 'lock', or 'atomic'")
    use_atomic_e = (
        backward_reduction != "lock"
        and de is not None
        and (de.dtype == torch.float32 or (accum_e_fp32 and de_accum_dtype != "fp32"))
    )
    use_atomic_c = (
        backward_reduction != "lock"
        and dc is not None
        and (dc.dtype == torch.float32 or (accum_c_fp32 and dc_accum_dtype != "fp32"))
    )
    fp16_scale_mode = os.getenv("CCE_FP16_ACCUM_SCALE", "auto")
    if fp16_scale_mode not in {"auto", "off"}:
        raise ValueError("CCE_FP16_ACCUM_SCALE must be 'auto' or 'off'")
    mile_weight_bound = (
        _mile_weight_bound(c.size(0), mile_gamma) if mile_weight is not None else 1.0
    )
    if patch_target_weight is not None:
        # Patch rows can contain a data-dependent number of valid targets.
        # B is a conservative bound for the normalized dense row coefficient.
        mile_weight_bound *= effective_b
    safe_scale = (
        _fp16_accum_scale(grad_scale, mile_weight_bound)
        if fp16_scale_mode == "auto" and dlse is None
        else 1.0
    )
    de_accum_scale = safe_scale if de is not None and de.dtype == torch.float16 else 1.0
    dc_accum_scale = (
        safe_scale
        if dc is not None and dc.dtype == torch.float16 and (mu is None or mu_fp16_fastpath)
        else 1.0
    )

    if bias is not None:
        assert bias_info is not None
        dbias = torch.zeros_like(bias, dtype=torch.float32) if bias_info.requires_grad else None
    else:
        dbias = None

    if de is not None:
        assert de.stride() == e.stride()

    if dc is not None:
        assert dc.stride() == c.stride()

    if dbias is not None:
        assert bias is not None
        assert dbias.stride() == bias.stride()

    if valids is not None:
        assert valids.ndim == 1
        B = valids.size(0)
    else:
        B = e.size(0)

    if dlse is not None:
        dlse = dlse.contiguous()
        if do.numel() > 1:
            assert dlse.size() == do.size()

    if do.numel() > 1:
        do = do.contiguous()
        lse = lse.contiguous()
        assert do.stride(0) == lse.stride(0), f"{do.stride()=}, {lse.stride()=}"

    if vocab_ordering is not None:
        assert vocab_ordering.ndim == 1
        assert vocab_ordering.numel() == c.size(0)
        assert vocab_ordering.stride(0) == 1

    # The one-hot target term is sparse in vocabulary.  Without softcapping,
    # apply it as an indexed O(B*D) update after the dense probability kernel
    # once the dense BxV work is large enough to repay the extra launch.  Tiny
    # problems keep the already-supported fused path, where launch latency is
    # more expensive than the target comparison.
    patch_targets = patch_target_weight is not None
    separate_target = (
        targets is not None and not patch_targets and softcap is None and B * c.size(0) >= 1 << 24
    )
    kernel_targets = None if separate_target or patch_targets else targets

    nd_locks = triton.cdiv(c.size(1), 64)
    if de is not None and not use_atomic_e:
        de_locks = e.new_zeros((triton.cdiv(B, CCE_LOCK_BLOCK_B), nd_locks), dtype=torch.int32)
        de_lock_sizes = de_locks.size()
    else:
        de_locks = None
        de_lock_sizes = (None, 1)

    if dc is not None and not use_atomic_c:
        dc_locks = c.new_zeros(
            (triton.cdiv(c.size(0), CCE_LOCK_BLOCK_V), nd_locks), dtype=torch.int32
        )
        dc_lock_sizes = dc_locks.size()
    else:
        dc_locks = None
        dc_lock_sizes = (None, 1)

    def launch_vocab_slice(
        c_slice: torch.Tensor,
        bias_slice: torch.Tensor | None,
        dc_slice: torch.Tensor | None,
        dbias_slice: torch.Tensor | None,
        ordering_slice: torch.Tensor | None,
        slice_v: int,
    ) -> None:
        def grid(META):
            return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(slice_v, META["BLOCK_V"]),)

        _cce_backward_kernel[grid](
            e,
            c_slice,
            bias_slice,
            lse,
            mile_weight,
            do,
            grad_scale,
            dlse,
            valids,
            ordering_slice,
            softcap,
            kernel_targets,
            de,
            de_locks,
            dc_slice,
            dc_locks,
            dbias_slice,
            B,
            e.size(1),
            slice_v,
            e.size(0),
            de_lock_sizes[1],
            dc_lock_sizes[1],
            e.stride(0),
            e.stride(1),
            c_slice.stride(0),
            c_slice.stride(1),
            1 if bias_slice is None else bias_slice.stride(0),
            1 if valids is None else valids.stride(0),
            filter_eps,
            de_accum_scale,
            dc_accum_scale,
            shift=shift,
            MODE=(
                (
                    bias_slice is not None
                    or valids is not None
                    or ordering_slice is not None
                    or kernel_targets is not None
                    or shift != 0
                )
                | ((softcap is not None or dlse is not None) << 1)
                | ((mile_weight is not None) << 2)
                | ((de is not None) << 3)
                | ((dc_slice is not None) << 4)
                | ((dbias_slice is not None) << 5)
                | (
                    ((filter_e_grad and de is not None) or (filter_c_grad and dc_slice is not None))
                    << 6
                )
                | ((do.numel() == 1) << 7)
                | ((use_atomic_e or use_atomic_c) << 8)
            ),
            B_BIN=b_bin_fn(B),
            LOCK_BLOCK_B=CCE_LOCK_BLOCK_B,
            LOCK_BLOCK_V=CCE_LOCK_BLOCK_V,
            USE_ATOMIC_E=use_atomic_e,
            USE_ATOMIC_C=use_atomic_c,
            SCALE_DE=de_accum_scale != 1.0,
            SCALE_DC=dc_accum_scale != 1.0,
            SHARED_ACCUM_SCALE=(
                de is not None
                and dc_slice is not None
                and de_accum_scale != 1.0
                and de_accum_scale == dc_accum_scale
            ),
            FILTER_E_GRAD=filter_e_grad and de is not None,
            FILTER_C_GRAD=filter_c_grad and dc_slice is not None,
        )

    launch_vocab_slice(
        c,
        bias,
        dc,
        dbias,
        vocab_ordering,
        c.size(0),
    )

    if separate_target:
        assert targets is not None
        target_block_b = 64
        target_block_d = 64
        _cce_target_backward_kernel[
            (triton.cdiv(B, target_block_b), triton.cdiv(e.size(1), target_block_d))
        ](
            e,
            c,
            mile_weight,
            do,
            valids,
            targets,
            de,
            dc,
            dbias,
            grad_scale,
            de_accum_scale,
            dc_accum_scale,
            B,
            e.size(1),
            c.size(0),
            e.size(0),
            e.stride(0),
            e.stride(1),
            c.stride(0),
            c.stride(1),
            1 if valids is None else valids.stride(0),
            shift,
            BLOCK_B=target_block_b,
            BLOCK_D=target_block_d,
            HAS_VALIDS=valids is not None,
            HAS_MILE=mile_weight is not None,
            ITEM_DO=do.numel() == 1,
            HAS_SHIFT=shift != 0,
            COMPUTE_DE=de is not None,
            COMPUTE_DC=dc is not None,
            COMPUTE_DBIAS=dbias is not None,
            num_warps=4,
        )

    if patch_targets:
        assert targets is not None
        assert patch_target_weight is not None
        patch_target_backward(
            e,
            c,
            targets,
            patch_target_weight,
            do,
            de,
            dc,
            dbias,
            grad_scale,
            de_accum_scale,
            dc_accum_scale,
        )

    if reduce_e_grad and de is not None:
        de = vp_reduce_e_grad(de, pg)

    if dbias is not None:
        assert bias_info is not None
        dbias = dbias.to(dtype=bias_info.dtype)

    if dc is not None:
        fused_mu_cast = (
            mu is not None
            and os.getenv("CCE_MU_FUSED_CAST", "1") != "0"
            and (dc.dtype == torch.float32 or mu_fp16_fastpath)
            and c_info.dtype != torch.float32
        )
        if mu is not None and not fused_mu_cast:
            assert mu_vocab_size is not None
            assert mu_loss_lambda is not None
            add_mu_loss_gradient_kernel(dc, mu, mu_vocab_size, do, mu_loss_lambda)
        if dc_accum_scale != 1.0:
            if fused_mu_cast:
                assert mu_vocab_size is not None
                assert mu_loss_lambda is not None
                dc = _finalize_scaled_gradient_with_mu(
                    dc,
                    c_info.dtype,
                    dc_accum_scale,
                    mu,
                    mu_vocab_size,
                    do,
                    mu_loss_lambda,
                )
            else:
                dc = _finalize_scaled_gradient(dc, c_info.dtype, dc_accum_scale)
        else:
            if fused_mu_cast:
                assert mu_vocab_size is not None
                assert mu_loss_lambda is not None
                dc = _finalize_scaled_gradient_with_mu(
                    dc,
                    c_info.dtype,
                    1.0,
                    mu,
                    mu_vocab_size,
                    do,
                    mu_loss_lambda,
                )
            else:
                dc = dc.to(dtype=c_info.dtype)

    if de is not None:
        if de_accum_scale != 1.0:
            de = _finalize_scaled_gradient(de, e_info.dtype, de_accum_scale)
        else:
            de = de.to(dtype=e_info.dtype)

    return de, dc, dbias
