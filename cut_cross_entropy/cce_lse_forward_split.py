# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Two-stage CCE forward reduction without cross-program spinlocks."""

import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_utils import tl_logaddexp, tl_softcapping


@triton.jit
def _cce_lse_split_partials_kernel(
    E,
    C,
    Bias,
    PartialLSE,
    PartialMeanLogit,
    LA,
    NegCorrectLogit,
    Valids,
    Targets,
    softcap,
    shift,
    B,
    V,
    D,
    BMax,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_biasv,
    stride_vb,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLITS: tl.constexpr,
    EVEN_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_MEAN_LOGIT: tl.constexpr,
    HAS_LA: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_split = tl.program_id(1).to(tl.int64)

    direct_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    offs_b = direct_offs_b
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    running_lse = tl.full((BLOCK_B,), -float("inf"), tl.float32)
    if HAS_MEAN_LOGIT:
        running_mean_logit = tl.zeros((BLOCK_B,), tl.float32)

    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_split_chunks = tl.cdiv(num_v_chunks, SPLITS)
    for split_chunk in range(0, num_split_chunks):
        pid_v = pid_split + split_chunk * SPLITS
        if pid_v < num_v_chunks:
            offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
            e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
            c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

            accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
            for d in range(0, tl.cdiv(D, BLOCK_D)):
                e_mask = offs_b[:, None] < BMax
                c_mask = offs_v[None, :] < V
                if not EVEN_D:
                    remaining_d = D - d * BLOCK_D
                    e_mask = e_mask & (offs_d[None, :] < remaining_d)
                    c_mask = c_mask & (offs_d[:, None] < remaining_d)

                e = tl.load(e_ptrs, mask=e_mask, other=0.0)
                c = tl.load(c_ptrs, mask=c_mask, other=0.0)
                accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)
                e_ptrs += BLOCK_D * stride_ed
                c_ptrs += BLOCK_D * stride_cd

            accum = accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
            if HAS_BIAS:
                bias = tl.load(Bias + offs_v * stride_biasv, mask=offs_v < V, other=0.0)
                accum += bias[None, :]

            logits = tl.where(offs_v[None, :] < V, accum, -float("inf"))
            if HAS_SOFTCAP:
                logits = tl_softcapping(logits, softcap)
            logits = logits.cast(tl.float32)

            if HAS_LA:
                valid_rows = direct_offs_b[:, None] < B
                avg = tl.sum(tl.where(valid_rows, logits, 0.0), 0) / B
                tl.atomic_add(LA + offs_v, avg, mask=offs_v < V)

            if HAS_TARGETS:
                target_offs_b = offs_b + shift if HAS_SHIFT else offs_b
                targets = tl.load(
                    Targets + target_offs_b,
                    mask=target_offs_b < BMax,
                    other=V + 1,
                )
                out_ptrs = tl.broadcast_to(
                    (NegCorrectLogit + direct_offs_b)[:, None], (BLOCK_B, BLOCK_V)
                )
                tl.store(out_ptrs, -logits, mask=targets[:, None] == offs_v[None, :])

            tile_max = tl.max(logits, axis=1)
            tile_exp = tl.exp(logits - tile_max[:, None])
            tile_exp_sum = tl.sum(tile_exp, axis=1)
            tile_lse = tile_max + tl.log(tile_exp_sum)
            new_lse = tl_logaddexp(running_lse, tile_lse)

            if HAS_MEAN_LOGIT:
                finite_logits = tl.where(offs_v[None, :] < V, logits, 0.0)
                tile_mean_logit = tl.sum(tile_exp * finite_logits, axis=1) / tile_exp_sum
                running_mean_logit = (
                    tl.exp(running_lse - new_lse) * running_mean_logit
                    + tl.exp(tile_lse - new_lse) * tile_mean_logit
                )
            running_lse = new_lse

    partial_offsets = pid_split * B + direct_offs_b
    output_mask = direct_offs_b < B
    tl.store(PartialLSE + partial_offsets, running_lse, mask=output_mask)
    if HAS_MEAN_LOGIT:
        tl.store(PartialMeanLogit + partial_offsets, running_mean_logit, mask=output_mask)


@triton.jit
def _cce_lse_split_reduce_kernel(
    PartialLSE,
    PartialMeanLogit,
    LSE,
    MeanLogit,
    B,
    BLOCK_B: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_MEAN_LOGIT: tl.constexpr,
):
    offs_b = (tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    mask = offs_b < B
    running_lse = tl.full((BLOCK_B,), -float("inf"), tl.float32)
    if HAS_MEAN_LOGIT:
        running_mean_logit = tl.zeros((BLOCK_B,), tl.float32)

    for split in range(0, SPLITS):
        partial_offsets = split * B + offs_b
        partial_lse = tl.load(PartialLSE + partial_offsets, mask=mask, other=-float("inf"))
        new_lse = tl_logaddexp(running_lse, partial_lse)
        if HAS_MEAN_LOGIT:
            partial_mean_logit = tl.load(
                PartialMeanLogit + partial_offsets, mask=mask, other=0.0
            )
            running_mean_logit = (
                tl.exp(running_lse - new_lse) * running_mean_logit
                + tl.exp(partial_lse - new_lse) * partial_mean_logit
            )
        running_lse = new_lse

    tl.store(LSE + offs_b, running_lse, mask=mask)
    if HAS_MEAN_LOGIT:
        tl.store(MeanLogit + offs_b, running_mean_logit, mask=mask)


_cce_lse_split_partials_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_MEAN_LOGIT": lambda args: args["PartialMeanLogit"] is not None,
        "HAS_LA": lambda args: args["LA"] is not None,
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "HAS_SHIFT": lambda args: args["shift"] != 0,
        "DOT_PRECISION": lambda args: "ieee"
        if args["PartialMeanLogit"] is not None
        else ("tf32" if torch.get_float32_matmul_precision() == "high" else "ieee"),
    }
)(_cce_lse_split_partials_kernel)


def split_count(b: int, v: int, block_b: int, block_v: int, device: torch.device) -> int:
    """Choose enough independent vocab reductions to occupy the GPU, capped for bounded state."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    b_tiles = triton.cdiv(b, block_b)
    v_tiles = triton.cdiv(v, block_v)
    desired = max(1, triton.cdiv(2 * sms, b_tiles))
    return min(v_tiles, 16, triton.next_power_of_2(desired))


def use_split_reduction(
    e: torch.Tensor, c: torch.Tensor, b: int, return_mean_logit: bool
) -> bool:
    """Conservative selector for regimes where the staged reduction wins."""
    if b == 0 or return_mean_logit or e.dtype == torch.float32:
        return False
    block_b = 128
    block_v = 128
    if split_count(b, c.size(0), block_b, block_v, e.device) <= 1:
        return False
    # Only the small/medium-token regime showed a repeated >=8% forward gain.
    # Larger B remains available through the explicit override, but does not
    # justify an extra launch/workspace in the automatic policy.
    return b <= 512


def cce_lse_forward_split(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    valids: torch.Tensor | None,
    softcap: float | None,
    targets: torch.Tensor | None,
    shift: int,
    return_logit_avg: bool,
    return_mean_logit: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if valids is not None:
        b = valids.numel()
    else:
        b = e.size(0)
    v, d = c.shape

    if e.dtype == torch.float32:
        block_b, block_v, block_d, num_warps, num_stages = 32, 128, 32, 4, 3
    else:
        block_b, block_v, block_d, num_warps, num_stages = 128, 128, 32, 4, 4
    splits = split_count(b, v, block_b, block_v, e.device)

    partial_lse = e.new_empty((splits, b), dtype=torch.float32)
    partial_mean = (
        e.new_empty((splits, b), dtype=torch.float32) if return_mean_logit else None
    )
    lse = e.new_empty((b,), dtype=torch.float32)
    mean_logit = e.new_empty((b,), dtype=torch.float32) if return_mean_logit else None
    logit_avg = e.new_zeros((v,), dtype=torch.float32) if return_logit_avg else None
    neg_correct_logit = e.new_zeros((b,), dtype=torch.float32) if targets is not None else None

    _cce_lse_split_partials_kernel[(triton.cdiv(b, block_b), splits)](
        e,
        c,
        bias,
        partial_lse,
        partial_mean,
        logit_avg,
        neg_correct_logit,
        valids,
        targets,
        softcap,
        shift,
        b,
        v,
        d,
        e.size(0),
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if bias is None else bias.stride(0),
        1 if valids is None else valids.stride(0),
        BLOCK_B=block_b,
        BLOCK_V=block_v,
        BLOCK_D=block_d,
        SPLITS=splits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    reduce_block_b = 256
    _cce_lse_split_reduce_kernel[(triton.cdiv(b, reduce_block_b),)](
        partial_lse,
        partial_mean,
        lse,
        mean_logit,
        b,
        BLOCK_B=reduce_block_b,
        SPLITS=splits,
        HAS_MEAN_LOGIT=return_mean_logit,
        num_warps=4,
    )
    return lse, logit_avg, neg_correct_logit, mean_logit
