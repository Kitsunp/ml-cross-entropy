# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from cut_cross_entropy.cce_utils import normalize_return_topk
from cut_cross_entropy.tl_autotune import CCE_LOCK_BLOCK_B, cce_forward_autotune
from cut_cross_entropy.tl_utils import b_bin_fn, tl_logaddexp, tl_softcapping


@triton.jit
def _topk_keys(
    logits,
    ids,
    valid,
    SCORE_DTYPE: tl.constexpr = tl.float32,
    PACKED_GLOBAL: tl.constexpr = False,
    MAX_K: tl.constexpr = 5,
):
    # Lexicographic score-descending/ID-ascending order, including signed-zero
    # ties. Reuse the actual projected/rounded CCE logits, not another GEMM.
    # Keep the full tile at 32 bits. Widen only winners when the global
    # score/ID order cannot also be represented losslessly in 32 bits.
    columns = tl.arange(0, logits.shape[1])
    vocab_offset = tl.min(ids, 1)
    tl.static_assert(not PACKED_GLOBAL or SCORE_DTYPE == tl.bfloat16 or SCORE_DTYPE == tl.float16)
    if (SCORE_DTYPE == tl.bfloat16 or SCORE_DTYPE == tl.float16) and logits.shape[1] <= 65536:
        # The *actual* logits have 16 significant storage bits. Pack their
        # monotone score and local tie-break ID into one scalar reduction;
        # unlike quantizing FP32, this loses no information. Widen only the
        # requested winners when a global score/ID key needs 64 bits.
        bits16 = logits.to(SCORE_DTYPE).to(tl.uint16, bitcast=True).to(tl.uint32)
        bits16 = tl.where(logits == 0.0, 0, bits16)
        ordered16 = tl.where((bits16 & 0x8000) != 0, (~bits16) & 0xFFFF, bits16 ^ 0x8000)
        packed = (ordered16 << 16) | (0xFFFF - columns[None, :])
        packed = tl.where(valid, packed, 0)
        packed = tl.where(
            valid & ~(tl.abs(logits) < float("inf")),
            0xFFFF0000 | (0xFFFF - columns[None, :]).to(tl.uint32),
            packed,
        )
        bad_key: tl.constexpr = 0xFFFFFFFF if PACKED_GLOBAL else 0xFFFFFFFFFFFFFFFF
        winners = ()
        for _ in tl.static_range(MAX_K):
            winner = tl.max(packed, 1)
            column = 0xFFFF - (winner & 0xFFFF)
            score = winner >> 16
            if PACKED_GLOBAL:
                key = (score << 16) | (0xFFFF - (vocab_offset + column).to(tl.uint32))
            else:
                score_bits = tl.where((score & 0x8000) != 0, score ^ 0x8000, (~score) & 0xFFFF)
                value = score_bits.to(tl.uint16).to(SCORE_DTYPE, bitcast=True).to(tl.float32)
                bits = value.to(tl.uint32, bitcast=True)
                ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
                key = (ordered.to(tl.uint64) << 32) | (
                    0xFFFFFFFF - (vocab_offset + column).to(tl.uint64)
                )
            key = tl.where(winner == 0, 0, key)
            key = tl.where(score == 0xFFFF, bad_key, key)
            winners += (key,)
            packed = tl.where(packed == winner[:, None], 0, packed)
        return winners

    else:
        bits = tl.where(logits == 0.0, 0, logits.to(tl.uint32, bitcast=True))
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        ordered = tl.where(valid, ordered, 0)
        ordered = tl.where(valid & ~(tl.abs(logits) < float("inf")), 0xFFFFFFFF, ordered)
        winners = ()
        for _ in tl.static_range(MAX_K):
            value, column = tl.max(
                ordered, 1, return_indices=True, return_indices_tie_break_left=True
            )
            key = (value.to(tl.uint64) << 32) | (0xFFFFFFFF - (vocab_offset + column).to(tl.uint64))
            key = tl.where(value == 0, 0, key)
            key = tl.where(value == 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF, key)
            winners += (key,)
            ordered = tl.where(columns[None, :] == column[:, None], 0, ordered)
        return winners


@triton.jit
def _merge_topk(previous, incoming):
    # Balanced order statistics for sorted inputs, specialized to 3 or 5.
    a, b, c = previous[0], previous[1], previous[2]
    u, v, w = incoming[0], incoming[1], incoming[2]
    result = (
        tl.maximum(a, u),
        tl.maximum(tl.maximum(b, v), tl.minimum(a, u)),
        tl.maximum(tl.maximum(c, w), tl.maximum(tl.minimum(a, v), tl.minimum(b, u))),
    )
    if len(previous) == 5:
        d, e = previous[3], previous[4]
        x, y = incoming[3], incoming[4]
        result += (
            tl.maximum(
                tl.maximum(d, x),
                tl.maximum(tl.maximum(tl.minimum(a, w), tl.minimum(b, v)), tl.minimum(c, u)),
            ),
            tl.maximum(
                tl.maximum(e, y),
                tl.maximum(
                    tl.maximum(tl.minimum(a, x), tl.minimum(b, w)),
                    tl.maximum(tl.minimum(c, v), tl.minimum(d, u)),
                ),
            ),
        )
    return result


@triton.jit
def _topk_counts_kernel(
    Keys,
    Targets,
    Valids,
    Out,
    B,
    BMax,
    V,
    shift: tl.constexpr,
    STRIDE_T: tl.constexpr,
    STRIDE_K: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    original = tl.load(Valids + rows, rows < B, BMax) if HAS_VALIDS else rows
    slot = tl.arange(0, BK)
    y = tl.load(
        Targets + (original + shift)[:, None] * STRIDE_T + slot[None, :] * STRIDE_K,
        (rows[:, None] < B) & (original[:, None] + shift < BMax) & (slot[None, :] < K),
        V,
    )
    valid = (rows[:, None] < B) & (y >= 0) & (y < V)
    id_mask: tl.constexpr = 0xFFFF if Keys.dtype.element_ty == tl.uint32 else 0xFFFFFFFF
    bad_key: tl.constexpr = 0xFFFFFFFF if Keys.dtype.element_ty == tl.uint32 else 0xFFFFFFFFFFFFFFFF
    hit = tl.full((BLOCK, BK), False, tl.int1)
    for rank in tl.static_range(TOPK_K):
        key = tl.load(Keys + rows * TOPK_K + rank, rows < B, 0)
        hit |= ((id_mask - (key & id_mask)).to(tl.int64)[:, None] == y) & (key[:, None] != 0)
        if rank == 0:
            tl.store(Out + tl.program_id(0) * 5, tl.sum(tl.sum((hit & valid).to(tl.float32), 1), 0))
            tl.store(
                Out + tl.program_id(0) * 5 + 4,
                tl.sum(tl.sum(((key[:, None] == bad_key) & valid).to(tl.float32), 1), 0),
            )
        if rank == 2:
            tl.store(
                Out + tl.program_id(0) * 5 + 1, tl.sum(tl.sum((hit & valid).to(tl.float32), 1), 0)
            )
        if rank == 4:
            tl.store(
                Out + tl.program_id(0) * 5 + 2, tl.sum(tl.sum((hit & valid).to(tl.float32), 1), 0)
            )
    if TOPK_K == 3:
        tl.store(Out + tl.program_id(0) * 5 + 2, 0.0)
    tl.store(Out + tl.program_id(0) * 5 + 3, tl.sum(tl.sum(valid.to(tl.float32), 1), 0))


def _topk_key_dtype(
    e: torch.Tensor, c: torch.Tensor, bias: torch.Tensor | None, softcap: float | None
) -> torch.dtype:
    # Conservative proof of the *post-bias* score dtype, without executing a
    # tensor operation or assuming E alone implies low-precision logits.
    if (
        e.dtype in (torch.float16, torch.bfloat16)
        and c.size(0) <= 65536
        and softcap is None
        and (bias is None or bias.dtype == e.dtype)
    ):
        return torch.uint32
    return torch.uint64


def _topk_counts(keys, targets, valids, shift, bmax, vocab_size):
    b = keys.size(0)
    if not b:
        return keys.new_zeros(5, dtype=torch.float32)
    partials = keys.new_empty((triton.cdiv(b, 256), 5), dtype=torch.float32)
    k = targets.size(1) if targets.ndim == 2 else 1
    _topk_counts_kernel[(triton.cdiv(b, 256),)](
        keys,
        targets,
        valids,
        partials,
        b,
        bmax,
        vocab_size,
        shift,
        targets.stride(0),
        targets.stride(1) if targets.ndim == 2 else 1,
        k,
        triton.next_power_of_2(k),
        valids is not None,
        256,
        keys.size(1),
    )
    return partials.sum(0)


@triton.jit
def _sum_pair(left_sum, left_weighted_sum, right_sum, right_weighted_sum):
    return left_sum + right_sum, left_weighted_sum + right_weighted_sum


@triton.jit
def _valid_embedding_mean_kernel(
    E,
    Valids,
    MeanE,
    B,
    BMax,
    D,
    stride_eb,
    stride_ed,
    stride_vb,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
):
    offs_d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_b = tl.arange(0, BLOCK_B).to(tl.int64)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for block_b in range(0, tl.cdiv(B, BLOCK_B)):
        direct_rows = block_b * BLOCK_B + offs_b
        if HAS_VALIDS:
            rows = tl.load(
                Valids + direct_rows * stride_vb,
                mask=direct_rows < B,
                other=BMax,
            ).to(tl.int64)
        else:
            rows = direct_rows
        values = tl.load(
            E + rows[:, None] * stride_eb + offs_d[None, :] * stride_ed,
            mask=(direct_rows[:, None] < B) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(values, axis=0)
    tl.store(MeanE + offs_d, accumulator / B, mask=offs_d < D)


@triton.jit
def _linear_logit_avg_kernel(
    MeanE,
    C,
    Bias,
    LogitAvg,
    V,
    D,
    stride_cv,
    stride_cd,
    stride_biasv,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    offs_v = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_d = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for block_d in range(0, tl.cdiv(D, BLOCK_D)):
        cols = block_d * BLOCK_D + offs_d
        mean_e = tl.load(MeanE + cols, mask=cols < D, other=0.0)
        c = tl.load(
            C + offs_v[:, None] * stride_cv + cols[None, :] * stride_cd,
            mask=(offs_v[:, None] < V) & (cols[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(c * mean_e[None, :], axis=1)
    if HAS_BIAS:
        accumulator += tl.load(Bias + offs_v * stride_biasv, mask=offs_v < V)
    tl.store(LogitAvg + offs_v, accumulator, mask=offs_v < V)


def _linear_logit_avg(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    valids: torch.Tensor | None,
) -> torch.Tensor:
    b = e.size(0) if valids is None else valids.numel()
    d = e.size(1)
    v = c.size(0)
    mean_e = e.new_empty((d,), dtype=torch.float32)
    _valid_embedding_mean_kernel[(triton.cdiv(d, 32),)](
        e,
        valids,
        mean_e,
        b,
        e.size(0),
        d,
        e.stride(0),
        e.stride(1),
        1 if valids is None else valids.stride(0),
        BLOCK_B=128,
        BLOCK_D=32,
        HAS_VALIDS=valids is not None,
        num_warps=4,
    )
    logit_avg = e.new_empty((v,), dtype=torch.float32)
    _linear_logit_avg_kernel[(triton.cdiv(v, 128),)](
        mean_e,
        c,
        bias,
        logit_avg,
        v,
        d,
        c.stride(0),
        c.stride(1),
        1 if bias is None else bias.stride(0),
        BLOCK_V=128,
        BLOCK_D=32,
        HAS_BIAS=bias is not None,
        num_warps=4,
        num_stages=3,
    )
    return logit_avg


@triton.jit
def _neg_correct_logit_kernel(
    E,
    C,
    Bias,
    Valids,
    Targets,
    Out,
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
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    direct_rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    if HAS_VALIDS:
        rows = tl.load(
            Valids + direct_rows * stride_vb,
            mask=direct_rows < B,
            other=BMax,
        ).to(tl.int64)
    else:
        rows = direct_rows.to(tl.int64)
    target_rows = rows + shift if HAS_SHIFT else rows
    targets = tl.load(
        Targets + target_rows,
        mask=(direct_rows < B) & (target_rows < BMax),
        other=V,
    ).to(tl.int64)
    valid_target = (direct_rows < B) & (targets >= 0) & (targets < V)
    offs_d = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_B, BLOCK_B), dtype=tl.float32)
    for block_d in range(0, tl.cdiv(D, BLOCK_D)):
        cols = block_d * BLOCK_D + offs_d
        e = tl.load(
            E + rows[:, None] * stride_eb + cols[None, :] * stride_ed,
            mask=(direct_rows[:, None] < B) & (cols[None, :] < D),
            other=0.0,
        )
        c = tl.load(
            C + cols[:, None] * stride_cd + targets[None, :] * stride_cv,
            mask=(cols[:, None] < D) & valid_target[None, :],
            other=0.0,
        )
        accumulator = tl.dot(e, c, accumulator, input_precision=DOT_PRECISION)
    diagonal = tl.arange(0, BLOCK_B)[:, None] == tl.arange(0, BLOCK_B)[None, :]
    logit = tl.sum(tl.where(diagonal, accumulator, 0.0), axis=1)
    logit = logit.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
    if HAS_BIAS:
        logit += tl.load(Bias + targets * stride_biasv, mask=valid_target, other=0.0)
    if HAS_SOFTCAP:
        logit = tl_softcapping(logit, softcap)
    tl.store(Out + direct_rows, -logit.to(tl.float32), mask=direct_rows < B)


def _neg_correct_logit(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    valids: torch.Tensor | None,
    targets: torch.Tensor,
    softcap: float | None,
    shift: int,
    dot_precision: str,
) -> torch.Tensor:
    b = e.size(0) if valids is None else valids.numel()
    out = e.new_empty((b,), dtype=torch.float32)
    _neg_correct_logit_kernel[(triton.cdiv(b, 16),)](
        e,
        c,
        bias,
        valids,
        targets,
        out,
        softcap,
        shift,
        b,
        c.size(0),
        e.size(1),
        e.size(0),
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if bias is None else bias.stride(0),
        1 if valids is None else valids.stride(0),
        BLOCK_B=16,
        BLOCK_D=32,
        HAS_BIAS=bias is not None,
        HAS_VALIDS=valids is not None,
        HAS_SOFTCAP=softcap is not None,
        HAS_SHIFT=shift != 0,
        DOT_PRECISION=dot_precision,
        num_warps=4,
        num_stages=3,
    )
    return out


def _cce_lse_forward_kernel(
    E,
    C,
    Bias,
    LSE,
    MeanLogit,
    LA,
    NegCorrectLogit,
    TopK,
    Locks,
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
    stride_tb,
    stride_tk,
    MODE,
    TOPK_K: tl.constexpr,
    # Meta-parameters
    B_BIN,
    LOCK_BLOCK_B: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,  #
    GROUP_B: tl.constexpr,  #
    EVEN_D: tl.constexpr,
    EVEN_V: tl.constexpr,
    FULL_B: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_MEAN_LOGIT: tl.constexpr,
    HAS_LA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    PATCH_SIZE: tl.constexpr,
    HAS_TOPK: tl.constexpr,
    VOCAB_OFFSET: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_pid_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_pid_in_group) // group_size_b).to(tl.int64)

    direct_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    offs_b = direct_offs_b
    if HAS_VALIDS:
        if FULL_B:
            offs_b = tl.load(Valids + stride_vb * offs_b).to(tl.int64)
        else:
            offs_b = tl.load(
                Valids + stride_vb * offs_b,
                mask=offs_b < B,
                other=BMax,
            ).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = True if (HAS_VALIDS and FULL_B) else offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)
        c_mask = True
        if not EVEN_V:
            c_mask = offs_v[None, :] < V
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
                mask=offs_v < V,
                other=0.0,
            )
        accum += bias[None, :]

    logits = accum
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)
    score_dtype: tl.constexpr = logits.dtype
    if not EVEN_V:
        logits = tl.where(offs_v[None, :] < V, logits, -float("inf"))
    logits = logits.cast(tl.float32)
    if HAS_LA:
        valid_rows = direct_offs_b[:, None] < B
        this_avg_logit = tl.sum(tl.where(valid_rows, logits, 0.0), 0) / B
        tl.atomic_add(
            LA + offs_v,
            this_avg_logit,
            mask=None if EVEN_V else offs_v < V,
        )

    if HAS_TARGETS:
        target_offs_b = offs_b + shift if HAS_SHIFT else offs_b
        for patch_slot in tl.static_range(0, PATCH_SIZE):
            this_targets = tl.load(
                Targets + target_offs_b * stride_tb + patch_slot * stride_tk,
                mask=target_offs_b < BMax,
                other=V + 1,
            )
            neg_correct_logit_ptrs = tl.broadcast_to(
                (NegCorrectLogit + direct_offs_b * PATCH_SIZE + patch_slot)[:, None],
                (BLOCK_B, BLOCK_V),
            )
            target_match = this_targets[:, None] == offs_v[None, :]
            if not EVEN_V:
                # Invalid positive targets can only match padded columns in the
                # final tile. Reuse its existing vocab mask instead of adding
                # two target-range comparisons to every hot tile.
                target_match &= offs_v[None, :] < V
            tl.store(neg_correct_logit_ptrs, -logits, mask=target_match)

    offs_b = direct_offs_b
    this_mx = tl.max(logits, axis=1)
    centered_logits = logits - this_mx[:, None]
    this_exp = tl.exp(centered_logits)
    if HAS_MEAN_LOGIT:
        finite_centered_logits = tl.where(offs_v[None, :] < V, centered_logits, 0.0)
        this_exp_sum, this_weighted_sum = tl.reduce(
            (this_exp, this_exp * finite_centered_logits),
            axis=1,
            combine_fn=_sum_pair,
        )
        this_mean_logit = this_mx + this_weighted_sum / this_exp_sum
    else:
        this_exp_sum = tl.sum(this_exp, axis=1)
    this_lse = this_mx + tl.log(this_exp_sum)
    if HAS_TOPK:
        tile_topk = _topk_keys(
            logits,
            offs_v[None, :] + VOCAB_OFFSET,
            offs_v[None, :] < V,
            score_dtype,
            TopK.dtype.element_ty == tl.uint32,
            TOPK_K,
        )

    o_mask = None if FULL_B else offs_b < B

    lse_ptrs = LSE + offs_b
    mean_logit_ptrs = MeanLogit + offs_b if HAS_MEAN_LOGIT else None

    this_locks = Locks + (pid_b * BLOCK_B // LOCK_BLOCK_B)
    while tl.atomic_cas(this_locks, 0, 1, sem="acquire", scope="gpu") == 1:
        pass

    if FULL_B:
        old_lse = tl.load(lse_ptrs, eviction_policy="evict_last")
    else:
        old_lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    if HAS_MEAN_LOGIT:
        if FULL_B:
            old_mean_logit = tl.load(mean_logit_ptrs, eviction_policy="evict_last")
        else:
            old_mean_logit = tl.load(
                mean_logit_ptrs,
                mask=o_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
        new_lse = tl_logaddexp(old_lse, this_lse)
        old_weight = tl.exp(old_lse - new_lse)
        this_weight = tl.exp(this_lse - new_lse)
        new_mean_logit = old_weight * old_mean_logit + this_weight * this_mean_logit
        tl.store(
            mean_logit_ptrs,
            new_mean_logit,
            mask=o_mask,
            eviction_policy="evict_last",
        )
    else:
        new_lse = tl_logaddexp(old_lse, this_lse)
    tl.store(lse_ptrs, new_lse, mask=o_mask, eviction_policy="evict_last")
    if HAS_TOPK:
        previous = ()
        for rank in tl.static_range(TOPK_K):
            previous += (tl.load(TopK + offs_b * TOPK_K + rank, mask=offs_b < B, other=0),)
        merged = _merge_topk(previous, tile_topk)
        for rank in tl.static_range(TOPK_K):
            tl.store(TopK + offs_b * TOPK_K + rank, merged[rank], mask=o_mask)

    tl.atomic_xchg(this_locks, 0, sem="release", scope="gpu")


_cce_lse_forward_kernel = triton.jit(
    _cce_lse_forward_kernel,
    do_not_specialize=["MODE", "B_BIN"],
)
_cce_lse_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "EVEN_V": lambda args: args["V"] % args["BLOCK_V"] == 0,
        "FULL_B": lambda args: args["B"] % args["BLOCK_B"] == 0,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_MEAN_LOGIT": lambda args: args["MeanLogit"] is not None,
        "HAS_LA": lambda args: args["LA"] is not None,
        "GROUP_B": lambda args: 16,
        # MiLe's entropy statistic is sensitive to the weighted-logit moment.
        # Keep the ordinary CE path on its configured fast precision, but use
        # IEEE products when that additional statistic is requested.
        "DOT_PRECISION": lambda args: (
            "ieee"
            if args["MeanLogit"] is not None
            else ("tf32" if torch.get_float32_matmul_precision() == "high" else "ieee")
        ),
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "HAS_TOPK": lambda args: args["TopK"] is not None,
        "HAS_SHIFT": lambda args: args["shift"] != 0,
    }
)(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = cce_forward_autotune()(_cce_lse_forward_kernel)  # type: ignore


@dataclass(slots=True)
class LSEReturn:
    lse: torch.Tensor
    logit_avg: torch.Tensor | None
    neg_correct_logit: torch.Tensor | None
    mean_logit: torch.Tensor | None
    topk_counts: torch.Tensor | None = None
    topk_keys: torch.Tensor | None = None


def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    targets: torch.Tensor | None = None,
    shift: int = 0,
    return_logit_avg: bool = False,
    return_mean_logit: bool = False,
    return_topk: bool | int = False,
    return_topk_keys: bool = False,
) -> LSEReturn:
    return_topk = normalize_return_topk(return_topk)
    if return_topk_keys and not return_topk:
        raise ValueError("return_topk_keys requires return_topk=3 or 5.")
    # Check constraints.
    assert e.shape[1] == c.shape[1], "Incompatible dimensions"
    assert e.is_contiguous(), "Matrix A must be contiguous"
    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B, _ = e.shape

    if bias is not None:
        assert bias.ndim == 1
        assert c.shape[0] == bias.shape[0]

    V, D = c.shape
    if return_topk and targets is None:
        raise ValueError("Top-k metrics require supervised targets.")
    if targets is not None:
        assert targets.ndim in (1, 2)
        patch_size = targets.size(1) if targets.ndim == 2 else 1
        target_stride_b = targets.stride(0)
        target_stride_k = targets.stride(1) if targets.ndim == 2 else 1
    else:
        patch_size = 1
        target_stride_b = 1
        target_stride_k = 1

    if B == 0:
        empty = e.new_empty((0,), dtype=torch.float32)
        correct = (
            e.new_empty((0, patch_size) if targets.ndim == 2 else (0,), dtype=torch.float32)
            if targets is not None
            else None
        )
        return LSEReturn(
            empty,
            e.new_zeros(V, dtype=torch.float32) if return_logit_avg else None,
            correct,
            e.new_empty(0, dtype=torch.float32) if return_mean_logit else None,
            e.new_zeros(5, dtype=torch.float32) if return_topk and not return_topk_keys else None,
            e.new_empty((0, return_topk), dtype=_topk_key_dtype(e, c, bias, softcap))
            if return_topk_keys else None,
        )

    reduction = os.getenv("CCE_FORWARD_REDUCTION", "auto")
    if reduction not in {"auto", "lock", "split"}:
        raise ValueError("CCE_FORWARD_REDUCTION must be 'auto', 'lock', or 'split'")
    if reduction != "lock":
        from cut_cross_entropy.cce_lse_forward_split import (
            cce_lse_forward_split,
            use_split_reduction,
        )

    if B > 0 and (
        reduction == "split"
        or (reduction == "auto" and use_split_reduction(e, c, B, return_mean_logit))
    ):
        return LSEReturn(
            *cce_lse_forward_split(
                e,
                c,
                bias,
                valids,
                softcap,
                targets,
                shift,
                return_logit_avg,
                return_mean_logit,
                return_topk,
                return_topk_keys,
            )
        )

    # Allocates output.
    lse = e.new_full((B,), -torch.inf, dtype=torch.float32)
    topk = (
        e.new_zeros((B, return_topk), dtype=_topk_key_dtype(e, c, bias, softcap))
        if return_topk
        else None
    )
    mean_logit = e.new_zeros((B,), dtype=torch.float32) if return_mean_logit else None
    # The fixed K=32 single-target path reconstructs targets with a diagonal
    # tl.dot. Patch targets stay fused because every slot must observe the exact
    # logit tile used by LSE; an independent K-target reduction could round
    # differently in one-class or highly peaked distributions.
    use_indexed_target = (
        targets is not None and targets.ndim == 1 and os.getenv("CCE_AUTOTUNE", "0") == "0"
    )
    kernel_targets = None if use_indexed_target else targets
    kernel_shift = shift if kernel_targets is not None else 0
    kernel_target_shape = (B, patch_size) if targets is not None and targets.ndim == 2 else (B,)
    kernel_neg_correct_logit = (
        e.new_zeros(kernel_target_shape, dtype=torch.float32)
        if kernel_targets is not None
        else None
    )
    assert lse.stride(0) == 1

    locks = e.new_full(
        (triton.cdiv(B, CCE_LOCK_BLOCK_B),),
        0,
        dtype=torch.uint32,
    )

    use_linear_logit_avg = return_logit_avg and softcap is None
    kernel_logit_avg = (
        e.new_full((V,), 0.0, dtype=torch.float32)
        if return_logit_avg and not use_linear_logit_avg
        else None
    )

    def launch_vocab_slice(
        c_slice: torch.Tensor,
        bias_slice: torch.Tensor | None,
        logit_avg_slice: torch.Tensor | None,
        slice_b: int,
        valids_slice: torch.Tensor | None,
        lse_slice: torch.Tensor,
        mean_logit_slice: torch.Tensor | None,
        locks_slice: torch.Tensor,
        vocab_offset: int,
    ) -> None:
        slice_v = c_slice.size(0)

        def grid(META) -> tuple[int]:
            return (triton.cdiv(slice_b, META["BLOCK_B"]) * triton.cdiv(slice_v, META["BLOCK_V"]),)

        _cce_lse_forward_kernel[grid](
            e,
            c_slice,
            bias_slice,
            lse_slice,
            mean_logit_slice,
            logit_avg_slice,
            kernel_neg_correct_logit,
            topk,
            locks_slice,
            valids_slice,
            kernel_targets,
            softcap,
            kernel_shift,
            slice_b,
            slice_v,
            D,  #
            e.size(0),
            e.stride(0),
            e.stride(1),  #
            c_slice.stride(0),
            c_slice.stride(1),  #
            1 if bias_slice is None else bias_slice.stride(0),
            1 if valids_slice is None else valids_slice.stride(0),
            target_stride_b,
            target_stride_k,
            # Normalize optional features into cost families. Individual constexpr
            # branches still compile independently, but equivalent modes share one
            # autotune result instead of fragmenting the timing cache.
            MODE=(
                (bias_slice is not None or valids_slice is not None)
                | ((softcap is not None) << 1)
                | (return_mean_logit << 2)
                | ((logit_avg_slice is not None) << 3)
                | ((kernel_targets is not None) << 4)
                | (bool(return_topk) << 5)
            ),
            B_BIN=b_bin_fn(slice_b),
            TOPK_K=return_topk,
            LOCK_BLOCK_B=CCE_LOCK_BLOCK_B,
            PATCH_SIZE=patch_size,
            VOCAB_OFFSET=vocab_offset,
        )

    fixed_tile_v = 128
    main_v = (
        V - (V % fixed_tile_v)
        if kernel_targets is None and os.getenv("CCE_AUTOTUNE", "0") == "0"
        else V
    )
    vocab_slices = [(0, main_v)] if main_v > 0 else []
    if main_v < V:
        vocab_slices.append((main_v, V))

    for vocab_start, vocab_stop in vocab_slices:
        launch_vocab_slice(
            c[vocab_start:vocab_stop],
            None if bias is None else bias[vocab_start:vocab_stop],
            (None if kernel_logit_avg is None else kernel_logit_avg[vocab_start:vocab_stop]),
            B,
            valids,
            lse,
            mean_logit,
            locks,
            vocab_start,
        )

    logit_avg = _linear_logit_avg(e, c, bias, valids) if use_linear_logit_avg else kernel_logit_avg
    neg_correct_logit = (
        _neg_correct_logit(
            e,
            c,
            bias,
            valids,
            targets,
            softcap,
            shift,
            "ieee"
            if return_mean_logit
            else ("tf32" if torch.get_float32_matmul_precision() == "high" else "ieee"),
        )
        if use_indexed_target
        else kernel_neg_correct_logit
    )
    return LSEReturn(
        lse,
        logit_avg,
        neg_correct_logit,
        mean_logit,
        _topk_counts(topk, targets, valids, shift, e.size(0), V)
        if return_topk and not return_topk_keys else None,
        topk if return_topk_keys else None,
    )
