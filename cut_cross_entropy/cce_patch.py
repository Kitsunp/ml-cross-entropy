# Copyright (C) 2026. All Rights Reserved.
"""Patch-level target kernels for Cut Cross Entropy.

The patch path keeps one embedding row per patch.  It computes one vocabulary
log-sum-exp for that row and only K indexed correct-class logits, avoiding a
K-fold repeat of the embedding and dense vocabulary reduction.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _patch_fraction_rows(
    LSE,
    MeanLogit,
    NegCorrectLogit,
    Targets,
    Fraction,
    B,
    V,
    stride_nb,
    stride_nk,
    stride_tb,
    stride_tk,
    PATCH_SIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_ENTROPY: tl.constexpr,
    GAMMA: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    slots = tl.arange(0, BLOCK_K)
    targets = tl.load(
        Targets + rows[:, None] * stride_tb + slots[None, :] * stride_tk,
        mask=(rows[:, None] < B) & (slots[None, :] < PATCH_SIZE),
        other=V,
    ).to(tl.int64)
    valid = (rows[:, None] < B) & (slots[None, :] < PATCH_SIZE)
    valid &= (targets >= 0) & (targets < V)
    counts = tl.sum(valid.to(tl.float32), axis=1)
    active = counts > 0
    lse = tl.load(LSE + rows, mask=(rows < B) & active, other=0).to(tl.float32)
    correct = tl.load(
        NegCorrectLogit + rows[:, None] * stride_nb + slots[None, :] * stride_nk,
        mask=valid,
        other=0,
    ).to(tl.float32)
    nll = tl.sum(tl.where(valid, lse[:, None] + correct, 0.0), axis=1)
    if GAMMA == 0.0:
        log_base = tl.full((BLOCK_B,), 0.0, tl.float32)
    else:
        mean = tl.load(MeanLogit + rows, mask=(rows < B) & active, other=0).to(tl.float32)
        entropy = tl.minimum(tl.maximum(lse - mean, 0.0), MAX_ENTROPY)
        log_base = tl.log(1.0 + entropy)
    fraction = tl.load(Fraction + rows, mask=(rows < B) & active, other=0).to(tl.float32)
    # Invalid active metadata poisons the objective instead of silently changing
    # its meaning. Excluded rows are neutral even if their storage contains NaN.
    fraction = tl.where(
        active & ~((fraction >= 0.0) & (fraction <= 1.0)),
        float("nan"),
        fraction,
    )
    return rows, counts, nll, log_base, fraction


@triton.jit
def _patch_fraction_components(
    log_base, counts, fraction, max_clean, max_corrupt, GAMMA: tl.constexpr
):
    clean = tl.where(counts > 0, 1.0 - fraction, 0.0)
    corrupt = tl.where(counts > 0, fraction, 0.0)
    # Subtract each group's own maximum BEFORE multiplying by gamma. A common
    # maximum can underflow every member of the lower-entropy group.
    clean_log = tl.where(clean > 0, log_base - max_clean, 0.0)
    corrupt_log = tl.where(corrupt > 0, log_base - max_corrupt, 0.0)
    clean_component = clean * tl.exp(clean_log * GAMMA)
    corrupt_component = corrupt * tl.exp(corrupt_log * GAMMA)
    return clean, corrupt, clean_component, corrupt_component


@triton.jit
def _patch_fraction_small_kernel(
    LSE,
    MeanLogit,
    NegCorrectLogit,
    Targets,
    Fraction,
    Objective,
    Unweighted,
    DenseWeight,
    TargetWeight,
    B,
    V,
    stride_nb,
    stride_nk,
    stride_tb,
    stride_tk,
    PATCH_SIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_ENTROPY: tl.constexpr,
    GAMMA: tl.constexpr,
    RETURN_UNWEIGHTED: tl.constexpr,
):
    rows, counts, nll, log_base, fraction = _patch_fraction_rows(
        LSE,
        MeanLogit,
        NegCorrectLogit,
        Targets,
        Fraction,
        B,
        V,
        stride_nb,
        stride_nk,
        stride_tb,
        stride_tk,
        PATCH_SIZE,
        BLOCK_B,
        BLOCK_K,
        MAX_ENTROPY,
        GAMMA,
    )
    max_clean = tl.max(tl.where((counts > 0) & (fraction < 1), log_base, 0.0), 0)
    max_corrupt = tl.max(tl.where((counts > 0) & (fraction > 0), log_base, 0.0), 0)
    clean, corrupt, part_clean, part_corrupt = _patch_fraction_components(
        log_base,
        counts,
        fraction,
        max_clean,
        max_corrupt,
        GAMMA,
    )
    n_clean = tl.sum(counts * clean, 0)
    n_corrupt = tl.sum(counts * corrupt, 0)
    a_clean = tl.sum(counts * part_clean, 0)
    a_corrupt = tl.sum(counts * part_corrupt, 0)
    scale_clean = n_clean / tl.where(a_clean > 0, a_clean, 1.0)
    scale_corrupt = n_corrupt / tl.where(a_corrupt > 0, a_corrupt, 1.0)
    count = tl.maximum(tl.sum(counts, 0), 1.0)
    weight = part_clean * scale_clean + part_corrupt * scale_corrupt
    target_weight = weight * (B / count)
    tl.store(TargetWeight + rows, target_weight, mask=rows < B)
    tl.store(DenseWeight + rows, target_weight * counts, mask=rows < B)
    tl.store(Objective, tl.sum(nll * weight, 0) / count)
    if RETURN_UNWEIGHTED:
        tl.store(Unweighted, tl.sum(nll, 0) / count)


@triton.jit
def _patch_fraction_prepare_kernel(
    LSE,
    MeanLogit,
    NegCorrectLogit,
    Targets,
    Fraction,
    RowData,
    Maxima,
    B,
    V,
    stride_nb,
    stride_nk,
    stride_tb,
    stride_tk,
    BLOCKS: tl.constexpr,
    PATCH_SIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_ENTROPY: tl.constexpr,
    GAMMA: tl.constexpr,
):
    rows, counts, nll, log_base, fraction = _patch_fraction_rows(
        LSE,
        MeanLogit,
        NegCorrectLogit,
        Targets,
        Fraction,
        B,
        V,
        stride_nb,
        stride_nk,
        stride_tb,
        stride_tk,
        PATCH_SIZE,
        BLOCK_B,
        BLOCK_K,
        MAX_ENTROPY,
        GAMMA,
    )
    tl.store(RowData + rows, counts, mask=rows < B)
    tl.store(RowData + B + rows, nll, mask=rows < B)
    tl.store(RowData + 2 * B + rows, log_base, mask=rows < B)
    tl.store(RowData + 3 * B + rows, fraction, mask=rows < B)
    max_clean = tl.max(tl.where((counts > 0) & (fraction < 1), log_base, 0.0), 0)
    max_corrupt = tl.max(tl.where((counts > 0) & (fraction > 0), log_base, 0.0), 0)
    tl.store(Maxima + tl.program_id(0), max_clean)
    tl.store(Maxima + BLOCKS + tl.program_id(0), max_corrupt)


@triton.jit
def _patch_fraction_partials_kernel(
    RowData,
    Maxima,
    Partials,
    B,
    BLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
    GAMMA: tl.constexpr,
):
    blocks = tl.arange(0, BLOCK_REDUCE)
    max_clean = tl.max(tl.load(Maxima + blocks, mask=blocks < BLOCKS, other=0), 0)
    max_corrupt = tl.max(tl.load(Maxima + BLOCKS + blocks, mask=blocks < BLOCKS, other=0), 0)
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    counts = tl.load(RowData + rows, mask=rows < B, other=0)
    nll = tl.load(RowData + B + rows, mask=rows < B, other=0)
    log_base = tl.load(RowData + 2 * B + rows, mask=rows < B, other=0)
    fraction = tl.load(RowData + 3 * B + rows, mask=rows < B, other=0)
    clean, corrupt, part_clean, part_corrupt = _patch_fraction_components(
        log_base,
        counts,
        fraction,
        max_clean,
        max_corrupt,
        GAMMA,
    )
    # Reuse the row workspace: downstream needs the two rescaled components,
    # not entropy or the original fraction. No vocabulary-sized temporaries.
    tl.store(RowData + 2 * B + rows, part_clean, mask=rows < B)
    tl.store(RowData + 3 * B + rows, part_corrupt, mask=rows < B)
    block = tl.program_id(0)
    tl.store(Partials + block, tl.sum(counts * clean, 0))
    tl.store(Partials + BLOCKS + block, tl.sum(counts * part_clean, 0))
    tl.store(Partials + 2 * BLOCKS + block, tl.sum(nll * part_clean, 0))
    tl.store(Partials + 3 * BLOCKS + block, tl.sum(counts * corrupt, 0))
    tl.store(Partials + 4 * BLOCKS + block, tl.sum(counts * part_corrupt, 0))
    tl.store(Partials + 5 * BLOCKS + block, tl.sum(nll * part_corrupt, 0))
    tl.store(Partials + 6 * BLOCKS + block, tl.sum(counts, 0))
    tl.store(Partials + 7 * BLOCKS + block, tl.sum(nll, 0))


@triton.jit
def _patch_fraction_finalize_kernel(
    Partials,
    Scales,
    Objective,
    Unweighted,
    BLOCKS: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
    RETURN_UNWEIGHTED: tl.constexpr,
):
    blocks = tl.arange(0, BLOCK_REDUCE)
    n_clean = tl.sum(tl.load(Partials + blocks, mask=blocks < BLOCKS, other=0), 0)
    a_clean = tl.sum(tl.load(Partials + BLOCKS + blocks, mask=blocks < BLOCKS, other=0), 0)
    q_clean = tl.sum(tl.load(Partials + 2 * BLOCKS + blocks, mask=blocks < BLOCKS, other=0), 0)
    n_corrupt = tl.sum(tl.load(Partials + 3 * BLOCKS + blocks, mask=blocks < BLOCKS, other=0), 0)
    a_corrupt = tl.sum(tl.load(Partials + 4 * BLOCKS + blocks, mask=blocks < BLOCKS, other=0), 0)
    q_corrupt = tl.sum(tl.load(Partials + 5 * BLOCKS + blocks, mask=blocks < BLOCKS, other=0), 0)
    count = tl.maximum(
        tl.sum(tl.load(Partials + 6 * BLOCKS + blocks, mask=blocks < BLOCKS, other=0), 0), 1.0
    )
    clean_scale = n_clean / tl.where(a_clean > 0, a_clean, 1.0) / count
    corrupt_scale = n_corrupt / tl.where(a_corrupt > 0, a_corrupt, 1.0) / count
    tl.store(Scales, clean_scale)
    tl.store(Scales + 1, corrupt_scale)
    tl.store(Objective, q_clean * clean_scale + q_corrupt * corrupt_scale)
    if RETURN_UNWEIGHTED:
        nll = tl.sum(tl.load(Partials + 7 * BLOCKS + blocks, mask=blocks < BLOCKS, other=0), 0)
        tl.store(Unweighted, nll / count)


@triton.jit
def _patch_fraction_weights_kernel(
    RowData, Scales, DenseWeight, TargetWeight, B, BLOCK_B: tl.constexpr
):
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    counts = tl.load(RowData + rows, mask=rows < B, other=0)
    clean = tl.load(RowData + 2 * B + rows, mask=rows < B, other=0)
    corrupt = tl.load(RowData + 3 * B + rows, mask=rows < B, other=0)
    weight = B * (clean * tl.load(Scales) + corrupt * tl.load(Scales + 1))
    tl.store(TargetWeight + rows, weight, mask=rows < B)
    tl.store(DenseWeight + rows, counts * weight, mask=rows < B)


def _patch_fraction_loss_forward(
    lse,
    mean_logit,
    neg_correct_logit,
    targets,
    vocab_size,
    mile_gamma,
    return_unweighted,
    fraction,
):
    rows, patch_size = targets.shape
    objective = lse.new_empty(())
    unweighted = lse.new_empty(()) if return_unweighted else objective
    dense_weight = torch.empty_like(lse)
    target_weight = torch.empty_like(lse)
    if rows == 0:
        objective.zero_()
        if return_unweighted:
            unweighted.zero_()
        return objective, unweighted, dense_weight, target_weight
    args = (
        lse,
        mean_logit,
        neg_correct_logit,
        targets,
        fraction,
    )
    strides = (
        rows,
        vocab_size,
        neg_correct_logit.stride(0),
        neg_correct_logit.stride(1),
        targets.stride(0),
        targets.stride(1),
    )
    opts = dict(
        PATCH_SIZE=patch_size,
        BLOCK_K=triton.next_power_of_2(patch_size),
        MAX_ENTROPY=math.log(vocab_size),
        GAMMA=mile_gamma,
        num_warps=4,
    )
    small_block = triton.next_power_of_2(rows)
    if small_block <= 128 and small_block * opts["BLOCK_K"] <= 1024:
        _patch_fraction_small_kernel[(1,)](
            *args,
            objective,
            unweighted,
            dense_weight,
            target_weight,
            *strides,
            BLOCK_B=small_block,
            RETURN_UNWEIGHTED=return_unweighted,
            **opts,
        )
    else:
        block_b = 128
        blocks = triton.cdiv(rows, block_b)
        row_data = lse.new_empty((4, rows))
        maxima = lse.new_empty((2, blocks))
        partials = lse.new_empty((8, blocks))
        scales = lse.new_empty((2,))
        _patch_fraction_prepare_kernel[(blocks,)](
            *args,
            row_data,
            maxima,
            *strides,
            BLOCKS=blocks,
            BLOCK_B=block_b,
            **opts,
        )
        _patch_fraction_partials_kernel[(blocks,)](
            row_data,
            maxima,
            partials,
            rows,
            BLOCKS=blocks,
            BLOCK_B=block_b,
            BLOCK_REDUCE=triton.next_power_of_2(blocks),
            GAMMA=mile_gamma,
            num_warps=4,
        )
        _patch_fraction_finalize_kernel[(1,)](
            partials,
            scales,
            objective,
            unweighted,
            BLOCKS=blocks,
            BLOCK_REDUCE=triton.next_power_of_2(blocks),
            RETURN_UNWEIGHTED=return_unweighted,
            num_warps=4,
        )
        _patch_fraction_weights_kernel[(blocks,)](
            row_data,
            scales,
            dense_weight,
            target_weight,
            rows,
            BLOCK_B=block_b,
            num_warps=4,
        )
    return objective, unweighted, dense_weight, target_weight


@triton.jit
def _patch_loss_small_kernel(
    LSE,
    MeanLogit,
    NegCorrectLogit,
    Targets,
    Objective,
    Unweighted,
    DenseWeight,
    TargetWeight,
    B,
    V,
    stride_nb,
    stride_nk,
    stride_tb,
    stride_tk,
    PATCH_SIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_MILE: tl.constexpr,
    MILE_GAMMA: tl.constexpr,
    RETURN_UNWEIGHTED: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_B)
    slots = tl.arange(0, BLOCK_K)
    row_mask = rows < B
    slot_mask = slots < PATCH_SIZE
    targets = tl.load(
        Targets + rows[:, None] * stride_tb + slots[None, :] * stride_tk,
        mask=row_mask[:, None] & slot_mask[None, :],
        other=V,
    ).to(tl.int64)
    valid = row_mask[:, None] & slot_mask[None, :] & (targets >= 0) & (targets < V)
    valid_f32 = valid.to(tl.float32)
    counts = tl.sum(valid_f32, axis=1)

    lse = tl.load(LSE + rows, mask=row_mask, other=0.0).to(tl.float32)
    neg_correct_logit = tl.load(
        NegCorrectLogit + rows[:, None] * stride_nb + slots[None, :] * stride_nk,
        mask=row_mask[:, None] & slot_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    row_nll = tl.sum((lse[:, None] + neg_correct_logit) * valid_f32, axis=1)

    if HAS_MILE:
        mean_logit = tl.load(MeanLogit + rows, mask=row_mask, other=0.0).to(tl.float32)
        entropy_base = 1.0 + tl.maximum(lse - mean_logit, 0.0)
        if MILE_GAMMA == 0.0:
            base_weight = tl.full((BLOCK_B,), 1.0, tl.float32)
        elif MILE_GAMMA == 1.0:
            base_weight = entropy_base
        else:
            base_weight = tl.exp(MILE_GAMMA * tl.log(entropy_base))
    else:
        base_weight = tl.full((BLOCK_B,), 1.0, tl.float32)
    base_weight = tl.where(row_mask, base_weight, 0.0)

    denominator = tl.maximum(tl.sum(base_weight * counts, axis=0), 1.0)
    weighted_nll = tl.sum(base_weight * row_nll, axis=0)
    target_weight = base_weight * (B / denominator)
    tl.store(TargetWeight + rows, target_weight, mask=row_mask)
    tl.store(DenseWeight + rows, target_weight * counts, mask=row_mask)

    tl.store(Objective, weighted_nll / denominator)
    if RETURN_UNWEIGHTED:
        total_valid = tl.maximum(tl.sum(counts, axis=0), 1.0)
        tl.store(Unweighted, tl.sum(row_nll, axis=0) / total_valid)


def patch_loss_forward(
    lse: torch.Tensor,
    mean_logit: torch.Tensor | None,
    neg_correct_logit: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    mile_gamma: float | None,
    return_unweighted: bool,
    mile_group_fraction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce patch targets and build detached backward row weights.

    The returned dense weight multiplies the softmax term once per row.  The
    target weight multiplies each sparse one-hot correction.  Scaling both by
    the static row count lets the existing backward retain its ``1 / B``
    reduction without reading a data-dependent valid-target count on the host.
    """
    if mile_group_fraction is not None:
        assert mile_gamma is not None and mean_logit is not None
        return _patch_fraction_loss_forward(
            lse,
            mean_logit,
            neg_correct_logit,
            targets,
            vocab_size,
            mile_gamma,
            return_unweighted,
            mile_group_fraction,
        )
    rows, patch_size = targets.shape
    block_b = triton.next_power_of_2(rows) if rows > 0 else 1
    block_k = triton.next_power_of_2(patch_size)
    if rows > 0 and block_b <= 128 and block_b * block_k <= 1024:
        objective = lse.new_empty(())
        unweighted = lse.new_empty(()) if return_unweighted else objective
        dense_weight = torch.empty_like(lse)
        target_weight = torch.empty_like(lse)
        _patch_loss_small_kernel[(1,)](
            lse,
            mean_logit,
            neg_correct_logit,
            targets,
            objective,
            unweighted,
            dense_weight,
            target_weight,
            rows,
            vocab_size,
            neg_correct_logit.stride(0),
            neg_correct_logit.stride(1),
            targets.stride(0),
            targets.stride(1),
            PATCH_SIZE=patch_size,
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            HAS_MILE=mile_gamma is not None,
            MILE_GAMMA=0.0 if mile_gamma is None else mile_gamma,
            RETURN_UNWEIGHTED=return_unweighted,
            num_warps=4,
        )
        return objective, unweighted, dense_weight, target_weight

    valid = (targets >= 0) & (targets < vocab_size)
    valid_f32 = valid.to(torch.float32)
    counts = valid_f32.sum(dim=1)
    nll = lse[:, None] + neg_correct_logit

    if mile_gamma is None:
        base_weight = torch.ones_like(lse)
    else:
        assert mean_logit is not None
        entropy_base = 1.0 + torch.clamp_min(lse - mean_logit, 0.0)
        if mile_gamma == 0.0:
            base_weight = torch.ones_like(entropy_base)
        elif mile_gamma == 1.0:
            base_weight = entropy_base
        else:
            base_weight = entropy_base.pow(mile_gamma)

    denominator = (base_weight * counts).sum().clamp_min(1.0)
    objective = (nll * valid_f32 * base_weight[:, None]).sum() / denominator
    if return_unweighted:
        total_valid = counts.sum().clamp_min(1.0)
        unweighted = (nll * valid_f32).sum() / total_valid
    else:
        unweighted = objective

    target_weight = base_weight * (lse.numel() / denominator)
    dense_weight = target_weight * counts
    return objective, unweighted, dense_weight, target_weight


@triton.jit
def _patch_target_backward_kernel(
    E,
    C,
    TargetWeight,
    dOut,
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
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_tb,
    stride_tk,
    PATCH_SIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_DE: tl.constexpr,
    COMPUTE_DC: tl.constexpr,
    COMPUTE_DBIAS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < B
    coefficient = -grad_scale * tl.load(dOut)
    coefficient *= tl.load(TargetWeight + rows, mask=row_mask, other=0.0)
    tile_d_mask = offs_d[None, :] < D

    de_update = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    for slot in tl.static_range(0, PATCH_SIZE):
        targets = tl.load(
            Targets + rows * stride_tb + slot * stride_tk,
            mask=row_mask,
            other=V,
        ).to(tl.int64)
        valid_target = row_mask & (targets >= 0) & (targets < V)
        tile_mask = valid_target[:, None] & tile_d_mask

        if COMPUTE_DE:
            c = tl.load(
                C + targets[:, None] * stride_cv + offs_d[None, :] * stride_cd,
                mask=tile_mask,
                other=0.0,
            )
            de_update += coefficient[:, None] * de_accum_scale * c

        if COMPUTE_DC:
            e = tl.load(
                E + rows[:, None] * stride_eb + offs_d[None, :] * stride_ed,
                mask=tile_mask,
                other=0.0,
            )
            tl.atomic_add(
                dC + targets[:, None] * stride_cv + offs_d[None, :] * stride_cd,
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

    if COMPUTE_DE:
        de_ptrs = dE + rows[:, None] * stride_eb + offs_d[None, :] * stride_ed
        old_de = tl.load(de_ptrs, mask=row_mask[:, None] & tile_d_mask, other=0.0)
        tl.store(
            de_ptrs,
            old_de + de_update,
            mask=row_mask[:, None] & tile_d_mask,
        )


def patch_target_backward(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    target_weight: torch.Tensor,
    grad_out: torch.Tensor,
    de: torch.Tensor | None,
    dc: torch.Tensor | None,
    dbias: torch.Tensor | None,
    grad_scale: float,
    de_accum_scale: float,
    dc_accum_scale: float,
) -> None:
    """Apply the K sparse target corrections to existing dense gradients."""
    assert targets.ndim == 2
    assert target_weight.ndim == 1
    assert grad_out.numel() == 1
    rows, patch_size = targets.shape
    block_b = 32
    block_d = 64
    _patch_target_backward_kernel[(triton.cdiv(rows, block_b), triton.cdiv(e.size(1), block_d))](
        e,
        c,
        target_weight,
        grad_out,
        targets,
        de,
        dc,
        dbias,
        grad_scale,
        de_accum_scale,
        dc_accum_scale,
        rows,
        e.size(1),
        c.size(0),
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        targets.stride(0),
        targets.stride(1),
        PATCH_SIZE=patch_size,
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        COMPUTE_DE=de is not None,
        COMPUTE_DC=dc is not None,
        COMPUTE_DBIAS=dbias is not None,
        num_warps=4,
    )
