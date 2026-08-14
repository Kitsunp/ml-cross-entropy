# Copyright (C) 2026. All Rights Reserved.
"""Patch-level target kernels for Cut Cross Entropy.

The patch path keeps one embedding row per patch.  It computes one vocabulary
log-sum-exp for that row and only K indexed correct-class logits, avoiding a
K-fold repeat of the embedding and dense vocabulary reduction.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce patch targets and build detached backward row weights.

    The returned dense weight multiplies the softmax term once per row.  The
    target weight multiplies each sparse one-hot correction.  Scaling both by
    the static row count lets the existing backward retain its ``1 / B``
    reduction without reading a data-dependent valid-target count on the host.
    """
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
