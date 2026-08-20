# Copyright (C) 2026. All Rights Reserved.
"""Fused token-wise MiLe weighting for the CCE path."""

import math

import torch
import triton
import triton.language as tl

_SINGLE_BLOCK_MAX_TOKENS = 16384
_MULTI_BLOCK_TOKENS = 256


@triton.jit
def _mile_single_block_kernel(
    LSE,
    MeanLogit,
    NLL,
    GroupMask,
    MiLeWeight,
    TokenLoss,
    UnweightedNLLSum,
    gamma,
    max_entropy,
    loss_scale,
    unweighted_nll_scale,
    B,
    GAMMA_MODE: tl.constexpr,
    HAS_GROUP_MASK: tl.constexpr,
    RETURN_METRICS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    mask = offs_b < B
    lse = tl.load(LSE + offs_b, mask=mask, other=0.0).to(tl.float32)
    mean_logit = tl.load(MeanLogit + offs_b, mask=mask, other=0.0).to(tl.float32)
    entropy = tl.minimum(tl.maximum(lse - mean_logit, 0.0), max_entropy)
    base = 1.0 + entropy
    if GAMMA_MODE == 0:
        weight = tl.full(base.shape, 1.0, tl.float32)
    elif GAMMA_MODE == 1:
        weight = base
    else:
        log_base = tl.log(base)
        max_log_base = tl.max(tl.where(mask, log_base, -float("inf")), axis=0)
        weight = tl.exp(gamma * (log_base - max_log_base))
    if HAS_GROUP_MASK:
        group = tl.load(GroupMask + offs_b, mask=mask, other=0).to(tl.int1)
        group_members = mask & group
        group_count = tl.sum(group_members.to(tl.float32), axis=0)
        other_count = B - group_count
        total_sum = tl.sum(tl.where(mask, weight, 0.0), axis=0)
        group_sum = tl.sum(tl.where(group_members, weight, 0.0), axis=0)
        other_sum = total_sum - group_sum
        group_scale = group_count / tl.maximum(group_sum, 1.0e-20)
        other_scale = other_count / tl.maximum(other_sum, 1.0e-20)
        weight *= tl.where(group, group_scale, other_scale)
    else:
        weight *= B / tl.sum(tl.where(mask, weight, 0.0), axis=0)
    nll = tl.load(NLL + offs_b, mask=mask, other=0.0).to(tl.float32)
    tl.store(MiLeWeight + offs_b, weight, mask=mask)
    tl.store(TokenLoss + offs_b, (nll * loss_scale) * weight, mask=mask)
    if RETURN_METRICS:
        tl.store(
            UnweightedNLLSum,
            tl.sum(tl.where(mask, nll * unweighted_nll_scale, 0.0), axis=0),
        )


@triton.jit
def _mile_log_base_max_kernel(
    LSE,
    MeanLogit,
    MiLeWeight,
    LogBaseMax,
    max_entropy,
    B,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs_b < B
    lse = tl.load(LSE + offs_b, mask=mask, other=0.0).to(tl.float32)
    mean_logit = tl.load(MeanLogit + offs_b, mask=mask, other=0.0).to(tl.float32)
    entropy = tl.minimum(tl.maximum(lse - mean_logit, 0.0), max_entropy)
    log_base = tl.log(1.0 + entropy)
    tl.store(MiLeWeight + offs_b, log_base, mask=mask)
    tl.atomic_max(LogBaseMax, tl.max(tl.where(mask, log_base, -float("inf")), axis=0))


@triton.jit
def _mile_weight_sum_kernel(
    LSE,
    MeanLogit,
    MiLeWeight,
    GroupMask,
    WeightSum,
    LogBaseMax,
    gamma,
    max_entropy,
    B,
    GAMMA_MODE: tl.constexpr,
    HAS_GROUP_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs_b < B
    if GAMMA_MODE == 0:
        weight = tl.full((BLOCK_B,), 1.0, tl.float32)
    elif GAMMA_MODE == 1:
        lse = tl.load(LSE + offs_b, mask=mask, other=0.0).to(tl.float32)
        mean_logit = tl.load(MeanLogit + offs_b, mask=mask, other=0.0).to(tl.float32)
        entropy = tl.minimum(tl.maximum(lse - mean_logit, 0.0), max_entropy)
        weight = 1.0 + entropy
    else:
        log_base = tl.load(MiLeWeight + offs_b, mask=mask, other=0.0).to(tl.float32)
        max_log_base = tl.load(LogBaseMax)
        weight = tl.exp(gamma * (log_base - max_log_base))
    tl.store(MiLeWeight + offs_b, weight, mask=mask)
    if HAS_GROUP_MASK:
        group = tl.load(GroupMask + offs_b, mask=mask, other=0).to(tl.int1)
        group_members = mask & group
        tl.atomic_add(
            WeightSum,
            tl.sum(tl.where(mask, weight, 0.0), axis=0),
        )
        tl.atomic_add(
            WeightSum + 1,
            tl.sum(tl.where(group_members, weight, 0.0), axis=0),
        )
        tl.atomic_add(WeightSum + 2, tl.sum(group_members.to(tl.float32), axis=0))
    else:
        tl.atomic_add(WeightSum, tl.sum(tl.where(mask, weight, 0.0), axis=0))


@triton.jit
def _mile_normalize_loss_kernel(
    NLL,
    MiLeWeight,
    GroupMask,
    WeightSum,
    TokenLoss,
    UnweightedNLLSum,
    B,
    loss_scale,
    unweighted_nll_scale,
    RETURN_METRICS: tl.constexpr,
    HAS_GROUP_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs_b < B
    weight = tl.load(MiLeWeight + offs_b, mask=mask, other=0.0).to(tl.float32)
    if HAS_GROUP_MASK:
        group = tl.load(GroupMask + offs_b, mask=mask, other=0).to(tl.int1)
        group_scale = tl.load(WeightSum + 2) / tl.maximum(tl.load(WeightSum + 1), 1.0e-20)
        other_scale = (B - tl.load(WeightSum + 2)) / tl.maximum(
            tl.load(WeightSum) - tl.load(WeightSum + 1), 1.0e-20
        )
        weight *= tl.where(group, group_scale, other_scale)
    else:
        weight *= B / tl.load(WeightSum)
    nll = tl.load(NLL + offs_b, mask=mask, other=0.0).to(tl.float32)
    tl.store(MiLeWeight + offs_b, weight, mask=mask)
    tl.store(TokenLoss + offs_b, (nll * loss_scale) * weight, mask=mask)
    if RETURN_METRICS:
        tl.atomic_add(
            UnweightedNLLSum,
            tl.sum(tl.where(mask, nll * unweighted_nll_scale, 0.0), axis=0),
        )


def cce_mile_forward_kernel(
    lse: torch.Tensor,
    mean_logit: torch.Tensor,
    nll: torch.Tensor,
    gamma: float,
    return_unweighted_nll_sum: bool = False,
    max_entropy: float = math.inf,
    *,
    return_unweighted_nll_mean: bool = False,
    mean_reduction: bool = False,
    group_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return MiLe weights, weighted token losses, and an optional NLL reduction."""
    if lse.shape != mean_logit.shape or lse.shape != nll.shape:
        raise ValueError("MiLe LSE, mean-logit, and NLL tensors must have matching shapes.")
    if math.isnan(max_entropy) or max_entropy < 0:
        raise ValueError(f"max_entropy must be non-negative and not NaN, got {max_entropy}.")
    if return_unweighted_nll_sum and return_unweighted_nll_mean:
        raise ValueError("Request either the unweighted NLL sum or mean, not both.")
    if group_mask is not None:
        if group_mask.shape != nll.shape:
            raise ValueError("MiLe group_mask must match the NLL shape.")
        if group_mask.dtype != torch.bool:
            raise TypeError("MiLe group_mask must be boolean.")
        if group_mask.device != nll.device:
            raise ValueError("MiLe group_mask and NLL must share a device.")
    if lse.numel() == 0:
        unweighted_nll = (
            torch.zeros((), device=nll.device, dtype=torch.float32)
            if return_unweighted_nll_sum or return_unweighted_nll_mean
            else None
        )
        return torch.empty_like(nll), torch.empty_like(nll), unweighted_nll

    mile_weight = torch.empty_like(nll)
    token_loss = torch.empty_like(nll)
    return_unweighted_nll = return_unweighted_nll_sum or return_unweighted_nll_mean
    unweighted_nll = (
        torch.zeros((), device=nll.device, dtype=torch.float32) if return_unweighted_nll else None
    )
    unweighted_nll_arg = unweighted_nll if unweighted_nll is not None else nll
    group_mask_arg = group_mask if group_mask is not None else nll
    loss_scale = 1.0 / nll.numel() if mean_reduction else 1.0
    unweighted_nll_scale = 1.0 / nll.numel() if return_unweighted_nll_mean else 1.0
    # Gamma 1 with a finite entropy bound cannot overflow its sum. General
    # powers are normalized in log space; the common scale cancels exactly.
    gamma_mode = 0 if gamma == 0.0 else 1 if gamma == 1.0 and math.isfinite(max_entropy) else 2
    # One program avoids global synchronization for ordinary microbatches;
    # larger vectors use bounded blocks to avoid excessive register pressure.
    if nll.numel() <= _SINGLE_BLOCK_MAX_TOKENS:
        block_b = triton.next_power_of_2(nll.numel())
        _mile_single_block_kernel[(1,)](
            lse,
            mean_logit,
            nll,
            group_mask_arg,
            mile_weight,
            token_loss,
            unweighted_nll_arg,
            gamma,
            max_entropy,
            loss_scale,
            unweighted_nll_scale,
            nll.numel(),
            GAMMA_MODE=gamma_mode,
            HAS_GROUP_MASK=group_mask is not None,
            RETURN_METRICS=return_unweighted_nll,
            BLOCK_B=block_b,
            num_warps=4 if block_b <= 4096 else 8,
            num_stages=1,
        )
        return mile_weight, token_loss, unweighted_nll

    weight_sum = torch.zeros(
        3 if group_mask is not None else 1,
        device=nll.device,
        dtype=torch.float32,
    )
    block_b = _MULTI_BLOCK_TOKENS
    grid = (triton.cdiv(nll.numel(), block_b),)
    log_base_max = nll
    if gamma_mode == 2:
        log_base_max = torch.full((), -torch.inf, device=nll.device, dtype=torch.float32)
        _mile_log_base_max_kernel[grid](
            lse,
            mean_logit,
            mile_weight,
            log_base_max,
            max_entropy,
            nll.numel(),
            BLOCK_B=block_b,
            num_warps=4,
            num_stages=1,
        )
    _mile_weight_sum_kernel[grid](
        lse,
        mean_logit,
        mile_weight,
        group_mask_arg,
        weight_sum,
        log_base_max,
        gamma,
        max_entropy,
        nll.numel(),
        GAMMA_MODE=gamma_mode,
        HAS_GROUP_MASK=group_mask is not None,
        BLOCK_B=block_b,
        num_warps=4,
        num_stages=1,
    )
    _mile_normalize_loss_kernel[grid](
        nll,
        mile_weight,
        group_mask_arg,
        weight_sum,
        token_loss,
        unweighted_nll_arg,
        nll.numel(),
        loss_scale,
        unweighted_nll_scale,
        RETURN_METRICS=return_unweighted_nll,
        HAS_GROUP_MASK=group_mask is not None,
        BLOCK_B=block_b,
        num_warps=4,
        num_stages=1,
    )
    return mile_weight, token_loss, unweighted_nll


def normalize_mile_weight_groups(
    mile_weight: torch.Tensor,
    nll: torch.Tensor,
    group_mask: torch.Tensor,
    *,
    mean_reduction: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize detached MiLe weights independently across two populations.

    MEAP intentionally raises uncertainty at corrupted input positions.  A
    single global MiLe normalization can therefore move most of the gradient
    mass onto that synthetic population.  This helper preserves MiLe within
    both populations while keeping mean weight one in each of them.
    """
    if mile_weight.shape != nll.shape or group_mask.shape != nll.shape:
        raise ValueError("MiLe weights, NLL, and group_mask must share a shape.")
    if group_mask.dtype != torch.bool:
        raise TypeError("MiLe group_mask must be boolean.")
    if group_mask.device != mile_weight.device:
        raise ValueError("MiLe group_mask and weights must share a device.")
    if mile_weight.numel() == 0:
        return mile_weight, nll

    weights_fp32 = mile_weight.float()
    group_fp32 = group_mask.float()
    group_count = group_fp32.sum()
    other_count = group_fp32.numel() - group_count
    group_sum = (weights_fp32 * group_fp32).sum()
    other_sum = weights_fp32.sum() - group_sum
    group_scale = torch.where(
        group_count > 0,
        group_count / group_sum.clamp_min(torch.finfo(torch.float32).tiny),
        torch.ones_like(group_count),
    )
    other_scale = torch.where(
        other_count > 0,
        other_count / other_sum.clamp_min(torch.finfo(torch.float32).tiny),
        torch.ones_like(other_count),
    )
    normalized = weights_fp32 * torch.where(group_mask, group_scale, other_scale)
    loss_scale = 1.0 / nll.numel() if mean_reduction else 1.0
    token_loss = nll.float() * normalized * loss_scale
    return normalized.to(mile_weight.dtype), token_loss.to(nll.dtype)
