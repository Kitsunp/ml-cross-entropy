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
    MiLeWeight,
    TokenLoss,
    UnweightedNLLSum,
    gamma,
    max_entropy,
    loss_scale,
    unweighted_nll_scale,
    B,
    GAMMA_MODE: tl.constexpr,
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
    WeightSum,
    LogBaseMax,
    gamma,
    max_entropy,
    B,
    GAMMA_MODE: tl.constexpr,
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
    tl.atomic_add(WeightSum, tl.sum(tl.where(mask, weight, 0.0), axis=0))


@triton.jit
def _mile_normalize_loss_kernel(
    NLL,
    MiLeWeight,
    WeightSum,
    TokenLoss,
    UnweightedNLLSum,
    B,
    loss_scale,
    unweighted_nll_scale,
    RETURN_METRICS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs_b < B
    weight = tl.load(MiLeWeight + offs_b, mask=mask, other=0.0).to(tl.float32)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return MiLe weights, weighted token losses, and an optional NLL reduction."""
    if lse.shape != mean_logit.shape or lse.shape != nll.shape:
        raise ValueError("MiLe LSE, mean-logit, and NLL tensors must have matching shapes.")
    if math.isnan(max_entropy) or max_entropy < 0:
        raise ValueError(f"max_entropy must be non-negative and not NaN, got {max_entropy}.")
    if return_unweighted_nll_sum and return_unweighted_nll_mean:
        raise ValueError("Request either the unweighted NLL sum or mean, not both.")
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
        torch.zeros((), device=nll.device, dtype=torch.float32)
        if return_unweighted_nll
        else None
    )
    unweighted_nll_arg = unweighted_nll if unweighted_nll is not None else nll
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
            mile_weight,
            token_loss,
            unweighted_nll_arg,
            gamma,
            max_entropy,
            loss_scale,
            unweighted_nll_scale,
            nll.numel(),
            GAMMA_MODE=gamma_mode,
            RETURN_METRICS=return_unweighted_nll,
            BLOCK_B=block_b,
            num_warps=4 if block_b <= 4096 else 8,
            num_stages=1,
        )
        return mile_weight, token_loss, unweighted_nll

    weight_sum = torch.zeros((), device=nll.device, dtype=torch.float32)
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
        weight_sum,
        log_base_max,
        gamma,
        max_entropy,
        nll.numel(),
        GAMMA_MODE=gamma_mode,
        BLOCK_B=block_b,
        num_warps=4,
        num_stages=1,
    )
    _mile_normalize_loss_kernel[grid](
        nll,
        mile_weight,
        weight_sum,
        token_loss,
        unweighted_nll_arg,
        nll.numel(),
        loss_scale,
        unweighted_nll_scale,
        RETURN_METRICS=return_unweighted_nll,
        BLOCK_B=block_b,
        num_warps=4,
        num_stages=1,
    )
    return mile_weight, token_loss, unweighted_nll
