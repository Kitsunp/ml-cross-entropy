# Copyright (C) 2026. All Rights Reserved.
"""Fused token-wise MiLe weighting for the CCE path."""

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
    gamma,
    B,
    GAMMA_MODE: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    mask = offs_b < B
    lse = tl.load(LSE + offs_b, mask=mask, other=0.0).to(tl.float32)
    mean_logit = tl.load(MeanLogit + offs_b, mask=mask, other=0.0).to(tl.float32)
    base = 1.0 + tl.maximum(lse - mean_logit, 0.0)
    if GAMMA_MODE == 0:
        weight = tl.full(base.shape, 1.0, tl.float32)
    elif GAMMA_MODE == 1:
        weight = base
    else:
        weight = tl.exp(gamma * tl.log(base))
    weight *= B / tl.sum(tl.where(mask, weight, 0.0), axis=0)
    nll = tl.load(NLL + offs_b, mask=mask, other=0.0).to(tl.float32)
    tl.store(MiLeWeight + offs_b, weight, mask=mask)
    tl.store(TokenLoss + offs_b, nll * weight, mask=mask)


@triton.jit
def _mile_weight_sum_kernel(
    LSE,
    MeanLogit,
    MiLeWeight,
    WeightSum,
    gamma,
    B,
    GAMMA_MODE: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs_b < B
    lse = tl.load(LSE + offs_b, mask=mask, other=0.0).to(tl.float32)
    mean_logit = tl.load(MeanLogit + offs_b, mask=mask, other=0.0).to(tl.float32)
    base = 1.0 + tl.maximum(lse - mean_logit, 0.0)
    if GAMMA_MODE == 0:
        weight = tl.full(base.shape, 1.0, tl.float32)
    elif GAMMA_MODE == 1:
        weight = base
    else:
        weight = tl.exp(gamma * tl.log(base))
    tl.store(MiLeWeight + offs_b, weight, mask=mask)
    tl.atomic_add(WeightSum, tl.sum(tl.where(mask, weight, 0.0), axis=0))


@triton.jit
def _mile_normalize_loss_kernel(
    NLL,
    MiLeWeight,
    WeightSum,
    TokenLoss,
    B,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs_b < B
    weight = tl.load(MiLeWeight + offs_b, mask=mask, other=0.0).to(tl.float32)
    weight *= B / tl.load(WeightSum)
    nll = tl.load(NLL + offs_b, mask=mask, other=0.0).to(tl.float32)
    tl.store(MiLeWeight + offs_b, weight, mask=mask)
    tl.store(TokenLoss + offs_b, nll * weight, mask=mask)


def cce_mile_forward_kernel(
    lse: torch.Tensor,
    mean_logit: torch.Tensor,
    nll: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached mean-normalized MiLe weights and weighted token losses."""
    if lse.shape != mean_logit.shape or lse.shape != nll.shape:
        raise ValueError("MiLe LSE, mean-logit, and NLL tensors must have matching shapes.")
    if lse.numel() == 0:
        return torch.empty_like(nll), torch.empty_like(nll)

    mile_weight = torch.empty_like(nll)
    token_loss = torch.empty_like(nll)
    gamma_mode = 0 if gamma == 0.0 else 1 if gamma == 1.0 else 2
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
            gamma,
            nll.numel(),
            GAMMA_MODE=gamma_mode,
            BLOCK_B=block_b,
            num_warps=4 if block_b <= 4096 else 8,
            num_stages=1,
        )
        return mile_weight, token_loss

    weight_sum = torch.zeros((), device=nll.device, dtype=torch.float32)
    block_b = _MULTI_BLOCK_TOKENS
    grid = (triton.cdiv(nll.numel(), block_b),)
    _mile_weight_sum_kernel[grid](
        lse,
        mean_logit,
        mile_weight,
        weight_sum,
        gamma,
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
        nll.numel(),
        BLOCK_B=block_b,
        num_warps=4,
        num_stages=1,
    )
    return mile_weight, token_loss
