# Copyright (C) 2026. Experimental MiLe reference implementation.
"""Exact, bounded-memory reference implementation of MiLe loss.

This module is intentionally a validation implementation, not the final fused
Triton kernel. It materializes only one ``(tokens, vocab_chunk)`` block at a
time and recomputes those blocks in backward.
"""

from __future__ import annotations

import math

import torch

from cut_cross_entropy.constants import IGNORE_INDEX


class _ChunkedMiLeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None,
        gamma: float,
        chunk_size: int,
    ) -> torch.Tensor:
        e32 = e.float()
        c32 = c.float()
        token_count = e.size(0)

        running_max = torch.full((token_count,), -torch.inf, device=e.device)
        running_sum = torch.zeros((token_count,), device=e.device)
        running_moment = torch.zeros((token_count,), device=e.device)

        for start in range(0, c.size(0), chunk_size):
            stop = min(start + chunk_size, c.size(0))
            logits = e32 @ c32[start:stop].T
            if bias is not None:
                logits = logits + bias[start:stop].float()

            block_max = logits.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            block_exp = torch.exp(logits - new_max[:, None])
            running_sum = running_sum * old_scale + block_exp.sum(dim=-1)
            running_moment = (
                running_moment * old_scale + (block_exp * logits).sum(dim=-1)
            )
            running_max = new_max

        lse = running_max + running_sum.log()
        mean_logit = running_moment / running_sum
        entropy = lse - mean_logit
        correct_logit = (e32 * c32[targets]).sum(dim=-1)
        if bias is not None:
            correct_logit = correct_logit + bias[targets].float()
        nll = lse - correct_logit
        weight = (1.0 + entropy).pow(gamma)
        weight = weight * weight.mean().reciprocal()
        losses = weight * nll

        saved_bias = bias if bias is not None else e.new_empty(0)
        ctx.save_for_backward(e, c, targets, saved_bias, lse, weight)
        ctx.has_bias = bias is not None
        ctx.chunk_size = chunk_size
        return losses

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        e, c, targets, saved_bias, lse, weight = ctx.saved_tensors
        bias = saved_bias if ctx.has_bias else None
        chunk_size = ctx.chunk_size

        e32 = e.float()
        c32 = c.float()
        grad_output = grad_output.float().reshape(-1, 1)
        weight = weight.reshape(-1, 1)

        grad_e32 = torch.zeros_like(e32) if ctx.needs_input_grad[0] else None
        grad_c32 = torch.zeros_like(c32) if ctx.needs_input_grad[1] else None
        grad_bias32 = (
            torch.zeros_like(saved_bias, dtype=torch.float32)
            if ctx.has_bias and ctx.needs_input_grad[3]
            else None
        )

        for start in range(0, c.size(0), chunk_size):
            stop = min(start + chunk_size, c.size(0))
            logits = e32 @ c32[start:stop].T
            if bias is not None:
                logits = logits + bias[start:stop].float()

            probabilities = torch.exp(logits - lse[:, None])
            ce_grad = probabilities.clone()
            target_mask = (targets >= start) & (targets < stop)
            if target_mask.any():
                rows = target_mask.nonzero(as_tuple=False).flatten()
                ce_grad[rows, targets[rows] - start] -= 1.0

            mile_grad = weight * ce_grad * grad_output

            if grad_e32 is not None:
                grad_e32.add_(mile_grad @ c32[start:stop])
            if grad_c32 is not None:
                grad_c32[start:stop] = mile_grad.T @ e32
            if grad_bias32 is not None:
                grad_bias32[start:stop] = mile_grad.sum(dim=0)

        grad_e = grad_e32.to(e.dtype) if grad_e32 is not None else None
        grad_c = grad_c32.to(c.dtype) if grad_c32 is not None else None
        grad_bias = (
            grad_bias32.to(saved_bias.dtype) if grad_bias32 is not None else None
        )
        return grad_e, grad_c, None, grad_bias, None, None


def linear_mile_loss(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    gamma: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    shift: bool | int = 0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute exact MiLe loss while bounding the materialized logit block.

    The loss is normalized ``(1 + H(p)) ** gamma * cross_entropy``. Entropy
    weights are always treated as stop-gradient quantities and normalized to
    mean one over valid targets, matching the released MiLe training code.
    Auxiliary statistics are O(tokens), while the temporary PyTorch logit
    block is O(tokens * chunk_size).

    This validation path currently excludes softcapping and vocabulary
    parallelism. The production target is to fuse the same statistics into
    CCE's Triton kernels.
    """
    if e.shape[:-1] != targets.shape:
        raise ValueError(f"Expected e.shape[:-1] == targets.shape, got {e.shape} and {targets.shape}.")
    if e.size(-1) != c.size(1):
        raise ValueError(f"Embedding dimensions differ: {e.size(-1)} and {c.size(1)}.")
    if bias is not None and bias.shape != (c.size(0),):
        raise ValueError(f"Expected bias shape {(c.size(0),)}, got {bias.shape}.")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be finite and non-negative, got {gamma}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unknown reduction {reduction}.")

    shift = int(shift)
    if shift < 0 or shift >= targets.size(-1):
        raise ValueError(f"shift must be in [0, {targets.size(-1)}), got {shift}.")
    if shift:
        e = e[..., :-shift, :]
        targets = targets[..., shift:]

    output_shape = targets.shape
    e = e.reshape(-1, e.size(-1))
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    valid_losses = _ChunkedMiLeFunction.apply(
        e[valid], c, targets[valid], bias, float(gamma), int(chunk_size)
    )

    if reduction == "mean":
        return valid_losses.mean()
    if reduction == "sum":
        return valid_losses.sum()

    losses = valid_losses.new_zeros(targets.shape)
    losses[valid] = valid_losses
    return losses.view(output_shape)
