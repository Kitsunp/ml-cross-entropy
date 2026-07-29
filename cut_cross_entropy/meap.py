# Copyright (C) 2026. All Rights Reserved.
"""Mask-Enhanced Autoregressive Prediction (MEAP) input corruption.

MEAP is deliberately separate from linear cross entropy: it corrupts token IDs
before the model forward while labels, causal attention, and padding stay clean.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _swap_or_not_permute(value, count, key0, key1):
    """Twelve keyed involution rounds forming a permutation over ``[0, count)``."""
    round_key = key0 ^ (key1 * 0x9E3779B1)
    for _ in tl.static_range(0, 12):
        round_key = round_key * 1664525 + 1013904223
        pivot = round_key % count
        partner = pivot + count - value
        partner = tl.where(partner >= count, partner - count, partner)
        pair_id = tl.minimum(value, partner)
        coin = pair_id ^ round_key
        coin ^= coin >> 16
        coin *= 0x7FEB352D
        coin ^= coin >> 15
        coin *= 0x846CA68B
        coin ^= coin >> 16
        value = tl.where((coin & 1) != 0, partner, value)
    return value


@triton.jit
def _meap_mask_inputs_kernel(
    InputIds,
    SelectionMask,
    OutputIds,
    OutputMask,
    sequence_length,
    mask_token_id,
    mask_ratio,
    seed,
    stride_ib,
    stride_it,
    stride_sb,
    stride_st,
    stride_ob,
    stride_ot,
    stride_mb,
    stride_mt,
    HAS_SELECTION_MASK: tl.constexpr,
    MASK_IS_PADDING: tl.constexpr,
    EXCLUDE_LAST: tl.constexpr,
    RETURN_MASK: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    positions = tl.arange(0, BLOCK_T)
    in_bounds = positions < sequence_length

    input_ptrs = InputIds + row * stride_ib + positions * stride_it
    input_ids = tl.load(input_ptrs, mask=in_bounds, other=0)
    if HAS_SELECTION_MASK:
        supplied_mask = tl.load(
            SelectionMask + row * stride_sb + positions * stride_st,
            mask=in_bounds,
            other=0,
        ).to(tl.int1)
        if MASK_IS_PADDING:
            eligible = in_bounds & ~supplied_mask
        else:
            eligible = in_bounds & supplied_mask
    else:
        eligible = in_bounds

    if EXCLUDE_LAST:
        last_eligible = tl.max(tl.where(eligible, positions, -1), axis=0)
        eligible &= positions != last_eligible

    eligible_count = tl.sum(eligible.to(tl.int32), axis=0)
    requested_count = (eligible_count.to(tl.float32) * mask_ratio).to(tl.int32)
    requested_count = tl.where(
        (mask_ratio > 0.0) & (eligible_count > 0),
        tl.maximum(requested_count, 1),
        0,
    )
    requested_count = tl.minimum(requested_count, eligible_count)

    # Convert sparse eligible positions to the dense rank domain [0, N).
    eligible_rank = tl.cumsum(eligible.to(tl.int32), axis=0) - 1
    eligible_rank = tl.where(eligible, eligible_rank, 0).to(tl.uint32)

    safe_count = tl.maximum(eligible_count, 1).to(tl.uint32)
    key0, key1, _, _ = tl.randint4x(seed, row)
    permuted_rank = _swap_or_not_permute(
        eligible_rank, safe_count, key0.to(tl.uint32), key1.to(tl.uint32)
    )

    selected = eligible & (permuted_rank < requested_count.to(tl.uint32))

    output = tl.where(selected, mask_token_id, input_ids)
    tl.store(OutputIds + row * stride_ob + positions * stride_ot, output, mask=in_bounds)
    if RETURN_MASK:
        tl.store(
            OutputMask + row * stride_mb + positions * stride_mt,
            selected,
            mask=in_bounds,
        )


def _validate_meap_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    eligible_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    seed: int,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, sequence), got {input_ids.shape}.")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"input_ids must be int32 or int64, got {input_ids.dtype}.")
    if not isinstance(mask_token_id, int):
        raise TypeError("mask_token_id must be an integer.")
    dtype_limits = torch.iinfo(input_ids.dtype)
    if not 0 <= mask_token_id <= dtype_limits.max:
        raise ValueError(
            f"mask_token_id must be in [0, {dtype_limits.max}] for {input_ids.dtype}."
        )
    if not math.isfinite(mask_ratio) or not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be finite and in [0, 1], got {mask_ratio}.")
    if not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be an integer in [0, 2**32 - 1].")
    if eligible_mask is not None and padding_mask is not None:
        raise ValueError("Pass either eligible_mask or padding_mask, not both.")
    selection_mask = padding_mask if padding_mask is not None else eligible_mask
    if selection_mask is not None:
        if selection_mask.shape != input_ids.shape:
            raise ValueError(
                "The selection mask must match input_ids shape, got "
                f"{selection_mask.shape} and {input_ids.shape}."
            )
        if selection_mask.dtype != torch.bool:
            raise TypeError(f"The selection mask must be boolean, got {selection_mask.dtype}.")
        if selection_mask.device != input_ids.device:
            raise ValueError("The selection mask and input_ids must be on the same device.")


def _torch_meap_mask_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    eligible_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    seed: int,
    exclude_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Readable reference used for validation and torch.compile benchmarks."""
    if padding_mask is not None:
        eligible = ~padding_mask
    elif eligible_mask is not None:
        eligible = eligible_mask.clone()
    else:
        eligible = torch.ones_like(input_ids, dtype=torch.bool)
    if exclude_last:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        last = torch.where(eligible, positions, -1).amax(dim=1)
        eligible &= positions.unsqueeze(0) != last.unsqueeze(1)

    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    scores = torch.rand(
        input_ids.shape,
        device=input_ids.device,
        dtype=torch.float32,
        generator=generator,
    ).masked_fill(~eligible, 2.0)
    order = scores.argsort(dim=1)
    eligible_count = eligible.sum(dim=1)
    requested_count = (eligible_count * mask_ratio).to(torch.long)
    if mask_ratio > 0.0:
        requested_count = torch.where(
            eligible_count > 0, requested_count.clamp_min(1), requested_count
        )
    ranks = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    selected_by_rank = ranks < requested_count.unsqueeze(1)
    selected = torch.zeros_like(eligible).scatter_(1, order, selected_by_rank)
    selected &= eligible
    return input_ids.masked_fill(selected, mask_token_id), selected


def _triton_meap_mask_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    eligible_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    seed: int,
    exclude_last: bool,
    return_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = input_ids.size(1)
    block_t = triton.next_power_of_2(sequence_length)
    if block_t > 4096:
        raise ValueError(
            "The fixed-count Triton MEAP kernel supports sequence lengths up to 4096."
        )
    output = torch.empty_like(input_ids)
    selected = torch.empty_like(input_ids, dtype=torch.bool) if return_mask else input_ids
    selection_mask = padding_mask if padding_mask is not None else eligible_mask
    eligible_arg = selection_mask if selection_mask is not None else input_ids
    _meap_mask_inputs_kernel[(input_ids.size(0),)](
        input_ids,
        eligible_arg,
        output,
        selected,
        sequence_length,
        mask_token_id,
        mask_ratio,
        seed,
        input_ids.stride(0),
        input_ids.stride(1),
        eligible_arg.stride(0),
        eligible_arg.stride(1),
        output.stride(0),
        output.stride(1),
        selected.stride(0),
        selected.stride(1),
        HAS_SELECTION_MASK=selection_mask is not None,
        MASK_IS_PADDING=padding_mask is not None,
        EXCLUDE_LAST=exclude_last,
        RETURN_MASK=return_mask,
        BLOCK_T=block_t,
        num_warps=4 if block_t <= 512 else 8,
        num_stages=1,
    )
    return output, selected


def meap_mask_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    *,
    enabled: bool = True,
    mask_ratio: float = 0.15,
    eligible_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    seed: int = 0,
    exclude_last: bool = True,
    return_mask: bool = False,
    implementation: str = "triton",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Corrupt autoregressive inputs with a fixed MEAP mask ratio per sequence.

    ``padding_mask`` is ``True`` where replacement is forbidden and avoids
    materializing its boolean inverse. Alternatively, ``eligible_mask`` is
    ``True`` only where replacement is allowed, including any BOS/EOS or other
    protected-token exclusions. Pass only one. Labels are not accepted
    intentionally: they must remain clean.

    The last eligible input is excluded by default because its hidden state has
    no clean next-token target when CCE is called with ``shift=1``. A positive
    ratio masks at least one token in each row that still has eligible tokens.
    """
    _validate_meap_inputs(
        input_ids, mask_token_id, mask_ratio, eligible_mask, padding_mask, seed
    )
    if implementation not in {"triton", "torch"}:
        raise ValueError(f"Unknown MEAP implementation {implementation!r}.")
    if not enabled:
        if return_mask:
            return input_ids, torch.zeros_like(input_ids, dtype=torch.bool)
        return input_ids

    if input_ids.numel() == 0:
        output = input_ids.clone()
        if return_mask:
            return output, torch.zeros_like(input_ids, dtype=torch.bool)
        return output

    if mask_ratio == 0.0:
        output = input_ids.clone()
        if return_mask:
            return output, torch.zeros_like(input_ids, dtype=torch.bool)
        return output

    if implementation == "torch":
        output, selected = _torch_meap_mask_inputs(
            input_ids,
            mask_token_id,
            mask_ratio,
            eligible_mask,
            padding_mask,
            seed,
            exclude_last,
        )
    else:
        if not input_ids.is_cuda:
            raise ValueError("The Triton MEAP implementation requires CUDA tensors.")
        output, selected = _triton_meap_mask_inputs(
            input_ids,
            mask_token_id,
            mask_ratio,
            eligible_mask,
            padding_mask,
            seed,
            exclude_last,
            return_mask,
        )

    return (output, selected) if return_mask else output


__all__ = ["meap_mask_inputs"]
