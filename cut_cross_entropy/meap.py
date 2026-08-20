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
from torch import nn


def meap_attention_diagnostics(
    clean_attention: torch.Tensor,
    masked_attention: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    eligible_mask: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
    visibility_mask: torch.Tensor | None = None,
    causal: bool = True,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compute the paired attention diagnostics used to evaluate MEAP.

    This is an evaluation-only helper. ``clean_attention`` and
    ``masked_attention`` must come from the same examples and checkpoint; the
    final dimension is the attended key/token dimension. ``selected_mask``
    identifies the keys replaced in the masked copy and may have leading
    singleton dimensions before its final sequence dimension. ``eligible_mask``
    can exclude padding or other invalid keys from both populations.
    ``query_mask`` excludes invalid query rows. For square self-attention it
    defaults to ``eligible_mask``. ``visibility_mask`` can provide an explicit
    query-key mask, while ``causal=True`` excludes future keys by default.

    The returned relative score decay and non-mask variance change distinguish
    MEAP's intended attention effect from the operational masking fraction.
    """
    if clean_attention.shape != masked_attention.shape:
        raise ValueError("clean_attention and masked_attention must share a shape.")
    if clean_attention.ndim < 2 or not clean_attention.is_floating_point():
        raise TypeError("attention tensors must be floating point with at least 2 dims.")
    if not masked_attention.is_floating_point():
        raise TypeError("attention tensors must be floating point.")
    if clean_attention.device != masked_attention.device:
        raise ValueError("attention tensors must share a device.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")

    def broadcast_key_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
        if mask.ndim < 1:
            raise ValueError(f"{name} must have a key dimension.")
        if mask.device != clean_attention.device:
            raise ValueError(f"{name} and attention tensors must share a device.")
        if mask.shape[-1] != clean_attention.shape[-1]:
            raise ValueError(f"{name} must match the attention key dimension.")
        result = mask.to(dtype=torch.bool)
        while result.ndim < clean_attention.ndim:
            result = result.unsqueeze(-2)
        try:
            return torch.broadcast_to(result, clean_attention.shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name} leading dimensions are not broadcastable to attention."
            ) from error

    def broadcast_query_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
        if mask.ndim < 1:
            raise ValueError(f"{name} must have a query dimension.")
        if mask.device != clean_attention.device:
            raise ValueError(f"{name} and attention tensors must share a device.")
        if mask.shape[-1] != clean_attention.shape[-2]:
            raise ValueError(f"{name} must match the attention query dimension.")
        result = mask.to(dtype=torch.bool).unsqueeze(-1)
        while result.ndim < clean_attention.ndim:
            result = result.unsqueeze(-3)
        try:
            return torch.broadcast_to(result, clean_attention.shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name} leading dimensions are not broadcastable to attention."
            ) from error

    def broadcast_visibility_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim < 2:
            raise ValueError("visibility_mask must have query and key dimensions.")
        if mask.device != clean_attention.device:
            raise ValueError("visibility_mask and attention tensors must share a device.")
        if mask.shape[-2:] != clean_attention.shape[-2:]:
            raise ValueError("visibility_mask must match the attention query and key dimensions.")
        result = mask.to(dtype=torch.bool)
        while result.ndim < clean_attention.ndim:
            result = result.unsqueeze(-3)
        try:
            return torch.broadcast_to(result, clean_attention.shape)
        except RuntimeError as error:
            raise ValueError(
                "visibility_mask leading dimensions are not broadcastable to attention."
            ) from error

    selected = broadcast_key_mask(selected_mask, "selected_mask")
    eligible = (
        torch.ones_like(selected)
        if eligible_mask is None
        else broadcast_key_mask(eligible_mask, "eligible_mask")
    )
    if (
        query_mask is None
        and eligible_mask is not None
        and (clean_attention.shape[-2] == clean_attention.shape[-1])
    ):
        query_mask = eligible_mask
    query_valid = (
        torch.ones_like(selected)
        if query_mask is None
        else broadcast_query_mask(query_mask, "query_mask")
    )
    visible = eligible & query_valid
    if causal:
        query_length, key_length = clean_attention.shape[-2:]
        query_positions = torch.arange(query_length, device=clean_attention.device).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=clean_attention.device).unsqueeze(0)
        visible = visible & (key_positions <= query_positions + max(key_length - query_length, 0))
    if visibility_mask is not None:
        visible = visible & broadcast_visibility_mask(visibility_mask)

    selected = selected & visible
    unselected = ~selected & visible
    selected_count = selected.sum(dim=-1)
    unselected_count = unselected.sum(dim=-1)
    valid_rows = (selected_count > 0) & (unselected_count > 0)
    if not bool(valid_rows.any()):
        raise ValueError("No attention row contains selected and unselected eligible keys.")

    clean_fp32 = clean_attention.float()
    masked_fp32 = masked_attention.float()
    selected_denominator = selected_count.clamp_min(1).float()
    unselected_denominator = unselected_count.clamp_min(1).float()
    clean_selected_per_row = (
        clean_fp32.masked_fill(~selected, 0.0).sum(dim=-1) / selected_denominator
    )
    masked_selected_per_row = (
        masked_fp32.masked_fill(~selected, 0.0).sum(dim=-1) / selected_denominator
    )
    clean_selected_mean = clean_selected_per_row.masked_select(valid_rows).mean()
    masked_selected_mean = masked_selected_per_row.masked_select(valid_rows).mean()

    clean_unselected_mean = (
        clean_fp32.masked_fill(~unselected, 0.0).sum(dim=-1) / unselected_denominator
    )
    masked_unselected_mean = (
        masked_fp32.masked_fill(~unselected, 0.0).sum(dim=-1) / unselected_denominator
    )
    clean_squared_error = (clean_fp32 - clean_unselected_mean.unsqueeze(-1)).square()
    masked_squared_error = (masked_fp32 - masked_unselected_mean.unsqueeze(-1)).square()
    clean_variance_per_row = (
        clean_squared_error.masked_fill(~unselected, 0.0).sum(dim=-1) / unselected_denominator
    )
    masked_variance_per_row = (
        masked_squared_error.masked_fill(~unselected, 0.0).sum(dim=-1) / unselected_denominator
    )
    clean_variance = clean_variance_per_row.masked_select(valid_rows).mean()
    masked_variance = masked_variance_per_row.masked_select(valid_rows).mean()

    return {
        "masked_attention_score_decay": clean_selected_mean - masked_selected_mean,
        "masked_attention_relative_decay": (clean_selected_mean - masked_selected_mean)
        / clean_selected_mean.abs().clamp_min(eps),
        "unmasked_attention_variance_change": masked_variance - clean_variance,
        "unmasked_attention_variance_relative_change": (masked_variance - clean_variance)
        / clean_variance.abs().clamp_min(eps),
    }


def apply_meap_embedding_override(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Replace MEAP positions after the complete input-embedding pipeline.

    Apply this function after dense or compositional token embeddings and after
    optional token-level augmentations such as Spelling Bee.  The final
    override keeps the mask representation independent from padding rows,
    Leviathan codebooks, and shared byte embeddings.

    ``input_ids`` must contain a dedicated mask ID that is absent from clean
    training data.  Labels and attention masks are intentionally not accepted.
    """
    if input_ids.ndim + 1 != embeddings.ndim:
        raise ValueError(
            "embeddings must have one trailing hidden dimension beyond input_ids, got "
            f"{input_ids.shape} and {embeddings.shape}."
        )
    if input_ids.shape != embeddings.shape[:-1]:
        raise ValueError(
            "input_ids shape must match the leading embedding dimensions, got "
            f"{input_ids.shape} and {embeddings.shape}."
        )
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"input_ids must be int32 or int64, got {input_ids.dtype}.")
    if not embeddings.is_floating_point():
        raise TypeError(f"embeddings must be floating point, got {embeddings.dtype}.")
    if mask_embedding.ndim != 1 or mask_embedding.shape[0] != embeddings.shape[-1]:
        raise ValueError(
            "mask_embedding must be a vector matching the hidden size, got "
            f"{mask_embedding.shape} and hidden size {embeddings.shape[-1]}."
        )
    if input_ids.device != embeddings.device or mask_embedding.device != embeddings.device:
        raise ValueError("input_ids, embeddings, and mask_embedding must share a device.")
    if not isinstance(mask_token_id, int) or mask_token_id < 0:
        raise ValueError("mask_token_id must be a non-negative integer.")

    selected = input_ids.eq(mask_token_id).unsqueeze(-1)
    broadcast_shape = (1,) * input_ids.ndim + (embeddings.shape[-1],)
    mask_value = mask_embedding.to(dtype=embeddings.dtype).view(broadcast_shape)
    return torch.where(selected, mask_value, embeddings)


class MEAPEmbeddingOverride(nn.Module):
    """Own a dedicated trainable representation for MEAP-masked positions.

    The module is intentionally backend agnostic.  Call it on the final token
    representations, after a dense embedding, Leviathan, Spelling Bee, or any
    combination of those paths.  Its single vector is saved in the model state
    dict and receives gradients only from MEAP-selected positions.
    """

    def __init__(
        self,
        hidden_size: int,
        mask_token_id: int,
        *,
        initializer_range: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")
        if not isinstance(mask_token_id, int) or mask_token_id < 0:
            raise ValueError("mask_token_id must be a non-negative integer.")
        if not math.isfinite(initializer_range) or initializer_range < 0.0:
            raise ValueError("initializer_range must be finite and non-negative.")

        self.hidden_size = hidden_size
        self.mask_token_id = mask_token_id
        self.initializer_range = float(initializer_range)
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.initializer_range)

    @torch.no_grad()
    def initialize_from(self, embedding: torch.Tensor) -> None:
        """Initialize from one existing *final* token representation.

        This supports a continuous checkpoint migration: compute the old PAD
        representation after every active embedding augmentation, then copy it
        here before changing MEAP to a dedicated reserved token ID.
        """
        if embedding.shape != self.weight.shape:
            raise ValueError(
                f"Expected an embedding with shape {self.weight.shape}, got {embedding.shape}."
            )
        self.weight.copy_(embedding.to(device=self.weight.device, dtype=self.weight.dtype))

    def forward(self, input_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        return apply_meap_embedding_override(
            input_ids,
            embeddings,
            self.weight,
            self.mask_token_id,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, mask_token_id={self.mask_token_id}, "
            f"initializer_range={self.initializer_range}"
        )


@triton.jit
def _fold_seed_to_uint32(seed):
    """Fold every bit of an int64 seed into the uint32 Philox seed."""
    seed_u64 = seed.to(tl.uint64)
    low = seed_u64.to(tl.uint32)
    high = (seed_u64 >> 32).to(tl.uint32)
    high ^= high >> 16
    high *= 0x7FEB352D
    high ^= high >> 15
    high *= 0x846CA68B
    high ^= high >> 16
    return low ^ high


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
    Metrics,
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
    RETURN_METRICS: tl.constexpr,
    SEED_IS_TENSOR: tl.constexpr,
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
    # A device scalar keeps a changing training seed out of Dynamo's Python
    # integer guards. Fold before randint4x so packed int64 step/rank seeds do
    # not silently alias when they differ only above bit 31. Seeds already in
    # the legacy uint32 range remain bit-for-bit unchanged.
    if SEED_IS_TENSOR:
        seed_value = _fold_seed_to_uint32(tl.load(seed))
    else:
        seed_value = seed
    key0, key1, _, _ = tl.randint4x(seed_value, row)
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
    if RETURN_METRICS:
        tl.atomic_add(Metrics, eligible_count)
        tl.atomic_add(Metrics + 1, requested_count)


def _validate_meap_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    eligible_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    seed: int | torch.Tensor,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, sequence), got {input_ids.shape}.")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"input_ids must be int32 or int64, got {input_ids.dtype}.")
    if not isinstance(mask_token_id, int):
        raise TypeError("mask_token_id must be an integer.")
    dtype_limits = torch.iinfo(input_ids.dtype)
    if not 0 <= mask_token_id <= dtype_limits.max:
        raise ValueError(f"mask_token_id must be in [0, {dtype_limits.max}] for {input_ids.dtype}.")
    if not math.isfinite(mask_ratio) or not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be finite and in [0, 1], got {mask_ratio}.")
    if isinstance(seed, torch.Tensor):
        if seed.ndim != 0 or seed.dtype not in (torch.int32, torch.int64):
            raise TypeError("a tensor seed must be a scalar int32 or int64 tensor.")
        if seed.device != input_ids.device:
            raise ValueError("a tensor seed and input_ids must be on the same device.")
    elif not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
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
    seed: int | torch.Tensor,
    exclude_last: bool,
    return_metrics: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
    metrics = (
        torch.stack((eligible_count.sum(), requested_count.sum())).to(torch.int32)
        if return_metrics
        else None
    )
    return input_ids.masked_fill(selected, mask_token_id), selected, metrics


def _triton_meap_mask_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    eligible_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    seed: int,
    exclude_last: bool,
    return_mask: bool,
    return_metrics: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    sequence_length = input_ids.size(1)
    block_t = triton.next_power_of_2(sequence_length)
    if block_t > 4096:
        raise ValueError("The fixed-count Triton MEAP kernel supports sequence lengths up to 4096.")
    output = torch.empty_like(input_ids)
    selected = torch.empty_like(input_ids, dtype=torch.bool) if return_mask else input_ids
    metrics = torch.zeros(2, device=input_ids.device, dtype=torch.int32) if return_metrics else None
    metrics_arg = metrics if metrics is not None else input_ids
    selection_mask = padding_mask if padding_mask is not None else eligible_mask
    eligible_arg = selection_mask if selection_mask is not None else input_ids
    _meap_mask_inputs_kernel[(input_ids.size(0),)](
        input_ids,
        eligible_arg,
        output,
        selected,
        metrics_arg,
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
        RETURN_METRICS=return_metrics,
        SEED_IS_TENSOR=isinstance(seed, torch.Tensor),
        BLOCK_T=block_t,
        num_warps=4 if block_t <= 512 else 8,
        num_stages=1,
    )
    return output, selected, metrics


def meap_mask_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    *,
    enabled: bool = True,
    mask_ratio: float = 0.15,
    eligible_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    seed: int | torch.Tensor = 0,
    exclude_last: bool = True,
    return_mask: bool = False,
    return_metrics: bool = False,
    implementation: str = "triton",
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Corrupt autoregressive inputs with a fixed MEAP mask ratio per sequence.

    ``padding_mask`` is ``True`` where replacement is forbidden and avoids
    materializing its boolean inverse. Alternatively, ``eligible_mask`` is
    ``True`` only where replacement is allowed, including any BOS/EOS or other
    protected-token exclusions. Pass only one. Labels are not accepted
    intentionally: they must remain clean.

    The last eligible input is excluded by default because its hidden state has
    no clean next-token target when CCE is called with ``shift=1``. A positive
    ratio masks at least one token in each row that still has eligible tokens.
    With ``return_metrics=True``, the final tuple item is a two-element device
    tensor ``[eligible_count, masked_count]`` reduced by the masking kernel.
    """
    _validate_meap_inputs(input_ids, mask_token_id, mask_ratio, eligible_mask, padding_mask, seed)
    if implementation not in {"triton", "torch"}:
        raise ValueError(f"Unknown MEAP implementation {implementation!r}.")
    if not enabled:
        if not return_mask and not return_metrics:
            return input_ids
        outputs = [input_ids]
        if return_mask:
            outputs.append(torch.zeros_like(input_ids, dtype=torch.bool))
        if return_metrics:
            outputs.append(torch.zeros(2, device=input_ids.device, dtype=torch.int32))
        return tuple(outputs)

    if input_ids.numel() == 0:
        output = input_ids.clone()
        outputs = [output]
        if return_mask:
            outputs.append(torch.zeros_like(input_ids, dtype=torch.bool))
        if return_metrics:
            outputs.append(torch.zeros(2, device=input_ids.device, dtype=torch.int32))
        return tuple(outputs) if len(outputs) > 1 else output

    if mask_ratio == 0.0 and not return_metrics:
        output = input_ids.clone()
        if return_mask:
            return output, torch.zeros_like(input_ids, dtype=torch.bool)
        return output

    if implementation == "torch":
        if isinstance(seed, torch.Tensor):
            raise TypeError(
                "implementation='torch' requires a Python integer seed; "
                "device-scalar seeds are supported by implementation='triton'."
            )
        output, selected, metrics = _torch_meap_mask_inputs(
            input_ids,
            mask_token_id,
            mask_ratio,
            eligible_mask,
            padding_mask,
            seed,
            exclude_last,
            return_metrics,
        )
    else:
        if not input_ids.is_cuda:
            raise ValueError("The Triton MEAP implementation requires CUDA tensors.")
        output, selected, metrics = _triton_meap_mask_inputs(
            input_ids,
            mask_token_id,
            mask_ratio,
            eligible_mask,
            padding_mask,
            seed,
            exclude_last,
            return_mask,
            return_metrics,
        )

    outputs = [output]
    if return_mask:
        outputs.append(selected)
    if return_metrics:
        assert metrics is not None
        outputs.append(metrics)
    return tuple(outputs) if len(outputs) > 1 else output


__all__ = [
    "MEAPEmbeddingOverride",
    "apply_meap_embedding_override",
    "meap_attention_diagnostics",
    "meap_mask_inputs",
]
