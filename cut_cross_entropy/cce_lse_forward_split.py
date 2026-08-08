# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Two-stage CCE forward reduction without cross-program spinlocks."""

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import CCE_LOCK_BLOCK_B
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


_SPLIT_V_TARGET_PROGRAMS_PER_SM = 2
_SPLIT_V_MEMORY_MULTIPLIER = 2
_SPLIT_V_BLOCK_D = 32
_SPLIT_V_DEFAULT_MAX_SPLITS = 64
_SPLIT_V_CC10_MAX_SPLITS = 32
_SPLIT_V_AUTO_MAX_B = 512
_SPLIT_V_CONFIG_CACHE_LIMIT = 128
_SPLIT_V_CONFIG_CACHE: dict[tuple[object, ...], "SplitVConfig"] = {}


@dataclass(frozen=True)
class SplitVConfig:
    """One analytically selected split-V launch configuration.

    The selector evaluates at most three static tile candidates.  It never
    launches a candidate, calls Triton's autotuner, or benchmarks the lock and
    split graphs while selecting a configuration.
    """

    block_b: int
    block_v: int
    block_d: int
    splits: int
    num_warps: int
    num_stages: int
    reduce_block_b: int
    reduce_num_warps: int
    score: float
    base_memory_bytes: int
    split_memory_bytes: int


@dataclass(frozen=True)
class _SplitVArchitectureProfile:
    max_splits: int
    validated: bool
    min_chunks_small: int = 16
    min_chunks_large: int = 8
    max_programs_per_sm: float = 2.0


def split_v_workspace_bytes(
    b: int, splits: int, return_mean_logit: bool = False
) -> int:
    """Bytes used by the FP32 LSE/weighted-logit partials."""
    return 4 * b * splits * (1 + int(return_mean_logit))


def _output_state_bytes(
    b: int,
    v: int,
    return_mean_logit: bool,
    return_logit_avg: bool,
    has_targets: bool,
) -> int:
    return 4 * b * (1 + int(return_mean_logit) + int(has_targets)) + 4 * v * int(
        return_logit_avg
    )


def _base_forward_memory_bytes(
    e: torch.Tensor,
    c: torch.Tensor,
    b: int,
    return_mean_logit: bool,
    return_logit_avg: bool,
    has_targets: bool,
) -> int:
    """Estimate the live lock-path footprint of the CCE forward."""
    output_bytes = _output_state_bytes(
        b, c.size(0), return_mean_logit, return_logit_avg, has_targets
    )
    lock_bytes = 4 * triton.cdiv(b, CCE_LOCK_BLOCK_B)
    return (
        e.numel() * e.element_size()
        + c.numel() * c.element_size()
        + output_bytes
        + lock_bytes
    )


def _split_forward_memory_bytes(
    base_memory_bytes: int,
    b: int,
    splits: int,
    return_mean_logit: bool,
) -> int:
    """Replace lock words with the split-V FP32 partial workspace."""
    lock_bytes = 4 * triton.cdiv(b, CCE_LOCK_BLOCK_B)
    return base_memory_bytes - lock_bytes + split_v_workspace_bytes(
        b, splits, return_mean_logit
    )


def _floor_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value.bit_length() - 1)


def _split_v_profile(
    device: torch.device,
    allow_unvalidated: bool = False,
) -> _SplitVArchitectureProfile | None:
    """Return a bounded profile; unknown architectures remain lock-only.

    CC12.x is the only profile measured in this workspace.  CC10.x (the
    Blackwell data-center family) has a conservative compatibility profile but
    is explicitly unvalidated here.  An explicit split request may opt into an
    unvalidated profile with ``CCE_SPLIT_V_ALLOW_UNVALIDATED=1``; automatic
    dispatch never does so.
    """
    major, _minor = torch.cuda.get_device_capability(device)
    if major == 12:
        return _SplitVArchitectureProfile(_SPLIT_V_DEFAULT_MAX_SPLITS, True)
    if major == 10 and allow_unvalidated:
        return _SplitVArchitectureProfile(
            _SPLIT_V_CC10_MAX_SPLITS,
            False,
            min_chunks_small=16,
            min_chunks_large=16,
            max_programs_per_sm=1.5,
        )
    return None


def _split_count_for_tile(
    b: int,
    v: int,
    block_b: int,
    block_v: int,
    sms: int,
    max_splits: int,
) -> int:
    """Choose a power-of-two S from occupancy and hard shape caps."""
    if b <= 0 or v <= 0:
        return 1
    b_tiles = triton.cdiv(b, block_b)
    v_tiles = triton.cdiv(v, block_v)
    target = max(1, triton.cdiv(_SPLIT_V_TARGET_PROGRAMS_PER_SM * sms, b_tiles))
    desired = triton.next_power_of_2(target)
    cap = min(v_tiles, max(1, max_splits))
    return min(desired, _floor_power_of_two(cap))


def split_count(
    b: int, v: int, block_b: int, block_v: int, device: torch.device
) -> int:
    """Compatibility helper using the conservative CC12 profile cap."""
    profile = _split_v_profile(device, allow_unvalidated=True)
    if profile is None:
        return 1
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return _split_count_for_tile(
        b, v, block_b, block_v, sms, profile.max_splits
    )


def _split_v_tile_candidates(
    e: torch.Tensor,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Small static candidate set; discarded candidates are never compiled."""
    if e.dtype == torch.float32:
        return ((32, 128, _SPLIT_V_BLOCK_D, 4, 3),)
    return (
        (64, 128, _SPLIT_V_BLOCK_D, 4, 4),
        (128, 128, _SPLIT_V_BLOCK_D, 4, 4),
        (128, 64, _SPLIT_V_BLOCK_D, 4, 4),
    )


def _shape_split_cap(
    v_tiles: int,
    d: int,
    profile: _SplitVArchitectureProfile,
) -> int:
    """Avoid S so large that each program performs only a tiny loop."""
    min_chunks = (
        profile.min_chunks_large
        if d >= 256 and v_tiles >= 256
        else profile.min_chunks_small
    )
    raw = max(1, v_tiles // min_chunks)
    return min(v_tiles, _floor_power_of_two(max(16, raw)))


def _candidate_cost(
    b: int,
    v: int,
    d: int,
    sms: int,
    block_b: int,
    block_v: int,
    splits: int,
    profile: _SplitVArchitectureProfile,
) -> float:
    """Analytic cost for the joint (tile,S) decision.

    It charges padded tile work, underfill, oversubscription, split reduction,
    and edge waste.  The coefficients are deliberately fixed by the
    architecture profile; no runtime search or global fitted model is used.
    """
    n_b = triton.cdiv(b, block_b)
    n_v = triton.cdiv(v, block_v)
    programs = n_b * splits
    chunks = triton.cdiv(n_v, splits)
    visits = programs * chunks
    logical_tiles = max(1, b * v)
    padded_work = visits * block_b * block_v / logical_tiles
    edge_efficiency = logical_tiles / max(1, visits * block_b * block_v)
    occupancy = min(1.0, programs / max(1, sms))
    pressure = programs / max(1, sms)
    # The 128x128 tile amortizes instruction/launch overhead on the large-D,
    # large-vocabulary regime.  This is a bounded architecture prior, not a
    # fitted runtime benchmark; smaller-D shapes retain the 64x128 candidate.
    tile_efficiency = (
        1.15 if block_b >= 128 and block_v >= 128 and d >= 256 and n_v >= 256 else 1.0
    )
    return (
        padded_work / (tile_efficiency * max(occupancy, 0.25))
        + 0.02 * pressure
        + 0.02 * splits / max(1, chunks)
        + 0.10 * max(0.0, 1.0 - occupancy)
        + 0.02 * max(0.0, pressure - 1.0)
        + 0.05 * (1.0 - edge_efficiency)
    )


def _reduce_launch_config(b: int) -> tuple[int, int]:
    if b <= 32:
        return 32, 2
    if b <= 64:
        return 64, 2
    if b <= 128:
        return 128, 4
    return 256, 4


def _free_memory_guard(
    e: torch.Tensor,
    c: torch.Tensor,
    split_memory_bytes: int,
) -> bool:
    """Reject a safe-relative split if external VRAM leaves no headroom."""
    if os.getenv("CCE_SPLIT_V_FREE_GUARD", "1").lower() in {"0", "false", "off"}:
        return True
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(e.device)
    except (AttributeError, RuntimeError):
        return True
    input_bytes = e.numel() * e.element_size() + c.numel() * c.element_size()
    required = max(0, split_memory_bytes - input_bytes)
    safety = float(os.getenv("CCE_SPLIT_V_FREE_FRACTION", "0.90"))
    if not 0.0 < safety <= 1.0:
        raise ValueError("CCE_SPLIT_V_FREE_FRACTION must be in (0, 1]")
    return required <= safety * free_bytes


def clear_split_v_config_cache() -> None:
    """Clear the bounded host-side policy cache (useful for experiments)."""
    _SPLIT_V_CONFIG_CACHE.clear()


def _split_v_config_cache_key(
    e: torch.Tensor,
    c: torch.Tensor,
    b: int,
    return_mean_logit: bool,
    return_logit_avg: bool,
    has_targets: bool,
    allow_unvalidated: bool,
) -> tuple[object, ...]:
    device_index = e.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(e.device)
    properties = torch.cuda.get_device_properties(e.device)
    return (
        device_index,
        capability,
        properties.multi_processor_count,
        properties.shared_memory_per_block,
        str(e.dtype),
        str(c.dtype),
        e.size(0),
        b,
        c.size(0),
        c.size(1),
        e.element_size(),
        c.element_size(),
        return_mean_logit,
        return_logit_avg,
        has_targets,
        allow_unvalidated,
    )


def _cache_split_v_config(
    key: tuple[object, ...], config: SplitVConfig
) -> None:
    _SPLIT_V_CONFIG_CACHE[key] = config
    while len(_SPLIT_V_CONFIG_CACHE) > _SPLIT_V_CONFIG_CACHE_LIMIT:
        _SPLIT_V_CONFIG_CACHE.pop(next(iter(_SPLIT_V_CONFIG_CACHE)))


def select_split_v_config(
    e: torch.Tensor,
    c: torch.Tensor,
    b: int,
    return_mean_logit: bool = False,
    return_logit_avg: bool = False,
    has_targets: bool = False,
    allow_unvalidated: bool = False,
) -> SplitVConfig:
    """Select one joint tile/S configuration with bounded arithmetic only."""
    key = _split_v_config_cache_key(
        e,
        c,
        b,
        return_mean_logit,
        return_logit_avg,
        has_targets,
        allow_unvalidated,
    )
    cached = _SPLIT_V_CONFIG_CACHE.get(key)
    refresh_free_guard = os.getenv("CCE_SPLIT_V_FREE_GUARD_REFRESH", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if cached is not None and not refresh_free_guard:
        return cached
    profile = _split_v_profile(e.device, allow_unvalidated=allow_unvalidated)
    base_memory_bytes = _base_forward_memory_bytes(
        e, c, b, return_mean_logit, return_logit_avg, has_targets
    )
    reduce_block_b, reduce_num_warps = _reduce_launch_config(b)
    candidates: list[SplitVConfig] = []
    free_memory_rejected = False
    if profile is not None:
        properties = torch.cuda.get_device_properties(e.device)
        partial_bytes = max(1, split_v_workspace_bytes(b, 1, return_mean_logit))
        memory_cap = max(1, base_memory_bytes // partial_bytes)
        for block_b, block_v, block_d, warps, stages in _split_v_tile_candidates(e):
            v_tiles = triton.cdiv(c.size(0), block_v)
            shape_cap = _shape_split_cap(v_tiles, c.size(1), profile)
            max_splits = min(profile.max_splits, v_tiles, memory_cap, shape_cap)
            splits = _split_count_for_tile(
                b,
                c.size(0),
                block_b,
                block_v,
                properties.multi_processor_count,
                max_splits=max(1, max_splits),
            )
            split_memory_bytes = _split_forward_memory_bytes(
                base_memory_bytes, b, splits, return_mean_logit
            )
            if split_memory_bytes > _SPLIT_V_MEMORY_MULTIPLIER * base_memory_bytes:
                continue
            if refresh_free_guard or cached is None:
                if not _free_memory_guard(e, c, split_memory_bytes):
                    free_memory_rejected = True
                    continue
            candidates.append(
                SplitVConfig(
                    block_b,
                    block_v,
                    block_d,
                    splits,
                    warps,
                    stages,
                    reduce_block_b,
                    reduce_num_warps,
                    -_candidate_cost(
                        b,
                        c.size(0),
                        c.size(1),
                        properties.multi_processor_count,
                        block_b,
                        block_v,
                        splits,
                        profile,
                    ),
                    base_memory_bytes,
                    split_memory_bytes,
                )
            )
    if candidates:
        selected = max(candidates, key=lambda candidate: candidate.score)
        _cache_split_v_config(key, selected)
        return selected

    # A S=1 config is a safe sentinel. The CCE wrapper treats it as a lock
    # fallback, so unsupported devices never launch a one-way staged reduction.
    block_b, block_v, block_d, warps, stages = _split_v_tile_candidates(e)[0]
    selected = SplitVConfig(
        block_b,
        block_v,
        block_d,
        1,
        warps,
        stages,
        reduce_block_b,
        reduce_num_warps,
        0.0,
        base_memory_bytes,
        _split_forward_memory_bytes(base_memory_bytes, b, 1, return_mean_logit),
    )
    if not free_memory_rejected:
        _cache_split_v_config(key, selected)
    return selected


def use_split_reduction(
    e: torch.Tensor,
    c: torch.Tensor,
    b: int,
    return_mean_logit: bool,
    return_logit_avg: bool = False,
    has_targets: bool = False,
    config: SplitVConfig | None = None,
) -> bool:
    """Automatic policy; unvalidated devices and large batches stay lock-only."""
    if b == 0 or return_mean_logit or e.dtype == torch.float32 or b > _SPLIT_V_AUTO_MAX_B:
        return False
    if _split_v_profile(e.device, allow_unvalidated=False) is None:
        return False
    if config is None:
        config = select_split_v_config(
            e, c, b, return_mean_logit, return_logit_avg, has_targets
        )
    if config.splits <= 1:
        return False
    properties = torch.cuda.get_device_properties(e.device)
    programs_per_sm = triton.cdiv(b, config.block_b) * config.splits / max(
        1, properties.multi_processor_count
    )
    chunks = triton.cdiv(
        triton.cdiv(c.size(0), config.block_v), config.splits
    )
    profile = _split_v_profile(e.device)
    assert profile is not None
    # A high CTA pressure with a short vocab loop is a known regression regime.
    return not (
        programs_per_sm > profile.max_programs_per_sm
        and chunks <= profile.min_chunks_small
    )


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
    config: SplitVConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if valids is not None:
        b = valids.numel()
    else:
        b = e.size(0)
    v, d = c.shape

    if config is None:
        config = select_split_v_config(
            e,
            c,
            b,
            return_mean_logit,
            return_logit_avg,
            targets is not None,
            allow_unvalidated=True,
        )
    block_b = config.block_b
    block_v = config.block_v
    block_d = config.block_d
    splits = config.splits
    num_warps = config.num_warps
    num_stages = config.num_stages
    reduce_block_b = config.reduce_block_b
    reduce_num_warps = config.reduce_num_warps

    if splits <= 1:
        raise ValueError("split-V requires at least two vocabulary splits")

    # These tensors are linearized in the reduction kernels.  Failing loudly
    # here is safer than silently treating a future non-contiguous view as a
    # contiguous vector.
    if valids is not None and valids.stride(0) != 1:
        raise ValueError("split-V valids tensor must be contiguous")
    if targets is not None and targets.stride(0) != 1:
        raise ValueError("split-V targets tensor must be contiguous")

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
    _cce_lse_split_reduce_kernel[(triton.cdiv(b, reduce_block_b),)](
        partial_lse,
        partial_mean,
        lse,
        mean_logit,
        b,
        BLOCK_B=reduce_block_b,
        SPLITS=splits,
        HAS_MEAN_LOGIT=return_mean_logit,
        num_warps=reduce_num_warps,
    )
    return lse, logit_avg, neg_correct_logit, mean_logit

