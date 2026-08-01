# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import CCE_LOCK_BLOCK_B, cce_forward_autotune
from cut_cross_entropy.tl_utils import b_bin_fn, tl_logaddexp, tl_softcapping


def _split_v_env_enabled() -> bool:
    """Return whether the optional split-V forward path was requested."""
    return os.getenv("CCE_SPLIT_V", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cce_lse_forward_kernel(
    E,
    C,
    Bias,
    LSE,
    MeanLogit,
    LA,
    NegCorrectLogit,
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
    MODE,
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
    HAS_SOFTCAP: tl.constexpr,
    HAS_MEAN_LOGIT: tl.constexpr,
    HAS_LA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
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
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(tl.int64)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

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
        this_avg_logit = tl.sum(tl.where(valid_rows, logits, 0.0), 0) / B
        tl.atomic_add(LA + offs_v, this_avg_logit, mask=offs_v < V)

    if HAS_TARGETS:
        if HAS_SHIFT:
            target_offs_b = offs_b + shift
        else:
            target_offs_b = offs_b

        this_targets = tl.load(Targets + target_offs_b, mask=target_offs_b < BMax, other=V + 1)

        offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)

        neg_correct_logit_ptrs = NegCorrectLogit + offs_b

        neg_correct_logit_ptrs = tl.broadcast_to(
            neg_correct_logit_ptrs[:, None], (BLOCK_B, BLOCK_V)
        )
        tl.store(neg_correct_logit_ptrs, -logits, mask=this_targets[:, None] == offs_v[None, :])
    else:
        offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)

    this_mx = tl.max(logits, axis=1)
    this_exp = tl.exp(logits - this_mx[:, None])
    this_exp_sum = tl.sum(this_exp, axis=1)
    this_lse = this_mx + tl.log(this_exp_sum)
    if HAS_MEAN_LOGIT:
        finite_logits = tl.where(offs_v[None, :] < V, logits, 0.0)
        this_mean_logit = tl.sum(this_exp * finite_logits, axis=1) / this_exp_sum

    o_mask = offs_b < B

    lse_ptrs = LSE + offs_b
    mean_logit_ptrs = MeanLogit + offs_b if HAS_MEAN_LOGIT else None

    this_locks = Locks + (pid_b * BLOCK_B // LOCK_BLOCK_B)
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    old_lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    new_lse = tl_logaddexp(old_lse, this_lse)
    if HAS_MEAN_LOGIT:
        old_mean_logit = tl.load(
            mean_logit_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last"
        )
        old_weight = tl.exp(old_lse - new_lse)
        this_weight = tl.exp(this_lse - new_lse)
        new_mean_logit = old_weight * old_mean_logit + this_weight * this_mean_logit
        tl.store(
            mean_logit_ptrs,
            new_mean_logit,
            mask=o_mask,
            eviction_policy="evict_last",
        )
    tl.store(lse_ptrs, new_lse, mask=o_mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(this_locks, 0)


_cce_lse_forward_kernel = triton.jit(
    _cce_lse_forward_kernel, do_not_specialize=["MODE", "B_BIN"]
)
_cce_lse_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_MEAN_LOGIT": lambda args: args["MeanLogit"] is not None,
        "HAS_LA": lambda args: args["LA"] is not None,
        "GROUP_B": lambda args: 8,
        # MiLe's entropy statistic is sensitive to the weighted-logit moment.
        # Keep the ordinary CE path on its configured fast precision, but use
        # IEEE products when that additional statistic is requested.
        "DOT_PRECISION": lambda args: "ieee"
        if args["MeanLogit"] is not None
        else (
            "tf32" if torch.get_float32_matmul_precision() == "high" else "ieee"
        ),
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
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
) -> LSEReturn:
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

    reduction = os.getenv("CCE_FORWARD_REDUCTION", "auto")
    if reduction not in {"auto", "lock", "split"}:
        raise ValueError("CCE_FORWARD_REDUCTION must be 'auto', 'lock', or 'split'")
    # Split-V is an opt-in optimization.  Keeping this flag off preserves the
    # historical lock-reduction path even though ``auto`` remains the default
    # value of CCE_FORWARD_REDUCTION.
    split_v_enabled = _split_v_env_enabled()
    split_requested = reduction == "split" or (
        reduction == "auto" and split_v_enabled
    )
    split_config = None
    if split_requested:
        from cut_cross_entropy.cce_lse_forward_split import (
            cce_lse_forward_split,
            clear_split_v_config_cache,
            select_split_v_config,
            use_split_reduction,
        )
        split_config = select_split_v_config(
            e,
            c,
            B,
            return_mean_logit,
            return_logit_avg,
            targets is not None,
            allow_unvalidated=reduction == "split"
            and os.getenv("CCE_SPLIT_V_ALLOW_UNVALIDATED", "0").lower()
            in {"1", "true", "yes", "on"},
        )

    if B > 0 and (
        (
            reduction == "split"
            and split_config is not None
            and split_config.splits > 1
        )
        or (
            reduction == "auto"
            and split_v_enabled
            and split_config is not None
            and use_split_reduction(
                e,
                c,
                B,
                return_mean_logit,
                return_logit_avg,
                targets is not None,
                config=split_config,
            )
        )
    ):

        try:
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
                    config=split_config,
                )
            )
        except torch.cuda.OutOfMemoryError:
            # A game or another process may consume VRAM after the cached
            # policy decision.  Automatic mode must preserve CCE's baseline
            # behavior; an explicit split request remains fail-fast.
            if reduction != "auto":
                raise
            torch.cuda.empty_cache()
            clear_split_v_config_cache()

    # Allocates output.
    lse = e.new_full((B,), -torch.inf, dtype=torch.float32)
    mean_logit = e.new_zeros((B,), dtype=torch.float32) if return_mean_logit else None
    neg_correct_logit = e.new_full((B,), 0.0, dtype=torch.float32) if targets is not None else None
    assert lse.stride(0) == 1

    locks = e.new_full(
        (triton.cdiv(B, CCE_LOCK_BLOCK_B),),
        0,
        dtype=torch.uint32,
    )

    if return_logit_avg:
        logit_avg = e.new_full((V,), 0.0, dtype=torch.float32)
    else:
        logit_avg = None

    # 1D launch kernel where each block gets its own program.
    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _cce_lse_forward_kernel[grid](
        e,
        c,
        bias,
        lse,
        mean_logit,
        logit_avg,
        neg_correct_logit,
        locks,
        valids,
        targets,
        softcap,
        shift,
        B,
        V,
        D,  #
        e.size(0),
        e.stride(0),
        e.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        1 if bias is None else bias.stride(0),
        1 if valids is None else valids.stride(0),
        # Normalize optional features into cost families. Individual constexpr
        # branches still compile independently, but equivalent modes share one
        # autotune result instead of fragmenting the timing cache.
        MODE=(
            (bias is not None or valids is not None or targets is not None or shift != 0)
            | ((softcap is not None) << 1)
            | (return_mean_logit << 2)
            | (return_logit_avg << 3)
        ),
        B_BIN=b_bin_fn(B),
        LOCK_BLOCK_B=CCE_LOCK_BLOCK_B,
    )

    return LSEReturn(lse, logit_avg, neg_correct_logit, mean_logit)

