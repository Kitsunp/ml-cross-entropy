"""Compiler-safe LEV boundary.

The raw Triton launch contains Python dispatch and tensor indexing that should
not be traced by TorchDynamo.  This module mirrors the CCE compiler boundary:
the CUDA custom op owns the launch, saves the lean LEV checkpoints, and calls a
second opaque custom op for the backward.  If the Triton implementation is
missing or the configuration is unsupported, the backend executes the verified
reference implementation instead. Runtime launch failures are propagated rather
than followed by reference work on the same CUDA stream; this is important when
the caller is using CUDA Graph Trees.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from cut_cross_entropy.torch_2_14 import (
    TORCH_2_14_CUDA_KERNEL_CONTEXT,
    TORCH_2_14_MEMORY_ANNOTATIONS,
    annotate_tensors,
    cuda_kernel_region,
)

from .backward_impl import (
    base_k_decompose,
    leviathan_backward,
    leviathan_forward_ref,
)
from .core import LeviathanConfig

try:
    from .forward_impl import leviathan_forward as _leviathan_forward
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional Triton
    _leviathan_forward = None

try:
    from .backward_kernels import (
        leviathan_backward_triton as _leviathan_backward_triton,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional Triton
    _leviathan_backward_triton = None


def _make_config(
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
    dtype: torch.dtype,
) -> LeviathanConfig:
    return LeviathanConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        generator_d_seed=d_seed,
        generator_num_modes=num_modes,
        generator_num_knots=num_knots,
        generator_spline_degree=spline_degree,
        generator_k=generator_k,
        generator_krank=krank,
        dtype=dtype,
    )


def _saved_or_reference(
    ids: torch.Tensor,
    params: dict[str, torch.Tensor],
    cfg: LeviathanConfig,
    *,
    save_intermediates: bool = True,
    return_seed: bool = False,
    fuse_gather: bool | None = None,
    mask_embedding: torch.Tensor | None = None,
    mask_token_id: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run LEV and optionally return the lean tensors required for backward."""
    if _leviathan_forward is not None:
        try:
            with torch.no_grad():
                embeds, saved = _leviathan_forward(
                    ids,
                    params,
                    cfg,
                    save_intermediates=save_intermediates,
                    return_seed=return_seed,
                    fuse_gather=fuse_gather,
                    mask_embedding=mask_embedding,
                    mask_token_id=mask_token_id,
                )
            if not save_intermediates and not return_seed:
                return embeds, {}
            if saved is not None:
                return embeds, saved
        except (TypeError, ValueError, AttributeError):
            # The reference path is the semantic fallback for unsupported
            # metadata/configurations. Do not catch CUDA launch failures here:
            # executing the reference after a failed launch can invalidate the
            # surrounding CUDA-graph/FlashAttention partition.
            pass

    with torch.no_grad():
        embeds, saved = leviathan_forward_ref(
            ids,
            params,
            cfg,
            save_intermediates=save_intermediates or return_seed,
        )
        if mask_embedding is not None:
            positions = ids.eq(int(mask_token_id)).unsqueeze(-1)
            embeds = torch.where(
                positions,
                mask_embedding.to(device=embeds.device, dtype=embeds.dtype),
                embeds,
            )
    if not save_intermediates and not return_seed:
        return embeds, {}
    if saved is None:  # pragma: no cover - the reference always saves here
        raise RuntimeError("Leviathan reference forward returned no checkpoints")
    return embeds, saved


def _leviathan_forward_impl(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
    *,
    return_seed: bool = False,
    fuse_gather: bool | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    params = {
        "codebooks": codebooks.detach(),
        "head_proj_weight": head_proj_weight.detach(),
        "head_norm_weight": head_norm_weight.detach(),
        "head_norm_bias": head_norm_bias.detach(),
        "head_spline_delta": head_spline_delta.detach(),
        "head_out_weight": head_out_weight.detach(),
        "knot_grid": knot_grid.detach(),
    }
    cfg = _make_config(
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
        codebooks.dtype,
    )
    has_meap = mask_embedding.numel() != 0
    forward_kwargs = {
        "return_seed": return_seed,
        "fuse_gather": fuse_gather,
        "mask_embedding": mask_embedding.detach() if has_meap else None,
        "mask_token_id": mask_token_id if has_meap else None,
    }
    if TORCH_2_14_CUDA_KERNEL_CONTEXT:
        with cuda_kernel_region("leviathan.forward", ids.device):
            embeds, saved = _saved_or_reference(
                ids.detach(),
                params,
                cfg,
                **forward_kwargs,
            )
    else:
        embeds, saved = _saved_or_reference(
            ids.detach(),
            params,
            cfg,
            **forward_kwargs,
        )

    z = saved["z"].contiguous()
    xhat = saved.get("x_hat_por_head")
    mean = saved.get("mean_por_head")
    rsqrt = saved.get("rsqrt_por_head")
    modes = saved.get("modes_por_head")
    if xhat is None:
        # Inference-with-seed uses this helper only to obtain z and does not
        # expose checkpoints to autograd.  Keep the tuple shape stable for
        # the fake/custom-op boundary without allocating a second checkpoint.
        xhat = codebooks.new_empty((num_modes, ids.numel(), d_seed))
        mean = torch.empty(
            (num_modes, ids.numel(), 1),
            dtype=torch.float32,
            device=codebooks.device,
        )
        rsqrt = torch.empty_like(mean)
    else:
        xhat = xhat.contiguous()
        mean = mean.contiguous()
        rsqrt = rsqrt.contiguous()
    has_modes = modes is not None
    if modes is None:
        # Keep the output metadata stable when a supported CUDA config falls
        # back at runtime.  The flag tells the backward op not to consume this
        # uninitialized placeholder.
        modes = codebooks.new_empty((ids.numel(), num_modes, krank))
    else:
        modes = modes.contiguous()
    mode_flag = codebooks.new_tensor(1 if has_modes else 0, dtype=torch.int8)
    if TORCH_2_14_MEMORY_ANNOTATIONS:
        annotate_tensors(
            "leviathan.forward",
            embeds=embeds,
            z=z,
            xhat=xhat,
            mean=mean,
            rsqrt=rsqrt,
            modes=modes,
            mode_flag=mode_flag,
        )
    return embeds, z, xhat, mean, rsqrt, modes, mode_flag


@torch.library.custom_op(
    "cut_cross_entropy::leviathan_forward_with_seed",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _leviathan_forward_with_seed_op(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Training boundary that exposes the kernel-produced seed to JTok.

    The forward implementation is shared with the legacy op.  The only
    policy difference is that it requests ``return_seed`` and the safe fused
    gather, so the returned ``z`` is the same stage-one result used by
    Leviathan itself.  Its registered backward combines the Leviathan output
    gradient and JTok's seed gradient before the codebook scatter.
    """
    return _leviathan_forward_impl(
        ids,
        codebooks,
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        mask_token_id,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
        return_seed=True,
        fuse_gather=True,
    )


@_leviathan_forward_with_seed_op.register_fake
def _leviathan_forward_with_seed_fake(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _leviathan_forward_fake(
        ids,
        codebooks,
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        mask_token_id,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    )


@torch.library.custom_op(
    "cut_cross_entropy::leviathan_forward",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _leviathan_forward_op(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _leviathan_forward_impl(
        ids,
        codebooks,
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        mask_token_id,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    )


@_leviathan_forward_op.register_fake
def _leviathan_forward_fake(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del (
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        mask_token_id,
        vocab_size,
        spline_degree,
        generator_k,
        num_knots,
    )
    n_tokens = ids.numel()
    xhat_dtype = torch.float16 if os.environ.get("LEV_SAVE_XH_FP16", "0") != "0" else torch.float32
    embeds = codebooks.new_empty((*ids.shape, hidden_size))
    z = codebooks.new_empty((n_tokens, d_seed))
    xhat = torch.empty(
        (num_modes, n_tokens, d_seed),
        dtype=xhat_dtype,
        device=codebooks.device,
    )
    mean = torch.empty(
        (num_modes, n_tokens, 1),
        dtype=torch.float32,
        device=codebooks.device,
    )
    rsqrt = torch.empty_like(mean)
    modes = codebooks.new_empty((n_tokens, num_modes, krank))
    mode_flag = torch.empty((), dtype=torch.int8, device=codebooks.device)
    return embeds, z, xhat, mean, rsqrt, modes, mode_flag


@torch.library.custom_op(
    "cut_cross_entropy::leviathan_inference",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _leviathan_inference_op(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> torch.Tensor:
    """CUDA inference boundary with no backward checkpoints."""
    params = {
        "codebooks": codebooks.detach(),
        "head_proj_weight": head_proj_weight.detach(),
        "head_norm_weight": head_norm_weight.detach(),
        "head_norm_bias": head_norm_bias.detach(),
        "head_spline_delta": head_spline_delta.detach(),
        "head_out_weight": head_out_weight.detach(),
        "knot_grid": knot_grid.detach(),
    }
    cfg = _make_config(
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
        codebooks.dtype,
    )
    has_meap = mask_embedding.numel() != 0
    if TORCH_2_14_CUDA_KERNEL_CONTEXT:
        with cuda_kernel_region("leviathan.inference", ids.device):
            embeds, _ = _saved_or_reference(
                ids.detach(),
                params,
                cfg,
                save_intermediates=False,
                mask_embedding=(mask_embedding.detach() if has_meap else None),
                mask_token_id=(mask_token_id if has_meap else None),
            )
    else:
        embeds, _ = _saved_or_reference(
            ids.detach(),
            params,
            cfg,
            save_intermediates=False,
            mask_embedding=(mask_embedding.detach() if has_meap else None),
            mask_token_id=(mask_token_id if has_meap else None),
        )
    if TORCH_2_14_MEMORY_ANNOTATIONS:
        annotate_tensors("leviathan.inference", embeds=embeds)
    return embeds


@_leviathan_inference_op.register_fake
def _leviathan_inference_fake(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> torch.Tensor:
    del (
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        mask_token_id,
        vocab_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    )
    return codebooks.new_empty((*ids.shape, hidden_size))


@torch.library.custom_op(
    "cut_cross_entropy::leviathan_inference_with_seed",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _leviathan_inference_with_seed_op(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference boundary returning embedding plus the kernel-produced seed."""
    params = {
        "codebooks": codebooks.detach(),
        "head_proj_weight": head_proj_weight.detach(),
        "head_norm_weight": head_norm_weight.detach(),
        "head_norm_bias": head_norm_bias.detach(),
        "head_spline_delta": head_spline_delta.detach(),
        "head_out_weight": head_out_weight.detach(),
        "knot_grid": knot_grid.detach(),
    }
    cfg = _make_config(
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
        codebooks.dtype,
    )
    has_meap = mask_embedding.numel() != 0
    forward_kwargs = {
        "save_intermediates": False,
        "return_seed": True,
        "fuse_gather": True,
        "mask_embedding": mask_embedding.detach() if has_meap else None,
        "mask_token_id": mask_token_id if has_meap else None,
    }
    if TORCH_2_14_CUDA_KERNEL_CONTEXT:
        with cuda_kernel_region("leviathan.inference", ids.device):
            embeds, saved = _saved_or_reference(
                ids.detach(), params, cfg, **forward_kwargs
            )
    else:
        embeds, saved = _saved_or_reference(
            ids.detach(), params, cfg, **forward_kwargs
        )
    seed = saved["z"].contiguous()
    if TORCH_2_14_MEMORY_ANNOTATIONS:
        annotate_tensors("leviathan.inference", embeds=embeds, z=seed)
    return embeds, seed


@_leviathan_inference_with_seed_op.register_fake
def _leviathan_inference_with_seed_fake(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        mask_token_id,
        vocab_size,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    )
    return (
        codebooks.new_empty((*ids.shape, hidden_size)),
        codebooks.new_empty((ids.numel(), d_seed)),
    )


def _compute_leviathan_grads(
    grad_out: torch.Tensor,
    params: dict[str, torch.Tensor],
    cfg: LeviathanConfig,
    saved: dict[str, Any],
    ids: torch.Tensor,
    has_modes: bool,
    seed_grad: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    grads = None
    if has_modes and _leviathan_backward_triton is not None:
        try:
            grads = _leviathan_backward_triton(
                grad_out,
                params,
                cfg,
                saved,
                ids,
                seed_grad=seed_grad,
            )
        except (TypeError, ValueError, AttributeError):
            grads = None
    if grads is None:
        # Keep the compiler-boundary fallback bounded just like the regular
        # autograd wrapper. This path is used when the Triton backward is
        # unavailable or rejects metadata; a long sequence must not make the
        # reference _head_backward materialize its full-N basis/phi workset.
        chunk = getattr(cfg, "backward_chunk", None) or 8192
        grads = leviathan_backward(
            grad_out,
            params,
            cfg,
            saved=saved,
            ids=ids,
            chunk=chunk,
            seed_grad=seed_grad,
        )
    return grads


@torch.library.custom_op(
    "cut_cross_entropy::leviathan_backward",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _leviathan_backward_op(
    grad_out: torch.Tensor,
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    z: torch.Tensor,
    xhat: torch.Tensor,
    mean: torch.Tensor,
    rsqrt: torch.Tensor,
    modes: torch.Tensor,
    modes_available: torch.Tensor,
    seed_grad: torch.Tensor,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    params = {
        "codebooks": codebooks,
        "head_proj_weight": head_proj_weight,
        "head_norm_weight": head_norm_weight,
        "head_norm_bias": head_norm_bias,
        "head_spline_delta": head_spline_delta,
        "head_out_weight": head_out_weight,
        "knot_grid": knot_grid,
    }
    cfg = _make_config(
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
        codebooks.dtype,
    )
    saved: dict[str, Any] = {
        "z": z,
        "x_hat_por_head": xhat,
        "mean_por_head": mean,
        "rsqrt_por_head": rsqrt,
        "knot_grid": knot_grid,
    }
    has_modes = bool(modes_available.item())
    if has_modes:
        saved["modes_por_head"] = modes

    if TORCH_2_14_CUDA_KERNEL_CONTEXT:
        with cuda_kernel_region("leviathan.backward", ids.device):
            grads = _compute_leviathan_grads(
                grad_out, params, cfg, saved, ids, has_modes,
                seed_grad=seed_grad,
            )
    else:
        grads = _compute_leviathan_grads(
            grad_out, params, cfg, saved, ids, has_modes,
            seed_grad=seed_grad,
        )
    if TORCH_2_14_MEMORY_ANNOTATIONS:
        annotate_tensors("leviathan.backward", **grads)
    return tuple(
        grads[key]
        for key in (
            "codebooks",
            "head_proj_weight",
            "head_norm_weight",
            "head_norm_bias",
            "head_spline_delta",
            "head_out_weight",
        )
    )


@_leviathan_backward_op.register_fake
def _leviathan_backward_fake(
    grad_out: torch.Tensor,
    ids: torch.Tensor,
    codebooks: torch.Tensor,
    head_proj_weight: torch.Tensor,
    head_norm_weight: torch.Tensor,
    head_norm_bias: torch.Tensor,
    head_spline_delta: torch.Tensor,
    head_out_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    z: torch.Tensor,
    xhat: torch.Tensor,
    mean: torch.Tensor,
    rsqrt: torch.Tensor,
    modes: torch.Tensor,
    modes_available: torch.Tensor,
    seed_grad: torch.Tensor,
    vocab_size: int,
    hidden_size: int,
    d_seed: int,
    num_modes: int,
    num_knots: int,
    spline_degree: int,
    generator_k: int,
    krank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del (
        grad_out,
        ids,
        knot_grid,
        z,
        xhat,
        mean,
        rsqrt,
        modes,
        modes_available,
        seed_grad,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    )
    return (
        torch.empty_like(codebooks),
        torch.empty_like(head_proj_weight),
        torch.empty_like(head_norm_weight),
        torch.empty_like(head_norm_bias),
        torch.empty_like(head_spline_delta),
        torch.empty_like(head_out_weight),
    )


def _leviathan_setup_context(ctx, inputs, output) -> None:
    (
        ids,
        codebooks,
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        mask_token_id,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    ) = inputs
    _embeds, z, xhat, mean, rsqrt, modes, mode_flag = output
    ctx.save_for_backward(
        ids,
        codebooks,
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        z,
        xhat,
        mean,
        rsqrt,
        modes,
        mode_flag,
    )
    ctx.config_values = (
        mask_token_id,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    )
    ctx.mark_non_differentiable(z, xhat, mean, rsqrt, modes, mode_flag)


def _leviathan_backward_impl(
    ctx,
    grad_out: torch.Tensor,
    seed_grad: torch.Tensor | None = None,
):
    (
        ids,
        codebooks,
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        z,
        xhat,
        mean,
        rsqrt,
        modes,
        mode_flag,
    ) = ctx.saved_tensors
    (
        mask_token_id,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    ) = ctx.config_values
    has_meap = mask_embedding.numel() != 0
    needs_mask_grad = ctx.needs_input_grad[8]
    needs_leviathan_grad = any(ctx.needs_input_grad[1:7])
    if has_meap:
        meap_positions = ids.eq(mask_token_id).unsqueeze(-1)
        grad_mask = (
            torch.where(
                meap_positions,
                grad_out,
                torch.zeros((), dtype=grad_out.dtype, device=grad_out.device),
            )
            .reshape(-1, hidden_size)
            .float()
            .sum(dim=0)
            .to(mask_embedding.dtype)
            if needs_mask_grad
            else None
        )
        grad_out_leviathan = torch.where(
            meap_positions,
            torch.zeros((), dtype=grad_out.dtype, device=grad_out.device),
            grad_out,
        )
    else:
        grad_mask = None
        grad_out_leviathan = grad_out

    if needs_leviathan_grad:
        d_codebooks, d_proj, d_norm_w, d_norm_b, d_delta, d_out = _leviathan_backward_op(
            grad_out_leviathan,
            ids,
            codebooks,
            head_proj_weight,
            head_norm_weight,
            head_norm_bias,
            head_spline_delta,
            head_out_weight,
            knot_grid,
            z,
            xhat,
            mean,
            rsqrt,
            modes,
            mode_flag,
            codebooks.new_empty(0) if seed_grad is None else seed_grad,
            vocab_size,
            hidden_size,
            d_seed,
            num_modes,
            num_knots,
            spline_degree,
            generator_k,
            krank,
        )
    else:
        d_codebooks = d_proj = d_norm_w = d_norm_b = d_delta = d_out = None
    return (
        None,
        d_codebooks,
        d_proj,
        d_norm_w,
        d_norm_b,
        d_delta,
        d_out,
        None,
        grad_mask,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _leviathan_backward(ctx, *grads):
    """Backward for the legacy embedding-only custom op."""
    return _leviathan_backward_impl(ctx, grads[0])


torch.library.register_autograd(
    _leviathan_forward_op,
    _leviathan_backward,
    setup_context=_leviathan_setup_context,
)


def _leviathan_with_seed_setup_context(ctx, inputs, output) -> None:
    """Save the same checkpoints while keeping the seed differentiable."""
    (
        ids,
        codebooks,
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        mask_token_id,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    ) = inputs
    _embeds, z, xhat, mean, rsqrt, modes, mode_flag = output
    ctx.save_for_backward(
        ids,
        codebooks,
        head_proj_weight,
        head_norm_weight,
        head_norm_bias,
        head_spline_delta,
        head_out_weight,
        knot_grid,
        mask_embedding,
        z,
        xhat,
        mean,
        rsqrt,
        modes,
        mode_flag,
    )
    ctx.config_values = (
        mask_token_id,
        vocab_size,
        hidden_size,
        d_seed,
        num_modes,
        num_knots,
        spline_degree,
        generator_k,
        krank,
    )
    # z is intentionally omitted: JTok/JTok-M must receive its gradient.
    ctx.mark_non_differentiable(xhat, mean, rsqrt, modes, mode_flag)


def _leviathan_with_seed_backward(ctx, *grads):
    """Merge d(embedding) and d(seed) before Leviathan's codebook scatter."""
    return _leviathan_backward_impl(ctx, grads[0], grads[1])


torch.library.register_autograd(
    _leviathan_forward_with_seed_op,
    _leviathan_with_seed_backward,
    setup_context=_leviathan_with_seed_setup_context,
)


def leviathan_embedding_compiler_safe(
    ids: torch.Tensor,
    params: dict[str, torch.Tensor],
    cfg: Any,
    knot_grid: torch.Tensor,
    *,
    mask_embedding: torch.Tensor | None = None,
    mask_token_id: int | None = None,
) -> torch.Tensor:
    """Run LEV through the opaque CUDA boundary or the model fallback.

    When both MEAP arguments are present, the CUDA path selects the dedicated
    vector in Leviathan's final GEMM epilogue. The backward sends masked rows
    only to ``mask_embedding`` and zeroes them before the LEV parameter path.
    """
    has_meap = mask_embedding is not None or mask_token_id is not None
    if has_meap:
        if mask_embedding is None or mask_token_id is None:
            raise ValueError("mask_embedding and mask_token_id must be provided together")
        if not isinstance(mask_token_id, int):
            raise TypeError("mask_token_id must be an integer")
        if not 0 <= mask_token_id < int(cfg.vocab_size):
            raise ValueError("mask_token_id must be inside the Leviathan vocabulary")
        if mask_embedding.ndim != 1 or mask_embedding.numel() != int(cfg.hidden_size):
            raise ValueError("mask_embedding must be a vector with cfg.hidden_size elements")
        if mask_embedding.device != params["codebooks"].device:
            raise ValueError("mask_embedding must be on the Leviathan device")
        if not mask_embedding.is_floating_point():
            raise TypeError("mask_embedding must be floating point")
    if not ids.is_cuda or not params["codebooks"].is_cuda:
        from .autograd_fn import leviathan_apply

        fallback_params = dict(params)
        fallback_params["knot_grid"] = knot_grid
        embeds = leviathan_apply(ids, fallback_params, cfg)
        if not has_meap:
            return embeds
        return torch.where(
            ids.eq(int(mask_token_id)).unsqueeze(-1),
            mask_embedding.to(device=embeds.device, dtype=embeds.dtype),
            embeds,
        )

    grad_enabled = torch.is_grad_enabled()
    leviathan_needs_backward = grad_enabled and any(
        tensor.requires_grad for name, tensor in params.items() if name != "knot_grid"
    )
    mask_needs_backward = grad_enabled and has_meap and mask_embedding.requires_grad
    needs_backward = leviathan_needs_backward
    op = _leviathan_forward_op if needs_backward else _leviathan_inference_op
    external_mask_grad = mask_needs_backward and not leviathan_needs_backward
    fuse_mask_in_kernel = has_meap and not external_mask_grad
    kernel_mask_embedding = (
        mask_embedding
        if fuse_mask_in_kernel and needs_backward
        else mask_embedding.detach()
        if fuse_mask_in_kernel
        else knot_grid.reshape(-1)[:0]
    )

    result = op(
        ids,
        params["codebooks"],
        params["head_proj_weight"],
        params["head_norm_weight"],
        params["head_norm_bias"],
        params["head_spline_delta"],
        params["head_out_weight"],
        knot_grid,
        kernel_mask_embedding,
        int(mask_token_id) if fuse_mask_in_kernel else -1,
        int(cfg.vocab_size),
        int(cfg.hidden_size),
        int(cfg.generator_d_seed),
        int(cfg.generator_num_modes),
        int(cfg.generator_num_knots),
        int(cfg.generator_spline_degree),
        int(cfg.generator_k),
        int(getattr(cfg, "generator_krank", params["head_spline_delta"].shape[-1])),
    )
    output = result[0] if needs_backward else result
    if external_mask_grad:
        # Frozen-LEV mask-only tuning needs a gradient only for the dedicated
        # vector. Keep the checkpoint-free inference launch and attach that
        # small gradient through the final selection instead of saving every
        # LEV intermediate and running its complete backward.
        output = torch.where(
            ids.eq(int(mask_token_id)).unsqueeze(-1),
            mask_embedding.to(device=output.device, dtype=output.dtype),
            output,
        )
    return output


def _leviathan_seed_from_codebooks(
    ids: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    """Build the differentiable compositional seed used by JTok.

    The legacy Leviathan custom op also materializes this seed internally,
    but marks it as a saved backward checkpoint. Reusing that tensor as a
    JTok input would silently cut the JTok -> codebook gradient. This small
    compiler-visible bridge repeats only the base-b lookup/sum, not the
    projection, spline, or output stages of Leviathan. Its value matches the
    kernel's stage-1 accumulation order and remains differentiable with
    respect to ``codebooks``.
    """
    flat_ids = ids.reshape(-1).long()
    num_codebooks, base, d_seed = codebooks.shape
    coords = base_k_decompose(flat_ids, base, num_codebooks)
    seed = torch.zeros(
        (flat_ids.numel(), d_seed),
        dtype=codebooks.dtype,
        device=codebooks.device,
    )
    for component in range(num_codebooks):
        seed = seed + codebooks[component].index_select(
            0, coords[:, component]
        )
    return seed.reshape(*ids.shape, d_seed)


def leviathan_embedding_with_seed_compiler_safe(
    ids: torch.Tensor,
    params: dict[str, torch.Tensor],
    cfg: Any,
    knot_grid: torch.Tensor,
    *,
    mask_embedding: torch.Tensor | None = None,
    mask_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Leviathan and expose the *same* kernel-produced seed to JTok.

    CUDA training uses a multi-output custom op.  Leviathan's A stage writes
    ``z`` once, JTok/JTok-M consumes that output, and the custom backward adds
    JTok's ``dL/dz`` to Leviathan's ``dL/dz`` before one codebook scatter.  The
    legacy embedding-only entry point is not changed.  CPU/reference callers
    retain the differentiable bridge because the Triton custom op is CUDA
    only.
    """
    has_meap = mask_embedding is not None or mask_token_id is not None
    if has_meap:
        if mask_embedding is None or mask_token_id is None:
            raise ValueError("mask_embedding and mask_token_id must be provided together")
        if not isinstance(mask_token_id, int):
            raise TypeError("mask_token_id must be an integer")
        if not 0 <= mask_token_id < int(cfg.vocab_size):
            raise ValueError("mask_token_id must be inside the Leviathan vocabulary")
        if mask_embedding.ndim != 1 or mask_embedding.numel() != int(cfg.hidden_size):
            raise ValueError("mask_embedding must be a vector with cfg.hidden_size elements")
        if mask_embedding.device != params["codebooks"].device:
            raise ValueError("mask_embedding must be on the Leviathan device")
        if not mask_embedding.is_floating_point():
            raise TypeError("mask_embedding must be floating point")

    if not ids.is_cuda or not params["codebooks"].is_cuda:
        embedding = leviathan_embedding_compiler_safe(
            ids,
            params,
            cfg,
            knot_grid,
            mask_embedding=mask_embedding,
            mask_token_id=mask_token_id,
        )
        seed = _leviathan_seed_from_codebooks(ids, params["codebooks"])
        return embedding, seed

    grad_enabled = torch.is_grad_enabled()
    needs_backward = grad_enabled and any(
        tensor.requires_grad for name, tensor in params.items() if name != "knot_grid"
    )
    mask_needs_backward = grad_enabled and has_meap and mask_embedding.requires_grad
    external_mask_grad = mask_needs_backward and not needs_backward
    fuse_mask_in_kernel = has_meap and not external_mask_grad
    kernel_mask_embedding = (
        mask_embedding
        if fuse_mask_in_kernel and needs_backward
        else mask_embedding.detach()
        if fuse_mask_in_kernel
        else knot_grid.reshape(-1)[:0]
    )
    args = (
        ids,
        params["codebooks"],
        params["head_proj_weight"],
        params["head_norm_weight"],
        params["head_norm_bias"],
        params["head_spline_delta"],
        params["head_out_weight"],
        knot_grid,
        kernel_mask_embedding,
        int(mask_token_id) if fuse_mask_in_kernel else -1,
        int(cfg.vocab_size),
        int(cfg.hidden_size),
        int(cfg.generator_d_seed),
        int(cfg.generator_num_modes),
        int(cfg.generator_num_knots),
        int(cfg.generator_spline_degree),
        int(cfg.generator_k),
        int(getattr(cfg, "generator_krank", params["head_spline_delta"].shape[-1])),
    )
    if needs_backward:
        result = _leviathan_forward_with_seed_op(*args)
        embedding, seed = result[0], result[1]
    else:
        embedding, seed = _leviathan_inference_with_seed_op(*args)
    if external_mask_grad:
        embedding = torch.where(
            ids.eq(int(mask_token_id)).unsqueeze(-1),
            mask_embedding.to(device=embedding.device, dtype=embedding.dtype),
            embedding,
        )
    return embedding, seed.reshape(*ids.shape, int(cfg.generator_d_seed))


__all__ = [
    "leviathan_embedding_compiler_safe",
    "leviathan_embedding_with_seed_compiler_safe",
]
