"""CUDA JTok/JTok-M kernels coupled to the Leviathan coordinate.

This module is deliberately separate from the legacy Leviathan embedding
operator.  The public ``jtok_apply`` and ``jtokm_apply`` functions consume the
continuous coordinate produced by Leviathan and never alter the legacy
``leviathan_embedding`` contract.  This makes the extension opt-in at the
model boundary: when JTok is disabled, this module is not called and the
original Leviathan kernel remains the only embedding path.

The implementation has two layers:

* ``*_reference`` functions are the semantic oracle and the training
  backward.  They use the same B-spline/product equations as the NeoLLM
  Torch implementation, but JTok-M evaluates only the selected experts.
* The CUDA path uses three small Triton stages: selected surface modes,
  selected-mode/output projection, and final normalization/modulation.  It
  never materializes ``[tokens, experts, modes, hidden]``.  The intermediate
  mode table is ``[tokens, selected_experts, modes]`` and is discarded before
  the caller receives the result.

The custom-op boundary follows the PyTorch 2.14 integration contract: fake
implementations describe output metadata and registered autograd recomputes
the compact reference path.  The ``backend="triton"`` mode is strict and
raises when the kernel cannot be used; ``backend="torch"`` is the explicit
reference baseline; ``backend="auto"`` is intended only for library callers
that deliberately want capability dispatch.

The backward currently prioritizes a bounded, auditable reference formula over
a second experimental Triton backward.  It saves only the operation inputs in
the custom-op context and recomputes selected surfaces, so no forward surface
or dense expert expansion is retained by autograd.  A future fused backward
can replace ``_jtok*_backward`` without changing the public contract.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F

from cut_cross_entropy.torch_2_14 import (
    TORCH_2_14_CUDA_KERNEL_CONTEXT,
    TORCH_2_14_MEMORY_ANNOTATIONS,
    annotate_tensors,
    cuda_kernel_region,
)

try:  # Triton is an optional dependency on CPU/macOS installations.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - CPU optional path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


Backend = Literal["auto", "torch", "triton"]
_SUPPORTED_KERNEL_DTYPES = (torch.float16, torch.bfloat16)
_BASIS_EPS = 1e-12
_PRODUCT_LOG_EPS = 1e-9
_DEFAULT_NORM_EPS = 1e-6


def _flatten_inputs(
    delta_m: torch.Tensor,
    z_tilde: torch.Tensor,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if delta_m.ndim < 2 or z_tilde.ndim < 2:
        raise ValueError("delta_m and z_tilde must have at least two dimensions")
    if delta_m.shape[:-1] != z_tilde.shape[:-1]:
        raise ValueError(
            "delta_m and z_tilde must share all leading dimensions; got "
            f"{tuple(delta_m.shape)} and {tuple(z_tilde.shape)}"
        )
    if delta_m.shape[-1] != hidden_size:
        raise ValueError(
            f"delta_m last dimension must be {hidden_size}, got {delta_m.shape[-1]}"
        )
    orig_shape = tuple(delta_m.shape)
    # A contiguous copy is explicit and differentiable.  The CUDA kernel has
    # a simple layout contract; callers can still pass transposed/sliced
    # tensors without silently taking a different numerical path.
    delta_flat = delta_m.reshape(-1, hidden_size).contiguous()
    z_flat = z_tilde.reshape(-1, z_tilde.shape[-1]).contiguous()
    return delta_flat, z_flat, orig_shape


def _valid_mask(
    valid_mask: Optional[torch.Tensor],
    n_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(n_tokens, dtype=torch.bool, device=device)
    mask = valid_mask.reshape(-1).to(device=device, dtype=torch.bool)
    if mask.numel() != n_tokens:
        raise ValueError(
            f"valid_mask must contain {n_tokens} elements, got {mask.numel()}"
        )
    return mask


def _empty_mask(device: torch.device) -> torch.Tensor:
    """Empty CUDA mask used to keep the custom-op schema shape-stable."""
    return torch.empty(0, device=device, dtype=torch.bool)


def _basis(z_flat: torch.Tensor, knot_grid: torch.Tensor) -> torch.Tensor:
    """Normalized quadratic B-spline basis, matching the model reference."""
    if knot_grid.ndim != 1 or knot_grid.numel() < 1:
        raise ValueError("knot_grid must be a non-empty one-dimensional tensor")
    scale = float(max(int(knot_grid.numel()) - 1, 0))
    z32 = z_flat.float().unsqueeze(-1)
    grid = knot_grid.float().view(1, 1, -1)
    distance = (z32 - grid).abs() * scale
    basis = torch.where(
        distance < 0.5,
        0.75 - distance.square(),
        torch.where(
            distance < 1.5,
            0.5 * (1.5 - distance).square(),
            torch.zeros_like(distance),
        ),
    )
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(_BASIS_EPS)


def _product_modes(
    basis: torch.Tensor,
    spline_coeff: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one or more expert surfaces in FP32.

    Args:
        basis: ``[N, d_seed, knots]``.
        spline_coeff: ``[E, modes, d_seed, knots]``.
    Returns:
        Product modes ``[N, E, modes]``.
    """
    phi = torch.einsum("ndg,emdg->nemd", basis, spline_coeff.float())
    log_mag = torch.log(phi.abs() + _PRODUCT_LOG_EPS).sum(dim=-1)
    negative = (phi < 0).to(torch.int32).sum(dim=-1)
    sign = 1.0 - 2.0 * (negative.remainder(2)).float()
    return sign * torch.exp(log_mag)


def _project_surfaces(
    z_flat: torch.Tensor,
    modes: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Project modes and the linear residual without an expert expansion."""
    # modes is [N,E,M], spline_out is [E,M,D].
    result = torch.einsum(
        "nem,emd->ned",
        modes.to(target_dtype),
        spline_out.to(target_dtype),
    )
    result = result + torch.einsum(
        "nd,edh->neh",
        z_flat.to(target_dtype),
        residual_out.to(target_dtype),
    )
    return result


def _all_surfaces_reference(
    z_flat: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    target_dtype: torch.dtype,
    knot_grid: torch.Tensor,
) -> torch.Tensor:
    """Evaluate every expert surface for the Torch/``torch.compile`` oracle.

    This is intentionally the dense reference path.  It is the fair baseline
    for the external JTok-M kernel: both paths perform the same routing and
    surface math, while only the external path avoids retaining the dense
    ``[tokens, experts, hidden]`` result.
    """
    basis = _basis(z_flat, knot_grid)
    modes = _product_modes(basis, spline_coeff)
    return _project_surfaces(
        z_flat,
        modes,
        spline_out,
        residual_out,
        target_dtype,
    )


def jtok_reference(
    delta_m: torch.Tensor,
    z_tilde: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    knot_grid: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    norm_eps: float = _DEFAULT_NORM_EPS,
) -> torch.Tensor:
    """Reference JTok transformation for the single base expert."""
    if spline_coeff.ndim != 4 or spline_coeff.shape[0] != 1:
        raise ValueError(
            "plain JTok expects spline_coeff with shape [1, modes, d_seed, knots]"
        )
    if spline_out.ndim != 3 or spline_out.shape[0] != 1:
        raise ValueError("plain JTok expects spline_out with shape [1, modes, hidden]")
    if residual_out.ndim != 3 or residual_out.shape[0] != 1:
        raise ValueError("plain JTok expects residual_out with shape [1, d_seed, hidden]")
    if scaler.ndim != 1 or scaler.numel() != delta_m.shape[-1]:
        raise ValueError("scaler must be a vector with delta_m.shape[-1] elements")

    dm, z, orig_shape = _flatten_inputs(delta_m, z_tilde, delta_m.shape[-1])
    mask = _valid_mask(valid_mask, dm.shape[0], dm.device)
    basis = _basis(z, knot_grid)
    modes = _product_modes(basis, spline_coeff)[..., 0, :]
    surfaces = _project_surfaces(
        z,
        modes.unsqueeze(1),
        spline_out,
        residual_out,
        dm.dtype,
    )[:, 0, :]
    direction = surfaces / (surfaces.norm(dim=-1, keepdim=True) + float(norm_eps))
    gate = 1.0 + scaler.to(dm.dtype) * direction
    output = dm * gate
    output = torch.where(mask.unsqueeze(-1), output, dm)
    return output.reshape(orig_shape)


def _selected_surface_reference(
    z: torch.Tensor,
    expert_idx: torch.Tensor,
    selected_weights: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    knot_grid: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Route-first JTok-M surface evaluation used by backward and oracle tests.

    The selected expert indices are gathered as a tensor instead of being
    partitioned with Python ``where`` loops.  Besides bounding the temporary
    surface to ``[tokens, top_k, modes, hidden]``, this keeps the reference
    backward traceable by ``torch.compile(fullgraph=True)``: routing values
    are data-dependent, so branches such as ``rows.numel() == 0`` are not
    legal during AOTAutograd capture.
    """
    if expert_idx.ndim != 2 or selected_weights.shape != expert_idx.shape:
        raise ValueError("expert_idx and selected_weights must have shape [N, top_k]")
    n_tokens, top_k = expert_idx.shape
    basis_all = _basis(z, knot_grid)
    # Gathering only the selected experts gives [N,K,M,d,g] coefficients,
    # [N,K,M,H] output projections and [N,K,d,H] residual projections.  It is
    # still route-first, but has no value-dependent control flow and no dense
    # [N,E,M,H] materialization.
    selected_coeff = spline_coeff[expert_idx]
    phi = torch.einsum("ndg,nkmdg->nkmd", basis_all, selected_coeff.float())
    log_mag = torch.log(phi.abs() + _PRODUCT_LOG_EPS).sum(dim=-1)
    negative = (phi < 0).to(torch.int32).sum(dim=-1)
    sign = 1.0 - 2.0 * negative.remainder(2).float()
    modes = sign * torch.exp(log_mag)
    selected_out = spline_out[expert_idx]
    values = torch.einsum(
        "nkm,nkmh->nkh",
        modes.to(target_dtype),
        selected_out.to(target_dtype),
    )
    selected_residual = residual_out[expert_idx]
    values = values + torch.einsum(
        "nd,nkdh->nkh",
        z.to(target_dtype),
        selected_residual.to(target_dtype),
    )
    weighted = values * selected_weights.to(target_dtype).unsqueeze(-1)
    return weighted.sum(dim=1)


def jtokm_reference(
    delta_m: torch.Tensor,
    z_tilde: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    router_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    *,
    top_k: int,
    valid_mask: Optional[torch.Tensor] = None,
    norm_eps: float = _DEFAULT_NORM_EPS,
    residual_scale: float = 1.0,
    expert_idx: Optional[torch.Tensor] = None,
    selected_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference JTok-M output and compact routing statistics."""
    del router_weight  # routing is performed by jtokm_apply before this call
    dm, z, orig_shape = _flatten_inputs(delta_m, z_tilde, delta_m.shape[-1])
    mask = _valid_mask(valid_mask, dm.shape[0], dm.device)
    if expert_idx is None or selected_weights is None:
        raise ValueError("jtokm_reference requires explicit expert_idx and weights")
    if expert_idx.shape[0] != dm.shape[0] or expert_idx.shape[1] != int(top_k):
        raise ValueError("expert_idx has an incompatible token/top-k shape")
    surfaces = _all_surfaces_reference(
        z, spline_coeff, spline_out, residual_out, dm.dtype, knot_grid
    )
    gather_idx = expert_idx.unsqueeze(-1).expand(
        dm.shape[0], int(top_k), dm.shape[-1]
    )
    mixed = (selected_weights.to(dm.dtype).unsqueeze(-1) * surfaces.gather(1, gather_idx)).sum(
        dim=1
    )
    direction = mixed / (mixed.norm(dim=-1, keepdim=True) + float(norm_eps))
    delta_r = float(residual_scale) * scaler.to(dm.dtype) * direction
    output = torch.where(mask.unsqueeze(-1), dm + delta_r, dm)
    # The router statistics are calculated by jtokm_apply, where logits are
    # still attached to the router graph.  These placeholders keep this
    # private helper focused on output semantics; callers that need metrics
    # should use jtokm_routing_stats directly.
    empty = dm.new_empty((0,), dtype=torch.float32)
    return output.reshape(orig_shape), empty, empty, mask.sum().float().detach(), expert_idx


def jtokm_routing_stats(
    logits: torch.Tensor,
    expert_idx: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    *,
    norm_eps: float = _DEFAULT_NORM_EPS,
) -> dict[str, torch.Tensor]:
    """Return JTok-M's differentiable balance terms and detached diagnostics."""
    if logits.ndim != 2 or expert_idx.ndim != 2:
        raise ValueError("logits and expert_idx must be two-dimensional")
    n_tokens, n_experts = logits.shape
    if expert_idx.shape[0] != n_tokens:
        raise ValueError("logits and expert_idx must have the same token count")
    mask = _valid_mask(valid_mask, n_tokens, logits.device)
    probabilities = torch.sigmoid(logits.float())
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        float(norm_eps)
    )
    valid_f = mask.to(probabilities.dtype).unsqueeze(-1)
    p_sum = (probabilities * valid_f).sum(dim=0)
    hard_onehot = torch.zeros_like(probabilities).scatter(1, expert_idx, 1.0)
    f_sum = (hard_onehot * valid_f).sum(dim=0).detach()
    valid_tokens = mask.sum().to(dtype=torch.float32).detach()

    denom = valid_tokens.clamp_min(1.0)
    p_i = p_sum / denom
    f_i = f_sum / (denom * float(expert_idx.shape[1]))
    f_mean = f_i.mean()
    load_cv = f_i.std(unbiased=False) / f_mean.clamp_min(float(norm_eps))
    load_entropy = -(f_i.clamp_min(float(norm_eps)) * f_i.clamp_min(float(norm_eps)).log()).sum()
    load_entropy = load_entropy / math.log(max(n_experts, 2))
    # An all-invalid batch is a valid padding/MEAP edge case.  Report neutral
    # zero-valued router metrics instead of the small epsilon entropy caused
    # by evaluating ``log(eps)`` on an empty population.
    has_valid_tokens = valid_tokens > 0
    load_entropy = torch.where(
        has_valid_tokens,
        load_entropy,
        torch.zeros_like(load_entropy),
    )
    token_entropy = -(
        probabilities.clamp_min(float(norm_eps))
        * probabilities.clamp_min(float(norm_eps)).log()
    ).sum(dim=-1)
    token_entropy = (token_entropy * mask.to(token_entropy.dtype)).sum() / denom
    active_experts = (f_sum > 0).sum().to(dtype=torch.float32)
    return {
        # p_sum stays differentiable because it is used by the auxiliary loss.
        "p_sum": p_sum,
        "f_sum": f_sum,
        # Per-expert normalized diagnostics are detached; callers must use
        # p_sum, not these views, when constructing the auxiliary loss.
        "expert_probability": p_i.detach(),
        "expert_fraction": f_i.detach(),
        "valid_tokens": valid_tokens,
        "load_cv": load_cv.detach(),
        "load_entropy": load_entropy.detach(),
        "router_entropy": token_entropy.detach(),
        "active_experts": active_experts.detach(),
        "max_load": f_i.max().detach() if f_i.numel() else f_i.new_zeros(()),
        "min_load": f_i.min().detach() if f_i.numel() else f_i.new_zeros(()),
        "invalid_fraction": (1.0 - mask.float().mean()).clamp(0.0, 1.0).detach(),
    }


def jtokm_auxiliary_loss(
    stats: dict[str, torch.Tensor],
    *,
    num_experts: int,
    top_k: int,
    weight: float,
) -> torch.Tensor:
    """Compute ``weight * E * sum_i(p_i*f_i)`` from compact reductions."""
    p_sum = stats["p_sum"]
    f_sum = stats["f_sum"].to(device=p_sum.device, dtype=torch.float32)
    valid_tokens = stats["valid_tokens"].to(device=p_sum.device).clamp_min(1.0)
    p_i = p_sum.float() / valid_tokens
    f_i = f_sum / (valid_tokens * float(top_k))
    return float(weight) * float(num_experts) * (p_i * f_i).sum()


def _check_common_kernel_inputs(
    delta: torch.Tensor,
    z: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    knot_grid: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if not delta.is_cuda:
        raise RuntimeError("JTok Triton kernels require CUDA tensors")
    if delta.dtype not in _SUPPORTED_KERNEL_DTYPES:
        raise TypeError(
            "JTok Triton kernels currently support only float16/bfloat16, got "
            f"{delta.dtype}"
        )
    if any(t.device != delta.device for t in (z, spline_coeff, spline_out, residual_out, scaler, knot_grid)):
        raise RuntimeError("all JTok inputs must be on the same CUDA device")
    if z.dtype != delta.dtype:
        raise TypeError("z_tilde and delta_m must have the same dtype")
    if spline_coeff.ndim != 4:
        raise ValueError("spline_coeff must have shape [experts, modes, d_seed, knots]")
    experts, modes, d_seed, knots = map(int, spline_coeff.shape)
    if spline_out.shape != (experts, modes, delta.shape[-1]):
        raise ValueError(
            "spline_out must have shape "
            f"[{experts}, {modes}, {delta.shape[-1]}], got {tuple(spline_out.shape)}"
        )
    if residual_out.shape != (experts, d_seed, delta.shape[-1]):
        raise ValueError(
            "residual_out must have shape "
            f"[{experts}, {d_seed}, {delta.shape[-1]}], got {tuple(residual_out.shape)}"
        )
    if scaler.shape != (delta.shape[-1],):
        raise ValueError("scaler must have one value per hidden dimension")
    if knot_grid.shape != (knots,):
        raise ValueError(f"knot_grid must have shape [{knots}]")
    if knots < 1 or modes < 1 or d_seed < 1 or experts < 1:
        raise ValueError("experts, modes, d_seed, and knots must all be positive")
    return experts, modes, d_seed, knots, int(delta.shape[-1])


if _TRITON_AVAILABLE:

    @triton.jit
    def _jtok_modes_kernel_masked(
        z_ptr,
        coeff_ptr,
        grid_ptr,
        expert_idx_ptr,
        valid_ptr,
        modes_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        KNOT_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // (TOP_K * NUM_MODES)
        mode_slot = pid % (TOP_K * NUM_MODES)
        slot = mode_slot // NUM_MODES
        mode = mode_slot % NUM_MODES
        row_mask = row < N
        expert = tl.load(expert_idx_ptr + row * TOP_K + slot, mask=row_mask, other=0).to(tl.int32)
        row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0)
        grid = tl.load(grid_ptr + tl.arange(0, KNOT_PAD), mask=tl.arange(0, KNOT_PAD) < NUM_KNOTS, other=0.0).to(tl.float32)
        scale = float(max(int(NUM_KNOTS) - 1, 0))
        log_acc = 0.0
        neg_acc = 0
        for d in tl.range(0, D_SEED):
            x = tl.load(z_ptr + row * D_SEED + d, mask=row_mask, other=0.0).to(tl.float32)
            distance = tl.abs(x - grid) * scale
            basis = tl.where(
                distance < 0.5,
                0.75 - distance * distance,
                tl.where(
                    distance < 1.5,
                    0.5 * (1.5 - distance) * (1.5 - distance),
                    0.0,
                ),
            )
            basis = tl.where(tl.arange(0, KNOT_PAD) < NUM_KNOTS, basis, 0.0)
            basis = basis / tl.maximum(tl.sum(basis, axis=0), 1e-12)
            coeff = tl.load(
                coeff_ptr
                + (((expert * NUM_MODES + mode) * D_SEED + d) * NUM_KNOTS)
                + tl.arange(0, KNOT_PAD),
                mask=tl.arange(0, KNOT_PAD) < NUM_KNOTS,
                other=0.0,
            ).to(tl.float32)
            phi = tl.sum(basis * coeff, axis=0)
            log_acc += tl.log(tl.abs(phi) + 1e-9)
            neg_acc += (phi < 0).to(tl.int32)
        value = (1.0 - 2.0 * (neg_acc & 1).to(tl.float32)) * tl.exp(log_acc)
        value = tl.where(row_valid & row_mask, value, 0.0)
        tl.store(
            modes_ptr + (row * TOP_K + slot) * NUM_MODES + mode,
            value.to(modes_ptr.dtype.element_ty),
            mask=row_mask,
        )

    @triton.jit
    def _jtok_modes_kernel_unmasked(
        z_ptr,
        coeff_ptr,
        grid_ptr,
        expert_idx_ptr,
        modes_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        KNOT_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // (TOP_K * NUM_MODES)
        mode_slot = pid % (TOP_K * NUM_MODES)
        slot = mode_slot // NUM_MODES
        mode = mode_slot % NUM_MODES
        row_mask = row < N
        expert = tl.load(expert_idx_ptr + row * TOP_K + slot, mask=row_mask, other=0).to(tl.int32)
        knot_offsets = tl.arange(0, KNOT_PAD)
        grid = tl.load(grid_ptr + knot_offsets, mask=knot_offsets < NUM_KNOTS, other=0.0).to(tl.float32)
        scale = float(max(int(NUM_KNOTS) - 1, 0))
        log_acc = 0.0
        neg_acc = 0
        for d in tl.range(0, D_SEED):
            x = tl.load(z_ptr + row * D_SEED + d, mask=row_mask, other=0.0).to(tl.float32)
            distance = tl.abs(x - grid) * scale
            basis = tl.where(
                distance < 0.5,
                0.75 - distance * distance,
                tl.where(
                    distance < 1.5,
                    0.5 * (1.5 - distance) * (1.5 - distance),
                    0.0,
                ),
            )
            basis = tl.where(knot_offsets < NUM_KNOTS, basis, 0.0)
            basis = basis / tl.maximum(tl.sum(basis, axis=0), 1e-12)
            coeff = tl.load(
                coeff_ptr
                + (((expert * NUM_MODES + mode) * D_SEED + d) * NUM_KNOTS)
                + knot_offsets,
                mask=knot_offsets < NUM_KNOTS,
                other=0.0,
            ).to(tl.float32)
            phi = tl.sum(basis * coeff, axis=0)
            log_acc += tl.log(tl.abs(phi) + 1e-9)
            neg_acc += (phi < 0).to(tl.int32)
        value = (1.0 - 2.0 * (neg_acc & 1).to(tl.float32)) * tl.exp(log_acc)
        tl.store(
            modes_ptr + (row * TOP_K + slot) * NUM_MODES + mode,
            value.to(modes_ptr.dtype.element_ty),
            mask=row_mask,
        )

    @triton.jit
    def _jtok_project_kernel(
        z_ptr,
        modes_ptr,
        spline_out_ptr,
        residual_out_ptr,
        expert_idx_ptr,
        selected_weights_ptr,
        surface_ptr,
        norm_ptr,
        N,
        D_SEED: tl.constexpr,
        HIDDEN: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)
        cols = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = pid_row < N
        col_mask = cols < HIDDEN
        mixed = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for slot in tl.range(0, TOP_K):
            expert = tl.load(expert_idx_ptr + pid_row * TOP_K + slot, mask=row_mask, other=0).to(tl.int32)
            weight = tl.load(selected_weights_ptr + pid_row * TOP_K + slot, mask=row_mask, other=0.0).to(tl.float32)
            values = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for mode in tl.range(0, NUM_MODES):
                mode_value = tl.load(
                    modes_ptr + (pid_row * TOP_K + slot) * NUM_MODES + mode,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                out_weight = tl.load(
                    spline_out_ptr + (expert * NUM_MODES + mode) * HIDDEN + cols,
                    mask=row_mask & col_mask,
                    other=0.0,
                ).to(tl.float32)
                values += mode_value * out_weight
            residual = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d in tl.range(0, D_SEED):
                z_value = tl.load(z_ptr + pid_row * D_SEED + d, mask=row_mask, other=0.0).to(tl.float32)
                residual_weight = tl.load(
                    residual_out_ptr + (expert * D_SEED + d) * HIDDEN + cols,
                    mask=row_mask & col_mask,
                    other=0.0,
                ).to(tl.float32)
                residual += z_value * residual_weight
            mixed += weight * (values + residual)
        mixed = tl.where(row_mask & col_mask, mixed, 0.0)
        tl.store(surface_ptr + pid_row * HIDDEN + cols, mixed.to(surface_ptr.dtype.element_ty), mask=row_mask & col_mask)
        tl.atomic_add(norm_ptr + pid_row, tl.sum(mixed * mixed, axis=0), mask=row_mask)

    @triton.jit
    def _jtok_finalize_kernel(
        delta_ptr,
        surface_ptr,
        scaler_ptr,
        norm_ptr,
        valid_ptr,
        output_ptr,
        N,
        HIDDEN: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NORM_EPS: tl.constexpr,
        RESIDUAL_SCALE: tl.constexpr,
        MIXTURE: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (row < N) & (cols < HIDDEN)
        valid = row < N
        if HAS_MASK:
            valid = valid & tl.load(valid_ptr + row, mask=row < N, other=0)
        norm = tl.sqrt(tl.load(norm_ptr + row, mask=row < N, other=0.0)) + NORM_EPS
        surface = tl.load(surface_ptr + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
        scaler = tl.load(scaler_ptr + cols, mask=cols < HIDDEN, other=0.0).to(tl.float32)
        direction = surface / norm
        delta = tl.load(delta_ptr + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
        if MIXTURE:
            value = delta + RESIDUAL_SCALE * scaler * direction
        else:
            value = delta * (1.0 + scaler * direction)
        value = tl.where(valid & mask, value, delta)
        tl.store(output_ptr + row * HIDDEN + cols, value.to(output_ptr.dtype.element_ty), mask=mask)


def _run_jtok_triton(
    delta: torch.Tensor,
    z: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    knot_grid: torch.Tensor,
    expert_idx: torch.Tensor,
    selected_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    norm_eps: float,
    residual_scale: float,
    mixture: bool,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:  # pragma: no cover - guarded by strict dispatch
        raise RuntimeError("Triton is not installed")
    experts, modes, d_seed, knots, hidden = _check_common_kernel_inputs(
        delta, z, spline_coeff, spline_out, residual_out, scaler, knot_grid
    )
    del experts
    n_tokens = int(delta.shape[0])
    top_k = int(expert_idx.shape[1])
    if top_k < 1 or top_k > int(spline_coeff.shape[0]):
        raise ValueError("top_k must be in [1, num_experts]")
    if selected_weights.shape != expert_idx.shape:
        raise ValueError("selected_weights and expert_idx must have the same shape")
    has_mask = valid_mask.numel() != 0
    if has_mask and valid_mask.numel() != n_tokens:
        raise ValueError("valid_mask has the wrong number of elements")

    # Mode values are intentionally stored in the activation dtype.  They are
    # transient kernel workspace, not an autograd checkpoint.
    mode_workspace = torch.empty(
        n_tokens,
        top_k,
        modes,
        device=delta.device,
        dtype=delta.dtype,
    )
    surface = torch.empty_like(delta)
    norm = torch.zeros(n_tokens, device=delta.device, dtype=torch.float32)
    output = torch.empty_like(delta)
    knot_pad = triton.next_power_of_2(knots)
    if has_mask:
        mode_grid = (n_tokens * top_k * modes,)
        _jtok_modes_kernel_masked[mode_grid](
            z,
            spline_coeff,
            knot_grid,
            expert_idx,
            valid_mask,
            mode_workspace,
            n_tokens,
            D_SEED=d_seed,
            NUM_KNOTS=knots,
            NUM_MODES=modes,
            TOP_K=top_k,
            KNOT_PAD=knot_pad,
        )
    else:
        mode_grid = (n_tokens * top_k * modes,)
        _jtok_modes_kernel_unmasked[mode_grid](
            z,
            spline_coeff,
            knot_grid,
            expert_idx,
            mode_workspace,
            n_tokens,
            D_SEED=d_seed,
            NUM_KNOTS=knots,
            NUM_MODES=modes,
            TOP_K=top_k,
            KNOT_PAD=knot_pad,
        )
    block_n = min(128, triton.next_power_of_2(hidden))
    project_grid = (n_tokens, triton.cdiv(hidden, block_n))
    _jtok_project_kernel[project_grid](
        z,
        mode_workspace,
        spline_out,
        residual_out,
        expert_idx,
        selected_weights,
        surface,
        norm,
        n_tokens,
        D_SEED=d_seed,
        HIDDEN=hidden,
        NUM_MODES=modes,
        TOP_K=top_k,
        BLOCK_N=block_n,
        HAS_MASK=has_mask,
        num_warps=4,
        num_stages=1,
    )
    final_grid = (n_tokens, triton.cdiv(hidden, block_n))
    _jtok_finalize_kernel[final_grid](
        delta,
        surface,
        scaler,
        norm,
        valid_mask,
        output,
        n_tokens,
        HIDDEN=hidden,
        BLOCK_N=block_n,
        NORM_EPS=float(norm_eps),
        RESIDUAL_SCALE=float(residual_scale),
        MIXTURE=bool(mixture),
        HAS_MASK=has_mask,
        num_warps=4,
        num_stages=1,
    )
    if TORCH_2_14_MEMORY_ANNOTATIONS:
        annotate_tensors(
            "jtok.mixture" if mixture else "jtok.forward",
            modes=mode_workspace,
            surface=surface,
            norm=norm,
            output=output,
        )
    return output


def _kernel_region(name: str, device: torch.device):
    if TORCH_2_14_CUDA_KERNEL_CONTEXT:
        return cuda_kernel_region(name, device)
    from contextlib import nullcontext

    return nullcontext()


@torch.library.custom_op(
    "cut_cross_entropy::jtok_forward",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _jtok_forward_op(
    delta: torch.Tensor,
    z: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    knot_grid: torch.Tensor,
    expert_idx: torch.Tensor,
    selected_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    with _kernel_region("jtok.forward", delta.device):
        return _run_jtok_triton(
            delta,
            z,
            spline_coeff,
            spline_out,
            residual_out,
            scaler,
            knot_grid,
            expert_idx,
            selected_weights,
            valid_mask,
            norm_eps=norm_eps,
            residual_scale=0.0,
            mixture=False,
        )


@_jtok_forward_op.register_fake
def _jtok_forward_fake(
    delta: torch.Tensor,
    z: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    knot_grid: torch.Tensor,
    expert_idx: torch.Tensor,
    selected_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    del z, spline_coeff, spline_out, residual_out, scaler, knot_grid
    del expert_idx, selected_weights, valid_mask, norm_eps
    return torch.empty_like(delta)


def _jtok_setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    del output
    (
        delta,
        z,
        spline_coeff,
        spline_out,
        residual_out,
        scaler,
        knot_grid,
        expert_idx,
        selected_weights,
        valid_mask,
        norm_eps,
    ) = inputs
    ctx.save_for_backward(
        delta,
        z,
        spline_coeff,
        spline_out,
        residual_out,
        scaler,
        knot_grid,
        expert_idx,
        selected_weights,
        valid_mask,
    )
    ctx.norm_eps = float(norm_eps)


def _autograd_recompute(
    reference_fn,
    tensors: tuple[torch.Tensor, ...],
    grad_out: torch.Tensor,
    **kwargs: Any,
) -> tuple[Optional[torch.Tensor], ...]:
    detached: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_positions: list[int] = []
    for position, tensor in enumerate(tensors):
        value = tensor.detach().requires_grad_(bool(tensor.requires_grad))
        detached.append(value)
        if value.requires_grad:
            targets.append(value)
            target_positions.append(position)
    gradients: list[Optional[torch.Tensor]] = [None] * len(tensors)
    if not targets:
        return tuple(gradients)
    with torch.enable_grad():
        output = reference_fn(*detached, **kwargs)
        computed = torch.autograd.grad(
            output,
            targets,
            grad_out,
            allow_unused=True,
        )
    for position, gradient in zip(target_positions, computed):
        gradients[position] = gradient
    return tuple(gradients)


def _jtok_backward(ctx: Any, grad_out: torch.Tensor):
    (
        delta,
        z,
        spline_coeff,
        spline_out,
        residual_out,
        scaler,
        knot_grid,
        expert_idx,
        selected_weights,
        valid_mask,
    ) = ctx.saved_tensors
    grads = _autograd_recompute(
        lambda d, zz, c, so, ro, s: jtok_reference(
            d,
            zz,
            c,
            so,
            ro,
            s,
            knot_grid,
            valid_mask=(valid_mask if valid_mask.numel() else None),
            norm_eps=ctx.norm_eps,
        ),
        (delta, z, spline_coeff, spline_out, residual_out, scaler),
        grad_out,
    )
    return (*grads, None, None, None, None, None)


torch.library.register_autograd(
    _jtok_forward_op,
    _jtok_backward,
    setup_context=_jtok_setup_context,
)


@torch.library.custom_op(
    "cut_cross_entropy::jtokm_forward",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _jtokm_forward_op(
    delta: torch.Tensor,
    z: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    expert_idx: torch.Tensor,
    selected_weights: torch.Tensor,
    knot_grid: torch.Tensor,
    valid_mask: torch.Tensor,
    norm_eps: float,
    residual_scale: float,
) -> torch.Tensor:
    with _kernel_region("jtokm.forward", delta.device):
        return _run_jtok_triton(
            delta,
            z,
            spline_coeff,
            spline_out,
            residual_out,
            scaler,
            knot_grid,
            expert_idx,
            selected_weights,
            valid_mask,
            norm_eps=norm_eps,
            residual_scale=residual_scale,
            mixture=True,
        )


@_jtokm_forward_op.register_fake
def _jtokm_forward_fake(
    delta: torch.Tensor,
    z: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    expert_idx: torch.Tensor,
    selected_weights: torch.Tensor,
    knot_grid: torch.Tensor,
    valid_mask: torch.Tensor,
    norm_eps: float,
    residual_scale: float,
) -> torch.Tensor:
    del z, spline_coeff, spline_out, residual_out, scaler
    del expert_idx, selected_weights, knot_grid, valid_mask, norm_eps, residual_scale
    return torch.empty_like(delta)


def _jtokm_setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    del output
    (
        delta,
        z,
        spline_coeff,
        spline_out,
        residual_out,
        scaler,
        expert_idx,
        selected_weights,
        knot_grid,
        valid_mask,
        norm_eps,
        residual_scale,
    ) = inputs
    ctx.save_for_backward(
        delta,
        z,
        spline_coeff,
        spline_out,
        residual_out,
        scaler,
        expert_idx,
        selected_weights,
        knot_grid,
        valid_mask,
    )
    ctx.norm_eps = float(norm_eps)
    ctx.residual_scale = float(residual_scale)


def _jtokm_backward(ctx: Any, grad_out: torch.Tensor):
    (
        delta,
        z,
        spline_coeff,
        spline_out,
        residual_out,
        scaler,
        expert_idx,
        selected_weights,
        knot_grid,
        valid_mask,
    ) = ctx.saved_tensors

    def reference(d, zz, c, so, ro, s, w):
        mixed = _selected_surface_reference(
            zz,
            expert_idx,
            w,
            c,
            so,
            ro,
            knot_grid,
            d.dtype,
        )
        mask = (
            _valid_mask(valid_mask, d.shape[0], d.device)
            if valid_mask.numel()
            else torch.ones(d.shape[0], device=d.device, dtype=torch.bool)
        )
        direction = mixed / (mixed.norm(dim=-1, keepdim=True) + ctx.norm_eps)
        update = d + ctx.residual_scale * s.to(d.dtype) * direction
        return torch.where(mask.unsqueeze(-1), update, d)

    grads = _autograd_recompute(
        reference,
        (delta, z, spline_coeff, spline_out, residual_out, scaler, selected_weights),
        grad_out,
    )
    d_delta, d_z, d_coeff, d_spline_out, d_residual_out, d_scaler, d_weights = grads
    return (
        d_delta,
        d_z,
        d_coeff,
        d_spline_out,
        d_residual_out,
        d_scaler,
        None,
        d_weights,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    _jtokm_forward_op,
    _jtokm_backward,
    setup_context=_jtokm_setup_context,
)


def _select_backend(backend: Backend, tensors: tuple[torch.Tensor, ...]) -> bool:
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"backend must be 'auto', 'torch', or 'triton', got {backend!r}")
    if backend == "torch":
        return False
    available = _TRITON_AVAILABLE and tensors[0].is_cuda
    if backend == "triton" and not available:
        raise RuntimeError("backend='triton' requires Triton and CUDA tensors")
    return available


def _prepare_mask_for_kernel(
    valid_mask: Optional[torch.Tensor],
    n_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_mask is None:
        return _empty_mask(device)
    return _valid_mask(valid_mask, n_tokens, device).contiguous()


def jtok_apply(
    delta_m: torch.Tensor,
    z_tilde: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    knot_grid: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    norm_eps: float = _DEFAULT_NORM_EPS,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Apply JTok using the explicit Torch or strict Triton backend."""
    if spline_coeff.shape[0] != 1:
        raise ValueError("plain JTok requires exactly one expert")
    dm, z, orig_shape = _flatten_inputs(delta_m, z_tilde, delta_m.shape[-1])
    coeff = spline_coeff.contiguous()
    out_weight = spline_out.contiguous()
    residual = residual_out.contiguous()
    scale = scaler.contiguous()
    grid = knot_grid.contiguous()
    mask = _prepare_mask_for_kernel(valid_mask, dm.shape[0], dm.device)
    expert_idx = torch.zeros(dm.shape[0], 1, device=dm.device, dtype=torch.long)
    selected_weights = torch.ones(dm.shape[0], 1, device=dm.device, dtype=torch.float32)
    use_kernel = _select_backend(
        backend,
        (dm, z, coeff, out_weight, residual, scale, grid),
    )
    if use_kernel:
        output = _jtok_forward_op(
            dm,
            z,
            coeff,
            out_weight,
            residual,
            scale,
            grid,
            expert_idx,
            selected_weights,
            mask,
            float(norm_eps),
        )
    else:
        output = jtok_reference(
            dm,
            z,
            coeff,
            out_weight,
            residual,
            scale,
            grid,
            valid_mask=(mask if mask.numel() else None),
            norm_eps=norm_eps,
        )
    return output.reshape(orig_shape)


def jtokm_apply(
    delta_m: torch.Tensor,
    z_tilde: torch.Tensor,
    router_state: torch.Tensor,
    spline_coeff: torch.Tensor,
    spline_out: torch.Tensor,
    residual_out: torch.Tensor,
    scaler: torch.Tensor,
    router_weight: torch.Tensor,
    knot_grid: torch.Tensor,
    *,
    top_k: int,
    valid_mask: Optional[torch.Tensor] = None,
    norm_eps: float = _DEFAULT_NORM_EPS,
    residual_scale: float = 1.0,
    compute_aux: bool = False,
    backend: Backend = "auto",
) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
    """Apply JTok-M and optionally return compact balance diagnostics."""
    dm, z, orig_shape = _flatten_inputs(delta_m, z_tilde, delta_m.shape[-1])
    h, _, _ = _flatten_inputs(router_state, z_tilde, delta_m.shape[-1])
    if h.shape != dm.shape:
        raise ValueError("router_state must have the same shape as delta_m")
    experts = int(spline_coeff.shape[0])
    if not 1 <= int(top_k) <= experts:
        raise ValueError(f"top_k must be in [1, {experts}], got {top_k}")
    if router_weight.shape != (experts, dm.shape[-1]):
        raise ValueError(
            f"router_weight must have shape [{experts}, {dm.shape[-1]}], "
            f"got {tuple(router_weight.shape)}"
        )
    mask = _prepare_mask_for_kernel(valid_mask, dm.shape[0], dm.device)
    h_norm = h * torch.rsqrt(h.float().square().mean(dim=-1, keepdim=True) + float(norm_eps))
    logits = F.linear(h_norm.to(router_weight.dtype), router_weight)
    top_values, expert_idx = torch.topk(logits, int(top_k), dim=-1)
    selected_prob = torch.sigmoid(top_values.float())
    selected_weights = selected_prob / selected_prob.sum(dim=-1, keepdim=True).clamp_min(
        float(norm_eps)
    )

    coeff = spline_coeff.contiguous()
    out_weight = spline_out.contiguous()
    residual = residual_out.contiguous()
    scale = scaler.contiguous()
    grid = knot_grid.contiguous()
    use_kernel = _select_backend(
        backend,
        (dm, z, coeff, out_weight, residual, scale, router_weight, grid),
    )
    if use_kernel:
        output = _jtokm_forward_op(
            dm,
            z,
            coeff,
            out_weight,
            residual,
            scale,
            expert_idx.contiguous(),
            selected_weights.contiguous(),
            grid,
            mask,
            float(norm_eps),
            float(residual_scale),
        )
    else:
        # Keep ``backend='torch'`` as the fair torch.compile baseline.  It
        # deliberately follows the model's dense reference path, including
        # the all-expert surface tensor.  The external Triton path below is
        # allowed to be lower-memory precisely because it avoids this tensor.
        all_surfaces = _all_surfaces_reference(
            z, coeff, out_weight, residual, dm.dtype, grid
        )
        gather_idx = expert_idx.unsqueeze(-1).expand(
            dm.shape[0], int(top_k), dm.shape[-1]
        )
        selected = all_surfaces.gather(1, gather_idx)
        mixed = (selected_weights.to(dm.dtype).unsqueeze(-1) * selected).sum(dim=1)
        direction = mixed / (mixed.norm(dim=-1, keepdim=True) + float(norm_eps))
        output = dm + float(residual_scale) * scale.to(dm.dtype) * direction
        output = torch.where(
            (mask if mask.numel() else torch.ones(dm.shape[0], device=dm.device, dtype=torch.bool)).unsqueeze(-1),
            output,
            dm,
        )
    stats = (
        jtokm_routing_stats(
            logits,
            expert_idx,
            mask if mask.numel() else None,
            norm_eps=norm_eps,
        )
        if compute_aux
        else None
    )
    return output.reshape(orig_shape), stats


__all__ = [
    "jtok_apply",
    "jtok_reference",
    "jtokm_apply",
    "jtokm_auxiliary_loss",
    "jtokm_reference",
    "jtokm_routing_stats",
]
