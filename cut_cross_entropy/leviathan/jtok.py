"""CUDA JTok/JTok-M kernels coupled to the Leviathan coordinate.

This module is deliberately separate from the legacy Leviathan embedding
operator.  The public ``jtok_apply`` and ``jtokm_apply`` functions consume the
continuous coordinate produced by Leviathan and never alter the legacy
``leviathan_embedding`` contract.  This makes the extension opt-in at the
model boundary: when JTok is disabled, this module is not called and the
original Leviathan kernel remains the only embedding path.

The implementation has two layers:

* ``*_reference`` functions are an explicit semantic oracle for A/B tests.
  They use the same B-spline/product equations as the NeoLLM Torch
  implementation, but JTok-M evaluates only the selected experts.  They are
  never selected implicitly by the external-kernel route.
* The CUDA path uses a fused selected-mode/output projection and a general
  final normalization/modulation stage.  When one hidden tile is sufficient,
  the latter is folded into the same Triton launch.  It never materializes
  ``[tokens, experts, modes, hidden]``; the wide route uses only a compact
  ``[tokens, top_k, modes]`` mode workspace.  The mode product is accumulated
  inside each output tile.

On Torch 2.14 the operation is registered with ``torch.library.triton_op`` so
the wrapped Triton launches are visible to ``torch.compile`` at the explicit
JTok boundary; this lets Inductor optimize the surrounding router and model
operations without replacing the surface kernel.  Older Torch versions use
the ``custom_op`` compatibility registration.  Fake metadata and registered
autograd are provided for both registrations.  The
``backend="triton"`` mode is strict and raises when the kernel cannot be used;
``backend="torch"`` is the explicit reference baseline; ``backend="auto"`` is
intended only for library callers that deliberately want capability dispatch.

The registered external-kernel backward is strict: it launches the Triton
single-tile or multi-tile implementation and raises when CUDA/Triton is not
available.  It retains only the compact ``[tokens, top_k, modes]`` forward
mode product needed by the wide backward; no dense expert expansion is
retained by autograd.  The Torch implementation remains available only
through the explicit ``backend="torch"`` oracle for comparison.
"""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple, Optional

import torch
import torch.nn.functional as F

try:
    from torch.library import triton_op as _torch_triton_op
    from torch.library import wrap_triton as _torch_wrap_triton

    _TRITON_OP_AVAILABLE = True
except (ImportError, AttributeError):  # pragma: no cover - older Torch
    _torch_triton_op = None
    _torch_wrap_triton = None
    _TRITON_OP_AVAILABLE = False

from cut_cross_entropy.torch_2_14 import (
    TORCH_2_14_CUDA_KERNEL_CONTEXT,
    TORCH_2_14_MEMORY_ANNOTATIONS,
    annotate_tensors,
    cuda_kernel_region,
    mark_warmup_incomplete_once,
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
# The compact wide path is faster at the boundary itself because it computes
# the B-spline mode product once per token/route instead of once per hidden
# lane.  Keep the dispatch boundary in one named constant so forward and
# backward cannot silently diverge when the tile policy is revisited.
_SINGLE_TILE_HIDDEN_LIMIT = 256
# The token-local spline derivative kernel can process all seed coordinates in
# one program.  Bound its register/work footprint by the padded coefficient
# tile rather than by a model-specific batch or hidden size.  This admits the
# real NeoLLM geometry (d_seed=128, knots=16, modes=4) while keeping larger
# spline surfaces on the established scalar-grid path.
_TOKEN_PROJECTION_BLOCK_WORK_LIMIT = 8192
# The wide mode evaluator can reuse the B-spline basis across modes when the
# padded coefficient surface fits one Triton program.  This is a geometry
# budget, not a model-specific batch/hidden constant; larger surfaces retain
# the established one-program-per-mode implementation below.
_MODE_EVALUATION_WORK_LIMIT = 8192
# Keep the established reduction width explicit.  The tile policy below uses
# this value whenever a complete-row program would exceed its geometry budget.
_BACKWARD_MULTI_TILE_BLOCK_H = 256
# A complete hidden row can be processed by one backward program only when
# the combined hidden/vector work remains below this budget.  This guard is
# deliberately expressed in terms of the actual kernel geometry rather than
# the NeoLLM model dimensions.  Larger rows retain the established 256-lane
# multi-tile reduction to avoid excessive register pressure.
_BACKWARD_SINGLE_TILE_WORK_LIMIT = 131072
# A complete row up to 512 lanes is safe for the current compact backward
# geometry when the work budget above also passes.  This is a resource guard,
# not a model-shape assumption: rows wider than it, or rows whose seed/mode
# work exceeds the budget, retain the 256-lane multi-tile reduction.
# A complete-row tile is a correctness property, while projection-block
# tuning is an independent kernel choice.
_BACKWARD_SINGLE_TILE_MAX_H = 512
# These configurations tune launch resources only.  They must not select the
# mathematical ownership mode: ``SINGLE_HIDDEN_TILE`` is supplied by the
# geometry planner because it determines whether direct stores are race-free.
# The candidate set is intentionally small; the first call for a new geometry
# benchmarks it once and the Autotuner keeps the winning configuration in its
# per-process cache.
if _TRITON_AVAILABLE:
    _JTOK_WIDE_BACKWARD_CONFIGS = [
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    _JTOK_PROJECTION_GRAD_CONFIGS = [
        triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=2),
    ]
    _JTOK_TOKEN_PROJECTION_CONFIGS = [
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
    ]
else:  # pragma: no cover - CPU-only installations do not import Triton
    _JTOK_WIDE_BACKWARD_CONFIGS = []
    _JTOK_PROJECTION_GRAD_CONFIGS = []
    _JTOK_TOKEN_PROJECTION_CONFIGS = []
_USE_COMPOSABLE_TRITON_OP = _TRITON_OP_AVAILABLE


class _BackwardTilePlan(NamedTuple):
    """Launch and reduction invariants for the wide JTok backward."""

    block_h: int
    num_tiles: int
    work_items: int
    direct_store: bool


def _can_use_vectorized_token_projection(
    d_seed: int,
    knots: int,
    modes: int,
) -> bool:
    """Return whether the compact token-gradient block fits its work budget."""
    if d_seed < 1 or knots < 1 or modes < 1:
        return False
    knot_pad = 1 << (int(knots) - 1).bit_length()
    return (
        d_seed <= 128
        and d_seed * knot_pad * modes <= _TOKEN_PROJECTION_BLOCK_WORK_LIMIT
    )


def _can_use_vectorized_mode_evaluation(
    d_seed: int,
    knots: int,
    modes: int,
) -> bool:
    """Return whether mode evaluation fits the vectorized geometry budget."""
    if d_seed < 1 or knots < 1 or modes < 1:
        return False
    knot_pad = 1 << (int(knots) - 1).bit_length()
    mode_pad = 1 << (int(modes) - 1).bit_length()
    return (
        d_seed <= 128
        and mode_pad <= 32
        and d_seed * knot_pad * mode_pad <= _MODE_EVALUATION_WORK_LIMIT
    )


def _can_use_route_vectorized_mode_evaluation(
    d_seed: int,
    knots: int,
    modes: int,
    top_k: int,
) -> bool:
    """Return whether Top-K routes can share one B-spline basis program.

    The basis depends only on ``z`` and is identical for every selected
    expert.  The route-vector kernel keeps at most two selected experts in a
    program; larger ``top_k`` values retain the route-independent kernel so
    its coefficient tile does not become register-bound.
    """
    if top_k != 2:
        return False
    return _can_use_vectorized_mode_evaluation(d_seed, knots, modes)


def _backward_hidden_tile_plan(
    hidden: int,
    d_seed: int,
    modes: int,
    top_k: int,
) -> _BackwardTilePlan:
    """Choose the hidden tile width and whether it is a complete-row tile.

    The multi-tile kernel keeps compact token-local reductions, but each
    additional hidden tile requires atomic accumulation into the mode,
    residual, and route-weight workspaces.  A single tile can use direct
    stores for those values.  ``direct_store`` is therefore a mathematical
    ownership invariant, not merely a performance hint: it is true only when
    exactly one program covers the complete hidden reduction.  The resource
    budget is independent of batch size and model-specific constants and
    rejects invalid or unusually wide geometries.
    """
    if hidden < 1 or d_seed < 1 or modes < 1 or top_k < 1:
        return _BackwardTilePlan(0, 0, 0, False)
    padded_hidden = 1 << (int(hidden) - 1).bit_length()
    single_work = padded_hidden * (int(d_seed) + int(modes) * int(top_k))
    single_tile = (
        padded_hidden <= _BACKWARD_SINGLE_TILE_MAX_H
        and single_work <= _BACKWARD_SINGLE_TILE_WORK_LIMIT
    )
    if single_tile:
        return _BackwardTilePlan(
            padded_hidden,
            1,
            single_work,
            True,
        )
    block_h = min(_BACKWARD_MULTI_TILE_BLOCK_H, padded_hidden)
    return _BackwardTilePlan(
        block_h,
        (int(hidden) + block_h - 1) // block_h,
        block_h * (int(d_seed) + int(modes) * int(top_k)),
        False,
    )


def _can_use_single_hidden_tile_backward(
    hidden: int,
    d_seed: int,
    modes: int,
    top_k: int,
) -> bool:
    """Return whether the wide backward can use one complete hidden tile."""
    return _backward_hidden_tile_plan(hidden, d_seed, modes, top_k).direct_store


def _jtok_op_decorator(name: str):
    """Select the composable Torch 2.14 registration when available."""
    if _USE_COMPOSABLE_TRITON_OP:
        return _torch_triton_op(name, mutates_args={})
    return torch.library.custom_op(
        name,
        mutates_args=(),
        device_types="cuda",
        tags=(torch.Tag.cudagraph_unsafe,),
    )


def _jtok_fake_registration(op):
    """Register fake metadata for the Torch 2.14/legacy op APIs."""
    if _USE_COMPOSABLE_TRITON_OP:
        return lambda function: function
    return op.register_fake


def _jtok_register_autograd(op, backward, *, setup_context):
    """Register the external Triton backward with either operator API."""
    if _USE_COMPOSABLE_TRITON_OP:
        op.register_autograd(backward, setup_context=setup_context)
    else:
        torch.library.register_autograd(
            op,
            backward,
            setup_context=setup_context,
        )


def _jtok_wrap_kernel(kernel):
    """Expose Triton launches to ``triton_op`` without breaking old Torch."""
    if not _USE_COMPOSABLE_TRITON_OP or _torch_wrap_triton is None:
        return kernel
    return _torch_wrap_triton(kernel)


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


def _flatten_aux_input(
    values: torch.Tensor,
    leading_shape: tuple[int, ...],
    width: int,
    name: str,
) -> torch.Tensor:
    """Flatten an auxiliary feature without touching the already-flat seed.

    JTok-M validates ``router_state`` against ``delta_m`` and ``z_tilde``.
    Calling ``_flatten_inputs(router_state, z_tilde, ...)`` for that validation
    used to reshape and re-contiguous the same Leviathan seed a second time.
    Keep the validation, but flatten only the auxiliary router feature.
    """
    if values.ndim < 2 or tuple(values.shape[:-1]) != leading_shape:
        raise ValueError(
            f"{name} must share all leading dimensions with delta_m; got "
            f"{tuple(values.shape)} and {leading_shape + (width,)}"
        )
    if values.shape[-1] != width:
        raise ValueError(
            f"{name} last dimension must be {width}, got {values.shape[-1]}"
        )
    return values.reshape(-1, width).contiguous()


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
        row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
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
    def _jtok_modes_kernel_vectorized(
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
        MODE_PAD: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """Evaluate all selected modes while sharing each token's basis.

        The legacy wide evaluator launches one program per
        ``(token, route, mode)``.  Every such program rereads the same seed
        coordinate and rebuilds the same normalized B-spline basis.  This
        variant launches one program per ``(token, route)`` and carries a
        padded mode vector, so the basis is formed once and the coefficient
        surface is reduced across modes in parallel.  ``MODE_PAD`` and the
        geometry guard in Python keep irregular mode counts masked without
        making a model-specific assumption.
        """
        pid = tl.program_id(0)
        row = pid // TOP_K
        slot = pid % TOP_K
        row_mask = row < N
        expert = tl.load(
            expert_idx_ptr + row * TOP_K + slot,
            mask=row_mask,
            other=0,
        ).to(tl.int32)
        if HAS_MASK:
            row_valid = tl.load(
                valid_ptr + row,
                mask=row_mask,
                other=0,
            ).to(tl.int1)
        else:
            row_valid = row_mask

        knot_offsets = tl.arange(0, KNOT_PAD)
        knot_mask = knot_offsets < NUM_KNOTS
        mode_offsets = tl.arange(0, MODE_PAD)
        mode_mask = mode_offsets < NUM_MODES
        grid = tl.load(
            grid_ptr + knot_offsets,
            mask=knot_mask,
            other=0.0,
        ).to(tl.float32)
        scale = float(max(int(NUM_KNOTS) - 1, 0))
        log_acc = tl.zeros((MODE_PAD,), dtype=tl.float32)
        neg_acc = tl.zeros((MODE_PAD,), dtype=tl.int32)

        for d in tl.range(0, D_SEED):
            x = tl.load(
                z_ptr + row * D_SEED + d,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
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
            basis = tl.where(knot_mask, basis, 0.0)
            basis = basis / tl.maximum(tl.sum(basis, axis=0), 1e-12)
            coeff = tl.load(
                coeff_ptr
                + (
                    (
                        (expert * NUM_MODES + mode_offsets[:, None]) * D_SEED
                        + d
                    )
                    * NUM_KNOTS
                    + knot_offsets[None, :]
                ),
                mask=(mode_mask[:, None] & knot_mask[None, :] & row_mask),
                other=0.0,
            ).to(tl.float32)
            phi = tl.sum(basis[None, :] * coeff, axis=1)
            log_acc += tl.log(tl.abs(phi) + 1e-9)
            neg_acc += (phi < 0).to(tl.int32)

        value = (1.0 - 2.0 * (neg_acc & 1).to(tl.float32)) * tl.exp(log_acc)
        value = tl.where(row_valid & row_mask & mode_mask, value, 0.0)
        tl.store(
            modes_ptr
            + (row * TOP_K + slot) * NUM_MODES
            + mode_offsets,
            value.to(modes_ptr.dtype.element_ty),
            mask=row_mask & mode_mask,
        )

    @triton.jit
    def _jtok_modes_kernel_route_vectorized(
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
        MODE_PAD: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """Evaluate two selected routes while sharing each token's basis.

        JTok-M selects multiple experts for the same token, but the normalized
        B-spline basis is a function of ``z`` only.  The ordinary vectorized
        kernel launches one program per route and repeats that basis work.
        This route-vector form keeps the two selected experts in one program,
        reuses the basis, and writes the same compact ``[N, TOP_K, M]``
        workspace.  The Python geometry guard restricts it to ``TOP_K=2`` so
        the extra coefficient tile does not make unusual routes register-bound.
        """
        row = tl.program_id(0)
        row_mask = row < N
        slots = tl.arange(0, TOP_K)
        route_mask = tl.load(
            expert_idx_ptr + row * TOP_K + slots,
            mask=row_mask,
            other=0,
        ).to(tl.int32)
        if HAS_MASK:
            row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
        else:
            row_valid = row_mask

        knot_offsets = tl.arange(0, KNOT_PAD)
        knot_mask = knot_offsets < NUM_KNOTS
        mode_offsets = tl.arange(0, MODE_PAD)
        mode_mask = mode_offsets < NUM_MODES
        grid = tl.load(
            grid_ptr + knot_offsets,
            mask=knot_mask,
            other=0.0,
        ).to(tl.float32)
        scale = float(max(int(NUM_KNOTS) - 1, 0))
        log_acc = tl.zeros((TOP_K, MODE_PAD), dtype=tl.float32)
        neg_acc = tl.zeros((TOP_K, MODE_PAD), dtype=tl.int32)

        for d in tl.range(0, D_SEED):
            x = tl.load(
                z_ptr + row * D_SEED + d,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
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
            basis = tl.where(knot_mask, basis, 0.0)
            basis = basis / tl.maximum(tl.sum(basis, axis=0), 1e-12)
            coeff = tl.load(
                coeff_ptr
                + (
                    (
                        (
                            route_mask[:, None, None] * NUM_MODES
                            + mode_offsets[None, :, None]
                        )
                        * D_SEED
                        + d
                    )
                    * NUM_KNOTS
                    + knot_offsets[None, None, :]
                ),
                mask=(
                    row_mask
                    & (slots[:, None, None] < TOP_K)
                    & mode_mask[None, :, None]
                    & knot_mask[None, None, :]
                ),
                other=0.0,
            ).to(tl.float32)
            phi = tl.sum(basis[None, None, :] * coeff, axis=2)
            log_acc += tl.log(tl.abs(phi) + 1e-9)
            neg_acc += (phi < 0).to(tl.int32)

        value = (1.0 - 2.0 * (neg_acc & 1).to(tl.float32)) * tl.exp(log_acc)
        value = tl.where(row_valid & row_mask, value, 0.0)
        value = tl.where(mode_mask[None, :], value, 0.0)
        tl.store(
            modes_ptr
            + row * TOP_K * NUM_MODES
            + slots[:, None] * NUM_MODES
            + mode_offsets[None, :],
            value.to(modes_ptr.dtype.element_ty),
            mask=row_mask & mode_mask[None, :],
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
            valid = valid & tl.load(valid_ptr + row, mask=row < N, other=0).to(tl.int1)
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

    @triton.jit
    def _jtok_project_fused_kernel(
        z_ptr,
        coeff_ptr,
        grid_ptr,
        spline_out_ptr,
        residual_out_ptr,
        expert_idx_ptr,
        selected_weights_ptr,
        valid_ptr,
        surface_ptr,
        norm_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_N: tl.constexpr,
        KNOT_PAD: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """Evaluate selected modes and project them in one row tile.

        The old path wrote a temporary ``[N, K, M]`` activation and launched a
        second kernel to consume it.  This kernel computes the same mode
        product in FP32 inside each hidden tile, then immediately accumulates
        the selected output and residual projections.  A tile may recompute
        the inexpensive mode product when ``HIDDEN > BLOCK_N``; the caller
        chooses a larger tile for this fused path to keep that duplication
        bounded.  The output and norm contracts are unchanged.
        """
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = row < N
        col_mask = cols < HIDDEN
        active_mask = row_mask & col_mask
        knot_offsets = tl.arange(0, KNOT_PAD)
        grid = tl.load(
            grid_ptr + knot_offsets,
            mask=knot_offsets < NUM_KNOTS,
            other=0.0,
        ).to(tl.float32)
        scale = float(max(int(NUM_KNOTS) - 1, 0))
        mixed = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for slot in tl.range(0, TOP_K):
            expert = tl.load(
                expert_idx_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0,
            ).to(tl.int32)
            weight = tl.load(
                selected_weights_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            values = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for mode in tl.range(0, NUM_MODES):
                log_acc = 0.0
                neg_acc = 0
                for d in tl.range(0, D_SEED):
                    x = tl.load(
                        z_ptr + row * D_SEED + d,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
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
                    basis = tl.where(
                        knot_offsets < NUM_KNOTS,
                        basis,
                        0.0,
                    )
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
                mode_value = (
                    1.0 - 2.0 * (neg_acc & 1).to(tl.float32)
                ) * tl.exp(log_acc)
                # Preserve the old three-stage path's activation-dtype
                # boundary: modes were stored as BF16/FP16 before projection.
                mode_value = mode_value.to(spline_out_ptr.dtype.element_ty).to(
                    tl.float32
                )
                out_weight = tl.load(
                    spline_out_ptr
                    + (expert * NUM_MODES + mode) * HIDDEN
                    + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                values += mode_value * out_weight

            residual = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d in tl.range(0, D_SEED):
                z_value = tl.load(
                    z_ptr + row * D_SEED + d,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_weight = tl.load(
                    residual_out_ptr + (expert * D_SEED + d) * HIDDEN + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                residual += z_value * residual_weight
            mixed += weight * (values + residual)

        if HAS_MASK:
            row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
            mixed = tl.where(row_valid, mixed, 0.0)
        mixed = tl.where(active_mask, mixed, 0.0)
        tl.store(
            surface_ptr + row * HIDDEN + cols,
            mixed.to(surface_ptr.dtype.element_ty),
            mask=active_mask,
        )
        tl.atomic_add(norm_ptr + row, tl.sum(mixed * mixed, axis=0), mask=row_mask)

    @triton.jit
    def _jtok_project_finalize_fused_kernel(
        delta_ptr,
        z_ptr,
        coeff_ptr,
        grid_ptr,
        spline_out_ptr,
        residual_out_ptr,
        expert_idx_ptr,
        selected_weights_ptr,
        modes_ptr,
        scaler_ptr,
        valid_ptr,
        output_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_N: tl.constexpr,
        KNOT_PAD: tl.constexpr,
        NORM_EPS: tl.constexpr,
        RESIDUAL_SCALE: tl.constexpr,
        MIXTURE: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """Fuse selected surface, norm, and modulation for one hidden tile.

        The reduction is valid only when one program owns the complete hidden
        row.  The Python dispatcher therefore selects this kernel only for
        ``hidden < 256``; the boundary itself uses the compact
        surface/norm/finalize sequence above.  Keeping this condition explicit
        avoids a hidden cross-tile reduction and preserves the odd-geometry
        contract.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        row_mask = row < N
        col_mask = cols < HIDDEN
        active_mask = row_mask & col_mask
        knot_offsets = tl.arange(0, KNOT_PAD)
        grid = tl.load(
            grid_ptr + knot_offsets,
            mask=knot_offsets < NUM_KNOTS,
            other=0.0,
        ).to(tl.float32)
        scale = float(max(int(NUM_KNOTS) - 1, 0))
        mixed = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if HAS_MASK:
            row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
        else:
            row_valid = row_mask

        for slot in tl.range(0, TOP_K):
            expert = tl.load(
                expert_idx_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0,
            ).to(tl.int32)
            weight = tl.load(
                selected_weights_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            values = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for mode in tl.range(0, NUM_MODES):
                log_acc = 0.0
                neg_acc = 0
                for d in tl.range(0, D_SEED):
                    x = tl.load(
                        z_ptr + row * D_SEED + d,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
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
                mode_value = (
                    1.0 - 2.0 * (neg_acc & 1).to(tl.float32)
                ) * tl.exp(log_acc)
                mode_value = mode_value.to(spline_out_ptr.dtype.element_ty).to(
                    tl.float32
                )
                mode_value = tl.where(row_valid, mode_value, 0.0)
                tl.store(
                    modes_ptr
                    + (row * TOP_K + slot) * NUM_MODES
                    + mode,
                    mode_value.to(modes_ptr.dtype.element_ty),
                    mask=row_mask,
                )
                out_weight = tl.load(
                    spline_out_ptr
                    + (expert * NUM_MODES + mode) * HIDDEN
                    + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                values += mode_value * out_weight

            residual = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d in tl.range(0, D_SEED):
                z_value = tl.load(
                    z_ptr + row * D_SEED + d,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_weight = tl.load(
                    residual_out_ptr + (expert * D_SEED + d) * HIDDEN + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                residual += z_value * residual_weight
            mixed += weight * (values + residual)

        mixed = tl.where(row_valid, mixed, 0.0)
        norm = tl.sqrt(tl.sum(mixed * mixed, axis=0)) + NORM_EPS
        surface = mixed.to(output_ptr.dtype.element_ty).to(tl.float32)
        scaler = tl.load(scaler_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        direction = surface / norm
        delta = tl.load(delta_ptr + row * HIDDEN + cols, mask=active_mask, other=0.0).to(
            tl.float32
        )
        if MIXTURE:
            value = delta + RESIDUAL_SCALE * scaler * direction
        else:
            value = delta * (1.0 + scaler * direction)
        value = tl.where(row_valid & col_mask, value, delta)
        tl.store(output_ptr + row * HIDDEN + cols, value.to(output_ptr.dtype.element_ty), mask=active_mask)

    @triton.jit
    def _jtok_backward_single_tile_kernel(
        delta_ptr,
        z_ptr,
        coeff_ptr,
        grid_ptr,
        spline_out_ptr,
        residual_out_ptr,
        expert_idx_ptr,
        selected_weights_ptr,
        scaler_ptr,
        grad_out_ptr,
        valid_ptr,
        grad_delta_ptr,
        grad_z_ptr,
        grad_coeff_ptr,
        grad_spline_out_ptr,
        grad_residual_out_ptr,
        grad_scaler_ptr,
        grad_weights_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        KNOT_PAD: tl.constexpr,
        NORM_EPS: tl.constexpr,
        RESIDUAL_SCALE: tl.constexpr,
        MIXTURE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        PARAM_GRADS: tl.constexpr,
    ):
        """Recompute one complete row and accumulate its backward gradients.

        This kernel intentionally handles only one hidden tile per token.  It
        is used when ``HIDDEN < 256`` so the norm reduction is local and no
        ``[tokens, top_k, modes, hidden]`` or per-token parameter-gradient
        workspace is needed.  Wider rows use the multi-tile Triton reduction
        below, so the registered external path remains kernel-only for every
        supported hidden size.

        The forward casts the surface to the activation dtype before dividing
        by the FP32 norm.  The backward mirrors that boundary: ``surface`` is
        the cast value while the norm derivative uses the pre-cast ``mixed``
        accumulator.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        row_mask = row < N
        col_mask = cols < HIDDEN
        active_mask = row_mask & col_mask
        safe_cols = tl.minimum(cols, HIDDEN - 1)
        if HAS_MASK:
            row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
        else:
            row_valid = row_mask

        knot_offsets = tl.arange(0, KNOT_PAD)
        grid = tl.load(
            grid_ptr + knot_offsets,
            mask=knot_offsets < NUM_KNOTS,
            other=0.0,
        ).to(tl.float32)
        grid_scale = float(max(int(NUM_KNOTS) - 1, 0))

        # Recompute the selected surface exactly as the fused forward kernel.
        mixed = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for slot in tl.range(0, TOP_K):
            expert = tl.load(
                expert_idx_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0,
            ).to(tl.int32)
            weight = tl.load(
                selected_weights_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            values = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for mode in tl.range(0, NUM_MODES):
                log_acc = 0.0
                negative = 0
                for d in tl.range(0, D_SEED):
                    x = tl.load(
                        z_ptr + row * D_SEED + d,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
                    distance = tl.abs(x - grid) * grid_scale
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
                    negative += (phi < 0).to(tl.int32)
                mode_value = (
                    1.0 - 2.0 * (negative & 1).to(tl.float32)
                ) * tl.exp(log_acc)
                mode_value = mode_value.to(
                    spline_out_ptr.dtype.element_ty
                ).to(tl.float32)
                out_weight = tl.load(
                    spline_out_ptr
                    + (expert * NUM_MODES + mode) * HIDDEN
                    + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                values += mode_value * out_weight

            residual_value = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for d in tl.range(0, D_SEED):
                z_value = tl.load(
                    z_ptr + row * D_SEED + d,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_weight = tl.load(
                    residual_out_ptr + (expert * D_SEED + d) * HIDDEN + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_value += z_value * residual_weight
            mixed += weight * (values + residual_value)

        mixed = tl.where(row_valid, mixed, 0.0)
        norm = tl.sqrt(tl.sum(mixed * mixed, axis=0)) + NORM_EPS
        surface = mixed.to(grad_delta_ptr.dtype.element_ty).to(tl.float32)
        grad_out = tl.load(
            grad_out_ptr + row * HIDDEN + cols,
            mask=active_mask,
            other=0.0,
        ).to(tl.float32)
        delta_value = tl.load(
            delta_ptr + row * HIDDEN + cols,
            mask=active_mask,
            other=0.0,
        ).to(tl.float32)
        scaler_value = tl.load(
            scaler_ptr + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        if MIXTURE:
            surface_grad_factor = RESIDUAL_SCALE * scaler_value * grad_out
            dot = tl.sum(surface_grad_factor * mixed, axis=0)
            grad_surface = (
                surface_grad_factor / norm
                - mixed * dot / (norm * norm * norm)
            )
            grad_delta = grad_out
            grad_scaler = RESIDUAL_SCALE * grad_out * surface / norm
        else:
            surface_grad_factor = grad_out * delta_value * scaler_value
            dot = tl.sum(surface_grad_factor * mixed, axis=0)
            grad_surface = (
                surface_grad_factor / norm
                - mixed * dot / (norm * norm * norm)
            )
            grad_delta = grad_out * (1.0 + scaler_value * surface / norm)
            grad_scaler = grad_out * delta_value * surface / norm

        valid_active = active_mask & row_valid
        grad_surface = tl.where(valid_active, grad_surface, 0.0)
        grad_scaler = tl.where(valid_active, grad_scaler, 0.0)
        tl.store(
            grad_delta_ptr + row * HIDDEN + cols,
            grad_delta.to(grad_delta_ptr.dtype.element_ty),
            mask=active_mask,
        )
        # Padded lanes carry zero gradients.  Clamp their address so the
        # higher-order Triton wrapper sees an unmasked vector atomic; this is
        # required by Torch 2.14's accessed-tensor analysis.
        tl.atomic_add(grad_scaler_ptr + safe_cols, grad_scaler)

        # Every selected expert receives one token's surface gradient.  The
        # parameter arrays are shared across rows, hence FP32 atomics are used
        # for the small persistent gradient tensors.
        for slot in tl.range(0, TOP_K):
            expert = tl.load(
                expert_idx_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0,
            ).to(tl.int32)
            weight = tl.load(
                selected_weights_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            slot_values = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for mode in tl.range(0, NUM_MODES):
                log_acc = 0.0
                negative = 0
                for d in tl.range(0, D_SEED):
                    x = tl.load(
                        z_ptr + row * D_SEED + d,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
                    distance = tl.abs(x - grid) * grid_scale
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
                    negative += (phi < 0).to(tl.int32)
                mode_value = (
                    1.0 - 2.0 * (negative & 1).to(tl.float32)
                ) * tl.exp(log_acc)
                mode_value = mode_value.to(
                    spline_out_ptr.dtype.element_ty
                ).to(tl.float32)
                out_weight = tl.load(
                    spline_out_ptr
                    + (expert * NUM_MODES + mode) * HIDDEN
                    + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                slot_values += mode_value * out_weight

                grad_value = weight * grad_surface
                grad_mode = tl.sum(grad_value * out_weight, axis=0)
                if PARAM_GRADS:
                    tl.atomic_add(
                        grad_spline_out_ptr
                        + (expert * NUM_MODES + mode) * HIDDEN
                        + safe_cols,
                        mode_value * grad_value,
                    )

                for d in tl.range(0, D_SEED):
                    x = tl.load(
                        z_ptr + row * D_SEED + d,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
                    diff = x - grid
                    distance = tl.abs(diff) * grid_scale
                    basis_raw = tl.where(
                        distance < 0.5,
                        0.75 - distance * distance,
                        tl.where(
                            distance < 1.5,
                            0.5 * (1.5 - distance) * (1.5 - distance),
                            0.0,
                        ),
                    )
                    basis_raw = tl.where(
                        knot_offsets < NUM_KNOTS, basis_raw, 0.0
                    )
                    sign_x = tl.where(
                        diff > 0.0,
                        1.0,
                        tl.where(diff < 0.0, -1.0, 0.0),
                    )
                    basis_derivative = tl.where(
                        distance < 0.5,
                        -2.0 * distance * grid_scale * sign_x,
                        tl.where(
                            distance < 1.5,
                            -(1.5 - distance) * grid_scale * sign_x,
                            0.0,
                        ),
                    )
                    basis_derivative = tl.where(
                        knot_offsets < NUM_KNOTS,
                        basis_derivative,
                        0.0,
                    )
                    raw_sum = tl.sum(basis_raw, axis=0)
                    safe_sum = tl.maximum(raw_sum, 1e-12)
                    coeff = tl.load(
                        coeff_ptr
                        + (((expert * NUM_MODES + mode) * D_SEED + d) * NUM_KNOTS)
                        + knot_offsets,
                        mask=knot_offsets < NUM_KNOTS,
                        other=0.0,
                    ).to(tl.float32)
                    weighted = tl.sum(basis_raw * coeff, axis=0)
                    derivative_sum = tl.sum(basis_derivative, axis=0)
                    derivative_weighted = tl.sum(
                        basis_derivative * coeff,
                        axis=0,
                    )
                    phi = weighted / safe_sum
                    dphi_dz = tl.where(
                        raw_sum > 1e-12,
                        (
                            derivative_weighted * safe_sum
                            - weighted * derivative_sum
                        )
                        / (safe_sum * safe_sum),
                        0.0,
                    )
                    phi_sign = tl.where(phi < 0.0, -1.0, 1.0)
                    grad_phi = (
                        grad_mode
                        * mode_value
                        * phi_sign
                        / (tl.abs(phi) + 1e-9)
                    )
                    grad_phi = tl.where(row_valid, grad_phi, 0.0)
                    tl.atomic_add(
                        grad_coeff_ptr
                        + ((expert * NUM_MODES + mode) * D_SEED + d) * NUM_KNOTS
                        + tl.minimum(knot_offsets, NUM_KNOTS - 1),
                        tl.where(
                            knot_offsets < NUM_KNOTS,
                            grad_phi * basis_raw / safe_sum,
                            0.0,
                        ),
                    )
                    tl.atomic_add(
                        grad_z_ptr + row * D_SEED + d,
                        grad_phi * dphi_dz,
                        mask=row_valid,
                    )

            residual_value = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for d in tl.range(0, D_SEED):
                z_value = tl.load(
                    z_ptr + row * D_SEED + d,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_weight = tl.load(
                    residual_out_ptr + (expert * D_SEED + d) * HIDDEN + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_value += z_value * residual_weight
                grad_value = weight * grad_surface
                tl.atomic_add(
                    grad_z_ptr + row * D_SEED + d,
                    tl.sum(grad_value * residual_weight, axis=0),
                    mask=row_valid,
                )
                if PARAM_GRADS:
                    tl.atomic_add(
                        grad_residual_out_ptr
                        + (expert * D_SEED + d) * HIDDEN
                        + safe_cols,
                        z_value * grad_value,
                    )
            slot_values += residual_value
            tl.store(
                grad_weights_ptr + row * TOP_K + slot,
                tl.sum(grad_surface * slot_values, axis=0),
                mask=row_mask,
            )

    @triton.jit
    def _jtok_backward_multi_tile_kernel(
        delta_ptr,
        z_ptr,
        spline_coeff_ptr,
        grid_ptr,
        spline_out_ptr,
        residual_out_ptr,
        expert_idx_ptr,
        selected_weights_ptr,
        modes_ptr,
        scaler_ptr,
        surface_ptr,
        norm_ptr,
        norm_dot_ptr,
        grad_out_ptr,
        valid_ptr,
        grad_delta_ptr,
        grad_z_ptr,
        grad_coeff_ptr,
        grad_spline_out_ptr,
        grad_residual_out_ptr,
        grad_scaler_ptr,
        grad_weights_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        KNOT_PAD: tl.constexpr,
        NORM_EPS: tl.constexpr,
        RESIDUAL_SCALE: tl.constexpr,
        MIXTURE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        PARAM_GRADS: tl.constexpr,
    ):
        """Backward for rows whose hidden dimension spans multiple tiles.

        ``_jtok_project_kernel`` has already written the pre-cast FP32 surface
        and the row-wise squared-norm reduction.  This pass consumes
        one hidden tile at a time and accumulates token-local gradients with
        atomics.  Projection-parameter gradients are disabled here for the
        wide path and reduced by ``_jtok_backward_projection_grad_kernel``
        over token blocks after this pass.  In particular, it never gathers
        ``[tokens, top_k, d_seed, hidden]`` residual weights or
        ``[tokens, top_k, modes, hidden]`` selected projections.
        """
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK_H + tl.arange(0, BLOCK_H)
        row_mask = row < N
        col_mask = cols < HIDDEN
        active_mask = row_mask & col_mask
        safe_cols = tl.minimum(cols, HIDDEN - 1)
        if HAS_MASK:
            row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
        else:
            row_valid = row_mask

        knot_offsets = tl.arange(0, KNOT_PAD)
        grid = tl.load(
            grid_ptr + knot_offsets,
            mask=knot_offsets < NUM_KNOTS,
            other=0.0,
        ).to(tl.float32)
        grid_scale = float(max(int(NUM_KNOTS) - 1, 0))
        mixed = tl.load(
            surface_ptr + row * HIDDEN + cols,
            mask=active_mask,
            other=0.0,
        ).to(tl.float32)
        norm = tl.sqrt(tl.load(norm_ptr + row, mask=row_mask, other=0.0)) + NORM_EPS
        grad_out = tl.load(
            grad_out_ptr + row * HIDDEN + cols,
            mask=active_mask,
            other=0.0,
        ).to(tl.float32)
        delta_value = tl.load(
            delta_ptr + row * HIDDEN + cols,
            mask=active_mask,
            other=0.0,
        ).to(tl.float32)
        scaler_value = tl.load(
            scaler_ptr + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        if MIXTURE:
            surface_grad_factor = RESIDUAL_SCALE * scaler_value * grad_out
            dot = tl.load(norm_dot_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
            grad_surface = (
                surface_grad_factor / norm
                - mixed * dot / (norm * norm * norm)
            )
            grad_delta = grad_out
            grad_scaler = RESIDUAL_SCALE * grad_out * mixed / norm
        else:
            surface_grad_factor = grad_out * delta_value * scaler_value
            dot = tl.load(norm_dot_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
            grad_surface = (
                surface_grad_factor / norm
                - mixed * dot / (norm * norm * norm)
            )
            grad_delta = grad_out * (1.0 + scaler_value * mixed / norm)
            grad_scaler = grad_out * delta_value * mixed / norm

        valid_active = active_mask & row_valid
        grad_surface = tl.where(valid_active, grad_surface, 0.0)
        grad_scaler = tl.where(valid_active, grad_scaler, 0.0)
        # Reuse the FP32 surface workspace as the hand-off buffer for the
        # block-reduced projection-gradient pass. Invalid/padded rows are
        # already zeroed above, so the second pass needs no mask tensor.
        tl.store(
            surface_ptr + row * HIDDEN + cols,
            grad_surface,
            mask=active_mask,
        )
        tl.store(
            grad_delta_ptr + row * HIDDEN + cols,
            grad_delta.to(grad_delta_ptr.dtype.element_ty),
            mask=active_mask,
        )
        tl.atomic_add(grad_scaler_ptr + safe_cols, grad_scaler)

        for slot in tl.range(0, TOP_K):
            expert = tl.load(
                expert_idx_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0,
            ).to(tl.int32)
            weight = tl.load(
                selected_weights_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            slot_values = tl.zeros((BLOCK_H,), dtype=tl.float32)

            for mode in tl.range(0, NUM_MODES):
                # The compact mode product is evaluated once per token/slot
                # before this tiled pass.  Recomputing it here for every
                # hidden tile multiplied the B-spline work by ceil(H/256)
                # and was the dominant cost on wide NeoLLM rows.
                mode_value = tl.load(
                    modes_ptr + (row * TOP_K + slot) * NUM_MODES + mode,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                out_weight = tl.load(
                    spline_out_ptr
                    + (expert * NUM_MODES + mode) * HIDDEN
                    + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                slot_values += mode_value * out_weight

                grad_value = weight * grad_surface
                grad_mode = tl.sum(grad_value * out_weight, axis=0)
                if PARAM_GRADS:
                    tl.atomic_add(
                        grad_spline_out_ptr
                        + (expert * NUM_MODES + mode) * HIDDEN
                        + safe_cols,
                        mode_value * grad_value,
                    )

                for d in tl.range(0, D_SEED):
                    x = tl.load(
                        z_ptr + row * D_SEED + d,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
                    diff = x - grid
                    distance = tl.abs(diff) * grid_scale
                    basis_raw = tl.where(
                        distance < 0.5,
                        0.75 - distance * distance,
                        tl.where(
                            distance < 1.5,
                            0.5 * (1.5 - distance) * (1.5 - distance),
                            0.0,
                        ),
                    )
                    basis_raw = tl.where(
                        knot_offsets < NUM_KNOTS,
                        basis_raw,
                        0.0,
                    )
                    sign_x = tl.where(
                        diff > 0.0,
                        1.0,
                        tl.where(diff < 0.0, -1.0, 0.0),
                    )
                    basis_derivative = tl.where(
                        distance < 0.5,
                        -2.0 * distance * grid_scale * sign_x,
                        tl.where(
                            distance < 1.5,
                            -(1.5 - distance) * grid_scale * sign_x,
                            0.0,
                        ),
                    )
                    basis_derivative = tl.where(
                        knot_offsets < NUM_KNOTS,
                        basis_derivative,
                        0.0,
                    )
                    raw_sum = tl.sum(basis_raw, axis=0)
                    safe_sum = tl.maximum(raw_sum, 1e-12)
                    coeff = tl.load(
                        spline_coeff_ptr
                        + (((expert * NUM_MODES + mode) * D_SEED + d) * NUM_KNOTS)
                        + knot_offsets,
                        mask=knot_offsets < NUM_KNOTS,
                        other=0.0,
                    ).to(tl.float32)
                    weighted = tl.sum(basis_raw * coeff, axis=0)
                    derivative_sum = tl.sum(basis_derivative, axis=0)
                    derivative_weighted = tl.sum(
                        basis_derivative * coeff,
                        axis=0,
                    )
                    phi = weighted / safe_sum
                    dphi_dz = tl.where(
                        raw_sum > 1e-12,
                        (
                            derivative_weighted * safe_sum
                            - weighted * derivative_sum
                        )
                        / (safe_sum * safe_sum),
                        0.0,
                    )
                    phi_sign = tl.where(phi < 0.0, -1.0, 1.0)
                    grad_phi = (
                        grad_mode
                        * mode_value
                        * phi_sign
                        / (tl.abs(phi) + 1e-9)
                    )
                    grad_phi = tl.where(row_valid, grad_phi, 0.0)
                    tl.atomic_add(
                        grad_coeff_ptr
                        + ((expert * NUM_MODES + mode) * D_SEED + d) * NUM_KNOTS
                        + tl.minimum(knot_offsets, NUM_KNOTS - 1),
                        tl.where(
                            knot_offsets < NUM_KNOTS,
                            grad_phi * basis_raw / safe_sum,
                            0.0,
                        ),
                    )
                    tl.atomic_add(
                        grad_z_ptr + row * D_SEED + d,
                        grad_phi * dphi_dz,
                        mask=row_valid,
                    )

            residual_value = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for d in tl.range(0, D_SEED):
                z_value = tl.load(
                    z_ptr + row * D_SEED + d,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_weight = tl.load(
                    residual_out_ptr + (expert * D_SEED + d) * HIDDEN + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_value += z_value * residual_weight
                grad_value = weight * grad_surface
                tl.atomic_add(
                    grad_z_ptr + row * D_SEED + d,
                    tl.sum(grad_value * residual_weight, axis=0),
                    mask=row_valid,
                )
                if PARAM_GRADS:
                    tl.atomic_add(
                        grad_residual_out_ptr
                        + (expert * D_SEED + d) * HIDDEN
                        + safe_cols,
                        z_value * grad_value,
                    )
            slot_values += residual_value
            tl.atomic_add(
                grad_weights_ptr + row * TOP_K + slot,
                tl.sum(grad_surface * slot_values, axis=0),
                mask=row_mask,
            )

    @triton.jit
    def _jtok_backward_norm_dot_kernel(
        surface_ptr,
        delta_ptr,
        scaler_ptr,
        grad_out_ptr,
        valid_ptr,
        dot_ptr,
        N,
        HIDDEN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NUM_TILES: tl.constexpr,
        RESIDUAL_SCALE: tl.constexpr,
        MIXTURE: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """Reduce the normalization dot product over the complete hidden row.

        For ``y = s / (||s|| + eps)``, the backward contains
        ``dot(a, s)`` over *all* hidden coordinates.  The tiled projection
        backward cannot form that scalar independently in each tile without
        changing the derivative.  This token-owned reduction writes one
        scalar per row, so the following tile pass can compute the exact same
        derivative while retaining its compact token/route workspaces.
        """
        row = tl.program_id(0)
        row_mask = row < N
        if HAS_MASK:
            row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
        else:
            row_valid = row_mask
        dot = tl.zeros((), dtype=tl.float32)
        for tile in tl.range(0, NUM_TILES):
            cols = tile * BLOCK_H + tl.arange(0, BLOCK_H)
            active = row_mask & (cols < HIDDEN) & row_valid
            mixed = tl.load(
                surface_ptr + row * HIDDEN + cols,
                mask=row_mask & (cols < HIDDEN),
                other=0.0,
            ).to(tl.float32)
            grad_out = tl.load(
                grad_out_ptr + row * HIDDEN + cols,
                mask=row_mask & (cols < HIDDEN),
                other=0.0,
            ).to(tl.float32)
            scaler = tl.load(
                scaler_ptr + cols,
                mask=cols < HIDDEN,
                other=0.0,
            ).to(tl.float32)
            if MIXTURE:
                factor = RESIDUAL_SCALE * scaler * grad_out
            else:
                delta = tl.load(
                    delta_ptr + row * HIDDEN + cols,
                    mask=row_mask & (cols < HIDDEN),
                    other=0.0,
                ).to(tl.float32)
                factor = grad_out * delta * scaler
            dot += tl.sum(tl.where(active, factor * mixed, 0.0), axis=0)
        tl.store(dot_ptr + row, dot, mask=row_mask)

    @triton.autotune(
        configs=_JTOK_WIDE_BACKWARD_CONFIGS,
        key=[
            "N",
            "D_SEED",
            "NUM_MODES",
            "TOP_K",
            "HIDDEN",
            "BLOCK_H",
            "MIXTURE",
            "HAS_MASK",
            "SINGLE_HIDDEN_TILE",
        ],
        reset_to_zero=[
            "grad_mode_ptr",
            "grad_residual_ptr",
            "grad_scaler_ptr",
            "grad_weights_ptr",
        ],
        # ``surface_ptr`` is intentionally reused as the input activation and
        # as the hand-off buffer for projection gradients.  Every benchmark
        # candidate overwrites it, so it must be restored before the next
        # candidate and before the selected configuration runs.
        restore_value=["surface_ptr"],
    )
    @triton.jit
    def _jtok_backward_multi_tile_fast_kernel(
        delta_ptr,
        z_ptr,
        spline_out_ptr,
        residual_out_ptr,
        expert_idx_ptr,
        selected_weights_ptr,
        modes_ptr,
        grad_mode_ptr,
        grad_residual_ptr,
        scaler_ptr,
        surface_ptr,
        norm_ptr,
        norm_dot_ptr,
        grad_out_ptr,
        valid_ptr,
        grad_delta_ptr,
        grad_scaler_ptr,
        grad_weights_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NORM_EPS: tl.constexpr,
        RESIDUAL_SCALE: tl.constexpr,
        MIXTURE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        SINGLE_HIDDEN_TILE: tl.constexpr,
    ):
        """Wide backward pass with token-local scalar reductions.

        Each hidden tile contributes only to the compact ``grad_mode`` and
        ``grad_residual`` workspaces.  The B-spline derivative is evaluated
        later once per token/slot/seed coordinate, rather than once per
        hidden tile.  The workspaces are proportional to
        ``tokens * top_k * (modes + d_seed)`` and never to hidden size.
        """
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK_H + tl.arange(0, BLOCK_H)
        row_mask = row < N
        col_mask = cols < HIDDEN
        active_mask = row_mask & col_mask
        safe_cols = tl.minimum(cols, HIDDEN - 1)
        if HAS_MASK:
            row_valid = tl.load(valid_ptr + row, mask=row_mask, other=0).to(tl.int1)
        else:
            row_valid = row_mask

        mixed = tl.load(
            surface_ptr + row * HIDDEN + cols,
            mask=active_mask,
            other=0.0,
        ).to(tl.float32)
        norm = tl.sqrt(tl.load(norm_ptr + row, mask=row_mask, other=0.0)) + NORM_EPS
        grad_out = tl.load(
            grad_out_ptr + row * HIDDEN + cols,
            mask=active_mask,
            other=0.0,
        ).to(tl.float32)
        delta_value = tl.load(
            delta_ptr + row * HIDDEN + cols,
            mask=active_mask,
            other=0.0,
        ).to(tl.float32)
        scaler_value = tl.load(
            scaler_ptr + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        if MIXTURE:
            surface_grad_factor = RESIDUAL_SCALE * scaler_value * grad_out
            grad_delta = grad_out
            grad_scaler = RESIDUAL_SCALE * grad_out * mixed / norm
        else:
            surface_grad_factor = grad_out * delta_value * scaler_value
            grad_delta = grad_out * (1.0 + scaler_value * mixed / norm)
            grad_scaler = grad_out * delta_value * mixed / norm

        if SINGLE_HIDDEN_TILE:
            # One program owns the complete hidden reduction, so the local
            # dot is mathematically global.  Multi-tile rows receive the
            # token-owned reduction produced by _jtok_backward_norm_dot_kernel.
            dot = tl.sum(surface_grad_factor * mixed, axis=0)
        else:
            dot = tl.load(norm_dot_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        grad_surface = (
            surface_grad_factor / norm
            - mixed * dot / (norm * norm * norm)
        )

        valid_active = active_mask & row_valid
        grad_surface = tl.where(valid_active, grad_surface, 0.0)
        grad_scaler = tl.where(valid_active, grad_scaler, 0.0)
        # The projection-gradient pass reuses this FP32 workspace after the
        # token-local derivative scalars have been accumulated.
        tl.store(
            surface_ptr + row * HIDDEN + cols,
            grad_surface,
            mask=active_mask,
        )
        tl.store(
            grad_delta_ptr + row * HIDDEN + cols,
            grad_delta.to(grad_delta_ptr.dtype.element_ty),
            mask=active_mask,
        )
        tl.atomic_add(grad_scaler_ptr + safe_cols, grad_scaler)

        for slot in tl.range(0, TOP_K):
            expert = tl.load(
                expert_idx_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0,
            ).to(tl.int32)
            weight = tl.load(
                selected_weights_ptr + row * TOP_K + slot,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            slot_values = tl.zeros((BLOCK_H,), dtype=tl.float32)
            grad_value = weight * grad_surface

            for mode in tl.range(0, NUM_MODES):
                mode_value = tl.load(
                    modes_ptr + (row * TOP_K + slot) * NUM_MODES + mode,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                out_weight = tl.load(
                    spline_out_ptr
                    + (expert * NUM_MODES + mode) * HIDDEN
                    + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                slot_values += mode_value * out_weight
                grad_mode_value = tl.sum(grad_value * out_weight, axis=0)
                if SINGLE_HIDDEN_TILE:
                    tl.store(
                        grad_mode_ptr
                        + (row * TOP_K + slot) * NUM_MODES
                        + mode,
                        tl.where(row_valid, grad_mode_value, 0.0),
                        mask=row_mask,
                    )
                else:
                    tl.atomic_add(
                        grad_mode_ptr
                        + (row * TOP_K + slot) * NUM_MODES
                        + mode,
                        grad_mode_value,
                        mask=row_valid,
                    )

            residual_value = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for d in tl.range(0, D_SEED):
                z_value = tl.load(
                    z_ptr + row * D_SEED + d,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_weight = tl.load(
                    residual_out_ptr + (expert * D_SEED + d) * HIDDEN + cols,
                    mask=active_mask,
                    other=0.0,
                ).to(tl.float32)
                residual_value += z_value * residual_weight
                grad_residual_value = tl.sum(grad_value * residual_weight, axis=0)
                if SINGLE_HIDDEN_TILE:
                    tl.store(
                        grad_residual_ptr
                        + (row * TOP_K + slot) * D_SEED
                        + d,
                        tl.where(row_valid, grad_residual_value, 0.0),
                        mask=row_mask,
                    )
                else:
                    tl.atomic_add(
                        grad_residual_ptr
                        + (row * TOP_K + slot) * D_SEED
                        + d,
                        grad_residual_value,
                        mask=row_valid,
                    )
            slot_values += residual_value
            grad_weight_value = tl.sum(grad_surface * slot_values, axis=0)
            if SINGLE_HIDDEN_TILE:
                tl.store(
                    grad_weights_ptr + row * TOP_K + slot,
                    grad_weight_value,
                    mask=row_mask,
                )
            else:
                tl.atomic_add(
                    grad_weights_ptr + row * TOP_K + slot,
                    grad_weight_value,
                    mask=row_mask,
                )

    @triton.jit
    def _jtok_backward_token_projection_grad_kernel(
        z_ptr,
        spline_coeff_ptr,
        grid_ptr,
        expert_idx_ptr,
        modes_ptr,
        grad_mode_ptr,
        grad_residual_ptr,
        valid_ptr,
        grad_z_ptr,
        grad_coeff_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        KNOT_PAD: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """Finish token-local spline and seed gradients once per coordinate."""
        row = tl.program_id(0)
        slot = tl.program_id(1)
        d = tl.program_id(2)
        row_mask = row < N
        safe_row = tl.minimum(row, N - 1)
        if HAS_MASK:
            row_valid = tl.load(valid_ptr + safe_row, mask=row_mask, other=0).to(tl.int1)
        else:
            row_valid = row_mask

        expert = tl.load(
            expert_idx_ptr + safe_row * TOP_K + slot,
            mask=row_mask,
            other=0,
        ).to(tl.int32)
        x = tl.load(
            z_ptr + safe_row * D_SEED + d,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        knot_offsets = tl.arange(0, KNOT_PAD)
        grid = tl.load(
            grid_ptr + knot_offsets,
            mask=knot_offsets < NUM_KNOTS,
            other=0.0,
        ).to(tl.float32)
        grid_scale = float(max(int(NUM_KNOTS) - 1, 0))
        diff = x - grid
        distance = tl.abs(diff) * grid_scale
        basis_raw = tl.where(
            distance < 0.5,
            0.75 - distance * distance,
            tl.where(
                distance < 1.5,
                0.5 * (1.5 - distance) * (1.5 - distance),
                0.0,
            ),
        )
        basis_raw = tl.where(knot_offsets < NUM_KNOTS, basis_raw, 0.0)
        sign_x = tl.where(
            diff > 0.0,
            1.0,
            tl.where(diff < 0.0, -1.0, 0.0),
        )
        basis_derivative = tl.where(
            distance < 0.5,
            -2.0 * distance * grid_scale * sign_x,
            tl.where(
                distance < 1.5,
                -(1.5 - distance) * grid_scale * sign_x,
                0.0,
            ),
        )
        basis_derivative = tl.where(
            knot_offsets < NUM_KNOTS,
            basis_derivative,
            0.0,
        )
        raw_sum = tl.sum(basis_raw, axis=0)
        safe_sum = tl.maximum(raw_sum, 1e-12)
        grad_z_value = tl.load(
            grad_residual_ptr
            + (safe_row * TOP_K + slot) * D_SEED
            + d,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        for mode in tl.range(0, NUM_MODES):
            mode_value = tl.load(
                modes_ptr + (safe_row * TOP_K + slot) * NUM_MODES + mode,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            grad_mode = tl.load(
                grad_mode_ptr + (safe_row * TOP_K + slot) * NUM_MODES + mode,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            coeff = tl.load(
                spline_coeff_ptr
                + ((expert * NUM_MODES + mode) * D_SEED + d) * NUM_KNOTS
                + knot_offsets,
                mask=knot_offsets < NUM_KNOTS,
                other=0.0,
            ).to(tl.float32)
            weighted = tl.sum(basis_raw * coeff, axis=0)
            derivative_sum = tl.sum(basis_derivative, axis=0)
            derivative_weighted = tl.sum(
                basis_derivative * coeff,
                axis=0,
            )
            phi = weighted / safe_sum
            dphi_dz = tl.where(
                raw_sum > 1e-12,
                (
                    derivative_weighted * safe_sum
                    - weighted * derivative_sum
                )
                / (safe_sum * safe_sum),
                0.0,
            )
            phi_sign = tl.where(phi < 0.0, -1.0, 1.0)
            grad_phi = (
                grad_mode
                * mode_value
                * phi_sign
                / (tl.abs(phi) + 1e-9)
            )
            grad_phi = tl.where(row_valid, grad_phi, 0.0)
            grad_z_value += grad_phi * dphi_dz
            tl.atomic_add(
                grad_coeff_ptr
                + ((expert * NUM_MODES + mode) * D_SEED + d) * NUM_KNOTS
                + tl.minimum(knot_offsets, NUM_KNOTS - 1),
                tl.where(
                    knot_offsets < NUM_KNOTS,
                    grad_phi * basis_raw / safe_sum,
                    0.0,
                ),
            )

        tl.atomic_add(
            grad_z_ptr + safe_row * D_SEED + d,
            grad_z_value,
            mask=row_mask & row_valid,
        )

    @triton.autotune(
        configs=_JTOK_TOKEN_PROJECTION_CONFIGS,
        key=[
            "N",
            "D_SEED",
            "NUM_KNOTS",
            "NUM_MODES",
            "TOP_K",
            "BLOCK_D",
            "HAS_MASK",
        ],
        reset_to_zero=["grad_z_ptr", "grad_coeff_ptr"],
    )
    @triton.jit
    def _jtok_backward_token_projection_grad_block_kernel(
        z_ptr,
        spline_coeff_ptr,
        grid_ptr,
        expert_idx_ptr,
        modes_ptr,
        grad_mode_ptr,
        grad_residual_ptr,
        valid_ptr,
        grad_z_ptr,
        grad_coeff_ptr,
        N,
        D_SEED: tl.constexpr,
        NUM_KNOTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        KNOT_PAD: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """Vectorized token-local spline gradients for the compact route.

        The scalar kernel above launches one program for every
        ``(token, route, seed-coordinate)``.  At the measured ``hidden=256``
        boundary that creates many tiny programs even though the selected
        JTok-M geometry has only 32 seed coordinates.  The same issue appears
        in the full NeoLLM geometry (``d_seed=128, knots=16, modes=4``), where
        the scalar grid multiplies the program count by 128.  This
        specialization keeps the same equations but processes a power-of-two
        seed block per ``(token, route)`` program.  The Python dispatcher
        restricts it by geometry so larger spline surfaces retain the
        established scalar-grid path.
        """
        row = tl.program_id(0)
        slot = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        d_mask = d < D_SEED
        row_mask = row < N
        safe_row = tl.minimum(row, N - 1)
        if HAS_MASK:
            row_valid = tl.load(
                valid_ptr + safe_row,
                mask=row_mask,
                other=0,
            ).to(tl.int1)
        else:
            row_valid = row_mask

        expert = tl.load(
            expert_idx_ptr + safe_row * TOP_K + slot,
            mask=row_mask,
            other=0,
        ).to(tl.int32)
        x = tl.load(
            z_ptr + safe_row * D_SEED + d,
            mask=row_mask & d_mask,
            other=0.0,
        ).to(tl.float32)
        knot_offsets = tl.arange(0, KNOT_PAD)
        knot_mask = knot_offsets < NUM_KNOTS
        grid = tl.load(
            grid_ptr + knot_offsets,
            mask=knot_mask,
            other=0.0,
        ).to(tl.float32)
        grid_scale = float(max(int(NUM_KNOTS) - 1, 0))
        diff = x[:, None] - grid[None, :]
        distance = tl.abs(diff) * grid_scale
        basis_raw = tl.where(
            distance < 0.5,
            0.75 - distance * distance,
            tl.where(
                distance < 1.5,
                0.5 * (1.5 - distance) * (1.5 - distance),
                0.0,
            ),
        )
        basis_raw = tl.where(knot_mask[None, :], basis_raw, 0.0)
        sign_x = tl.where(
            diff > 0.0,
            1.0,
            tl.where(diff < 0.0, -1.0, 0.0),
        )
        basis_derivative = tl.where(
            distance < 0.5,
            -2.0 * distance * grid_scale * sign_x,
            tl.where(
                distance < 1.5,
                -(1.5 - distance) * grid_scale * sign_x,
                0.0,
            ),
        )
        basis_derivative = tl.where(
            knot_mask[None, :], basis_derivative, 0.0
        )
        raw_sum = tl.sum(basis_raw, axis=1)
        safe_sum = tl.maximum(raw_sum, 1e-12)
        grad_z_value = tl.load(
            grad_residual_ptr
            + (safe_row * TOP_K + slot) * D_SEED
            + d,
            mask=row_mask & d_mask,
            other=0.0,
        ).to(tl.float32)

        for mode in tl.range(0, NUM_MODES):
            mode_value = tl.load(
                modes_ptr + (safe_row * TOP_K + slot) * NUM_MODES + mode,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            grad_mode = tl.load(
                grad_mode_ptr + (safe_row * TOP_K + slot) * NUM_MODES + mode,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            coeff = tl.load(
                spline_coeff_ptr
                + ((expert * NUM_MODES + mode) * D_SEED) * NUM_KNOTS
                + d[:, None] * NUM_KNOTS
                + knot_offsets[None, :],
                mask=(row_mask & d_mask)[:, None] & knot_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weighted = tl.sum(basis_raw * coeff, axis=1)
            derivative_sum = tl.sum(basis_derivative, axis=1)
            derivative_weighted = tl.sum(
                basis_derivative * coeff,
                axis=1,
            )
            phi = weighted / safe_sum
            dphi_dz = tl.where(
                raw_sum > 1e-12,
                (
                    derivative_weighted * safe_sum
                    - weighted * derivative_sum
                )
                / (safe_sum * safe_sum),
                0.0,
            )
            phi_sign = tl.where(phi < 0.0, -1.0, 1.0)
            grad_phi = (
                grad_mode
                * mode_value
                * phi_sign
                / (tl.abs(phi) + 1e-9)
            )
            grad_phi = tl.where(row_valid, grad_phi, 0.0)
            grad_z_value += grad_phi * dphi_dz
            tl.atomic_add(
                grad_coeff_ptr
                + ((expert * NUM_MODES + mode) * D_SEED) * NUM_KNOTS
                + d[:, None] * NUM_KNOTS
                + tl.minimum(knot_offsets, NUM_KNOTS - 1)[None, :],
                tl.where(
                    knot_mask[None, :],
                    grad_phi[:, None] * basis_raw / safe_sum[:, None],
                    0.0,
                ),
                mask=(row_mask & d_mask)[:, None] & knot_mask[None, :],
            )

        tl.atomic_add(
            grad_z_ptr + safe_row * D_SEED + d,
            grad_z_value,
            mask=row_mask & row_valid & d_mask,
        )

    @triton.autotune(
        configs=_JTOK_PROJECTION_GRAD_CONFIGS,
        key=[
            "N",
            "D_SEED",
            "HIDDEN",
            "NUM_EXPERTS",
            "NUM_MODES",
            "TOP_K",
            "BLOCK_H",
        ],
        reset_to_zero=["grad_spline_out_ptr", "grad_residual_out_ptr"],
    )
    @triton.jit
    def _jtok_backward_projection_grad_kernel(
        grad_surface_ptr,
        z_ptr,
        modes_ptr,
        expert_idx_ptr,
        selected_weights_ptr,
        grad_spline_out_ptr,
        grad_residual_out_ptr,
        N,
        D_SEED: tl.constexpr,
        HIDDEN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        NUM_MODES: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """Reduce projection gradients by token block instead of by token.

        The wide backward already has the complete ``grad_surface`` in the
        temporary surface buffer.  The previous implementation launched one
        token/tile program and atomically added every element of
        ``grad_spline_out`` and ``grad_residual_out``.  This kernel scans one
        token block for one expert and performs one vector atomic per output
        parameter tile.  It deliberately uses a block-local expert match,
        rather than sorting or compacting routes with Torch, so it remains
        graph-safe for dynamic top-k routing and does not introduce a second
        data-dependent execution path.

        The scan is over the small expert dimension and the temporary is
        bounded by ``BLOCK_M * BLOCK_H`` registers.  No
        ``[tokens, experts, modes, hidden]`` tensor is formed.
        """
        expert = tl.program_id(0)
        row_block = tl.program_id(1)
        hidden_block = tl.program_id(2)
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
        row_mask = rows < N
        col_mask = cols < HIDDEN
        matrix_mask = row_mask[:, None] & col_mask[None, :]
        grad_surface = tl.load(
            grad_surface_ptr
            + rows[:, None] * HIDDEN
            + cols[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)

        # Route metadata is invariant across the mode and seed-coordinate
        # reductions below.  Loading it inside both loops made the wide
        # backward reread ``expert_idx`` and ``selected_weights`` up to
        # ``NUM_MODES + D_SEED`` times per token block.  Keep the compact
        # [BLOCK_M, TOP_K] route tile in registers and reuse it for both
        # projection-gradient families.  This is only a register tile; it
        # does not materialize an expert-expanded activation.
        route_slots = tl.arange(0, TOP_K)
        route_mask = row_mask[:, None]
        route_experts = tl.load(
            expert_idx_ptr
            + rows[:, None] * TOP_K
            + route_slots[None, :],
            mask=route_mask,
            other=0,
        ).to(tl.int32)
        route_weights = tl.load(
            selected_weights_ptr
            + rows[:, None] * TOP_K
            + route_slots[None, :],
            mask=route_mask,
            other=0.0,
        ).to(tl.float32)
        expert_weight = tl.sum(
            tl.where(route_experts == expert, route_weights, 0.0), axis=1
        )

        # The same grad_surface tile is reused for all modes.  This keeps the
        # hidden projection read traffic close to one pass per expert/block.
        for mode in tl.range(0, NUM_MODES):
            mode_value = tl.load(
                modes_ptr
                + (rows[:, None] * TOP_K + route_slots[None, :]) * NUM_MODES
                + mode,
                mask=route_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient = tl.sum(
                tl.where(
                    route_experts == expert,
                    route_weights * mode_value,
                    0.0,
                ),
                axis=1,
            )
            partial = tl.sum(
                grad_surface * coefficient[:, None],
                axis=0,
            )
            tl.atomic_add(
                grad_spline_out_ptr
                + (expert * NUM_MODES + mode) * HIDDEN
                + cols,
                partial,
                mask=col_mask,
            )

        for d in tl.range(0, D_SEED):
            z_value = tl.load(
                z_ptr + rows * D_SEED + d,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient = expert_weight * z_value
            partial = tl.sum(
                grad_surface * coefficient[:, None],
                axis=0,
            )
            tl.atomic_add(
                grad_residual_out_ptr
                + (expert * D_SEED + d) * HIDDEN
                + cols,
                partial,
                mask=col_mask,
            )



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
) -> tuple[torch.Tensor, torch.Tensor]:
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

    knot_pad = triton.next_power_of_2(knots)
    # Keep the compact mode product as an autograd output.  The backward
    # consumes it directly instead of launching the B-spline evaluator a
    # second time.  Its size is tokens * top_k * modes, not a surface-sized
    # activation; the fused single-tile path writes the same contract.
    modes_buffer = torch.empty(
        (n_tokens, top_k, modes), device=delta.device, dtype=spline_out.dtype
    )
    single_tile = hidden < _SINGLE_TILE_HIDDEN_LIMIT
    if single_tile:
        output = torch.empty_like(delta)
        block_n = triton.next_power_of_2(hidden)
        _jtok_wrap_kernel(_jtok_project_finalize_fused_kernel)[(n_tokens,)](
            delta,
            z,
            spline_coeff,
            knot_grid,
            spline_out,
            residual_out,
            expert_idx,
            selected_weights,
            modes_buffer,
            scaler,
            valid_mask,
            output,
            n_tokens,
            D_SEED=d_seed,
            NUM_KNOTS=knots,
            NUM_MODES=modes,
            TOP_K=top_k,
            HIDDEN=hidden,
            BLOCK_N=block_n,
            KNOT_PAD=knot_pad,
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
                output=output,
            )
        return output, modes_buffer

    # Keep the wide-path surface in FP32.  The forward kernel reduces its
    # squared norm before the activation-dtype write-back; retaining the
    # pre-cast surface here lets the tiled backward use the same value for the
    # normalization derivative without rebuilding a Torch reference graph.
    # A compact [tokens, top_k, modes] mode buffer is much smaller than the
    # former dense expert surface and avoids recalculating the B-spline
    # product for every hidden tile.  It is intentionally scoped to this
    # invocation rather than retained by the autograd context.
    if _can_use_route_vectorized_mode_evaluation(d_seed, knots, modes, top_k):
        mode_grid = (n_tokens,)
        _jtok_wrap_kernel(_jtok_modes_kernel_route_vectorized)[mode_grid](
            z,
            spline_coeff,
            knot_grid,
            expert_idx,
            valid_mask,
            modes_buffer,
            n_tokens,
            D_SEED=d_seed,
            NUM_KNOTS=knots,
            NUM_MODES=modes,
            TOP_K=top_k,
            KNOT_PAD=knot_pad,
            MODE_PAD=triton.next_power_of_2(modes),
            HAS_MASK=has_mask,
            num_warps=4,
            num_stages=1,
        )
    elif _can_use_vectorized_mode_evaluation(d_seed, knots, modes):
        mode_grid = (n_tokens * top_k,)
        _jtok_wrap_kernel(_jtok_modes_kernel_vectorized)[mode_grid](
            z,
            spline_coeff,
            knot_grid,
            expert_idx,
            valid_mask,
            modes_buffer,
            n_tokens,
            D_SEED=d_seed,
            NUM_KNOTS=knots,
            NUM_MODES=modes,
            TOP_K=top_k,
            KNOT_PAD=knot_pad,
            MODE_PAD=triton.next_power_of_2(modes),
            HAS_MASK=has_mask,
            num_warps=4,
            num_stages=1,
        )
    else:
        modes_grid = (n_tokens * top_k * modes,)
        if has_mask:
            _jtok_wrap_kernel(_jtok_modes_kernel_masked)[modes_grid](
                z,
                spline_coeff,
                knot_grid,
                expert_idx,
                valid_mask,
                modes_buffer,
                n_tokens,
                D_SEED=d_seed,
                NUM_KNOTS=knots,
                NUM_MODES=modes,
                TOP_K=top_k,
                KNOT_PAD=knot_pad,
                num_warps=4,
                num_stages=1,
            )
        else:
            _jtok_wrap_kernel(_jtok_modes_kernel_unmasked)[modes_grid](
                z,
                spline_coeff,
                knot_grid,
                expert_idx,
                modes_buffer,
                n_tokens,
                D_SEED=d_seed,
                NUM_KNOTS=knots,
                NUM_MODES=modes,
                TOP_K=top_k,
                KNOT_PAD=knot_pad,
                num_warps=4,
                num_stages=1,
            )
    surface = torch.empty(
        (n_tokens, hidden), device=delta.device, dtype=torch.float32
    )
    norm = torch.zeros(n_tokens, device=delta.device, dtype=torch.float32)
    output = torch.empty_like(delta)
    # The wide path keeps only the compact [tokens, top_k, modes] mode
    # activation.  It is negligible relative to a dense expert surface and
    # prevents the B-spline product from being recomputed for every hidden
    # tile.  The single-tile path above still folds final
    # normalization/modulation into one Triton boundary.
    block_n = min(256, triton.next_power_of_2(hidden))
    project_grid = (n_tokens, triton.cdiv(hidden, block_n))
    _jtok_wrap_kernel(_jtok_project_kernel)[project_grid](
        z,
        modes_buffer,
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
    _jtok_wrap_kernel(_jtok_finalize_kernel)[final_grid](
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
            surface=surface,
            norm=norm,
            output=output,
        )
    mark_warmup_incomplete_once(
        "jtok.mixture" if mixture else "jtok.forward",
        (
            delta.device.index,
            delta.dtype,
            int(n_tokens),
            int(hidden),
            int(d_seed),
            int(knots),
            int(modes),
            int(top_k),
            bool(has_mask),
        ),
    )
    return output, modes_buffer


def _run_jtok_backward_triton(
    grad_out: torch.Tensor,
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
    modes_buffer: torch.Tensor,
    *,
    norm_eps: float,
    residual_scale: float,
    mixture: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the Triton backward for both compact and wide hidden rows.

    Parameter gradients accumulate into small FP32 workspaces and are cast
    only after the kernel completes.  The workspaces are proportional to the
    trainable surface parameters, not to ``tokens × experts × modes × hidden``.
    Hidden sizes below 256 use the compact single-tile program.  Wider rows
    use the geometry planner: a complete-row program is selected when the
    hidden width and combined seed/mode work fit the resource budget;
    otherwise the established 256-lane two-pass tiled reduction is used.  The
    tiled path's first pass writes the selected FP32 surface and row norm, and
    its second pass accumulates parameter gradients per hidden tile.  There
    is deliberately no Torch/autograd fallback in this registered
    external-kernel path.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if delta.shape != grad_out.shape:
        raise ValueError("grad_out and delta must have the same shape")
    experts, modes, d_seed, knots, hidden = _check_common_kernel_inputs(
        delta,
        z,
        spline_coeff,
        spline_out,
        residual_out,
        scaler,
        knot_grid,
    )
    top_k = int(expert_idx.shape[1])
    if top_k < 1 or top_k > experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if selected_weights.shape != expert_idx.shape:
        raise ValueError("selected_weights and expert_idx must have the same shape")
    if valid_mask.numel() not in (0, int(delta.shape[0])):
        raise ValueError("valid_mask has the wrong number of elements")
    n_tokens = int(delta.shape[0])
    expected_modes_shape = (n_tokens, top_k, modes)
    if tuple(modes_buffer.shape) != expected_modes_shape:
        raise ValueError(
            "modes_buffer must preserve the forward compact mode product with "
            f"shape {expected_modes_shape}, got {tuple(modes_buffer.shape)}"
        )
    if modes_buffer.dtype != spline_out.dtype or modes_buffer.device != delta.device:
        raise ValueError("modes_buffer must match spline_out dtype and delta device")
    grad_delta = torch.empty_like(delta)
    # The kernel accumulates all shared gradients in FP32.  This also avoids
    # dtype-dependent atomic behavior for BF16 parameter gradients.
    grad_z_accum = torch.zeros(
        (n_tokens, d_seed), device=delta.device, dtype=torch.float32
    )
    grad_coeff_accum = torch.zeros(
        spline_coeff.shape, device=delta.device, dtype=torch.float32
    )
    grad_spline_out_accum = torch.zeros(
        spline_out.shape, device=delta.device, dtype=torch.float32
    )
    grad_residual_out_accum = torch.zeros(
        residual_out.shape, device=delta.device, dtype=torch.float32
    )
    grad_scaler_accum = torch.zeros(
        scaler.shape, device=delta.device, dtype=torch.float32
    )
    grad_weights = torch.empty_like(selected_weights)
    if n_tokens == 0:
        return (
            grad_delta,
            grad_z_accum.to(z.dtype),
            grad_coeff_accum.to(spline_coeff.dtype),
            grad_spline_out_accum.to(spline_out.dtype),
            grad_residual_out_accum.to(residual_out.dtype),
            grad_scaler_accum.to(scaler.dtype),
            grad_weights,
        )

    if hidden < _SINGLE_TILE_HIDDEN_LIMIT:
        block_h = triton.next_power_of_2(hidden)
        _jtok_wrap_kernel(_jtok_backward_single_tile_kernel)[(n_tokens,)](
            delta,
            z,
            spline_coeff,
            knot_grid,
            spline_out,
            residual_out,
            expert_idx,
            selected_weights,
            scaler,
            grad_out,
            valid_mask,
            grad_delta,
            grad_z_accum,
            grad_coeff_accum,
            grad_spline_out_accum,
            grad_residual_out_accum,
            grad_scaler_accum,
            grad_weights,
            n_tokens,
            D_SEED=d_seed,
            NUM_KNOTS=knots,
            NUM_MODES=modes,
            TOP_K=top_k,
            HIDDEN=hidden,
            BLOCK_H=block_h,
            KNOT_PAD=triton.next_power_of_2(knots),
            NORM_EPS=float(norm_eps),
            RESIDUAL_SCALE=float(residual_scale),
            MIXTURE=bool(mixture),
            HAS_MASK=bool(valid_mask.numel()),
            PARAM_GRADS=True,
            num_warps=4,
            num_stages=1,
        )
    else:
        # Every hidden tile contributes to the same token-level gradient
        # tensors.  At the one-tile boundary the fast kernel writes each
        # token exactly once, so it can skip the memset and use stores for
        # those token-local workspaces; wider rows retain FP32 atomics.
        tile_plan = _backward_hidden_tile_plan(
            hidden,
            d_seed,
            modes,
            top_k,
        )
        block_h = tile_plan.block_h
        single_hidden_tile = tile_plan.direct_store
        if not single_hidden_tile:
            grad_weights.zero_()
        # Wide hidden rows accumulate mode and residual contractions once per
        # hidden tile, then finish the B-spline derivative once per token.
        # These FP32 buffers are compact: their size is independent of hidden.
        if single_hidden_tile:
            grad_mode_buffer = torch.empty(
                (n_tokens, top_k, modes), device=delta.device, dtype=torch.float32
            )
            grad_residual_buffer = torch.empty(
                (n_tokens, top_k, d_seed), device=delta.device, dtype=torch.float32
            )
        else:
            grad_mode_buffer = torch.zeros(
                (n_tokens, top_k, modes), device=delta.device, dtype=torch.float32
            )
            grad_residual_buffer = torch.zeros(
                (n_tokens, top_k, d_seed), device=delta.device, dtype=torch.float32
            )
        surface = torch.empty(
            (n_tokens, hidden), device=delta.device, dtype=torch.float32
        )
        norm = torch.zeros(n_tokens, device=delta.device, dtype=torch.float32)
        project_grid = (n_tokens, triton.cdiv(hidden, block_h))
        _jtok_wrap_kernel(_jtok_project_kernel)[project_grid](
            z,
            modes_buffer,
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
            BLOCK_N=block_h,
            HAS_MASK=bool(valid_mask.numel()),
            num_warps=4,
            num_stages=1,
        )
        # The normalization derivative couples every hidden tile through one
        # scalar dot product.  Compute it once per token before the tiled
        # gradient pass; a complete-row tile computes the same scalar locally
        # and does not need this launch or workspace initialization.
        # The single-tile specialization never reads ``norm_dot`` because it
        # computes the complete reduction locally.  Reuse ``norm`` there so
        # the optimization does not introduce a dead FP32 allocation; only
        # genuinely multi-tile rows receive a second per-token scalar buffer.
        norm_dot = norm if single_hidden_tile else torch.empty_like(norm)
        if not single_hidden_tile:
            _jtok_wrap_kernel(_jtok_backward_norm_dot_kernel)[(n_tokens,)](
                surface,
                delta,
                scaler,
                grad_out,
                valid_mask,
                norm_dot,
                n_tokens,
                HIDDEN=hidden,
                BLOCK_H=block_h,
                NUM_TILES=tile_plan.num_tiles,
                RESIDUAL_SCALE=float(residual_scale),
                MIXTURE=bool(mixture),
                HAS_MASK=bool(valid_mask.numel()),
                num_warps=4,
                num_stages=1,
            )
        _jtok_wrap_kernel(_jtok_backward_multi_tile_fast_kernel)[project_grid](
            delta,
            z,
            spline_out,
            residual_out,
            expert_idx,
            selected_weights,
            modes_buffer,
            grad_mode_buffer,
            grad_residual_buffer,
            scaler,
            surface,
            norm,
            norm_dot,
            grad_out,
            valid_mask,
            grad_delta,
            grad_scaler_accum,
            grad_weights,
            n_tokens,
            D_SEED=d_seed,
            NUM_MODES=modes,
            TOP_K=top_k,
            HIDDEN=hidden,
            BLOCK_H=block_h,
            NORM_EPS=float(norm_eps),
            RESIDUAL_SCALE=float(residual_scale),
            MIXTURE=bool(mixture),
            HAS_MASK=bool(valid_mask.numel()),
            SINGLE_HIDDEN_TILE=bool(single_hidden_tile),
        )
        # Vectorizing the seed-coordinate reduction removes one tiny program
        # per coordinate.  It is safe for wide hidden rows too: the preceding
        # multi-tile kernel has already reduced their token-local mode and
        # residual buffers.  The work-budget predicate keeps larger spline
        # geometries on the established scalar-grid path.
        if _can_use_vectorized_token_projection(d_seed, knots, modes):
            block_d = triton.next_power_of_2(d_seed)
            token_projection_grid = (n_tokens, top_k)
            _jtok_wrap_kernel(_jtok_backward_token_projection_grad_block_kernel)[
                token_projection_grid
            ](
                z,
                spline_coeff,
                knot_grid,
                expert_idx,
                modes_buffer,
                grad_mode_buffer,
                grad_residual_buffer,
                valid_mask,
                grad_z_accum,
                grad_coeff_accum,
                n_tokens,
                D_SEED=d_seed,
                NUM_KNOTS=knots,
                NUM_MODES=modes,
                TOP_K=top_k,
                KNOT_PAD=triton.next_power_of_2(knots),
                BLOCK_D=block_d,
                HAS_MASK=bool(valid_mask.numel()),
            )
        else:
            token_projection_grid = (n_tokens, top_k, d_seed)
            _jtok_wrap_kernel(_jtok_backward_token_projection_grad_kernel)[
                token_projection_grid
            ](
                z,
                spline_coeff,
                knot_grid,
                expert_idx,
                modes_buffer,
                grad_mode_buffer,
                grad_residual_buffer,
                valid_mask,
                grad_z_accum,
                grad_coeff_accum,
                n_tokens,
                D_SEED=d_seed,
                NUM_KNOTS=knots,
                NUM_MODES=modes,
                TOP_K=top_k,
                KNOT_PAD=triton.next_power_of_2(knots),
                HAS_MASK=bool(valid_mask.numel()),
                num_warps=4,
                num_stages=1,
            )
        # Keep projection tiling independent from the hidden-reduction
        # ownership mode.  The old coupling selected BLOCK_M=64 whenever the
        # preceding pass used a complete hidden tile; on the full training
        # geometry that made the projection-gradient kernel substantially
        # slower even though the single-tile reduction itself was valid.
        # Triton's autotuner now benchmarks BLOCK_M and launch resources for
        # this exact token/hidden geometry.
        projection_block_h = 128
        def projection_grid(meta):
            return (
                experts,
                triton.cdiv(n_tokens, meta["BLOCK_M"]),
                triton.cdiv(hidden, meta["BLOCK_H"]),
            )

        _jtok_wrap_kernel(_jtok_backward_projection_grad_kernel)[projection_grid](
            surface,
            z,
            modes_buffer,
            expert_idx,
            selected_weights,
            grad_spline_out_accum,
            grad_residual_out_accum,
            n_tokens,
            D_SEED=d_seed,
            HIDDEN=hidden,
            NUM_EXPERTS=experts,
            NUM_MODES=modes,
            TOP_K=top_k,
            BLOCK_H=projection_block_h,
        )
    mark_warmup_incomplete_once(
        "jtok.mixture.backward" if mixture else "jtok.backward",
        (
            grad_out.device.index,
            grad_out.dtype,
            int(n_tokens),
            int(hidden),
            int(d_seed),
            int(knots),
            int(modes),
            int(top_k),
            bool(valid_mask.numel()),
        ),
    )
    return (
        grad_delta,
        grad_z_accum.to(z.dtype),
        grad_coeff_accum.to(spline_coeff.dtype),
        grad_spline_out_accum.to(spline_out.dtype),
        grad_residual_out_accum.to(residual_out.dtype),
        grad_scaler_accum.to(scaler.dtype),
        grad_weights,
    )


def _run_jtok_backward_formula(
    grad_out: torch.Tensor,
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
    *,
    norm_eps: float,
    residual_scale: float,
    mixture: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Vectorized backward formula used to audit the Triton reduction.

    This is intentionally written from the derivative equations rather than
    through ``torch.autograd.grad``.  It is an experiment for Torch 2.14's
    AOTAutograd path: Inductor can see the contractions and choose GEMM-like
    reductions, while the external forward still owns the selected surface.
    The temporary dispatch switch is removed after the controlled comparison;
    the formula itself remains useful as the wide-geometry reference.
    """
    n_tokens, hidden = delta.shape
    experts, modes, d_seed, knots = map(int, spline_coeff.shape)
    dtype = delta.dtype
    valid = (
        valid_mask.reshape(n_tokens).to(device=delta.device, dtype=torch.bool)
        if valid_mask.numel()
        else torch.ones(n_tokens, device=delta.device, dtype=torch.bool)
    )
    valid_f = valid.to(torch.float32).unsqueeze(-1)
    grad_out_f = grad_out.float()
    delta_f = delta.float()
    z_f = z.float()
    coeff_f = spline_coeff.float()
    out_f = spline_out.float()
    residual_f = residual_out.float()
    scaler_f = scaler.float()
    weights_f = selected_weights.float()

    # ``one_hot`` is only [tokens, top_k, experts].  It avoids gathering a
    # [tokens, top_k, modes, hidden] projection and lets Inductor lower the
    # shared-parameter reductions as contractions.
    one_hot = F.one_hot(expert_idx, num_classes=experts).to(torch.float32)
    basis_scale = float(max(knots - 1, 0))
    diff = z_f.unsqueeze(-1) - knot_grid.float().view(1, 1, knots)
    distance = diff.abs() * basis_scale
    raw_basis = torch.where(
        distance < 0.5,
        0.75 - distance.square(),
        torch.where(
            distance < 1.5,
            0.5 * (1.5 - distance).square(),
            torch.zeros_like(distance),
        ),
    )
    raw_sum = raw_basis.sum(dim=-1, keepdim=True)
    safe_sum = raw_sum.clamp_min(_BASIS_EPS)
    basis = raw_basis / safe_sum
    phi = torch.einsum("ndg,nkmdg->nkmd", basis, coeff_f[expert_idx])
    log_mag = torch.log(phi.abs() + _PRODUCT_LOG_EPS).sum(dim=-1)
    negative = (phi < 0).to(torch.int32).sum(dim=-1)
    mode_sign = 1.0 - 2.0 * negative.remainder(2).float()
    modes_value = mode_sign * torch.exp(log_mag)
    # The forward kernel explicitly rounds the mode scalar to activation dtype
    # before the projection.  Preserve that boundary in the formula path.
    modes_active = modes_value.to(spline_out.dtype).float()

    # selected_values is [N,K,H], not [N,E,H] or [N,K,M,H].
    selected_values = torch.einsum(
        "nke,nkm,emh->nkh", one_hot, modes_active, out_f
    )
    selected_values = selected_values + torch.einsum(
        "nke,nd,edh->nkh", one_hot, z_f, residual_f
    )
    mixed = (weights_f.unsqueeze(-1) * selected_values).sum(dim=1)
    mixed_f = mixed.float()
    surface = mixed.to(dtype).float()
    norm = torch.sqrt((mixed_f * mixed_f).sum(dim=-1, keepdim=True)) + float(norm_eps)
    direction = surface / norm

    if mixture:
        grad_surface_factor = (
            float(residual_scale) * scaler_f.unsqueeze(0) * grad_out_f
        )
        grad_delta_f = grad_out_f
        grad_scaler = (
            float(residual_scale) * grad_out_f * surface / norm * valid_f
        ).sum(dim=0)
    else:
        grad_surface_factor = grad_out_f * delta_f * scaler_f.unsqueeze(0)
        grad_delta_f = grad_out_f * (
            1.0 + scaler_f.unsqueeze(0) * surface / norm
        )
        grad_scaler = (
            grad_out_f * delta_f * surface / norm * valid_f
        ).sum(dim=0)
    grad_surface_factor = grad_surface_factor * valid_f
    dot = (grad_surface_factor * mixed_f).sum(dim=-1, keepdim=True)
    grad_surface = (
        grad_surface_factor / norm
        - mixed_f * dot / (norm * norm * norm)
    ) * valid_f
    grad_value = grad_surface.unsqueeze(1) * weights_f.unsqueeze(-1)

    # Shared projection gradients and the gradient of the selected mixture
    # weights.  The one-hot contraction keeps accumulation deterministic at
    # the mathematical level and removes per-token global atomics.
    grad_spline_out = torch.einsum(
        "nke,nkh,nkm->emh", one_hot, grad_value, modes_active
    )
    grad_residual_out = torch.einsum(
        "nke,nkh,nd->edh", one_hot, grad_value, z_f
    )
    grad_weights = (grad_surface * selected_values).sum(dim=-1)
    grad_mode = torch.einsum(
        "nkh,emh,nke->nkm", grad_value, out_f, one_hot
    )
    grad_z_residual = torch.einsum(
        "nkh,edh,nke->nd", grad_value, residual_f, one_hot
    )
    grad_phi = (
        grad_mode
        * modes_value
        * torch.where(phi < 0.0, -1.0, 1.0)
        / (phi.abs() + _PRODUCT_LOG_EPS)
    )
    grad_coeff = torch.einsum(
        "nkmd,nke,ndg->emdg", grad_phi, one_hot, basis
    )
    grad_basis = torch.einsum(
        "nkmd,nke,emdg->ndg", grad_phi, one_hot, coeff_f
    )
    grad_raw_basis = (
        grad_basis - basis * grad_basis.sum(dim=-1, keepdim=True)
    ) / safe_sum
    sign_diff = torch.where(diff > 0.0, 1.0, torch.where(diff < 0.0, -1.0, 0.0))
    basis_derivative = torch.where(
        distance < 0.5,
        -2.0 * distance * basis_scale * sign_diff,
        torch.where(
            distance < 1.5,
            -(1.5 - distance) * basis_scale * sign_diff,
            torch.zeros_like(distance),
        ),
    )
    grad_z = grad_z_residual + (grad_raw_basis * basis_derivative).sum(dim=-1)

    if mixture:
        grad_delta = grad_delta_f
    else:
        grad_delta = torch.where(valid_f.bool(), grad_delta_f, grad_out_f)
    return (
        grad_delta.to(dtype),
        grad_z.to(z.dtype),
        grad_coeff.to(spline_coeff.dtype),
        grad_spline_out.to(spline_out.dtype),
        grad_residual_out.to(residual_out.dtype),
        grad_scaler.to(scaler.dtype),
        grad_weights.to(selected_weights.dtype),
    )


def _kernel_region(name: str, device: torch.device):
    # ``triton_op`` traces this body under ``torch.compile``. The optional
    # 2.14 allocator/profiler context is eager-runtime scaffolding and must not
    # become Python control flow or a nested allocator context in the traced
    # graph. The custom-op compatibility path still receives it at runtime.
    is_compiling = getattr(
        getattr(torch, "compiler", None), "is_compiling", lambda: False
    )
    if TORCH_2_14_CUDA_KERNEL_CONTEXT and not is_compiling():
        return cuda_kernel_region(name, device)
    from contextlib import nullcontext

    return nullcontext()


@_jtok_op_decorator("cut_cross_entropy::jtok_backward")
def _jtok_backward_op(
    grad_out: torch.Tensor,
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
    modes_buffer: torch.Tensor,
    norm_eps: float,
    residual_scale: float,
    mixture: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _run_jtok_backward_triton(
        grad_out,
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
        modes_buffer,
        norm_eps=norm_eps,
        residual_scale=residual_scale,
        mixture=mixture,
    )


@_jtok_fake_registration(_jtok_backward_op)
def _jtok_backward_fake(
    grad_out: torch.Tensor,
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
    modes_buffer: torch.Tensor,
    norm_eps: float,
    residual_scale: float,
    mixture: bool,
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
        grad_out,
        expert_idx,
        knot_grid,
        valid_mask,
        modes_buffer,
        norm_eps,
        residual_scale,
        mixture,
    )
    return (
        torch.empty_like(delta),
        torch.empty_like(z),
        torch.empty_like(spline_coeff),
        torch.empty_like(spline_out),
        torch.empty_like(residual_out),
        torch.empty_like(scaler),
        torch.empty_like(selected_weights),
    )


@_jtok_op_decorator("cut_cross_entropy::jtok_forward")
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
) -> tuple[torch.Tensor, torch.Tensor]:
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


@_jtok_fake_registration(_jtok_forward_op)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    modes = torch.empty(
        (delta.shape[0], expert_idx.shape[1], spline_coeff.shape[1]),
        device=delta.device,
        dtype=spline_out.dtype,
    )
    del z, residual_out, scaler, knot_grid
    del expert_idx, selected_weights, valid_mask, norm_eps
    return torch.empty_like(delta), modes


def _jtok_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, modes_buffer = output
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
        modes_buffer,
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


def _jtok_backward(
    ctx: Any,
    grad_out: torch.Tensor,
    grad_modes: Optional[torch.Tensor] = None,
):
    del grad_modes
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
        modes_buffer,
    ) = ctx.saved_tensors
    if not (_TRITON_AVAILABLE and delta.is_cuda):
        raise RuntimeError(
            "The registered JTok kernel backward requires CUDA and Triton; "
            "the external-kernel path never falls back to the Torch reference."
        )
    grads = _jtok_backward_op(
        grad_out,
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
        modes_buffer,
        float(ctx.norm_eps),
        0.0,
        False,
    )[:6]
    return (*grads, None, None, None, None, None)


_jtok_register_autograd(
    _jtok_forward_op,
    _jtok_backward,
    setup_context=_jtok_setup_context,
)


@_jtok_op_decorator("cut_cross_entropy::jtokm_forward")
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
) -> tuple[torch.Tensor, torch.Tensor]:
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


@_jtok_fake_registration(_jtokm_forward_op)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    modes = torch.empty(
        (delta.shape[0], expert_idx.shape[1], spline_coeff.shape[1]),
        device=delta.device,
        dtype=spline_out.dtype,
    )
    del z, residual_out, scaler
    del expert_idx, selected_weights, knot_grid, valid_mask, norm_eps, residual_scale
    return torch.empty_like(delta), modes


def _jtokm_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, modes_buffer = output
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
        modes_buffer,
    )
    ctx.norm_eps = float(norm_eps)
    ctx.residual_scale = float(residual_scale)


def _jtokm_backward(
    ctx: Any,
    grad_out: torch.Tensor,
    grad_modes: Optional[torch.Tensor] = None,
):
    del grad_modes
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
        modes_buffer,
    ) = ctx.saved_tensors

    if not (_TRITON_AVAILABLE and delta.is_cuda):
        raise RuntimeError(
            "The registered JTok-M kernel backward requires CUDA and Triton; "
            "the external-kernel path never falls back to the Torch reference."
        )
    grads = _jtok_backward_op(
        grad_out,
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
        modes_buffer,
        float(ctx.norm_eps),
        float(ctx.residual_scale),
        True,
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


_jtok_register_autograd(
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
        output, _ = _jtok_forward_op(
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
    h = _flatten_aux_input(
        router_state,
        tuple(delta_m.shape[:-1]),
        delta_m.shape[-1],
        "router_state",
    )
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
        output, _ = _jtokm_forward_op(
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
