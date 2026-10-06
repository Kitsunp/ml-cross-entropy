"""Shared compact-support contract for Leviathan's quadratic spline.

The production Triton kernels keep their own device-side implementation, but
the support geometry is defined here once so forward, backward, and the
future JToK sidecar can be checked against the same reference.  The compact
form is valid only for the canonical unit grid, including its model-dtype
rounding: three mathematical neighbours plus a fourth slot for boundary
contributions possible on the stored floating-point knot grid.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import torch


# Keep a bounded reference to each tiny grid: detached custom-op inputs alias
# its storage/version counter. Retaining that storage also prevents pointer
# reuse from producing a false cache hit after the caller releases a grid.
_GRID_CHECKS: OrderedDict[tuple, tuple[torch.Tensor, bool]] = OrderedDict()


def is_uniform_unit_grid(
    knot_grid: Optional[torch.Tensor],
    num_knots: int,
) -> bool:
    """Recognize exact canonical unit-grid representations, not approximate grids.

    ``None`` means the kernel's default grid and is therefore valid.  Exact
    equality is intentional. Also accept FP32 linspace rounded to BF16/FP16
    and optionally promoted again: model.to(dtype) rounds the frozen buffer.
    The kernels must use those stored knots, not replace them with FP32 knots.
    Limit low-precision grids so their scaled rounding error is below 1/2;
    only one of the two outer halo knots can then be active at a time.
    Checks are cached by storage and mutation version. Normal in-place
    PyTorch writes invalidate the result; writes through ``.data`` or external
    pointers bypass version tracking and are unsupported for a frozen grid.
    """
    if knot_grid is None:
        return True
    if knot_grid.numel() != num_knots or not knot_grid.is_floating_point():
        return False
    try:
        version = knot_grid._version
    except RuntimeError:  # inference tensors have no mutation counter
        version = None
    key = None
    if version is not None:
        key = (
            knot_grid.device, knot_grid.dtype, knot_grid.data_ptr(),
            tuple(knot_grid.shape), tuple(knot_grid.stride()), version, num_knots,
        )
        cached = _GRID_CHECKS.get(key)
        if cached is not None:
            _GRID_CHECKS.move_to_end(key)
            return cached[1]
    if knot_grid.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("validate the compact spline grid before CUDA graph capture")
    flat = knot_grid.reshape(-1)
    valid = False
    if knot_grid.dtype in (torch.float32, torch.float64):
        expected = torch.linspace(0.0, 1.0, num_knots, dtype=knot_grid.dtype,
                                  device=knot_grid.device)
        valid = bool(torch.equal(flat, expected))
    if not valid:
        canonical = torch.linspace(0.0, 1.0, num_knots, dtype=torch.float32,
                                   device=knot_grid.device)
        # On [0,1], nearest-rounding error <= 1/512 (BF16) or 1/4096
        # (FP16), plus <= 2**-24 for the original FP32 linspace. These
        # limits keep (K-1)*error < 1/2, covering support with four slots.
        for storage_dtype, max_knots in ((torch.bfloat16, 256), (torch.float16, 2048)):
            if num_knots > max_knots:
                continue
            if knot_grid.dtype not in (torch.float32, torch.float64, storage_dtype):
                continue
            expected = canonical.to(storage_dtype).to(knot_grid.dtype)
            if torch.equal(flat, expected):
                valid = True
                break
    if key is not None:
        _GRID_CHECKS[key] = (knot_grid, valid)
        if len(_GRID_CHECKS) > 64:
            _GRID_CHECKS.popitem(last=False)
    return valid


def compact_quadratic_support(
    x: torch.Tensor,
    num_knots: int,
    knot_grid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference compact support for normalized quadratic B-splines.

    Returns ``(indices, basis, basis_derivative)``. The last dimension has size
    four and contains the local knot indices (including the boundary halo), normalized
    basis weights, and derivatives with respect to ``x``.  The formula is
    equivalent to the dense reference for ``x`` in ``[0, 1]`` and a recognized
    canonical grid, using its stored values. It rejects custom grids rather than
    silently applying the wrong geometry.
    """
    if num_knots < 3:
        raise ValueError("quadratic compact support needs at least 3 knots")
    if not is_uniform_unit_grid(knot_grid, num_knots):
        raise ValueError("compact support requires a uniform unit knot grid")

    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x_work = x.to(dtype=work_dtype)
    scale = float(num_knots - 1)
    # Select the mathematical support first, then account for rounded knots.
    left = torch.floor(x_work * scale - 0.5).to(torch.long)
    left = left.clamp(0, num_knots - 3)

    if knot_grid is None:
        grid = torch.linspace(
            0.0, 1.0, num_knots, dtype=work_dtype, device=x.device
        )
    else:
        grid = knot_grid.to(device=x.device, dtype=work_dtype).reshape(-1)
    previous = grid[(left - 1).clamp_min(0)]
    previous_active = (left > 0) & ((x_work - previous).abs() * scale < 1.5)
    left = torch.where(previous_active, left - 1, left)
    q = torch.arange(4, device=x.device, dtype=torch.long)
    indices = left.unsqueeze(-1) + q
    valid = indices < num_knots
    selected_grid = grid[indices.clamp_max(num_knots - 1)]
    distance = (x_work.unsqueeze(-1) - selected_grid).abs() * scale
    weights = torch.where(
        distance < 0.5,
        0.75 - distance.square(),
        torch.where(
            distance < 1.5,
            0.5 * (1.5 - distance).square(),
            torch.zeros_like(distance),
        ),
    )
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    total = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = weights / total
    derivative_distance = torch.where(
        distance < 0.5,
        -2.0 * distance,
        torch.where(
            distance < 1.5,
            -(1.5 - distance),
            torch.zeros_like(distance),
        ),
    )
    raw_derivative = (
        derivative_distance
        * torch.sign(x_work.unsqueeze(-1) - selected_grid)
        * scale
    )
    raw_derivative = torch.where(valid, raw_derivative, torch.zeros_like(raw_derivative))
    derivative = (
        raw_derivative - weights * raw_derivative.sum(dim=-1, keepdim=True)
    ) / total
    # Match the device contract's partition-of-unity derivative.
    derivative = torch.stack(
        (derivative[..., 0],
         -((derivative[..., 0] + derivative[..., 2]) + derivative[..., 3]),
         derivative[..., 2], derivative[..., 3]),
        dim=-1,
    )
    # Out-of-grid halo slots carry zero weight/derivative and alias the last
    # legal knot in the reference's scatter representation.
    return indices.clamp_max(num_knots - 1), weights, derivative
