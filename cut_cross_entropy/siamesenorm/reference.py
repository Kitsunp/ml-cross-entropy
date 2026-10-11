"""The paired RMSNorm boundary; residual updates remain with the caller."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def output_dtype(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor | None) -> torch.dtype:
    """RMSNorm keeps stream dtype; pointwise scale/add follow promotion."""
    scaled = x.dtype if scale is None else torch.promote_types(x.dtype, scale.dtype)
    return torch.promote_types(scaled, y.dtype)


def siamesenorm_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    weight_x: torch.Tensor,
    weight_y: torch.Tensor,
    weight_out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
    post_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute RMS_out(scale * RMS_x(x) + RMS_y(y)).

    All affine vectors are independent; no parameter is centered or tied. The
    optional outer norm and scale are static architecture choices. Calling this
    twice in a layer does not move the attention or MLP dependency boundaries.
    """
    a = F.rms_norm(x, (x.shape[-1],), weight_x, eps)
    b = F.rms_norm(y, (y.shape[-1],), weight_y, eps)
    if scale is not None:
        a = a * scale
    mixed = a + b
    output = (
        F.rms_norm(mixed, (mixed.shape[-1],), weight_out, eps) if weight_out is not None else mixed
    )
    if post_scale is not None:
        output = output * post_scale
    return output if out_dtype is None else output.to(out_dtype)
