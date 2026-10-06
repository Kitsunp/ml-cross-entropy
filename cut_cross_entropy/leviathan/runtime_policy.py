"""Internal architecture policy for Leviathan kernel specializations."""

from __future__ import annotations

import os

import torch

try:
    from .spline_support import is_uniform_unit_grid
except ImportError:  # pragma: no cover - flat remote staging layout
    from spline_support import is_uniform_unit_grid


def use_dot_specialization(
    device: torch.device,
    *,
    d_seed: int,
    num_knots: int,
    krank: int,
) -> bool:
    """Use the numerically validated SM120 tensor-core specialization.

    ``LEV_DOT`` remains an existing developer-only diagnostic override.  The
    normal API needs no flag: unsupported architectures and unvalidated
    geometries retain the deterministic scalar implementation.
    """
    override = os.environ.get("LEV_DOT")
    if override is not None:
        return override == "1"
    if (
        getattr(device, "type", None) != "cuda"
        or d_seed != 128
        or num_knots != 16
        or krank != 64
    ):
        return False
    try:
        return torch.cuda.get_device_capability(device) >= (12, 0)
    except (RuntimeError, TypeError, AttributeError):
        return False


def compact_spline_requested() -> bool:
    """Whether the opt-in compact quadratic-spline candidate was requested."""
    value = os.environ.get("LEV_COMPACT_SPLINE", "0").strip().lower()
    return value in {"1", "true", "on", "yes"}


def compact_spline_supported(
    device: torch.device,
    *,
    d_seed: int,
    num_knots: int,
    krank: int,
    knot_grid: torch.Tensor | None = None,
) -> bool:
    """Return whether the compact candidate is safe for this launch.

    The first implementation is intentionally narrow: it composes with the
    already validated SM120 dot path and the standard uniform K=16 grid.  A
    requested but unsupported candidate is rejected by the host plumbing so a
    benchmark cannot silently report the dense fallback as compact.
    """
    if (getattr(device, "type", None) != "cuda"
            or (d_seed, num_knots, krank) != (128, 16, 64)):
        return False
    if os.environ.get("LEV_DOT") == "1":
        # The dense diagnostic override must not widen the compact contract.
        # Automatic dot selection already checks the architecture below.
        try:
            if torch.cuda.get_device_capability(device) < (12, 0):
                return False
        except (RuntimeError, TypeError, AttributeError):
            return False
    return (
        use_dot_specialization(
            device,
            d_seed=d_seed,
            num_knots=num_knots,
            krank=krank,
        )
        and is_uniform_unit_grid(knot_grid, num_knots)
    )
