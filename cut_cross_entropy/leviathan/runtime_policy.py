"""Internal architecture policy for Leviathan kernel specializations."""

from __future__ import annotations

import os

import torch


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
