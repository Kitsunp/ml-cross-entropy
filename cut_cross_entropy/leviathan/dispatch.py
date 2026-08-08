"""Dispatch for the LEV forward: supports() + automatic fallback to reference.

supports(cfg)   -> True only for configs the Triton kernels implement exactly.
                   Unsupported configs fall back to the leviathan_core
                   reference (LeviathanGenerator), which handles everything
                   (fp32 models, odd shapes, k>4, ...) at reference speed.
leviathan_embedding(ids, params, cfg) -> embeds (kernel path or reference)
"""

from __future__ import annotations

import math

import torch

# Keep the integrated package importable on CPU-only environments.  Triton is
# optional for the reference fallback, so do not eagerly import the legacy
# ``forward/`` worktree (or fail when Triton is not installed).
forward_impl = None
try:
    from . import forward_impl as forward_impl  # type: ignore
except ImportError:
    try:  # flat layout: leviathan_kernels/forward_impl.py
        import forward_impl as forward_impl  # type: ignore
    except ImportError:
        try:  # original worktree layout: forward/forward_impl.py
            from forward import forward_impl as forward_impl  # type: ignore
        except ImportError:
            pass

# design-time chunking constant of kernel A (see forward_impl.D_CHUNK)
_MIN_D_CHUNK = 4


def supports(cfg, variant: str = "exact") -> bool:
    """True if the Triton forward implements this config exactly.

    Ranges (documented in README):
        k       2..4            (generator_k)
        d_seed  32..256, power of 2 (generator_d_seed)
        krank   16..128, power of 2 (generator_krank)
        kappa   8..32, power of 2 (generator_num_knots)
        h       2..16           (generator_num_modes)
        D       64..2048, %16==0 (hidden_size)
        dtype   torch.bfloat16  (kernel path; fp32 models fall back)
        b       derived: ceil(vocab_size ** (1/k)), requires b**k >= vocab_size
    """
    try:
        k = cfg.generator_k
        d = cfg.generator_d_seed
        krank = cfg.generator_krank
        kappa = cfg.generator_num_knots
        h = cfg.generator_num_modes
        D = cfg.hidden_size
        b = math.ceil(cfg.vocab_size ** (1.0 / k))
        ok = (variant in ("exact", "fast")
              and 2 <= k <= 4
              and 32 <= d <= 256 and d & (d - 1) == 0
              and 16 <= krank <= 128 and krank & (krank - 1) == 0
              and 8 <= kappa <= 32 and kappa & (kappa - 1) == 0
              and 2 <= h <= 16
              and 64 <= D <= 2048 and D % 16 == 0
              and b ** k >= cfg.vocab_size
              and getattr(cfg, "dtype", torch.bfloat16) == torch.bfloat16)
        return bool(ok)
    except (AttributeError, TypeError, ValueError):
        return False


def _reference_forward(ids, params, cfg):
    """Differentiable oracle path using the supplied parameter tensors.

    Constructing a temporary module and loading a state dict here would copy
    the inputs into fresh ``Parameter`` objects and disconnect their autograd
    history.  The dict-based reference keeps the original tensors in the
    computation graph and also honors an explicitly supplied knot grid.
    """
    from .backward_impl import leviathan_forward_ref

    embeds, _ = leviathan_forward_ref(
        ids,
        params,
        cfg,
        save_intermediates=False,
    )
    return embeds


def leviathan_embedding(ids, params, cfg, save_intermediates=False,
                        variant: str = "exact"):
    """embeds [*ids.shape, D] -- Triton kernel path, or reference fallback.

    The kernel path is used when supports(cfg) and a launch backend is
    available (CUDA, or TRITON_INTERPRET=1 for CPU sanity checks).  In every
    other case this falls back to the leviathan_core reference module.
    """
    if forward_impl is not None and supports(cfg, variant):
        try:
            embeds, _ = forward_impl.leviathan_forward(
                ids, params, cfg, save_intermediates=save_intermediates,
                variant=variant)
            return embeds
        except (RuntimeError, TypeError, ValueError):
            # no CUDA / wrong device / dtype mismatch -> reference
            pass
    return _reference_forward(ids, params, cfg)
