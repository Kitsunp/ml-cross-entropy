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
    """Oracle path: build the reference module from cfg and load the params."""
    from .core import LeviathanConfig, LeviathanGenerator

    c = LeviathanConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        generator_d_seed=cfg.generator_d_seed,
        generator_num_modes=cfg.generator_num_modes,
        generator_num_knots=cfg.generator_num_knots,
        generator_spline_degree=cfg.generator_spline_degree,
        generator_k=cfg.generator_k,
        generator_krank=cfg.generator_krank,
        dtype=getattr(cfg, "dtype", torch.bfloat16),
    )
    gen = LeviathanGenerator(c)
    device = next(iter(params.values())).device
    gen = gen.to(device=device, dtype=params["codebooks"].dtype)
    state = {k: v for k, v in params.items() if k in gen.state_dict()}
    gen.load_state_dict(state, strict=False)
    if "knot_grid" in params:
        gen.knot_grid = params["knot_grid"].to(device=device,
                                               dtype=torch.float32)
    return gen(ids)


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
