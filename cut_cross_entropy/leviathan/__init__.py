"""Integrated Leviathan kernels and model consumer for the LEV layer.

Layout (flat, per contract): forward_impl.py (Triton forward),
backward_impl.py (backward; CPU-torch implementation, verified 23/23),
autograd_fn.py (LeviathanFunction — the single integrable entry point),
dispatch.py (supports() + leviathan_embedding with reference fallback).
"""
from .autograd_fn import (  # noqa: F401
    _HAS_KERNEL_FORWARD,
    LeviathanFunction,
    leviathan_apply,
)
from .backward_impl import leviathan_backward, leviathan_forward_ref  # noqa: F401
from .compiler import leviathan_embedding_compiler_safe  # noqa: F401
from .core import LeviathanConfig, LeviathanGenerator, build_generator  # noqa: F401
from .dispatch import leviathan_embedding, supports  # noqa: F401
from .jtok import (  # noqa: F401
    jtok_apply,
    jtok_reference,
    jtokm_apply,
    jtokm_auxiliary_loss,
    jtokm_reference,
    jtokm_routing_stats,
)
from .model import LeviathanEmbedding, LeviathanForCausalLM  # noqa: F401
from .neollm import (  # noqa: F401
    apply_neollm_jtok,
    make_triton_leviathan_generator,
    replace_leviathan_generator,
)

__all__ = [
    "LeviathanFunction",
    "leviathan_apply",
    "leviathan_backward",
    "leviathan_forward_ref",
    "leviathan_embedding",
    "leviathan_embedding_compiler_safe",
    "supports",
    "LeviathanConfig",
    "LeviathanGenerator",
    "build_generator",
    "LeviathanEmbedding",
    "LeviathanForCausalLM",
    "make_triton_leviathan_generator",
    "replace_leviathan_generator",
    "apply_neollm_jtok",
    "jtok_apply",
    "jtok_reference",
    "jtokm_apply",
    "jtokm_auxiliary_loss",
    "jtokm_reference",
    "jtokm_routing_stats",
    "_HAS_KERNEL_FORWARD",
]
