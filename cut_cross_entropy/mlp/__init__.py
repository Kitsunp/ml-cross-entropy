"""Reference contract and CuTe backends for the NeoLLM training MLP."""

from typing import Any

from .compiler import gupn, gupn_uses_cute
from .reference import (
    fan_reference,
    gupn_backward_reference,
    gupn_reference,
    neollm_mlp_reference,
)


def fp8_gupn(*args: Any, **kwargs: Any):
    """Load the optional TorchAO FP8 route only when it is requested."""
    from .fp8 import fp8_gupn as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "fan_reference",
    "fp8_gupn",
    "gupn",
    "gupn_reference",
    "gupn_uses_cute",
    "gupn_backward_reference",
    "neollm_mlp_reference",
]
