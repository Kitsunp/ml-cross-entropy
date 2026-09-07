"""Optional CUDA integration hooks introduced in PyTorch 2.14.

The helpers in this module are intentionally no-ops on older PyTorch releases.
They add optional profiler attribution and tell CUDA Graph Trees when a newly
observed external-kernel geometry needs one more warmup iteration. They do not
change kernel math, launch geometry, allocator behavior, or the global compiler
policy.
"""

from __future__ import annotations

import os
import threading
from contextlib import ExitStack, contextmanager
from typing import Hashable, Iterator

import torch

from cut_cross_entropy.utils import is_package_greater_or_equal


TORCH_2_14_OR_NEWER = is_package_greater_or_equal("torch", "2.14")
TORCH_2_14_CUDA_INTEGRATION = bool(
    TORCH_2_14_OR_NEWER
    and os.environ.get("CUT_CROSS_ENTROPY_TORCH_2_14_INTEGRATION", "1") != "0"
)
TORCH_2_14_GRAPH_ANNOTATIONS = bool(
    TORCH_2_14_CUDA_INTEGRATION
    and os.environ.get("CUT_CROSS_ENTROPY_GRAPH_ANNOTATIONS", "0") == "1"
)
TORCH_2_14_MEMORY_ANNOTATIONS = bool(
    TORCH_2_14_CUDA_INTEGRATION
    and os.environ.get("CUT_CROSS_ENTROPY_MEMORY_ANNOTATIONS", "0") == "1"
)

if TORCH_2_14_GRAPH_ANNOTATIONS:
    try:
        from torch.cuda.graph_annotations import mark_kernels as _mark_kernels
    except (ImportError, AttributeError):  # pragma: no cover - patched/vendor builds
        _mark_kernels = None
else:
    _mark_kernels = None

_mark_warmup_incomplete = (
    getattr(torch.compiler, "cudagraph_mark_warmup_incomplete", None)
    if TORCH_2_14_CUDA_INTEGRATION
    else None
)
_annotate_tensor = (
    getattr(torch.cuda.memory, "_annotate_tensor", None)
    if TORCH_2_14_MEMORY_ANNOTATIONS
    else None
)
TORCH_2_14_CUDA_KERNEL_CONTEXT = bool(_mark_kernels is not None)

_WARMED_GEOMETRIES: set[tuple[str, Hashable]] = set()
_WARMED_GEOMETRIES_LOCK = threading.Lock()


def _kernel_annotation(name: str) -> dict[str, str]:
    component, phase = name.split(".", 1)
    return {
        "name": name,
        "library": "cut_cross_entropy",
        "component": component,
        "phase": phase,
        "operator": f"cut_cross_entropy::{component}_{phase}",
    }


@contextmanager
def cuda_kernel_region(name: str, device: torch.device) -> Iterator[None]:
    """Attribute one external-kernel region with PyTorch 2.14 APIs."""
    del device
    with ExitStack() as stack:
        if _mark_kernels is not None:
            stack.enter_context(
                _mark_kernels(
                    _kernel_annotation(name),
                    backward=False,
                )
            )
        yield


def mark_warmup_incomplete_once(name: str, geometry: Hashable) -> None:
    """Request one additional CUDA Graph Trees warmup for a new geometry.

    Call this after a successful first launch.  The PyTorch API is a no-op
    outside CUDA Graph Trees warmup, while the local cache keeps steady-state
    calls free of repeated compiler hooks.
    """
    if _mark_warmup_incomplete is None:
        return
    key = (name, geometry)
    if key in _WARMED_GEOMETRIES:
        return
    with _WARMED_GEOMETRIES_LOCK:
        if key in _WARMED_GEOMETRIES:
            return
        _mark_warmup_incomplete()
        _WARMED_GEOMETRIES.add(key)


def annotate_tensors(name: str, **tensors: torch.Tensor | None) -> None:
    """Label CUDA allocations for ``memory_viz`` when explicitly requested."""
    if _annotate_tensor is None:
        return
    for role, tensor in tensors.items():
        if tensor is not None and tensor.is_cuda:
            _annotate_tensor(tensor, f"cut_cross_entropy::{name}.{role}")


__all__ = [
    "TORCH_2_14_CUDA_INTEGRATION",
    "TORCH_2_14_CUDA_KERNEL_CONTEXT",
    "TORCH_2_14_GRAPH_ANNOTATIONS",
    "TORCH_2_14_MEMORY_ANNOTATIONS",
    "TORCH_2_14_OR_NEWER",
    "annotate_tensors",
    "cuda_kernel_region",
    "mark_warmup_incomplete_once",
]
