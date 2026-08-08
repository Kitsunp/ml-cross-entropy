"""Adapter for the ``modeling_neollm.LeviathanGenerator`` class.

The destination project does not own NeoLLM's large modeling file.  This
module therefore supplies a drop-in subclass/factory instead of copying that
file and its unrelated transformer components into the CCE package.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .autograd_fn import _HAS_KERNEL_FORWARD
from .compiler import leviathan_embedding_compiler_safe
from .dispatch import supports


def make_triton_leviathan_generator(reference_cls: type[nn.Module]) -> type[nn.Module]:
    """Create a state-dict-compatible Triton subclass of a model generator."""

    class TritonLeviathanGenerator(reference_cls):
        use_leviathan_triton = True

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            # This is the original model implementation's fallback boundary:
            # CPU, unsupported shapes, and an explicitly disabled kernel keep
            # using ``LeviathanGenerator.forward`` unchanged.
            if (
                not self.use_leviathan_triton
                or not token_ids.is_cuda
                or not _HAS_KERNEL_FORWARD
                or not supports(self.config, dtype_override=self.codebooks.dtype)
            ):
                return super().forward(token_ids)

            params = {
                "codebooks": self.codebooks,
                "head_proj_weight": self.head_proj_weight,
                "head_norm_weight": self.head_norm_weight,
                "head_norm_bias": self.head_norm_bias,
                "head_spline_delta": self.head_spline_delta,
                "head_out_weight": self.head_out_weight,
            }
            return leviathan_embedding_compiler_safe(
                token_ids,
                params,
                self.config,
                self.knot_grid,
            )

    TritonLeviathanGenerator.__name__ = f"{reference_cls.__name__}Triton"
    TritonLeviathanGenerator.__qualname__ = TritonLeviathanGenerator.__name__
    return TritonLeviathanGenerator


def replace_leviathan_generator(
    model: nn.Module,
    *,
    use_kernel: bool = True,
) -> nn.Module:
    """Replace ``model.model.token_generator`` without changing state names.

    Call this after constructing ``NeoLLMForCausalLM`` and before FSDP wrapping
    or ``torch.compile``.  The replacement subclasses the model's own
    ``LeviathanGenerator``, so its parameter names and initialization contract
    remain unchanged.  The returned object is the same model instance.
    """
    container: Any = getattr(model, "model", model)
    generator = getattr(container, "token_generator", None)
    if generator is None:
        raise ValueError(
            "The model has no `token_generator`; construct NeoLLM with "
            "`use_token_generator=True` first."
        )

    generator_cls = type(generator)
    if getattr(generator_cls, "use_leviathan_triton", False):
        generator.use_leviathan_triton = bool(use_kernel)
        return model

    triton_cls = make_triton_leviathan_generator(generator_cls)
    replacement = triton_cls(generator.config)
    first_parameter = next(generator.parameters())
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in generator.named_parameters()
    }
    original_knot_grid = getattr(generator, "knot_grid", None)
    if original_knot_grid is not None:
        original_knot_grid = original_knot_grid.detach().clone()
    replacement.to(device=first_parameter.device, dtype=first_parameter.dtype)
    if original_knot_grid is not None:
        # ``knot_grid`` is a non-persistent buffer.  Restore it after the
        # module cast so an FP32 grid is not rounded to BF16 by replacement.
        replacement._buffers["knot_grid"] = original_knot_grid.to(
            device=first_parameter.device
        )
    replacement.load_state_dict(generator.state_dict(), strict=True)
    for name, parameter in replacement.named_parameters():
        if name in original_requires_grad:
            parameter.requires_grad_(original_requires_grad[name])
    replacement.train(generator.training)
    replacement.use_leviathan_triton = bool(use_kernel)
    container.token_generator = replacement
    return model


__all__ = [
    "make_triton_leviathan_generator",
    "replace_leviathan_generator",
]
