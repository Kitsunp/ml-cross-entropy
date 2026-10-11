"""Trainer-side phase control for graph-stable patch-level training."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Literal, Optional

import torch

PatchTrainingPhase = Literal["patch", "token"]


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.") from error


def pool_patch_inputs(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    patch_size: int,
    coordinates: torch.Tensor | None = None,
    corruption_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool effective embeddings and optional clean token geometry.

    Inputs end in ``[raw_tokens, features]`` and masks in ``[raw_tokens]``.
    Embeddings must already include every composition and MEAP override. For
    each K-token patch, average embeddings over valid slots, but coordinates
    only over valid, uncorrupted slots. This preserves the existing exclusion
    of MEAP positions from JToK/JToK-M geometry gradients and router counts.
    It is an explicit curriculum extension, not equivalence to token training.

    Return ``(embedding, coordinate, core_valid, geometry_valid, fraction)``.
    All leading output domains have ``raw_tokens // K`` rows. ``fraction`` is
    the fraction of valid current input slots corrupted by MEAP, not a mask
    on future target slots. Empty embeddings/coordinates are zero and their
    corresponding validity is false. The coordinate placeholder must never
    enter a router or auxiliary population when ``geometry_valid`` is false.

    Sums use FP32 for FP16/BF16 inputs and preserve FP64 for reference tests;
    pooled values return to their own input dtype. The VJP distributes each
    pooled cotangent uniformly over its contributing slots, with no detach.
    Tensor values do not control Python branches. This helper does not form
    labels, run an embedding generator, or claim full-model compile stability.
    """
    patch_size = _integer(patch_size, "patch_size")
    if patch_size < 1:
        raise ValueError("patch_size must be positive.")
    if embeddings.ndim < 2 or not embeddings.is_floating_point():
        raise ValueError("embeddings must be floating point with a token and feature dimension.")
    raw_tokens = embeddings.size(-2)
    if raw_tokens == 0 or raw_tokens % patch_size:
        raise ValueError("raw token count must be positive and divisible by patch_size.")
    domain = embeddings.shape[:-1]
    for name, mask in (("valid_mask", valid_mask), ("corruption_mask", corruption_mask)):
        if mask is None:
            continue
        if mask.shape != domain or mask.dtype != torch.bool or mask.device != embeddings.device:
            raise ValueError(f"{name} must be boolean and match the embedding token domain/device.")
    if coordinates is not None:
        if (
            coordinates.shape[:-1] != domain
            or not coordinates.is_floating_point()
            or coordinates.device != embeddings.device
        ):
            raise ValueError(
                "coordinates must be floating point and match the token domain/device."
            )

    rows = raw_tokens // patch_size
    grouped_valid = valid_mask.unflatten(-1, (rows, patch_size))
    counts = grouped_valid.sum(dim=-1)
    if corruption_mask is None:
        grouped_clean = grouped_valid
        fraction = counts.new_zeros(counts.shape, dtype=torch.float32)
    else:
        grouped_corrupt = corruption_mask.unflatten(-1, (rows, patch_size)) & grouped_valid
        grouped_clean = grouped_valid & ~grouped_corrupt
        fraction = grouped_corrupt.sum(dim=-1).float() / counts.clamp_min(1).float()
    clean_counts = grouped_clean.sum(dim=-1)

    def pool(values, members, member_counts):
        grouped = values.unflatten(-2, (rows, patch_size))
        accumulation_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        # Selection before summation avoids 0 * NaN at an excluded slot.
        total = torch.where(members.unsqueeze(-1), grouped, 0).sum(dim=-2, dtype=accumulation_dtype)
        return (total / member_counts.clamp_min(1).unsqueeze(-1)).to(values.dtype).contiguous()

    pooled_embeddings = pool(embeddings, grouped_valid, counts)
    pooled_coordinates = (
        None if coordinates is None else pool(coordinates, grouped_clean, clean_counts)
    )
    return (
        pooled_embeddings,
        pooled_coordinates,
        counts.gt(0).contiguous(),
        clean_counts.gt(0).contiguous(),
        fraction.contiguous(),
    )


@dataclass(frozen=True)
class PatchTrainingSchedule:
    """Select the patch or token phase outside the compiled model.

    ``global_step`` is the number of completed optimizer updates before the
    next update, not a microbatch, example or token counter. All microbatches
    accumulated into that update must receive the same value. With
    ``patch_training_steps=90_000``, values ``0..89_999`` use patch targets and
    value ``90_000`` selects the first token update. On resume, supply the
    restored update count, not a new counter starting at zero.

    The CCE flag remains enabled in both phases; token targets are packed into
    a fixed final dimension so changing phases does not change the loss
    signature. Data consumption can be logged independently but cannot change
    this schedule's phase boundary.
    """

    patch_training_steps: int
    patch_size: int
    ignore_index: int = -100

    def __post_init__(self) -> None:
        patch_training_steps = _integer(self.patch_training_steps, "patch_training_steps")
        patch_size = _integer(self.patch_size, "patch_size")
        ignore_index = _integer(self.ignore_index, "ignore_index")
        if patch_training_steps < 0:
            raise ValueError("patch_training_steps must be non-negative.")
        if patch_size < 1:
            raise ValueError("patch_size must be positive.")
        object.__setattr__(self, "patch_training_steps", patch_training_steps)
        object.__setattr__(self, "patch_size", patch_size)
        object.__setattr__(self, "ignore_index", ignore_index)

    def phase(self, global_step: int) -> PatchTrainingPhase:
        """Select the next update from the completed optimizer-update count."""
        step = _integer(global_step, "global_step")
        if step < 0:
            raise ValueError("global_step must be non-negative.")
        return "patch" if step < self.patch_training_steps else "token"

    def is_patch_step(self, global_step: int) -> bool:
        """Return whether ``global_step`` belongs to patch-level training."""
        return self.phase(global_step) == "patch"

    def is_transition_step(self, global_step: int) -> bool:
        """Return whether this is the first token-level optimizer step."""
        step = _integer(global_step, "global_step")
        if step < 0:
            raise ValueError("global_step must be non-negative.")
        return step == self.patch_training_steps

    def prepare_patch_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Validate patch targets and normalize their compiled input layout."""
        if targets.ndim < 1 or targets.size(-1) != self.patch_size:
            raise ValueError(
                f"Patch targets must end in patch_size={self.patch_size}, got {tuple(targets.shape)}."
            )
        # The paper-aligned ``labels[..., K:].unflatten(...)`` view retains the
        # skipped prefix in its leading stride. Token-phase targets do not. A
        # one-time contiguous copy outside the compiled core prevents Dynamo
        # from specializing a second graph on that layout difference.
        return targets.contiguous()

    def prepare_token_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Pack ordinary token targets into ``[..., K]`` without changing K."""
        packed = targets.new_full((*targets.shape, self.patch_size), self.ignore_index)
        packed[..., 0] = targets
        return packed

    def targets_for_step(
        self,
        global_step: int,
        *,
        patch_targets: Optional[torch.Tensor] = None,
        token_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select and shape targets before entering a compiled training step."""
        if self.is_patch_step(global_step):
            if patch_targets is None:
                raise ValueError("patch_targets are required during patch-level training.")
            return self.prepare_patch_targets(patch_targets)
        if token_targets is None:
            raise ValueError("token_targets are required during token-level training.")
        return self.prepare_token_targets(token_targets)
