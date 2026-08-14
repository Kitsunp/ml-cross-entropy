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


@dataclass(frozen=True)
class PatchTrainingSchedule:
    """Select the patch or token phase outside the compiled model.

    Steps are zero-based: with ``patch_training_steps=90_000``, steps
    ``0..89_999`` use patch targets and step ``90_000`` is the first token
    step. The CCE flag remains enabled in both phases; token targets are packed
    into a fixed final dimension so changing phases does not change the loss
    signature.
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
        """Return the phase for a zero-based optimizer step."""
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
