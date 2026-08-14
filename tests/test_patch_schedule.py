from __future__ import annotations

import pytest
import torch

from cut_cross_entropy import PatchTrainingSchedule


def test_patch_schedule_has_an_exact_zero_based_boundary() -> None:
    schedule = PatchTrainingSchedule(patch_training_steps=3, patch_size=4)

    assert [schedule.phase(step) for step in range(5)] == [
        "patch",
        "patch",
        "patch",
        "token",
        "token",
    ]
    assert schedule.is_patch_step(2)
    assert not schedule.is_patch_step(3)
    assert schedule.is_transition_step(3)
    assert not schedule.is_transition_step(2)


def test_patch_schedule_keeps_a_fixed_target_shape_across_phases() -> None:
    schedule = PatchTrainingSchedule(patch_training_steps=2, patch_size=4, ignore_index=-100)
    patch_targets = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    token_targets = torch.tensor([[9, 10]])

    patch = schedule.targets_for_step(1, patch_targets=patch_targets)
    token = schedule.targets_for_step(2, token_targets=token_targets)

    assert patch is patch_targets
    assert patch.shape == token.shape == (1, 2, 4)
    torch.testing.assert_close(token[..., 0], token_targets)
    assert torch.all(token[..., 1:] == -100)


def test_patch_targets_normalize_the_shifted_view_layout() -> None:
    schedule = PatchTrainingSchedule(patch_training_steps=2, patch_size=4)
    labels = torch.arange(2 * 16).view(2, 16)
    shifted = labels[..., 4:].unflatten(-1, (3, 4))

    assert not shifted.is_contiguous()
    prepared = schedule.prepare_patch_targets(shifted)

    assert prepared.is_contiguous()
    assert prepared.storage_offset() == 0
    torch.testing.assert_close(prepared, shifted)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"patch_training_steps": -1, "patch_size": 4}, ValueError, "non-negative"),
        ({"patch_training_steps": 1, "patch_size": 0}, ValueError, "positive"),
        ({"patch_training_steps": True, "patch_size": 4}, TypeError, "not bool"),
    ],
)
def test_patch_schedule_validates_configuration(kwargs, error, message: str) -> None:
    with pytest.raises(error, match=message):
        PatchTrainingSchedule(**kwargs)


def test_patch_schedule_rejects_missing_or_misaligned_phase_targets() -> None:
    schedule = PatchTrainingSchedule(patch_training_steps=2, patch_size=4)

    with pytest.raises(ValueError, match="patch_targets are required"):
        schedule.targets_for_step(0)
    with pytest.raises(ValueError, match="token_targets are required"):
        schedule.targets_for_step(2)
    with pytest.raises(ValueError, match="must end in patch_size=4"):
        schedule.targets_for_step(0, patch_targets=torch.ones(2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="global_step must be non-negative"):
        schedule.phase(-1)
