from __future__ import annotations

import pytest
import torch

from cut_cross_entropy import PatchTrainingSchedule, pool_patch_inputs


def test_pool_patch_inputs_separates_embedding_and_clean_geometry_domains():
    embeddings = torch.arange(24, dtype=torch.float64).reshape(1, 12, 2)
    coordinates = embeddings / 100
    valid = torch.tensor(
        [[True, True, False, True, True, True, True, True, False, False, False, False]]
    )
    corrupt = torch.tensor(
        [[False, True, False, False, True, True, True, True, True, False, True, False]]
    )
    embeddings = embeddings.masked_fill(~valid.unsqueeze(-1), torch.nan)
    coordinates = coordinates.masked_fill((~valid | corrupt).unsqueeze(-1), torch.nan)
    e, z, core_valid, geometry_valid, fraction = pool_patch_inputs(
        embeddings, valid, patch_size=4, coordinates=coordinates, corruption_mask=corrupt
    )
    torch.testing.assert_close(e[0, 0], embeddings[0, [0, 1, 3]].mean(dim=0))
    torch.testing.assert_close(e[0, 1], embeddings[0, 4:8].mean(dim=0))
    torch.testing.assert_close(e[0, 2], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(z[0, 0], coordinates[0, [0, 3]].mean(dim=0))
    torch.testing.assert_close(z[0, 1:], torch.zeros(2, 2, dtype=torch.float64))
    assert core_valid.tolist() == [[True, True, False]]
    assert geometry_valid.tolist() == [[True, False, False]]
    torch.testing.assert_close(fraction, torch.tensor([[1 / 3, 1.0, 0.0]]))


def test_pool_patch_inputs_has_the_masked_pooling_adjoint():
    torch.manual_seed(812)
    base = torch.randn(1, 8, 2, dtype=torch.float64, requires_grad=True)
    coordinates = torch.rand(1, 8, 3, dtype=torch.float64, requires_grad=True)
    mask_embedding = torch.randn(2, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True, False, True, True, True, True, True]])
    corrupt = torch.tensor([[False, True, False, False, True, True, True, True]])
    effective = torch.where(corrupt.unsqueeze(-1), mask_embedding, base)
    e, z, _, _, _ = pool_patch_inputs(
        effective, valid, patch_size=4, coordinates=coordinates, corruption_mask=corrupt
    )
    ge = torch.randn_like(e)
    gz = torch.randn_like(z)
    ((e * ge).sum() + (z * gz).sum()).backward()
    expected_base = torch.zeros_like(base)
    expected_base[0, [0, 3]] = ge[0, 0] / 3
    expected_z = torch.zeros_like(coordinates)
    expected_z[0, [0, 3]] = gz[0, 0] / 2
    torch.testing.assert_close(base.grad, expected_base, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(coordinates.grad, expected_z, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(mask_embedding.grad, ge[0, 0] / 3 + ge[0, 1], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_pool_patch_inputs_k1_identity_and_consistent_core_layout(dtype):
    torch.manual_seed(813)
    base = torch.randn(2, 17, 5, dtype=dtype)
    embeddings = base[:, 1:]
    coordinates = torch.rand(2, 16, 3, dtype=torch.float32)
    for k, inputs in ((4, embeddings), (1, embeddings[:, :4])):
        z = coordinates[:, : inputs.size(1)]
        valid = torch.ones(inputs.shape[:-1], dtype=torch.bool)
        e, pooled_z, core_valid, geometry_valid, fraction = pool_patch_inputs(
            inputs, valid, patch_size=k, coordinates=z
        )
        assert e.dtype == dtype and pooled_z.dtype == coordinates.dtype
        assert e.shape == (2, 4, 5) and pooled_z.shape == (2, 4, 3)
        assert all(
            t.is_contiguous() and t.storage_offset() == 0
            for t in (e, pooled_z, core_valid, geometry_valid, fraction)
        )
        assert core_valid.all() and geometry_valid.all() and not fraction.any()
        if k == 1:
            assert torch.equal(e, inputs)
            assert torch.equal(pooled_z, z)


def test_pool_patch_inputs_is_local_to_the_current_input_patch():
    embeddings = torch.arange(12.0).reshape(1, 12, 1)
    valid = torch.ones(1, 12, dtype=torch.bool)
    before = pool_patch_inputs(embeddings, valid, patch_size=4)
    embeddings[:, 8:] += 100
    after = pool_patch_inputs(embeddings, valid, patch_size=4)
    assert before[1] is after[1] is None
    torch.testing.assert_close(before[0][:, :2], after[0][:, :2], rtol=0, atol=0)
    assert not torch.equal(before[0][:, 2:], after[0][:, 2:])


@pytest.mark.parametrize(
    "case",
    [
        "zero_k",
        "bool_k",
        "indivisible",
        "integer_embedding",
        "mask_shape",
        "mask_dtype",
        "coordinate_shape",
    ],
)
def test_pool_patch_inputs_rejects_invalid_metadata(case):
    e = torch.ones(2, 8, 3)
    mask = torch.ones(2, 8, dtype=torch.bool)
    kwargs = {"patch_size": 4}
    if case == "zero_k":
        kwargs["patch_size"] = 0
    elif case == "bool_k":
        kwargs["patch_size"] = True
    elif case == "indivisible":
        kwargs["patch_size"] = 3
    elif case == "integer_embedding":
        e = e.long()
    elif case == "mask_shape":
        mask = mask[:, :7]
    elif case == "mask_dtype":
        mask = mask.float()
    else:
        kwargs["coordinates"] = torch.ones(2, 7, 4)
    with pytest.raises((ValueError, TypeError)):
        pool_patch_inputs(e, mask, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pool_patch_inputs_prewarmed_producers_keep_value_changes_graph_stable(monkeypatch):
    """Two raw shapes are prewarmed, not falsely called one producer graph.

    This is a preparation/VJP test, not a full generator/backbone/CCE or real
    training transition. Those composed paths require their own warmup gate.
    """
    from torch._dynamo.utils import counters

    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    torch.manual_seed(814)

    def prepare(e, z, valid, corrupt, k):
        return pool_patch_inputs(e, valid, patch_size=k, coordinates=z, corruption_mask=corrupt)

    compiled = torch.compile(prepare, mode="max-autotune", fullgraph=True)
    inputs = {}
    for k in (1, 4):
        e = torch.randn(2, 4 * k, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        z = torch.rand(2, 4 * k, 3, device="cuda", requires_grad=True)
        valid = torch.ones(2, 4 * k, device="cuda", dtype=torch.bool)
        corrupt = torch.zeros_like(valid)
        inputs[k] = (e, z, valid, corrupt)

    def step(k, check=False):
        e, z, valid, corrupt = inputs[k]
        torch.compiler.cudagraph_mark_step_begin()
        e.grad = z.grad = None
        actual = compiled(e, z, valid, corrupt, k)
        (actual[0].float().square().mean() + actual[1].square().mean()).backward()
        if check:
            refs = [v.detach().clone().requires_grad_() for v in (e, z)]
            expected = prepare(*refs, valid, corrupt, k)
            (expected[0].float().square().mean() + expected[1].square().mean()).backward()
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            for a, b in zip((e, z), refs):
                torch.testing.assert_close(a.grad, b.grad)

    for _ in range(5):
        for k in (4, 1):
            step(k)
    warmed = (counters["stats"]["unique_graphs"], sum(counters["graph_break"].values()))
    for case in ("mixed", "all_corrupt", "all_invalid"):
        for k in (4, 1):
            _, _, valid, corrupt = inputs[k]
            valid.fill_(case != "all_invalid")
            corrupt.fill_(case == "all_corrupt")
            if case == "mixed":
                corrupt[:, ::2] = True
            step(k, check=True)
            assert warmed == (
                counters["stats"]["unique_graphs"],
                sum(counters["graph_break"].values()),
            )


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


@pytest.mark.parametrize("accumulation_steps", [1, 4])
def test_patch_schedule_counts_updates_not_microbatches_or_data(accumulation_steps) -> None:
    schedule = PatchTrainingSchedule(patch_training_steps=2, patch_size=4)
    completed_updates = 0
    consumed_tokens = 0
    phases = []
    for update in range(4):
        for microbatch in range(accumulation_steps):
            # Unequal batch populations must not move a step-based boundary.
            consumed_tokens += (update + 1) * (microbatch + 1) * 127
            phases.append(schedule.phase(completed_updates))
        completed_updates += 1

    assert consumed_tokens > 0
    assert phases == ["patch"] * (2 * accumulation_steps) + ["token"] * (2 * accumulation_steps)


@pytest.mark.parametrize("restored_updates, expected", [(1, "patch"), (2, "token"), (7, "token")])
def test_patch_schedule_resumes_from_completed_updates(restored_updates, expected) -> None:
    schedule = PatchTrainingSchedule(patch_training_steps=2, patch_size=4)
    assert schedule.phase(restored_updates) == expected
    assert schedule.is_transition_step(restored_updates) == (restored_updates == 2)


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
