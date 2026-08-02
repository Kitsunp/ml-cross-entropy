# Copyright (C) 2026. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy import linear_cross_entropy, meap_mask_inputs

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _expected_count(eligible_count: torch.Tensor, ratio: float) -> torch.Tensor:
    count = (eligible_count * ratio).to(torch.long)
    if ratio > 0:
        count = torch.where(eligible_count > 0, count.clamp_min(1), count)
    return count


def _fold_seed_to_uint32(seed: int) -> int:
    low = seed & 0xFFFFFFFF
    high = (seed >> 32) & 0xFFFFFFFF
    high ^= high >> 16
    high = (high * 0x7FEB352D) & 0xFFFFFFFF
    high ^= high >> 15
    high = (high * 0x846CA68B) & 0xFFFFFFFF
    high ^= high >> 16
    return low ^ high


@pytest.mark.parametrize("implementation", ["torch"])
def test_meap_fixed_count_padding_and_last_token(implementation: str) -> None:
    input_ids = torch.arange(24).view(3, 8)
    eligible = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False, False],
            [False, False, False, False, False, False, False, False],
        ]
    )
    output, selected, metrics = meap_mask_inputs(
        input_ids,
        99,
        mask_ratio=0.4,
        eligible_mask=eligible,
        seed=17,
        return_mask=True,
        return_metrics=True,
        implementation=implementation,
    )
    effective = eligible.clone()
    effective[0, 7] = False
    effective[1, 4] = False
    assert torch.equal(selected.sum(1), _expected_count(effective.sum(1), 0.4))
    assert not selected[~effective].any()
    assert torch.equal(output[selected], torch.full_like(output[selected], 99))
    assert torch.equal(output[~selected], input_ids[~selected])
    assert int(metrics[0]) == int(effective.sum())
    assert int(metrics[1]) == int(selected.sum())


def test_meap_explicit_disable_is_noop() -> None:
    input_ids = torch.arange(12).view(2, 6)
    output = meap_mask_inputs(input_ids, 99, enabled=False)
    assert output is input_ids
    output, selected = meap_mask_inputs(input_ids, 99, enabled=False, return_mask=True)
    assert output is input_ids
    assert not selected.any()


@pytest.mark.parametrize("shape", [(0, 8), (3, 0)])
def test_meap_empty_input(shape: tuple[int, int]) -> None:
    input_ids = torch.empty(shape, dtype=torch.long)
    output, selected, metrics = meap_mask_inputs(
        input_ids,
        99,
        implementation="torch",
        return_mask=True,
        return_metrics=True,
    )
    assert output.shape == shape
    assert selected.shape == shape
    assert torch.equal(metrics, torch.zeros(2, dtype=torch.int32))


@pytest.mark.parametrize(
    "input_ids,mask_ratio,error",
    [
        (torch.zeros(2, dtype=torch.long), 0.15, ValueError),
        (torch.zeros(2, 3), 0.15, TypeError),
        (torch.zeros(2, 3, dtype=torch.long), -0.1, ValueError),
        (torch.zeros(2, 3, dtype=torch.long), 1.1, ValueError),
    ],
)
def test_meap_rejects_invalid_inputs(
    input_ids: torch.Tensor, mask_ratio: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        meap_mask_inputs(input_ids, 99, mask_ratio=mask_ratio, implementation="torch")


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("sequence_length", [17, 64, 127])
def test_meap_triton_invariants(dtype: torch.dtype, sequence_length: int) -> None:
    batch_size = 7
    input_ids = torch.arange(
        batch_size * sequence_length, device="cuda", dtype=dtype
    ).view(batch_size, sequence_length)
    lengths = torch.tensor([sequence_length - row for row in range(batch_size)], device="cuda")
    positions = torch.arange(sequence_length, device="cuda")
    eligible = positions.unsqueeze(0) < lengths.unsqueeze(1)
    output, selected, metrics = meap_mask_inputs(
        input_ids,
        32000,
        mask_ratio=0.15,
        eligible_mask=eligible,
        seed=1234,
        return_mask=True,
        return_metrics=True,
    )
    effective_count = (lengths - 1).clamp_min(0)
    assert torch.equal(selected.sum(1), _expected_count(effective_count, 0.15))
    assert not selected[~eligible].any()
    assert not selected[torch.arange(batch_size), lengths - 1].any()
    assert torch.equal(output[selected], torch.full_like(output[selected], 32000))
    assert torch.equal(output[~selected], input_ids[~selected])
    assert int(metrics[0]) == int(effective_count.sum())
    assert int(metrics[1]) == int(selected.sum())

    output_without_mask, metrics_without_mask = meap_mask_inputs(
        input_ids,
        32000,
        mask_ratio=0.15,
        eligible_mask=eligible,
        seed=1234,
        return_metrics=True,
    )
    assert torch.equal(output_without_mask, output)
    assert torch.equal(metrics_without_mask, metrics)


@skip_no_cuda
@pytest.mark.parametrize("mask_ratio", [0.0, 0.01, 0.15, 0.5, 1.0])
@pytest.mark.parametrize("sequence_length", [63, 512, 1025, 4096])
def test_meap_permutation_exact_count_with_sparse_eligibility(
    mask_ratio: float, sequence_length: int
) -> None:
    input_ids = torch.arange(3 * sequence_length, device="cuda").view(3, sequence_length)
    positions = torch.arange(sequence_length, device="cuda")
    eligible = (positions.unsqueeze(0) % torch.tensor([[3], [5], [7]], device="cuda")) != 0
    output, selected = meap_mask_inputs(
        input_ids,
        999_999,
        mask_ratio=mask_ratio,
        eligible_mask=eligible,
        seed=5,
        exclude_last=False,
        return_mask=True,
    )
    assert torch.equal(selected.sum(1), _expected_count(eligible.sum(1), mask_ratio))
    assert not selected[~eligible].any()
    assert torch.equal(output[selected], torch.full_like(output[selected], 999_999))
    assert torch.equal(output[~selected], input_ids[~selected])


@skip_no_cuda
def test_meap_accepts_padding_mask_without_materializing_inverse() -> None:
    input_ids = torch.arange(4 * 64, device="cuda").view(4, 64)
    padding_mask = torch.arange(64, device="cuda").unsqueeze(0) >= torch.tensor(
        [[64], [57], [41], [19]], device="cuda"
    )
    direct = meap_mask_inputs(
        input_ids,
        999,
        padding_mask=padding_mask,
        seed=77,
        return_mask=True,
    )
    inverse = meap_mask_inputs(
        input_ids,
        999,
        eligible_mask=~padding_mask,
        seed=77,
        return_mask=True,
    )
    assert torch.equal(direct[0], inverse[0])
    assert torch.equal(direct[1], inverse[1])


@skip_no_cuda
def test_meap_triton_seed_is_reproducible() -> None:
    input_ids = torch.arange(8 * 64, device="cuda").view(8, 64)
    first = meap_mask_inputs(input_ids, 999, seed=81, return_mask=True)
    second = meap_mask_inputs(input_ids, 999, seed=81, return_mask=True)
    different = meap_mask_inputs(input_ids, 999, seed=82, return_mask=True)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[1], different[1])


@skip_no_cuda
@pytest.mark.parametrize("seed", [0, 81, 0xFFFFFFFE])
def test_meap_device_scalar_seed_matches_python_integer(seed: int) -> None:
    input_ids = torch.arange(8 * 64, device="cuda").view(8, 64)
    tensor_seed = torch.tensor(seed, device="cuda", dtype=torch.int64)
    from_integer = meap_mask_inputs(input_ids, 999, seed=seed, return_mask=True)
    from_tensor = meap_mask_inputs(
        input_ids, 999, seed=tensor_seed, return_mask=True
    )
    assert torch.equal(from_integer[0], from_tensor[0])
    assert torch.equal(from_integer[1], from_tensor[1])


@skip_no_cuda
def test_meap_device_scalar_seed_folds_high_bits() -> None:
    input_ids = torch.arange(64 * 64, device="cuda").view(64, 64)
    low_seed = 81
    packed_seed = low_seed + (1 << 32)
    folded_seed = _fold_seed_to_uint32(packed_seed)
    assert folded_seed != low_seed

    _, from_low = meap_mask_inputs(
        input_ids, 999_999, seed=low_seed, return_mask=True
    )
    _, from_packed = meap_mask_inputs(
        input_ids,
        999_999,
        seed=torch.tensor(packed_seed, device="cuda", dtype=torch.int64),
        return_mask=True,
    )
    _, from_folded = meap_mask_inputs(
        input_ids, 999_999, seed=folded_seed, return_mask=True
    )
    assert not torch.equal(from_packed, from_low)
    assert torch.equal(from_packed, from_folded)


@skip_no_cuda
@pytest.mark.parametrize("seed", [1, 2027, 0xFFFFFFFE])
def test_meap_permutation_has_no_obvious_position_or_adjacency_bias(seed: int) -> None:
    batch_size, sequence_length = 4096, 64
    input_ids = torch.arange(sequence_length, device="cuda").expand(batch_size, -1)
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    eligible[:, -1] = False
    _, selected = meap_mask_inputs(
        input_ids,
        999,
        mask_ratio=0.15,
        eligible_mask=eligible,
        seed=seed,
        exclude_last=False,
        return_mask=True,
    )

    eligible_count = sequence_length - 1
    selected_count = int(_expected_count(torch.tensor(eligible_count), 0.15))
    expected_frequency = batch_size * selected_count / eligible_count
    frequencies = selected[:, :-1].sum(dim=0).float()
    chi_square = ((frequencies - expected_frequency).square() / expected_frequency).sum()
    assert float(chi_square) < 2.5 * (eligible_count - 1)

    adjacent_pairs = (selected[:, :-2] & selected[:, 1:-1]).sum()
    expected_adjacent_pairs = (
        batch_size * (eligible_count - 1) * selected_count * (selected_count - 1)
        / (eligible_count * (eligible_count - 1))
    )
    relative_error = abs(float(adjacent_pairs) - expected_adjacent_pairs) / expected_adjacent_pairs
    assert relative_error < 0.08


@skip_no_cuda
def test_meap_combines_with_cce_mile_and_mu_loss() -> None:
    torch.manual_seed(91)
    vocab_size, width = 257, 32
    token_weight = torch.randn(vocab_size, width, device="cuda", dtype=torch.bfloat16)
    token_weight.requires_grad_(True)
    classifier = torch.randn(vocab_size, width, device="cuda", dtype=torch.bfloat16)
    classifier.requires_grad_(True)
    clean_ids = torch.randint(0, vocab_size - 1, (4, 64), device="cuda")
    padding_mask = torch.zeros_like(clean_ids, dtype=torch.bool)
    padding_mask[1, -7:] = True
    clean_ids = clean_ids.masked_fill(padding_mask, 0)
    targets = clean_ids.masked_fill(padding_mask, -100)

    masked_ids, selected = meap_mask_inputs(
        clean_ids,
        vocab_size - 1,
        enabled=True,
        mask_ratio=0.15,
        padding_mask=padding_mask,
        seed=2026,
        return_mask=True,
    )
    hidden = torch.nn.functional.embedding(masked_ids, token_weight)

    loss, metrics = linear_cross_entropy(
        hidden,
        classifier,
        targets,
        shift=1,
        impl="cce_kahan_full_c",
        mile_enabled=True,
        mile_gamma=1.0,
        mu_loss_enabled=True,
        mu_loss_lambda=1e-4,
        return_loss_metrics=True,
    )
    torch.testing.assert_close(
        loss.detach(),
        metrics["ntp_ce_unweighted"]
        + metrics["mile_reweighting_delta"]
        + metrics["mu_loss"],
        rtol=2e-5,
        atol=2e-5,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert selected.any()
    assert token_weight.grad is not None and torch.isfinite(token_weight.grad).all()
    assert classifier.grad is not None and torch.isfinite(classifier.grad).all()
