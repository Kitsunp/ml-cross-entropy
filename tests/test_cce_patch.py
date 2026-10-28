from __future__ import annotations

import math

import pytest
import torch
import triton
import triton.language as tl

from cut_cross_entropy import PatchTrainingSchedule, linear_cross_entropy
from cut_cross_entropy.cce_lse_forward import _merge_topk, _topk_key_dtype, _topk_keys
from cut_cross_entropy.cce_patch import patch_loss_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@triton.jit
def _topk_leaf_contract_kernel(
    X, Out, WIDTH: tl.constexpr, VALID: tl.constexpr, OFFSET: tl.constexpr, NATIVE: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.arange(0, WIDTH)
    values = tl.load(X + row * WIDTH + column)[None, :]
    winners = _topk_keys(
        values if NATIVE else values.to(tl.float32),
        (column + OFFSET)[None, :],
        (column < VALID)[None, :],
        values.dtype,
        Out.dtype.element_ty == tl.uint32,
    )
    for rank in tl.static_range(5):
        tl.store(Out + row * 5 + rank, tl.sum(winners[rank], 0))


@triton.jit
def _topk_merge_contract_kernel(A, B, Out):
    row = tl.program_id(0)
    left = ()
    right = ()
    for rank in tl.static_range(5):
        left += (tl.load(A + row * 5 + rank),)
        right += (tl.load(B + row * 5 + rank),)
    merged = _merge_topk(left, right)
    for rank in tl.static_range(5):
        tl.store(Out + row * 5 + rank, merged[rank])


@pytest.mark.parametrize(
    "dtype,global_packed",
    [
        (torch.float16, False),
        (torch.float16, True),
        (torch.bfloat16, False),
        (torch.bfloat16, True),
        (torch.float32, False),
    ],
)
@pytest.mark.parametrize("valid_columns", [0, 3, 251])
@pytest.mark.parametrize("native", [False, True], ids=["widened", "native"])
def test_topk_leaf_encoding_preserves_exact_bits_and_ids(
    dtype, global_packed, valid_columns, native
):
    import struct

    torch.manual_seed(1927)
    x = torch.randn(9, 256, dtype=torch.float32).to(dtype)
    tiny = torch.finfo(dtype).tiny
    x[0, :8] = torch.tensor([0.0, -0.0, tiny, -tiny, tiny / 2, -tiny / 2, 1, -1], dtype=dtype)
    x[1].zero_()
    x[2] = -torch.arange(256).to(dtype)
    x[3, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=dtype)
    # Make the subnormals winners, rather than hiding them under random
    # order-one scores. Cover positive and all-negative subnormal tiles.
    x[4] = tiny / 256
    x[4, :8] = torch.arange(1, 9, dtype=torch.float32).to(dtype) * (tiny / 16)
    x[5] = -x[4]
    x[6].fill_(-tiny)
    x[6, :8] = torch.tensor([0.0, -0.0, tiny, -tiny, tiny / 2, -tiny / 2, 0, 0], dtype=dtype)
    # Padded columns may be much larger than every valid score.
    x[:, 251:] = torch.finfo(dtype).max
    key_dtype = torch.uint32 if global_packed else torch.uint64
    offset = 64100 if global_packed else 65537
    output = torch.empty((9, 5), device="cuda", dtype=key_dtype)
    _topk_leaf_contract_kernel[(9,)](x.cuda(), output, 256, valid_columns, offset, native)
    expected = []
    for row in x.float().tolist():
        keys = []
        for column, value in enumerate(row[:valid_columns]):
            if not math.isfinite(value):
                keys.append(0xFFFFFFFF if global_packed else 0xFFFFFFFFFFFFFFFF)
                continue
            if global_packed:
                # Independent host bit representation, not kernel decoding.
                bits16 = (
                    struct.unpack("H", struct.pack("e", value))[0]
                    if dtype == torch.float16
                    else struct.unpack("I", struct.pack("f", value))[0] >> 16
                )
                bits16 = 0 if value == 0 else bits16
                ordered16 = (~bits16 & 0xFFFF) if bits16 & 0x8000 else bits16 ^ 0x8000
                keys.append((ordered16 << 16) | (0xFFFF - (offset + column)))
                continue
            bits = 0 if value == 0 else struct.unpack("I", struct.pack("f", value))[0]
            ordered = (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits ^ 0x80000000
            keys.append((ordered << 32) | (0xFFFFFFFF - (offset + column)))
        expected.append((sorted(keys, reverse=True) + [0] * 5)[:5])
    torch.testing.assert_close(
        output.cpu(), torch.tensor(expected, dtype=key_dtype), rtol=0, atol=0
    )


@pytest.mark.parametrize("vocab", [65536, 65537])
@pytest.mark.parametrize("reduction", ["lock", "split"])
def test_topk_global_key_width_boundary(vocab, reduction, monkeypatch):
    from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel

    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    e = torch.zeros((7, 32), device="cuda", dtype=torch.bfloat16)
    e[:, 0] = 1
    c = torch.zeros((vocab, 32), device="cuda", dtype=torch.bfloat16)
    c[-1, 0] = 1
    labels = torch.tensor([vocab - 1, 0, 1, 2, 3, 4, -100], device="cuda")
    result = cce_lse_forward_kernel(e, c, targets=labels, return_topk=True)
    torch.testing.assert_close(
        result.topk_counts, torch.tensor([1.0, 3.0, 5.0, 6.0, 0.0], device="cuda")
    )
    assert _topk_key_dtype(e, c, None, None) == (torch.uint32 if vocab <= 65536 else torch.uint64)
    assert _topk_key_dtype(e, c, torch.zeros(vocab, device="cuda"), None) == torch.uint64
    assert _topk_key_dtype(e, c, None, 7.0) == torch.uint64


@pytest.mark.parametrize("dtype,bits", [(torch.uint32, 32), (torch.uint64, 64)], ids=["u32", "u64"])
def test_topk_merge_network_matches_multiset_order_statistics(dtype, bits):
    import random

    generator = random.Random(4291)
    pairs = [([0] * 5, [0] * 5), ([2**bits - 1] * 5, [0] * 5)]
    pairs += [
        (
            sorted([generator.randrange(8) for _ in range(5)], reverse=True),
            sorted([generator.getrandbits(bits) for _ in range(5)], reverse=True),
        )
        for _ in range(127)
    ]
    # Also cover interleaving, equal keys, and nonfinite sentinels.
    pairs += [([9, 7, 5, 3, 1], [10, 8, 6, 4, 2]), ([7] * 5, [7] * 5)]
    a = torch.tensor([pair[0] for pair in pairs], dtype=dtype, device="cuda")
    b = torch.tensor([pair[1] for pair in pairs], dtype=dtype, device="cuda")
    output = torch.empty_like(a)
    _topk_merge_contract_kernel[(len(pairs),)](a, b, output)
    expected = torch.tensor(
        [sorted(left + right, reverse=True)[:5] for left, right in pairs], dtype=dtype
    )
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("reduction", ["lock", "split"])
@pytest.mark.parametrize("softcap", [None, 7.0])
def test_topk_fp32_bias_is_not_requantized(reduction, softcap, monkeypatch):
    from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel

    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    e = torch.zeros((9, 32), device="cuda", dtype=torch.bfloat16)
    c = torch.zeros((259, 32), device="cuda", dtype=torch.bfloat16)
    # All scores round to the same BF16 number, but the actual FP32-bias
    # projection has distinct winners. A low-precision assumption fails.
    bias = 1 + torch.arange(259, device="cuda", dtype=torch.float32) * 1e-6
    labels = torch.tensor([258, 257, 256, 255, 254, 253, 0, -100, 259], device="cuda")
    result = cce_lse_forward_kernel(e, c, bias, targets=labels, softcap=softcap, return_topk=True)
    torch.testing.assert_close(
        result.topk_counts, torch.tensor([1.0, 3.0, 5.0, 7.0, 0.0], device="cuda")
    )


@pytest.mark.parametrize("reduction", ["lock", "split"])
@pytest.mark.parametrize("mile_statistic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_full_token_topk_uses_actual_lse_logits(reduction, mile_statistic, dtype, monkeypatch):
    from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel

    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    torch.manual_seed(443)
    e = torch.randint(-3, 4, (13, 32), device="cuda").to(dtype)
    c = torch.randint(-3, 4, (259, 32), device="cuda").to(dtype)
    bias = torch.zeros(259, device="cuda", dtype=dtype)
    # Deliberate ties, a tail-vocabulary winner, ignored and invalid targets.
    e[0].zero_()
    e[1].zero_()
    e[1, 0] = 4
    c[-1, 0] = 128
    targets = torch.tensor([3, 258, -100, 0, 258, 1, 259, 4, 5, 6, 7, 8, 9], device="cuda")
    actual = cce_lse_forward_kernel(
        e, c, bias, targets=targets, return_mean_logit=mile_statistic, return_topk=True
    )
    ordinary = cce_lse_forward_kernel(e, c, bias, targets=targets, return_mean_logit=mile_statistic)
    torch.testing.assert_close(actual.lse, ordinary.lse)
    if mile_statistic:
        torch.testing.assert_close(actual.mean_logit, ordinary.mean_logit)
    logits = (e.float() @ c.float().T).to(dtype) + bias
    top = logits.float().argsort(dim=-1, descending=True, stable=True)[:, :5]
    valid = (targets >= 0) & (targets < c.shape[0])
    expected = torch.tensor(
        [((top[:, :k] == targets[:, None]).any(-1) & valid).sum().item() for k in (1, 3, 5)]
        + [valid.sum().item(), 0.0],
        device="cuda",
    )
    torch.testing.assert_close(actual.topk_counts, expected)


@pytest.mark.parametrize("reduction", ["lock", "split"])
def test_full_token_topk_compact_and_fixed_k1_targets(reduction, monkeypatch):
    from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel

    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    e = torch.zeros((8, 32), device="cuda", dtype=torch.bfloat16)
    c = torch.zeros((17, 32), device="cuda", dtype=torch.bfloat16)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, -100, 7], device="cuda")
    valids = torch.tensor([0, 1, 2, 3, 4, 5], device="cuda")
    actual = cce_lse_forward_kernel(e, c, targets=labels, valids=valids, return_topk=True)
    torch.testing.assert_close(
        actual.topk_counts, torch.tensor([1.0, 3.0, 5.0, 6.0, 0.0], device="cuda")
    )
    slots = torch.full((8, 4), -1, device="cuda", dtype=torch.long)
    slots[:, 0] = labels
    fixed = cce_lse_forward_kernel(e, c, targets=slots, return_topk=True)
    torch.testing.assert_close(
        fixed.topk_counts, torch.tensor([1.0, 3.0, 5.0, 7.0, 0.0], device="cuda")
    )


@pytest.mark.parametrize("mile", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_full_token_topk_public_api_preserves_loss_and_gradients(mile, compiled):
    from cut_cross_entropy import linear_cross_entropy

    torch.manual_seed(814)
    hidden = torch.randint(-2, 3, (2, 9, 32), device="cuda").to(torch.bfloat16).requires_grad_()
    weight = torch.randint(-2, 3, (259, 32), device="cuda").to(torch.bfloat16).requires_grad_()
    labels = torch.tensor(
        [[1, 2, -100, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, -100, 15, 16]], device="cuda"
    )

    def objective(e, c, y):
        return linear_cross_entropy(
            e,
            c,
            y,
            shift=1,
            impl="cce_exact",
            return_loss_metrics=True,
            return_topk=True,
            mile_enabled=mile,
            mu_loss_enabled=True,
        )

    run = torch.compile(objective, fullgraph=True, mode="max-autotune") if compiled else objective
    loss, metrics = run(hidden, weight, labels)
    counts = metrics["topk_counts"].detach().clone()
    loss.backward()
    gradients = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    reference, _ = linear_cross_entropy(
        hidden,
        weight,
        labels,
        shift=1,
        impl="cce_exact",
        return_loss_metrics=True,
        mile_enabled=mile,
        mu_loss_enabled=True,
    )
    reference.backward()
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(hidden.grad, gradients[0], atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, gradients[1], atol=0, rtol=0)
    logits = (hidden[:, :-1].detach().float() @ weight.detach().float().T).to(torch.bfloat16)
    top = logits.float().argsort(dim=-1, descending=True, stable=True)[..., :5]
    valid = labels[:, 1:] != -100
    expected = torch.tensor(
        [((top[..., :k] == labels[:, 1:, None]).any(-1) & valid).sum().item() for k in (1, 3, 5)]
        + [valid.sum().item(), 0.0],
        device="cuda",
    )
    torch.testing.assert_close(counts, expected)


@pytest.mark.parametrize("reduction", ["lock", "split"])
@pytest.mark.parametrize("rows,vocab", [(0, 3), (257, 3), (257, 259)])
def test_full_token_topk_empty_small_vocab_and_full_row_tiles(reduction, rows, vocab, monkeypatch):
    from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel

    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    e = torch.zeros((rows, 32), device="cuda", dtype=torch.bfloat16)
    c = torch.zeros((vocab, 32), device="cuda", dtype=torch.bfloat16)
    targets = torch.arange(rows, device="cuda") % vocab
    targets[::17] = -100
    actual = cce_lse_forward_kernel(e, c, targets=targets, return_topk=True)
    expected = torch.tensor(
        [((targets >= 0) & (targets < k)).sum().item() for k in (1, 3, 5)]
        + [(targets >= 0).sum().item(), 0.0],
        device="cuda",
    )
    torch.testing.assert_close(actual.topk_counts, expected)


@pytest.mark.parametrize("mile", [False, True])
def test_full_tile_topk_preserves_bf16_objective_and_vjp(mile, monkeypatch):
    """Exercise the large eight-warp epilogue, not just tiny adaptive tiles."""
    from cut_cross_entropy import linear_cross_entropy

    monkeypatch.setenv("CCE_FORWARD_REDUCTION", "lock")
    torch.manual_seed(917)
    hidden = (
        (torch.randint(-1, 2, (257, 128), device="cuda") / 4).to(torch.bfloat16).requires_grad_()
    )
    weight = (
        (torch.randint(-1, 2, (259, 128), device="cuda") / 4).to(torch.bfloat16).requires_grad_()
    )
    labels = torch.arange(257, device="cuda") % 259
    labels[::17] = -100
    options = dict(impl="cce_exact", return_loss_metrics=True, mile_enabled=mile)
    measured, metrics = linear_cross_entropy(hidden, weight, labels, return_topk=True, **options)
    measured.backward()
    grads = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    reference, _ = linear_cross_entropy(hidden, weight, labels, **options)
    reference.backward()
    torch.testing.assert_close(measured, reference, atol=2e-6, rtol=2e-6)
    # Reduction scheduling may change FP32 roundoff before the final BF16
    # gradient cast. Permit one BF16 relative ULP, not an arbitrary percentage.
    torch.testing.assert_close(hidden.grad, grads[0], atol=1e-5, rtol=1 / 128)
    torch.testing.assert_close(weight.grad, grads[1], atol=1e-5, rtol=1 / 128)
    logits = (hidden.detach().float() @ weight.detach().float().T).to(torch.bfloat16)
    top = logits.float().argsort(dim=-1, descending=True, stable=True)[:, :5]
    valid = labels != -100
    expected = torch.tensor(
        [((top[:, :k] == labels[:, None]).any(-1) & valid).sum().item() for k in (1, 3, 5)]
        + [valid.sum().item(), 0.0],
        device="cuda",
    )
    torch.testing.assert_close(metrics["topk_counts"], expected)


@pytest.mark.parametrize(
    ("rows", "patch_size", "mile_gamma", "return_unweighted"),
    [
        (1, 1, None, False),
        (3, 8, 0.0, True),
        (30, 4, 1.0, True),
        (65, 3, 0.5, True),
        (129, 4, 2.0, True),
    ],
)
def test_patch_loss_fused_small_and_fallback_match_reference(
    rows: int,
    patch_size: int,
    mile_gamma: float | None,
    return_unweighted: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(700 + rows + patch_size)
    vocab = 257
    lse = torch.rand(rows, device="cuda", generator=generator, dtype=torch.float32) + 2.0
    mean_logit = torch.randn(rows, device="cuda", generator=generator, dtype=torch.float32)
    neg_correct_logit = -torch.rand(
        rows,
        patch_size,
        device="cuda",
        generator=generator,
        dtype=torch.float32,
    )
    targets = torch.randint(
        vocab,
        (rows, patch_size),
        device="cuda",
        generator=generator,
    )
    targets[0, 0] = -1
    if rows > 1:
        targets[1, -1] = vocab

    actual = patch_loss_forward(
        lse,
        mean_logit if mile_gamma is not None else None,
        neg_correct_logit,
        targets,
        vocab,
        mile_gamma,
        return_unweighted,
    )

    valid = (targets >= 0) & (targets < vocab)
    valid_f32 = valid.float()
    counts = valid_f32.sum(dim=1)
    row_nll = ((lse[:, None] + neg_correct_logit) * valid_f32).sum(dim=1)
    if mile_gamma is None:
        base_weight = torch.ones_like(lse)
    else:
        base_weight = (1.0 + torch.clamp_min(lse - mean_logit, 0.0)).pow(mile_gamma)
    denominator = (base_weight * counts).sum().clamp_min(1.0)
    expected_objective = (base_weight * row_nll).sum() / denominator
    expected_unweighted = row_nll.sum() / counts.sum().clamp_min(1.0)
    expected_target_weight = base_weight * (rows / denominator)
    expected_dense_weight = expected_target_weight * counts

    torch.testing.assert_close(actual[0], expected_objective, rtol=1e-5, atol=1e-6)
    if return_unweighted:
        torch.testing.assert_close(actual[1], expected_unweighted, rtol=1e-5, atol=1e-6)
    else:
        assert actual[1].data_ptr() == actual[0].data_ptr()
    torch.testing.assert_close(actual[2], expected_dense_weight, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[3], expected_target_weight, rtol=1e-5, atol=1e-6)


def _dense_patch_loss(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    mile_gamma: float | None,
    mu_loss_lambda: float | None,
    mile_group_fraction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = e.reshape(-1, e.size(-1)).float() @ c.float().T
    if bias is not None:
        logits = logits + bias.float()
    flat_targets = targets.reshape(logits.size(0), targets.size(-1))
    valid = (flat_targets >= 0) & (flat_targets < c.size(0))
    safe_targets = torch.where(valid, flat_targets, torch.zeros_like(flat_targets))
    log_probs = logits.log_softmax(dim=-1)
    nll = -log_probs.gather(1, safe_targets)
    valid_f32 = valid.float()
    counts = valid_f32.sum(dim=1)

    if mile_gamma is None:
        base_weight = torch.ones(logits.size(0), device=logits.device)
    else:
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)
        base_weight = (1.0 + entropy).pow(mile_gamma).detach()

    if mile_group_fraction is None:
        denominator = (base_weight * counts).sum().clamp_min(1.0)
        objective = (nll * valid_f32 * base_weight[:, None]).sum() / denominator
    else:
        fraction = mile_group_fraction.flatten()
        weight = torch.zeros_like(base_weight)
        for population in (1.0 - fraction, fraction):
            group_count = (counts * population).sum()
            group_normalizer = (counts * population * base_weight).sum()
            weight += (
                population
                * base_weight
                * group_count
                / torch.where(group_normalizer > 0, group_normalizer, 1.0)
            )
        weight = weight.detach()
        objective = (nll * valid_f32 * weight[:, None]).sum() / counts.sum().clamp_min(1.0)
    unweighted = (nll * valid_f32).sum() / counts.sum().clamp_min(1.0)
    mu_loss = objective.new_zeros(())
    if mu_loss_lambda is not None:
        mu_loss = mu_loss_lambda * c.float().mean(dim=0).square().sum()
    metrics = {
        "ntp_ce_unweighted": unweighted.detach(),
        "mile_reweighting_delta": (objective - unweighted).detach(),
        "mu_loss": mu_loss.detach(),
    }
    return objective + mu_loss, metrics


def _assert_relative_gradient(actual: torch.Tensor, expected: torch.Tensor) -> None:
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(
        1e-12
    )
    assert float(relative_l2) < 4e-2, f"Relative L2 gradient error: {float(relative_l2)}"


@pytest.mark.parametrize(
    ("mile_gamma", "mu_loss_lambda"),
    [(None, None), (1.0, None), (None, 1e-4), (1.0, 1e-4)],
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_patch_cce_matches_dense_forward_backward(
    mile_gamma: float | None,
    mu_loss_lambda: float | None,
    has_bias: bool,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20_260_813)
    rows, patch_size, vocab, dim = 19, 4, 257, 64
    e_data = torch.randn(rows, dim, device="cuda", dtype=dtype, generator=generator) / 8
    c_data = torch.randn(vocab, dim, device="cuda", dtype=dtype, generator=generator)
    bias_data = (
        torch.randn(vocab, device="cuda", dtype=dtype, generator=generator) / 20
        if has_bias
        else None
    )
    targets = torch.randint(
        vocab,
        (rows, patch_size),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )
    targets[1, 3] = -100
    targets[4, 1:] = -100
    targets[7, 0] = -1
    targets[8, 1] = vocab + 3
    targets[9, 2] = targets[9, 0]

    e_ref = e_data.clone().requires_grad_(True)
    c_ref = c_data.clone().requires_grad_(True)
    bias_ref = bias_data.clone().requires_grad_(True) if bias_data is not None else None
    expected, expected_metrics = _dense_patch_loss(
        e_ref,
        c_ref,
        targets,
        bias_ref,
        mile_gamma=mile_gamma,
        mu_loss_lambda=mu_loss_lambda,
    )
    expected.backward()

    e_test = e_data.clone().requires_grad_(True)
    c_test = c_data.clone().requires_grad_(True)
    bias_test = bias_data.clone().requires_grad_(True) if bias_data is not None else None
    actual, actual_metrics = linear_cross_entropy(
        e_test,
        c_test,
        targets,
        bias=bias_test,
        impl="cce_exact",
        filter_eps=None,
        return_loss_metrics=True,
        mile_enabled=mile_gamma is not None,
        mile_gamma=1.0 if mile_gamma is None else mile_gamma,
        mu_loss_enabled=mu_loss_lambda is not None,
        mu_loss_lambda=1e-4 if mu_loss_lambda is None else mu_loss_lambda,
        patch_training_enabled=True,
    )
    actual.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    for name in expected_metrics:
        torch.testing.assert_close(
            actual_metrics[name], expected_metrics[name], rtol=3e-2, atol=3e-2
        )
    assert e_test.grad is not None and e_ref.grad is not None
    assert c_test.grad is not None and c_ref.grad is not None
    _assert_relative_gradient(e_test.grad, e_ref.grad)
    _assert_relative_gradient(c_test.grad, c_ref.grad)
    if bias_test is not None:
        assert bias_test.grad is not None and bias_ref is not None and bias_ref.grad is not None
        _assert_relative_gradient(bias_test.grad, bias_ref.grad)


def test_token_phase_with_padded_slots_matches_single_target_cce() -> None:
    generator = torch.Generator(device="cuda").manual_seed(91)
    rows, patch_size, vocab, dim = 31, 4, 263, 64
    e_data = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16, generator=generator) / 8
    c_data = torch.randn(vocab, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    targets = torch.randint(vocab, (rows,), device="cuda", generator=generator)
    patch_targets = torch.full((rows, patch_size), -100, device="cuda", dtype=torch.long)
    patch_targets[:, 0] = targets

    base_e = e_data.clone().requires_grad_(True)
    base_c = c_data.clone().requires_grad_(True)
    base = linear_cross_entropy(
        base_e,
        base_c,
        targets,
        impl="cce_exact",
        filter_eps=None,
        mile_enabled=True,
        mu_loss_enabled=True,
    )
    base.backward()

    patch_e = e_data.clone().requires_grad_(True)
    patch_c = c_data.clone().requires_grad_(True)
    patch = linear_cross_entropy(
        patch_e,
        patch_c,
        patch_targets,
        impl="cce_exact",
        filter_eps=None,
        mile_enabled=True,
        mu_loss_enabled=True,
        patch_training_enabled=True,
    )
    patch.backward()

    torch.testing.assert_close(patch, base, rtol=3e-2, atol=3e-2)
    assert patch_e.grad is not None and base_e.grad is not None
    assert patch_c.grad is not None and base_c.grad is not None
    _assert_relative_gradient(patch_e.grad, base_e.grad)
    _assert_relative_gradient(patch_c.grad, base_c.grad)


@pytest.mark.parametrize("target_slots", [1, 4])
@pytest.mark.parametrize("impl", ["cce_exact", "cce_kahan_full_c"])
def test_patch_k1_cotangent_cancellation_matches_token_before_bf16_cast(target_slots, impl):
    """The confident-target cancellation must occur BEFORE dot quantization.

    logits=[6,0] makes p_target close to one. Separately rounding p and then
    subtracting its target can amplify BF16 error in the small cotangent.
    This kernel-only regression contains no private model/data/provenance.
    """
    e = torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16)
    e[0, 0] = 1
    c = torch.zeros(2, 64, device="cuda", dtype=torch.bfloat16)
    c[0, 0] = 6
    target = torch.zeros(1, device="cuda", dtype=torch.long)
    packed = target.new_full((1, target_slots), -100)
    packed[:, 0] = target
    gradients = []
    for patch in (False, True):
        hi, ci = e.clone().requires_grad_(), c.clone().requires_grad_()
        options = dict(
            impl=impl,
            mile_enabled=True,
            mile_gamma=1.0,
            mu_loss_enabled=True,
        )
        if patch:
            options.update(patch_training_enabled=True, mile_group_fraction=e.new_zeros(1).float())
        loss = linear_cross_entropy(hi, ci, packed if patch else target, **options)
        gradients.append(torch.autograd.grad(loss, hi)[0])
    torch.testing.assert_close(gradients[1], gradients[0], atol=0, rtol=0)
    expected = 6 * (torch.tensor([6.0, 0.0], device="cuda", dtype=torch.float64).softmax(0)[0] - 1)
    torch.testing.assert_close(gradients[1][0, 0].double(), expected, atol=0, rtol=0.004)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_patch_width_above_fusion_budget_keeps_sparse_path_correct(dtype, monkeypatch):
    import importlib

    backward = importlib.import_module("cut_cross_entropy.cce_backward")
    sparse = backward.patch_target_backward
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return sparse(*args, **kwargs)

    monkeypatch.setattr(backward, "patch_target_backward", counted)
    generator = torch.Generator(device="cuda").manual_seed(2021)
    e = (torch.randn(19, 64, device="cuda", dtype=dtype, generator=generator) / 8).requires_grad_()
    c = torch.randn(257, 64, device="cuda", dtype=dtype, generator=generator).requires_grad_()
    targets = torch.randint(257, (19, 5), device="cuda", generator=generator)
    targets[0] = -100
    targets[1, 1] = targets[1, 0]
    targets[2, 0] = 258
    fraction = (torch.arange(19, device="cuda") % 5).float() / 4
    er, cr = [value.detach().clone().requires_grad_() for value in (e, c)]
    expected, _ = _dense_patch_loss(
        er, cr, targets, None, mile_gamma=1.0, mu_loss_lambda=1e-4, mile_group_fraction=fraction
    )
    actual = linear_cross_entropy(
        e,
        c,
        targets,
        impl="cce_exact",
        patch_training_enabled=True,
        mile_enabled=True,
        mile_group_fraction=fraction,
        mu_loss_enabled=True,
    )
    actual.backward()
    expected.backward()
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    _assert_relative_gradient(e.grad, er.grad)
    _assert_relative_gradient(c.grad, cr.grad)


@pytest.mark.parametrize("impl", ["cce_exact", "cce_kahan_full_c"])
def test_patch_repeated_target_does_not_add_sequential_subtraction_residue(impl):
    """Three equal targets must not leave FP32 residue from three subtractions.

    Compare the same CCE arithmetic, not an exact-softmax claim at large
    logits: even the ordinary path has finite-precision LSE error here.
    """
    hidden = torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16)
    hidden[0, 0] = 1
    classifier = torch.zeros(2, 64, device="cuda", dtype=torch.bfloat16)
    classifier[0, 0] = 16
    packed = torch.tensor([[0, 0, 0, -100]], device="cuda")
    gradients = []
    for patch in (False, True):
        e = hidden.clone().requires_grad_()
        c = classifier.clone().requires_grad_()
        loss = linear_cross_entropy(
            e,
            c,
            packed if patch else packed[:, 0],
            impl=impl,
            patch_training_enabled=patch,
            mile_enabled=True,
            mile_group_fraction=torch.zeros(1, device="cuda") if patch else None,
            mu_loss_enabled=True,
        )
        gradients.append(torch.autograd.grad(loss, e)[0])
    torch.testing.assert_close(gradients[1], gradients[0], atol=0, rtol=0)


@pytest.mark.parametrize("dim", [128, 512])
def test_patch_large_bf16_tile_matches_dense_after_row_duplicate_folding(dim):
    generator = torch.Generator(device="cuda").manual_seed(2023)
    e = torch.randn(129, dim, device="cuda", dtype=torch.bfloat16, generator=generator) / 8
    c = torch.randn(257, dim, device="cuda", dtype=torch.bfloat16, generator=generator) / 4
    e, c = e.requires_grad_(), c.requires_grad_()
    targets = torch.randint(257, (129, 4), device="cuda", generator=generator)
    targets[0] = -100
    targets[1:9, 1:3] = targets[1:9, :1]
    targets[9, 3] = 258
    fraction = (torch.arange(129, device="cuda") % 5).float() / 4
    er, cr = [value.detach().clone().requires_grad_() for value in (e, c)]
    expected, _ = _dense_patch_loss(
        er,
        cr,
        targets,
        None,
        mile_gamma=1.0,
        mu_loss_lambda=1e-4,
        mile_group_fraction=fraction,
    )
    actual = linear_cross_entropy(
        e,
        c,
        targets,
        impl="cce_exact",
        patch_training_enabled=True,
        mile_enabled=True,
        mile_group_fraction=fraction,
        mu_loss_enabled=True,
    )
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    _assert_relative_gradient(e.grad, er.grad)
    _assert_relative_gradient(c.grad, cr.grad)


def test_patch_target_logit_cannot_exceed_single_class_lse() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7)
    rows, patch_size, dim = 17, 4, 64
    e = torch.randn(
        rows,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    c = torch.randn(
        1,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    targets = torch.zeros((rows, patch_size), device="cuda", dtype=torch.long)

    loss = linear_cross_entropy(
        e,
        c,
        targets,
        impl="cce_exact",
        filter_eps=None,
        patch_training_enabled=True,
    )

    assert float(loss.detach()) >= 0.0
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("patch_size", [1, 8])
@pytest.mark.parametrize("forward_reduction", ["lock", "split"])
def test_patch_extreme_sizes_with_all_invalid_targets_are_zero(
    monkeypatch: pytest.MonkeyPatch,
    patch_size: int,
    forward_reduction: str,
) -> None:
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", forward_reduction)
    generator = torch.Generator(device="cuda").manual_seed(117 + patch_size)
    rows, vocab, dim = 3, 257, 33
    e = torch.randn(
        rows,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    c = torch.randn(
        vocab,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    targets = torch.full((rows, patch_size), -100, device="cuda", dtype=torch.long)
    if patch_size > 1:
        targets[0, 0] = -1
        targets[1, 1] = vocab

    loss = linear_cross_entropy(
        e,
        c,
        targets,
        impl="cce_exact",
        filter_eps=None,
        patch_training_enabled=True,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0.0, atol=0.0)
    assert e.grad is not None and c.grad is not None
    torch.testing.assert_close(e.grad, torch.zeros_like(e.grad), rtol=0.0, atol=0.0)
    torch.testing.assert_close(c.grad, torch.zeros_like(c.grad), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("forward_reduction", ["lock", "split"])
def test_patch_out_of_range_target_is_ignored_before_padded_vocab_load(
    monkeypatch: pytest.MonkeyPatch,
    forward_reduction: str,
) -> None:
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", forward_reduction)
    generator = torch.Generator(device="cuda").manual_seed(83)
    rows, patch_size, vocab, dim = 17, 4, 257, 64
    e = torch.randn(
        rows,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    c = torch.randn(
        vocab,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    targets = torch.randint(
        vocab,
        (rows, patch_size),
        device="cuda",
        generator=generator,
    )
    targets[3, 2] = vocab + 3

    actual = linear_cross_entropy(
        e,
        c,
        targets,
        impl="cce_exact",
        filter_eps=None,
        patch_training_enabled=True,
    )
    expected, _metrics = _dense_patch_loss(
        e,
        c,
        targets,
        None,
        mile_gamma=None,
        mu_loss_lambda=None,
    )

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_patch_training_compiles_five_steps_without_phase_recompile() -> None:
    generator = torch.Generator(device="cuda").manual_seed(44)
    batch, sequence, patch_size, vocab, dim = 2, 8, 4, 257, 64
    base_e = torch.randn(
        batch,
        sequence,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    base_c = torch.randn(vocab, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    schedule = PatchTrainingSchedule(patch_training_steps=3, patch_size=patch_size)
    raw_patch_targets = torch.randint(
        vocab,
        (batch, (sequence + 1) * patch_size),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )
    shifted_patch_targets = raw_patch_targets[..., patch_size:].unflatten(
        -1, (sequence, patch_size)
    )
    assert not shifted_patch_targets.is_contiguous()
    patch_targets = schedule.prepare_patch_targets(shifted_patch_targets)
    token_ids = torch.randint(
        vocab,
        (batch, sequence),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )
    token_targets = schedule.prepare_token_targets(token_ids)
    assert patch_targets.stride() == token_targets.stride()
    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    def step(e, c, labels):
        return linear_cross_entropy(
            e,
            c,
            labels,
            impl="cce_exact",
            filter_eps=None,
            mile_enabled=True,
            mu_loss_enabled=True,
            patch_training_enabled=True,
        )

    explanation = torch._dynamo.explain(step)(base_e, base_c, patch_targets)
    compiled = torch.compile(step, backend=counting_backend, fullgraph=True)
    for labels in (
        patch_targets,
        patch_targets.roll(1, dims=-1),
        patch_targets.roll(2, dims=-1),
        token_targets,
        token_targets.roll(1, dims=1),
    ):
        e = base_e.clone().requires_grad_(True)
        c = base_c.clone().requires_grad_(True)
        loss = compiled(e, c, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert e.grad is not None and torch.isfinite(e.grad).all()
        assert c.grad is not None and torch.isfinite(c.grad).all()
    torch.cuda.synchronize()

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert compile_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reduction": "sum"}, "reduction='mean'"),
        ({"shift": 1}, "does not support shift"),
        ({"softcap": 20.0}, "does not currently support softcap"),
        ({"return_lse": True}, "does not currently support return_lse"),
    ],
)
def test_patch_training_rejects_unsupported_options(kwargs, message: str) -> None:
    e = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(31, 16, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(31, (4, 2), device="cuda")
    with pytest.raises(ValueError, match=message):
        linear_cross_entropy(e, c, targets, patch_training_enabled=True, **kwargs)


def _fractional_reduction_reference(lse, mean, neg_correct, targets, vocab, gamma, fraction):
    """FP64 oracle over supplied statistics, independent of the Triton decomposition."""
    valid = (targets >= 0) & (targets < vocab)
    counts = valid.sum(-1).double()
    nll = torch.where(valid, lse.double()[:, None] + neg_correct.double(), 0).sum(-1)
    fraction = torch.where(counts > 0, fraction.double(), 0.0)
    entropy = (lse.double() - mean.double()).clamp(0.0, math.log(vocab))
    log_base = torch.where(counts > 0, (1.0 + entropy).log(), 0.0)
    weight = torch.zeros_like(counts)
    for population in (1.0 - fraction, fraction):
        population = torch.where(counts > 0, population, 0.0)
        member = (counts > 0) & (population > 0)
        maximum = torch.where(member, log_base, 0.0).max()
        log_ratio = torch.where(member, log_base - maximum, 0.0)
        component = population * (gamma * log_ratio).exp()
        numerator = (counts * population).sum()
        normalizer = (counts * component).sum()
        weight += component * numerator / torch.where(normalizer > 0, normalizer, 1.0)
    count = counts.sum().clamp_min(1.0)
    target_weight = weight * (lse.numel() / count)
    return (nll * weight).sum() / count, nll.sum() / count, counts * target_weight, target_weight


@pytest.mark.parametrize("rows", [33, 129, 513, 8193])
@pytest.mark.parametrize("gamma", [0.0, 1.0, 10000.0])
def test_fractional_patch_reduction_matches_fp64(rows, gamma):
    generator = torch.Generator(device="cuda").manual_seed(1700 + rows)
    vocab, patch_size = 257, 4
    lse = torch.rand(rows, device="cuda", generator=generator) + 6
    mean = torch.rand(rows, device="cuda", generator=generator) * 6
    neg = -torch.rand((rows, patch_size), device="cuda", generator=generator)
    targets = torch.randint(vocab, (rows, patch_size), device="cuda", generator=generator)
    fraction = torch.randint(5, (rows,), device="cuda", generator=generator).float() / 4
    targets[0] = -100
    lse[0], mean[0], neg[0], fraction[0] = float("nan"), float("nan"), float("nan"), float("nan")
    targets[1, 2] = vocab
    neg[1, 2] = float("nan")
    actual = patch_loss_forward(lse, mean, neg, targets, vocab, gamma, True, fraction)
    expected = _fractional_reduction_reference(lse, mean, neg, targets, vocab, gamma, fraction)
    for value, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(
            value.double(), reference, rtol=2e-3 if gamma > 1 else 2e-5, atol=2e-5
        )
    torch.testing.assert_close(actual[2].sum() / rows, lse.new_ones(()), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("rows", [17, 259])
@pytest.mark.parametrize("population", ["clean", "corrupted", "binary", "tiny", "empty"])
def test_fractional_patch_group_local_scaling_extremes(rows, population):
    vocab = 257
    lse = torch.full((rows,), 6.0, device="cuda")
    mean = torch.full_like(lse, 5.9)
    mean[rows // 2 :] = 1.0
    fraction = torch.zeros_like(lse)
    if population == "corrupted":
        fraction.fill_(1)
    elif population == "binary":
        fraction[rows // 2 :] = 1
    elif population == "tiny":
        fraction.fill_(1e-7)
    targets = torch.zeros((rows, 4), device="cuda", dtype=torch.long)
    neg = torch.full_like(targets, -2.0, dtype=torch.float32)
    if population == "empty":
        targets.fill_(-100)
        lse.fill_(float("nan"))
        fraction.fill_(float("nan"))
    actual = patch_loss_forward(lse, mean, neg, targets, vocab, 10000.0, True, fraction)
    expected = _fractional_reduction_reference(lse, mean, neg, targets, vocab, 10000.0, fraction)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.double(), reference, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("bad_fraction", [-0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize("rows", [17, 259])
def test_fractional_patch_invalid_active_metadata_is_not_clamped(rows, bad_fraction):
    lse = torch.full((rows,), 6.0, device="cuda")
    mean = torch.zeros_like(lse)
    targets = torch.zeros((rows, 4), device="cuda", dtype=torch.long)
    neg = torch.zeros_like(targets, dtype=torch.float32)
    fraction = torch.zeros_like(lse)
    fraction[0] = bad_fraction
    actual = patch_loss_forward(lse, mean, neg, targets, 257, 1.0, True, fraction)
    assert torch.isnan(actual[0])


@pytest.mark.parametrize("patch_size", [1, 4])
@pytest.mark.parametrize("gamma", [0.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fractional_patch_cce_dense_forward_backward(patch_size, gamma, dtype):
    generator = torch.Generator(device="cuda").manual_seed(1711)
    batch, sequence, vocab, dim = 2, 9, 257, 64
    e = (
        torch.randn((batch, sequence, dim), device="cuda", dtype=dtype, generator=generator) / 8
    ).requires_grad_()
    c = torch.randn((vocab, dim), device="cuda", dtype=dtype, generator=generator).requires_grad_()
    bias = torch.randn(vocab, device="cuda", dtype=dtype, generator=generator).requires_grad_()
    targets = torch.randint(
        vocab, (batch, sequence, patch_size), device="cuda", generator=generator
    )
    targets[0, 0] = -100
    targets[1, 2, 0] = vocab + 1
    fraction = (
        torch.randint(5, (batch, sequence * 2), device="cuda", generator=generator).float()[
            ..., ::2
        ]
        / 4
    )
    fraction = fraction.T.contiguous().T  # Explicit noncontiguous metadata contract.
    reference_e, reference_c, reference_bias = [
        x.detach().clone().requires_grad_() for x in (e, c, bias)
    ]
    expected, expected_metrics = _dense_patch_loss(
        reference_e,
        reference_c,
        targets,
        reference_bias,
        mile_gamma=gamma,
        mu_loss_lambda=1e-4,
        mile_group_fraction=fraction,
    )
    actual, actual_metrics = linear_cross_entropy(
        e,
        c,
        targets,
        bias=bias,
        impl="cce_exact",
        filter_eps=None,
        patch_training_enabled=True,
        mile_enabled=True,
        mile_gamma=gamma,
        mile_group_fraction=fraction,
        mu_loss_enabled=True,
        return_loss_metrics=True,
    )
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    for name in expected_metrics:
        torch.testing.assert_close(
            actual_metrics[name], expected_metrics[name], rtol=3e-2, atol=3e-2
        )
    for value, reference in zip(
        (e, c, bias), (reference_e, reference_c, reference_bias), strict=True
    ):
        assert value.grad is not None and reference.grad is not None
        _assert_relative_gradient(value.grad, reference.grad)


@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
def test_fractional_patch_binary_k1_matches_existing_token_grouping(gamma):
    generator = torch.Generator(device="cuda").manual_seed(1777)
    e = torch.randn(
        (2, 9, 64), device="cuda", dtype=torch.bfloat16, generator=generator
    ).requires_grad_()
    c = torch.randn(
        (257, 64), device="cuda", dtype=torch.bfloat16, generator=generator
    ).requires_grad_()
    targets = torch.randint(257, (2, 9), device="cuda", generator=generator)
    targets[0, 0] = -100
    groups = torch.zeros_like(targets, dtype=torch.bool)
    groups[:, 2::3] = True
    patch_e, patch_c = [x.detach().clone().requires_grad_() for x in (e, c)]
    common = dict(
        impl="cce_exact", filter_eps=None, mile_enabled=True, mile_gamma=gamma, mu_loss_enabled=True
    )
    token = linear_cross_entropy(e, c, targets, mile_group_mask=groups, **common)
    patch = linear_cross_entropy(
        patch_e,
        patch_c,
        targets[..., None],
        patch_training_enabled=True,
        mile_group_fraction=groups.float(),
        **common,
    )
    token.backward()
    patch.backward()
    torch.testing.assert_close(patch, token, rtol=3e-2, atol=3e-2)
    _assert_relative_gradient(patch_e.grad, e.grad)
    _assert_relative_gradient(patch_c.grad, c.grad)


@pytest.mark.parametrize(
    "case", ["disabled_patch", "disabled_mile", "shape", "dtype", "device", "gradient", "gamma"]
)
def test_fractional_patch_rejects_invalid_metadata(case):
    e = torch.randn((2, 4, 64), device="cuda", dtype=torch.bfloat16)
    c = torch.randn((31, 64), device="cuda", dtype=torch.bfloat16)
    targets = torch.zeros((2, 4, 4), device="cuda", dtype=torch.long)
    fraction = torch.zeros((2, 4), device="cuda")
    opts = dict(patch_training_enabled=True, mile_enabled=True)
    if case == "disabled_patch":
        opts["patch_training_enabled"] = False
        targets = targets[..., 0]
    elif case == "disabled_mile":
        opts["mile_enabled"] = False
    elif case == "shape":
        fraction = torch.zeros_like(targets, dtype=torch.float32)
    elif case == "dtype":
        fraction = fraction.bool()
    elif case == "device":
        fraction = fraction.cpu()
    elif case == "gradient":
        fraction.requires_grad_()
    elif case == "gamma":
        opts["mile_gamma"] = 1e100
    with pytest.raises((ValueError, TypeError), match="mile_group_fraction|Fractional MiLe"):
        linear_cross_entropy(e, c, targets, mile_group_fraction=fraction, **opts)


@pytest.mark.parametrize(
    ("sequence", "storage_dtype"),
    [(9, torch.bfloat16), (65, torch.float32), (129, torch.bfloat16)],
    ids=["small-bf16", "multi-fp32-autocast", "large-bf16-patch-tile"],
)
def test_fractional_patch_fullgraph_inductor_forward_backward_reuses_warm_graph(
    sequence, storage_dtype
):
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    torch._dynamo.reset()
    counters.clear()
    generator = torch.Generator(device="cuda").manual_seed(1781)
    dim = 128 if sequence == 129 else 64
    e = torch.randn(
        (2, sequence, dim), device="cuda", dtype=storage_dtype, generator=generator
    ).requires_grad_()
    c = torch.randn(
        (257, dim), device="cuda", dtype=storage_dtype, generator=generator
    ).requires_grad_()
    labels = torch.randint(257, (2, sequence, 4), device="cuda", generator=generator)
    token_labels = torch.full_like(labels, -100)
    token_labels[..., 0] = labels[..., 0]
    fractions = torch.rand((2, sequence), device="cuda", generator=generator)
    variants = [
        (labels, fractions),
        (token_labels, (fractions > 0.5).float()),
        (labels, torch.ones_like(fractions)),
        (torch.full_like(labels, -100), fractions),
        (labels, torch.zeros_like(fractions)),
    ]

    def step(hidden, classifier, target, fraction):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=storage_dtype == torch.float32):
            loss, values = linear_cross_entropy(
                hidden,
                classifier,
                target,
                impl="cce_exact",
                filter_eps=None,
                patch_training_enabled=True,
                mile_enabled=True,
                mile_gamma=2.0,
                mile_group_fraction=fraction,
                mu_loss_enabled=True,
                return_loss_metrics=True,
            )
        return loss + hidden.sum() * 0.0, values

    with torch._inductor.config.patch(force_disable_caches=True):
        compiled = torch.compile(step, fullgraph=True, mode="max-autotune", dynamic=False)
        for target, fraction in variants * 3:
            torch.compiler.cudagraph_mark_step_begin()
            e.grad = c.grad = None
            loss, _ = compiled(e, c, target, fraction)
            loss.backward()
        torch.cuda.synchronize()
        before = (
            dict(counters["stats"]),
            dict(counters["aot_autograd"]),
            dict(counters["graph_break"]),
            metrics.generated_kernel_count,
        )
        for target, fraction in variants * 2:
            torch.compiler.cudagraph_mark_step_begin()
            e.grad = c.grad = None
            expected, _ = step(e, c, target, fraction)
            loss, _ = compiled(e, c, target, fraction)
            loss.backward()
            torch.testing.assert_close(loss, expected, rtol=3e-3, atol=3e-3)
            assert torch.isfinite(e.grad).all() and torch.isfinite(c.grad).all()
        torch.cuda.synchronize()
        after = (
            dict(counters["stats"]),
            dict(counters["aot_autograd"]),
            dict(counters["graph_break"]),
            metrics.generated_kernel_count,
        )
    assert before == after
    assert counters["stats"]["unique_graphs"] == 1
    assert not counters["graph_break"]
