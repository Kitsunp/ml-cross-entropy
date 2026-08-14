from __future__ import annotations

import pytest
import torch

from cut_cross_entropy import PatchTrainingSchedule, linear_cross_entropy
from cut_cross_entropy.cce_patch import patch_loss_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


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

    denominator = (base_weight * counts).sum().clamp_min(1.0)
    objective = (nll * valid_f32 * base_weight[:, None]).sum() / denominator
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
