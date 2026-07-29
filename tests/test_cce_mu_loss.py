# Copyright (C) 2026. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy import LinearCrossEntropy, linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.mu_loss import (
    add_mu_loss_gradient_kernel,
    mu_loss_forward_kernel,
)

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mu_loss_kernels_match_dense(dtype: torch.dtype) -> None:
    torch.manual_seed(31)
    coefficient = 3e-4
    c = torch.randn(137, 61, device="cuda", dtype=dtype)

    loss, mu, vocab_size = mu_loss_forward_kernel(c, coefficient)
    expected_mu = c.float().mean(dim=0)
    expected_loss = coefficient * expected_mu.square().sum()
    torch.testing.assert_close(mu, expected_mu, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-6, atol=2e-6)
    assert float(vocab_size) == c.size(0)

    dc = torch.randn_like(c, dtype=torch.float32)
    expected_dc = dc.clone()
    d_out = torch.tensor(0.37, device="cuda")
    add_mu_loss_gradient_kernel(dc, mu, vocab_size, d_out, coefficient)
    expected_dc += d_out * (2.0 * coefficient / c.size(0)) * expected_mu
    torch.testing.assert_close(dc, expected_dc, rtol=2e-6, atol=2e-6)


def _dense_loss(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    coefficient: float,
    mile_enabled: bool,
) -> torch.Tensor:
    logits = (e @ c.T).float()
    if bias is not None:
        logits = logits + bias.float()
    flat_targets = targets.flatten()
    valid = flat_targets != IGNORE_INDEX
    safe_targets = flat_targets.masked_fill(~valid, 0)
    log_probs = logits.flatten(0, -2).log_softmax(dim=-1)
    nll = -log_probs.gather(1, safe_targets[:, None]).squeeze(1)
    if mile_enabled:
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)
        weights = (1.0 + entropy).detach()
        weights = weights * weights[valid].mean().reciprocal()
        nll = nll * weights
    ce = nll[valid].mean()
    return ce + coefficient * c.float().mean(dim=0).square().sum()


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce_exact", "cce_kahan_full_c"])
@pytest.mark.parametrize("mile_enabled", [False, True])
def test_cce_mu_loss_matches_dense(impl: str, mile_enabled: bool) -> None:
    torch.manual_seed(37)
    coefficient = 7e-4
    e_data = torch.randn(3, 17, 64, device="cuda", dtype=torch.bfloat16) / 8
    c_data = torch.randn(263, 64, device="cuda", dtype=torch.bfloat16)
    bias_data = torch.randn(263, device="cuda", dtype=torch.bfloat16) * 0.02
    targets = torch.randint(0, 263, (3, 17), device="cuda")
    targets[0, 5] = IGNORE_INDEX
    targets[2, 11] = IGNORE_INDEX

    e_ref = e_data.clone().requires_grad_(True)
    c_ref = c_data.clone().requires_grad_(True)
    bias_ref = bias_data.clone().requires_grad_(True)
    expected = _dense_loss(e_ref, c_ref, targets, bias_ref, coefficient, mile_enabled)

    e_test = e_data.clone().requires_grad_(True)
    c_test = c_data.clone().requires_grad_(True)
    bias_test = bias_data.clone().requires_grad_(True)
    actual = linear_cross_entropy(
        e_test,
        c_test,
        targets,
        bias_test,
        impl=impl,
        mile_enabled=mile_enabled,
        mu_loss_enabled=True,
        mu_loss_lambda=coefficient,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    expected.backward()
    actual.backward()
    for test_tensor, ref_tensor in ((e_test, e_ref), (c_test, c_ref), (bias_test, bias_ref)):
        assert test_tensor.grad is not None and ref_tensor.grad is not None
        torch.testing.assert_close(test_tensor.grad, ref_tensor.grad, rtol=5e-2, atol=3e-2)


@skip_no_cuda
def test_mu_loss_only_changes_classifier_gradient() -> None:
    torch.manual_seed(41)
    coefficient = 0.2
    e_data = torch.randn(23, 32, device="cuda") / 32**0.5
    c_data = torch.randn(131, 32, device="cuda")
    bias_data = torch.randn(131, device="cuda") * 0.01
    targets = torch.randint(0, 131, (23,), device="cuda")

    gradients = []
    losses = []
    for enabled in (False, True):
        e = e_data.clone().requires_grad_(True)
        c = c_data.clone().requires_grad_(True)
        bias = bias_data.clone().requires_grad_(True)
        loss = linear_cross_entropy(
            e,
            c,
            targets,
            bias,
            impl="cce_exact",
            mu_loss_enabled=enabled,
            mu_loss_lambda=coefficient,
        )
        (0.37 * loss).backward()
        losses.append(loss.detach())
        gradients.append((e.grad, c.grad, bias.grad))

    base_e, base_c, base_bias = gradients[0]
    mu_e, mu_c, mu_bias = gradients[1]
    assert base_e is not None and mu_e is not None
    assert base_c is not None and mu_c is not None
    assert base_bias is not None and mu_bias is not None
    # The CE kernel is launched independently in each branch, so floating-point
    # reduction order can differ by a few ULPs even though mu-loss has no path
    # to the embeddings.
    torch.testing.assert_close(mu_e, base_e, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(mu_bias, base_bias, rtol=0, atol=0)
    expected_delta = (0.37 * 2.0 * coefficient / c_data.size(0)) * c_data.mean(dim=0)
    torch.testing.assert_close(
        mu_c - base_c,
        expected_delta.unsqueeze(0).expand_as(c_data),
        rtol=2e-3,
        atol=2e-6,
    )
    expected_loss_delta = coefficient * c_data.mean(dim=0).square().sum()
    torch.testing.assert_close(losses[1] - losses[0], expected_loss_delta)


@skip_no_cuda
def test_mu_loss_weight_tying_adds_regularizer_once() -> None:
    torch.manual_seed(42)
    coefficient = 5e-3
    vocab_size, embedding_dim = 113, 32
    weight_data = torch.randn(
        vocab_size, embedding_dim, device="cuda", dtype=torch.float32
    )
    input_ids = torch.randint(0, vocab_size, (19,), device="cuda")
    targets = torch.randint(0, vocab_size, (19,), device="cuda")

    weight_ref = weight_data.clone().requires_grad_(True)
    e_ref = torch.nn.functional.embedding(input_ids, weight_ref)
    expected = _dense_loss(e_ref, weight_ref, targets, None, coefficient, False)
    expected.backward()

    weight_test = weight_data.clone().requires_grad_(True)
    e_test = torch.nn.functional.embedding(input_ids, weight_test)
    actual = linear_cross_entropy(
        e_test,
        weight_test,
        targets,
        impl="cce_exact",
        mu_loss_enabled=True,
        mu_loss_lambda=coefficient,
    )
    actual.backward()

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert weight_test.grad is not None and weight_ref.grad is not None
    torch.testing.assert_close(weight_test.grad, weight_ref.grad, rtol=3e-3, atol=3e-4)


@skip_no_cuda
def test_mu_loss_module_and_explicit_disable() -> None:
    torch.manual_seed(43)
    e_data = torch.randn(9, 32, device="cuda", dtype=torch.bfloat16)
    c_data = torch.randn(127, 32, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, 127, (9,), device="cuda")
    module = LinearCrossEntropy(
        impl="cce_exact", mu_loss_enabled=True, mu_loss_lambda=1e-3
    )
    module_loss = module(e_data, c_data, targets)
    function_loss = linear_cross_entropy(
        e_data,
        c_data,
        targets,
        impl="cce_exact",
        mu_loss_enabled=True,
        mu_loss_lambda=1e-3,
    )
    torch.testing.assert_close(module_loss, function_loss)

    base = linear_cross_entropy(e_data, c_data, targets, impl="cce_exact")
    disabled = linear_cross_entropy(
        e_data,
        c_data,
        targets,
        impl="cce_exact",
        mu_loss_enabled=False,
        mu_loss_lambda=99.0,
    )
    torch.testing.assert_close(disabled, base, rtol=0, atol=0)


def test_mu_loss_rejects_unsupported_options() -> None:
    e = torch.randn(2, 3)
    c = torch.randn(5, 3)
    targets = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="mu_loss_enabled"):
        linear_cross_entropy(e, c, targets, impl="torch_compile", mu_loss_enabled=True)
    with pytest.raises(ValueError, match="mu_loss_lambda"):
        linear_cross_entropy(
            e, c, targets, impl="cce", mu_loss_enabled=True, mu_loss_lambda=-1.0
        )
    with pytest.raises(ValueError, match="reduction='mean'"):
        linear_cross_entropy(
            e, c, targets, impl="cce", mu_loss_enabled=True, reduction="sum"
        )
