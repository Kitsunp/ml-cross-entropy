import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.cce_mile import (
    cce_mile_forward_kernel,
    normalize_mile_weight_groups,
)
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _assert_relative_gradient(actual: torch.Tensor, expected: torch.Tensor) -> None:
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(
        1e-12
    )
    assert float(relative_l2) < 4e-2, f"Relative L2 gradient error: {float(relative_l2)}"


def _dense_mile(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    gamma: float,
    softcap: float | None,
    shift: int,
    reduction: str,
    group_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shift:
        e = e[..., :-shift, :]
        targets = targets[..., shift:]
        if group_mask is not None:
            group_mask = group_mask[..., :-shift]
    output_shape = targets.shape
    logits = e.reshape(-1, e.size(-1)) @ c.T
    if bias is not None:
        logits = logits + bias
    if softcap is not None:
        logits = softcapping(logits, softcap)
    logits = logits.float()
    flat_targets = targets.reshape(-1)
    valid = flat_targets != IGNORE_INDEX
    safe_targets = flat_targets.masked_fill(~valid, 0)
    log_probs = logits.log_softmax(dim=-1)
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum(dim=-1)
    nll = -log_probs.gather(1, safe_targets[:, None]).squeeze(1)
    weights = (1.0 + entropy).pow(gamma).detach()
    if group_mask is None:
        weights = weights * weights[valid].mean().reciprocal()
    else:
        flat_group = group_mask.reshape(-1)
        for population in (False, True):
            members = valid & (flat_group == population)
            if bool(members.any()):
                weights[members] *= weights[members].mean().reciprocal()
    losses = weights * nll
    losses = losses.masked_fill(~valid, 0.0)
    lse = logits.logsumexp(dim=-1).masked_fill(~valid, 0.0)
    if reduction == "mean":
        loss = losses.sum() / valid.sum()
    elif reduction == "sum":
        loss = losses.sum()
    else:
        loss = losses.view(output_shape)
        lse = lse.view(output_shape)
    return loss, lse


@skip_no_cuda
@pytest.mark.parametrize("dtype,error_tol", [(torch.float32, 2e-4), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("has_bias", [False, True])
def test_mean_logit(
    dtype: torch.dtype, error_tol: float, softcap: float | None, has_bias: bool
) -> None:
    torch.manual_seed(3)
    e = torch.randn(33, 61, device="cuda", dtype=dtype) / 61**0.5
    c = torch.randn(137, 61, device="cuda", dtype=dtype)
    bias = torch.randn(137, device="cuda", dtype=dtype) * 0.02 if has_bias else None
    logits = e @ c.T
    if bias is not None:
        logits = logits + bias
    if softcap is not None:
        logits = softcapping(logits, softcap)
    logits = logits.float()
    expected = (logits.softmax(dim=-1) * logits).sum(dim=-1)
    actual = cce_lse_forward_kernel(e, c, bias, softcap=softcap, return_mean_logit=True).mean_logit
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=error_tol, atol=error_tol)


CASES = [
    (0.0, None, False, 0, False, "mean"),
    (0.5, None, True, 0, True, "sum"),
    (1.0, 20.0, False, 1, False, "none"),
    (1.0, None, True, 1, True, "mean"),
    (2.0, 20.0, True, 0, True, "none"),
]


@skip_no_cuda
@pytest.mark.parametrize("size", [1, 257, 4097, 20000])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
def test_cce_mile_forward_kernel_matches_torch(size: int, gamma: float) -> None:
    torch.manual_seed(4)
    lse = torch.rand(size, device="cuda", dtype=torch.float32) * 8
    entropy = torch.rand(size, device="cuda", dtype=torch.float32) * 5
    mean_logit = lse - entropy
    nll = torch.rand(size, device="cuda", dtype=torch.float32) * 10

    expected_weight = (1.0 + entropy).pow(gamma)
    expected_weight *= expected_weight.mean().reciprocal()
    expected_loss = nll * expected_weight
    actual_weight, actual_loss, unweighted_nll_sum = cce_mile_forward_kernel(
        lse, mean_logit, nll, gamma, return_unweighted_nll_sum=True
    )
    assert unweighted_nll_sum is not None

    torch.testing.assert_close(actual_weight, expected_weight, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_weight.mean(), torch.ones((), device="cuda"))
    torch.testing.assert_close(unweighted_nll_sum, nll.sum(), rtol=2e-5, atol=2e-5)


@skip_no_cuda
def test_mile_group_normalization_preserves_each_population_mass() -> None:
    torch.manual_seed(37)
    size = 4097
    entropy = torch.rand(size, device="cuda", dtype=torch.float32) * 5
    lse = torch.rand(size, device="cuda", dtype=torch.float32) * 8
    mean_logit = lse - entropy
    nll = torch.rand(size, device="cuda", dtype=torch.float32) * 10
    group_mask = torch.arange(size, device="cuda") % 7 == 0
    weight, _loss, _ = cce_mile_forward_kernel(lse, mean_logit, nll, 1.0)
    weight, loss = normalize_mile_weight_groups(weight, nll, group_mask, mean_reduction=True)

    torch.testing.assert_close(weight[group_mask].mean(), torch.ones((), device="cuda"))
    torch.testing.assert_close(weight[~group_mask].mean(), torch.ones((), device="cuda"))
    torch.testing.assert_close(loss, nll * weight / size)


@pytest.mark.parametrize("size", [257, 16385])
@pytest.mark.parametrize("population", ["clean", "masked", "sparse"])
@skip_no_cuda
def test_fused_mile_group_normalization_handles_population_edges(
    size: int,
    population: str,
) -> None:
    torch.manual_seed(38)
    entropy = torch.rand(size, device="cuda", dtype=torch.float32) * 7
    lse = torch.rand(size, device="cuda", dtype=torch.float32) * 8
    mean_logit = lse - entropy
    nll = torch.rand(size, device="cuda", dtype=torch.float32) * 10
    if population == "clean":
        group_mask = torch.zeros(size, device="cuda", dtype=torch.bool)
    elif population == "masked":
        group_mask = torch.ones(size, device="cuda", dtype=torch.bool)
    else:
        group_mask = torch.arange(size, device="cuda") % 127 == 0

    weight, loss, _ = cce_mile_forward_kernel(
        lse,
        mean_logit,
        nll,
        1.0,
        mean_reduction=True,
        group_mask=group_mask,
    )
    assert torch.isfinite(weight).all()
    assert torch.isfinite(loss).all()
    for members in (group_mask, ~group_mask):
        if bool(members.any()):
            torch.testing.assert_close(
                weight[members].mean(),
                torch.ones((), device="cuda"),
                rtol=2e-4,
                atol=2e-4,
            )
    torch.testing.assert_close(loss, nll * weight / size, rtol=2e-5, atol=2e-5)


@skip_no_cuda
def test_cce_mile_group_mask_matches_dense_with_shift_and_ignored_targets() -> None:
    torch.manual_seed(39)
    e_data = torch.randn(3, 17, 64, device="cuda", dtype=torch.bfloat16) / 8
    c_data = torch.randn(257, 64, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, 257, (3, 17), device="cuda")
    targets[1, 9:] = IGNORE_INDEX
    group_mask = torch.zeros_like(targets, dtype=torch.bool)
    group_mask[:, 1::5] = True

    e_ref = e_data.clone().requires_grad_(True)
    c_ref = c_data.clone().requires_grad_(True)
    expected, _ = _dense_mile(e_ref, c_ref, targets, None, 1.0, None, 1, "mean", group_mask)
    e_actual = e_data.clone().requires_grad_(True)
    c_actual = c_data.clone().requires_grad_(True)
    actual = linear_cross_entropy(
        e_actual,
        c_actual,
        targets,
        shift=1,
        impl="cce_kahan_full_c",
        mile_enabled=True,
        mile_gamma=1.0,
        mile_group_mask=group_mask,
    )
    expected.backward()
    actual.backward()

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    _assert_relative_gradient(e_actual.grad, e_ref.grad)
    _assert_relative_gradient(c_actual.grad, c_ref.grad)


@skip_no_cuda
def test_cce_loss_metrics_reconcile_without_materializing_logits() -> None:
    torch.manual_seed(41)
    e = torch.randn(4, 16, 64, device="cuda", dtype=torch.bfloat16) / 8
    e.requires_grad_(True)
    c = torch.randn(257, 64, device="cuda", dtype=torch.bfloat16)
    c.requires_grad_(True)
    targets = torch.randint(0, 257, (4, 16), device="cuda")
    targets[1, -5:] = IGNORE_INDEX

    loss, metrics = linear_cross_entropy(
        e,
        c,
        targets,
        shift=1,
        impl="cce_kahan_full_c",
        mile_enabled=True,
        mile_gamma=1.0,
        mu_loss_enabled=True,
        mu_loss_lambda=1e-4,
        return_loss_metrics=True,
    )

    reconstructed = (
        metrics["ntp_ce_unweighted"] + metrics["mile_reweighting_delta"] + metrics["mu_loss"]
    )
    torch.testing.assert_close(reconstructed, loss.detach(), rtol=2e-5, atol=2e-5)
    assert not any(metric.requires_grad for metric in metrics.values())
    loss.backward()
    assert e.grad is not None and torch.isfinite(e.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()


@skip_no_cuda
def test_plain_cce_loss_metrics_have_zero_extension_terms() -> None:
    torch.manual_seed(43)
    e = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(257, 64, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, 257, (32,), device="cuda")

    loss, metrics = linear_cross_entropy(
        e,
        c,
        targets,
        impl="cce_exact",
        return_loss_metrics=True,
    )

    torch.testing.assert_close(metrics["ntp_ce_unweighted"], loss.detach())
    torch.testing.assert_close(metrics["mile_reweighting_delta"], torch.zeros_like(loss))
    torch.testing.assert_close(metrics["mu_loss"], torch.zeros_like(loss))


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce_exact", "cce_kahan_full_c"])
@pytest.mark.parametrize("gamma,softcap,has_bias,shift,invalids,reduction", CASES)
def test_cce_mile_matches_dense(
    impl: str,
    gamma: float,
    softcap: float | None,
    has_bias: bool,
    shift: int,
    invalids: bool,
    reduction: str,
) -> None:
    torch.manual_seed(5)
    e_data = torch.randn(4, 16, 64, device="cuda", dtype=torch.bfloat16) / 64**0.5
    c_data = torch.randn(137, 64, device="cuda", dtype=torch.bfloat16)
    bias_data = torch.randn(137, device="cuda", dtype=torch.bfloat16) * 0.02 if has_bias else None
    targets = torch.randint(0, 137, (4, 16), device="cuda")
    if invalids:
        targets[0, 3] = IGNORE_INDEX
        targets[2, 7] = IGNORE_INDEX

    e_ref = e_data.detach().clone().requires_grad_(True)
    c_ref = c_data.detach().clone().requires_grad_(True)
    bias_ref = bias_data.detach().clone().requires_grad_(True) if bias_data is not None else None
    e_test = e_data.detach().clone().requires_grad_(True)
    c_test = c_data.detach().clone().requires_grad_(True)
    bias_test = bias_data.detach().clone().requires_grad_(True) if bias_data is not None else None

    expected, _ = _dense_mile(
        e_ref,
        c_ref,
        targets,
        bias_ref,
        gamma,
        softcap,
        shift,
        reduction,
    )
    actual = linear_cross_entropy(
        e_test,
        c_test,
        targets,
        bias_test,
        softcap=softcap,
        shift=shift,
        reduction=reduction,
        impl=impl,
        mile_enabled=True,
        mile_gamma=gamma,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    upstream = torch.randn_like(expected) if reduction == "none" else None
    expected.backward(upstream)
    actual.backward(upstream)
    assert e_test.grad is not None and e_ref.grad is not None
    assert c_test.grad is not None and c_ref.grad is not None
    _assert_relative_gradient(e_test.grad, e_ref.grad)
    _assert_relative_gradient(c_test.grad, c_ref.grad)
    if has_bias:
        assert bias_test is not None and bias_ref is not None
        assert bias_test.grad is not None and bias_ref.grad is not None
        _assert_relative_gradient(bias_test.grad, bias_ref.grad)


@skip_no_cuda
def test_cce_mile_return_lse_gradient() -> None:
    torch.manual_seed(9)
    e_data = torch.randn(3, 12, 64, device="cuda", dtype=torch.bfloat16) / 8
    c_data = torch.randn(131, 64, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, 131, (3, 12), device="cuda")
    targets[1, 4] = IGNORE_INDEX

    e_ref = e_data.detach().clone().requires_grad_(True)
    c_ref = c_data.detach().clone().requires_grad_(True)
    expected_loss, expected_lse = _dense_mile(e_ref, c_ref, targets, None, 1.0, 20.0, 1, "none")
    expected = expected_loss.mean() + 0.01 * expected_lse.square().mean()

    e_test = e_data.detach().clone().requires_grad_(True)
    c_test = c_data.detach().clone().requires_grad_(True)
    actual_loss, actual_lse = linear_cross_entropy(
        e_test,
        c_test,
        targets,
        softcap=20.0,
        shift=1,
        reduction="none",
        impl="cce_kahan_full_c",
        mile_enabled=True,
        mile_gamma=1.0,
        return_lse=True,
    )
    actual = actual_loss.mean() + 0.01 * actual_lse.square().mean()
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-2, atol=2e-2)
    assert e_test.grad is not None and e_ref.grad is not None
    assert c_test.grad is not None and c_ref.grad is not None
    _assert_relative_gradient(e_test.grad, e_ref.grad)
    _assert_relative_gradient(c_test.grad, c_ref.grad)


@skip_no_cuda
def test_cce_mile_bias_gradient_sums_to_zero() -> None:
    torch.manual_seed(13)
    e = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) / 8
    c = torch.randn(511, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.zeros(511, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, 511, (64,), device="cuda")
    loss = linear_cross_entropy(
        e,
        c,
        targets,
        bias,
        impl="cce_kahan_full_c",
        mile_enabled=True,
        mile_gamma=1.0,
    )
    loss.backward()
    assert bias.grad is not None
    assert abs(float(bias.grad.float().sum())) < 2e-2


@skip_no_cuda
def test_cce_mile_kahan_full_c_keeps_classifier_gradient_complete() -> None:
    torch.manual_seed(17)
    vocab_size = 65536
    e_data = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16) * 0.01
    c_data = torch.full((vocab_size, 32), 0.01, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, vocab_size, (8,), device="cuda")

    grads = {}
    for impl in ("cce_exact", "cce_kahan_full_c"):
        e = e_data.detach().clone().requires_grad_(True)
        c = c_data.detach().clone().requires_grad_(True)
        bias = torch.zeros(vocab_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        linear_cross_entropy(
            e, c, targets, bias, impl=impl, mile_enabled=True, mile_gamma=1.0
        ).backward()
        assert e.grad is not None and c.grad is not None and bias.grad is not None
        grads[impl] = (e.grad, c.grad, bias.grad)

    exact_e, exact_c, exact_bias = grads["cce_exact"]
    full_c_e, full_c_c, full_c_bias = grads["cce_kahan_full_c"]
    _assert_relative_gradient(full_c_c, exact_c)
    _assert_relative_gradient(full_c_bias, exact_bias)
    assert float((full_c_e.float() - exact_e.float()).norm()) > 0.0


def test_mile_rejects_unsupported_options() -> None:
    e = torch.randn(2, 3)
    c = torch.randn(5, 3)
    targets = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="mile_enabled"):
        linear_cross_entropy(e, c, targets, impl="torch_compile", mile_enabled=True)
    with pytest.raises(ValueError, match="return_loss_metrics"):
        linear_cross_entropy(e, c, targets, impl="torch_compile", return_loss_metrics=True)
    with pytest.raises(ValueError, match="mile_gamma"):
        linear_cross_entropy(e, c, targets, impl="cce", mile_enabled=True, mile_gamma=-1.0)
    with pytest.raises(ValueError, match="mile_enabled"):
        linear_cross_entropy(
            e,
            c,
            targets,
            impl="cce",
            mile_group_mask=torch.zeros_like(targets, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="return_loss_metrics"):
        linear_cross_entropy(
            e,
            c,
            targets,
            impl="cce",
            reduction="sum",
            return_loss_metrics=True,
        )


@skip_no_cuda
def test_mile_explicit_disable_uses_plain_cce() -> None:
    torch.manual_seed(29)
    e_data = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    c_data = torch.randn(257, 64, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, 257, (32,), device="cuda")

    outputs = []
    gradients = []
    for kwargs in (
        {},
        {
            "mile_enabled": False,
            "mile_gamma": 3.0,
        },
    ):
        e = e_data.detach().clone().requires_grad_(True)
        c = c_data.detach().clone().requires_grad_(True)
        loss = linear_cross_entropy(e, c, targets, impl="cce_exact", **kwargs)
        loss.backward()
        outputs.append(loss.detach())
        gradients.append((e.grad, c.grad))

    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(gradients[1][0], gradients[0][0], rtol=0, atol=0)
    torch.testing.assert_close(gradients[1][1], gradients[0][1], rtol=0, atol=0)
