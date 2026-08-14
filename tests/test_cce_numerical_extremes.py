import math

import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.cce_mile import cce_mile_forward_kernel
from cut_cross_entropy.constants import IGNORE_INDEX


skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _stable_normalized_entropy_weight(
    lse: torch.Tensor,
    mean_logit: torch.Tensor,
    gamma: float,
    vocab_size: int,
) -> torch.Tensor:
    entropy = (lse.double() - mean_logit.double()).clamp(0.0, math.log(vocab_size))
    if gamma == 0.0:
        return torch.ones_like(entropy, dtype=torch.float32)
    log_weight = gamma * torch.log1p(entropy)
    scaled = torch.exp(log_weight - log_weight.max())
    return (scaled / scaled.mean()).float()


@skip_no_cuda
@pytest.mark.parametrize("size", [16_384, 16_385, 32_704])
@pytest.mark.parametrize("gamma", [1.0, 2.0, 32.0])
def test_mile_normalization_stays_finite_at_fp32_extremes(size: int, gamma: float) -> None:
    vocab_size = 151_936
    # These scalars are finite, but the old direct subtraction/power/sum path
    # overflows even though the normalized weights have a finite solution.
    lse = torch.full((size,), 3.0e38, device="cuda", dtype=torch.float32)
    mean_logit = torch.full_like(lse, -3.0e38)
    nll = torch.linspace(0.0, 32.0, size, device="cuda", dtype=torch.float32)

    expected_weight = _stable_normalized_entropy_weight(lse, mean_logit, gamma, vocab_size)
    actual_weight, actual_loss, nll_sum = cce_mile_forward_kernel(
        lse,
        mean_logit,
        nll,
        gamma,
        return_unweighted_nll_sum=True,
        max_entropy=math.log(vocab_size),
    )
    torch.cuda.synchronize()

    assert torch.isfinite(actual_weight).all()
    assert torch.isfinite(actual_loss).all()
    assert nll_sum is not None and torch.isfinite(nll_sum)
    torch.testing.assert_close(actual_weight.mean(), torch.ones((), device="cuda"))
    torch.testing.assert_close(actual_weight, expected_weight, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_loss, nll * expected_weight, rtol=2e-5, atol=2e-5)


@skip_no_cuda
def test_mile_large_finite_dynamic_range_matches_stable_reference() -> None:
    size = 32_704
    vocab_size = 151_936
    entropy = torch.logspace(-20, 38, size, device="cuda", dtype=torch.float32)
    lse = entropy
    mean_logit = torch.zeros_like(entropy)
    nll = torch.ones_like(entropy)

    expected_weight = _stable_normalized_entropy_weight(lse, mean_logit, 8.0, vocab_size)
    actual_weight, actual_loss, _ = cce_mile_forward_kernel(
        lse,
        mean_logit,
        nll,
        8.0,
        max_entropy=math.log(vocab_size),
    )
    torch.cuda.synchronize()

    assert torch.isfinite(actual_weight).all()
    torch.testing.assert_close(actual_weight, expected_weight, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(actual_loss, expected_weight, rtol=3e-5, atol=3e-5)


@skip_no_cuda
def test_mile_mean_reduction_scales_before_a_finite_product_overflows() -> None:
    size = 32_704
    vocab_size = 151_936
    max_entropy = math.log(vocab_size)
    entropy = torch.zeros(size, device="cuda", dtype=torch.float32)
    entropy[0] = max_entropy
    lse = entropy
    mean_logit = torch.zeros_like(entropy)
    nll = torch.zeros_like(entropy)
    nll[0] = 3.0e38

    expected_weight = _stable_normalized_entropy_weight(lse, mean_logit, 1.0, vocab_size)
    expected_mean = (nll.double() * expected_weight.double()).mean()
    actual_weight, scaled_token_loss, unweighted_mean = cce_mile_forward_kernel(
        lse,
        mean_logit,
        nll,
        1.0,
        max_entropy=max_entropy,
        return_unweighted_nll_mean=True,
        mean_reduction=True,
    )
    actual_mean = scaled_token_loss.sum()
    torch.cuda.synchronize()

    assert torch.isfinite(actual_weight).all()
    assert torch.isfinite(scaled_token_loss).all()
    assert torch.isfinite(actual_mean)
    assert unweighted_mean is not None and torch.isfinite(unweighted_mean)
    torch.testing.assert_close(actual_mean.double(), expected_mean, rtol=2e-5, atol=0.0)
    torch.testing.assert_close(
        unweighted_mean.double(), nll.double().mean(), rtol=2e-5, atol=0.0
    )


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce_exact", "cce_kahan_full_c"])
def test_out_of_range_targets_are_safely_ignored(impl: str) -> None:
    torch.manual_seed(20_260_814)
    rows, vocab, dim = 65, 137, 64
    e_data = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16) / dim**0.5
    c_data = torch.randn(vocab, dim, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[3] = -1
    targets[7] = vocab
    targets[11] = torch.iinfo(torch.int64).min
    targets[13] = torch.iinfo(torch.int64).max
    targets[17] = IGNORE_INDEX
    safe_targets = targets.clone()
    invalid = (safe_targets != IGNORE_INDEX) & (
        (safe_targets < 0) | (safe_targets >= vocab)
    )
    safe_targets[invalid] = IGNORE_INDEX

    e_ref = e_data.detach().clone().requires_grad_(True)
    c_ref = c_data.detach().clone().requires_grad_(True)
    logits = e_ref[:-1].float() @ c_ref.float().T
    expected = F.cross_entropy(
        logits, safe_targets[1:], ignore_index=IGNORE_INDEX, reduction="mean"
    )
    expected.backward()

    e_actual = e_data.detach().clone().requires_grad_(True)
    c_actual = c_data.detach().clone().requires_grad_(True)
    actual = linear_cross_entropy(
        e_actual,
        c_actual,
        targets,
        shift=1,
        reduction="mean",
        impl=impl,
    )
    actual.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(actual)
    assert e_actual.grad is not None and torch.isfinite(e_actual.grad).all()
    assert c_actual.grad is not None and torch.isfinite(c_actual.grad).all()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    assert e_ref.grad is not None and c_ref.grad is not None
    rel_e = (e_actual.grad.float() - e_ref.grad.float()).norm() / e_ref.grad.float().norm()
    rel_c = (c_actual.grad.float() - c_ref.grad.float()).norm() / c_ref.grad.float().norm()
    assert float(rel_e) < 4e-2
    assert float(rel_c) < 4e-2
