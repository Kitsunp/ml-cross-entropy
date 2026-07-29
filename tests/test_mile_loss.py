import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy import linear_mile_loss
from cut_cross_entropy.constants import IGNORE_INDEX


def _dense_mile(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    gamma: float,
    reduction: str,
    shift: int,
) -> torch.Tensor:
    if shift:
        e = e[..., :-shift, :]
        targets = targets[..., shift:]
    logits = e.reshape(-1, e.size(-1)).float() @ c.float().T
    if bias is not None:
        logits = logits + bias.float()
    flat_targets = targets.reshape(-1)
    valid = flat_targets != IGNORE_INDEX
    logits = logits[valid]
    flat_targets = flat_targets[valid]
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    weights = (1.0 + entropy).pow(gamma).detach()
    weights = weights * weights.mean().reciprocal()
    losses = weights * -log_probs.gather(1, flat_targets[:, None]).squeeze(1)
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    output = losses.new_zeros(targets.numel())
    output[valid] = losses
    return output.view(targets.shape)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("shift", [0, 1])
@pytest.mark.parametrize("has_bias", [False, True])
def test_chunked_mile_matches_dense(
    gamma: float, reduction: str, shift: int, has_bias: bool
) -> None:
    torch.manual_seed(7)
    e_ref = torch.randn(2, 5, 6, dtype=torch.float32, requires_grad=True)
    c_ref = torch.randn(11, 6, dtype=torch.float32, requires_grad=True)
    bias_ref = torch.randn(11, dtype=torch.float32, requires_grad=True) if has_bias else None
    targets = torch.randint(0, 11, (2, 5))
    targets[0, 2] = IGNORE_INDEX

    e_test = e_ref.detach().clone().requires_grad_(True)
    c_test = c_ref.detach().clone().requires_grad_(True)
    bias_test = bias_ref.detach().clone().requires_grad_(True) if bias_ref is not None else None

    expected = _dense_mile(e_ref, c_ref, targets, bias_ref, gamma, reduction, shift)
    actual = linear_mile_loss(
        e_test,
        c_test,
        targets,
        bias_test,
        gamma=gamma,
        reduction=reduction,
        shift=shift,
        chunk_size=4,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    grad = torch.randn_like(expected) if reduction == "none" else None
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(e_test.grad, e_ref.grad, rtol=8e-5, atol=8e-5)
    torch.testing.assert_close(c_test.grad, c_ref.grad, rtol=8e-5, atol=8e-5)
    if has_bias:
        assert bias_test is not None and bias_ref is not None
        torch.testing.assert_close(bias_test.grad, bias_ref.grad, rtol=8e-5, atol=8e-5)


def test_gamma_zero_matches_cross_entropy() -> None:
    torch.manual_seed(11)
    e = torch.randn(8, 5, requires_grad=True)
    c = torch.randn(13, 5, requires_grad=True)
    targets = torch.randint(0, 13, (8,))
    expected = F.cross_entropy(e @ c.T, targets)
    actual = linear_mile_loss(e, c, targets, gamma=0.0, chunk_size=3)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("gamma", [-1.0, float("inf")])
def test_invalid_gamma(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        linear_mile_loss(
            torch.randn(2, 3),
            torch.randn(5, 3),
            torch.zeros(2, dtype=torch.long),
            gamma=gamma,
        )
