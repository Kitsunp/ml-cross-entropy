"""Sparse Copy-Paste loss/entropy/gradient against an independent dense oracle."""

import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy.copy_paste import _copy_paste_sparse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("slots,patch_size", [(1, 1), (4, 4)])
@pytest.mark.parametrize("empty_context", [False, True])
def test_sparse_mixture_against_dense(slots, patch_size, empty_context):
    torch.manual_seed(213)
    b, t, s, v = 2, 4, 4 * patch_size, 13
    device = "cuda"
    logits = torch.randn(b, t, v, device=device)
    source = torch.randint(v, (b, s), device=device)
    source[:, 1] = source[:, 0]  # repeated tokens are one vocabulary class
    mask = torch.ones(b, s, dtype=torch.bool, device=device)
    mask[0, -1] = False
    if empty_context:
        mask.zero_()
    targets = torch.randint(v, (b, t, slots), device=device)
    targets[0, 0, 0] = -100
    targets[1, 2, 0] = source[1, 1]
    safe = targets.clamp(0, v - 1)
    negative = -logits.gather(-1, safe)
    lse = logits.logsumexp(-1)
    mean = (logits.softmax(-1) * logits).sum(-1)
    score = torch.randn(b, t, s, device=device)
    gate = torch.randn(b, t, device=device)
    sink = torch.randn(b, t, device=device)
    context_logits = logits.gather(-1, source[:, None].expand(-1, t, -1))
    order = source.argsort(-1)
    top = torch.empty((b, t, 3), device=device, dtype=torch.uint64)
    inputs = [score, negative, lse, gate, sink]
    inputs = [x.detach().requires_grad_() for x in inputs]
    score, negative, lse, gate, sink = inputs
    actual, entropy, _ = _copy_paste_sparse(
        score,
        context_logits,
        source,
        mask,
        targets,
        negative,
        lse,
        mean,
        gate,
        sink,
        order,
        top,
        v,
        patch_size,
        True,
        False,
        False,
    )
    positions = torch.arange(s, device=device)
    allowed = mask[:, None, :] & (
        positions[None, None, :] < (torch.arange(t, device=device)[None, :, None] + 1) * patch_size
    )
    score_masked = score.masked_fill(~allowed, -torch.inf)
    alpha = torch.cat((sink[..., None], score_masked), dim=-1).softmax(-1)
    c = torch.sigmoid(gate - lse)
    a = 1 - c + c * alpha[..., 0]
    matching = source[:, None, None, :] == safe[..., None]
    copy = (alpha[..., None, 1:] * matching).sum(-1)
    base = torch.exp(-negative - lse[..., None])
    probability = a[..., None] * base + c[..., None] * copy
    expected = torch.where((targets >= 0) & (targets < v), -probability.log(), 0.0)
    probs = a[..., None] * logits.softmax(-1)
    probs = probs.scatter_add(
        -1, source[:, None, :].expand(-1, t, -1), c[..., None] * alpha[..., 1:]
    )
    expected_entropy = -(probs * probs.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=2e-5, atol=2e-5)
    weight = torch.rand_like(actual)
    gradients = torch.autograd.grad((actual * weight).sum(), inputs, retain_graph=True)
    expected_gradients = torch.autograd.grad((expected * weight).sum(), inputs)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=6e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_mixture_torch_compile():
    torch.manual_seed(99)
    b, t, s, v = 2, 4, 4, 13
    device = "cuda"
    ids = torch.randint(v, (b, s), device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    target = torch.randint(v, (b, t, 1), device=device)
    logits = torch.randn(b, t, v, device=device)
    context = logits.gather(-1, ids[:, None, :].expand(-1, t, -1))
    order = ids.argsort(-1)
    keys = torch.empty(0, device=device, dtype=torch.uint64)
    mean = (logits.softmax(-1) * logits).sum(-1)
    values = [
        torch.randn(b, t, s, device=device),
        -logits.gather(-1, target),
        logits.logsumexp(-1),
        torch.randn(b, t, device=device),
        torch.randn(b, t, device=device),
    ]
    values = [value.detach().requires_grad_() for value in values]

    def loss(score, negative, lse, gate, sink):
        return _copy_paste_sparse(
            score,
            context,
            ids,
            mask,
            target,
            negative,
            lse,
            mean,
            gate,
            sink,
            order,
            keys,
            v,
            1,
            True,
            False,
            False,
        )[0].sum()

    expected = loss(*values)
    reference_grads = torch.autograd.grad(expected, values, retain_graph=True)
    compiled = torch.compile(loss, fullgraph=True, backend="inductor")
    actual = compiled(*values)
    actual_grads = torch.autograd.grad(actual, values)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_grad, expected_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "gate_value,sink_value", [(100.0, -100.0), (-100.0, 100.0), (100.0, 100.0), (-100.0, -100.0)]
)
def test_extreme_logits_keep_finite_loss_and_gradients(gate_value, sink_value):
    device = "cuda"
    score = torch.tensor([[[100.0, -100.0, 0.0]]], device=device, requires_grad=True)
    negative = torch.tensor([[[200.0]]], device=device, requires_grad=True)
    lse = torch.zeros(1, 1, device=device, requires_grad=True)
    gate = torch.full((1, 1), gate_value, device=device, requires_grad=True)
    sink = torch.full((1, 1), sink_value, device=device, requires_grad=True)
    source = torch.tensor([[1, 1, 2]], device=device)
    mask = torch.ones_like(source, dtype=torch.bool)
    target = torch.tensor([[[0]]], device=device)  # Never appears in context.
    empty = torch.empty(0, device=device)
    out = _copy_paste_sparse(
        score,
        empty,
        source,
        mask,
        target,
        negative,
        lse,
        empty,
        gate,
        sink,
        source.new_empty(0),
        torch.empty(0, device=device, dtype=torch.uint64),
        3,
        3,
        False,
        False,
        False,
    )[0]
    ds = [x.detach().double().requires_grad_() for x in (score, negative, lse, gate, sink)]
    scores64, negative64, lse64, gate64, sink64 = ds
    norm = torch.logsumexp(torch.cat((sink64[..., None], scores64), -1), -1)
    gate_delta = gate64 - lse64
    log_a = torch.logaddexp(-F.softplus(gate_delta), -F.softplus(-gate_delta) + sink64 - norm)
    expected = -(log_a - negative64[..., 0] - lse64)
    torch.testing.assert_close(out.double(), expected[..., None], rtol=1e-5, atol=1e-4)
    actual_grads = torch.autograd.grad(out.sum(), (score, negative, lse, gate, sink))
    expected_grads = torch.autograd.grad(expected.sum(), ds)
    for actual, reference in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), reference, rtol=2e-4, atol=1e-6)
