"""CCE statistics and sparse Copy-Paste composition against dense Torch."""

import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy.copy_paste import _copy_paste_sparse, _copy_paste_stats, copy_paste_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("slots,patch_size", [(1, 1), (4, 1), (4, 4)])
def test_cce_sparse_composition(dtype, slots, patch_size):
    torch.manual_seed(42)
    b, t, d, v, s = 2, 5, 32, 73, 5 * patch_size
    h = torch.randn(b, t, d, device="cuda", dtype=dtype, requires_grad=True)
    c = torch.randn(v, d, device="cuda", dtype=dtype, requires_grad=True)
    score = torch.randn(b, t, s, device="cuda", requires_grad=True)
    gate = torch.randn(b, t, device="cuda", requires_grad=True)
    sink = torch.randn(b, t, device="cuda", requires_grad=True)
    ids = torch.randint(v, (b, s), device="cuda")
    ids[:, 1] = ids[:, 0]
    mask = torch.ones((b, s), dtype=torch.bool, device="cuda")
    mask[0, -1] = False
    target = torch.randint(v, (b, t, slots), device="cuda")
    if patch_size == 1 and slots > 1:
        target[..., 1:] = -100
    target[0, 0, 0] = -100
    target[1, 3, 0] = ids[1, 1]
    lse, neg, mean, keys = _copy_paste_stats(h, c, target, None, True, patch_size == 1)
    with torch.no_grad():
        context_logits = (
            torch.matmul(h.detach(), c.detach()[ids].transpose(1, 2)).float().contiguous()
        )
    nll, entropy, counts = _copy_paste_sparse(
        score,
        context_logits,
        ids,
        mask,
        target,
        neg,
        lse,
        mean,
        gate,
        sink,
        ids.argsort(-1),
        keys,
        v,
        patch_size,
        True,
        patch_size == 1,
        dtype == torch.bfloat16,
    )
    logits = F.linear(h, c).float()
    alpha = torch.cat(
        (
            sink[..., None],
            score.masked_fill(
                ~(
                    mask[:, None, :]
                    & (
                        torch.arange(s, device="cuda")[None, None, :]
                        < (torch.arange(t, device="cuda")[None, :, None] + 1) * patch_size
                    )
                ),
                -torch.inf,
            ),
        ),
        -1,
    ).softmax(-1)
    copy_gate = torch.sigmoid(gate - logits.logsumexp(-1))
    base_weight = 1 - copy_gate + copy_gate * alpha[..., 0]
    probs = base_weight[..., None] * logits.softmax(-1)
    probs = probs.scatter_add(
        -1, ids[:, None, :].expand(-1, t, -1), copy_gate[..., None] * alpha[..., 1:]
    )
    safe = target.clamp(0, v - 1)
    expected_nll = torch.where((target >= 0) & (target < v), -probs.gather(-1, safe).log(), 0.0)
    expected_entropy = -(probs * probs.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
    tol = 0.025 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(nll, expected_nll, rtol=tol, atol=tol)
    torch.testing.assert_close(entropy, expected_entropy, rtol=tol, atol=tol)
    if patch_size == 1:
        y = safe[..., 0]
        yp = probs.gather(-1, y[..., None])
        order = torch.arange(v, device="cuda")
        rank = ((probs > yp) | ((probs == yp) & (order < y[..., None]))).sum(-1)
        valid = target[..., 0] >= 0
        expected_counts = torch.stack(
            (
                (valid & (rank == 0)).sum(),
                (valid & (rank < 3)).sum(),
                valid.new_zeros((), dtype=torch.long),
                valid.sum(),
                valid.new_zeros((), dtype=torch.long),
            )
        ).float()
        torch.testing.assert_close(counts.sum(0), expected_counts, rtol=0, atol=0)
    grad_weight = torch.rand_like(nll)
    variables = (h, c, score, gate, sink)
    actual_gradients = torch.autograd.grad((nll * grad_weight).sum(), variables, retain_graph=True)
    expected_gradients = torch.autograd.grad((expected_nll * grad_weight).sum(), variables)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.07 if dtype == torch.bfloat16 else 2e-4,
            atol=0.04 if dtype == torch.bfloat16 else 2e-5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vocab", [37, 65537])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("reduction", ["lock", "split"])
def test_topk_ties_and_repeated_copy_ids(vocab, empty, reduction, monkeypatch):
    """Both packed-key formats preserve lower-ID ties after repeated copies."""
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    b, t, d = 1, 4, 8
    h = torch.zeros(b, t, d, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(vocab, d, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[7, 7, 5, 5]], device="cuda")
    mask = (
        torch.zeros_like(ids, dtype=torch.bool) if empty else torch.ones_like(ids, dtype=torch.bool)
    )
    target = torch.tensor([[[0], [7], [5], [6]]], device="cuda")
    scores = torch.zeros((b, t, t), device="cuda")
    gate = torch.full((b, t), 20.0, device="cuda")
    sink = torch.full((b, t), -20.0, device="cuda")
    lse, neg, mean, keys = _copy_paste_stats(h, c, target, None, True, True)
    candidates = torch.zeros_like(scores, dtype=torch.bfloat16)
    nll, entropy, counts = _copy_paste_sparse(
        scores,
        candidates,
        ids,
        mask,
        target,
        neg,
        lse,
        mean,
        gate,
        sink,
        ids.argsort(-1),
        keys,
        vocab,
        1,
        True,
        True,
        True,
    )
    allowed = mask[:, None, :] & (
        torch.arange(t, device="cuda")[None, None, :]
        <= torch.arange(t, device="cuda")[None, :, None]
    )
    alpha = torch.cat((sink[..., None], scores.masked_fill(~allowed, -torch.inf)), -1).softmax(-1)
    cp_gate = torch.sigmoid(gate - lse)
    probs = (
        ((1 - cp_gate + cp_gate * alpha[..., 0]) / vocab)[..., None].expand(-1, -1, vocab).clone()
    )
    probs.scatter_add_(-1, ids[:, None, :].expand(-1, t, -1), cp_gate[..., None] * alpha[..., 1:])
    target_prob = probs.gather(-1, target)
    rank = (
        (probs > target_prob)
        | ((probs == target_prob) & (torch.arange(vocab, device="cuda") < target))
    ).sum(-1)
    expected = torch.stack(
        (
            (rank == 0).sum(),
            (rank < 3).sum(),
            rank.new_zeros(()),
            rank.new_full((), b * t),
            rank.new_zeros(()),
        )
    ).float()
    torch.testing.assert_close(counts.sum(0), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stats_bias_and_repeated_targets(dtype):
    torch.manual_seed(19)
    h = torch.randn(2, 3, 32, device="cuda", dtype=dtype, requires_grad=True)
    c = torch.randn(23, 32, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(23, device="cuda", dtype=dtype, requires_grad=True)
    targets = torch.tensor([[[1, 1], [2, 4], [-100, 7]], [[5, 5], [4, 3], [0, 0]]], device="cuda")
    lse, neg, _, _ = _copy_paste_stats(h, c, targets, bias, False, False)
    logits = F.linear(h, c, bias).float()
    expected_lse = logits.logsumexp(-1)
    expected_neg = -logits.gather(-1, targets.clamp_min(0))
    expected_neg = torch.where(targets >= 0, expected_neg, 0.0)
    tol = 0.1 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=tol)
    torch.testing.assert_close(neg, expected_neg, rtol=1e-5, atol=tol)
    upstream_lse, upstream_neg = torch.randn_like(lse), torch.randn_like(neg)
    actual = torch.autograd.grad(
        (lse * upstream_lse).sum() + (neg * upstream_neg).sum(), (h, c, bias), retain_graph=True
    )
    expected = torch.autograd.grad(
        (expected_lse * upstream_lse).sum() + (expected_neg * upstream_neg).sum(), (h, c, bias)
    )
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0.05 if dtype == torch.bfloat16 else 2e-4, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_invalid_targets_and_masked_context():
    b, t, d, v = 1, 3, 32, 31
    h = torch.randn(b, t, d, device="cuda", requires_grad=True)
    c = torch.randn(v, d, device="cuda", requires_grad=True)
    query = torch.randn_like(h, requires_grad=True)
    keys = torch.randn_like(h, requires_grad=True)
    gate = torch.randn(b, t, device="cuda", requires_grad=True)
    sink = torch.randn(b, t, device="cuda", requires_grad=True)
    ids = torch.zeros(b, t, dtype=torch.long, device="cuda")
    mask = torch.zeros_like(ids, dtype=torch.bool)
    targets = torch.full((b, t, 1), -100, device="cuda")
    loss, ce, delta, mu, counts = copy_paste_loss(
        h,
        c,
        None,
        query,
        keys,
        ids,
        mask,
        targets,
        gate,
        sink,
        use_mile=True,
        use_mu=True,
        need_topk=True,
    )
    loss.backward()
    assert ce.item() == 0 and delta.item() == 0
    torch.testing.assert_close(loss.detach(), mu)
    torch.testing.assert_close(counts, torch.zeros_like(counts))
    for value in (h, c, query, keys, gate, sink):
        assert value.grad is not None and torch.isfinite(value.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch,time", [(0, 2), (1, 0)])
def test_empty_batch_or_predicted_rows(batch, time):
    d, v = 8, 17
    h = torch.empty(batch, time, d, device="cuda", requires_grad=True)
    c = torch.randn(v, d, device="cuda", requires_grad=True)
    query = torch.empty_like(h)
    keys = torch.empty(batch, 0, d, device="cuda")
    ids = torch.empty(batch, 0, dtype=torch.long, device="cuda")
    mask = torch.empty_like(ids, dtype=torch.bool)
    targets = torch.empty(batch, time, 1, dtype=torch.long, device="cuda")
    gate = torch.empty(batch, time, device="cuda")
    sink = torch.empty_like(gate)
    loss, ce, delta, mu, counts = copy_paste_loss(
        h,
        c,
        None,
        query,
        keys,
        ids,
        mask,
        targets,
        gate,
        sink,
        use_mile=True,
        use_mu=True,
        need_topk=True,
    )
    torch.testing.assert_close(loss.detach(), mu)
    assert ce.item() == 0 and delta.item() == 0
    torch.testing.assert_close(counts, torch.zeros_like(counts))
    loss.backward()
    assert torch.isfinite(c.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_complete_copy_paste_torch_compile_fullgraph():
    torch.manual_seed(71)
    b, t, d, v = 2, 5, 32, 37
    inputs = [
        torch.randn(b, t, d, device="cuda", requires_grad=True),
        torch.randn(v, d, device="cuda", requires_grad=True),
        torch.randn(b, t, d, device="cuda", requires_grad=True),
        torch.randn(b, t, d, device="cuda", requires_grad=True),
        torch.randn(b, t, device="cuda", requires_grad=True),
        torch.randn(b, t, device="cuda", requires_grad=True),
    ]
    ids = torch.randint(v, (b, t), device="cuda")
    mask = torch.ones_like(ids, dtype=torch.bool)
    targets = torch.randint(v, (b, t, 1), device="cuda")
    groups = torch.rand(b, t, device="cuda")

    def run(h, c, query, keys, gate, sink):
        return copy_paste_loss(
            h,
            c,
            None,
            query,
            keys,
            ids,
            mask,
            targets,
            gate,
            sink,
            group_fraction=groups,
            use_mile=True,
            use_mu=True,
            need_topk=True,
        )

    eager = run(*inputs)
    eager_grad = torch.autograd.grad(eager[0], inputs, retain_graph=True)
    compiled = torch.compile(run, backend="inductor", fullgraph=True)(*inputs)
    compiled_grad = torch.autograd.grad(compiled[0], inputs)
    for actual, expected in zip(compiled, eager):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual, expected in zip(compiled_grad, eager_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
