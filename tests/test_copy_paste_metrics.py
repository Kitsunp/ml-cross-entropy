"""Copy diagnostics are token-weighted and do not affect loss gradients."""

import math

import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.copy_paste import _copy_paste_sparse, _copy_paste_stats, copy_paste_loss
from cut_cross_entropy.utils import TensorInfo


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("patch_size", [1, 4])
@pytest.mark.parametrize("all_invalid", [False, True])
@pytest.mark.parametrize("reduction", ["lock", "split"])
def test_copy_metric_sums_match_dense_reference(patch_size, all_invalid, reduction, monkeypatch):
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    torch.manual_seed(37)
    batch, rows, dimensions, vocab = 2, 3, 16, 13
    source = rows * patch_size
    slots = patch_size
    hidden = torch.randn(batch, rows, dimensions, device="cuda", requires_grad=True)
    classifier = torch.randn(vocab, dimensions, device="cuda", requires_grad=True)
    query = torch.randn_like(hidden, requires_grad=True)
    keys = torch.randn(batch, source, dimensions, device="cuda", requires_grad=True)
    gate = torch.randn(batch, rows, device="cuda", requires_grad=True)
    sink = torch.randn(batch, rows, device="cuda", requires_grad=True)
    ids = torch.randint(0, vocab - 1, (batch, source), device="cuda")
    ids[:, 1] = ids[:, 0]  # repeated source IDs must contribute only once per token
    source_mask = torch.ones_like(ids, dtype=torch.bool)
    source_mask[0, -1] = False
    targets = torch.randint(0, vocab, (batch, rows, slots), device="cuda")
    targets[0, 0, 0] = ids[0, 0]
    targets[1, 0, 0] = vocab - 1  # absent from the source
    targets[0, -1, -1] = -100
    if all_invalid:
        targets.fill_(-100)

    args = (hidden, classifier, None, query, keys, ids, source_mask, targets, gate, sink)
    options = dict(patch_size=patch_size, use_mile=True, use_mu=True, need_topk=patch_size == 1)
    ordinary = copy_paste_loss(*args, **options)
    measured = copy_paste_loss(*args, **options, return_copy_metrics=True)
    for actual, expected in zip(measured[:5], ordinary):
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    variables = (hidden, classifier, query, keys, gate, sink)
    actual_grads = torch.autograd.grad(measured[0], variables, retain_graph=True)
    reference_grads = torch.autograd.grad(ordinary[0], variables)
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    logits = F.linear(hidden.detach(), classifier.detach())
    lm = logits.softmax(-1)
    c = torch.sigmoid(gate.detach() - logits.logsumexp(-1))
    scores = torch.matmul(query.detach(), keys.detach().transpose(-1, -2)) / math.sqrt(dimensions)
    positions = torch.arange(source, device="cuda")
    allowed = source_mask[:, None, :] & (
        positions[None, None, :]
        < (torch.arange(rows, device="cuda")[None, :, None] + 1) * patch_size
    )
    alpha = torch.cat(
        (sink.detach()[..., None], scores.masked_fill(~allowed, -torch.inf)), -1
    ).softmax(-1)
    s = alpha[..., 0]
    valid = (targets >= 0) & (targets < vocab)
    safe = targets.clamp(0, vocab - 1)
    matches = allowed[:, :, None, :] & (ids[:, None, None, :] == safe[..., None])
    copy_target = (alpha[..., 1:].unsqueeze(-2) * matches).sum(-1)
    base_target = lm.gather(-1, safe)
    mixture_target = (1 - c + c * s)[..., None] * base_target + c[..., None] * copy_target
    responsibility = c[..., None] * copy_target / mixture_target
    count_per_row = valid.sum(-1)
    expected = torch.stack(
        (
            (c * count_per_row).sum(),
            (s * count_per_row).sum(),
            (c * (1 - s) * count_per_row).sum(),
            torch.where(valid, responsibility, 0.0).sum(),
            (matches.any(-1) & valid).sum(),
            valid.sum(),
        )
    ).float()
    torch.testing.assert_close(measured[5], expected, rtol=1e-4, atol=2e-5)
    assert not measured[5].requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_copy_metrics_compile_and_empty_rows():
    device = "cuda"
    h = torch.randn(1, 2, 16, device=device, requires_grad=True)
    c = torch.randn(19, 16, device=device, requires_grad=True)
    q = torch.randn_like(h, requires_grad=True)
    k = torch.randn_like(h, requires_grad=True)
    ids = torch.tensor([[1, 2]], device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    targets = torch.tensor([[[1], [7]]], device=device)
    gate = torch.randn(1, 2, device=device, requires_grad=True)
    sink = torch.randn(1, 2, device=device, requires_grad=True)

    def run(h, c, q, k, gate, sink):
        return copy_paste_loss(
            h,
            c,
            None,
            q,
            k,
            ids,
            mask,
            targets,
            gate,
            sink,
            use_mile=True,
            need_topk=True,
            return_copy_metrics=True,
        )

    expected = run(h, c, q, k, gate, sink)
    actual = torch.compile(run, fullgraph=True)(h, c, q, k, gate, sink)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
    actual[0].backward()
    assert all(torch.isfinite(x.grad).all() for x in (h, c, q, k, gate, sink))

    empty = copy_paste_loss(
        h[:, :0],
        c,
        None,
        q[:, :0],
        k,
        ids,
        mask,
        targets[:, :0],
        gate[:, :0],
        sink[:, :0],
        return_copy_metrics=True,
    )
    assert empty[5].shape == (6,)
    assert torch.count_nonzero(empty[5]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("pattern", ["zero", "partial", "nonzero", "nan"])
def test_zero_row_skip_preserves_dense_lse_gradient(pattern):
    torch.manual_seed(91)
    rows, dimensions, vocab = 257, 32, 131
    h = torch.randn(rows, dimensions, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(vocab, dimensions, device="cuda", dtype=torch.bfloat16)
    lse = F.linear(h.float(), c.float()).logsumexp(-1)
    do = torch.randn(rows, device="cuda")
    if pattern == "zero":
        do.zero_()
    elif pattern in ("partial", "nan"):
        do[:128] = 0
        if pattern == "nan":
            do[13] = float("nan")
    options = dict(
        do=do,
        dlse=None,
        e=h,
        e_info=TensorInfo(torch.float32, True),
        c=c,
        c_info=TensorInfo(torch.float32, True),
        bias=None,
        bias_info=None,
        lse=lse,
        mile_weight=None,
        patch_target_weight=None,
        mu=None,
        mu_vocab_size=None,
        mu_loss_lambda=None,
        valids=None,
        softcap=None,
        filter_eps=None,
        targets=None,
        accum_e_fp32=True,
        accum_c_fp32=True,
        filter_e_grad=False,
        filter_c_grad=False,
    )
    baseline = cce_backward_kernel(**options, skip_zero_rows=False)
    candidate = cce_backward_kernel(**options, skip_zero_rows=True)
    for actual, expected in zip(candidate[:2], baseline[:2]):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reduction", ["lock", "split"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [128, 160])
def test_empty_target_skip_keeps_supervised_cce_statistics(reduction, dtype, rows, monkeypatch):
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    torch.manual_seed(42)
    h = torch.randn(2, rows, 32, device="cuda", dtype=dtype)
    c = torch.randn(101, 32, device="cuda", dtype=dtype)
    targets = torch.randint(0, 101, (2, rows, 4), device="cuda")
    targets[:, : rows // 2] = -100  # entire kernel tiles with no valid target
    targets[:, 3 * rows // 4 :, 1:] = -100  # partially populated patch rows
    reference = _copy_paste_stats(h, c, targets, None, True, True, False)
    skipped = _copy_paste_stats(h, c, targets, None, True, True, True)
    valid = (targets >= 0).any(-1)
    for actual, expected in zip(skipped, reference):
        if actual.dtype in (torch.uint32, torch.uint64):
            signed = torch.int32 if actual.dtype == torch.uint32 else torch.int64
            actual, expected = actual.view(signed), expected.view(signed)
        torch.testing.assert_close(actual[valid], expected[valid], rtol=1e-5, atol=1e-5)
    assert torch.isfinite(skipped[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("patch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compacted_loss_and_gradients_with_partial_rows(patch_size, dtype):
    torch.manual_seed(105)
    batch, rows, d, vocab = 2, 160, 32, 101
    source = rows * patch_size
    h = (torch.randn(batch, rows, d, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    c = (torch.randn(vocab, d, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    q = (torch.randn_like(h) * 0.2).requires_grad_()
    k = (torch.randn(batch, source, d, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    gate = torch.randn(batch, rows, device="cuda", requires_grad=True)
    sink = torch.randn(batch, rows, device="cuda", requires_grad=True)
    ids = torch.randint(vocab, (batch, source), device="cuda")
    ids[:, 1] = ids[:, 0]
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, -20:] = False
    targets = torch.randint(vocab, (batch, rows, patch_size), device="cuda")
    targets[:, :80] = -100
    if patch_size > 1:
        targets[:, 120:, 1:] = -100
    group = torch.rand(batch, rows, device="cuda")
    variables = (h, c, q, k, gate, sink)
    actual = copy_paste_loss(
        h,
        c,
        None,
        q,
        k,
        ids,
        mask,
        targets,
        gate,
        sink,
        patch_size=patch_size,
        group_fraction=group,
        use_mile=True,
        use_mu=True,
        need_topk=patch_size == 1,
        return_copy_metrics=True,
    )

    logits = F.linear(h, c).float()
    cgate = torch.sigmoid(gate - logits.logsumexp(-1))
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(d)
    allowed = mask[:, None, :] & (
        torch.arange(source, device="cuda")[None, None, :]
        < (torch.arange(rows, device="cuda")[None, :, None] + 1) * patch_size
    )
    alpha = torch.cat((sink[..., None], scores.masked_fill(~allowed, -torch.inf)), -1).softmax(-1)
    p = (1 - cgate + cgate * alpha[..., 0])[..., None] * logits.softmax(-1)
    p = p.scatter_add(-1, ids[:, None, :].expand(-1, rows, -1), cgate[..., None] * alpha[..., 1:])
    valid = (targets >= 0) & (targets < vocab)
    safe = targets.clamp(0, vocab - 1)
    nll = torch.where(valid, -p.gather(-1, safe).log(), 0.0)
    entropy = -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1).detach()
    raw = 1 + entropy
    row_count = valid.sum(-1).float()
    group_scale = (row_count * group).sum() / (row_count * group * raw).sum().clamp_min(1e-20)
    other = 1 - group
    other_scale = (row_count * other).sum() / (row_count * other * raw).sum().clamp_min(1e-20)
    weights = raw * (group * group_scale + other * other_scale)
    expected_ce = nll.sum() / valid.sum().clamp_min(1)
    expected_weighted = (nll * weights[..., None]).sum() / valid.sum().clamp_min(1)
    expected_mu = 1e-4 * c.float().mean(0).square().sum()
    expected = expected_weighted + expected_mu
    tolerance = 0.03 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(actual[0], expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual[1], expected_ce, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        actual[2], expected_weighted - expected_ce, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(actual[3], expected_mu, rtol=tolerance, atol=tolerance)
    if patch_size == 1:
        y = safe[..., 0]
        target_probability = p.gather(-1, y[..., None])
        token_ids = torch.arange(vocab, device="cuda")
        rank = (
            (p > target_probability) | ((p == target_probability) & (token_ids < y[..., None]))
        ).sum(-1)
        target_valid = valid[..., 0]
        expected_topk = torch.stack(
            (
                (target_valid & (rank == 0)).sum(),
                (target_valid & (rank < 3)).sum(),
                torch.zeros((), device="cuda"),
                target_valid.sum(),
                torch.zeros((), device="cuda"),
            )
        ).float()
        torch.testing.assert_close(actual[4], expected_topk, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual[0], variables, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, variables)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad.float(), expected_grad.float(), rtol=tolerance, atol=tolerance * 0.01
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("key_dtype", [torch.uint32, torch.uint64])
@pytest.mark.parametrize("lse_value", [float("inf"), -float("inf"), float("nan"), 0.0])
def test_nonfinite_vocabulary_never_counts_token_zero_as_a_topk_hit(key_dtype, lse_value):
    device = "cuda"
    ids = torch.tensor([[0, 1]], device=device)
    scores = torch.zeros((1, 1, 2), device=device)
    context_logits = torch.zeros_like(scores)
    mask = torch.ones_like(ids, dtype=torch.bool)
    targets = torch.tensor([[[0]]], device=device)
    lse = torch.tensor([[lse_value]], device=device)
    negative = torch.zeros_like(targets, dtype=torch.float32)
    gate = sink = torch.zeros((1, 1), device=device)
    # CCE marks a nonfinite winner with an all-ones packed key.
    signed_dtype = torch.int32 if key_dtype == torch.uint32 else torch.int64
    bad_keys = torch.full((1, 1, 3), -1, device=device, dtype=signed_dtype).view(key_dtype)
    _, _, counts = _copy_paste_sparse(
        scores,
        context_logits,
        ids,
        mask,
        targets,
        negative,
        lse,
        torch.zeros_like(lse),
        gate,
        sink,
        ids.argsort(-1),
        bad_keys,
        7,
        1,
        False,
        True,
        True,
    )
    torch.testing.assert_close(
        counts.sum(0), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], device=device)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reduction", ["lock", "split"])
def test_nonfinite_vocab_is_reported_by_the_complete_loss(reduction, monkeypatch):
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    h = torch.ones((1, 1, 16), device="cuda")
    c = torch.zeros((7, 16), device="cuda")
    c[1, 0] = float("inf")
    ids = torch.tensor([[0]], device="cuda")
    target = torch.tensor([[[0]]], device="cuda")
    gate = sink = torch.zeros((1, 1), device="cuda")
    result = copy_paste_loss(
        h,
        c,
        None,
        h,
        h,
        ids,
        torch.ones_like(ids, dtype=torch.bool),
        target,
        gate,
        sink,
        need_topk=True,
    )
    torch.testing.assert_close(result[4], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_value", [None, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("gamma", [100.0, 1000.0])
def test_ignored_rows_cannot_change_high_gamma_mile_normalization(group_value, gamma):
    d, vocab = 16, 7
    h = torch.tensor([[[3.0] * d, [-3.0] * d]], device="cuda")
    c = torch.full((vocab, d), -3.0, device="cuda")
    c[0] = 3.0
    q = torch.zeros_like(h)
    ids = torch.zeros((1, 2), device="cuda", dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    targets = torch.tensor([[[1], [-100]]], device="cuda")
    gate = torch.tensor([[-100.0, 0.0]], device="cuda")
    sink = torch.zeros_like(gate)
    group = None if group_value is None else torch.full_like(gate, group_value)
    loss, ce, delta, _, _ = copy_paste_loss(
        h,
        c,
        None,
        q,
        q,
        ids,
        mask,
        targets,
        gate,
        sink,
        use_mile=True,
        mile_gamma=gamma,
        group_fraction=group,
    )
    # A single valid prediction must have normalized MiLe weight exactly one.
    torch.testing.assert_close(loss, ce, atol=1e-4, rtol=1e-6)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-4, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_high_gamma_normalizes_disjoint_mile_groups_separately():
    d, vocab = 16, 7
    h = torch.tensor([[[3.0] * d, [0.0] * d]], device="cuda", requires_grad=True)
    c = torch.full((vocab, d), -3.0, device="cuda")
    c[0] = 3.0
    c.requires_grad_()
    q = torch.zeros_like(h)
    ids = torch.zeros((1, 2), device="cuda", dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    targets = torch.ones((1, 2, 1), device="cuda", dtype=torch.long)
    gate = torch.full((1, 2), -100.0, device="cuda")
    group = torch.tensor([[1.0, 0.0]], device="cuda")
    loss, ce, delta, _, _ = copy_paste_loss(
        h,
        c,
        None,
        q,
        q,
        ids,
        mask,
        targets,
        gate,
        torch.zeros_like(gate),
        use_mile=True,
        mile_gamma=1000.0,
        group_fraction=group,
    )
    # Each population contains one token, so each normalized weight is one.
    torch.testing.assert_close(loss, ce, atol=1e-4, rtol=1e-6)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-4, rtol=0)
    grads = torch.autograd.grad(loss, (h, c))
    assert all(torch.isfinite(gradient).all() for gradient in grads)
