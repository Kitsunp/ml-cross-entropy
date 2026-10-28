"""Sparse, causal Copy-Paste LM reductions for CCE.

The full mixture is ``a * softmax(logits) + c * copy`` where ``a=1-c+c*s``
and ``s`` is the copy sink probability.  Entropy and rank need only the base
distribution's sufficient statistics and its top three classes, plus classes
which actually receive copy mass.  Repeated source IDs are combined before
entropy/ranking; no query-by-vocabulary matrix is constructed.

The lower-level operation takes already-computed vocabulary statistics and
pointer scores.  Its backward differentiates the exact target log-probability;
entropy and hit counts are detached diagnostics, matching MiLe's contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import TensorInfo


@triton.jit
def _cp_logaddexp(x, y):
    maximum = tl.maximum(x, y)
    finite_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    value = finite_max + tl.log(tl.exp(x - finite_max) + tl.exp(y - finite_max))
    return tl.where(maximum == -float("inf"), -float("inf"), value)


@triton.jit
def _cp_segment_sum(left, left_head, right, right_head):
    return tl.where(right_head, right, left + right), left_head | right_head


@triton.jit
def _cp_softplus(x):
    return tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))


@triton.jit
def _cp_decode_key(key, PACKED32: tl.constexpr, BF16_SCORE: tl.constexpr):
    if PACKED32:
        token = (0xFFFF - (key & 0xFFFF)).to(tl.int64)
        ordered = (key >> 16).to(tl.uint32)
        bits = tl.where((ordered & 0x8000) != 0, ordered ^ 0x8000, (~ordered) & 0xFFFF)
        if BF16_SCORE:
            score = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            score = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        valid = (key != 0) & (key != 0xFFFFFFFF)
    else:
        token = (0xFFFFFFFF - (key & 0xFFFFFFFF)).to(tl.int64)
        ordered = (key >> 32).to(tl.uint32)
        bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
        score = bits.to(tl.uint32).to(tl.float32, bitcast=True)
        valid = (key != 0) & (key != 0xFFFFFFFFFFFFFFFF)
    return token, score, valid


@triton.jit
def _copy_paste_sparse_forward_kernel(
    Scores,
    ContextLogits,
    SourceIds,
    SourceMask,
    Targets,
    NegCorrect,
    LSE,
    MeanLogit,
    Gate,
    Sink,
    SortedIndices,
    BaseTopKeys,
    NLL,
    Entropy,
    Counts,
    ROWS: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    PATCH_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NEED_ENTROPY: tl.constexpr,
    NEED_TOPK: tl.constexpr,
    PACKED32: tl.constexpr,
    BF16_SCORE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // T
    position = row % T
    i = tl.arange(0, BLOCK_S)
    token = tl.load(SourceIds + batch * S + i, i < S, 0).to(tl.int64)
    allowed = (i < S) & (i < (position + 1) * PATCH_SIZE)
    allowed &= tl.load(SourceMask + batch * S + i, i < S, 0).to(tl.int1)
    score = tl.load(Scores + row * S + i, i < S, 0).to(tl.float32)
    score = tl.where(allowed, score, -float("inf"))
    sink = tl.load(Sink + row).to(tl.float32)
    maximum = tl.maximum(tl.max(score, 0), sink)
    normalizer = maximum + tl.log(tl.sum(tl.exp(score - maximum), 0) + tl.exp(sink - maximum))
    lse = tl.load(LSE + row).to(tl.float32)
    gate_delta = tl.load(Gate + row).to(tl.float32) - lse
    log_c = -_cp_softplus(-gate_delta)
    log_not_c = -_cp_softplus(gate_delta)
    log_a = _cp_logaddexp(log_not_c, log_c + sink - normalizer)

    target_prob = 0.0
    target_token = tl.full((), -1, tl.int64)
    target_valid = tl.full((), False, tl.int1)
    for slot in tl.static_range(K):
        y = tl.load(Targets + row * K + slot).to(tl.int64)
        valid = (y >= 0) & (y < V)
        negative = tl.load(NegCorrect + row * K + slot).to(tl.float32)
        match = allowed & (token == y)
        match_max = tl.max(tl.where(match, score, -float("inf")), 0)
        safe_match_max = tl.where(match_max == -float("inf"), 0.0, match_max)
        match_sum = tl.sum(tl.where(match, tl.exp(score - safe_match_max), 0.0), 0)
        log_r = tl.where(
            match_sum > 0.0, tl.log(match_sum) + safe_match_max - normalizer, -float("inf")
        )
        base_log = log_a - negative - lse
        log_probability = _cp_logaddexp(base_log, log_c + log_r)
        tl.store(NLL + row * K + slot, tl.where(valid, -log_probability, 0.0))
        if slot == 0:
            target_prob = tl.exp(base_log) + tl.exp(log_c + log_r)
            target_token = y
            target_valid = valid

    if NEED_ENTROPY or NEED_TOPK:
        sorted_pos = tl.load(SortedIndices + batch * S + i, i < S, 0).to(tl.int64)
        sorted_ids = tl.load(SourceIds + batch * S + sorted_pos, i < S, 0).to(tl.int64)
        previous_pos = tl.load(SortedIndices + batch * S + i - 1, (i > 0) & (i < S), 0).to(tl.int64)
        next_pos = tl.load(SortedIndices + batch * S + i + 1, i + 1 < S, 0).to(tl.int64)
        previous_id = tl.load(SourceIds + batch * S + previous_pos, (i > 0) & (i < S), -1).to(
            tl.int64
        )
        next_id = tl.load(SourceIds + batch * S + next_pos, i + 1 < S, -1).to(tl.int64)
        sorted_allowed = (i < S) & (sorted_pos < (position + 1) * PATCH_SIZE)
        sorted_allowed &= tl.load(SourceMask + batch * S + sorted_pos, i < S, 0).to(tl.int1)
        sorted_score = tl.load(Scores + row * S + sorted_pos, i < S, 0).to(tl.float32)
        alpha = tl.where(sorted_allowed, tl.exp(sorted_score - normalizer), 0.0)
        segment_start = (i == 0) | (sorted_ids != previous_id)
        segment_last = (i < S) & ((i + 1 == S) | (sorted_ids != next_id))
        group_alpha, _ = tl.associative_scan((alpha, segment_start), 0, _cp_segment_sum)
        copy_mass = tl.exp(log_c) * group_alpha
        source_logits = tl.load(ContextLogits + row * S + sorted_pos, i < S, 0).to(tl.float32)
        base_mass = tl.exp(log_a + source_logits - lse)
        mixed = base_mass + copy_mass
        active = segment_last & (copy_mass > 0.0)
        if NEED_ENTROPY:
            mean = tl.load(MeanLogit + row).to(tl.float32)
            base_entropy = tl.minimum(tl.maximum(lse - mean, 0.0), tl.log(float(V)))
            before = tl.where(base_mass > 0.0, base_mass * tl.log(base_mass), 0.0)
            after = tl.where(mixed > 0.0, mixed * tl.log(mixed), 0.0)
            entropy = tl.exp(log_a) * (base_entropy - log_a) + tl.sum(
                tl.where(active, before - after, 0.0), 0
            )
            tl.store(Entropy + row, tl.minimum(tl.maximum(entropy, 0.0), tl.log(float(V))))
        if NEED_TOPK:
            rank = tl.full((), 0, tl.int32)
            excluded = tl.full((BLOCK_S,), False, tl.int1)
            for k in tl.static_range(3):
                key = tl.load(BaseTopKeys + row * 3 + k)
                candidate, candidate_logit, key_valid = _cp_decode_key(key, PACKED32, BF16_SCORE)
                matching = active & (sorted_ids == candidate)
                candidate_copy = tl.sum(tl.where(matching, copy_mass, 0.0), 0)
                candidate_prob = tl.exp(log_a + candidate_logit - lse) + candidate_copy
                rank += (
                    key_valid
                    & (
                        (candidate_prob > target_prob)
                        | ((candidate_prob == target_prob) & (candidate < target_token))
                    )
                ).to(tl.int32)
                excluded |= sorted_ids == candidate
            rank += tl.sum(
                (
                    active
                    & ~excluded
                    & (
                        (mixed > target_prob)
                        | ((mixed == target_prob) & (sorted_ids < target_token))
                    )
                ).to(tl.int32),
                0,
            )
            finite = tl.abs(target_prob) < float("inf")
            keep = target_valid & finite
            tl.store(Counts + row * 5, (keep & (rank == 0)).to(tl.float32))
            tl.store(Counts + row * 5 + 1, (keep & (rank < 3)).to(tl.float32))
            tl.store(Counts + row * 5 + 2, 0.0)
            tl.store(Counts + row * 5 + 3, target_valid.to(tl.float32))
            tl.store(Counts + row * 5 + 4, (target_valid & ~finite).to(tl.float32))


@triton.jit
def _copy_paste_sparse_backward_kernel(
    Scores,
    SourceIds,
    SourceMask,
    Targets,
    NegCorrect,
    LSE,
    Gate,
    Sink,
    dNLL,
    dScores,
    dNegCorrect,
    dLSE,
    dGate,
    dSink,
    T: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    PATCH_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // T
    position = row % T
    i = tl.arange(0, BLOCK_S)
    token = tl.load(SourceIds + batch * S + i, i < S, 0).to(tl.int64)
    allowed = (i < S) & (i < (position + 1) * PATCH_SIZE)
    allowed &= tl.load(SourceMask + batch * S + i, i < S, 0).to(tl.int1)
    score = tl.load(Scores + row * S + i, i < S, 0).to(tl.float32)
    score = tl.where(allowed, score, -float("inf"))
    sink = tl.load(Sink + row).to(tl.float32)
    maximum = tl.maximum(tl.max(score, 0), sink)
    normalizer = maximum + tl.log(tl.sum(tl.exp(score - maximum), 0) + tl.exp(sink - maximum))
    alpha = tl.exp(score - normalizer)
    s = tl.exp(sink - normalizer)
    lse = tl.load(LSE + row).to(tl.float32)
    gate_delta = tl.load(Gate + row).to(tl.float32) - lse
    log_c = -_cp_softplus(-gate_delta)
    log_not_c = -_cp_softplus(gate_delta)
    log_a = _cp_logaddexp(log_not_c, log_c + sink - normalizer)
    c = tl.exp(log_c)
    a_sink_fraction = tl.exp(log_c + sink - normalizer - log_a)
    dlog_a_dgate = -tl.exp(log_not_c + log_c - log_a) * tl.maximum(1.0 - s, 0.0)
    dscores = tl.full((BLOCK_S,), 0.0, tl.float32)
    dlse = tl.full((), 0.0, tl.float32)
    dgate = tl.full((), 0.0, tl.float32)
    dsink = tl.full((), 0.0, tl.float32)
    for slot in tl.static_range(K):
        y = tl.load(Targets + row * K + slot).to(tl.int64)
        valid = (y >= 0) & (y < V)
        negative = tl.load(NegCorrect + row * K + slot).to(tl.float32)
        upstream = tl.where(valid, tl.load(dNLL + row * K + slot).to(tl.float32), 0.0)
        match = allowed & (token == y)
        match_max = tl.max(tl.where(match, score, -float("inf")), 0)
        safe_match_max = tl.where(match_max == -float("inf"), 0.0, match_max)
        match_sum = tl.sum(tl.where(match, tl.exp(score - safe_match_max), 0.0), 0)
        log_match = tl.where(match_sum > 0.0, tl.log(match_sum) + safe_match_max, -float("inf"))
        log_r = log_match - normalizer
        base_log = log_a - negative - lse
        log_probability = _cp_logaddexp(base_log, log_c + log_r)
        base_responsibility = tl.exp(base_log - log_probability)
        copy_responsibility = tl.exp(log_c + log_r - log_probability)
        target_alpha = tl.where(
            match, tl.exp(score - tl.where(match_sum > 0.0, log_match, 0.0)), 0.0
        )
        dscores += upstream * (
            alpha * (base_responsibility * a_sink_fraction + copy_responsibility)
            - copy_responsibility * target_alpha
        )
        dlse += upstream * (
            base_responsibility * (1.0 + dlog_a_dgate) + copy_responsibility * (1.0 - c)
        )
        dgate -= upstream * (base_responsibility * dlog_a_dgate + copy_responsibility * (1.0 - c))
        dsink += upstream * (
            -base_responsibility * a_sink_fraction * (1.0 - s) + copy_responsibility * s
        )
        tl.store(dNegCorrect + row * K + slot, upstream * base_responsibility)
    tl.store(dScores + row * S + i, dscores, i < S)
    tl.store(dLSE + row, dlse)
    tl.store(dGate + row, dgate)
    tl.store(dSink + row, dsink)


@torch.library.custom_op(
    "cut_cross_entropy::copy_paste_sparse", mutates_args=(), device_types="cuda"
)
def _copy_paste_sparse(
    scores: torch.Tensor,
    context_logits: torch.Tensor,
    source_ids: torch.Tensor,
    source_mask: torch.Tensor,
    targets: torch.Tensor,
    neg_correct: torch.Tensor,
    lse: torch.Tensor,
    mean_logit: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    sorted_indices: torch.Tensor,
    base_topkeys: torch.Tensor,
    vocab_size: int,
    patch_size: int,
    need_entropy: bool,
    need_topk: bool,
    bf16_score: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, time, source = scores.shape
    slots = targets.shape[-1]
    nll = torch.empty_like(neg_correct, dtype=torch.float32)
    entropy = torch.zeros_like(lse, dtype=torch.float32)
    counts = lse.new_zeros((batch * time, 5), dtype=torch.float32)
    if batch * time:
        _copy_paste_sparse_forward_kernel[(batch * time,)](
            scores,
            context_logits,
            source_ids,
            source_mask,
            targets,
            neg_correct,
            lse,
            mean_logit,
            gate,
            sink,
            sorted_indices,
            base_topkeys,
            nll,
            entropy,
            counts,
            batch * time,
            time,
            source,
            slots,
            vocab_size,
            patch_size,
            triton.next_power_of_2(source),
            need_entropy,
            need_topk,
            base_topkeys.dtype == torch.uint32,
            bf16_score,
        )
    return nll, entropy, counts


@_copy_paste_sparse.register_fake
def _copy_paste_sparse_fake(
    scores,
    context_logits,
    source_ids,
    source_mask,
    targets,
    neg_correct,
    lse,
    mean_logit,
    gate,
    sink,
    sorted_indices,
    base_topkeys,
    vocab_size,
    patch_size,
    need_entropy,
    need_topk,
    bf16_score,
):
    return (
        torch.empty_like(neg_correct, dtype=torch.float32),
        torch.empty_like(lse, dtype=torch.float32),
        lse.new_empty((scores.shape[0] * scores.shape[1], 5), dtype=torch.float32),
    )


@torch.library.custom_op(
    "cut_cross_entropy::copy_paste_sparse_backward", mutates_args=(), device_types="cuda"
)
def _copy_paste_sparse_backward(
    scores: torch.Tensor,
    source_ids: torch.Tensor,
    source_mask: torch.Tensor,
    targets: torch.Tensor,
    neg_correct: torch.Tensor,
    lse: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    grad_nll: torch.Tensor,
    vocab_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, time, source = scores.shape
    d_scores = torch.empty_like(scores)
    d_negative = torch.empty_like(neg_correct)
    d_lse = torch.empty_like(lse)
    d_gate = torch.empty_like(gate)
    d_sink = torch.empty_like(sink)
    if batch * time:
        _copy_paste_sparse_backward_kernel[(batch * time,)](
            scores,
            source_ids,
            source_mask,
            targets,
            neg_correct,
            lse,
            gate,
            sink,
            grad_nll.contiguous(),
            d_scores,
            d_negative,
            d_lse,
            d_gate,
            d_sink,
            time,
            source,
            targets.shape[-1],
            vocab_size,
            patch_size,
            triton.next_power_of_2(source),
        )
    return d_scores, d_negative, d_lse, d_gate, d_sink


@_copy_paste_sparse_backward.register_fake
def _copy_paste_sparse_backward_fake(
    scores,
    source_ids,
    source_mask,
    targets,
    neg_correct,
    lse,
    gate,
    sink,
    grad_nll,
    vocab_size,
    patch_size,
):
    return (
        torch.empty_like(scores),
        torch.empty_like(neg_correct),
        torch.empty_like(lse),
        torch.empty_like(gate),
        torch.empty_like(sink),
    )


def _copy_paste_sparse_setup(ctx, inputs, output):
    (
        scores,
        _context_logits,
        source_ids,
        source_mask,
        targets,
        neg_correct,
        lse,
        _mean_logit,
        gate,
        sink,
        _sorted_indices,
        _base_topkeys,
        vocab_size,
        patch_size,
        _need_entropy,
        _need_topk,
        _bf16_score,
    ) = inputs
    ctx.save_for_backward(scores, source_ids, source_mask, targets, neg_correct, lse, gate, sink)
    ctx.vocab_size = vocab_size
    ctx.patch_size = patch_size
    ctx.mark_non_differentiable(output[1], output[2])


def _copy_paste_sparse_autograd(ctx, grad_nll, _grad_entropy, _grad_counts):
    scores, source_ids, source_mask, targets, neg_correct, lse, gate, sink = ctx.saved_tensors
    d_scores, d_negative, d_lse, d_gate, d_sink = _copy_paste_sparse_backward(
        scores,
        source_ids,
        source_mask,
        targets,
        neg_correct,
        lse,
        gate,
        sink,
        grad_nll,
        ctx.vocab_size,
        ctx.patch_size,
    )
    return (
        d_scores,
        None,
        None,
        None,
        None,
        d_negative,
        d_lse,
        None,
        d_gate,
        d_sink,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_copy_paste_sparse.register_autograd(
    _copy_paste_sparse_autograd, setup_context=_copy_paste_sparse_setup
)


@triton.jit
def _copy_paste_target_backward_kernel(
    E,
    C,
    Targets,
    dNegative,
    dE,
    dC,
    dBias,
    N: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    embedding = tl.load(
        E + rows[:, None] * D + dims[None, :], (rows[:, None] < N) & (dims[None, :] < D), 0
    ).to(tl.float32)
    delta_e = tl.full((BLOCK_N, BLOCK_D), 0.0, tl.float32)
    for slot in tl.static_range(K):
        target = tl.load(Targets + rows * K + slot, rows < N, -1).to(tl.int64)
        valid = (rows < N) & (target >= 0) & (target < V)
        negative_grad = tl.load(dNegative + rows * K + slot, rows < N, 0).to(tl.float32)
        value = -tl.where(valid, negative_grad, 0.0)
        c = tl.load(
            C + target[:, None] * D + dims[None, :], valid[:, None] & (dims[None, :] < D), 0
        ).to(tl.float32)
        delta_e += value[:, None] * c
        tl.atomic_add(
            dC + target[:, None] * D + dims[None, :],
            value[:, None] * embedding,
            valid[:, None] & (dims[None, :] < D),
            sem="relaxed",
        )
        if HAS_BIAS:
            if tl.program_id(1) == 0:
                tl.atomic_add(dBias + target, value, valid, sem="relaxed")
    tl.store(
        dE + rows[:, None] * D + dims[None, :],
        tl.load(
            dE + rows[:, None] * D + dims[None, :], (rows[:, None] < N) & (dims[None, :] < D), 0
        )
        + delta_e,
        (rows[:, None] < N) & (dims[None, :] < D),
    )


@torch.library.custom_op(
    "cut_cross_entropy::copy_paste_stats", mutates_args=(), device_types="cuda"
)
def _copy_paste_stats(
    hidden: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    need_entropy: bool,
    need_topk: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, time, dimensions = hidden.shape
    rows = batch * time
    slots = targets.shape[-1]
    result = cce_lse_forward_kernel(
        hidden.contiguous().reshape(rows, dimensions),
        classifier.contiguous(),
        bias,
        targets=targets.contiguous().reshape(rows, slots),
        return_mean_logit=need_entropy,
        return_topk=3 if need_topk else False,
        return_topk_keys=need_topk,
    )
    assert result.neg_correct_logit is not None
    mean = (
        result.mean_logit.reshape(batch, time)
        if need_entropy
        else hidden.new_empty(0, dtype=torch.float32)
    )
    keys = (
        result.topk_keys.reshape(batch, time, 3)
        if need_topk
        else hidden.new_empty(0, dtype=torch.uint64)
    )
    return (
        result.lse.reshape(batch, time),
        result.neg_correct_logit.reshape(batch, time, slots),
        mean,
        keys,
    )


@_copy_paste_stats.register_fake
def _copy_paste_stats_fake(hidden, classifier, targets, bias, need_entropy, need_topk):
    batch, time, _ = hidden.shape
    dtype = (
        torch.uint32
        if hidden.dtype in (torch.float16, torch.bfloat16)
        and classifier.shape[0] <= 65536
        and (bias is None or bias.dtype == hidden.dtype)
        else torch.uint64
    )
    return (
        hidden.new_empty((batch, time), dtype=torch.float32),
        hidden.new_empty(targets.shape, dtype=torch.float32),
        hidden.new_empty((batch, time) if need_entropy else (0,), dtype=torch.float32),
        hidden.new_empty(
            (batch, time, 3) if need_topk else (0,), dtype=dtype if need_topk else torch.uint64
        ),
    )


@torch.library.custom_op(
    "cut_cross_entropy::copy_paste_stats_backward", mutates_args=(), device_types="cuda"
)
def _copy_paste_stats_backward(
    hidden: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    lse: torch.Tensor,
    grad_lse: torch.Tensor,
    grad_negative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, time, dimensions = hidden.shape
    rows = batch * time
    flat_hidden = hidden.contiguous().reshape(rows, dimensions)
    classifier = classifier.contiguous()
    flat_targets = targets.contiguous().reshape(rows, targets.shape[-1])
    grad_e, grad_c, grad_bias = cce_backward_kernel(
        do=grad_lse.contiguous().reshape(rows),
        dlse=None,
        e=flat_hidden,
        e_info=TensorInfo(torch.float32, True),
        c=classifier,
        c_info=TensorInfo(torch.float32, True),
        bias=bias,
        bias_info=TensorInfo(torch.float32, True) if bias is not None else None,
        lse=lse.contiguous().reshape(rows),
        mile_weight=None,
        patch_target_weight=None,
        mu=None,
        mu_vocab_size=None,
        mu_loss_lambda=None,
        valids=None,
        softcap=None,
        filter_eps=None,
        targets=None,
        grad_scale=1.0,
        accum_e_fp32=True,
        accum_c_fp32=True,
        auto_mixed_grad_accum=False,
        filter_e_grad=False,
        filter_c_grad=False,
    )
    assert grad_e is not None and grad_c is not None
    if rows:
        _copy_paste_target_backward_kernel[(triton.cdiv(rows, 16), triton.cdiv(dimensions, 64))](
            flat_hidden,
            classifier,
            flat_targets,
            grad_negative.contiguous(),
            grad_e,
            grad_c,
            grad_bias if grad_bias is not None else grad_e,
            rows,
            dimensions,
            classifier.shape[0],
            targets.shape[-1],
            bias is not None,
            16,
            64,
        )
    return (
        grad_e.reshape_as(hidden).to(hidden.dtype),
        grad_c.to(classifier.dtype),
        grad_bias.to(bias.dtype) if bias is not None else hidden.new_empty(0),
    )


@_copy_paste_stats_backward.register_fake
def _copy_paste_stats_backward_fake(
    hidden, classifier, targets, bias, lse, grad_lse, grad_negative
):
    return (
        torch.empty_like(hidden),
        torch.empty_like(classifier),
        torch.empty_like(bias) if bias is not None else hidden.new_empty(0),
    )


def _copy_paste_stats_setup(ctx, inputs, output):
    hidden, classifier, targets, bias, _need_entropy, _need_topk = inputs
    ctx.save_for_backward(hidden, classifier, targets, bias, output[0])
    ctx.mark_non_differentiable(output[2], output[3])


def _copy_paste_stats_autograd(ctx, grad_lse, grad_negative, _grad_mean, _grad_keys):
    hidden, classifier, targets, bias, lse = ctx.saved_tensors
    if grad_lse is None:
        grad_lse = torch.zeros_like(lse)
    if grad_negative is None:
        grad_negative = torch.zeros_like(targets, dtype=torch.float32)
    grad_e, grad_c, grad_bias = _copy_paste_stats_backward(
        hidden,
        classifier,
        targets,
        bias,
        lse,
        grad_lse,
        grad_negative,
    )
    return grad_e, grad_c, None, grad_bias if bias is not None else None, None, None


_copy_paste_stats.register_autograd(
    _copy_paste_stats_autograd, setup_context=_copy_paste_stats_setup
)


def copy_paste_loss(
    hidden: torch.Tensor,
    classifier: torch.Tensor,
    bias: torch.Tensor | None,
    query: torch.Tensor,
    keys: torch.Tensor,
    source_ids: torch.Tensor,
    source_mask: torch.Tensor,
    targets: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    *,
    patch_size: int = 1,
    group_fraction: torch.Tensor | None = None,
    mile_gamma: float = 1.0,
    use_mile: bool = False,
    use_mu: bool = False,
    mu_lambda: float = 1e-4,
    need_topk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute mixed NTP loss and detached CE/MiLe/mu/Top-1/3 statistics.

    ``hidden`` and ``query`` describe the predicted rows [B,T,D]; ``keys``
    and source IDs describe the actually observed raw tokens [B,S,...]. For a
    pooled patch of size K, its K source tokens share a key and all become
    visible at the end of that patch. ``targets`` is [B,T,target_slots], with
    negative or out-of-vocabulary IDs ignored. MiLe weights use the exact
    mixture entropy and are detached; fractional MEAP groups are normalized
    using the same valid-target counts as the reference implementation.
    Top-1/3 use slot zero in the token phase, including a padded slot layout.

    The vocabulary projection uses CCE rather than storing [B,T,V] logits.
    Pointer scores and, when requested, sparse candidate logits require
    O(B*T*S) temporary storage. The implementation supports CUDA and normal
    autocast/torch.compile execution; its backward uses FP32 accumulators.
    """
    if hidden.ndim != 3 or query.shape != hidden.shape or classifier.ndim != 2:
        raise ValueError("Copy-Paste requires hidden/query [B,T,D] and classifier [V,D].")
    batch, rows, dimensions = hidden.shape
    if (
        classifier.shape[1] != dimensions
        or targets.ndim != 3
        or targets.shape[:2] != (batch, rows)
        or gate.shape != (batch, rows)
        or sink.shape != (batch, rows)
    ):
        raise ValueError("Copy-Paste target, gate, sink and classifier geometry must match hidden.")
    if (
        keys.ndim != 3
        or keys.shape[0] != batch
        or keys.shape[2] != dimensions
        or source_ids.shape != keys.shape[:2]
        or source_mask.shape != source_ids.shape
    ):
        raise ValueError("Copy-Paste source IDs/mask and keys must share [B,S].")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("Copy-Paste requires a positive integer patch size.")
    if targets.shape[-1] == 0 or classifier.shape[0] == 0 or dimensions == 0:
        raise ValueError("Copy-Paste requires nonempty target, vocabulary and hidden dimensions.")
    if batch and rows and source_ids.shape[1] == 0:
        raise ValueError("Copy-Paste requires a nonempty source for predicted rows.")
    if targets.dtype not in (torch.int32, torch.int64) or source_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Copy-Paste target and source IDs must be integer tensors.")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != classifier.shape[0]):
        raise ValueError("Copy-Paste classifier bias must match the vocabulary.")
    if not math.isfinite(float(mile_gamma)) or mile_gamma < 0:
        raise ValueError("Copy-Paste MiLe gamma must be finite and non-negative.")
    if not math.isfinite(float(mu_lambda)) or mu_lambda < 0:
        raise ValueError("Copy-Paste mu-loss lambda must be finite and non-negative.")
    if need_topk and patch_size != 1:
        raise ValueError("Copy-Paste Top-1/3 metrics require the token phase (patch_size=1).")
    if group_fraction is not None and group_fraction.shape != (batch, rows):
        raise ValueError("Copy-Paste MiLe groups must match the predicted row geometry.")
    if not hidden.is_cuda:
        raise ValueError("The fused Copy-Paste loss requires CUDA.")
    tensors = (
        classifier,
        query,
        keys,
        source_ids,
        source_mask,
        targets,
        gate,
        sink,
        bias,
        group_fraction,
    )
    if any(value is not None and value.device != hidden.device for value in tensors):
        raise ValueError("Copy-Paste inputs must share a CUDA device.")
    if batch == 0 or rows == 0:
        zero = hidden.sum() * 0.0
        mu = float(mu_lambda) * classifier.float().mean(0).square().sum() if use_mu else zero
        return (
            zero + mu,
            zero.detach(),
            zero.detach(),
            mu.detach(),
            zero.new_zeros(5) if need_topk else None,
        )

    compute_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    )
    projected_hidden = hidden.to(compute_dtype).contiguous()
    projected_classifier = classifier.to(compute_dtype).contiguous()
    projected_bias = None if bias is None else bias.to(compute_dtype).contiguous()
    targets = targets.contiguous()
    source_ids = source_ids.to(dtype=torch.long).contiguous()
    source_mask = source_mask.to(dtype=torch.bool).contiguous()
    lse, neg_correct, mean_logit, topkeys = _copy_paste_stats(
        projected_hidden,
        projected_classifier,
        targets,
        projected_bias,
        use_mile,
        need_topk,
    )
    scores = (
        torch.matmul(query.float(), keys.float().transpose(-1, -2)) * (dimensions**-0.5)
    ).contiguous()
    if use_mile or need_topk:
        with torch.no_grad():
            source_classifier = F.embedding(source_ids, projected_classifier)
            context_logits = torch.matmul(
                projected_hidden.detach(), source_classifier.transpose(-1, -2)
            )
            if projected_bias is not None:
                context_logits = context_logits + F.embedding(
                    source_ids, projected_bias[:, None]
                ).transpose(1, 2)
            context_logits = context_logits.contiguous()
            sorted_indices = torch.argsort(source_ids, dim=-1).contiguous()
    else:
        context_logits = hidden.new_empty(0)
        sorted_indices = source_ids.new_empty(0)
    nll, entropy, partial_counts = _copy_paste_sparse(
        scores,
        context_logits,
        source_ids,
        source_mask,
        targets,
        neg_correct,
        lse,
        mean_logit,
        gate.contiguous(),
        sink.contiguous(),
        sorted_indices,
        topkeys,
        classifier.shape[0],
        patch_size,
        use_mile,
        need_topk,
        compute_dtype == torch.bfloat16,
    )
    valid = (targets >= 0) & (targets < classifier.shape[0])
    valid_count = valid.sum().to(torch.float32)
    denominator = valid_count.clamp_min(1.0)
    ce = nll.sum() / denominator
    weighted = ce
    if use_mile:
        log_base = torch.log1p(entropy.detach())
        raw = torch.exp(float(mile_gamma) * (log_base - log_base.amax()))
        row_count = valid.sum(-1).to(torch.float32)
        if group_fraction is None:
            weights = raw * (valid_count / (raw * row_count).sum().clamp_min(1e-20))
        else:
            group = group_fraction.float()
            other = 1.0 - group
            group_scale = (row_count * group).sum() / (row_count * group * raw).sum().clamp_min(
                1e-20
            )
            other_scale = (row_count * other).sum() / (row_count * other * raw).sum().clamp_min(
                1e-20
            )
            weights = raw * (group * group_scale + other * other_scale)
        weighted = (nll * weights.unsqueeze(-1)).sum() / denominator
    mu = (
        float(mu_lambda) * classifier.float().mean(0).square().sum() if use_mu else ce.new_zeros(())
    )
    topk = partial_counts.sum(0) if need_topk else None
    return weighted + mu, ce.detach(), (weighted - ce).detach(), mu.detach(), topk
