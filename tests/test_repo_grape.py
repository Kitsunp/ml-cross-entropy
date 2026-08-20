from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy.repo_grape import repo_grape, repo_grape_supported

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@dataclass(frozen=True)
class Case:
    batch: int
    query_heads: int
    key_heads: int
    sequence: int
    head_dim: int
    rot_half: int
    sequence_pseudo_factor: int = 1


def _inputs(
    case: Case,
    *,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260820)
    effective_sequence = case.sequence * case.sequence_pseudo_factor
    q = torch.randn(
        case.batch,
        effective_sequence,
        case.query_heads,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    ).transpose(1, 2)
    k = torch.randn(
        case.batch,
        effective_sequence,
        case.key_heads,
        case.head_dim,
        device="cuda",
        dtype=dtype,
    ).transpose(1, 2)
    z = torch.randn(
        case.batch,
        case.query_heads,
        case.sequence,
        device="cuda",
        dtype=torch.bfloat16,
    )
    position_ids = torch.arange(case.sequence, device="cuda").expand(case.batch, -1)
    inv_freq = torch.exp(
        -math.log(10_000.0)
        * torch.arange(case.rot_half, device="cuda", dtype=torch.float32)
        / case.rot_half
    )
    alpha = torch.linspace(0.8, 1.2, case.query_heads, device="cuda")
    log_scale = (
        torch.randn(
            case.query_heads,
            case.head_dim // 2,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    q_weight = torch.randn(case.head_dim, device="cuda") * 0.02 + 1.0
    k_weight = torch.randn(case.head_dim, device="cuda") * 0.02 + 1.0
    if requires_grad:
        for tensor in (q, k, z, alpha, log_scale, q_weight, k_weight):
            tensor.requires_grad_(True)
    return q, k, z, position_ids, inv_freq, alpha, log_scale, q_weight, k_weight


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    z: torch.Tensor,
    position_ids: torch.Tensor | None,
    inv_freq: torch.Tensor,
    alpha: torch.Tensor,
    log_scale: torch.Tensor,
    q_norm_weight: torch.Tensor | None,
    k_norm_weight: torch.Tensor | None,
    *,
    attention_scaling: float,
    momentum_gamma: float,
    rms_norm_eps: float,
    sequence_pseudo_factor: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q_norm_weight is not None:
        q = F.rms_norm(q, (q.shape[-1],), q_norm_weight, rms_norm_eps)
        k = F.rms_norm(k, (k.shape[-1],), k_norm_weight, rms_norm_eps)
    batch, repo_heads, base_sequence = z.shape
    if position_ids is None:
        positions = torch.arange(base_sequence, device=z.device, dtype=torch.float32).view(
            1, base_sequence
        )
    else:
        positions = position_ids.float()
    if positions.shape[0] == 1 and batch != 1:
        positions = positions.expand(batch, -1)
    coordinates = positions.unsqueeze(1) + alpha.view(1, repo_heads, 1) * (
        z.float() - positions.unsqueeze(1)
    )
    if sequence_pseudo_factor > 1:
        factor = sequence_pseudo_factor
        offsets = torch.arange(factor, device=z.device, dtype=torch.float32)
        coordinates = (coordinates.unsqueeze(-1) * factor + offsets.view(1, 1, 1, factor)).reshape(
            batch, repo_heads, base_sequence * factor
        )

    query_heads = q.shape[1]
    key_heads = k.shape[1]
    head_pseudo_factor = query_heads // repo_heads
    base_key_heads = key_heads // head_pseudo_factor
    queries_per_key = repo_heads // base_key_heads
    effective_sequence = q.shape[2]
    rot_half = inv_freq.numel()
    rotary_dim = 2 * rot_half
    query_coordinates = (
        coordinates.unsqueeze(2)
        .expand(batch, repo_heads, head_pseudo_factor, effective_sequence)
        .reshape(batch, query_heads, effective_sequence)
    )
    frequency = inv_freq.view(1, rot_half) * torch.exp(log_scale[:, :rot_half])
    query_frequency = (
        frequency.unsqueeze(1)
        .expand(repo_heads, head_pseudo_factor, rot_half)
        .reshape(query_heads, rot_half)
    )
    phase = query_coordinates.unsqueeze(-1) * query_frequency.view(1, query_heads, 1, rot_half)
    query_cosine = phase.cos()
    query_sine = phase.sin()
    key_cosine = query_cosine.reshape(
        batch,
        base_key_heads,
        queries_per_key,
        head_pseudo_factor,
        effective_sequence,
        rot_half,
    ).mean(2)
    key_sine = query_sine.reshape(
        batch,
        base_key_heads,
        queries_per_key,
        head_pseudo_factor,
        effective_sequence,
        rot_half,
    ).mean(2)
    resultant_sq = key_cosine.square() + key_sine.square()
    good = resultant_sq > torch.finfo(torch.float32).eps
    inverse = torch.rsqrt(torch.where(good, resultant_sq, torch.ones_like(resultant_sq)))
    key_cosine = torch.where(good, key_cosine * inverse, torch.ones_like(key_cosine))
    key_sine = torch.where(good, key_sine * inverse, torch.zeros_like(key_sine))
    key_cosine = key_cosine.reshape(batch, key_heads, effective_sequence, rot_half)
    key_sine = key_sine.reshape(batch, key_heads, effective_sequence, rot_half)

    scale = torch.tensor(attention_scaling, device=q.device, dtype=torch.float32)
    q_first, q_second = q[..., :rot_half], q[..., rot_half:rotary_dim]
    k_first, k_second = k[..., :rot_half], k[..., rot_half:rotary_dim]
    q_rotated = torch.cat(
        (
            scale * (q_first * query_cosine - q_second * query_sine),
            scale * (q_second * query_cosine + q_first * query_sine),
            q[..., rotary_dim:],
        ),
        dim=-1,
    )
    k_rotated = torch.cat(
        (
            scale * (k_first * key_cosine - k_second * key_sine),
            scale * (k_second * key_cosine + k_first * key_sine),
            k[..., rotary_dim:],
        ),
        dim=-1,
    )

    def momentum(x: torch.Tensor) -> torch.Tensor:
        previous = F.pad(x[..., :-1, :], (0, 0, 1, 0))
        return (x + momentum_gamma * (x - previous)).to(output_dtype)

    return momentum(q_rotated), momentum(k_rotated)


@pytest.mark.parametrize(
    "case",
    (
        Case(2, 8, 8, 17, 64, 8),
        Case(2, 8, 4, 17, 64, 8),
        Case(1, 8, 2, 31, 64, 16),
        Case(2, 8, 4, 13, 64, 8, 2),
        Case(1, 16, 4, 9, 128, 32),
    ),
)
@pytest.mark.parametrize("fuse_norm", (False, True))
def test_forward_matches_reference(case: Case, fuse_norm: bool) -> None:
    dtype = torch.bfloat16 if fuse_norm else torch.float32
    q, k, z, position_ids, inv_freq, alpha, log_scale, q_weight, k_weight = _inputs(
        case, dtype=dtype
    )
    norm_weights = (q_weight, k_weight) if fuse_norm else (None, None)
    expected = _reference(
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        *norm_weights,
        attention_scaling=1.0,
        momentum_gamma=0.1,
        rms_norm_eps=1.0e-6,
        sequence_pseudo_factor=case.sequence_pseudo_factor,
        output_dtype=torch.bfloat16,
    )
    actual = repo_grape(
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        1.0,
        sequence_pseudo_factor=case.sequence_pseudo_factor,
        momentum_gamma=0.1,
        output_dtype=torch.bfloat16,
        q_norm_weight=norm_weights[0],
        k_norm_weight=norm_weights[1],
        rms_norm_eps=1.0e-6,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=8.0e-3, atol=2.0e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=8.0e-3, atol=2.0e-2)


def test_no_position_ids_and_fp32_output() -> None:
    case = Case(2, 8, 4, 19, 64, 8)
    q, k, z, _position_ids, inv_freq, alpha, log_scale, q_weight, k_weight = _inputs(
        case, dtype=torch.float32
    )
    expected = _reference(
        q,
        k,
        z,
        None,
        inv_freq,
        alpha,
        log_scale,
        q_weight,
        k_weight,
        attention_scaling=0.75,
        momentum_gamma=0.0,
        rms_norm_eps=1.0e-6,
        sequence_pseudo_factor=1,
        output_dtype=torch.float32,
    )
    actual = repo_grape(
        q,
        k,
        z,
        None,
        inv_freq,
        alpha,
        log_scale,
        0.75,
        q_norm_weight=q_weight,
        k_norm_weight=k_weight,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=8.0e-6, atol=8.0e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=8.0e-6, atol=8.0e-6)


def test_circular_resultant_fallback_is_finite_identity() -> None:
    q = torch.zeros((1, 2, 1, 8), device="cuda")
    k = torch.randn((1, 1, 1, 8), device="cuda")
    z = torch.tensor([[[0.0], [math.pi]]], device="cuda")
    position_ids = torch.zeros((1, 1), device="cuda", dtype=torch.int64)
    inv_freq = torch.ones((1,), device="cuda")
    alpha = torch.ones((2,), device="cuda")
    log_scale = torch.zeros((2, 4), device="cuda")
    _q_out, k_out = repo_grape(q, k, z, position_ids, inv_freq, alpha, log_scale, 1.0)
    assert torch.isfinite(k_out).all()
    torch.testing.assert_close(k_out, k)


@pytest.mark.parametrize(
    ("case", "dtype", "output_dtype"),
    (
        (Case(2, 8, 4, 17, 64, 8), torch.bfloat16, torch.bfloat16),
        (Case(2, 8, 4, 13, 64, 8, 2), torch.bfloat16, torch.bfloat16),
        (Case(1, 8, 4, 9, 64, 8), torch.float32, torch.float32),
    ),
)
def test_training_gradients_and_compile_match_reference(
    case: Case,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    inputs = _inputs(case, dtype=dtype, requires_grad=True)
    q, k, z, position_ids, inv_freq, alpha, log_scale, q_weight, k_weight = inputs
    expected_outputs = _reference(
        *inputs[:7],
        q_weight,
        k_weight,
        attention_scaling=1.0,
        momentum_gamma=0.1,
        rms_norm_eps=1.0e-6,
        sequence_pseudo_factor=case.sequence_pseudo_factor,
        output_dtype=output_dtype,
    )
    torch.manual_seed(11)
    grad_outputs = tuple(torch.randn_like(output) for output in expected_outputs)
    differentiable = (q, k, z, alpha, log_scale, q_weight, k_weight)
    expected_grads = torch.autograd.grad(expected_outputs, differentiable, grad_outputs)

    def compiled_forward(q, k, z, position_ids, inv_freq, alpha, log_scale, qw, kw):
        return repo_grape(
            q,
            k,
            z,
            position_ids,
            inv_freq,
            alpha,
            log_scale,
            1.0,
            sequence_pseudo_factor=case.sequence_pseudo_factor,
            momentum_gamma=0.1,
            output_dtype=output_dtype,
            q_norm_weight=qw,
            k_norm_weight=kw,
        )

    compiled = torch.compile(compiled_forward, mode="max-autotune", fullgraph=True)
    actual_outputs = compiled(*inputs)
    actual_grads = torch.autograd.grad(actual_outputs, differentiable, grad_outputs)
    if dtype == torch.float32:
        torch.testing.assert_close(actual_outputs[0], expected_outputs[0], rtol=8.0e-6, atol=8.0e-6)
        torch.testing.assert_close(actual_outputs[1], expected_outputs[1], rtol=8.0e-6, atol=8.0e-6)
        tolerances = (
            (5.0e-5, 5.0e-5),
            (5.0e-5, 5.0e-5),
            (1.0e-3, 1.0e-3),
            (5.0e-5, 5.0e-5),
            (5.0e-5, 5.0e-5),
            (5.0e-5, 5.0e-5),
            (5.0e-5, 5.0e-5),
        )
    else:
        torch.testing.assert_close(actual_outputs[0], expected_outputs[0], rtol=8.0e-3, atol=2.0e-2)
        torch.testing.assert_close(actual_outputs[1], expected_outputs[1], rtol=8.0e-3, atol=2.0e-2)
        tolerances = (
            (8.0e-4, 4.0e-3),
            (8.0e-4, 4.0e-3),
            (8.0e-3, 8.0e-3),
            (1.0e-3, 4.0e-3),
            (1.0e-3, 4.0e-3),
            (5.0e-4, 3.0e-3),
            (5.0e-4, 3.0e-3),
        )
    for actual, expected, (rtol, atol) in zip(actual_grads, expected_grads, tolerances):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("sequence", (1, 3, 5, 63, 65))
def test_boundary_sequence_lengths_match_reference(sequence: int) -> None:
    case = Case(1, 4, 2, sequence, 32, 8)
    inputs = _inputs(case, dtype=torch.bfloat16)
    q, k, z, position_ids, inv_freq, alpha, log_scale, q_weight, k_weight = inputs
    expected = _reference(
        *inputs[:7],
        q_weight,
        k_weight,
        attention_scaling=1.0,
        momentum_gamma=0.1,
        rms_norm_eps=1.0e-6,
        sequence_pseudo_factor=1,
        output_dtype=torch.bfloat16,
    )
    actual = repo_grape(
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        1.0,
        momentum_gamma=0.1,
        output_dtype=torch.bfloat16,
        q_norm_weight=q_weight,
        k_norm_weight=k_weight,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=8.0e-3, atol=2.0e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=8.0e-3, atol=2.0e-2)


def test_reset_position_ids_and_no_momentum_match_reference() -> None:
    case = Case(2, 8, 4, 9, 64, 8)
    inputs = _inputs(case, dtype=torch.bfloat16)
    q, k, z, _position_ids, inv_freq, alpha, log_scale, q_weight, k_weight = inputs
    position_ids = torch.tensor(
        ((0, 1, 2, 3, 0, 1, 2, 2, 7), (11, 12, 4, 5, 6, 0, 1, 2, 3)),
        device="cuda",
    )
    expected = _reference(
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        q_weight,
        k_weight,
        attention_scaling=0.75,
        momentum_gamma=0.0,
        rms_norm_eps=1.0e-6,
        sequence_pseudo_factor=1,
        output_dtype=torch.bfloat16,
    )
    actual = repo_grape(
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        0.75,
        momentum_gamma=0.0,
        output_dtype=torch.bfloat16,
        q_norm_weight=q_weight,
        k_norm_weight=k_weight,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=8.0e-3, atol=2.0e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=8.0e-3, atol=2.0e-2)


def test_no_grad_inference_does_not_build_autograd_state() -> None:
    case = Case(1, 8, 4, 65, 64, 16)
    inputs = _inputs(case, dtype=torch.bfloat16, requires_grad=True)
    q, k, z, position_ids, inv_freq, alpha, log_scale, q_weight, k_weight = inputs
    with torch.no_grad():
        outputs = repo_grape(
            q,
            k,
            z,
            position_ids,
            inv_freq,
            alpha,
            log_scale,
            1.0,
            momentum_gamma=0.1,
            output_dtype=torch.bfloat16,
            q_norm_weight=q_weight,
            k_norm_weight=k_weight,
        )
    assert outputs[0].grad_fn is None
    assert outputs[1].grad_fn is None
    assert torch.isfinite(outputs[0]).all()
    assert torch.isfinite(outputs[1]).all()


def test_support_contract_rejects_incompatible_inputs() -> None:
    case = Case(1, 8, 4, 8, 64, 8)
    q, k, z, _position_ids, inv_freq, alpha, log_scale, q_weight, k_weight = _inputs(
        case, dtype=torch.bfloat16
    )
    assert repo_grape_supported(
        q,
        k,
        z,
        inv_freq,
        alpha,
        log_scale,
        q_norm_weight=q_weight,
        k_norm_weight=k_weight,
    )
    assert not repo_grape_supported(
        q.cpu(),
        k,
        z,
        inv_freq,
        alpha,
        log_scale,
        q_norm_weight=q_weight,
        k_norm_weight=k_weight,
    )
