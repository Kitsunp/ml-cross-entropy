# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from collections.abc import Callable

import pytest
import torch

from cut_cross_entropy.cce_lse_forward import _neg_correct_logit, cce_lse_forward_kernel
from cut_cross_entropy.indexed_dot import indexed_neg_dot_forward_kernel
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def cce_lse_kernel_indexed_dot(
    e: torch.Tensor,
    c: torch.Tensor,
    inds: torch.Tensor,
    bias: torch.Tensor | None = None,
    shift: int = 0,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    lse_return = cce_lse_forward_kernel(e, c, bias, valids, softcap, inds, shift)
    assert lse_return.neg_correct_logit is not None

    return lse_return.neg_correct_logit.to(out_dtype)


@skip_no_cuda
@pytest.mark.parametrize(
    "dtype,error_tol", [(torch.float32, 5e-6), (torch.float16, 2.5e-3), (torch.bfloat16, 2.5e-2)]
)
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("shape", [(256, 512, 512), (255, 507, 512), (255, 507, 497)])
@pytest.mark.parametrize("fn", [cce_lse_kernel_indexed_dot, indexed_neg_dot_forward_kernel])
def test_indexed_dot(
    dtype: torch.dtype,
    error_tol: float,
    softcap: float | None,
    has_bias: bool,
    shape: tuple[int, int, int],
    fn: Callable[..., torch.Tensor],
):
    # This test's tight FP32 tolerance validates the IEEE path. Do not inherit
    # a process-global TF32 policy from a test that ran earlier in the session.
    torch.set_float32_matmul_precision("highest")
    torch.cuda.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = torch.randn((N, D), device="cuda", dtype=dtype) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    if has_bias:
        bias = torch.randn(V, device="cuda", dtype=dtype)
    else:
        bias = None

    inds = torch.randint(0, V, size=(N,), device="cuda")

    gt = e.float() @ c.float().T

    if bias is not None:
        gt += bias.float()

    if softcap is not None:
        gt = softcapping(gt, softcap)

    gt = -gt.gather(dim=1, index=inds.view(-1, 1)).view(-1)

    ref = e @ c.T

    if bias is not None:
        ref += bias

    if softcap is not None:
        ref = softcapping(ref, softcap)

    ref = -ref.gather(dim=1, index=inds.view(-1, 1)).view(-1)

    cce_neg_dot = fn(e, c, inds, bias=bias, softcap=softcap)

    expected_error = (gt - ref.float()).abs()
    cce_error = (gt - cce_neg_dot.float()).abs()

    assert (
        cce_error <= (expected_error + error_tol)
    ).all(), f"{(cce_error - expected_error).relu().max()=}"


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_indexed_dot_masks_out_of_range_targets(dtype: torch.dtype) -> None:
    torch.cuda.manual_seed(20_260_811)
    e = torch.randn((8, 64), device="cuda", dtype=dtype)
    c = torch.randn((7, 64), device="cuda", dtype=dtype)
    bias = torch.randn((7,), device="cuda", dtype=dtype)
    targets = torch.tensor([-1, 0, 1, 6, 7, 2, 3, 4], device="cuda")

    new_indexed = _neg_correct_logit(e, c, bias, None, targets, None, 0, "ieee")
    legacy_indexed = indexed_neg_dot_forward_kernel(e, c, targets, bias=bias)
    torch.cuda.synchronize()

    invalid = (targets < 0) | (targets >= c.size(0))
    torch.testing.assert_close(new_indexed[invalid], torch.zeros_like(new_indexed[invalid]))
    torch.testing.assert_close(
        legacy_indexed[invalid], torch.zeros_like(legacy_indexed[invalid])
    )


@skip_no_cuda
@pytest.mark.parametrize("fn", [cce_lse_kernel_indexed_dot, indexed_neg_dot_forward_kernel])
def test_indexed_dot_fp32_high_policy_has_bounded_tf32_error(
    fn: Callable[..., torch.Tensor],
) -> None:
    torch.set_float32_matmul_precision("high")
    torch.cuda.manual_seed(29)
    rows, vocab, dim = 256, 512, 512
    e = torch.randn((rows, dim), device="cuda", dtype=torch.float32) / (dim**0.5)
    c = torch.randn((vocab, dim), device="cuda", dtype=torch.float32)
    targets = torch.randint(0, vocab, (rows,), device="cuda")

    # Compute the selected logits directly in FP64. A dense FP32 matmul would
    # inherit the same TF32 policy and is not a precision-independent oracle.
    reference = -(e.double() * c[targets].double()).sum(dim=1)
    actual = fn(e, c, targets)

    max_error = (actual.double() - reference).abs().max()
    assert float(max_error) < 4e-3


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("packed_shift", [False, True])
@pytest.mark.parametrize("forward_reduction", ["lock", "split"])
def test_target_logit_uses_same_reduction_as_lse(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    softcap: float | None,
    has_bias: bool,
    packed_shift: bool,
    forward_reduction: str,
) -> None:
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", forward_reduction)
    torch.cuda.manual_seed(20_260_811)
    e = torch.randn((512, 512), device="cuda", dtype=dtype)
    c = torch.randn((1, 512), device="cuda", dtype=dtype)
    targets = torch.zeros((512,), device="cuda", dtype=torch.long)
    bias = torch.randn((1,), device="cuda", dtype=dtype) if has_bias else None
    if packed_shift:
        valids = torch.arange(0, 511, 2, device="cuda", dtype=torch.int32)
        shift = 1
    else:
        valids = None
        shift = 0

    result = cce_lse_forward_kernel(
        e,
        c,
        bias=bias,
        valids=valids,
        softcap=softcap,
        targets=targets,
        shift=shift,
    )
    assert result.neg_correct_logit is not None
    loss = result.lse + result.neg_correct_logit

    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0.0, atol=0.0)
