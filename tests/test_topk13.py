from __future__ import annotations

import pytest
import torch

from cut_cross_entropy import SUPPORTED_TOPK, linear_cross_entropy
from cut_cross_entropy.cce_utils import normalize_return_topk


@pytest.mark.parametrize("value,expected", [(False, 0), (True, 5), (0, 0), (1, 5), (3, 3), (5, 5)])
def test_return_topk_normalization(value, expected):
    assert normalize_return_topk(value) == expected
    assert SUPPORTED_TOPK == (3, 5)


@pytest.mark.parametrize("value", [-1, 2, 4, 6, 3.0, "3", None])
def test_reject_unsupported_rank(value):
    with pytest.raises(ValueError, match="return_topk"):
        normalize_return_topk(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reduction", ["lock", "split"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_top1_top3_without_top5_dense_loss_and_gradient_parity(
    monkeypatch, reduction, dtype, compiled
):
    import torch._inductor.config as config

    if compiled and dtype == torch.float32:
        pytest.skip("CCE's existing fullgraph boundary requires FP16/BF16 compute")
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    torch.manual_seed(914)
    base_e = torch.randn(2, 11, 16, device="cuda", dtype=dtype)
    base_c = torch.randn(19, 16, device="cuda", dtype=dtype)
    labels = torch.randint(0, 19, (2, 11), device="cuda")
    labels[0, 3:5] = -100
    labels[1, -3:] = -100
    valid = labels[:, 1:] != -100
    logits = (base_e[:, :-1].float() @ base_c.float().T).to(dtype).float()[valid]
    order = logits.argsort(dim=-1, descending=True, stable=True)
    targets = labels[:, 1:][valid]
    expected = torch.stack([(order[:, :k] == targets[:, None]).any(-1).sum() for k in (1, 3)])
    results = []
    for rank in (3, True):
        e = base_e.clone().requires_grad_()
        c = base_c.clone().requires_grad_()

        def loss(a, b):
            return linear_cross_entropy(
                a,
                b,
                labels,
                shift=1,
                impl="cce_kahan_full_c",
                mile_enabled=True,
                mu_loss_enabled=True,
                return_loss_metrics=True,
                return_topk=rank,
            )

        with config.patch(compile_threads=2):
            fn = torch.compile(loss, fullgraph=True, mode="max-autotune") if compiled else loss
            value, metrics = fn(e, c)
            counts = metrics["topk_counts"].detach().clone()
            torch.testing.assert_close(counts[:2], expected.float(), rtol=0, atol=0)
            assert counts[3] == valid.sum() and counts[4] == 0
            if rank == 3:
                assert counts[2] == 0  # Reserved wire slot, not measured Top-5.
            value.backward()
            results.append((value.detach().clone(), e.grad.clone(), c.grad.clone()))
    for a, b in zip(results[0], results[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reduction", ["lock", "split"])
def test_top3_ties_empty_and_changed_valid_counts(monkeypatch, reduction):
    import torch._inductor.config as config

    monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
    e = torch.zeros(2, 7, 16, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros(9, 16, device="cuda", dtype=torch.bfloat16)
    labels = torch.tensor([[3, 0, 1, 2, 3, -100, 4], [2, 2, 1, 0, 8, 7, 6]], device="cuda")

    def loss(a, b, y):
        return linear_cross_entropy(a, b, y, shift=1, return_loss_metrics=True, return_topk=3)

    with config.patch(compile_threads=2):
        fn = torch.compile(loss, fullgraph=True, mode="max-autotune")
        for empty in (False, True, False):
            y = torch.full_like(labels, -100) if empty else labels
            _, metrics = fn(e, c, y)
            values = y[:, 1:]
            valid = values != -100
            expected = torch.stack(
                (
                    (values.eq(0) & valid).sum(),
                    ((values < 3) & valid).sum(),
                    values.new_zeros(()),
                    valid.sum(),
                    values.new_zeros(()),
                )
            )
            torch.testing.assert_close(metrics["topk_counts"], expected.float(), rtol=0, atol=0)
