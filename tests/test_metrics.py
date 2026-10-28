from __future__ import annotations

import math

import pytest
import torch

from cut_cross_entropy import finalize_lm_metrics, linear_cross_entropy


def totals(device="cpu"):
    return torch.tensor(
        [23.0, 26.0, 10.0, 12.0, 15.0, 3.0, 3.0, 7.0, 9.0, 10.0, 0.0],
        device=device,
        dtype=torch.float64,
    )


def expected(raw):
    values = raw.cpu().tolist()
    derived = []
    for start in (0, 3):
        ce, mile, count = values[start : start + 3]
        if count:
            ce, mile = ce / count, mile / count
            derived.extend((ce, math.exp(ce), mile, math.exp(mile)))
        else:
            derived.extend((0.0,) * 4)
    derived.extend(hit / max(values[9], 1) for hit in values[6:9])
    return torch.tensor(values + derived, dtype=torch.float64, device=raw.device)


@pytest.mark.parametrize("scale", [1, 2, 2**25])
def test_finalize_is_token_weighted_and_preserves_raw_counts(scale):
    raw = totals() * scale
    original = raw.clone()
    output = finalize_lm_metrics(raw)
    torch.testing.assert_close(output, expected(raw), rtol=2e-14, atol=1e-14)
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
    assert output.data_ptr() != raw.data_ptr()
    assert output.dtype == torch.float64 and not output.requires_grad
    torch.testing.assert_close(
        output,
        finalize_lm_metrics(totals()).mul(torch.tensor([scale] * 11 + [1] * 11)),
        rtol=2e-14,
        atol=1e-14,
    )


def test_empty_phases_and_topk_have_zero_placeholders():
    torch.testing.assert_close(
        finalize_lm_metrics(torch.zeros(11, dtype=torch.float64)),
        torch.zeros(22, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_preserves_raw_health_flags_without_synchronizing_or_repairing_them():
    raw = totals()
    raw[10] = 2
    assert finalize_lm_metrics(raw)[10] == 2
    raw[0] = float("nan")
    assert torch.isnan(finalize_lm_metrics(raw)[12])


def test_float64_exponential_range():
    raw = totals()
    raw[0] = 1000
    assert math.isclose(finalize_lm_metrics(raw)[12], math.exp(100), rel_tol=2e-14)
    raw[0] = 10000
    assert math.isinf(finalize_lm_metrics(raw)[12])


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(10, dtype=torch.float64),
        torch.zeros(11),
        torch.zeros(22, dtype=torch.float64)[::2],
        torch.zeros(11, dtype=torch.float64, requires_grad=True),
    ],
)
def test_rejects_wrong_shape_dtype_stride_or_grad(value):
    with pytest.raises(ValueError):
        finalize_lm_metrics(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_compiled_dynamic_values_and_graph_capture():
    import torch._inductor.config as config

    with config.patch(compile_threads=2):
        compiled = torch.compile(finalize_lm_metrics, fullgraph=True, mode="max-autotune")
        raw = totals("cuda")
        for scale in (1, 2, 2**25):
            torch.testing.assert_close(
                compiled(raw * scale), expected(raw * scale), rtol=2e-14, atol=1e-14
            )
    for _ in range(3):
        finalize_lm_metrics(raw)
    torch.cuda.synchronize()  # Test boundary, never inside finalization.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = finalize_lm_metrics(raw)
    graph.replay()
    torch.testing.assert_close(output, expected(raw), rtol=2e-14, atol=1e-14)
    raw.mul_(2)
    graph.replay()
    torch.testing.assert_close(output, expected(raw), rtol=2e-14, atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_exponential_range_and_empty_phases():
    raw = totals("cuda")
    for ce_sum in (0.0, 1000.0, 10000.0):
        raw[0] = ce_sum
        cpu = finalize_lm_metrics(raw.cpu())
        torch.testing.assert_close(finalize_lm_metrics(raw).cpu(), cpu, rtol=2e-14, atol=1e-14)
    torch.testing.assert_close(
        finalize_lm_metrics(raw.zero_()), torch.zeros_like(raw).new_zeros(22)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_finalization_one_kernel_and_no_host_scalar_reads_or_copies():
    raw = totals("cuda")
    for _ in range(3):
        finalize_lm_metrics(raw)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        for _ in range(3):
            finalize_lm_metrics(raw)
    names = [event.name for event in profile.events()]
    assert not any(
        name in ("aten::item", "aten::_local_scalar_dense", "aten::_to_copy") for name in names
    )
    assert not any("Memcpy DtoH" in name for name in names)
    kernels = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == 3 and all("_finalize_lm_metrics_kernel" in name for name in kernels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mile_enabled", [False, True])
def test_compiled_cce_components_and_topk_finalize_against_dense(mile_enabled):
    import torch._inductor.config as config

    torch.manual_seed(784)
    e = torch.randn(2, 7, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    c = torch.randn(17, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, -100, -100, 7, 9], [4, 5, 6, 8, 10, 11, 12]], device="cuda")
    valid = labels[:, 1:] != -100
    # CCE rounds dot products to the embedding dtype before softmax/Top-k.
    logits = (e.detach()[:, :-1].float() @ c.detach().float().T).bfloat16().float()[valid]
    targets = labels[:, 1:][valid]
    lse = logits.logsumexp(-1)
    nll = lse - logits.gather(1, targets[:, None]).flatten()
    entropy = lse - (logits.softmax(-1) * logits).sum(-1)
    weight = 1 + entropy
    objective = (weight * nll).sum() / weight.sum() if mile_enabled else nll.mean()
    order = logits.argsort(dim=-1, descending=True, stable=True)
    hits = torch.stack([(order[:, :k] == targets[:, None]).any(-1).sum() for k in (1, 3, 5)])

    def loss(a, b):
        return linear_cross_entropy(
            a,
            b,
            labels,
            shift=1,
            impl="cce_kahan_full_c",
            mile_enabled=mile_enabled,
            mu_loss_enabled=True,
            return_loss_metrics=True,
            return_topk=True,
        )

    with config.patch(compile_threads=2):
        value, metrics = torch.compile(loss, fullgraph=True, mode="max-autotune")(e, c)
        ce = metrics["ntp_ce_unweighted"].detach().clone()
        mile = ce + metrics["mile_reweighting_delta"].detach().clone()
        counts = metrics["topk_counts"].detach().double().clone()
        count = counts[3]
        sums = torch.stack((ce.double() * count, mile.double() * count, count))
        raw = torch.cat((sums, sums.new_zeros(3), counts))
        output = finalize_lm_metrics(raw)
        torch.testing.assert_close(output[12], nll.mean().double().exp(), rtol=5e-5, atol=0)
        torch.testing.assert_close(output[14], objective.double().exp(), rtol=5e-5, atol=0)
        torch.testing.assert_close(counts[:3], hits.double(), rtol=0, atol=0)
        torch.testing.assert_close(output[19:22], hits.double() / valid.sum(), rtol=0, atol=0)
        assert count == valid.sum() and counts[4] == 0
        value.backward()
        assert torch.isfinite(e.grad).all() and torch.isfinite(c.grad).all()
