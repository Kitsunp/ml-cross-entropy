"""Focused tests for the optional Leviathan Triton integration.

The tests deliberately exercise the new package boundary only.  Existing CCE
tests remain untouched; CCE's data-dependent loss operator has its own graph
policy and is not silently changed by this integration.
"""

from __future__ import annotations

import pytest
import torch

from cut_cross_entropy.leviathan import (
    LeviathanConfig,
    LeviathanEmbedding,
    LeviathanGenerator,
    make_triton_leviathan_generator,
)


def _config(*, dtype: torch.dtype) -> LeviathanConfig:
    return LeviathanConfig(
        vocab_size=4096,
        hidden_size=128,
        generator_d_seed=64,
        generator_num_modes=2,
        generator_num_knots=16,
        generator_k=3,
        generator_krank=16,
        dtype=dtype,
    )


def test_reference_fallback_is_trainable_on_cpu() -> None:
    cfg = _config(dtype=torch.float32)
    model = LeviathanEmbedding(cfg, use_reference=True)
    ids = torch.randint(cfg.vocab_size, (2, 8))

    output = model(ids)
    output.square().mean().backward()

    assert output.shape == (2, 8, cfg.hidden_size)
    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_neollm_adapter_preserves_reference_fallback() -> None:
    cfg = _config(dtype=torch.float32)
    reference = LeviathanGenerator(cfg)
    adapter_cls = make_triton_leviathan_generator(LeviathanGenerator)
    adapted = adapter_cls(cfg)
    adapted.load_state_dict(reference.state_dict())
    adapted.use_leviathan_triton = False
    ids = torch.randint(cfg.vocab_size, (2, 8))

    torch.testing.assert_close(adapted(ids), reference(ids), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_kernel_is_finite_and_stays_under_vram_budget() -> None:
    cfg = _config(dtype=torch.bfloat16)
    model = LeviathanEmbedding(cfg).cuda()
    ids = torch.randint(cfg.vocab_size, (1, 32), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    output = model(ids)
    output.float().square().mean().backward()
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert torch.cuda.max_memory_allocated() / 1e9 <= 10.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lev_boundary_has_no_dynamo_graph_break() -> None:
    cfg = _config(dtype=torch.bfloat16)
    model = LeviathanEmbedding(cfg).cuda()
    ids = torch.randint(cfg.vocab_size, (1, 32), device="cuda")
    report = torch._dynamo.explain(lambda value: model(value))(ids)

    assert report.graph_count == 1
    assert report.graph_break_count == 0
