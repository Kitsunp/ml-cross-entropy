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
    leviathan_forward_ref,
    leviathan_embedding,
    leviathan_embedding_compiler_safe,
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


def _detached_params(generator: LeviathanGenerator) -> dict[str, torch.Tensor]:
    return {
        name: getattr(generator, name).detach().clone().requires_grad_()
        for name in (
            "codebooks",
            "head_proj_weight",
            "head_norm_weight",
            "head_norm_bias",
            "head_spline_delta",
            "head_out_weight",
        )
    }


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


def test_reference_dispatch_preserves_grads_and_custom_knot_grid() -> None:
    cfg = _config(dtype=torch.float32)
    generator = LeviathanGenerator(cfg)
    params = _detached_params(generator)
    knot_grid = torch.linspace(0.0, 1.0, cfg.generator_num_knots).pow(1.7)
    params["knot_grid"] = knot_grid
    ids = torch.randint(cfg.vocab_size, (2, 8))

    output = leviathan_embedding(ids, params, cfg)
    expected, _ = leviathan_forward_ref(
        ids,
        params,
        cfg,
        save_intermediates=False,
    )
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)

    compiler_output = leviathan_embedding_compiler_safe(
        ids,
        {key: value for key, value in params.items() if key != "knot_grid"},
        cfg,
        knot_grid,
    )
    torch.testing.assert_close(compiler_output, expected, rtol=0.0, atol=0.0)

    output.float().square().mean().backward()
    assert all(
        params[name].grad is not None and torch.isfinite(params[name].grad).all()
        for name in params
        if name != "knot_grid"
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
    assert {parameter.dtype for parameter in model.generator.parameters()} == {
        torch.bfloat16
    }
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
def test_cuda_inference_skips_backward_checkpoints() -> None:
    import cut_cross_entropy.leviathan.compiler as compiler

    cfg = _config(dtype=torch.bfloat16)
    model = LeviathanEmbedding(cfg).cuda()
    ids = torch.randint(cfg.vocab_size, (1, 32), device="cuda")
    original_forward = compiler._leviathan_forward
    calls: list[bool] = []

    def tracked_forward(*args, **kwargs):
        calls.append(bool(kwargs["save_intermediates"]))
        return original_forward(*args, **kwargs)

    compiler._leviathan_forward = tracked_forward
    try:
        with torch.no_grad():
            output = model(ids)
        for parameter in model.generator.parameters():
            parameter.requires_grad_(False)
        frozen_output = model(ids)
        torch.cuda.synchronize()
    finally:
        compiler._leviathan_forward = original_forward

    assert calls == [False, False]
    assert not output.requires_grad
    assert not frozen_output.requires_grad
    assert torch.isfinite(output).all()
    assert torch.isfinite(frozen_output).all()
    assert torch.cuda.max_memory_allocated() / 1e9 <= 10.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_compiler_fallback_backward_is_chunked() -> None:
    import cut_cross_entropy.leviathan.compiler as compiler

    cfg = _config(dtype=torch.bfloat16)
    model = LeviathanEmbedding(cfg).cuda()
    ids = torch.randint(cfg.vocab_size, (1, 32), device="cuda")
    original_kernel_backward = compiler._leviathan_backward_triton
    original_reference_backward = compiler.leviathan_backward
    chunks: list[int | None] = []

    def tracked_backward(*args, **kwargs):
        chunks.append(kwargs.get("chunk"))
        return original_reference_backward(*args, **kwargs)

    compiler._leviathan_backward_triton = None
    compiler.leviathan_backward = tracked_backward
    try:
        output = model(ids)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
    finally:
        compiler._leviathan_backward_triton = original_kernel_backward
        compiler.leviathan_backward = original_reference_backward

    assert chunks == [8192]
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
