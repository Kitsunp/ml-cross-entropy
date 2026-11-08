"""Regression coverage for masked overflow and LEV spline gradients."""
import pytest
import torch

from cut_cross_entropy.leviathan import (
    LeviathanConfig, LeviathanGenerator, leviathan_embedding_compiler_safe,
    leviathan_embedding_with_seed_compiler_safe, leviathan_forward_ref,
)

NAMES = ("codebooks", "head_proj_weight", "head_norm_weight", "head_norm_bias",
         "head_spline_delta", "head_out_weight")


def _config():
    return LeviathanConfig(vocab_size=64402, hidden_size=512, generator_d_seed=128,
                           generator_num_modes=8, generator_num_knots=16,
                           generator_k=3, generator_krank=64, dtype=torch.bfloat16)


def test_backward_custom_op_keeps_cached_graph_call_signature():
    schema = torch.ops.cut_cross_entropy.leviathan_backward.default._schema
    arguments = [argument.name for argument in schema.arguments]
    assert len(arguments) == 24
    assert arguments[-9:] == ["seed_grad", "vocab_size", "hidden_size", "d_seed",
                             "num_modes", "num_knots", "spline_degree", "generator_k", "krank"]


def _masked_only_overflow(generator):
    """Token 16 overflows its discarded product; token 1 has all modes 1."""
    with torch.no_grad():
        generator.codebooks.zero_()
        pattern = torch.cat((torch.ones(64), -torch.ones(64))).to(generator.codebooks)
        generator.codebooks[2, 16].copy_(pattern)
        generator.codebooks[2, 1].copy_(-pattern)
        generator.head_proj_weight.copy_(
            torch.eye(128, device=generator.codebooks.device, dtype=torch.bfloat16).expand(8, -1, -1)
        )
        generator.head_norm_weight.fill_(1)
        generator.head_norm_bias.zero_()
        generator.head_spline_delta.zero_()
        generator.head_spline_delta[:, :64, 8:11, :].fill_(1.015625)
        generator.head_spline_delta[:, 64:, 5:8, :].fill_(1.015625)


class _Observed(torch.nn.Module):
    def __init__(self, generator, with_seed=False, mask_token_id=16):
        super().__init__()
        self.generator = generator
        self.mask_embedding = torch.nn.Parameter(torch.full(
            (generator.config.hidden_size,), 0.25,
            device=generator.codebooks.device, dtype=torch.bfloat16,
        ))
        self.with_seed = with_seed
        self.mask_token_id = mask_token_id

    def forward(self, ids):
        g = self.generator
        params = {name: getattr(g, name) for name in NAMES}
        fn = leviathan_embedding_with_seed_compiler_safe if self.with_seed else leviathan_embedding_compiler_safe
        return fn(ids, params, g.config, g.knot_grid, mask_embedding=self.mask_embedding,
                  mask_token_id=self.mask_token_id, return_dynamics_checkpoints=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dot", ["0", "1"])
def test_masked_overflow_isolated_in_compiled_forward_backward(monkeypatch, dot):
    monkeypatch.setenv("LEV_DOT", dot)
    torch.manual_seed(1234)
    generator = LeviathanGenerator(_config()).cuda()
    _masked_only_overflow(generator)
    model = _Observed(generator)
    compiled = torch.compile(model, fullgraph=True, mode="max-autotune")
    control_ids = torch.tensor([[1, 1, 1, 1]], device="cuda")
    ids = torch.tensor([[16, 1, 1, 16]], device="cuda")
    grad = torch.ones(1, 4, 512, device="cuda", dtype=torch.bfloat16)
    control_grad = grad.clone()
    control_grad[:, (0, 3)] = 0

    control, _ = compiled(control_ids)
    control.backward(control_grad)
    expected = {name: getattr(generator, name).grad.detach().clone() for name in NAMES}
    model.zero_grad(set_to_none=True)
    output, (_, _, modes) = compiled(ids)
    output.backward(grad)
    assert torch.isfinite(output).all()
    assert torch.isfinite(modes).all()
    assert torch.count_nonzero(modes.reshape(4, -1)[[0, 3]]) == 0
    torch.testing.assert_close(output[:, 1:3], control[:, 1:3], rtol=0, atol=0)
    torch.testing.assert_close(model.mask_embedding.grad, torch.full_like(model.mask_embedding, 2))
    for name in NAMES:
        actual = getattr(generator, name).grad
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, expected[name], rtol=0.02, atol=0.001, msg=name)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_torch_meap_reference_skips_discarded_product(device):
    torch.manual_seed(1234)
    g = LeviathanGenerator(_config()).to(device)
    _masked_only_overflow(g)
    ids = torch.tensor([[16, 1, 1, 16]], device=device)
    params = {name: getattr(g, name) for name in NAMES}
    params["knot_grid"] = g.knot_grid
    e, _ = leviathan_forward_ref(ids, params, g.config, mask_token_id=16)
    mask = torch.nn.Parameter(torch.full((512,), 0.25, device=device, dtype=torch.bfloat16))
    result = torch.where(ids.eq(16).unsqueeze(-1), mask, e)
    result.float().sum().backward()
    assert torch.isfinite(result).all()
    assert all(torch.isfinite(p.grad).all() for p in g.parameters())
    torch.testing.assert_close(mask.grad, torch.full_like(mask, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dot", ["0", "1"])
def test_constant_splines_have_no_coordinate_gradient(monkeypatch, dot):
    monkeypatch.setenv("LEV_DOT", dot)
    torch.manual_seed(1234)
    g = LeviathanGenerator(_config()).cuda()
    with torch.no_grad():
        g.head_spline_delta.fill_(0.1015625)
    ids = torch.tensor([[7, 13, 99, 1000]], device="cuda")
    params = {name: getattr(g, name) for name in NAMES}
    output = leviathan_embedding_compiler_safe(ids, params, g.config, g.knot_grid)
    output.backward(torch.randn_like(output))
    assert torch.count_nonzero(g.codebooks.grad) == 0
    assert torch.count_nonzero(g.head_proj_weight.grad) == 0
    assert torch.count_nonzero(g.head_norm_weight.grad) == 0
    assert torch.count_nonzero(g.head_norm_bias.grad) == 0
    assert torch.isfinite(g.head_spline_delta.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dot", ["0", "1"])
def test_exact_zero_matches_torch_epsilon_convention(monkeypatch, dot):
    monkeypatch.setenv("LEV_DOT", dot)
    torch.manual_seed(1234)
    g = LeviathanGenerator(_config()).cuda()
    with torch.no_grad():
        g.head_spline_delta.fill_(0.1015625)
        g.head_spline_delta[:, 0, :, :].fill_(-1)
        g.head_out_weight.zero_()
        g.head_out_weight[0, 0, 0] = 1
    ids = torch.tensor([[7, 13, 99, 1000]], device="cuda")
    params = {name: getattr(g, name) for name in NAMES}
    output = leviathan_embedding_compiler_safe(ids, params, g.config, g.knot_grid)
    grad = torch.zeros_like(output)
    grad[0, 0, 0] = 1
    output.backward(grad)
    assert torch.count_nonzero(g.head_spline_delta.grad[:, 0]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dot", ["0", "1"])
def test_masked_seed_overflow_does_not_contaminate_gradients(monkeypatch, dot):
    monkeypatch.setenv("LEV_DOT", dot)
    torch.manual_seed(1234)
    g = LeviathanGenerator(_config()).cuda()
    maximum = torch.finfo(torch.bfloat16).max
    with torch.no_grad():
        g.codebooks.zero_()
        g.codebooks[1, 0].fill_(maximum * 0.5)
        g.codebooks[2, 16].fill_(maximum * 0.75)
        g.codebooks[2, 1].fill_(-maximum * 0.5)
        g.head_spline_delta.zero_()
    model = _Observed(g)
    fn = torch.compile(model, fullgraph=True, mode="max-autotune")
    ids = torch.tensor([[16, 1, 1, 16]], device="cuda")
    output, (seed, _, _) = fn(ids)
    assert not torch.isfinite(seed.reshape(4, -1)[[0, 3]]).all()
    assert torch.isfinite(output).all()
    output.float().sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in g.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_noncontiguous_ids_keep_meap_rows_isolated():
    torch.manual_seed(1234)
    g = LeviathanGenerator(_config()).cuda()
    _masked_only_overflow(g)
    model = _Observed(g)
    ids = torch.tensor([[16, 1], [1, 16], [16, 1], [1, 16]], device="cuda").t()
    assert not ids.is_contiguous()
    output, _ = torch.compile(model, fullgraph=True)(ids)
    output.float().sum().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in g.parameters())
    torch.testing.assert_close(model.mask_embedding.grad, torch.full_like(model.mask_embedding, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_mask_id_survives_compiled_checkpoint_metadata():
    torch.manual_seed(1234)
    g = LeviathanGenerator(_config()).cuda()
    model = _Observed(g, mask_token_id=60000)
    ids = torch.full((1, 4), 60000, device="cuda", dtype=torch.long)
    output, _ = torch.compile(model, fullgraph=True)(ids)
    output.float().sum().backward()
    torch.testing.assert_close(model.mask_embedding.grad, torch.full_like(model.mask_embedding, 4))
    assert all(torch.count_nonzero(p.grad) == 0 for p in g.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_jtok_seed_is_not_replaced_by_meap():
    torch.manual_seed(1234)
    generator = LeviathanGenerator(_config()).cuda()
    model = _Observed(generator, with_seed=True)
    ids = torch.tensor([[16, 1, 16, 2]], device="cuda")
    _, seed, _ = model(ids)
    expected = generator._base_k_decompose(ids)
    expected_seed = torch.zeros_like(seed)
    for component in range(generator.k):
        expected_seed = expected_seed + generator.codebooks[component][expected[..., component]]
    torch.testing.assert_close(seed, expected_seed, rtol=0, atol=0)

    class _Loss(torch.nn.Module):
        def __init__(self, observed):
            super().__init__()
            self.observed = observed

        def forward(self, value):
            output, seed, _ = self.observed(value)
            return output.float().sum() + seed.float().sum()

    compiled = torch.compile(_Loss(model), fullgraph=True, mode="max-autotune")
    compiled(ids).backward()
    assert torch.isfinite(generator.codebooks.grad).all()
    assert generator.codebooks.grad.abs().sum() > 0
