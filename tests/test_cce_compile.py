from __future__ import annotations

import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.cce_compile import _cce_backward_op, _cce_forward_op

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    e = torch.randn(
        4,
        32,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    c = torch.randn(
        2048,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    targets = torch.randint(
        2048,
        (4, 32),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )
    targets[0, 24:] = -100
    targets[1, 29:] = -100
    return e, c, targets


def _call(
    e,
    c,
    targets,
    mile_enabled: bool,
    mu_loss_enabled: bool,
    *,
    bias=None,
    return_loss_metrics: bool = True,
):
    return linear_cross_entropy(
        e,
        c,
        targets,
        bias=bias,
        shift=1,
        impl="cce_kahan_full_c",
        reduction="mean",
        return_loss_metrics=return_loss_metrics,
        mile_enabled=mile_enabled,
        mile_gamma=1.0,
        mu_loss_enabled=mu_loss_enabled,
        mu_loss_lambda=1e-4,
    )


def _operator_args(
    e,
    c,
    targets,
    *,
    bias=None,
    filter_eps=None,
    filter_e_grad=False,
    filter_c_grad=False,
    compute_dtype_is_bf16=None,
    forward_used_autocast=False,
):
    if compute_dtype_is_bf16 is None:
        compute_dtype_is_bf16 = e.dtype == torch.bfloat16
    return (
        e,
        c,
        targets,
        bias,
        e.requires_grad,
        c.requires_grad,
        bias.requires_grad if bias is not None else False,
        -100,
        None,
        1,
        False,
        filter_eps,
        True,
        True,
        filter_e_grad,
        filter_c_grad,
        True,
        False,
        1.0,
        False,
        1e-4,
        compute_dtype_is_bf16,
        forward_used_autocast,
    )


def _backward_operator_args(forward_args):
    outputs = _cce_forward_op(*forward_args)
    return (
        torch.ones_like(outputs[0]),
        *forward_args[:4],
        *outputs[2:9],
        forward_args[-1],
        *forward_args[4:-2],
    )


@pytest.mark.parametrize(
    ("mile_enabled", "mu_loss_enabled"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_compiler_boundary_matches_eager(mile_enabled: bool, mu_loss_enabled: bool):
    base_e, base_c, targets = _inputs()
    eager_e = base_e.clone().requires_grad_(True)
    eager_c = base_c.clone().requires_grad_(True)
    compiled_e = base_e.clone().requires_grad_(True)
    compiled_c = base_c.clone().requires_grad_(True)

    eager_loss, eager_metrics = _call(eager_e, eager_c, targets, mile_enabled, mu_loss_enabled)
    eager_loss.backward()

    def tail(e, c, labels):
        loss, metrics = _call(e, c, labels, mile_enabled, mu_loss_enabled)
        # Verify that Dynamo can continue the model graph after CCE rather than
        # accepting a graph that merely terminates at the custom operator.
        return loss + e.sum() * 0.0, metrics

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        explanation = torch._dynamo.explain(tail)(compiled_e, compiled_c, targets)
        compiled = torch.compile(tail, fullgraph=True, mode="reduce-overhead")
        compiled_loss, compiled_metrics = compiled(compiled_e, compiled_c, targets)
        compiled_loss.backward()
    torch.cuda.synchronize()

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=2e-4, atol=2e-4)
    for name in eager_metrics:
        torch.testing.assert_close(
            compiled_metrics[name], eager_metrics[name], rtol=2e-4, atol=2e-4
        )
    # FP32 atomics and MiLe's parallel reductions are order-nondeterministic;
    # compare their numerical contract rather than requiring bit identity.
    for compiled_grad, eager_grad in (
        (compiled_e.grad, eager_e.grad),
        (compiled_c.grad, eager_c.grad),
    ):
        assert torch.isfinite(compiled_grad).all()
        relative_l2 = (
            compiled_grad.float() - eager_grad.float()
        ).norm() / eager_grad.float().norm()
        assert relative_l2 < 3e-3


def test_compiler_boundary_does_not_specialize_on_valid_label_count():
    base_e, base_c, base_targets = _inputs()
    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    def tail(e, c, labels):
        loss, _metrics = _call(e, c, labels, True, True)
        return loss + e.sum() * 0.0

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(tail, backend=counting_backend, fullgraph=True)
        for first_padding_position in (0, 8, 16, 24, 31):
            e = base_e.clone().requires_grad_(True)
            c = base_c.clone().requires_grad_(True)
            targets = base_targets.clone()
            targets[:, first_padding_position:] = -100
            compiled(e, c, targets).backward()
    torch.cuda.synchronize()
    assert compile_count == 1


@pytest.mark.parametrize(
    ("mile_enabled", "mu_loss_enabled"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_compiler_boundary_fp32_autocast_matches_eager(
    mile_enabled: bool,
    mu_loss_enabled: bool,
):
    base_e, base_c, targets = _inputs()
    base_e = base_e.float()
    base_c = base_c.float()
    eager_e = base_e.clone().requires_grad_(True)
    eager_c = base_c.clone().requires_grad_(True)
    compiled_e = base_e.clone().requires_grad_(True)
    compiled_c = base_c.clone().requires_grad_(True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_loss, eager_metrics = _call(
            eager_e, eager_c, targets, mile_enabled, mu_loss_enabled
        )
    eager_loss.backward()

    def tail(e, c, labels):
        loss, metrics = _call(e, c, labels, mile_enabled, mu_loss_enabled)
        return loss + e.sum() * 0.0, metrics

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(tail, fullgraph=True, mode="reduce-overhead")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            compiled_loss, compiled_metrics = compiled(
                compiled_e, compiled_c, targets
            )
        compiled_loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(compiled_loss, eager_loss, rtol=2e-4, atol=2e-4)
    for name in eager_metrics:
        torch.testing.assert_close(
            compiled_metrics[name], eager_metrics[name], rtol=2e-4, atol=2e-4
        )
    for compiled_grad, eager_grad in (
        (compiled_e.grad, eager_e.grad),
        (compiled_c.grad, eager_c.grad),
    ):
        assert compiled_grad.dtype == torch.float32
        assert torch.isfinite(compiled_grad).all()
        relative_l2 = (
            compiled_grad.float() - eager_grad.float()
        ).norm() / eager_grad.float().norm()
        assert relative_l2 < 3e-3


@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
def test_compiler_boundary_fp32_autocast_does_not_specialize_on_valid_count(
    autocast_dtype,
):
    """Reproduce FP32 RMSNorm outputs with changing padding under autocast.

    Before the regression fix, the public dtype guard rejected these otherwise
    supported inputs. Dynamo then traced CCE's data-dependent compaction and
    specialized the Triton path for every valid-token count, reaching its
    default recompile limit on the ninth distinct batch.
    """
    base_e, base_c, base_targets = _inputs()
    base_e = base_e.float()
    base_c = base_c.float()
    compile_count = 0
    saw_cce_boundary = False

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count, saw_cce_boundary
        compile_count += 1
        saw_cce_boundary |= any(
            node.op == "call_function" and "cce_forward" in str(node.target)
            for node in graph_module.graph.nodes
        )
        return graph_module.forward

    def tail(e, c, labels):
        loss, metrics = _call(e, c, labels, True, True)
        return loss + metrics["mu_loss"] * 0.0 + e.sum() * 0.0

    # Ten distinct counts cross the same default eight-recompile threshold that
    # the server reached when its ninth valid-token count entered CCE.
    first_padding_positions = (2, 3, 6, 9, 12, 15, 18, 21, 24, 31)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(tail, backend=counting_backend, fullgraph=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            for first_padding_position in first_padding_positions:
                e = base_e.clone().requires_grad_(True)
                c = base_c.clone().requires_grad_(True)
                targets = base_targets.clone()
                targets[:, first_padding_position:] = -100
                loss = compiled(e, c, targets)
                loss.backward()
                assert torch.isfinite(loss)
                assert e.grad.dtype == torch.float32
                assert c.grad.dtype == torch.float32
    torch.cuda.synchronize()

    assert saw_cce_boundary
    assert compile_count == 1


def test_compiler_boundary_supports_bias_without_metrics():
    base_e, base_c, targets = _inputs()
    generator = torch.Generator(device="cuda").manual_seed(5678)
    base_bias = torch.randn(
        base_c.size(0), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    eager_e = base_e.clone().requires_grad_(True)
    eager_c = base_c.clone().requires_grad_(True)
    eager_bias = base_bias.clone().requires_grad_(True)
    compiled_e = base_e.clone().requires_grad_(True)
    compiled_c = base_c.clone().requires_grad_(True)
    compiled_bias = base_bias.clone().requires_grad_(True)

    eager_loss = _call(
        eager_e,
        eager_c,
        targets,
        False,
        False,
        bias=eager_bias,
        return_loss_metrics=False,
    )
    eager_loss.backward()

    def tail(e, c, labels, bias):
        loss = _call(
            e,
            c,
            labels,
            False,
            False,
            bias=bias,
            return_loss_metrics=False,
        )
        return loss + e.sum() * 0.0

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(tail, fullgraph=True, mode="reduce-overhead")
        compiled_loss = compiled(compiled_e, compiled_c, targets, compiled_bias)
        compiled_loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(compiled_loss, eager_loss, rtol=2e-4, atol=2e-4)
    for compiled_grad, eager_grad in (
        (compiled_e.grad, eager_e.grad),
        (compiled_c.grad, eager_c.grad),
        (compiled_bias.grad, eager_bias.grad),
    ):
        assert torch.isfinite(compiled_grad).all()
        relative_l2 = (
            compiled_grad.float() - eager_grad.float()
        ).norm() / eager_grad.float().norm()
        assert relative_l2 < 3e-3


@pytest.mark.parametrize(
    ("input_dtype", "autocast_dtype"),
    [(torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)],
)
def test_compiler_boundary_auto_eps_uses_autocast_dtype(input_dtype, autocast_dtype):
    base_e, base_c, targets = _inputs()
    base_e = base_e.to(input_dtype)
    base_c = base_c.to(input_dtype)
    compiled_e = base_e.clone().requires_grad_(True)
    compiled_c = base_c.clone().requires_grad_(True)
    captured_filter_eps = []

    def inspect_backend(graph_module, _example_inputs):
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and "cce_forward" in str(node.target):
                captured_filter_eps.append(node.args[11])
        return graph_module.forward

    def tail(e, c, labels):
        return _call(
            e,
            c,
            labels,
            False,
            False,
            return_loss_metrics=False,
        )

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(tail, backend=inspect_backend, fullgraph=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            compiled_loss = compiled(compiled_e, compiled_c, targets)
    torch.cuda.synchronize()

    assert torch.isfinite(compiled_loss)
    assert captured_filter_eps == [torch.finfo(autocast_dtype).eps / 32]
    with torch.autocast("cuda", dtype=autocast_dtype):
        result = torch.library.opcheck(
            _cce_forward_op,
            _operator_args(
                compiled_e,
                compiled_c,
                targets,
                compute_dtype_is_bf16=autocast_dtype == torch.bfloat16,
                forward_used_autocast=True,
            ),
            test_utils=("test_schema", "test_faketensor"),
        )
    assert set(result.values()) == {"SUCCESS"}


def test_compiler_boundary_disables_filters_when_eps_is_none():
    e, c, targets = _inputs()
    e.requires_grad_(True)
    c.requires_grad_(True)
    captured_filter_config = []

    def inspect_backend(graph_module, _example_inputs):
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and "cce_forward" in str(node.target):
                captured_filter_config.append((node.args[11], node.args[14], node.args[15]))
        return graph_module.forward

    def tail(embeddings, classifier, labels):
        return linear_cross_entropy(
            embeddings,
            classifier,
            labels,
            shift=1,
            impl="cce_exact",
            reduction="mean",
        )

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(tail, backend=inspect_backend, fullgraph=True)
        loss = compiled(e, c, targets)
        loss.backward()
    torch.cuda.synchronize()

    assert captured_filter_config == [(None, False, False)]
    assert torch.isfinite(loss)
    assert torch.isfinite(e.grad).all()
    assert torch.isfinite(c.grad).all()


@pytest.mark.parametrize(
    ("mile_enabled", "mile_gamma", "mu_loss_enabled", "mu_loss_lambda", "message"),
    [
        (True, -1.0, False, 1e-4, "mile_gamma must be finite and non-negative"),
        (False, 1.0, True, float("nan"), "mu_loss_lambda must be finite and non-negative"),
    ],
)
def test_compiler_boundary_preserves_objective_validation(
    monkeypatch,
    mile_enabled,
    mile_gamma,
    mu_loss_enabled,
    mu_loss_lambda,
    message,
):
    e, c, targets = _inputs()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(ValueError, match=message):
        linear_cross_entropy(
            e,
            c,
            targets,
            shift=1,
            impl="cce_kahan_full_c",
            mile_enabled=mile_enabled,
            mile_gamma=mile_gamma,
            mu_loss_enabled=mu_loss_enabled,
            mu_loss_lambda=mu_loss_lambda,
        )


def test_compiler_boundary_preserves_shape_validation(monkeypatch):
    e, c, targets = _inputs()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(AssertionError):
        linear_cross_entropy(
            e[:, :-1],
            c,
            targets,
            shift=1,
            impl="cce_kahan_full_c",
        )


def test_compiler_operator_registration_contract():
    e, c, targets = _inputs()
    e.requires_grad_(True)
    c.requires_grad_(True)
    args = _operator_args(e, c, targets)
    result = torch.library.opcheck(_cce_forward_op, args, rtol=3e-3, atol=3e-3)
    assert set(result.values()) == {"SUCCESS"}


def test_compiler_operator_noncontiguous_embedding_contract():
    base_e, c, targets = _inputs()
    e = base_e.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not e.is_contiguous()
    e.requires_grad_(True)
    c.requires_grad_(True)
    forward_args = _operator_args(e, c, targets)
    result = torch.library.opcheck(
        _cce_backward_op,
        _backward_operator_args(forward_args),
        test_utils=("test_schema", "test_faketensor"),
        rtol=3e-3,
        atol=3e-3,
    )
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.parametrize("gradient_owner", ["c", "bias"])
def test_compiler_operator_partial_gradient_logit_avg_contract(gradient_owner):
    e, c, targets = _inputs()
    bias = torch.randn(c.size(0), device="cuda", dtype=c.dtype)
    c.requires_grad_(gradient_owner == "c")
    bias.requires_grad_(gradient_owner == "bias")
    result = torch.library.opcheck(
        _cce_forward_op,
        _operator_args(
            e,
            c,
            targets,
            bias=bias,
            filter_eps=1e-4,
            filter_e_grad=True,
            filter_c_grad=False,
        ),
        test_utils=("test_schema", "test_faketensor"),
        rtol=3e-3,
        atol=3e-3,
    )
    assert set(result.values()) == {"SUCCESS"}
