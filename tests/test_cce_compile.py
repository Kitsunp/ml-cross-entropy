from __future__ import annotations

import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.cce_compile import _cce_forward_op

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


def test_compiler_operator_registration_contract():
    e, c, targets = _inputs()
    e.requires_grad_(True)
    c.requires_grad_(True)
    args = (
        e,
        c,
        targets,
        None,
        True,
        True,
        False,
        -100,
        None,
        1,
        False,
        None,
        True,
        True,
        False,
        False,
        True,
        False,
        1.0,
        False,
        1e-4,
    )
    result = torch.library.opcheck(_cce_forward_op, args, rtol=3e-3, atol=3e-3)
    assert set(result.values()) == {"SUCCESS"}
