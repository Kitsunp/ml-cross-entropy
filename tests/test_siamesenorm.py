"""Generic normalization regressions, independent of any model or router."""

import pytest
import torch

from cut_cross_entropy.siamesenorm import siamesenorm, siamesenorm_reference


def test_cpu_auto_and_shape_contract():
    x = torch.randn(3, 17, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    w = [torch.randn(17, requires_grad=True) for _ in range(4)]
    torch.testing.assert_close(
        siamesenorm(x, y, *w, backend="auto"), siamesenorm_reference(x, y, *w)
    )
    with pytest.raises(ValueError, match="identical"):
        siamesenorm(x, y[:, :-1], *w)
    with pytest.raises(ValueError, match="positive"):
        siamesenorm(x, y, *w, eps=float("nan"))
    with pytest.raises(ValueError, match="forced"):
        siamesenorm(x, y, *w, backend="triton")


def test_fixed_epilogue_contract():
    x = torch.randn(3, 17, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    w = torch.ones(17, requires_grad=True)
    post = torch.tensor(0.37)
    expected = (siamesenorm_reference(x, y, w, w, w) * post).bfloat16()
    actual = siamesenorm(x, y, w, w, w, post_scale=post, out_dtype=torch.bfloat16, backend="auto")
    torch.testing.assert_close(actual, expected)
    for invalid in (post.reshape(1), post.double(), post.clone().requires_grad_()):
        with pytest.raises(ValueError, match="fixed scalar"):
            siamesenorm(x, y, w, w, post_scale=invalid)
    with pytest.raises(ValueError, match="out_dtype"):
        siamesenorm(x, y, w, w, out_dtype=torch.float16)


def test_reference_vjp_keeps_epsilon_radial_term():
    """Verify the derived paired adjoint independently in double precision.

    A non-negligible epsilon distinguishes RMSNorm's true VJP from an
    incorrect exact tangent projection. Native GPU rounding has separate tests.
    """
    torch.manual_seed(186)
    x, y = [torch.randn(3, 7, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    wx, wy, wo, scale = [torch.randn(7, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    eps, post = 0.25, torch.tensor(0.37, dtype=torch.float64)
    g = torch.randn_like(x)
    output = siamesenorm_reference(x, y, wx, wy, wo, scale, eps=eps, post_scale=post)
    output.backward(g)
    with torch.no_grad():
        rx = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
        ry = (y.square().mean(-1, keepdim=True) + eps).rsqrt()
        a, b = x * rx * wx, y * ry * wy
        z = a * scale + b
        rz = (z.square().mean(-1, keepdim=True) + eps).rsqrt()
        gh = g * post
        qz = gh * wo
        gz = rz * qz - z * rz.pow(3) * (qz * z).mean(-1, keepdim=True)
        ga, gb = gz * scale, gz
        qx, qy = ga * wx, gb * wy
        dx = rx * qx - x * rx.pow(3) * (qx * x).mean(-1, keepdim=True)
        dy = ry * qy - y * ry.pow(3) * (qy * y).mean(-1, keepdim=True)
        grads = (
            dx,
            dy,
            (ga * x * rx).sum(0),
            (gb * y * ry).sum(0),
            (gh * z * rz).sum(0),
            (gz * a).sum(0),
        )
        for value, expected in zip((x, y, wx, wy, wo, scale), grads):
            torch.testing.assert_close(value.grad, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize(
    "stream_dtype,affine_dtype,consumer_dtype",
    [
        (torch.float32, torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "outer,scaled,post_value",
    [(True, False, 0.37), (False, True, -0.71), (True, True, 0.0), (False, False, None)],
)
def test_fused_epilogue_native_vjp(
    monkeypatch, backend, stream_dtype, affine_dtype, consumer_dtype, outer, scaled, post_value
):
    """Keep native mix rounding/VJP even when consumer dtype differs.

    The partial last row group and 129 groups exercise both parameter passes.
    Zero post-scale must yield exactly zero gradients, not unused gradients.
    """
    from cut_cross_entropy.siamesenorm import _cute, _triton

    if backend == "cute" and not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")

    def forbidden(*args, **kwargs):
        raise AssertionError("epilogue must not use the other backend")

    other = _triton if backend == "cute" else _cute
    monkeypatch.setattr(other, "forward", forbidden)
    monkeypatch.setattr(other, "backward", forbidden)
    torch.manual_seed(183)
    args = [torch.randn(1025, 512, device="cuda", dtype=stream_dtype) for _ in range(2)]
    args += [torch.randn(512, device="cuda", dtype=affine_dtype) * 0.2 + 1 for _ in range(3)]
    # A native-dtype scale keeps the intended mix dtype, even with FP32 affines.
    args += [torch.randn(512, device="cuda", dtype=stream_dtype) * 0.2 + 1]
    args[4] = args[4] if outer else None
    args[5] = args[5] if scaled else None
    args = [a.requires_grad_() if a is not None else None for a in args]
    refs = [a.detach().clone().requires_grad_() if a is not None else None for a in args]
    post = (
        torch.tensor(post_value, device="cuda", dtype=stream_dtype)
        if post_value is not None
        else None
    )
    expected = siamesenorm_reference(*refs)
    if post is not None:
        expected = expected * post
    expected = expected.to(consumer_dtype)
    actual = siamesenorm(*args, backend=backend, post_scale=post, out_dtype=consumer_dtype)
    # A FP32 consumer does not undo prior BF16 rounding. Isolate the fusion
    # against the same backend with an ordinary Torch scalar multiply/cast.
    controls = [a.detach().clone().requires_grad_() if a is not None else None for a in args]
    unfused = siamesenorm(*controls, backend=backend)
    if post is not None:
        unfused = unfused * post
    unfused = unfused.to(consumer_dtype)
    torch.testing.assert_close(actual, unfused, rtol=0, atol=0)
    assert actual.dtype == consumer_dtype
    g = torch.randn_like(actual)
    actual.backward(g)
    expected.backward(g)
    unfused.backward(g)
    for a, b in zip(args, controls):
        if a is not None:
            rms = b.grad.float().square().mean().sqrt().clamp_min(1e-8)
            error = (a.grad.float() - b.grad.float()).square().mean().sqrt() / rms
            # FP32 reduction layout can differ when G changes storage dtype;
            # do not require identical summation bits across those layouts.
            assert error.item() < (2e-6 if a.dtype == torch.float32 else 3e-5)
    for a, b in [(actual, expected)] + [
        (a.grad, b.grad) for a, b in zip(args, refs) if a is not None
    ]:
        assert a.dtype == b.dtype
        assert torch.isfinite(a).all()
        if post_value == 0.0:
            assert torch.count_nonzero(a) == 0
        rms = b.float().square().mean().sqrt().clamp_min(1e-8)
        error = (a.float() - b.float()).square().mean().sqrt() / rms
        tolerance = 2e-5 if stream_dtype == a.dtype == torch.float32 else 3e-3
        assert error.item() < tolerance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("post_value", [0.0, 0.37])
def test_fused_epilogue_autocast_consumer_preserves_continuation(backend, post_value):
    """An existing Linear autocast is movable; unrelated state loss is not.

    Compare identical backend math with and without the epilogue, including
    consumer parameter gradients and independent gradients of the live streams.
    """
    import copy

    from cut_cross_entropy.siamesenorm import _cute

    if backend == "cute" and not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")
    torch.manual_seed(185)
    args = [torch.randn(17, 512, device="cuda", requires_grad=True) for _ in range(2)]
    args += [
        torch.ones(512, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    ]
    refs = [a.detach().clone().requires_grad_() for a in args]
    consumer = torch.nn.Linear(512, 37, device="cuda", dtype=torch.bfloat16)
    control = copy.deepcopy(consumer)
    post = torch.tensor(post_value, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = control(siamesenorm(*refs, backend=backend) * post).sin()
        actual = consumer(
            siamesenorm(*args, backend=backend, post_scale=post, out_dtype=torch.bfloat16)
        ).sin()
        # This separate activation tap must keep its original FP32 state.
        loss = actual.float().square().mean() + 0.01 * args[0].square().mean()
        expected_loss = expected.float().square().mean() + 0.01 * refs[0].square().mean()
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    leaves = list(zip(args, refs)) + list(zip(consumer.parameters(), control.parameters()))
    for a, b in leaves:
        rms = b.grad.float().square().mean().sqrt().clamp_min(1e-8)
        error = (a.grad.float() - b.grad.float()).square().mean().sqrt() / rms
        assert error.item() < (2e-6 if a.dtype == torch.float32 else 3e-5)
    if post_value == 0.0:
        # Normalized branch is silent; the continuation/auxiliary tap is not.
        torch.testing.assert_close(args[0].grad, 0.02 * args[0] / args[0].numel())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_compiled_epilogue_tensor_values_do_not_recompile(monkeypatch, backend):
    """Scale values may change; dtype, shape and epilogue topology stay fixed."""
    from torch._dynamo.utils import counters

    from cut_cross_entropy.siamesenorm import _cute

    if backend == "cute" and not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    torch.manual_seed(184)
    args = [torch.randn(17, 512, device="cuda", requires_grad=True) for _ in range(2)]
    args += [
        torch.ones(512, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    ]
    post = torch.tensor(0.37, device="cuda")
    g = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(
        lambda *values: siamesenorm(
            *values[:-1], post_scale=values[-1], out_dtype=torch.bfloat16, backend=backend
        ),
        mode="max-autotune",
        fullgraph=True,
    )
    for _ in range(5):
        torch.compiler.cudagraph_mark_step_begin()
        for a in args:
            a.grad = None
        compiled(*args, post).backward(g)
    warmed = (
        counters["stats"]["unique_graphs"],
        sum(counters["graph_break"].values()),
        len(_cute._CACHE),
    )
    for value in (0.0, -0.71, 0.375):
        torch.compiler.cudagraph_mark_step_begin()
        post.fill_(value)
        for a in args:
            a.grad = None
        refs = [a.detach().clone().requires_grad_() for a in args]
        expected = (siamesenorm_reference(*refs) * post).bfloat16()
        expected.backward(g)
        actual = compiled(*args, post)
        actual.backward(g)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
        for a, b in zip(args, refs):
            rms = b.grad.float().square().mean().sqrt().clamp_min(1e-8)
            error = (a.grad.float() - b.grad.float()).square().mean().sqrt() / rms
            assert error.item() < (2e-5 if a.dtype == torch.float32 else 3e-3)
        assert warmed == (
            counters["stats"]["unique_graphs"],
            sum(counters["graph_break"].values()),
            len(_cute._CACHE),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("outer,scale", [(True, True), (True, False), (False, True)])
def test_native_dtype_boundaries(backend, dtype, outer, scale):
    """Detect extra normalization casts and lost rounding before scaled addition."""
    from cut_cross_entropy.siamesenorm import _cute

    if backend == "cute" and not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")
    torch.manual_seed(103)
    args = [torch.randn(9, 64, device="cuda", dtype=dtype) for _ in range(2)]
    args += [torch.randn(64, device="cuda", dtype=dtype) * 0.2 + 1 for _ in range(4)]
    args[4] = args[4] if outer else None
    args[5] = args[5] if scale else None
    args = [a.requires_grad_() if a is not None else None for a in args]
    refs = [a.detach().clone().requires_grad_() if a is not None else None for a in args]
    actual = siamesenorm(*args, backend=backend)
    expected = siamesenorm_reference(*refs)
    g = torch.randn_like(actual)
    actual.backward(g)
    expected.backward(g)
    tolerance = 2e-5 if dtype == torch.float32 else 3e-3
    for a, b in [(actual, expected)] + [
        (a.grad, b.grad) for a, b in zip(args, refs) if a is not None
    ]:
        err = (a.float() - b.float()).square().mean().sqrt()
        rms = b.float().square().mean().sqrt().clamp_min(1e-8)
        assert (err / rms).item() < tolerance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_odd_width_noncontiguous_and_offset():
    x = torch.randn(5, 67, device="cuda")
    y = torch.randn_like(x)
    w = torch.ones(67, device="cuda")
    torch.testing.assert_close(
        siamesenorm(x, y, w, w, backend="triton"), siamesenorm_reference(x, y, w, w)
    )
    torch.testing.assert_close(
        siamesenorm(x.T, y.T, w[:5], w[:5], backend="auto"),
        siamesenorm_reference(x.T, y.T, w[:5], w[:5]),
    )
    # A contiguous view may still have a nonzero offset and scalar alignment.
    x = torch.randn(1 + 5 * 64, device="cuda")[1:].reshape(5, 64)
    y = x * 0.7
    w = torch.ones(64, device="cuda")
    from cut_cross_entropy.siamesenorm import _cute

    if _cute.is_available():
        torch.testing.assert_close(
            siamesenorm(x, y, w, w, backend="cute"), siamesenorm_reference(x, y, w, w)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_independent_backend_and_hierarchical_reduction(monkeypatch, backend, dtype):
    """129 row groups require two reduction levels and a masked final group."""
    from cut_cross_entropy.siamesenorm import _cute, _triton

    if backend == "cute" and not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")

    def forbidden(*args, **kwargs):
        raise AssertionError("a backend must not call the other backend")

    other = _triton if backend == "cute" else _cute
    monkeypatch.setattr(other, "forward", forbidden)
    monkeypatch.setattr(other, "backward", forbidden)
    torch.manual_seed(81)
    args = [torch.randn(1025, 64, device="cuda", dtype=dtype) for _ in range(2)]
    args += [torch.randn(64, device="cuda", dtype=dtype) * 0.2 + 1 for _ in range(4)]
    args = [a.requires_grad_() for a in args]
    refs = [a.detach().clone().requires_grad_() for a in args]
    actual = siamesenorm(*args, backend=backend)
    expected = siamesenorm_reference(*refs)
    g = torch.randn_like(actual)
    actual.backward(g)
    expected.backward(g)
    for a, b in [(actual, expected)] + [(a.grad, b.grad) for a, b in zip(args, refs)]:
        rms = b.float().square().mean().sqrt().clamp_min(1e-8)
        error = (a.float() - b.float()).square().mean().sqrt() / rms
        assert error.item() < (2e-5 if dtype == torch.float32 else 3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("affine_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [64, 512])
def test_compiled_backend_dispatch_and_stable_graph(monkeypatch, backend, affine_dtype, width):
    """AOT backward must retain backend selection, including CUDA Graph replay."""
    from torch._dynamo.utils import counters

    from cut_cross_entropy.siamesenorm import _cute, _triton

    if backend == "cute" and not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")

    def forbidden(*args, **kwargs):
        raise AssertionError("compiled backward must retain its selected backend")

    other = _triton if backend == "cute" else _cute
    monkeypatch.setattr(other, "forward", forbidden)
    monkeypatch.setattr(other, "backward", forbidden)
    if backend == "triton":
        import cut_cross_entropy.siamesenorm as norm_module

        if norm_module.triton_op is not None:
            # A Torch 2.14 compile must see launches rather than quietly using
            # the opaque compatibility boundary (forward or backward).
            monkeypatch.setattr(norm_module, "_forward", forbidden)
            monkeypatch.setattr(norm_module, "_backward_op", forbidden)
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    torch.manual_seed(82)
    args = [torch.randn(1025, width, device="cuda", requires_grad=True) for _ in range(2)]
    args += [
        torch.ones(width, device="cuda", dtype=affine_dtype, requires_grad=True) for _ in range(4)
    ]
    refs = [a.detach().clone().requires_grad_() for a in args]
    g = torch.randn_like(args[0])
    expected = siamesenorm_reference(*refs)
    expected.backward(g)
    compiled = torch.compile(
        lambda *values: siamesenorm(*values, backend=backend),
        mode="max-autotune",
        fullgraph=True,
    )
    for _ in range(5):
        torch.compiler.cudagraph_mark_step_begin()
        for a in args:
            a.grad = None
        compiled(*args).backward(g)
    warmed = (counters["stats"]["unique_graphs"], sum(counters["graph_break"].values()))
    torch.compiler.cudagraph_mark_step_begin()
    for a in args:
        a.grad = None
    actual = compiled(*args)
    actual.backward(g)
    assert warmed == (counters["stats"]["unique_graphs"], sum(counters["graph_break"].values()))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for a, b in zip(args, refs):
        rms = b.grad.float().square().mean().sqrt().clamp_min(1e-8)
        error = (a.grad.float() - b.grad.float()).square().mean().sqrt() / rms
        assert error.item() < (2e-5 if a.dtype == torch.float32 else 3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [32, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cute_supported_width_edges(width, dtype):
    """Cover one item per lane and the advertised shared-memory width limit."""
    from cut_cross_entropy.siamesenorm import _cute

    if not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")
    torch.manual_seed(83)
    args = [torch.randn(17, width, device="cuda", dtype=dtype) for _ in range(2)]
    args += [torch.randn(width, device="cuda", dtype=dtype) * 0.2 + 1 for _ in range(4)]
    args = [a.requires_grad_() for a in args]
    refs = [a.detach().clone().requires_grad_() for a in args]
    actual = siamesenorm(*args, backend="cute")
    expected = siamesenorm_reference(*refs)
    g = torch.randn_like(actual)
    actual.backward(g)
    expected.backward(g)
    for a, b in [(actual, expected)] + [(a.grad, b.grad) for a, b in zip(args, refs)]:
        rms = b.float().square().mean().sqrt().clamp_min(1e-8)
        error = (a.float() - b.float()).square().mean().sqrt() / rms
        assert error.item() < (2e-5 if dtype == torch.float32 else 3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize(
    "stream_x,stream_y,affine,scale_dtype",
    [
        (torch.float32, torch.float32, torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32),
        (torch.bfloat16, torch.float32, torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.bfloat16, torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("outer,scaled", [(True, True), (True, False), (False, True)])
def test_mixed_precision_streams_and_affines(
    backend, stream_x, stream_y, affine, scale_dtype, outer, scaled
):
    """Residual streams can promote to FP32 while RMSNorm weights stay BF16.

    Also cover asymmetric streams and FP32 scales promoting a BF16 sum. The
    output and every leaf gradient must keep native Torch dtype semantics.
    """
    from cut_cross_entropy.siamesenorm import _cute

    if backend == "cute" and not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")
    torch.manual_seed(84)
    args = [
        torch.randn(17, 64, device="cuda", dtype=stream_x),
        torch.randn(17, 64, device="cuda", dtype=stream_y),
    ]
    args += [torch.randn(64, device="cuda", dtype=affine) * 0.2 + 1 for _ in range(3)]
    args += [torch.randn(64, device="cuda", dtype=scale_dtype) * 0.2 + 1]
    args[4] = args[4] if outer else None
    args[5] = args[5] if scaled else None
    args = [a.requires_grad_() if a is not None else None for a in args]
    refs = [a.detach().clone().requires_grad_() if a is not None else None for a in args]
    actual = siamesenorm(*args, backend=backend)
    expected = siamesenorm_reference(*refs)
    assert actual.dtype == expected.dtype
    g = torch.randn_like(expected)
    actual.backward(g)
    expected.backward(g)
    for a, b in [(actual, expected)] + [
        (a.grad, b.grad) for a, b in zip(args, refs) if a is not None
    ]:
        assert a.dtype == b.dtype
        assert torch.isfinite(a).all()
        rms = b.float().square().mean().sqrt().clamp_min(1e-8)
        error = (a.float() - b.float()).square().mean().sqrt() / rms
        assert error.item() < (2e-5 if a.dtype == torch.float32 else 3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "outer,scaled", [(False, False), (False, True), (True, False), (True, True)]
)
def test_compact_optional_parameter_workspace(monkeypatch, backend, dtype, outer, scaled):
    """Only existing affine gradients need row partials and shared reductions."""
    from cut_cross_entropy.siamesenorm import _cute

    if backend == "cute" and not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")
    rows, width = 1025, 64
    planes = 2 + int(outer) + int(scaled)
    groups, chunks = 129, 2
    sizes = []
    original_empty = torch.empty

    def tracked_empty(*args, **kwargs):
        if args and isinstance(args[0], tuple):
            sizes.append(args[0])
        return original_empty(*args, **kwargs)

    torch.manual_seed(85)
    args = [
        torch.randn(rows, width, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)
    ]
    args += [torch.ones(width, device="cuda", dtype=dtype, requires_grad=True) for _ in range(4)]
    args[4] = args[4] if outer else None
    args[5] = args[5] if scaled else None
    refs = [a.detach().clone().requires_grad_() if a is not None else None for a in args]
    monkeypatch.setattr(torch, "empty", tracked_empty)
    actual = siamesenorm(*args, backend=backend)
    expected = siamesenorm_reference(*refs)
    g = torch.randn_like(expected)
    actual.backward(g)
    expected.backward(g)
    if backend == "cute":
        assert (planes * groups * width,) in sizes
        assert (planes, chunks * width) in sizes
        if planes < 4:
            assert (4 * groups * width,) not in sizes
    else:
        assert (planes, groups, width) in sizes
        assert (planes, chunks, width) in sizes
        if planes < 4:
            assert (4, groups, width) not in sizes
            assert (4, chunks, width) not in sizes
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        for a, b in zip(args, refs):
            if a is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-4, rtol=2e-4)
    else:
        for a, b in [(actual, expected)] + [
            (a.grad, b.grad) for a, b in zip(args, refs) if a is not None
        ]:
            assert a.dtype == b.dtype and torch.isfinite(a).all()
            error = (a.float() - b.float()).square().mean().sqrt()
            rms = b.float().square().mean().sqrt().clamp_min(1e-8)
            assert (error / rms).item() < 3e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 7, 9, 1025])
@pytest.mark.parametrize("scaled", [False, True])
def test_cute_bf16_single_warp_masked_rows(monkeypatch, rows, scaled):
    """A row-uniform guard must not leak unwritten parameter scratch.

    Seven and nine rows mask opposite halves of the two-row owner loop;
    1025 rows also exercise the second parameter reduction level. Keep the
    BF16 arithmetic and affine cast boundaries, not just FP32 stream tests.
    """
    from cut_cross_entropy.siamesenorm import _cute, _triton

    if not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")

    def forbidden(*args, **kwargs):
        raise AssertionError("CuTe must remain independent of Triton")

    monkeypatch.setattr(_triton, "forward", forbidden)
    monkeypatch.setattr(_triton, "backward", forbidden)
    assert _cute._backward_row_lanes(512) == 32
    torch.manual_seed(185)
    args = [torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    args += [torch.randn(512, device="cuda", dtype=torch.bfloat16) * 0.1 + 1 for _ in range(4)]
    args[5] = args[5] if scaled else None
    args = [a.requires_grad_() if a is not None else None for a in args]
    refs = [a.detach().clone().requires_grad_() if a is not None else None for a in args]
    actual = siamesenorm(*args, backend="cute")
    expected = siamesenorm_reference(*refs)
    g = torch.randn_like(expected)
    actual.backward(g)
    expected.backward(g)
    for a, b in [(actual, expected)] + [
        (a.grad, b.grad) for a, b in zip(args, refs) if a is not None
    ]:
        assert a.dtype == b.dtype
        assert torch.isfinite(a).all()
        rms = b.float().square().mean().sqrt().clamp_min(1e-8)
        error = (a.float() - b.float()).square().mean().sqrt() / rms
        assert error.item() < 3e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 7, 9, 1025])
@pytest.mark.parametrize("width", [512, 544])
@pytest.mark.parametrize(
    "outer,scaled", [(True, True), (True, False), (False, True), (False, False)]
)
def test_cute_wide_row_cooperation_and_partial_blocks(monkeypatch, rows, width, outer, scaled):
    """No CTA barrier may be conditional on a partial final row owner.

    Width 544 is legal but not divisible by 64, so it retains one warp per row.
    A single valid owner, a missing second row and a two-level reduction are
    different synchronization/indexing cases, not benchmark repetitions.
    """
    from cut_cross_entropy.siamesenorm import _cute

    if not _cute.is_available():
        pytest.skip("CuTe optional dependency is not installed")
    # This layout remains an explicit numerical experiment: increased
    # potential occupancy did not improve its measured real-training latency.
    monkeypatch.setattr(
        _cute, "_backward_row_lanes", lambda d: 64 if d >= 512 and d % 64 == 0 else 32
    )
    torch.manual_seed(86)
    args = [torch.randn(rows, width, device="cuda", requires_grad=True) for _ in range(2)]
    args += [
        (torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).requires_grad_()
        for _ in range(4)
    ]
    args[4] = args[4] if outer else None
    args[5] = args[5] if scaled else None
    refs = [a.detach().clone().requires_grad_() if a is not None else None for a in args]
    actual = siamesenorm(*args, backend="cute")
    expected = siamesenorm_reference(*refs)
    g = torch.randn_like(expected)
    actual.backward(g)
    expected.backward(g)
    for a, b in [(actual, expected)] + [
        (a.grad, b.grad) for a, b in zip(args, refs) if a is not None
    ]:
        assert torch.isfinite(a).all()
        error = (a.float() - b.float()).square().mean().sqrt()
        rms = b.float().square().mean().sqrt().clamp_min(1e-8)
        assert (error / rms).item() < (2e-5 if a.dtype == torch.float32 else 3e-3)
