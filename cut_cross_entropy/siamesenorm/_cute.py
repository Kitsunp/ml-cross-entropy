"""Optional independent CuTe forward/backward for paired normalization."""

from __future__ import annotations

import threading

import torch

from .reference import output_dtype

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream
except ImportError:
    cuda = cutlass = cute = from_dlpack = make_fake_stream = None


if cute is not None:

    @cute.jit
    def _rounded(value: cutlass.Float32, bf16: cutlass.Constexpr[bool]):
        if cutlass.const_expr(bf16):
            # A real conversion boundary, not an optimizable cast pair: an
            # LLVM contraction must not replace round(a*s)+b by round(a*s+b).
            return cutlass.Float32(cutlass.BFloat16(cute.arch.cvt_f32_bf16(value.ir_value())))
        else:
            return value

    @cute.kernel
    def _kernel(
        X: cute.Tensor,
        Y: cute.Tensor,
        WX: cute.Tensor,
        WY: cute.Tensor,
        WO: cute.Tensor,
        S: cute.Tensor,
        Post: cute.Tensor,
        O: cute.Tensor,
        Stats: cute.Tensor,
        eps: cutlass.Constexpr[float],
        outer: cutlass.Constexpr[bool],
        scale: cutlass.Constexpr[bool],
        post: cutlass.Constexpr[bool],
        mix_bf16: cutlass.Constexpr[bool],
    ):
        t, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = t % 32
        row = block * 4 + t // 32
        if row < X.shape[0]:
            width = X.shape[1]
            items = width // 32
            xs = cute.make_rmem_tensor(items, cutlass.Float32)
            ys = cute.make_rmem_tensor(items, cutlass.Float32)
            mix = cute.make_rmem_tensor(items, cutlass.Float32)
            xx = cutlass.Float32(0)
            yy = cutlass.Float32(0)
            for i in cutlass.range_constexpr(items):
                c = lane + i * 32
                x = cutlass.Float32(X[row, c])
                y = cutlass.Float32(Y[row, c])
                xs[i], ys[i] = x, y
                xx = xx + x * x
                yy = yy + y * y
            rx = cute.math.rsqrt(cute.arch.warp_reduction_sum(xx) / width + eps)
            ry = cute.math.rsqrt(cute.arch.warp_reduction_sum(yy) / width + eps)
            zz = cutlass.Float32(0)
            for i in cutlass.range_constexpr(items):
                c = lane + i * 32
                a = xs[i] * rx
                b = ys[i] * ry
                a = _rounded(a * cutlass.Float32(WX[c]), X.element_type == cutlass.BFloat16)
                b = _rounded(b * cutlass.Float32(WY[c]), Y.element_type == cutlass.BFloat16)
                if cutlass.const_expr(scale):
                    a = _rounded(
                        a * cutlass.Float32(S[c]),
                        X.element_type == cutlass.BFloat16 and S.element_type == cutlass.BFloat16,
                    )
                z = _rounded(a + b, mix_bf16)
                mix[i] = z
                zz = zz + z * z
            rz = cutlass.Float32(1)
            if cutlass.const_expr(outer):
                rz = cute.math.rsqrt(cute.arch.warp_reduction_sum(zz) / width + eps)
            for i in cutlass.range_constexpr(items):
                c = lane + i * 32
                z = cutlass.Float32(mix[i])
                if cutlass.const_expr(outer):
                    z = z * rz * cutlass.Float32(WO[c])
                # A native BF16 store already supplies its cast. An explicit
                # intrinsic round-trip is needed only before a multiply or a
                # wider consumer store, not when the store is the boundary.
                if cutlass.const_expr(post or O.element_type != cutlass.BFloat16):
                    z = _rounded(z, mix_bf16)
                if cutlass.const_expr(post):
                    z = z * cutlass.Float32(Post[0])
                    if cutlass.const_expr(O.element_type != cutlass.BFloat16):
                        z = _rounded(z, mix_bf16)
                O[row, c] = O.element_type(z)
            if lane == 0:
                Stats[row, 0] = rx
                Stats[row, 1] = ry
                Stats[row, 2] = rz

    @cute.jit
    def _launch(
        X: cute.Tensor,
        Y: cute.Tensor,
        WX: cute.Tensor,
        WY: cute.Tensor,
        WO: cute.Tensor,
        S: cute.Tensor,
        Post: cute.Tensor,
        O: cute.Tensor,
        Stats: cute.Tensor,
        stream: cuda.CUstream,
        eps: cutlass.Constexpr[float],
        outer: cutlass.Constexpr[bool],
        scale: cutlass.Constexpr[bool],
        post: cutlass.Constexpr[bool],
        mix_bf16: cutlass.Constexpr[bool],
    ):
        _kernel(X, Y, WX, WY, WO, S, Post, O, Stats, eps, outer, scale, post, mix_bf16).launch(
            grid=((X.shape[0] + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def _backward_rows(
        X: cute.Tensor,
        Y: cute.Tensor,
        WX: cute.Tensor,
        WY: cute.Tensor,
        WO: cute.Tensor,
        S: cute.Tensor,
        Post: cute.Tensor,
        G: cute.Tensor,
        Stats: cute.Tensor,
        DX: cute.Tensor,
        DY: cute.Tensor,
        Partials: cute.Tensor,
        outer: cutlass.Constexpr[bool],
        scale: cutlass.Constexpr[bool],
        row_lanes: cutlass.Constexpr[int],
        post: cutlass.Constexpr[bool],
        mix_bf16: cutlass.Constexpr[bool],
    ):
        t, _, _ = cute.arch.thread_idx()
        group, _, _ = cute.arch.block_idx()
        lane, owner = t % row_lanes, t // row_lanes
        subgroup = (t % row_lanes) // 32
        width = X.shape[1]
        items = width // row_lanes
        groups = (X.shape[0] + 7) // 8
        x_bf16 = X.element_type == cutlass.BFloat16
        y_bf16 = Y.element_type == cutlass.BFloat16
        scaled_bf16 = x_bf16 and (not scale or S.element_type == cutlass.BFloat16)
        planes = 2 + int(outer) + int(scale)
        scale_plane = 2 + int(outer)
        allocator = cutlass.utils.SmemAllocator()
        params = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((4, planes, width), stride=(planes * width, width, 1)),
            byte_alignment=16,
        )
        if cutlass.const_expr(row_lanes == 64):
            # Separate planes avoid reusing a dot before its partner warp
            # consumed it. Barriers below are outside the partial-row guard.
            dots = allocator.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_layout((3, 4, 2), stride=(8, 2, 1)),
                byte_alignment=16,
            )
        for i in cutlass.range_constexpr(items):
            c = lane + i * row_lanes
            for p in cutlass.range_constexpr(planes):
                params[owner, p, c] = cutlass.Float32(0)
        # Four row owners per CTA, each processing two rows. Wide rows use
        # two warps per owner, halving per-thread vectors without extra global
        # reads or increasing the number/size of parameter partial groups.
        for local in cutlass.range(2, unroll=1):
            row = group * 8 + owner + local * 4
            # One warp owns the entire row on the regular path. Keep its
            # live vectors inside one row-valid region, rather than carrying
            # zero-filled vectors across three masked regions. The cooperative
            # experiment still requires every owner at each CTA barrier.
            if row < X.shape[0] or row_lanes == 64:
                xs = cute.make_rmem_tensor(items, cutlass.Float32)
                ys = cute.make_rmem_tensor(items, cutlass.Float32)
                gx = cute.make_rmem_tensor(items, cutlass.Float32)
                gy = cute.make_rmem_tensor(items, cutlass.Float32)
                rx, ry, rz = cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(1)
                dotz = cutlass.Float32(0)
                if row < X.shape[0]:
                    rx = cutlass.Float32(Stats[row, 0])
                    ry = cutlass.Float32(Stats[row, 1])
                    rz = cutlass.Float32(Stats[row, 2])
                    for i in cutlass.range_constexpr(items):
                        c = lane + i * row_lanes
                        x, y = cutlass.Float32(X[row, c]), cutlass.Float32(Y[row, c])
                        xs[i], ys[i] = x, y
                        if cutlass.const_expr(outer):
                            a = _rounded(x * rx * cutlass.Float32(WX[c]), x_bf16)
                            b = _rounded(y * ry * cutlass.Float32(WY[c]), y_bf16)
                            if cutlass.const_expr(scale):
                                a = _rounded(a * cutlass.Float32(S[c]), scaled_bf16)
                            z = _rounded(a + b, mix_bf16)
                            g = cutlass.Float32(G[row, c])
                            if cutlass.const_expr(mix_bf16 and G.element_type != cutlass.BFloat16):
                                g = _rounded(g, True)
                            if cutlass.const_expr(post):
                                g = _rounded(g * cutlass.Float32(Post[0]), mix_bf16)
                            dotz = dotz + g * cutlass.Float32(WO[c]) * z
                            params[owner, 2, c] = cutlass.Float32(params[owner, 2, c]) + g * z * rz
                if cutlass.const_expr(outer):
                    dotz = cute.arch.warp_reduction_sum(dotz)
                    if cutlass.const_expr(row_lanes == 64):
                        if lane % 32 == 0:
                            dots[0, owner, subgroup] = dotz
                        cute.arch.sync_threads()
                        dotz = cutlass.Float32(dots[0, owner, 0]) + cutlass.Float32(
                            dots[0, owner, 1]
                        )
                    dotz = dotz / width
                dotx, doty = cutlass.Float32(0), cutlass.Float32(0)
                if row < X.shape[0]:
                    for i in cutlass.range_constexpr(items):
                        c = lane + i * row_lanes
                        x, y = cutlass.Float32(xs[i]), cutlass.Float32(ys[i])
                        # Recompute cheap affine intermediates rather than keeping
                        # a, z and g live across the outer warp reduction.
                        a = _rounded(x * rx * cutlass.Float32(WX[c]), x_bf16)
                        g = cutlass.Float32(G[row, c])
                        if cutlass.const_expr(mix_bf16 and G.element_type != cutlass.BFloat16):
                            g = _rounded(g, True)
                        if cutlass.const_expr(post):
                            g = _rounded(g * cutlass.Float32(Post[0]), mix_bf16)
                        if cutlass.const_expr(outer):
                            b = _rounded(y * ry * cutlass.Float32(WY[c]), y_bf16)
                            sa = a
                            if cutlass.const_expr(scale):
                                sa = _rounded(a * cutlass.Float32(S[c]), scaled_bf16)
                            z = _rounded(sa + b, mix_bf16)
                            g = _rounded(
                                rz * (g * cutlass.Float32(WO[c]) - z * rz * rz * dotz), mix_bf16
                            )
                        gb = _rounded(g, y_bf16)
                        ga = _rounded(g, scaled_bf16)
                        if cutlass.const_expr(scale):
                            params[owner, scale_plane, c] = cutlass.Float32(
                                params[owner, scale_plane, c]
                            ) + _rounded(ga * a, scaled_bf16)
                            ga = _rounded(ga * cutlass.Float32(S[c]), scaled_bf16)
                        ga = _rounded(ga, x_bf16)
                        params[owner, 0, c] = cutlass.Float32(params[owner, 0, c]) + ga * xs[i] * rx
                        params[owner, 1, c] = cutlass.Float32(params[owner, 1, c]) + gb * ys[i] * ry
                        gx[i] = ga * cutlass.Float32(WX[c])
                        gy[i] = gb * cutlass.Float32(WY[c])
                        dotx = dotx + gx[i] * xs[i]
                        doty = doty + gy[i] * ys[i]
                dotx = cute.arch.warp_reduction_sum(dotx)
                doty = cute.arch.warp_reduction_sum(doty)
                if cutlass.const_expr(row_lanes == 64):
                    if lane % 32 == 0:
                        dots[1, owner, subgroup] = dotx
                        dots[2, owner, subgroup] = doty
                    cute.arch.sync_threads()
                    dotx = cutlass.Float32(dots[1, owner, 0]) + cutlass.Float32(dots[1, owner, 1])
                    doty = cutlass.Float32(dots[2, owner, 0]) + cutlass.Float32(dots[2, owner, 1])
                dotx, doty = dotx / width, doty / width
                if row < X.shape[0]:
                    for i in cutlass.range_constexpr(items):
                        c = lane + i * row_lanes
                        DX[row, c] = DX.element_type(rx * (gx[i] - xs[i] * rx * rx * dotx))
                        DY[row, c] = DY.element_type(ry * (gy[i] - ys[i] * ry * ry * doty))
                if cutlass.const_expr(row_lanes == 64 and not outer):
                    # With an outer norm the next dotz barrier protects reuse;
                    # otherwise prevent the next row overwriting dotx/doty early.
                    cute.arch.sync_threads()
        cute.arch.sync_threads()
        if owner == 0:
            for i in cutlass.range_constexpr(items):
                c = lane + i * row_lanes
                for p in cutlass.range_constexpr(planes):
                    value = cutlass.Float32(0)
                    for w in cutlass.range_constexpr(4):
                        value = value + cutlass.Float32(params[w, p, c])
                    Partials[(p * groups + group) * width + c] = value

    @cute.kernel
    def _param_reduce(
        P: cute.Tensor,
        DWX: cute.Tensor,
        DWY: cute.Tensor,
        DWO: cute.Tensor,
        DS: cute.Tensor,
        width: cutlass.Constexpr[int],
        groups: cutlass.Constexpr[int],
        chunk_rows: cutlass.Constexpr[int],
        outer: cutlass.Constexpr[bool],
        scale: cutlass.Constexpr[bool],
    ):
        t, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        lane, warp = t % 32, t // 32
        c = bx * 16 + t % 16
        planes = 2 + int(outer) + int(scale)
        sums = cute.make_rmem_tensor(planes, cutlass.Float32)
        for p in cutlass.range_constexpr(planes):
            sums[p] = cutlass.Float32(0)
        for r in range(t // 16, chunk_rows, 8):
            row = by * chunk_rows + r
            if row < groups and c < width:
                for p in cutlass.range_constexpr(planes):
                    sums[p] = sums[p] + cutlass.Float32(P[(p * groups + row) * width + c])
        allocator = cutlass.utils.SmemAllocator()
        partial = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((4, planes, 16), stride=(planes * 16, 16, 1)),
            byte_alignment=16,
        )
        for p in cutlass.range_constexpr(planes):
            v = cutlass.Float32(sums[p])
            v = v + cute.arch.shuffle_sync_bfly(v, 16)
            if lane < 16:
                partial[warp, p, lane] = v
        cute.arch.sync_threads()
        if t < 16 and c < width:
            result = cute.make_rmem_tensor(planes, cutlass.Float32)
            for p in cutlass.range_constexpr(planes):
                value = cutlass.Float32(0)
                for w in cutlass.range_constexpr(4):
                    value = value + cutlass.Float32(partial[w, p, t])
                result[p] = value
            DWX[by * width + c] = DWX.element_type(result[0])
            DWY[by * width + c] = DWY.element_type(result[1])
            if cutlass.const_expr(outer):
                DWO[by * width + c] = DWO.element_type(result[2])
            if cutlass.const_expr(scale):
                DS[by * width + c] = DS.element_type(result[2 + int(outer)])

    @cute.jit
    def _launch_backward(
        X: cute.Tensor,
        Y: cute.Tensor,
        WX: cute.Tensor,
        WY: cute.Tensor,
        WO: cute.Tensor,
        S: cute.Tensor,
        Post: cute.Tensor,
        G: cute.Tensor,
        Stats: cute.Tensor,
        DX: cute.Tensor,
        DY: cute.Tensor,
        P: cute.Tensor,
        Mid: cute.Tensor,
        MX: cute.Tensor,
        MY: cute.Tensor,
        MO: cute.Tensor,
        MS: cute.Tensor,
        DWX: cute.Tensor,
        DWY: cute.Tensor,
        DWO: cute.Tensor,
        DS: cute.Tensor,
        stream: cuda.CUstream,
        outer: cutlass.Constexpr[bool],
        scale: cutlass.Constexpr[bool],
        row_lanes: cutlass.Constexpr[int],
        post: cutlass.Constexpr[bool],
        mix_bf16: cutlass.Constexpr[bool],
    ):
        groups = (X.shape[0] + 7) // 8
        width = X.shape[1]
        chunks = (groups + 127) // 128
        _backward_rows(
            X,
            Y,
            WX,
            WY,
            WO,
            S,
            Post,
            G,
            Stats,
            DX,
            DY,
            P,
            outer,
            scale,
            row_lanes,
            post,
            mix_bf16,
        ).launch(grid=(groups, 1, 1), block=(4 * row_lanes, 1, 1), stream=stream)
        if cutlass.const_expr(chunks > 1):
            _param_reduce(P, MX, MY, MO, MS, width, groups, 128, outer, scale).launch(
                grid=((width + 15) // 16, chunks, 1), block=(128, 1, 1), stream=stream
            )
            _param_reduce(Mid, DWX, DWY, DWO, DS, width, chunks, chunks, outer, scale).launch(
                grid=((width + 15) // 16, 1, 1), block=(128, 1, 1), stream=stream
            )
        else:
            _param_reduce(P, DWX, DWY, DWO, DS, width, groups, groups, outer, scale).launch(
                grid=((width + 15) // 16, 1, 1), block=(128, 1, 1), stream=stream
            )


_CACHE = {}
_LOCK = threading.Lock()


def is_available() -> bool:
    return cute is not None and hasattr(cute.arch, "cvt_f32_bf16")


def _backward_row_lanes(width: int) -> int:
    """Keep the measured compact path; cooperative layout is test-only.

    The 64-lane experiment increased potential occupancy but regressed row
    latency. Tests explicitly override this planner before compilation to
    preserve its masked-row/scratch-reuse regressions. Not a runtime toggle.
    """
    return 32


def forward(x, y, wx, wy, wo, scale, eps, post_scale=None, out_dtype=None):
    if not is_available():
        raise RuntimeError("CuTe DSL is not installed")
    mix_dtype = output_dtype(x, y, scale)
    output = torch.empty_like(x, dtype=mix_dtype if out_dtype is None else out_dtype)
    stats = torch.empty((x.shape[0], 3), device=x.device, dtype=torch.float32)
    tensors = (
        x,
        y,
        wx,
        wy,
        wx if wo is None else wo,
        wx if scale is None else scale,
        wx if post_scale is None else post_scale.reshape(1),
        output,
        stats,
    )
    key = (
        x.device.index,
        tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors),
        eps,
        wo is not None,
        scale is not None,
        post_scale is not None,
    )
    with torch.cuda.device(x.device):
        with _LOCK:
            fn = _CACHE.get(key)
            if fn is None:
                # No alignment promise beyond the scalar element size: offset
                # contiguous tensors are legal and need no hidden clone.
                descriptors = [from_dlpack(t, enable_tvm_ffi=True) for t in tensors]
                fn = cute.compile(
                    _launch,
                    *descriptors,
                    make_fake_stream(),
                    eps=eps,
                    outer=wo is not None,
                    scale=scale is not None,
                    post=post_scale is not None,
                    mix_bf16=mix_dtype == torch.bfloat16,
                    options="--enable-tvm-ffi",
                )
                _CACHE[key] = fn
        fn(*tensors, cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    return output, stats


def backward(g, x, y, wx, wy, wo, scale, stats, post_scale=None):
    rows, d = x.shape
    row_lanes = _backward_row_lanes(d)
    groups = (rows + 7) // 8
    chunks = (groups + 127) // 128
    planes = 2 + int(wo is not None) + int(scale is not None)
    dx, dy = torch.empty_like(x), torch.empty_like(y)
    dwx, dwy = torch.empty_like(wx), torch.empty_like(wy)
    dwo = torch.empty_like(wo) if wo is not None else None
    ds = torch.empty_like(scale) if scale is not None else None
    partials = torch.empty((planes * groups * d,), device=x.device, dtype=torch.float32)
    mid = torch.empty((planes, chunks * d), device=x.device, dtype=torch.float32)
    tensors = (
        x,
        y,
        wx,
        wy,
        wx if wo is None else wo,
        wx if scale is None else scale,
        wx if post_scale is None else post_scale.reshape(1),
        g.contiguous(),
        stats,
        dx,
        dy,
        partials,
        mid.reshape(-1),
        mid[0],
        mid[1],
        mid[2] if wo is not None else mid[0],
        mid[2 + int(wo is not None)] if scale is not None else mid[0],
        dwx,
        dwy,
        dwx if dwo is None else dwo,
        dwx if ds is None else ds,
    )
    key = (
        "backward",
        x.device.index,
        row_lanes,
        tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors),
        wo is not None,
        scale is not None,
        post_scale is not None,
    )
    with torch.cuda.device(x.device):
        with _LOCK:
            fn = _CACHE.get(key)
            if fn is None:
                descriptors = [from_dlpack(t, enable_tvm_ffi=True) for t in tensors]
                fn = cute.compile(
                    _launch_backward,
                    *descriptors,
                    make_fake_stream(),
                    outer=wo is not None,
                    scale=scale is not None,
                    row_lanes=row_lanes,
                    post=post_scale is not None,
                    mix_bf16=output_dtype(x, y, scale) == torch.bfloat16,
                    options="--enable-tvm-ffi",
                )
                _CACHE[key] = fn
        fn(*tensors, cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    return dx, dy, dwx, dwy, dwo, ds
