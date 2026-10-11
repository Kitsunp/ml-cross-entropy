"""Fused paired RMSNorm with FP32 reductions and deterministic affine gradients."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from .reference import output_dtype

try:
    from torch.library import wrap_triton
except ImportError:  # Older supported Torch retains the opaque operator route.
    wrap_triton = None


def _launch(kernel):
    return kernel if wrap_triton is None else wrap_triton(kernel)


def _device_context(device):
    # The compiled graph owns device/stream selection; keep eager scaffolding
    # out of a compiler-visible triton_op body.
    return nullcontext() if torch.compiler.is_compiling() else torch.cuda.device(device)


@triton.jit
def _forward(
    X,
    Y,
    WX,
    WY,
    WO,
    S,
    Post,
    O,
    Stats,
    D: tl.constexpr,
    EPS: tl.constexpr,
    OUTER: tl.constexpr,
    SCALE: tl.constexpr,
    POST: tl.constexpr,
    MIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + r * D + c, c < D, 0).to(tl.float32)
    y = tl.load(Y + r * D + c, c < D, 0).to(tl.float32)
    wx = tl.load(WX + c, c < D, 0).to(tl.float32)
    wy = tl.load(WY + c, c < D, 0).to(tl.float32)
    rx = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    ry = tl.rsqrt(tl.sum(y * y, 0) / D + EPS)
    # Preserve the dtype boundaries of native Torch RMSNorm and pointwise ops.
    a = x * rx * wx
    b = y * ry * wy
    a = a.to(X.dtype.element_ty).to(tl.float32)
    b = b.to(Y.dtype.element_ty).to(tl.float32)
    if SCALE:
        s = tl.load(S + c, c < D, 0).to(tl.float32)
        scaled_dtype: tl.constexpr = (
            tl.float32
            if X.dtype.element_ty == tl.float32 or S.dtype.element_ty == tl.float32
            else tl.bfloat16
        )
        a = (a * s).to(scaled_dtype).to(tl.float32)
    z = (a + b).to(MIX).to(tl.float32)
    rz = tl.full((), 1.0, tl.float32)
    if OUTER:
        rz = tl.rsqrt(tl.sum(z * z, 0) / D + EPS)
        wo = tl.load(WO + c, c < D, 0).to(tl.float32)
        z = z * rz * wo
    z = z.to(MIX).to(tl.float32)
    if POST:
        z = (z * tl.load(Post).to(tl.float32)).to(MIX).to(tl.float32)
    tl.store(O + r * D + c, z, c < D)
    tl.store(Stats + r * 3, rx)
    tl.store(Stats + r * 3 + 1, ry)
    tl.store(Stats + r * 3 + 2, rz)


@triton.jit
def _backward(
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
    ROWS: tl.constexpr,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    OUTER: tl.constexpr,
    SCALE: tl.constexpr,
    POST: tl.constexpr,
    MIX: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
):
    group = tl.program_id(0)
    r = group * BR + tl.arange(0, BR)
    c = tl.arange(0, BD)
    mask = (r[:, None] < ROWS) & (c[None, :] < D)
    x = tl.load(X + r[:, None] * D + c[None, :], mask, 0).to(tl.float32)
    y = tl.load(Y + r[:, None] * D + c[None, :], mask, 0).to(tl.float32)
    g = tl.load(G + r[:, None] * D + c[None, :], mask, 0).to(tl.float32)
    # Cast VJP precedes the scalar-multiply VJP. Consumer dtype is not the
    # dtype of the mixed/outer-normalized activation.
    g = g.to(MIX).to(tl.float32)
    if POST:
        g = (g * tl.load(Post).to(tl.float32)).to(MIX).to(tl.float32)
    wx = tl.load(WX + c, c < D, 0).to(tl.float32)
    wy = tl.load(WY + c, c < D, 0).to(tl.float32)
    rx = tl.load(Stats + r * 3, r < ROWS, 0)
    ry = tl.load(Stats + r * 3 + 1, r < ROWS, 0)
    nx = x * rx[:, None]
    ny = y * ry[:, None]
    a = (nx * wx[None, :]).to(X.dtype.element_ty).to(tl.float32)
    b = (ny * wy[None, :]).to(Y.dtype.element_ty).to(tl.float32)
    s = tl.full((BD,), 1.0, tl.float32)
    if SCALE:
        scaled_dtype: tl.constexpr = (
            tl.float32
            if X.dtype.element_ty == tl.float32 or S.dtype.element_ty == tl.float32
            else tl.bfloat16
        )
        s = tl.load(S + c, c < D, 0).to(tl.float32)
    else:
        scaled_dtype: tl.constexpr = X.dtype.element_ty
    sa = (a * s[None, :]).to(scaled_dtype).to(tl.float32)
    z = (sa + b).to(MIX).to(tl.float32)
    gwo = tl.full((BD,), 0.0, tl.float32)
    if OUTER:
        rz = tl.load(Stats + r * 3 + 2, r < ROWS, 0)
        wo = tl.load(WO + c, c < D, 0).to(tl.float32)
        nz = z * rz[:, None]
        gwo = tl.sum(g * nz, axis=0)
        gz = g * wo[None, :]
        dot = tl.sum(gz * z, axis=1) / D
        g = (rz[:, None] * (gz - z * (rz * rz * dot)[:, None])).to(MIX).to(tl.float32)
    gs = tl.full((BD,), 0.0, tl.float32)
    gb = g.to(Y.dtype.element_ty).to(tl.float32)
    ga = g.to(scaled_dtype).to(tl.float32)
    if SCALE:
        # The separate pointwise multiply returns a gradient in input dtype
        # before broadcast reduction; RMSNorm affine gradients do not.
        gs = tl.sum((ga * a).to(scaled_dtype).to(tl.float32), axis=0)
    ga = (ga * s[None, :]).to(scaled_dtype).to(X.dtype.element_ty).to(tl.float32)
    gwx = tl.sum(ga * nx, axis=0)
    gwy = tl.sum(gb * ny, axis=0)
    gxnorm = ga * wx[None, :]
    gynorm = gb * wy[None, :]
    dotx = tl.sum(gxnorm * x, axis=1) / D
    doty = tl.sum(gynorm * y, axis=1) / D
    dx = rx[:, None] * (gxnorm - x * (rx * rx * dotx)[:, None])
    dy = ry[:, None] * (gynorm - y * (ry * ry * doty)[:, None])
    tl.store(DX + r[:, None] * D + c[None, :], dx, mask)
    tl.store(DY + r[:, None] * D + c[None, :], dy, mask)
    offset = group * D + c
    tl.store(P + offset, gwx, c < D)
    tl.store(P + GROUPS * D + offset, gwy, c < D)
    if OUTER:
        tl.store(P + 2 * GROUPS * D + offset, gwo, c < D)
    if SCALE:
        scale_plane: tl.constexpr = 2 + OUTER
        tl.store(P + scale_plane * GROUPS * D + offset, gs, c < D)


@triton.jit
def _reduce_params(
    P,
    DWX,
    DWY,
    DWO,
    DS,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    OUTER: tl.constexpr,
    SCALE: tl.constexpr,
    BC: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    r = tl.arange(0, 128)
    sx = tl.full((128, BC), 0.0, tl.float32)
    sy = tl.full((128, BC), 0.0, tl.float32)
    so = tl.full((128, BC), 0.0, tl.float32)
    ss = tl.full((128, BC), 0.0, tl.float32)
    scale_plane: tl.constexpr = 2 + OUTER
    for start in range(tl.cdiv(CHUNK_ROWS, 128)):
        rows = tl.program_id(1) * CHUNK_ROWS + start * 128 + r
        offset = rows[:, None] * D + c[None, :]
        mask = (rows[:, None] < GROUPS) & (c[None, :] < D)
        sx += tl.load(P + offset, mask, 0)
        sy += tl.load(P + GROUPS * D + offset, mask, 0)
        if OUTER:
            so += tl.load(P + 2 * GROUPS * D + offset, mask, 0)
        if SCALE:
            ss += tl.load(P + scale_plane * GROUPS * D + offset, mask, 0)
    out = tl.program_id(1) * D + c
    tl.store(DWX + out, tl.sum(sx, 0), c < D)
    tl.store(DWY + out, tl.sum(sy, 0), c < D)
    if OUTER:
        tl.store(DWO + out, tl.sum(so, 0), c < D)
    if SCALE:
        tl.store(DS + out, tl.sum(ss, 0), c < D)


def forward(x, y, wx, wy, wo, scale, eps, post_scale=None, out_dtype=None):
    rows, d = x.shape
    mix_dtype = output_dtype(x, y, scale)
    output = torch.empty_like(x, dtype=mix_dtype if out_dtype is None else out_dtype)
    stats = torch.empty((rows, 3), device=x.device, dtype=torch.float32)
    with _device_context(x.device):
        _launch(_forward)[(rows,)](
            x,
            y,
            wx,
            wy,
            wo,
            scale,
            post_scale,
            output,
            stats,
            d,
            eps,
            wo is not None,
            scale is not None,
            post_scale is not None,
            tl.float32 if mix_dtype == torch.float32 else tl.bfloat16,
            triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, stats


def _backward_num_warps(width: int) -> int:
    # The 16-warp width-512 experiment did not improve the full training step.
    return 8


def backward(g, x, y, wx, wy, wo, scale, stats, post_scale=None):
    rows, d = x.shape
    groups = triton.cdiv(rows, 8)
    planes = 2 + int(wo is not None) + int(scale is not None)
    scale_plane = 2 + int(wo is not None)
    dx, dy = torch.empty_like(x), torch.empty_like(y)
    dwx, dwy = torch.empty_like(wx), torch.empty_like(wy)
    dwo = torch.empty_like(wo) if wo is not None else None
    ds = torch.empty_like(scale) if scale is not None else None
    partials = torch.empty((planes, groups, d), device=x.device, dtype=torch.float32)
    with _device_context(x.device):
        _launch(_backward)[(groups,)](
            x,
            y,
            wx,
            wy,
            wo,
            scale,
            post_scale,
            g.contiguous(),
            stats,
            dx,
            dy,
            partials,
            rows,
            d,
            groups,
            wo is not None,
            scale is not None,
            post_scale is not None,
            tl.float32 if output_dtype(x, y, scale) == torch.float32 else tl.bfloat16,
            8,
            triton.next_power_of_2(d),
            num_warps=_backward_num_warps(d),
            enable_fp_fusion=False,
        )
        # A width-only grid underutilizes large GPUs while reading all row
        # partials. Split that traversal across rows, then reduce its compact
        # FP32 outputs. Both passes have fixed ordering and no atomics.
        chunks = triton.cdiv(groups, 128)
        if chunks > 1:
            reduced = torch.empty((planes, chunks, d), device=x.device, dtype=torch.float32)
            _launch(_reduce_params)[(triton.cdiv(d, 16), chunks)](
                partials,
                reduced[0],
                reduced[1],
                reduced[2] if wo is not None else None,
                reduced[scale_plane] if scale is not None else None,
                d,
                groups,
                wo is not None,
                scale is not None,
                16,
                128,
                num_warps=4,
            )
            partials = reduced
            groups = chunks
        _launch(_reduce_params)[(triton.cdiv(d, 16), 1)](
            partials,
            dwx,
            dwy,
            dwo,
            ds,
            d,
            groups,
            wo is not None,
            scale is not None,
            16,
            groups,
            num_warps=4,
        )
    return dx, dy, dwx, dwy, dwo, ds
