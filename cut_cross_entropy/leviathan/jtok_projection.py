"""Opt-in FP32/IEEE split-N projection VJP shared by JTok and JTok-M.

No token-by-expert activation is materialized. Each program owns a parameter
tile and token split, then a fixed-order reducer stores each final gradient.
"""
from __future__ import annotations

from typing import NamedTuple

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):  # Optional on CPU-only installs.
    triton = None
    tl = None


class ProjectionGradPlan(NamedTuple):
    splits: int
    block_f: int
    block_h: int
    block_n: int
    workspace_bytes: int


def projection_grad_plan(
    tokens: int, experts: int, modes: int, d_seed: int, hidden: int,
    *, splits: int = 8,
) -> ProjectionGradPlan:
    """Bound parameter-sized scratch before allocating or launching CUDA."""
    if tokens < 0 or min(experts, modes, d_seed, hidden, splits) < 1:
        raise ValueError("projection split dimensions must be positive; tokens may be zero")
    actual_splits = min(splits, max(1, (tokens + 31) // 32))
    workspace_bytes = experts * actual_splits * (modes + d_seed) * hidden * 4
    if workspace_bytes > 128 * 1024 * 1024:
        raise ValueError("projection split workspace exceeds the explicit 128 MiB experiment budget")
    return ProjectionGradPlan(actual_splits, 16, 64, 32, workspace_bytes)


if triton is not None:
    @triton.jit
    def _jtok_projection_split_kernel(
        grad_surface_ptr, z_ptr, modes_ptr, expert_idx_ptr, selected_weights_ptr,
        partial_ptr, N,
        D_SEED: tl.constexpr, NUM_MODES: tl.constexpr, HIDDEN: tl.constexpr,
        TOP_K: tl.constexpr, SPLITS: tl.constexpr,
        BLOCK_F: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        expert = tl.program_id(0)
        split = tl.program_id(1)
        tile = tl.program_id(2)
        hidden_tiles = tl.cdiv(HIDDEN, BLOCK_H)
        features = tile // hidden_tiles * BLOCK_F + tl.arange(0, BLOCK_F)
        hidden = tile % hidden_tiles * BLOCK_H + tl.arange(0, BLOCK_H)
        total_features: tl.constexpr = NUM_MODES + D_SEED
        mode_feature = features < NUM_MODES
        seed_feature = (features >= NUM_MODES) & (features < total_features)
        seed = tl.maximum(features - NUM_MODES, 0)
        blocks_per_split = tl.cdiv(tl.cdiv(N, BLOCK_N), SPLITS)
        acc = tl.zeros((BLOCK_F, BLOCK_H), tl.float32)
        for block in range(blocks_per_split):
            rows = (split * blocks_per_split + block) * BLOCK_N + tl.arange(0, BLOCK_N)
            row_mask = rows < N
            alpha = tl.zeros((BLOCK_N,), tl.float32)
            beta = tl.zeros((BLOCK_F, BLOCK_N), tl.float32)
            for route in tl.static_range(0, TOP_K):
                route_expert = tl.load(expert_idx_ptr + rows * TOP_K + route,
                                       mask=row_mask, other=-1).to(tl.int32)
                route_weight = tl.load(selected_weights_ptr + rows * TOP_K + route,
                                       mask=row_mask, other=0.0).to(tl.float32)
                selected = tl.where(route_expert == expert, route_weight, 0.0)
                mode_value = tl.load(
                    modes_ptr + (rows[None, :] * TOP_K + route) * NUM_MODES
                    + features[:, None],
                    mask=mode_feature[:, None] & row_mask[None, :], other=0.0,
                ).to(tl.float32)
                alpha += selected
                beta += selected[None, :] * mode_value
            coordinates = tl.load(
                z_ptr + rows[None, :] * D_SEED + seed[:, None],
                mask=seed_feature[:, None] & row_mask[None, :], other=0.0,
            ).to(tl.float32)
            coefficients = tl.where(mode_feature[:, None], beta,
                                    alpha[None, :] * coordinates)
            grad_surface = tl.load(
                grad_surface_ptr + rows[:, None] * HIDDEN + hidden[None, :],
                mask=row_mask[:, None] & (hidden < HIDDEN)[None, :], other=0.0,
            ).to(tl.float32)
            # No new BF16 quantization and no implicit TF32 approximation.
            acc = tl.dot(coefficients, grad_surface, acc, input_precision="ieee")
        offsets = ((expert * SPLITS + split) * total_features + features[:, None]) * HIDDEN
        tl.store(partial_ptr + offsets + hidden[None, :], acc,
                 mask=(features < total_features)[:, None] & (hidden < HIDDEN)[None, :])

    @triton.jit
    def _jtok_projection_split_reduce_kernel(
        partial_ptr, grad_spline_out_ptr, grad_residual_out_ptr,
        D_SEED: tl.constexpr, NUM_MODES: tl.constexpr, HIDDEN: tl.constexpr,
        SPLITS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        expert = tl.program_id(0)
        total_features: tl.constexpr = NUM_MODES + D_SEED
        linear = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = linear < total_features * HIDDEN
        feature = linear // HIDDEN
        value = tl.zeros((BLOCK,), tl.float32)
        for split in range(SPLITS):
            value += tl.load(partial_ptr + (expert * SPLITS + split) * total_features * HIDDEN
                             + linear, mask=mask, other=0.0)
        tl.store(grad_spline_out_ptr + expert * NUM_MODES * HIDDEN + linear,
                 value, mask=mask & (feature < NUM_MODES))
        tl.store(grad_residual_out_ptr + expert * D_SEED * HIDDEN + linear - NUM_MODES * HIDDEN,
                 value, mask=mask & (feature >= NUM_MODES))
