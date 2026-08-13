from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import torch

from . import _cute_gupn

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
except Exception as error:
    cuda = None
    cutlass = None
    cute = None
    make_fake_stream = None
    _IMPORT_ERROR: Exception | None = error
else:
    _IMPORT_ERROR = None


WARP_SIZE = _cute_gupn.WARP_SIZE
VECTOR_WIDTH = _cute_gupn.VECTOR_WIDTH
BACKWARD_WARPS = 8
SCALAR_COUNT = 20
PARAMETER_COUNT = 6
SWIZZLE_B = _cute_gupn.SWIZZLE_B
SWIZZLE_M = _cute_gupn.SWIZZLE_M
SWIZZLE_S = _cute_gupn.SWIZZLE_S


if cute is not None and cutlass is not None and cuda is not None:
    @cute.kernel
    def gupn_exclusive_backward_rows_kernel(
        grad_output: cute.Tensor,
        gate0: cute.Tensor,
        up: cute.Tensor,
        gate_row: cute.Tensor,
        polynorm_weight: cute.Tensor,
        polynorm_bias: cute.Tensor,
        exclusive_logits: cute.Tensor,
        down_column: cute.Tensor,
        seeds: cute.Tensor,
        gate_partials: cute.Tensor,
        column_partials: cute.Tensor,
        parameter_partials: cute.Tensor,
        grad_gate0: cute.Tensor,
        grad_up: cute.Tensor,
        num_warps: cutlass.Constexpr[int],
        use_xor: cutlass.Constexpr[bool],
        dropout_p: cutlass.Constexpr[float],
        dropout_threshold: cutlass.Constexpr[int],
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = thread % WARP_SIZE
        warp = thread // WARP_SIZE
        row = block * num_warps + warp
        active_row = row < gate0.shape[0]
        vector_width = gate0.shape[2]
        groups = gate0.shape[1]
        hidden = groups * vector_width
        group_tiles = (groups + WARP_SIZE - 1) // WARP_SIZE

        allocator = cutlass.utils.SmemAllocator()
        cache_layout = cute.make_layout(
            (num_warps * groups, vector_width),
            stride=(vector_width, 1),
        )
        if cutlass.const_expr(use_xor):
            cache_layout = cute.make_composed_layout(
                cute.make_swizzle(
                    SWIZZLE_B,
                    SWIZZLE_M,
                    SWIZZLE_S,
                ),
                0,
                cache_layout,
            )
        gate_cache = allocator.allocate_tensor(
            element_type=gate0.element_type,
            layout=cache_layout,
            byte_alignment=16,
        )
        row_scalars = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(
                (num_warps, SCALAR_COUNT),
                stride=(SCALAR_COUNT, 1),
            ),
            byte_alignment=16,
        )
        vector_scratch = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(
                (num_warps, WARP_SIZE, vector_width),
                stride=(WARP_SIZE * vector_width, vector_width, 1),
            ),
            byte_alignment=16,
        )
        parameter_scratch = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(
                (num_warps, PARAMETER_COUNT),
                stride=(PARAMETER_COUNT, 1),
            ),
            byte_alignment=16,
        )

        gate0_values = cute.make_rmem_tensor(vector_width, gate0.element_type)
        gate_values = cute.make_rmem_tensor(vector_width, gate0.element_type)
        multiplier_values = cute.make_rmem_tensor(vector_width, gate_row.element_type)

        # Recompute and cache only G.  The eight warps own eight independent rows,
        # so every reduction below remains warp-local and deterministic.
        sum2 = cutlass.Float32(0.0)
        sum4 = cutlass.Float32(0.0)
        sum6 = cutlass.Float32(0.0)
        for tile in range(group_tiles):
            group = tile * WARP_SIZE + lane
            for item in cutlass.range_constexpr(vector_width):
                gate0_values[item] = cutlass.BFloat16(0.0)
                multiplier_values[item] = cutlass.BFloat16(0.0)
                gate_values[item] = cutlass.BFloat16(0.0)
            if group < groups:
                if active_row:
                    cute.autovec_copy(gate0[row, group, None], gate0_values)
                    cute.autovec_copy(gate_row[group, None], multiplier_values)
                for item in cutlass.range_constexpr(vector_width):
                    gate = cutlass.BFloat16(
                        cutlass.Float32(gate0_values[item])
                        * cutlass.Float32(multiplier_values[item])
                    )
                    gate_values[item] = gate
                    value = cutlass.Float32(gate)
                    value2 = value * value
                    value3 = value * value2
                    sum2 = sum2 + value2
                    sum4 = sum4 + value2 * value2
                    sum6 = sum6 + value3 * value3
                cute.autovec_copy(
                    gate_values,
                    gate_cache[warp * groups + group, None],
                )
        sum2 = cute.arch.warp_reduction_sum(sum2)
        sum4 = cute.arch.warp_reduction_sum(sum4)
        sum6 = cute.arch.warp_reduction_sum(sum6)
        if lane == 0:
            inv_width = cutlass.Float32(1.0) / cutlass.Float32(hidden)
            eps = cutlass.Float32(1.0e-6)
            row_scalars[warp, 0] = cute.math.rsqrt(sum2 * inv_width + eps)
            row_scalars[warp, 1] = cute.math.rsqrt(sum4 * inv_width + eps)
            row_scalars[warp, 2] = cute.math.rsqrt(sum6 * inv_width + eps)
        cute.arch.sync_threads()

        inv1_bf16 = cutlass.BFloat16(row_scalars[warp, 0])
        inv2_bf16 = cutlass.BFloat16(row_scalars[warp, 1])
        inv3_bf16 = cutlass.BFloat16(row_scalars[warp, 2])

        reference_norm_sq = cutlass.Float32(0.0)
        dot2 = cutlass.Float32(0.0)
        dot3 = cutlass.Float32(0.0)
        for tile in range(group_tiles):
            group = tile * WARP_SIZE + lane
            if group < groups:
                cute.autovec_copy(
                    gate_cache[warp * groups + group, None],
                    gate_values,
                )
                for item in cutlass.range_constexpr(vector_width):
                    value = cutlass.Float32(gate_values[item])
                    value2 = value * value
                    value3 = value * value2
                    x1 = cutlass.BFloat16(
                        value * cutlass.Float32(inv1_bf16)
                    )
                    x2 = cutlass.BFloat16(
                        value2 * cutlass.Float32(inv2_bf16)
                    )
                    x3 = cutlass.BFloat16(
                        value3 * cutlass.Float32(inv3_bf16)
                    )
                    x1f = cutlass.Float32(x1)
                    reference_norm_sq = reference_norm_sq + x1f * x1f
                    dot2 = dot2 + cutlass.Float32(x2) * x1f
                    dot3 = dot3 + cutlass.Float32(x3) * x1f
        reference_norm_sq = cute.arch.warp_reduction_sum(reference_norm_sq)
        dot2 = cute.arch.warp_reduction_sum(dot2)
        dot3 = cute.arch.warp_reduction_sum(dot3)
        if lane == 0:
            denominator = cute.math.max(
                reference_norm_sq,
                cutlass.Float32(1.0e-6),
            )
            alpha2 = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0)
                + cute.math.exp(-cutlass.Float32(exclusive_logits[0]))
            )
            alpha3 = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0)
                + cute.math.exp(-cutlass.Float32(exclusive_logits[1]))
            )
            row_scalars[warp, 3] = reference_norm_sq
            row_scalars[warp, 4] = denominator
            row_scalars[warp, 5] = dot2
            row_scalars[warp, 6] = dot3
            row_scalars[warp, 7] = dot2 / denominator
            row_scalars[warp, 8] = dot3 / denominator
            row_scalars[warp, 9] = alpha2
            row_scalars[warp, 10] = alpha3
        cute.arch.sync_threads()

        projection2 = cutlass.BFloat16(row_scalars[warp, 7])
        projection3 = cutlass.BFloat16(row_scalars[warp, 8])
        alpha2 = cutlass.BFloat16(row_scalars[warp, 9])
        alpha3 = cutlass.BFloat16(row_scalars[warp, 10])
        residual2_sq = cutlass.Float32(0.0)
        residual3_sq = cutlass.Float32(0.0)
        for tile in range(group_tiles):
            group = tile * WARP_SIZE + lane
            if group < groups:
                cute.autovec_copy(
                    gate_cache[warp * groups + group, None],
                    gate_values,
                )
                for item in cutlass.range_constexpr(vector_width):
                    value = cutlass.Float32(gate_values[item])
                    value2 = value * value
                    value3 = value * value2
                    x1 = cutlass.BFloat16(
                        value * cutlass.Float32(inv1_bf16)
                    )
                    x2 = cutlass.BFloat16(
                        value2 * cutlass.Float32(inv2_bf16)
                    )
                    x3 = cutlass.BFloat16(
                        value3 * cutlass.Float32(inv3_bf16)
                    )
                    coeff2 = cutlass.BFloat16(
                        cutlass.Float32(alpha2)
                        * cutlass.Float32(projection2)
                    )
                    coeff3 = cutlass.BFloat16(
                        cutlass.Float32(alpha3)
                        * cutlass.Float32(projection3)
                    )
                    residual2 = cutlass.BFloat16(
                        cutlass.Float32(x2)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff2)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    residual3 = cutlass.BFloat16(
                        cutlass.Float32(x3)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff3)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    residual2f = cutlass.Float32(residual2)
                    residual3f = cutlass.Float32(residual3)
                    residual2_sq = residual2_sq + residual2f * residual2f
                    residual3_sq = residual3_sq + residual3f * residual3f
        residual2_sq = cute.arch.warp_reduction_sum(residual2_sq)
        residual3_sq = cute.arch.warp_reduction_sum(residual3_sq)
        if lane == 0:
            inv_width = cutlass.Float32(1.0) / cutlass.Float32(hidden)
            eps = cutlass.Float32(1.0e-6)
            row_scalars[warp, 11] = cute.math.rsqrt(
                residual2_sq * inv_width + eps
            )
            row_scalars[warp, 12] = cute.math.rsqrt(
                residual3_sq * inv_width + eps
            )
        cute.arch.sync_threads()

        residual2_inv = cutlass.BFloat16(row_scalars[warp, 11])
        residual3_inv = cutlass.BFloat16(row_scalars[warp, 12])
        w3 = polynorm_weight[0]
        w2 = polynorm_weight[1]
        w1 = polynorm_weight[2]
        shift = polynorm_bias[0]
        seed0 = cutlass.Uint32(seeds[0])
        seed1 = cutlass.Uint32(seeds[1])
        dropout_scale = cutlass.Float32(1.0 / (1.0 - dropout_p))
        up_values = cute.make_rmem_tensor(vector_width, up.element_type)
        grad_output_values = cute.make_rmem_tensor(
            vector_width,
            grad_output.element_type,
        )
        column_values = cute.make_rmem_tensor(
            vector_width,
            down_column.element_type,
        )
        grad_up_values = cute.make_rmem_tensor(vector_width, grad_up.element_type)
        dropout_values = cute.make_rmem_tensor(vector_width, cutlass.Float32)

        weight3_sum = cutlass.Float32(0.0)
        weight2_sum = cutlass.Float32(0.0)
        weight1_sum = cutlass.Float32(0.0)
        bias_sum = cutlass.Float32(0.0)
        dot_residual2 = cutlass.Float32(0.0)
        dot_residual3 = cutlass.Float32(0.0)

        # Recompute the forward epilogue, replay Philox, emit dU, and form the
        # first deterministic parameter partials.
        for tile in range(group_tiles):
            group = tile * WARP_SIZE + lane
            for item in cutlass.range_constexpr(vector_width):
                up_values[item] = cutlass.BFloat16(0.0)
                grad_output_values[item] = cutlass.BFloat16(0.0)
                column_values[item] = cutlass.BFloat16(0.0)
                grad_up_values[item] = cutlass.BFloat16(0.0)
                dropout_values[item] = cutlass.Float32(1.0)
                vector_scratch[warp, lane, item] = cutlass.Float32(0.0)
            if group < groups:
                cute.autovec_copy(
                    gate_cache[warp * groups + group, None],
                    gate_values,
                )
                if active_row:
                    cute.autovec_copy(up[row, group, None], up_values)
                    cute.autovec_copy(
                        grad_output[row, group, None],
                        grad_output_values,
                    )
                    cute.autovec_copy(
                        down_column[group, None],
                        column_values,
                    )
                    if cutlass.const_expr(dropout_p != 0.0):
                        counter = (
                            cutlass.Uint64(row) * cutlass.Uint64(hidden // 4)
                            + cutlass.Uint64(group)
                        )
                        c0 = cutlass.Uint32(counter)
                        c1 = cutlass.Uint32(counter >> 32)
                        c2 = cutlass.Uint32(seeds[2])
                        c3 = cutlass.Uint32(seeds[3])
                        k0 = seed0
                        k1 = seed1
                        for _ in cutlass.range_constexpr(10):
                            product0 = (
                                cutlass.Uint64(c0)
                                * cutlass.Uint64(0xD2511F53)
                            )
                            product1 = (
                                cutlass.Uint64(c2)
                                * cutlass.Uint64(0xCD9E8D57)
                            )
                            lo0 = cutlass.Uint32(product0)
                            hi0 = cutlass.Uint32(product0 >> 32)
                            lo1 = cutlass.Uint32(product1)
                            hi1 = cutlass.Uint32(product1 >> 32)
                            c0, c1, c2, c3 = (
                                hi1 ^ c1 ^ k0,
                                lo1,
                                hi0 ^ c3 ^ k1,
                                lo0,
                            )
                            k0 = k0 + cutlass.Uint32(0x9E3779B9)
                            k1 = k1 + cutlass.Uint32(0xBB67AE85)
                        dropout_values[0] = (
                            c0 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[1] = (
                            c1 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[2] = (
                            c2 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[3] = (
                            c3 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale

                for item in cutlass.range_constexpr(vector_width):
                    value = cutlass.Float32(gate_values[item])
                    value2 = value * value
                    value3 = value * value2
                    x1 = cutlass.BFloat16(
                        value * cutlass.Float32(inv1_bf16)
                    )
                    x2 = cutlass.BFloat16(
                        value2 * cutlass.Float32(inv2_bf16)
                    )
                    x3 = cutlass.BFloat16(
                        value3 * cutlass.Float32(inv3_bf16)
                    )
                    coeff2 = cutlass.BFloat16(
                        cutlass.Float32(alpha2)
                        * cutlass.Float32(projection2)
                    )
                    coeff3 = cutlass.BFloat16(
                        cutlass.Float32(alpha3)
                        * cutlass.Float32(projection3)
                    )
                    residual2 = cutlass.BFloat16(
                        cutlass.Float32(x2)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff2)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    residual3 = cutlass.BFloat16(
                        cutlass.Float32(x3)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff3)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    exclusive2 = cutlass.BFloat16(
                        cutlass.Float32(residual2)
                        * cutlass.Float32(residual2_inv)
                    )
                    exclusive3 = cutlass.BFloat16(
                        cutlass.Float32(residual3)
                        * cutlass.Float32(residual3_inv)
                    )
                    poly3 = cutlass.BFloat16(
                        cutlass.Float32(w3) * cutlass.Float32(exclusive3)
                    )
                    poly2 = cutlass.BFloat16(
                        cutlass.Float32(w2) * cutlass.Float32(exclusive2)
                    )
                    poly1 = cutlass.BFloat16(
                        cutlass.Float32(w1) * cutlass.Float32(x1)
                    )
                    poly = cutlass.BFloat16(
                        cutlass.Float32(poly3) + cutlass.Float32(poly2)
                    )
                    poly = cutlass.BFloat16(
                        cutlass.Float32(poly) + cutlass.Float32(poly1)
                    )
                    poly = cutlass.BFloat16(
                        cutlass.Float32(poly) + cutlass.Float32(shift)
                    )
                    dropped = cutlass.BFloat16(
                        cutlass.Float32(poly) * dropout_values[item]
                    )
                    hadamard = cutlass.BFloat16(
                        cutlass.Float32(dropped)
                        * cutlass.Float32(up_values[item])
                    )
                    grad = cutlass.Float32(grad_output_values[item])
                    column = cutlass.Float32(column_values[item])
                    up_value = cutlass.Float32(up_values[item])
                    grad_activation = (
                        grad * column * up_value * dropout_values[item]
                    )
                    grad_up_values[item] = cutlass.BFloat16(
                        grad * column * cutlass.Float32(dropped)
                    )
                    vector_scratch[warp, lane, item] = (
                        grad * cutlass.Float32(hadamard)
                    )
                    weight3_sum = (
                        weight3_sum
                        + grad_activation * cutlass.Float32(exclusive3)
                    )
                    weight2_sum = (
                        weight2_sum
                        + grad_activation * cutlass.Float32(exclusive2)
                    )
                    weight1_sum = (
                        weight1_sum
                        + grad_activation * cutlass.Float32(x1)
                    )
                    bias_sum = bias_sum + grad_activation
                    grad_exclusive2 = (
                        grad_activation * cutlass.Float32(w2)
                    )
                    grad_exclusive3 = (
                        grad_activation * cutlass.Float32(w3)
                    )
                    dot_residual2 = (
                        dot_residual2
                        + grad_exclusive2 * cutlass.Float32(residual2)
                    )
                    dot_residual3 = (
                        dot_residual3
                        + grad_exclusive3 * cutlass.Float32(residual3)
                    )
                if active_row:
                    cute.autovec_copy(
                        grad_up_values,
                        grad_up[row, group, None],
                    )
            cute.arch.sync_threads()
            if thread < WARP_SIZE * vector_width:
                slot = thread // vector_width
                item = thread % vector_width
                output_group = tile * WARP_SIZE + slot
                if output_group < groups:
                    value = cutlass.Float32(0.0)
                    for source_warp in cutlass.range_constexpr(num_warps):
                        value = (
                            value
                            + cutlass.Float32(
                                vector_scratch[source_warp, slot, item]
                            )
                        )
                    column_partials[block, output_group, item] = value
            cute.arch.sync_threads()

        weight3_sum = cute.arch.warp_reduction_sum(weight3_sum)
        weight2_sum = cute.arch.warp_reduction_sum(weight2_sum)
        weight1_sum = cute.arch.warp_reduction_sum(weight1_sum)
        bias_sum = cute.arch.warp_reduction_sum(bias_sum)
        dot_residual2 = cute.arch.warp_reduction_sum(dot_residual2)
        dot_residual3 = cute.arch.warp_reduction_sum(dot_residual3)
        if lane == 0:
            parameter_scratch[warp, 0] = weight3_sum
            parameter_scratch[warp, 1] = weight2_sum
            parameter_scratch[warp, 2] = weight1_sum
            parameter_scratch[warp, 3] = bias_sum
            row_scalars[warp, 13] = dot_residual2
            row_scalars[warp, 14] = dot_residual3
        cute.arch.sync_threads()

        # Backpropagate through the residual RMS normalizations and collect the
        # two projection-dot derivatives.
        dot_grad_reference2 = cutlass.Float32(0.0)
        dot_grad_reference3 = cutlass.Float32(0.0)
        inv_residual2 = cutlass.Float32(row_scalars[warp, 11])
        inv_residual3 = cutlass.Float32(row_scalars[warp, 12])
        inv_width = cutlass.Float32(1.0) / cutlass.Float32(hidden)
        for tile in range(group_tiles):
            group = tile * WARP_SIZE + lane
            if group < groups:
                cute.autovec_copy(
                    gate_cache[warp * groups + group, None],
                    gate_values,
                )
                for item in cutlass.range_constexpr(vector_width):
                    up_values[item] = cutlass.BFloat16(0.0)
                    grad_output_values[item] = cutlass.BFloat16(0.0)
                    column_values[item] = cutlass.BFloat16(0.0)
                    dropout_values[item] = cutlass.Float32(1.0)
                if active_row:
                    cute.autovec_copy(up[row, group, None], up_values)
                    cute.autovec_copy(
                        grad_output[row, group, None],
                        grad_output_values,
                    )
                    cute.autovec_copy(
                        down_column[group, None],
                        column_values,
                    )
                    if cutlass.const_expr(dropout_p != 0.0):
                        counter = (
                            cutlass.Uint64(row) * cutlass.Uint64(hidden // 4)
                            + cutlass.Uint64(group)
                        )
                        c0 = cutlass.Uint32(counter)
                        c1 = cutlass.Uint32(counter >> 32)
                        c2 = cutlass.Uint32(seeds[2])
                        c3 = cutlass.Uint32(seeds[3])
                        k0 = seed0
                        k1 = seed1
                        for _ in cutlass.range_constexpr(10):
                            product0 = (
                                cutlass.Uint64(c0)
                                * cutlass.Uint64(0xD2511F53)
                            )
                            product1 = (
                                cutlass.Uint64(c2)
                                * cutlass.Uint64(0xCD9E8D57)
                            )
                            lo0 = cutlass.Uint32(product0)
                            hi0 = cutlass.Uint32(product0 >> 32)
                            lo1 = cutlass.Uint32(product1)
                            hi1 = cutlass.Uint32(product1 >> 32)
                            c0, c1, c2, c3 = (
                                hi1 ^ c1 ^ k0,
                                lo1,
                                hi0 ^ c3 ^ k1,
                                lo0,
                            )
                            k0 = k0 + cutlass.Uint32(0x9E3779B9)
                            k1 = k1 + cutlass.Uint32(0xBB67AE85)
                        dropout_values[0] = (
                            c0 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[1] = (
                            c1 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[2] = (
                            c2 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[3] = (
                            c3 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                for item in cutlass.range_constexpr(vector_width):
                    value = cutlass.Float32(gate_values[item])
                    value2 = value * value
                    value3 = value * value2
                    x1 = cutlass.BFloat16(
                        value * cutlass.Float32(inv1_bf16)
                    )
                    x2 = cutlass.BFloat16(
                        value2 * cutlass.Float32(inv2_bf16)
                    )
                    x3 = cutlass.BFloat16(
                        value3 * cutlass.Float32(inv3_bf16)
                    )
                    coeff2 = cutlass.BFloat16(
                        cutlass.Float32(alpha2)
                        * cutlass.Float32(projection2)
                    )
                    coeff3 = cutlass.BFloat16(
                        cutlass.Float32(alpha3)
                        * cutlass.Float32(projection3)
                    )
                    residual2 = cutlass.BFloat16(
                        cutlass.Float32(x2)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff2)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    residual3 = cutlass.BFloat16(
                        cutlass.Float32(x3)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff3)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    grad_activation = (
                        cutlass.Float32(grad_output_values[item])
                        * cutlass.Float32(column_values[item])
                        * cutlass.Float32(up_values[item])
                        * dropout_values[item]
                    )
                    grad_exclusive2 = (
                        grad_activation * cutlass.Float32(w2)
                    )
                    grad_exclusive3 = (
                        grad_activation * cutlass.Float32(w3)
                    )
                    residual2f = cutlass.Float32(residual2)
                    residual3f = cutlass.Float32(residual3)
                    grad_residual2 = (
                        grad_exclusive2 * inv_residual2
                        - residual2f
                        * inv_residual2
                        * inv_residual2
                        * inv_residual2
                        * cutlass.Float32(row_scalars[warp, 13])
                        * inv_width
                    )
                    grad_residual3 = (
                        grad_exclusive3 * inv_residual3
                        - residual3f
                        * inv_residual3
                        * inv_residual3
                        * inv_residual3
                        * cutlass.Float32(row_scalars[warp, 14])
                        * inv_width
                    )
                    dot_grad_reference2 = (
                        dot_grad_reference2
                        + grad_residual2 * cutlass.Float32(x1)
                    )
                    dot_grad_reference3 = (
                        dot_grad_reference3
                        + grad_residual3 * cutlass.Float32(x1)
                    )
        dot_grad_reference2 = cute.arch.warp_reduction_sum(
            dot_grad_reference2
        )
        dot_grad_reference3 = cute.arch.warp_reduction_sum(
            dot_grad_reference3
        )
        if lane == 0:
            row_scalars[warp, 15] = dot_grad_reference2
            row_scalars[warp, 16] = dot_grad_reference3
            grad_alpha2 = (
                -cutlass.Float32(projection2) * dot_grad_reference2
            )
            grad_alpha3 = (
                -cutlass.Float32(projection3) * dot_grad_reference3
            )
            alpha2f = cutlass.Float32(alpha2)
            alpha3f = cutlass.Float32(alpha3)
            parameter_scratch[warp, 4] = (
                grad_alpha2 * alpha2f * (cutlass.Float32(1.0) - alpha2f)
            )
            parameter_scratch[warp, 5] = (
                grad_alpha3 * alpha3f * (cutlass.Float32(1.0) - alpha3f)
            )
        cute.arch.sync_threads()

        # Compute the three polynomial-branch gradient dots.
        dot_grad_x1 = cutlass.Float32(0.0)
        dot_grad_x2 = cutlass.Float32(0.0)
        dot_grad_x3 = cutlass.Float32(0.0)
        denominator = cutlass.Float32(row_scalars[warp, 4])
        denominator_live = (
            cutlass.Float32(row_scalars[warp, 3]) >= denominator
        )
        for tile in range(group_tiles):
            group = tile * WARP_SIZE + lane
            if group < groups:
                cute.autovec_copy(
                    gate_cache[warp * groups + group, None],
                    gate_values,
                )
                for item in cutlass.range_constexpr(vector_width):
                    up_values[item] = cutlass.BFloat16(0.0)
                    grad_output_values[item] = cutlass.BFloat16(0.0)
                    column_values[item] = cutlass.BFloat16(0.0)
                    dropout_values[item] = cutlass.Float32(1.0)
                if active_row:
                    cute.autovec_copy(up[row, group, None], up_values)
                    cute.autovec_copy(
                        grad_output[row, group, None],
                        grad_output_values,
                    )
                    cute.autovec_copy(
                        down_column[group, None],
                        column_values,
                    )
                    if cutlass.const_expr(dropout_p != 0.0):
                        counter = (
                            cutlass.Uint64(row) * cutlass.Uint64(hidden // 4)
                            + cutlass.Uint64(group)
                        )
                        c0 = cutlass.Uint32(counter)
                        c1 = cutlass.Uint32(counter >> 32)
                        c2 = cutlass.Uint32(seeds[2])
                        c3 = cutlass.Uint32(seeds[3])
                        k0 = seed0
                        k1 = seed1
                        for _ in cutlass.range_constexpr(10):
                            product0 = (
                                cutlass.Uint64(c0)
                                * cutlass.Uint64(0xD2511F53)
                            )
                            product1 = (
                                cutlass.Uint64(c2)
                                * cutlass.Uint64(0xCD9E8D57)
                            )
                            lo0 = cutlass.Uint32(product0)
                            hi0 = cutlass.Uint32(product0 >> 32)
                            lo1 = cutlass.Uint32(product1)
                            hi1 = cutlass.Uint32(product1 >> 32)
                            c0, c1, c2, c3 = (
                                hi1 ^ c1 ^ k0,
                                lo1,
                                hi0 ^ c3 ^ k1,
                                lo0,
                            )
                            k0 = k0 + cutlass.Uint32(0x9E3779B9)
                            k1 = k1 + cutlass.Uint32(0xBB67AE85)
                        dropout_values[0] = (
                            c0 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[1] = (
                            c1 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[2] = (
                            c2 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[3] = (
                            c3 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                for item in cutlass.range_constexpr(vector_width):
                    value = cutlass.Float32(gate_values[item])
                    value2 = value * value
                    value3 = value * value2
                    x1 = cutlass.BFloat16(
                        value * cutlass.Float32(inv1_bf16)
                    )
                    x2 = cutlass.BFloat16(
                        value2 * cutlass.Float32(inv2_bf16)
                    )
                    x3 = cutlass.BFloat16(
                        value3 * cutlass.Float32(inv3_bf16)
                    )
                    coeff2 = cutlass.BFloat16(
                        cutlass.Float32(alpha2)
                        * cutlass.Float32(projection2)
                    )
                    coeff3 = cutlass.BFloat16(
                        cutlass.Float32(alpha3)
                        * cutlass.Float32(projection3)
                    )
                    residual2 = cutlass.BFloat16(
                        cutlass.Float32(x2)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff2)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    residual3 = cutlass.BFloat16(
                        cutlass.Float32(x3)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff3)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    grad_activation = (
                        cutlass.Float32(grad_output_values[item])
                        * cutlass.Float32(column_values[item])
                        * cutlass.Float32(up_values[item])
                        * dropout_values[item]
                    )
                    grad_exclusive2 = (
                        grad_activation * cutlass.Float32(w2)
                    )
                    grad_exclusive3 = (
                        grad_activation * cutlass.Float32(w3)
                    )
                    residual2f = cutlass.Float32(residual2)
                    residual3f = cutlass.Float32(residual3)
                    grad_residual2 = (
                        grad_exclusive2 * inv_residual2
                        - residual2f
                        * inv_residual2
                        * inv_residual2
                        * inv_residual2
                        * cutlass.Float32(row_scalars[warp, 13])
                        * inv_width
                    )
                    grad_residual3 = (
                        grad_exclusive3 * inv_residual3
                        - residual3f
                        * inv_residual3
                        * inv_residual3
                        * inv_residual3
                        * cutlass.Float32(row_scalars[warp, 14])
                        * inv_width
                    )
                    x1f = cutlass.Float32(x1)
                    x2f = cutlass.Float32(x2)
                    x3f = cutlass.Float32(x3)
                    alpha2f = cutlass.Float32(alpha2)
                    alpha3f = cutlass.Float32(alpha3)
                    dot_ref2 = cutlass.Float32(row_scalars[warp, 15])
                    dot_ref3 = cutlass.Float32(row_scalars[warp, 16])
                    grad_x2 = (
                        grad_residual2
                        - alpha2f * dot_ref2 / denominator * x1f
                    )
                    grad_x3 = (
                        grad_residual3
                        - alpha3f * dot_ref3 / denominator * x1f
                    )
                    grad_ref2 = (
                        -alpha2f
                        * cutlass.Float32(projection2)
                        * grad_residual2
                        - alpha2f * dot_ref2 / denominator * x2f
                    )
                    grad_ref3 = (
                        -alpha3f
                        * cutlass.Float32(projection3)
                        * grad_residual3
                        - alpha3f * dot_ref3 / denominator * x3f
                    )
                    if denominator_live:
                        grad_ref2 = (
                            grad_ref2
                            + alpha2f
                            * dot_ref2
                            * (cutlass.Float32(2.0) * row_scalars[warp, 5])
                            / (denominator * denominator)
                            * x1f
                        )
                        grad_ref3 = (
                            grad_ref3
                            + alpha3f
                            * dot_ref3
                            * (cutlass.Float32(2.0) * row_scalars[warp, 6])
                            / (denominator * denominator)
                            * x1f
                        )
                    grad_x1 = (
                        grad_activation * cutlass.Float32(w1)
                        + grad_ref2
                        + grad_ref3
                    )
                    dot_grad_x1 = dot_grad_x1 + grad_x1 * value
                    dot_grad_x2 = dot_grad_x2 + grad_x2 * value2
                    dot_grad_x3 = dot_grad_x3 + grad_x3 * value3
        dot_grad_x1 = cute.arch.warp_reduction_sum(dot_grad_x1)
        dot_grad_x2 = cute.arch.warp_reduction_sum(dot_grad_x2)
        dot_grad_x3 = cute.arch.warp_reduction_sum(dot_grad_x3)
        if lane == 0:
            row_scalars[warp, 17] = dot_grad_x1
            row_scalars[warp, 18] = dot_grad_x2
            row_scalars[warp, 19] = dot_grad_x3
        cute.arch.sync_threads()

        # Final pass: emit dG0 and aggregate the gate-row multiplier gradient.
        inv1 = cutlass.Float32(row_scalars[warp, 0])
        inv2 = cutlass.Float32(row_scalars[warp, 1])
        inv3 = cutlass.Float32(row_scalars[warp, 2])
        dot_poly1 = cutlass.Float32(row_scalars[warp, 17])
        dot_poly2 = cutlass.Float32(row_scalars[warp, 18])
        dot_poly3 = cutlass.Float32(row_scalars[warp, 19])
        grad_gate0_values = cute.make_rmem_tensor(
            vector_width,
            grad_gate0.element_type,
        )
        for tile in range(group_tiles):
            group = tile * WARP_SIZE + lane
            for item in cutlass.range_constexpr(vector_width):
                gate0_values[item] = cutlass.BFloat16(0.0)
                multiplier_values[item] = cutlass.BFloat16(0.0)
                grad_gate0_values[item] = cutlass.BFloat16(0.0)
                vector_scratch[warp, lane, item] = cutlass.Float32(0.0)
            if group < groups:
                cute.autovec_copy(
                    gate_cache[warp * groups + group, None],
                    gate_values,
                )
                if active_row:
                    cute.autovec_copy(gate0[row, group, None], gate0_values)
                    cute.autovec_copy(gate_row[group, None], multiplier_values)
                for item in cutlass.range_constexpr(vector_width):
                    dropout_values[item] = cutlass.Float32(1.0)
                    up_values[item] = cutlass.BFloat16(0.0)
                    grad_output_values[item] = cutlass.BFloat16(0.0)
                    column_values[item] = cutlass.BFloat16(0.0)
                if cutlass.const_expr(dropout_p != 0.0):
                    if active_row:
                        counter = (
                            cutlass.Uint64(row)
                            * cutlass.Uint64(hidden // 4)
                            + cutlass.Uint64(group)
                        )
                        c0 = cutlass.Uint32(counter)
                        c1 = cutlass.Uint32(counter >> 32)
                        c2 = cutlass.Uint32(seeds[2])
                        c3 = cutlass.Uint32(seeds[3])
                        k0 = seed0
                        k1 = seed1
                        for _ in cutlass.range_constexpr(10):
                            product0 = (
                                cutlass.Uint64(c0)
                                * cutlass.Uint64(0xD2511F53)
                            )
                            product1 = (
                                cutlass.Uint64(c2)
                                * cutlass.Uint64(0xCD9E8D57)
                            )
                            lo0 = cutlass.Uint32(product0)
                            hi0 = cutlass.Uint32(product0 >> 32)
                            lo1 = cutlass.Uint32(product1)
                            hi1 = cutlass.Uint32(product1 >> 32)
                            c0, c1, c2, c3 = (
                                hi1 ^ c1 ^ k0,
                                lo1,
                                hi0 ^ c3 ^ k1,
                                lo0,
                            )
                            k0 = k0 + cutlass.Uint32(0x9E3779B9)
                            k1 = k1 + cutlass.Uint32(0xBB67AE85)
                        dropout_values[0] = (
                            c0 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[1] = (
                            c1 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[2] = (
                            c2 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                        dropout_values[3] = (
                            c3 >= cutlass.Uint32(dropout_threshold)
                        ) * dropout_scale
                if active_row:
                    cute.autovec_copy(up[row, group, None], up_values)
                    cute.autovec_copy(
                        grad_output[row, group, None],
                        grad_output_values,
                    )
                    cute.autovec_copy(
                        down_column[group, None],
                        column_values,
                    )
                for item in cutlass.range_constexpr(vector_width):
                    value = cutlass.Float32(gate_values[item])
                    value2 = value * value
                    value3 = value * value2
                    x1 = cutlass.BFloat16(
                        value * cutlass.Float32(inv1_bf16)
                    )
                    x2 = cutlass.BFloat16(
                        value2 * cutlass.Float32(inv2_bf16)
                    )
                    x3 = cutlass.BFloat16(
                        value3 * cutlass.Float32(inv3_bf16)
                    )
                    coeff2 = cutlass.BFloat16(
                        cutlass.Float32(alpha2)
                        * cutlass.Float32(projection2)
                    )
                    coeff3 = cutlass.BFloat16(
                        cutlass.Float32(alpha3)
                        * cutlass.Float32(projection3)
                    )
                    residual2 = cutlass.BFloat16(
                        cutlass.Float32(x2)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff2)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    residual3 = cutlass.BFloat16(
                        cutlass.Float32(x3)
                        - cutlass.Float32(
                            cutlass.BFloat16(
                                cutlass.Float32(coeff3)
                                * cutlass.Float32(x1)
                            )
                        )
                    )
                    grad_activation = (
                        cutlass.Float32(grad_output_values[item])
                        * cutlass.Float32(column_values[item])
                        * cutlass.Float32(up_values[item])
                        * dropout_values[item]
                    )
                    grad_exclusive2 = (
                        grad_activation * cutlass.Float32(w2)
                    )
                    grad_exclusive3 = (
                        grad_activation * cutlass.Float32(w3)
                    )
                    residual2f = cutlass.Float32(residual2)
                    residual3f = cutlass.Float32(residual3)
                    grad_residual2 = (
                        grad_exclusive2 * inv_residual2
                        - residual2f
                        * inv_residual2
                        * inv_residual2
                        * inv_residual2
                        * cutlass.Float32(row_scalars[warp, 13])
                        * inv_width
                    )
                    grad_residual3 = (
                        grad_exclusive3 * inv_residual3
                        - residual3f
                        * inv_residual3
                        * inv_residual3
                        * inv_residual3
                        * cutlass.Float32(row_scalars[warp, 14])
                        * inv_width
                    )
                    x1f = cutlass.Float32(x1)
                    x2f = cutlass.Float32(x2)
                    x3f = cutlass.Float32(x3)
                    alpha2f = cutlass.Float32(alpha2)
                    alpha3f = cutlass.Float32(alpha3)
                    dot_ref2 = cutlass.Float32(row_scalars[warp, 15])
                    dot_ref3 = cutlass.Float32(row_scalars[warp, 16])
                    grad_x2 = (
                        grad_residual2
                        - alpha2f * dot_ref2 / denominator * x1f
                    )
                    grad_x3 = (
                        grad_residual3
                        - alpha3f * dot_ref3 / denominator * x1f
                    )
                    grad_ref2 = (
                        -alpha2f
                        * cutlass.Float32(projection2)
                        * grad_residual2
                        - alpha2f * dot_ref2 / denominator * x2f
                    )
                    grad_ref3 = (
                        -alpha3f
                        * cutlass.Float32(projection3)
                        * grad_residual3
                        - alpha3f * dot_ref3 / denominator * x3f
                    )
                    if denominator_live:
                        grad_ref2 = (
                            grad_ref2
                            + alpha2f
                            * dot_ref2
                            * (cutlass.Float32(2.0) * row_scalars[warp, 5])
                            / (denominator * denominator)
                            * x1f
                        )
                        grad_ref3 = (
                            grad_ref3
                            + alpha3f
                            * dot_ref3
                            * (cutlass.Float32(2.0) * row_scalars[warp, 6])
                            / (denominator * denominator)
                            * x1f
                        )
                    grad_x1 = (
                        grad_activation * cutlass.Float32(w1)
                        + grad_ref2
                        + grad_ref3
                    )
                    grad_gate = (
                        inv1 * grad_x1
                        - inv1 * inv1 * inv1 * dot_poly1 * inv_width * value
                        + cutlass.Float32(2.0)
                        * inv2
                        * grad_x2
                        * value
                        - cutlass.Float32(2.0)
                        * inv2
                        * inv2
                        * inv2
                        * dot_poly2
                        * inv_width
                        * value
                        * value2
                        + cutlass.Float32(3.0)
                        * inv3
                        * grad_x3
                        * value2
                        - cutlass.Float32(3.0)
                        * inv3
                        * inv3
                        * inv3
                        * dot_poly3
                        * inv_width
                        * value
                        * value2
                        * value2
                    )
                    grad_gate0_values[item] = cutlass.BFloat16(
                        grad_gate * cutlass.Float32(multiplier_values[item])
                    )
                    vector_scratch[warp, lane, item] = (
                        grad_gate * cutlass.Float32(gate0_values[item])
                    )
                if active_row:
                    cute.autovec_copy(
                        grad_gate0_values,
                        grad_gate0[row, group, None],
                    )
            cute.arch.sync_threads()
            if thread < WARP_SIZE * vector_width:
                slot = thread // vector_width
                item = thread % vector_width
                output_group = tile * WARP_SIZE + slot
                if output_group < groups:
                    value = cutlass.Float32(0.0)
                    for source_warp in cutlass.range_constexpr(num_warps):
                        value = (
                            value
                            + cutlass.Float32(
                                vector_scratch[source_warp, slot, item]
                            )
                        )
                    gate_partials[block, output_group, item] = value
            cute.arch.sync_threads()

        if warp == 0:
            if lane < PARAMETER_COUNT:
                value = cutlass.Float32(0.0)
                for source_warp in cutlass.range_constexpr(num_warps):
                    value = value + cutlass.Float32(
                        parameter_scratch[source_warp, lane]
                    )
                parameter_partials[block, lane] = value


    @cute.kernel
    def gupn_backward_vector_reduce_kernel(
        partials: cute.Tensor,
        output: cute.Tensor,
    ):
        thread, _, _ = cute.arch.thread_idx()
        group, _, _ = cute.arch.block_idx()
        lane = thread % WARP_SIZE
        warp = thread // WARP_SIZE
        reduction_warps = 8
        reduction_threads = reduction_warps * WARP_SIZE
        vector_width = partials.shape[2]
        allocator = cutlass.utils.SmemAllocator()
        warp_sums = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(
                (reduction_warps, vector_width),
                stride=(vector_width, 1),
            ),
            byte_alignment=16,
        )
        values = cute.make_rmem_tensor(vector_width, cutlass.Float32)
        for item in cutlass.range_constexpr(vector_width):
            values[item] = cutlass.Float32(0.0)
        for block in range(thread, partials.shape[0], reduction_threads):
            for item in cutlass.range_constexpr(vector_width):
                values[item] = (
                    values[item]
                    + cutlass.Float32(partials[block, group, item])
                )
        for item in cutlass.range_constexpr(vector_width):
            values[item] = cute.arch.warp_reduction_sum(values[item])
        if lane == 0:
            for item in cutlass.range_constexpr(vector_width):
                warp_sums[warp, item] = values[item]
        cute.arch.sync_threads()
        if warp == 0:
            for item in cutlass.range_constexpr(vector_width):
                value = cutlass.Float32(0.0)
                if lane < reduction_warps:
                    value = cutlass.Float32(warp_sums[lane, item])
                value = cute.arch.warp_reduction_sum(value)
                if lane == 0:
                    output[group, item] = cutlass.BFloat16(value)


    @cute.kernel
    def gupn_backward_parameter_reduce_kernel(
        partials: cute.Tensor,
        grad_weight: cute.Tensor,
        grad_bias: cute.Tensor,
        grad_logits: cute.Tensor,
    ):
        thread, _, _ = cute.arch.thread_idx()
        lane = thread % WARP_SIZE
        warp = thread // WARP_SIZE
        reduction_warps = 8
        reduction_threads = reduction_warps * WARP_SIZE
        allocator = cutlass.utils.SmemAllocator()
        warp_sums = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(
                (reduction_warps, PARAMETER_COUNT),
                stride=(PARAMETER_COUNT, 1),
            ),
            byte_alignment=16,
        )
        values = cute.make_rmem_tensor(PARAMETER_COUNT, cutlass.Float32)
        for item in cutlass.range_constexpr(PARAMETER_COUNT):
            values[item] = cutlass.Float32(0.0)
        for block in range(thread, partials.shape[0], reduction_threads):
            for item in cutlass.range_constexpr(PARAMETER_COUNT):
                values[item] = (
                    values[item] + cutlass.Float32(partials[block, item])
                )
        for item in cutlass.range_constexpr(PARAMETER_COUNT):
            values[item] = cute.arch.warp_reduction_sum(values[item])
        if lane == 0:
            for item in cutlass.range_constexpr(PARAMETER_COUNT):
                warp_sums[warp, item] = values[item]
        cute.arch.sync_threads()
        if warp == 0:
            for item in cutlass.range_constexpr(PARAMETER_COUNT):
                value = cutlass.Float32(0.0)
                if lane < reduction_warps:
                    value = cutlass.Float32(warp_sums[lane, item])
                value = cute.arch.warp_reduction_sum(value)
                if lane == 0:
                    if item < 3:
                        grad_weight[item] = cutlass.BFloat16(value)
                    elif item == 3:
                        grad_bias[0] = cutlass.BFloat16(value)
                    else:
                        grad_logits[item - 4] = cutlass.BFloat16(value)


    @cute.jit
    def _launch_backward_on_stream(
        grad_output: cute.Tensor,
        gate0: cute.Tensor,
        up: cute.Tensor,
        gate_row: cute.Tensor,
        polynorm_weight: cute.Tensor,
        polynorm_bias: cute.Tensor,
        exclusive_logits: cute.Tensor,
        down_column: cute.Tensor,
        seeds: cute.Tensor,
        gate_partials: cute.Tensor,
        column_partials: cute.Tensor,
        parameter_partials: cute.Tensor,
        grad_gate0: cute.Tensor,
        grad_up: cute.Tensor,
        grad_gate_row: cute.Tensor,
        grad_weight: cute.Tensor,
        grad_bias: cute.Tensor,
        grad_logits: cute.Tensor,
        grad_column: cute.Tensor,
        stream: cuda.CUstream,
        num_warps: cutlass.Constexpr[int],
        use_xor: cutlass.Constexpr[bool],
        dropout_p: cutlass.Constexpr[float],
        dropout_threshold: cutlass.Constexpr[int],
    ):
        blocks = (gate0.shape[0] + num_warps - 1) // num_warps
        gupn_exclusive_backward_rows_kernel(
            grad_output,
            gate0,
            up,
            gate_row,
            polynorm_weight,
            polynorm_bias,
            exclusive_logits,
            down_column,
            seeds,
            gate_partials,
            column_partials,
            parameter_partials,
            grad_gate0,
            grad_up,
            num_warps,
            use_xor,
            dropout_p,
            dropout_threshold,
        ).launch(
            grid=(blocks, 1, 1),
            block=(num_warps * WARP_SIZE, 1, 1),
            stream=stream,
        )
        gupn_backward_vector_reduce_kernel(
            gate_partials,
            grad_gate_row,
        ).launch(
            grid=(gate_partials.shape[1], 1, 1),
            block=(8 * WARP_SIZE, 1, 1),
            stream=stream,
        )
        gupn_backward_vector_reduce_kernel(
            column_partials,
            grad_column,
        ).launch(
            grid=(column_partials.shape[1], 1, 1),
            block=(8 * WARP_SIZE, 1, 1),
            stream=stream,
        )
        gupn_backward_parameter_reduce_kernel(
            parameter_partials,
            grad_weight,
            grad_bias,
            grad_logits,
        ).launch(
            grid=(1, 1, 1),
            block=(8 * WARP_SIZE, 1, 1),
            stream=stream,
        )


@dataclass(frozen=True)
class _KernelKey:
    device_index: int
    capability: tuple[int, int]
    shapes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    dropout_p: float
    use_xor: bool


_CACHE: dict[_KernelKey, Any] = {}
_CACHE_LOCK = threading.Lock()


def is_available() -> bool:
    return _IMPORT_ERROR is None and _cute_gupn.is_available()


def _compile(
    tensors: tuple[torch.Tensor, ...],
    dropout_p: float,
    use_xor: bool,
):
    if cute is None or make_fake_stream is None:
        raise RuntimeError("NVIDIA CuTe DSL is unavailable") from _IMPORT_ERROR
    device_index = tensors[0].device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = _KernelKey(
        device_index=device_index,
        capability=torch.cuda.get_device_capability(device_index),
        shapes=tuple(tuple(tensor.shape) for tensor in tensors),
        strides=tuple(tuple(tensor.stride()) for tensor in tensors),
        dtypes=tuple(tensor.dtype for tensor in tensors),
        dropout_p=float(dropout_p),
        use_xor=bool(use_xor),
    )
    dropout_threshold = math.ceil(dropout_p * (1 << 32))
    with _CACHE_LOCK:
        compiled = _CACHE.get(key)
        if compiled is None:
            compiled = cute.compile(
                _launch_backward_on_stream,
                *(_cute_gupn._descriptors(tensors)),
                make_fake_stream(),
                num_warps=BACKWARD_WARPS,
                use_xor=use_xor,
                dropout_p=dropout_p,
                dropout_threshold=dropout_threshold,
                options="--enable-tvm-ffi",
            )
            _CACHE[key] = compiled
        return compiled


def backward(
    grad_output: torch.Tensor,
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column: torch.Tensor,
    seeds: torch.Tensor,
    *,
    dropout_p: float,
    use_xor: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    dropout_p = float(dropout_p)
    _cute_gupn._validate(
        gate0,
        up,
        gate_row,
        polynorm_weight,
        polynorm_bias,
        exclusive_logits,
        down_column,
        seeds,
        dropout_p,
    )
    if grad_output.shape != gate0.shape:
        raise ValueError("grad_output must match gate0")
    if grad_output.device != gate0.device or grad_output.dtype != gate0.dtype:
        raise ValueError("grad_output must match gate0 dtype and device")

    aligned = tuple(
        _cute_gupn._aligned_contiguous(tensor)
        for tensor in (
            grad_output,
            gate0,
            up,
            gate_row,
            polynorm_weight,
            polynorm_bias,
            exclusive_logits,
            down_column,
            seeds,
        )
    )
    shape = _cute_gupn._kernel_shape(aligned[1])
    blocks = (shape[0] + BACKWARD_WARPS - 1) // BACKWARD_WARPS
    grad_gate0 = torch.empty_like(aligned[1])
    grad_up = torch.empty_like(aligned[2])
    grad_gate_row = torch.empty_like(aligned[3])
    grad_weight = torch.empty_like(aligned[4])
    grad_bias = torch.empty_like(aligned[5])
    grad_logits = torch.empty_like(aligned[6])
    grad_column = torch.empty_like(aligned[7])
    gate_partials = torch.empty(
        (blocks, shape[1], shape[2]),
        device=gate0.device,
        dtype=torch.float32,
    )
    column_partials = torch.empty_like(gate_partials)
    parameter_partials = torch.empty(
        (blocks, PARAMETER_COUNT),
        device=gate0.device,
        dtype=torch.float32,
    )
    tensors = (
        aligned[0].view(shape),
        aligned[1].view(shape),
        aligned[2].view(shape),
        aligned[3].view(shape[1:]),
        aligned[4],
        aligned[5],
        aligned[6],
        aligned[7].view(shape[1:]),
        aligned[8],
        gate_partials,
        column_partials,
        parameter_partials,
        grad_gate0.view(shape),
        grad_up.view(shape),
        grad_gate_row.view(shape[1:]),
        grad_weight,
        grad_bias,
        grad_logits,
        grad_column.view(shape[1:]),
    )
    with torch.cuda.device(gate0.device):
        compiled = _compile(tensors, dropout_p, use_xor)
        compiled(*tensors, _cute_gupn._stream(gate0.device))
    return (
        grad_gate0,
        grad_up,
        grad_gate_row,
        grad_weight,
        grad_bias,
        grad_logits,
        grad_column,
    )


__all__ = ["backward", "is_available"]
