from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream
except Exception as error:
    cuda = None
    cutlass = None
    cute = None
    from_dlpack = None
    make_fake_stream = None
    _IMPORT_ERROR: Exception | None = error
else:
    _IMPORT_ERROR = None


WARP_SIZE = 32
VECTOR_WIDTH = 4
FORWARD_WARPS = 8
BACKWARD_WARPS = 4
SWIZZLE_B = 2
SWIZZLE_M = 4
SWIZZLE_S = 3
DESCRIPTOR_ALIGNMENT = 32
MAX_DROPOUT_P = ((1 << 32) - 1) / (1 << 32)


if cute is not None and cutlass is not None and cuda is not None:
    @cute.kernel
    def polynorm_forward_kernel(
        x: cute.Tensor,
        seeds: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor,
        output: cute.Tensor,
        stats: cute.Tensor,
        num_warps: cutlass.Constexpr[int],
        save_stats: cutlass.Constexpr[bool],
        dropout_p: cutlass.Constexpr[float],
        dropout_threshold: cutlass.Constexpr[int],
    ):
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = thread % WARP_SIZE
        warp = thread // WARP_SIZE
        block_threads = num_warps * WARP_SIZE
        allocator = cutlass.utils.SmemAllocator()
        partials = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((num_warps, 3)),
            byte_alignment=16,
        )
        norms = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((3,)),
            byte_alignment=16,
        )

        sum2 = cutlass.Float32(0.0)
        sum4 = cutlass.Float32(0.0)
        sum6 = cutlass.Float32(0.0)
        vector_width = x.shape[2]
        hidden = x.shape[1] * vector_width
        values = cute.make_rmem_tensor(vector_width, x.element_type)
        for group in range(thread, x.shape[1], block_threads):
            cute.autovec_copy(x[row, group, None], values)
            for item in cutlass.range_constexpr(vector_width):
                value = cutlass.Float32(values[item])
                value2 = value * value
                value3 = value * value2
                sum2 = sum2 + value2
                sum4 = sum4 + value2 * value2
                sum6 = sum6 + value3 * value3

        sum2 = cute.arch.warp_reduction_sum(sum2)
        sum4 = cute.arch.warp_reduction_sum(sum4)
        sum6 = cute.arch.warp_reduction_sum(sum6)
        if lane == 0:
            partials[warp, 0] = sum2
            partials[warp, 1] = sum4
            partials[warp, 2] = sum6
        cute.arch.sync_threads()

        if warp == 0:
            block_sum2 = cutlass.Float32(0.0)
            block_sum4 = cutlass.Float32(0.0)
            block_sum6 = cutlass.Float32(0.0)
            if lane < num_warps:
                block_sum2 = cutlass.Float32(partials[lane, 0])
                block_sum4 = cutlass.Float32(partials[lane, 1])
                block_sum6 = cutlass.Float32(partials[lane, 2])
            block_sum2 = cute.arch.warp_reduction_sum(block_sum2)
            block_sum4 = cute.arch.warp_reduction_sum(block_sum4)
            block_sum6 = cute.arch.warp_reduction_sum(block_sum6)
            inv_width = cutlass.Float32(1.0) / cutlass.Float32(hidden)
            eps = cutlass.Float32(1.0e-6)
            if lane == 0:
                norms[0] = cute.math.rsqrt(block_sum2 * inv_width + eps)
                norms[1] = cute.math.rsqrt(block_sum4 * inv_width + eps)
                norms[2] = cute.math.rsqrt(block_sum6 * inv_width + eps)
                if cutlass.const_expr(save_stats):
                    stats[row, 0] = norms[0]
                    stats[row, 1] = norms[1]
                    stats[row, 2] = norms[2]
        cute.arch.sync_threads()

        inv1 = cutlass.Float32(norms[0])
        inv2 = cutlass.Float32(norms[1])
        inv3 = cutlass.Float32(norms[2])
        w3 = cutlass.Float32(weight[0])
        w2 = cutlass.Float32(weight[1])
        w1 = cutlass.Float32(weight[2])
        shift = cutlass.Float32(bias[0])
        seed0 = cutlass.Uint32(seeds[0])
        seed1 = cutlass.Uint32(seeds[1])
        dropout_scale = cutlass.Float32(1.0 / (1.0 - dropout_p))
        output_values = cute.make_rmem_tensor(vector_width, output.element_type)
        dropout_values = cute.make_rmem_tensor(vector_width, cutlass.Float32)
        for group in range(thread, x.shape[1], block_threads):
            cute.autovec_copy(x[row, group, None], values)
            if cutlass.const_expr(dropout_p == 0.0):
                for item in cutlass.range_constexpr(vector_width):
                    dropout_values[item] = cutlass.Float32(1.0)
            else:
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
                    product0 = cutlass.Uint64(c0) * cutlass.Uint64(0xD2511F53)
                    product1 = cutlass.Uint64(c2) * cutlass.Uint64(0xCD9E8D57)
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
                value = cutlass.Float32(values[item])
                value2 = value * value
                value3 = value * value2
                result = dropout_values[item] * (
                    w3 * value3 * inv3
                    + w2 * value2 * inv2
                    + w1 * value * inv1
                    + shift
                )
                if cutlass.const_expr(output.element_type == cutlass.BFloat16):
                    output_values[item] = cutlass.BFloat16(result)
                else:
                    output_values[item] = result
            cute.autovec_copy(output_values, output[row, group, None])


    @cute.kernel
    def polynorm_backward_rows_kernel(
        x: cute.Tensor,
        seeds: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        stats: cute.Tensor,
        partials: cute.Tensor,
        grad_x: cute.Tensor,
        num_warps: cutlass.Constexpr[int],
        swizzle_b: cutlass.Constexpr[int],
        swizzle_m: cutlass.Constexpr[int],
        swizzle_s: cutlass.Constexpr[int],
        dropout_p: cutlass.Constexpr[float],
        dropout_threshold: cutlass.Constexpr[int],
    ):
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = thread % WARP_SIZE
        warp = thread // WARP_SIZE
        block_threads = num_warps * WARP_SIZE
        allocator = cutlass.utils.SmemAllocator()
        vector_width = x.shape[2]
        hidden = x.shape[1] * vector_width
        value_layout = cute.make_layout(
            (x.shape[1], vector_width), stride=(vector_width, 1)
        )
        grad_layout = cute.make_layout(
            (x.shape[1], vector_width), stride=(vector_width, 1)
        )
        value_layout = cute.make_composed_layout(
            cute.make_swizzle(swizzle_b, swizzle_m, swizzle_s), 0, value_layout
        )
        grad_layout = cute.make_composed_layout(
            cute.make_swizzle(swizzle_b, swizzle_m, swizzle_s), 0, grad_layout
        )
        values = allocator.allocate_tensor(
            element_type=x.element_type,
            layout=value_layout,
            byte_alignment=16,
        )
        grads = allocator.allocate_tensor(
            element_type=grad_output.element_type,
            layout=grad_layout,
            byte_alignment=16,
        )
        warp_partials = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((num_warps, 4)),
            byte_alignment=16,
        )
        totals = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((4,)),
            byte_alignment=16,
        )

        dot1 = cutlass.Float32(0.0)
        dot2 = cutlass.Float32(0.0)
        dot3 = cutlass.Float32(0.0)
        grad_sum = cutlass.Float32(0.0)
        value_fragment = cute.make_rmem_tensor(vector_width, x.element_type)
        grad_fragment = cute.make_rmem_tensor(vector_width, grad_output.element_type)
        dropout_values = cute.make_rmem_tensor(vector_width, cutlass.Float32)
        seed0 = cutlass.Uint32(seeds[0])
        seed1 = cutlass.Uint32(seeds[1])
        dropout_scale = cutlass.Float32(1.0 / (1.0 - dropout_p))
        for group in range(thread, x.shape[1], block_threads):
            cute.autovec_copy(x[row, group, None], value_fragment)
            cute.autovec_copy(grad_output[row, group, None], grad_fragment)
            cute.autovec_copy(value_fragment, values[group, None])
            cute.autovec_copy(grad_fragment, grads[group, None])
            if cutlass.const_expr(dropout_p == 0.0):
                for item in cutlass.range_constexpr(vector_width):
                    dropout_values[item] = cutlass.Float32(1.0)
            else:
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
                    product0 = cutlass.Uint64(c0) * cutlass.Uint64(0xD2511F53)
                    product1 = cutlass.Uint64(c2) * cutlass.Uint64(0xCD9E8D57)
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
                value = cutlass.Float32(value_fragment[item])
                grad = cutlass.Float32(grad_fragment[item]) * dropout_values[item]
                value2 = value * value
                value3 = value * value2
                dot1 = dot1 + grad * value
                dot2 = dot2 + grad * value2
                dot3 = dot3 + grad * value3
                grad_sum = grad_sum + grad

        dot1 = cute.arch.warp_reduction_sum(dot1)
        dot2 = cute.arch.warp_reduction_sum(dot2)
        dot3 = cute.arch.warp_reduction_sum(dot3)
        grad_sum = cute.arch.warp_reduction_sum(grad_sum)
        if lane == 0:
            warp_partials[warp, 0] = dot1
            warp_partials[warp, 1] = dot2
            warp_partials[warp, 2] = dot3
            warp_partials[warp, 3] = grad_sum
        cute.arch.sync_threads()

        if warp == 0:
            block_dot1 = cutlass.Float32(0.0)
            block_dot2 = cutlass.Float32(0.0)
            block_dot3 = cutlass.Float32(0.0)
            block_grad_sum = cutlass.Float32(0.0)
            if lane < num_warps:
                block_dot1 = cutlass.Float32(warp_partials[lane, 0])
                block_dot2 = cutlass.Float32(warp_partials[lane, 1])
                block_dot3 = cutlass.Float32(warp_partials[lane, 2])
                block_grad_sum = cutlass.Float32(warp_partials[lane, 3])
            block_dot1 = cute.arch.warp_reduction_sum(block_dot1)
            block_dot2 = cute.arch.warp_reduction_sum(block_dot2)
            block_dot3 = cute.arch.warp_reduction_sum(block_dot3)
            block_grad_sum = cute.arch.warp_reduction_sum(block_grad_sum)
            if lane == 0:
                totals[0] = block_dot1
                totals[1] = block_dot2
                totals[2] = block_dot3
                totals[3] = block_grad_sum
                inv1 = cutlass.Float32(stats[row, 0])
                inv2 = cutlass.Float32(stats[row, 1])
                inv3 = cutlass.Float32(stats[row, 2])
                partials[row, 0] = block_dot3 * inv3
                partials[row, 1] = block_dot2 * inv2
                partials[row, 2] = block_dot1 * inv1
                partials[row, 3] = block_grad_sum
        cute.arch.sync_threads()

        dot1 = cutlass.Float32(totals[0])
        dot2 = cutlass.Float32(totals[1])
        dot3 = cutlass.Float32(totals[2])
        inv1 = cutlass.Float32(stats[row, 0])
        inv2 = cutlass.Float32(stats[row, 1])
        inv3 = cutlass.Float32(stats[row, 2])
        w3 = cutlass.Float32(weight[0])
        w2 = cutlass.Float32(weight[1])
        w1 = cutlass.Float32(weight[2])
        inv_width = cutlass.Float32(1.0) / cutlass.Float32(hidden)
        direct1 = w1 * inv1
        direct2 = cutlass.Float32(2.0) * w2 * inv2
        direct3 = cutlass.Float32(3.0) * w3 * inv3
        corr1 = w1 * inv1 * inv1 * inv1 * dot1 * inv_width
        corr2 = (
            cutlass.Float32(2.0)
            * w2
            * inv2
            * inv2
            * inv2
            * dot2
            * inv_width
        )
        corr3 = (
            cutlass.Float32(3.0)
            * w3
            * inv3
            * inv3
            * inv3
            * dot3
            * inv_width
        )
        grad_x_fragment = cute.make_rmem_tensor(vector_width, grad_x.element_type)
        for group in range(thread, x.shape[1], block_threads):
            cute.autovec_copy(values[group, None], value_fragment)
            cute.autovec_copy(grads[group, None], grad_fragment)
            if cutlass.const_expr(dropout_p == 0.0):
                for item in cutlass.range_constexpr(vector_width):
                    dropout_values[item] = cutlass.Float32(1.0)
            else:
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
                    product0 = cutlass.Uint64(c0) * cutlass.Uint64(0xD2511F53)
                    product1 = cutlass.Uint64(c2) * cutlass.Uint64(0xCD9E8D57)
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
                value = cutlass.Float32(value_fragment[item])
                grad = cutlass.Float32(grad_fragment[item]) * dropout_values[item]
                value2 = value * value
                direct = direct1 + value * (direct2 + direct3 * value)
                correction = value * (corr1 + value2 * (corr2 + corr3 * value2))
                result = grad * direct - correction
                if cutlass.const_expr(grad_x.element_type == cutlass.BFloat16):
                    grad_x_fragment[item] = cutlass.BFloat16(result)
                else:
                    grad_x_fragment[item] = result
            cute.autovec_copy(grad_x_fragment, grad_x[row, group, None])


    @cute.kernel
    def polynorm_backward_params_kernel(
        partials: cute.Tensor,
        grad_weight: cute.Tensor,
        grad_bias: cute.Tensor,
    ):
        thread, _, _ = cute.arch.thread_idx()
        lane = thread % WARP_SIZE
        warp = thread // WARP_SIZE
        parameter_warps = 8
        parameter_threads = parameter_warps * WARP_SIZE
        allocator = cutlass.utils.SmemAllocator()
        warp_sums = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((parameter_warps, 4)),
            byte_alignment=16,
        )
        value0 = cutlass.Float32(0.0)
        value1 = cutlass.Float32(0.0)
        value2 = cutlass.Float32(0.0)
        value3 = cutlass.Float32(0.0)
        for row in range(thread, partials.shape[0], parameter_threads):
            value0 = value0 + cutlass.Float32(partials[row, 0])
            value1 = value1 + cutlass.Float32(partials[row, 1])
            value2 = value2 + cutlass.Float32(partials[row, 2])
            value3 = value3 + cutlass.Float32(partials[row, 3])
        value0 = cute.arch.warp_reduction_sum(value0)
        value1 = cute.arch.warp_reduction_sum(value1)
        value2 = cute.arch.warp_reduction_sum(value2)
        value3 = cute.arch.warp_reduction_sum(value3)
        if lane == 0:
            warp_sums[warp, 0] = value0
            warp_sums[warp, 1] = value1
            warp_sums[warp, 2] = value2
            warp_sums[warp, 3] = value3
        cute.arch.sync_threads()

        if warp == 0:
            block0 = cutlass.Float32(0.0)
            block1 = cutlass.Float32(0.0)
            block2 = cutlass.Float32(0.0)
            block3 = cutlass.Float32(0.0)
            if lane < parameter_warps:
                block0 = cutlass.Float32(warp_sums[lane, 0])
                block1 = cutlass.Float32(warp_sums[lane, 1])
                block2 = cutlass.Float32(warp_sums[lane, 2])
                block3 = cutlass.Float32(warp_sums[lane, 3])
            block0 = cute.arch.warp_reduction_sum(block0)
            block1 = cute.arch.warp_reduction_sum(block1)
            block2 = cute.arch.warp_reduction_sum(block2)
            block3 = cute.arch.warp_reduction_sum(block3)
            if lane == 0:
                if cutlass.const_expr(grad_weight.element_type == cutlass.BFloat16):
                    grad_weight[0] = cutlass.BFloat16(block0)
                    grad_weight[1] = cutlass.BFloat16(block1)
                    grad_weight[2] = cutlass.BFloat16(block2)
                    grad_bias[0] = cutlass.BFloat16(block3)
                else:
                    grad_weight[0] = block0
                    grad_weight[1] = block1
                    grad_weight[2] = block2
                    grad_bias[0] = block3


    @cute.jit
    def _launch_forward_on_stream(
        x: cute.Tensor,
        seeds: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor,
        output: cute.Tensor,
        stats: cute.Tensor,
        stream: cuda.CUstream,
        num_warps: cutlass.Constexpr[int],
        save_stats: cutlass.Constexpr[bool],
        dropout_p: cutlass.Constexpr[float],
        dropout_threshold: cutlass.Constexpr[int],
    ):
        polynorm_forward_kernel(
            x,
            seeds,
            weight,
            bias,
            output,
            stats,
            num_warps,
            save_stats,
            dropout_p,
            dropout_threshold,
        ).launch(
            grid=(x.shape[0], 1, 1),
            block=(num_warps * WARP_SIZE, 1, 1),
            stream=stream,
        )


    @cute.jit
    def _launch_backward_on_stream(
        x: cute.Tensor,
        seeds: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        stats: cute.Tensor,
        partials: cute.Tensor,
        grad_x: cute.Tensor,
        grad_weight: cute.Tensor,
        grad_bias: cute.Tensor,
        stream: cuda.CUstream,
        num_warps: cutlass.Constexpr[int],
        swizzle_b: cutlass.Constexpr[int],
        swizzle_m: cutlass.Constexpr[int],
        swizzle_s: cutlass.Constexpr[int],
        dropout_p: cutlass.Constexpr[float],
        dropout_threshold: cutlass.Constexpr[int],
    ):
        polynorm_backward_rows_kernel(
            x,
            seeds,
            weight,
            grad_output,
            stats,
            partials,
            grad_x,
            num_warps,
            swizzle_b,
            swizzle_m,
            swizzle_s,
            dropout_p,
            dropout_threshold,
        ).launch(
            grid=(x.shape[0], 1, 1),
            block=(num_warps * WARP_SIZE, 1, 1),
            stream=stream,
        )
        polynorm_backward_params_kernel(
            partials, grad_weight, grad_bias
        ).launch(
            grid=(1, 1, 1),
            block=(8 * WARP_SIZE, 1, 1),
            stream=stream,
        )


@dataclass(frozen=True)
class _KernelKey:
    operation: str
    device_index: int
    capability: tuple[int, int]
    shapes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    dropout_p: float


_CACHE: dict[_KernelKey, Any] = {}
_CACHE_LOCK = threading.Lock()


def is_available() -> bool:
    return _IMPORT_ERROR is None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def _key(
    operation: str,
    tensors: tuple[torch.Tensor, ...],
    dropout_p: float,
) -> _KernelKey:
    device_index = tensors[0].device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _KernelKey(
        operation=operation,
        device_index=device_index,
        capability=torch.cuda.get_device_capability(device_index),
        shapes=tuple(tuple(tensor.shape) for tensor in tensors),
        strides=tuple(tuple(tensor.stride()) for tensor in tensors),
        dtypes=tuple(tensor.dtype for tensor in tensors),
        dropout_p=float(dropout_p),
    )


def _descriptors(tensors: tuple[torch.Tensor, ...]) -> list[Any]:
    if from_dlpack is None:
        raise RuntimeError("NVIDIA CuTe DSL is unavailable") from _IMPORT_ERROR
    return [
        from_dlpack(
            tensor,
            assumed_align=DESCRIPTOR_ALIGNMENT,
            enable_tvm_ffi=True,
        )
        for tensor in tensors
    ]


def _compile(
    operation: str,
    tensors: tuple[torch.Tensor, ...],
    dropout_p: float,
):
    if cute is None or make_fake_stream is None:
        raise RuntimeError("NVIDIA CuTe DSL is unavailable") from _IMPORT_ERROR
    key = _key(operation, tensors, dropout_p)
    dropout_threshold = math.ceil(dropout_p * (1 << 32))
    with _CACHE_LOCK:
        compiled = _CACHE.get(key)
        if compiled is not None:
            return compiled
        if operation in ("forward", "inference"):
            compiled = cute.compile(
                _launch_forward_on_stream,
                *_descriptors(tensors),
                make_fake_stream(),
                num_warps=FORWARD_WARPS,
                save_stats=operation == "forward",
                dropout_p=dropout_p,
                dropout_threshold=dropout_threshold,
                options="--enable-tvm-ffi",
            )
        elif operation == "backward":
            compiled = cute.compile(
                _launch_backward_on_stream,
                *_descriptors(tensors),
                make_fake_stream(),
                num_warps=BACKWARD_WARPS,
                swizzle_b=SWIZZLE_B,
                swizzle_m=SWIZZLE_M,
                swizzle_s=SWIZZLE_S,
                dropout_p=dropout_p,
                dropout_threshold=dropout_threshold,
                options="--enable-tvm-ffi",
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")
        _CACHE[key] = compiled
        return compiled


def _stream(device: torch.device) -> Any:
    if cuda is None:
        raise RuntimeError("NVIDIA CuTe DSL is unavailable") from _IMPORT_ERROR
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _aligned_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous() and tensor.data_ptr() % DESCRIPTOR_ALIGNMENT == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def backward_shared_memory_bytes(hidden: int, dtype: torch.dtype) -> int:
    element_size = 2 if dtype == torch.bfloat16 else 4
    row_storage = 2 * hidden * element_size
    reduction_storage = (BACKWARD_WARPS * 4 + 4) * 4
    return row_storage + reduction_storage + 64


def _kernel_shape(tensor: torch.Tensor) -> tuple[int, int, int]:
    if tensor.ndim != 2 or tensor.shape[1] % VECTOR_WIDTH:
        raise ValueError(
            f"input must be 2D with hidden divisible by {VECTOR_WIDTH}"
        )
    return tensor.shape[0], tensor.shape[1] // VECTOR_WIDTH, VECTOR_WIDTH


def forward(
    x: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    dropout_p: float,
    save_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _aligned_contiguous(x)
    seeds = _aligned_contiguous(seeds)
    weight = _aligned_contiguous(weight)
    bias = _aligned_contiguous(bias)
    output = torch.empty_like(x)
    stats_rows = x.shape[0] if save_stats else 1
    stats = torch.empty((stats_rows, 3), device=x.device, dtype=torch.float32)
    shape = _kernel_shape(x)
    tensors = (
        x.view(shape),
        seeds,
        weight,
        bias,
        output.view(shape),
        stats,
    )
    operation = "forward" if save_stats else "inference"
    with torch.cuda.device(x.device):
        compiled = _compile(operation, tensors, dropout_p)
        compiled(*tensors, _stream(x.device))
    return output, stats


def backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    stats: torch.Tensor,
    *,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_output = _aligned_contiguous(grad_output)
    x = _aligned_contiguous(x)
    seeds = _aligned_contiguous(seeds)
    weight = _aligned_contiguous(weight)
    stats = _aligned_contiguous(stats)
    partials = torch.empty((x.shape[0], 4), device=x.device, dtype=torch.float32)
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty((1,), device=weight.device, dtype=weight.dtype)
    shape = _kernel_shape(x)
    tensors = (
        x.view(shape),
        seeds,
        weight,
        grad_output.view(shape),
        stats,
        partials,
        grad_x.view(shape),
        grad_weight,
        grad_bias,
    )
    with torch.cuda.device(x.device):
        compiled = _compile("backward", tensors, dropout_p)
        compiled(*tensors, _stream(x.device))
    return grad_x, grad_weight, grad_bias


__all__ = [
    "backward",
    "backward_shared_memory_bytes",
    "forward",
    "import_error",
    "is_available",
]
