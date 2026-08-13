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
DESCRIPTOR_ALIGNMENT = 32
SWIZZLE_B = 2
SWIZZLE_M = 4
SWIZZLE_S = 3
MAX_DROPOUT_P = ((1 << 32) - 1) / (1 << 32)


if cute is not None and cutlass is not None and cuda is not None:
    @cute.kernel
    def gupn_exclusive_forward_kernel(
        gate0: cute.Tensor,
        up: cute.Tensor,
        gate_row: cute.Tensor,
        polynorm_weight: cute.Tensor,
        polynorm_bias: cute.Tensor,
        exclusive_logits: cute.Tensor,
        down_column: cute.Tensor,
        seeds: cute.Tensor,
        output: cute.Tensor,
        num_warps: cutlass.Constexpr[int],
        use_xor: cutlass.Constexpr[bool],
        dropout_p: cutlass.Constexpr[float],
        dropout_threshold: cutlass.Constexpr[int],
    ):
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = thread % WARP_SIZE
        warp = thread // WARP_SIZE
        block_threads = num_warps * WARP_SIZE
        vector_width = gate0.shape[2]
        hidden = gate0.shape[1] * vector_width

        allocator = cutlass.utils.SmemAllocator()
        cache_layout = cute.make_layout(
            (gate0.shape[1], vector_width),
            stride=(vector_width, 1),
        )
        if cutlass.const_expr(use_xor):
            cache_layout = cute.make_composed_layout(
                cute.make_swizzle(SWIZZLE_B, SWIZZLE_M, SWIZZLE_S),
                0,
                cache_layout,
            )
        gate_cache = allocator.allocate_tensor(
            element_type=gate0.element_type,
            layout=cache_layout,
            byte_alignment=16,
        )
        partials = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((num_warps, 3)),
            byte_alignment=16,
        )
        scalars = allocator.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((9,)),
            byte_alignment=16,
        )

        gate0_values = cute.make_rmem_tensor(vector_width, gate0.element_type)
        multiplier_values = cute.make_rmem_tensor(vector_width, gate_row.element_type)
        gate_values = cute.make_rmem_tensor(vector_width, gate0.element_type)

        # Stage 1: apply the gate row multiplier with a BF16 boundary, cache G,
        # and reduce the three raw PolyNorm moments.
        sum2 = cutlass.Float32(0.0)
        sum4 = cutlass.Float32(0.0)
        sum6 = cutlass.Float32(0.0)
        for group in range(thread, gate0.shape[1], block_threads):
            cute.autovec_copy(gate0[row, group, None], gate0_values)
            cute.autovec_copy(gate_row[group, None], multiplier_values)
            for item in cutlass.range_constexpr(vector_width):
                scaled = cutlass.Float32(gate0_values[item]) * cutlass.Float32(
                    multiplier_values[item]
                )
                gate = cutlass.BFloat16(scaled)
                gate_values[item] = gate
                value = cutlass.Float32(gate)
                value2 = value * value
                value3 = value * value2
                sum2 = sum2 + value2
                sum4 = sum4 + value2 * value2
                sum6 = sum6 + value3 * value3
            cute.autovec_copy(gate_values, gate_cache[group, None])

        sum2 = cute.arch.warp_reduction_sum(sum2)
        sum4 = cute.arch.warp_reduction_sum(sum4)
        sum6 = cute.arch.warp_reduction_sum(sum6)
        if lane == 0:
            partials[warp, 0] = sum2
            partials[warp, 1] = sum4
            partials[warp, 2] = sum6
        cute.arch.sync_threads()
        if warp == 0:
            total2 = cutlass.Float32(0.0)
            total4 = cutlass.Float32(0.0)
            total6 = cutlass.Float32(0.0)
            if lane < num_warps:
                total2 = cutlass.Float32(partials[lane, 0])
                total4 = cutlass.Float32(partials[lane, 1])
                total6 = cutlass.Float32(partials[lane, 2])
            total2 = cute.arch.warp_reduction_sum(total2)
            total4 = cute.arch.warp_reduction_sum(total4)
            total6 = cute.arch.warp_reduction_sum(total6)
            if lane == 0:
                inv_width = cutlass.Float32(1.0) / cutlass.Float32(hidden)
                eps = cutlass.Float32(1.0e-6)
                scalars[0] = cute.math.rsqrt(total2 * inv_width + eps)
                scalars[1] = cute.math.rsqrt(total4 * inv_width + eps)
                scalars[2] = cute.math.rsqrt(total6 * inv_width + eps)
        cute.arch.sync_threads()

        inv1_bf16 = cutlass.BFloat16(scalars[0])
        inv2_bf16 = cutlass.BFloat16(scalars[1])
        inv3_bf16 = cutlass.BFloat16(scalars[2])

        # Stage 2: build the three normalized branches at the same BF16
        # boundary as the model, then reduce the exclusive projections.
        reference_norm_sq = cutlass.Float32(0.0)
        dot2 = cutlass.Float32(0.0)
        dot3 = cutlass.Float32(0.0)
        for group in range(thread, gate0.shape[1], block_threads):
            cute.autovec_copy(gate_cache[group, None], gate_values)
            for item in cutlass.range_constexpr(vector_width):
                value = cutlass.Float32(gate_values[item])
                value2 = value * value
                value3 = value * value2
                x1 = cutlass.BFloat16(value * cutlass.Float32(inv1_bf16))
                x2 = cutlass.BFloat16(value2 * cutlass.Float32(inv2_bf16))
                x3 = cutlass.BFloat16(value3 * cutlass.Float32(inv3_bf16))
                x1f = cutlass.Float32(x1)
                reference_norm_sq = reference_norm_sq + x1f * x1f
                dot2 = dot2 + cutlass.Float32(x2) * x1f
                dot3 = dot3 + cutlass.Float32(x3) * x1f

        reference_norm_sq = cute.arch.warp_reduction_sum(reference_norm_sq)
        dot2 = cute.arch.warp_reduction_sum(dot2)
        dot3 = cute.arch.warp_reduction_sum(dot3)
        if lane == 0:
            partials[warp, 0] = reference_norm_sq
            partials[warp, 1] = dot2
            partials[warp, 2] = dot3
        cute.arch.sync_threads()
        if warp == 0:
            total_ref = cutlass.Float32(0.0)
            total_dot2 = cutlass.Float32(0.0)
            total_dot3 = cutlass.Float32(0.0)
            if lane < num_warps:
                total_ref = cutlass.Float32(partials[lane, 0])
                total_dot2 = cutlass.Float32(partials[lane, 1])
                total_dot3 = cutlass.Float32(partials[lane, 2])
            total_ref = cute.arch.warp_reduction_sum(total_ref)
            total_dot2 = cute.arch.warp_reduction_sum(total_dot2)
            total_dot3 = cute.arch.warp_reduction_sum(total_dot3)
            if lane == 0:
                proj_eps = cutlass.Float32(1.0e-6)
                denominator = cute.math.max(total_ref, proj_eps)
                scalars[3] = total_dot2 / denominator
                scalars[4] = total_dot3 / denominator
                logit2 = cutlass.Float32(exclusive_logits[0])
                logit3 = cutlass.Float32(exclusive_logits[1])
                scalars[5] = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.math.exp(-logit2)
                )
                scalars[6] = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.math.exp(-logit3)
                )
        cute.arch.sync_threads()

        projection2 = cutlass.BFloat16(scalars[3])
        projection3 = cutlass.BFloat16(scalars[4])
        alpha2 = cutlass.BFloat16(scalars[5])
        alpha3 = cutlass.BFloat16(scalars[6])

        # Stage 3: reproduce the two BF16 residual branches and obtain their
        # independent renormalization factors.
        residual2_sq = cutlass.Float32(0.0)
        residual3_sq = cutlass.Float32(0.0)
        for group in range(thread, gate0.shape[1], block_threads):
            cute.autovec_copy(gate_cache[group, None], gate_values)
            for item in cutlass.range_constexpr(vector_width):
                value = cutlass.Float32(gate_values[item])
                value2 = value * value
                value3 = value * value2
                x1 = cutlass.BFloat16(value * cutlass.Float32(inv1_bf16))
                x2 = cutlass.BFloat16(value2 * cutlass.Float32(inv2_bf16))
                x3 = cutlass.BFloat16(value3 * cutlass.Float32(inv3_bf16))
                coeff2 = cutlass.BFloat16(
                    cutlass.Float32(alpha2) * cutlass.Float32(projection2)
                )
                coeff3 = cutlass.BFloat16(
                    cutlass.Float32(alpha3) * cutlass.Float32(projection3)
                )
                component2 = cutlass.BFloat16(
                    cutlass.Float32(coeff2) * cutlass.Float32(x1)
                )
                component3 = cutlass.BFloat16(
                    cutlass.Float32(coeff3) * cutlass.Float32(x1)
                )
                residual2 = cutlass.BFloat16(
                    cutlass.Float32(x2) - cutlass.Float32(component2)
                )
                residual3 = cutlass.BFloat16(
                    cutlass.Float32(x3) - cutlass.Float32(component3)
                )
                residual2f = cutlass.Float32(residual2)
                residual3f = cutlass.Float32(residual3)
                residual2_sq = residual2_sq + residual2f * residual2f
                residual3_sq = residual3_sq + residual3f * residual3f

        residual2_sq = cute.arch.warp_reduction_sum(residual2_sq)
        residual3_sq = cute.arch.warp_reduction_sum(residual3_sq)
        if lane == 0:
            partials[warp, 0] = residual2_sq
            partials[warp, 1] = residual3_sq
        cute.arch.sync_threads()
        if warp == 0:
            total_residual2 = cutlass.Float32(0.0)
            total_residual3 = cutlass.Float32(0.0)
            if lane < num_warps:
                total_residual2 = cutlass.Float32(partials[lane, 0])
                total_residual3 = cutlass.Float32(partials[lane, 1])
            total_residual2 = cute.arch.warp_reduction_sum(total_residual2)
            total_residual3 = cute.arch.warp_reduction_sum(total_residual3)
            if lane == 0:
                inv_width = cutlass.Float32(1.0) / cutlass.Float32(hidden)
                eps = cutlass.Float32(1.0e-6)
                scalars[7] = cute.math.rsqrt(total_residual2 * inv_width + eps)
                scalars[8] = cute.math.rsqrt(total_residual3 * inv_width + eps)
        cute.arch.sync_threads()

        residual2_inv = cutlass.BFloat16(scalars[7])
        residual3_inv = cutlass.BFloat16(scalars[8])
        w3 = polynorm_weight[0]
        w2 = polynorm_weight[1]
        w1 = polynorm_weight[2]
        shift = polynorm_bias[0]
        seed0 = cutlass.Uint32(seeds[0])
        seed1 = cutlass.Uint32(seeds[1])
        dropout_scale = cutlass.Float32(1.0 / (1.0 - dropout_p))
        up_values = cute.make_rmem_tensor(vector_width, up.element_type)
        column_values = cute.make_rmem_tensor(vector_width, down_column.element_type)
        output_values = cute.make_rmem_tensor(vector_width, output.element_type)
        dropout_values = cute.make_rmem_tensor(vector_width, cutlass.Float32)

        # Stage 4: exclusive PolyNorm epilogue, Philox, Hadamard and down
        # column multiplier.  Every former BF16 tensor boundary is retained.
        for group in range(thread, gate0.shape[1], block_threads):
            cute.autovec_copy(gate_cache[group, None], gate_values)
            cute.autovec_copy(up[row, group, None], up_values)
            cute.autovec_copy(down_column[group, None], column_values)
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
                value = cutlass.Float32(gate_values[item])
                value2 = value * value
                value3 = value * value2
                x1 = cutlass.BFloat16(value * cutlass.Float32(inv1_bf16))
                x2 = cutlass.BFloat16(value2 * cutlass.Float32(inv2_bf16))
                x3 = cutlass.BFloat16(value3 * cutlass.Float32(inv3_bf16))
                coeff2 = cutlass.BFloat16(
                    cutlass.Float32(alpha2) * cutlass.Float32(projection2)
                )
                coeff3 = cutlass.BFloat16(
                    cutlass.Float32(alpha3) * cutlass.Float32(projection3)
                )
                component2 = cutlass.BFloat16(
                    cutlass.Float32(coeff2) * cutlass.Float32(x1)
                )
                component3 = cutlass.BFloat16(
                    cutlass.Float32(coeff3) * cutlass.Float32(x1)
                )
                residual2 = cutlass.BFloat16(
                    cutlass.Float32(x2) - cutlass.Float32(component2)
                )
                residual3 = cutlass.BFloat16(
                    cutlass.Float32(x3) - cutlass.Float32(component3)
                )
                exclusive2 = cutlass.BFloat16(
                    cutlass.Float32(residual2) * cutlass.Float32(residual2_inv)
                )
                exclusive3 = cutlass.BFloat16(
                    cutlass.Float32(residual3) * cutlass.Float32(residual3_inv)
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
                    cutlass.Float32(dropped) * cutlass.Float32(up_values[item])
                )
                result = cutlass.BFloat16(
                    cutlass.Float32(hadamard)
                    * cutlass.Float32(column_values[item])
                )
                output_values[item] = result
            cute.autovec_copy(output_values, output[row, group, None])


    @cute.jit
    def _launch_forward_on_stream(
        gate0: cute.Tensor,
        up: cute.Tensor,
        gate_row: cute.Tensor,
        polynorm_weight: cute.Tensor,
        polynorm_bias: cute.Tensor,
        exclusive_logits: cute.Tensor,
        down_column: cute.Tensor,
        seeds: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
        num_warps: cutlass.Constexpr[int],
        use_xor: cutlass.Constexpr[bool],
        dropout_p: cutlass.Constexpr[float],
        dropout_threshold: cutlass.Constexpr[int],
    ):
        gupn_exclusive_forward_kernel(
            gate0,
            up,
            gate_row,
            polynorm_weight,
            polynorm_bias,
            exclusive_logits,
            down_column,
            seeds,
            output,
            num_warps,
            use_xor,
            dropout_p,
            dropout_threshold,
        ).launch(
            grid=(gate0.shape[0], 1, 1),
            block=(num_warps * WARP_SIZE, 1, 1),
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
    return _IMPORT_ERROR is None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def _aligned_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous() and tensor.data_ptr() % DESCRIPTOR_ALIGNMENT == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _descriptors(tensors: tuple[torch.Tensor, ...]) -> list[Any]:
    if from_dlpack is None:
        raise RuntimeError("NVIDIA CuTe DSL is unavailable") from _IMPORT_ERROR
    return [
        from_dlpack(tensor, assumed_align=DESCRIPTOR_ALIGNMENT, enable_tvm_ffi=True)
        for tensor in tensors
    ]


def _stream(device: torch.device) -> Any:
    if cuda is None:
        raise RuntimeError("NVIDIA CuTe DSL is unavailable") from _IMPORT_ERROR
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _kernel_shape(tensor: torch.Tensor) -> tuple[int, int, int]:
    if tensor.ndim != 2 or tensor.shape[1] % VECTOR_WIDTH:
        raise ValueError(
            f"input must be 2D with hidden divisible by {VECTOR_WIDTH}"
        )
    return tensor.shape[0], tensor.shape[1] // VECTOR_WIDTH, VECTOR_WIDTH


def _validate(
    gate0: torch.Tensor,
    up: torch.Tensor,
    gate_row: torch.Tensor,
    polynorm_weight: torch.Tensor,
    polynorm_bias: torch.Tensor,
    exclusive_logits: torch.Tensor,
    down_column: torch.Tensor,
    seeds: torch.Tensor,
    dropout_p: float,
) -> None:
    if not is_available():
        raise RuntimeError("NVIDIA CuTe DSL is unavailable") from _IMPORT_ERROR
    if not gate0.is_cuda or gate0.dtype != torch.bfloat16 or gate0.ndim != 2:
        raise ValueError("gate0 must be a 2D CUDA BF16 tensor")
    if up.shape != gate0.shape or up.dtype != gate0.dtype or up.device != gate0.device:
        raise ValueError("up must match gate0")
    hidden = gate0.shape[1]
    expected = (
        ("gate_row", gate_row, (hidden,)),
        ("polynorm_weight", polynorm_weight, (3,)),
        ("polynorm_bias", polynorm_bias, (1,)),
        ("exclusive_logits", exclusive_logits, (2,)),
        ("down_column", down_column, (hidden,)),
        ("seeds", seeds, (4,)),
    )
    for name, tensor, shape in expected:
        if tensor.shape != shape or tensor.device != gate0.device:
            raise ValueError(f"{name} must have shape {shape} on {gate0.device}")
    for name, tensor, _shape in expected[:-1]:
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16")
    if seeds.dtype != torch.int64:
        raise ValueError("seeds must be int64")
    if hidden % VECTOR_WIDTH:
        raise ValueError(f"hidden must be divisible by {VECTOR_WIDTH}")
    if not 0.0 <= dropout_p <= MAX_DROPOUT_P:
        raise ValueError(f"dropout_p must be in [0, {MAX_DROPOUT_P}]")


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
                _launch_forward_on_stream,
                *_descriptors(tensors),
                make_fake_stream(),
                num_warps=FORWARD_WARPS,
                use_xor=use_xor,
                dropout_p=dropout_p,
                dropout_threshold=dropout_threshold,
                options="--enable-tvm-ffi",
            )
            _CACHE[key] = compiled
        return compiled


def forward(
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
) -> torch.Tensor:
    """Run the exclusive GUPN forward stage; this internal API has no autograd."""
    dropout_p = float(dropout_p)
    _validate(
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
    aligned = tuple(
        _aligned_contiguous(tensor)
        for tensor in (
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
    output = torch.empty_like(aligned[0])
    shape = _kernel_shape(aligned[0])
    tensors = (
        aligned[0].view(shape),
        aligned[1].view(shape),
        aligned[2].view(shape[1:]),
        aligned[3],
        aligned[4],
        aligned[5],
        aligned[6].view(shape[1:]),
        aligned[7],
        output.view(shape),
    )
    with torch.cuda.device(gate0.device):
        compiled = _compile(tensors, dropout_p, use_xor)
        compiled(*tensors, _stream(gate0.device))
    return output


__all__ = ["forward", "import_error", "is_available"]
