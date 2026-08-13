# Copyright (C) 2026. All Rights Reserved.
"""Compiler-safe boundary for the CUDA CCE training path.

The CCE implementation intentionally contains data-dependent compaction,
Python-side Triton autotuning, and custom reduction launch logic.  Letting
TorchDynamo trace those internals creates many small specialized graphs.  The
operators below keep that implementation opaque to Dynamo while registering
an explicit autograd formula, so the surrounding model remains one FX graph.
"""

from __future__ import annotations

import torch

from cut_cross_entropy.cce import CCEParams, LinearCrossEntropyFunction
from cut_cross_entropy.utils import TensorInfo, _build_flat_valids


class _FunctionContext:
    """Minimal context used to reuse the existing, tested CCE kernel drivers."""

    def __init__(self) -> None:
        self.saved_tensors: tuple[torch.Tensor | None, ...] = ()

    def save_for_backward(self, *tensors: torch.Tensor | None) -> None:
        self.saved_tensors = tensors

    def mark_non_differentiable(self, *tensors: torch.Tensor) -> None:
        del tensors


def _empty(ref: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    return ref.new_empty((0,), dtype=dtype)


def _pack_optional(value: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    return value if value is not None else _empty(ref)


def _unpack_optional(value: torch.Tensor, present: bool) -> torch.Tensor | None:
    return value if present else None


def _maximum_valid_rows(
    targets: torch.Tensor, shift: int, patch_training_enabled: bool
) -> int | torch.SymInt:
    """Return the input-shape capacity of CCE's compact valid-token domain."""
    if patch_training_enabled:
        return targets.numel() // targets.size(-1)
    return targets[..., shift:].numel()


def _pad_valid_rows(value: torch.Tensor, capacity: int) -> torch.Tensor:
    """Expose a static compiler shape while preserving the compact prefix."""
    padding = capacity - value.size(0)
    if padding < 0:
        raise RuntimeError(
            f"CCE produced {value.size(0)} valid rows for a capacity of {capacity}."
        )
    return torch.nn.functional.pad(value, (0, padding)) if padding else value


def _prepare_forward_inputs(
    e: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    shift: int,
    patch_training_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    batch_shape = targets.size()[:-1] if patch_training_enabled else targets.size()
    e = e.contiguous().flatten(0, -2)
    targets = targets.contiguous()
    if patch_training_enabled:
        targets = targets.reshape(-1, targets.size(-1))
        targets = torch.where(targets == ignore_index, -1, targets)
        return e, targets, _empty(e, dtype=torch.int64), batch_shape
    valids = _build_flat_valids(targets, ignore_index, shift)
    # The compiler-safe path is deliberately restricted to shift > 0.  In
    # that regime _build_flat_valids always returns an index tensor, including
    # the valid all-token and all-ignored edge cases.
    assert valids is not None
    targets = targets.flatten()
    if targets.data_ptr() % 16:
        targets = torch.nn.functional.pad(targets, (0, 1))[:-1]
    return e, targets, valids, batch_shape


def _prepare_backward_inputs(
    e: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    patch_training_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Restore the flattened layouts used by the existing backward kernels."""
    batch_shape = targets.size()[:-1] if patch_training_enabled else targets.size()
    e = e.contiguous().flatten(0, -2)
    if patch_training_enabled:
        targets = targets.contiguous().reshape(-1, targets.size(-1))
        targets = torch.where(targets == ignore_index, -1, targets)
        return e, targets, batch_shape
    targets = targets.contiguous().flatten()
    if targets.data_ptr() % 16:
        targets = torch.nn.functional.pad(targets, (0, 1))[:-1]
    return e, targets, batch_shape


@torch.library.custom_op(
    "cut_cross_entropy::cce_backward",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _cce_backward_op(
    grad_loss: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    lse: torch.Tensor,
    valids: torch.Tensor,
    mile_weight: torch.Tensor,
    patch_target_weight: torch.Tensor,
    mu: torch.Tensor,
    mu_vocab_size: torch.Tensor,
    compute_dtype_witness: torch.Tensor,
    forward_used_autocast: bool,
    e_requires_grad: bool,
    c_requires_grad: bool,
    bias_requires_grad: bool,
    ignore_index: int,
    softcap: float | None,
    shift: int,
    return_loss_metrics: bool,
    filter_eps: float | None,
    accum_e_fp32: bool,
    accum_c_fp32: bool,
    filter_e_grad: bool,
    filter_c_grad: bool,
    auto_mixed_grad_accum: bool,
    mile_enabled: bool,
    mile_gamma: float,
    mu_loss_enabled: bool,
    mu_loss_lambda: float,
    patch_training_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_e = e
    original_c = c
    original_bias = bias
    # The compiler-facing forward pads data-dependent saved rows to a static
    # input-shape capacity. Rebuild the compact indices inside this opaque,
    # CUDAGraph-unsafe operator and expose only the populated prefixes to the
    # existing backward implementation. This keeps Dynamo/AOT shapes static
    # without changing the CCE kernels or requiring a global compiler flag.
    if patch_training_enabled:
        runtime_valids = None
    else:
        runtime_valids = _build_flat_valids(targets, ignore_index, shift)
        assert runtime_valids is not None
        valid_count = runtime_valids.size(0)
        lse = lse[:valid_count]
        if mile_enabled:
            mile_weight = mile_weight[:valid_count]
    e, targets, batch_shape = _prepare_backward_inputs(
        e, targets, ignore_index, patch_training_enabled
    )
    compute_dtype = compute_dtype_witness.dtype
    e = e.to(dtype=compute_dtype)
    c = c.to(dtype=compute_dtype)
    if bias is not None:
        bias = bias.to(dtype=compute_dtype)

    params = CCEParams(
        targets=targets,
        valids=runtime_valids,
        softcap=softcap,
        reduction="mean",
        filter_eps=filter_eps,
        shift=shift,
        batch_shape=batch_shape,
        accum_e_fp32=accum_e_fp32,
        accum_c_fp32=accum_c_fp32,
        filter_e_grad=filter_e_grad,
        filter_c_grad=filter_c_grad,
        vocab_parallel_options=None,
        return_lse=False,
        return_loss_metrics=return_loss_metrics,
        auto_mixed_grad_accum=auto_mixed_grad_accum,
        mile_gamma=mile_gamma if mile_enabled else None,
        mu_loss_lambda=mu_loss_lambda if mu_loss_enabled else None,
        patch_training_enabled=patch_training_enabled,
    )
    ctx = _FunctionContext()
    ctx.saved_tensors = (
        e,
        c,
        bias,
        lse,
        targets,
        runtime_valids,
        _unpack_optional(mile_weight, mile_enabled or patch_training_enabled),
        _unpack_optional(patch_target_weight, patch_training_enabled),
        _unpack_optional(mu, mu_loss_enabled),
        _unpack_optional(mu_vocab_size, mu_loss_enabled),
    )
    ctx.params = params
    ctx.e_info = TensorInfo(original_e.dtype, e_requires_grad)
    ctx.c_info = TensorInfo(original_c.dtype, c_requires_grad)
    ctx.bias_info = (
        TensorInfo(original_bias.dtype, bias_requires_grad) if original_bias is not None else None
    )
    # Recreate the custom_fwd/custom_bwd contract from the eager autograd
    # function. The saved operands already have the witnessed compute dtype,
    # while _fwd_used_autocast makes custom_bwd restore the same CUDA autocast
    # context used by eager. Marking it false changes kernel-side dtype policy
    # for FP32 storage inputs even though their saved tensors are FP16/BF16.
    ctx._fwd_used_autocast = forward_used_autocast
    ctx._dtype = compute_dtype
    de, dc, dbias, _params_grad = LinearCrossEntropyFunction.backward(ctx, grad_loss, None, None)
    return (
        de.view_as(original_e) if de is not None else _empty(original_e),
        dc if dc is not None else _empty(original_c),
        dbias if dbias is not None else _empty(original_e),
    )


@_cce_backward_op.register_fake
def _cce_backward_fake(
    grad_loss: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    lse: torch.Tensor,
    valids: torch.Tensor,
    mile_weight: torch.Tensor,
    patch_target_weight: torch.Tensor,
    mu: torch.Tensor,
    mu_vocab_size: torch.Tensor,
    compute_dtype_witness: torch.Tensor,
    forward_used_autocast: bool,
    e_requires_grad: bool,
    c_requires_grad: bool,
    bias_requires_grad: bool,
    ignore_index: int,
    softcap: float | None,
    shift: int,
    return_loss_metrics: bool,
    filter_eps: float | None,
    accum_e_fp32: bool,
    accum_c_fp32: bool,
    filter_e_grad: bool,
    filter_c_grad: bool,
    auto_mixed_grad_accum: bool,
    mile_enabled: bool,
    mile_gamma: float,
    mu_loss_enabled: bool,
    mu_loss_lambda: float,
    patch_training_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        grad_loss,
        targets,
        lse,
        valids,
        mile_weight,
        patch_target_weight,
        mu,
        mu_vocab_size,
        compute_dtype_witness,
        forward_used_autocast,
        ignore_index,
        softcap,
        shift,
        return_loss_metrics,
        filter_eps,
        accum_e_fp32,
        accum_c_fp32,
        filter_e_grad,
        auto_mixed_grad_accum,
        mile_enabled,
        mile_gamma,
        mu_loss_enabled,
        mu_loss_lambda,
        patch_training_enabled,
    )
    # The real path makes e contiguous before the kernel and reshapes that
    # contiguous gradient back to the original shape.
    de = e.new_empty(e.shape) if e_requires_grad else _empty(e)
    dc = torch.empty_like(c) if c_requires_grad else _empty(c)
    dbias = (
        torch.empty_like(bias)
        if bias is not None and bias_requires_grad
        else _empty(e)
    )
    return de, dc, dbias


@torch.library.custom_op(
    "cut_cross_entropy::cce_forward",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _cce_forward_op(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    e_requires_grad: bool,
    c_requires_grad: bool,
    bias_requires_grad: bool,
    ignore_index: int,
    softcap: float | None,
    shift: int,
    return_loss_metrics: bool,
    filter_eps: float | None,
    accum_e_fp32: bool,
    accum_c_fp32: bool,
    filter_e_grad: bool,
    filter_c_grad: bool,
    auto_mixed_grad_accum: bool,
    mile_enabled: bool,
    mile_gamma: float,
    mu_loss_enabled: bool,
    mu_loss_lambda: float,
    patch_training_enabled: bool,
    compute_dtype_is_bf16: bool,
    forward_used_autocast: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    valid_capacity = _maximum_valid_rows(targets, shift, patch_training_enabled)
    e, targets, valids, batch_shape = _prepare_forward_inputs(
        e, targets, ignore_index, shift, patch_training_enabled
    )
    # Custom-op backend implementations run below Autograd.  Recreate only the
    # metadata used by the existing kernel driver; no input storage is mutated.
    e = e.detach().requires_grad_(e_requires_grad)
    c = c.detach().requires_grad_(c_requires_grad)
    if bias is not None:
        bias = bias.detach().requires_grad_(bias_requires_grad)
    params = CCEParams(
        targets=targets,
        valids=None if patch_training_enabled else valids,
        softcap=softcap,
        reduction="mean",
        filter_eps=filter_eps,
        shift=shift,
        batch_shape=batch_shape,
        accum_e_fp32=accum_e_fp32,
        accum_c_fp32=accum_c_fp32,
        filter_e_grad=filter_e_grad,
        filter_c_grad=filter_c_grad,
        vocab_parallel_options=None,
        return_lse=False,
        return_loss_metrics=return_loss_metrics,
        auto_mixed_grad_accum=auto_mixed_grad_accum,
        mile_gamma=mile_gamma if mile_enabled else None,
        mu_loss_lambda=mu_loss_lambda if mu_loss_enabled else None,
        patch_training_enabled=patch_training_enabled,
    )
    kernel_ctx = _FunctionContext()
    compute_dtype = torch.bfloat16 if compute_dtype_is_bf16 else torch.float16
    # Inductor may invoke the opaque custom op without the ambient autocast
    # context that was active while Dynamo captured its caller. Recreate that
    # context explicitly so this backend follows the eager forward's cast
    # order, including evaluating mu-loss before e/c/bias are converted.
    with torch.autocast(
        "cuda", dtype=compute_dtype, enabled=forward_used_autocast
    ):
        loss, _ret_lse, loss_metrics = LinearCrossEntropyFunction.forward(
            kernel_ctx, e, c, bias, params
        )
    (
        saved_e,
        _saved_c,
        _saved_bias,
        lse,
        _saved_targets,
        saved_valids,
        mile_weight,
        patch_target_weight,
        mu,
        mu_vocab_size,
    ) = kernel_ctx.saved_tensors
    if patch_training_enabled:
        assert saved_valids is None
        saved_valids = _empty(saved_e, dtype=torch.int64)
    else:
        assert saved_valids is not None
        lse = _pad_valid_rows(lse, valid_capacity)
        saved_valids = _pad_valid_rows(saved_valids, valid_capacity)
        if mile_weight is not None:
            mile_weight = _pad_valid_rows(mile_weight, valid_capacity)
    return (
        loss,
        _pack_optional(loss_metrics, loss),
        lse,
        saved_valids,
        _pack_optional(mile_weight, loss),
        _pack_optional(patch_target_weight, loss),
        _pack_optional(mu, loss),
        _pack_optional(mu_vocab_size, loss),
        saved_e.new_empty(
            (),
            dtype=compute_dtype,
        ),
    )


@_cce_forward_op.register_fake
def _cce_forward_fake(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    e_requires_grad: bool,
    c_requires_grad: bool,
    bias_requires_grad: bool,
    ignore_index: int,
    softcap: float | None,
    shift: int,
    return_loss_metrics: bool,
    filter_eps: float | None,
    accum_e_fp32: bool,
    accum_c_fp32: bool,
    filter_e_grad: bool,
    filter_c_grad: bool,
    auto_mixed_grad_accum: bool,
    mile_enabled: bool,
    mile_gamma: float,
    mu_loss_enabled: bool,
    mu_loss_lambda: float,
    patch_training_enabled: bool,
    compute_dtype_is_bf16: bool,
    forward_used_autocast: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del (
        ignore_index,
        softcap,
        accum_e_fp32,
        accum_c_fp32,
        auto_mixed_grad_accum,
        filter_eps,
        filter_e_grad,
        filter_c_grad,
        mile_gamma,
        mu_loss_lambda,
        forward_used_autocast,
    )
    valid_capacity = _maximum_valid_rows(targets, shift, patch_training_enabled)
    lse = e.new_empty((valid_capacity,), dtype=torch.float32)
    # torch.nonzero, used by _build_flat_valids, returns int64 indices.
    valids = (
        _empty(e, dtype=torch.int64)
        if patch_training_enabled
        else e.new_empty((valid_capacity,), dtype=torch.int64)
    )
    # Each absent optional gets its own placeholder. Returning the same empty
    # tensor object more than once would declare output aliasing in FakeTensor,
    # which violates the functional custom-op contract.
    mile_weight = (
        lse.new_empty((valid_capacity,))
        if mile_enabled or patch_training_enabled
        else _empty(e, dtype=torch.float32)
    )
    patch_target_weight = (
        lse.new_empty((valid_capacity,))
        if patch_training_enabled
        else _empty(e, dtype=torch.float32)
    )
    mu = (
        e.new_empty((c.size(1),), dtype=torch.float32)
        if mu_loss_enabled
        else _empty(e, dtype=torch.float32)
    )
    mu_vocab_size = (
        e.new_empty((), dtype=torch.float32)
        if mu_loss_enabled
        else _empty(e, dtype=torch.float32)
    )
    metrics = (
        e.new_empty((3,), dtype=torch.float32)
        if return_loss_metrics
        else _empty(e, dtype=torch.float32)
    )
    return (
        e.new_empty((), dtype=torch.float32),
        metrics,
        lse,
        valids,
        mile_weight,
        patch_target_weight,
        mu,
        mu_vocab_size,
        e.new_empty(
            (),
            dtype=torch.bfloat16 if compute_dtype_is_bf16 else torch.float16,
        ),
    )


def _setup_context(ctx, inputs, output) -> None:
    (
        e,
        c,
        targets,
        bias,
        e_requires_grad,
        c_requires_grad,
        bias_requires_grad,
        ignore_index,
        softcap,
        shift,
        return_loss_metrics,
        filter_eps,
        accum_e_fp32,
        accum_c_fp32,
        filter_e_grad,
        filter_c_grad,
        auto_mixed_grad_accum,
        mile_enabled,
        mile_gamma,
        mu_loss_enabled,
        mu_loss_lambda,
        patch_training_enabled,
        _compute_dtype_is_bf16,
        forward_used_autocast,
    ) = inputs
    (
        _loss,
        metrics,
        lse,
        valids,
        mile_weight,
        patch_target_weight,
        mu,
        mu_vocab_size,
        witness,
    ) = output
    tensors = [e, c, targets]
    if bias is not None:
        tensors.append(bias)
    tensors.extend([lse, valids, mile_weight, patch_target_weight, mu, mu_vocab_size, witness])
    ctx.save_for_backward(*tensors)
    ctx.has_bias = bias is not None
    ctx.e_requires_grad = e_requires_grad
    ctx.c_requires_grad = c_requires_grad
    ctx.bias_requires_grad = bias_requires_grad
    ctx.ignore_index = ignore_index
    ctx.softcap = softcap
    ctx.shift = shift
    ctx.return_loss_metrics = return_loss_metrics
    ctx.filter_eps = filter_eps
    ctx.accum_e_fp32 = accum_e_fp32
    ctx.accum_c_fp32 = accum_c_fp32
    ctx.filter_e_grad = filter_e_grad
    ctx.filter_c_grad = filter_c_grad
    ctx.auto_mixed_grad_accum = auto_mixed_grad_accum
    ctx.mile_enabled = mile_enabled
    ctx.mile_gamma = mile_gamma
    ctx.mu_loss_enabled = mu_loss_enabled
    ctx.mu_loss_lambda = mu_loss_lambda
    ctx.patch_training_enabled = patch_training_enabled
    ctx.forward_used_autocast = forward_used_autocast
    ctx.mark_non_differentiable(
        metrics,
        lse,
        valids,
        mile_weight,
        patch_target_weight,
        mu,
        mu_vocab_size,
        witness,
    )


def _backward(ctx, *grads):
    grad_loss = grads[0]
    saved = list(ctx.saved_tensors)
    e, c, targets = saved[:3]
    offset = 3
    if ctx.has_bias:
        bias = saved[offset]
        offset += 1
    else:
        bias = None
    lse, valids, mile_weight, patch_target_weight, mu, mu_vocab_size, witness = saved[offset:]
    de, dc, dbias = _cce_backward_op(
        grad_loss,
        e,
        c,
        targets,
        bias,
        lse,
        valids,
        mile_weight,
        patch_target_weight,
        mu,
        mu_vocab_size,
        witness,
        ctx.forward_used_autocast,
        ctx.e_requires_grad,
        ctx.c_requires_grad,
        ctx.bias_requires_grad,
        ctx.ignore_index,
        ctx.softcap,
        ctx.shift,
        ctx.return_loss_metrics,
        ctx.filter_eps,
        ctx.accum_e_fp32,
        ctx.accum_c_fp32,
        ctx.filter_e_grad,
        ctx.filter_c_grad,
        ctx.auto_mixed_grad_accum,
        ctx.mile_enabled,
        ctx.mile_gamma,
        ctx.mu_loss_enabled,
        ctx.mu_loss_lambda,
        ctx.patch_training_enabled,
    )
    return (
        de,
        dc,
        None,
        dbias if ctx.has_bias else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(_cce_forward_op, _backward, setup_context=_setup_context)


def compiler_cce_linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    ignore_index: int,
    softcap: float | None,
    shift: int,
    return_loss_metrics: bool,
    filter_eps: float | None,
    accum_e_fp32: bool,
    accum_c_fp32: bool,
    filter_e_grad: bool,
    filter_c_grad: bool,
    auto_mixed_grad_accum: bool,
    mile_enabled: bool,
    mile_gamma: float,
    mu_loss_enabled: bool,
    mu_loss_lambda: float,
    patch_training_enabled: bool,
    compute_dtype_is_bf16: bool,
    forward_used_autocast: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the supported training subset as one compiler-visible custom op."""
    outputs = _cce_forward_op(
        e,
        c,
        targets,
        bias,
        e.requires_grad,
        c.requires_grad,
        bias.requires_grad if bias is not None else False,
        ignore_index,
        softcap,
        shift,
        return_loss_metrics,
        filter_eps,
        accum_e_fp32,
        accum_c_fp32,
        filter_e_grad,
        filter_c_grad,
        auto_mixed_grad_accum,
        mile_enabled,
        mile_gamma,
        mu_loss_enabled,
        mu_loss_lambda,
        patch_training_enabled,
        compute_dtype_is_bf16,
        forward_used_autocast,
    )
    return outputs[0], outputs[1] if return_loss_metrics else None
