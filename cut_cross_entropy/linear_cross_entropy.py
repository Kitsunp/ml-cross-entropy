# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import platform
import warnings
from typing import TYPE_CHECKING, Literal, overload

import torch
import torch.nn as nn

from cut_cross_entropy.cce_utils import CCEPreset, CCEPresets, LinearCrossEntropyImpl
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import (
    CCE_OPTS_DOC,
    DTENSOR_NOTE,
    IMPL_DOC,
    LINEAR_CROSS_ENTROPY_DOC,
    add_doc_end,
    add_doc_start,
)
from cut_cross_entropy.torch_compile import torch_compile_linear_cross_entropy
from cut_cross_entropy.utils import (
    CCEWarning,
    _handle_eps,
    is_torch_greater_or_equal_2_5,
    maybe_type_as,
    to_full_tensor,
)
from cut_cross_entropy.vocab_parallel import VocabParallelOptions

warnings.filterwarnings("once", category=CCEWarning, module="cut_cross_entropy")

# Resolve package metadata once at import.  Calling importlib.metadata from a
# compiled training step forces a graph break even though the result is process
# invariant.
TORCH_GREATER_OR_EQUAL_2_5 = is_torch_greater_or_equal_2_5()

PLATFORM_SYSTEM = platform.system()

if TYPE_CHECKING or PLATFORM_SYSTEM != "Darwin":
    from cut_cross_entropy.cce import _validate_cce_inputs, cce_linear_cross_entropy
    from cut_cross_entropy.cce_compile import compiler_cce_linear_cross_entropy

    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.CCE
else:
    cce_linear_cross_entropy = None
    compiler_cce_linear_cross_entropy = None
    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.TORCH_COMPILE

if TYPE_CHECKING or TORCH_GREATER_OR_EQUAL_2_5:
    import torch.distributed.tensor


is_d_tensor_error_message = (
    "Received {name} as a torch.distributed.tensor.DTensor. This is not supported. "
)


@overload
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: Literal[False] = False,
    return_loss_metrics: Literal[False] = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
    mile_enabled: bool = False,
    mile_gamma: float = 1.0,
    mu_loss_enabled: bool = False,
    mu_loss_lambda: float = 1e-4,
    patch_training_enabled: bool = False,
) -> torch.Tensor: ...


@overload
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: Literal[True] = True,
    return_loss_metrics: Literal[False] = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
    mile_enabled: bool = False,
    mile_gamma: float = 1.0,
    mu_loss_enabled: bool = False,
    mu_loss_lambda: float = 1e-4,
    patch_training_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: Literal[False] = False,
    return_loss_metrics: Literal[True] = True,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
    mile_enabled: bool = False,
    mile_gamma: float = 1.0,
    mu_loss_enabled: bool = False,
    mu_loss_lambda: float = 1e-4,
    patch_training_enabled: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


@overload
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: Literal[True] = True,
    return_loss_metrics: Literal[True] = True,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
    mile_enabled: bool = False,
    mile_gamma: float = 1.0,
    mu_loss_enabled: bool = False,
    mu_loss_lambda: float = 1e-4,
    patch_training_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: ...


@overload
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: bool = False,
    return_loss_metrics: bool = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
    mile_enabled: bool = False,
    mile_gamma: float = 1.0,
    mu_loss_enabled: bool = False,
    mu_loss_lambda: float = 1e-4,
    patch_training_enabled: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, dict[str, torch.Tensor]]
    | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
): ...


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
@add_doc_start(*(doc_str + " Only valid for the cce implementation." for doc_str in CCE_OPTS_DOC))
@add_doc_start(IMPL_DOC)
@add_doc_end(DTENSOR_NOTE)
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: bool = False,
    return_loss_metrics: bool = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
    mile_enabled: bool = False,
    mile_gamma: float = 1.0,
    mu_loss_enabled: bool = False,
    mu_loss_lambda: float = 1e-4,
    patch_training_enabled: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, dict[str, torch.Tensor]]
    | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
):
    """
    :param vocab_parallel_options: Used to enable vocab parallelism."""

    if TORCH_GREATER_OR_EQUAL_2_5:
        maybe_tensor_inputs = dict(e=e, targets=targets)
        for k, v in maybe_tensor_inputs.items():
            if isinstance(v, torch.distributed.tensor.DTensor):
                raise ValueError(is_d_tensor_error_message.format(name=k))

        c = maybe_type_as(to_full_tensor(c), e)
        bias = maybe_type_as(to_full_tensor(bias), e)

    if isinstance(impl, LinearCrossEntropyImpl):
        impl = impl.name.lower()

    if isinstance(shift, int) and (shift < 0 or shift >= targets.size(-1)):
        raise ValueError(f"Shift must be in the range [0, {targets.size(-1)}). Got {shift}.")

    if vocab_parallel_options is not None:
        expected_v_dim_size = vocab_parallel_options.stop - vocab_parallel_options.start
        if c.size(0) != expected_v_dim_size:
            raise ValueError(f"Expected c.size(0) to be {expected_v_dim_size}, got {c.size(0)}.")

    if bias is not None and bias.size(0) != c.size(0):
        raise ValueError(
            f"Bias has a different number of elements than c. {bias.size(0)} vs. {c.size(0)}."
        )

    if patch_training_enabled:
        if int(shift) != 0:
            raise ValueError(
                "patch_training_enabled does not support shift; align patches upstream."
            )
        if softcap is not None:
            raise ValueError("patch_training_enabled does not currently support softcap.")
        if return_lse:
            raise ValueError("patch_training_enabled does not currently support return_lse.")
        if vocab_parallel_options is not None:
            raise ValueError("patch_training_enabled does not currently support vocab parallelism.")

    if impl in CCEPresets.names:
        if platform.system() == "Darwin":
            raise RuntimeError(
                "CCE does not support MacOS. Please use torch_compile when running on MacOS instead."
            )

        cce_opts = CCEPresets.build_for_impl(
            impl,
            CCEPreset(
                filter_eps=filter_eps,
                accum_e_fp32=accum_e_fp32,
                accum_c_fp32=accum_c_fp32,
                filter_e_grad=filter_e_grad,
                filter_c_grad=filter_c_grad,
            ),
        )

        # Eligibility must follow the dtype used by the CUDA kernels, not only
        # the input storage dtype.  RMSNorm can legitimately return FP32 while
        # CUDA autocast makes CCE compute in FP16/BF16; rejecting that case
        # exposes the data-dependent valid-index compaction and Triton launch
        # code to Dynamo, specializing the graph once per valid-token count.
        cuda_autocast_enabled = torch.is_autocast_enabled("cuda")
        compute_dtype = (
            torch.get_autocast_dtype("cuda")
            if cuda_autocast_enabled
            else e.dtype
        )
        use_compiler_boundary = (
            torch.compiler.is_compiling()
            and vocab_parallel_options is None
            and reduction == "mean"
            and not return_lse
            and (int(shift) > 0 or patch_training_enabled)
            and e.is_cuda
            and compute_dtype in (torch.float16, torch.bfloat16)
        )
        if use_compiler_boundary:
            assert compiler_cce_linear_cross_entropy is not None
            _validate_cce_inputs(
                e,
                c,
                targets,
                reduction,
                return_loss_metrics,
                mile_enabled,
                mile_gamma,
                mu_loss_enabled,
                mu_loss_lambda,
                patch_training_enabled,
            )
            resolved_filter_eps = _handle_eps(cce_opts["filter_eps"], compute_dtype)
            filter_e_grad = (
                cce_opts["filter_e_grad"] and resolved_filter_eps is not None
            )
            filter_c_grad = (
                cce_opts["filter_c_grad"] and resolved_filter_eps is not None
            )
            loss, loss_metrics = compiler_cce_linear_cross_entropy(
                e,
                c,
                targets,
                bias,
                ignore_index,
                softcap,
                int(shift),
                return_loss_metrics,
                resolved_filter_eps,
                cce_opts["accum_e_fp32"],
                cce_opts["accum_c_fp32"],
                filter_e_grad,
                filter_c_grad,
                impl == "cce_kahan_full_c",
                mile_enabled,
                mile_gamma,
                mu_loss_enabled,
                mu_loss_lambda,
                patch_training_enabled,
                compute_dtype == torch.bfloat16,
                cuda_autocast_enabled,
            )
            lse = None
        else:
            assert cce_linear_cross_entropy is not None
            loss, lse, loss_metrics = cce_linear_cross_entropy(
                e,
                c,
                targets,
                bias,
                ignore_index,
                softcap,
                reduction,
                shift,
                **cce_opts,
                vocab_parallel_options=vocab_parallel_options,
                return_lse=return_lse,
                return_loss_metrics=return_loss_metrics,
                _auto_mixed_grad_accum=impl == "cce_kahan_full_c",
                mile_enabled=mile_enabled,
                mile_gamma=mile_gamma,
                mu_loss_enabled=mu_loss_enabled,
                mu_loss_lambda=mu_loss_lambda,
                patch_training_enabled=patch_training_enabled,
            )
    elif impl == "torch_compile":
        if return_loss_metrics:
            raise ValueError("return_loss_metrics is only supported by CCE implementations.")
        if mile_enabled:
            raise ValueError("mile_enabled is only supported by CCE implementations.")
        if mu_loss_enabled:
            raise ValueError("mu_loss_enabled is only supported by CCE implementations.")
        if patch_training_enabled:
            raise ValueError("patch_training_enabled is only supported by CCE implementations.")
        loss, lse = torch_compile_linear_cross_entropy(
            e,
            c,
            targets,
            bias,
            ignore_index,
            softcap,
            reduction,
            shift,
            vocab_parallel_options=vocab_parallel_options,
            return_lse=return_lse,
        )
        loss_metrics = None
    else:
        raise NotImplementedError(f"{impl} is not implemented.")

    if return_loss_metrics:
        assert loss_metrics is not None
        metrics = {
            "ntp_ce_unweighted": loss_metrics[0],
            "mile_reweighting_delta": loss_metrics[1],
            "mu_loss": loss_metrics[2],
        }
        if return_lse:
            assert lse is not None
            return loss, lse, metrics
        return loss, metrics
    if return_lse:
        assert lse is not None
        return loss, lse
    else:
        return loss


class LinearCrossEntropy(nn.Module):
    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        softcap: float | None = None,
        reduction: str = "mean",
        shift: bool | int = 0,
        filter_eps: float | str | None = "auto",
        accum_e_fp32: bool = False,
        accum_c_fp32: bool = False,
        filter_e_grad: bool = True,
        filter_c_grad: bool = True,
        impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
        return_lse: bool = False,
        return_loss_metrics: bool = False,
        mile_enabled: bool = False,
        mile_gamma: float = 1.0,
        mu_loss_enabled: bool = False,
        mu_loss_lambda: float = 1e-4,
        patch_training_enabled: bool = False,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.softcap = softcap
        self.reduction = reduction
        self.filter_eps = filter_eps
        self.shift = shift

        self.accum_e_fp32 = accum_e_fp32
        self.accum_c_fp32 = accum_c_fp32

        self.filter_e_grad = filter_e_grad
        self.filter_c_grad = filter_c_grad

        self.impl = impl
        self.return_lse = return_lse
        self.return_loss_metrics = return_loss_metrics
        self.mile_enabled = mile_enabled
        self.mile_gamma = mile_gamma
        self.mu_loss_enabled = mu_loss_enabled
        self.mu_loss_lambda = mu_loss_lambda
        self.patch_training_enabled = patch_training_enabled

    def forward(
        self,
        e: torch.Tensor,
        c: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        return linear_cross_entropy(
            e,
            c,
            targets,
            bias=bias,
            ignore_index=self.ignore_index,
            softcap=self.softcap,
            reduction=self.reduction,
            shift=self.shift,
            filter_eps=self.filter_eps,
            accum_e_fp32=self.accum_e_fp32,
            accum_c_fp32=self.accum_c_fp32,
            filter_e_grad=self.filter_e_grad,
            filter_c_grad=self.filter_c_grad,
            impl=self.impl,
            return_lse=self.return_lse,
            return_loss_metrics=self.return_loss_metrics,
            mile_enabled=self.mile_enabled,
            mile_gamma=self.mile_gamma,
            mu_loss_enabled=self.mu_loss_enabled,
            mu_loss_lambda=self.mu_loss_lambda,
            patch_training_enabled=self.patch_training_enabled,
        )
