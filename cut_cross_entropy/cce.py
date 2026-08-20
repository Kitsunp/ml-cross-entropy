# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.amp

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.cce_mile import cce_mile_forward_kernel
from cut_cross_entropy.cce_patch import patch_loss_forward
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import CCE_OPTS_DOC, LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.mu_loss import mu_loss_forward_kernel
from cut_cross_entropy.utils import (
    TensorInfo,
    _build_flat_valids,
    _handle_eps,
    handle_reduction_none,
)
from cut_cross_entropy.vocab_parallel.utils import (
    VocabParallelOptions,
    vp_reduce_correct_logit,
    vp_reduce_lse,
    vp_reduce_mean_logit,
)


@dataclass
class CCEParams:
    targets: torch.Tensor
    valids: torch.Tensor | None
    softcap: float | None
    reduction: str
    filter_eps: float | None
    shift: int
    batch_shape: torch.Size
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool
    vocab_parallel_options: VocabParallelOptions | None
    return_lse: bool
    return_loss_metrics: bool
    auto_mixed_grad_accum: bool
    mile_gamma: float | None
    mile_group_mask: torch.Tensor | None
    mu_loss_lambda: float | None
    patch_training_enabled: bool


@torch.compiler.assume_constant_result
def _cuda_is_bf16_supported() -> bool:
    """Evaluate the process/device capability without emitting an FX node."""
    return torch.cuda.is_bf16_supported()


def _validate_cce_inputs(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    reduction: str,
    return_loss_metrics: bool,
    mile_enabled: bool,
    mile_gamma: float,
    mu_loss_enabled: bool,
    mu_loss_lambda: float,
    patch_training_enabled: bool,
    mile_group_mask: torch.Tensor | None = None,
) -> None:
    """Keep eager and compiler-boundary input validation identical."""
    if patch_training_enabled:
        if targets.ndim < 2 or targets.size(-1) < 1:
            raise ValueError(
                "Patch targets must have shape (..., patch_size) with patch_size >= 1."
            )
        if e.size()[0:-1] != targets.size()[0:-1]:
            raise ValueError(
                "With patch_training_enabled=True, e.shape[:-1] must equal targets.shape[:-1]."
            )
        if reduction != "mean":
            raise ValueError("patch_training_enabled currently requires reduction='mean'.")
    else:
        assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)
    if mile_enabled and (not math.isfinite(mile_gamma) or mile_gamma < 0):
        raise ValueError(f"mile_gamma must be finite and non-negative, got {mile_gamma}.")
    if mile_group_mask is not None:
        if not mile_enabled:
            raise ValueError("mile_group_mask requires mile_enabled=True.")
        if patch_training_enabled:
            raise ValueError("mile_group_mask is not supported by patch training.")
        if mile_group_mask.shape != targets.shape:
            raise ValueError("mile_group_mask must match the targets shape.")
        if mile_group_mask.dtype != torch.bool:
            raise TypeError("mile_group_mask must be boolean.")
        if mile_group_mask.device != targets.device:
            raise ValueError("mile_group_mask and targets must share a device.")
    if mu_loss_enabled and (not math.isfinite(mu_loss_lambda) or mu_loss_lambda < 0):
        raise ValueError(f"mu_loss_lambda must be finite and non-negative, got {mu_loss_lambda}.")
    if mu_loss_enabled and reduction != "mean":
        raise ValueError("mu_loss_enabled requires reduction='mean'.")
    if return_loss_metrics and reduction != "mean":
        raise ValueError("return_loss_metrics requires reduction='mean'.")
    if not _cuda_is_bf16_supported():
        raise RuntimeError(
            "Cut Cross Entropy requires an ampere GPU or newer. "
            "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
        )


class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor | None,
        params: CCEParams,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        e_info = TensorInfo(e.dtype, e.requires_grad)
        c_info = TensorInfo(c.dtype, c.requires_grad)

        bias_info = None
        if bias is not None:
            bias_info = TensorInfo(bias.dtype, bias.requires_grad)

        if params.mu_loss_lambda is not None:
            vp_opts = params.vocab_parallel_options
            mu_loss, mu, mu_vocab_size = mu_loss_forward_kernel(
                c,
                params.mu_loss_lambda,
                pg=None if vp_opts is None else vp_opts.group,
                vocab_parallel=vp_opts is not None,
            )
        else:
            mu_loss = None
            mu = None
            mu_vocab_size = None

        if torch.is_autocast_enabled():
            e = e.to(dtype=torch.get_autocast_gpu_dtype())
            c = c.to(dtype=torch.get_autocast_gpu_dtype())

            if bias is not None:
                bias = bias.to(dtype=torch.get_autocast_gpu_dtype())

        targets = params.targets
        if (vp_opts := params.vocab_parallel_options) is not None:
            is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
            targets = torch.where(
                is_my_target,
                targets - vp_opts.start,
                ## NB
                # The backward kernel already uses
                # c.size(0) + 1 as the padding value to ensure that
                # (targets.size(0) % block_size) == 0, so for targets
                # that aren't in this VP rank's range, we can just consider
                # them as padded and all work work as expected.
                targets.new_full((), c.size(0) + 1),
            )

        ret = cce_lse_forward_kernel(
            e=e,
            c=c,
            bias=bias,
            valids=params.valids,
            softcap=params.softcap,
            return_logit_avg=False,
            shift=params.shift,
            targets=targets,
            return_mean_logit=params.mile_gamma is not None,
        )
        lse = ret.lse
        mean_logit = ret.mean_logit

        if params.patch_training_enabled:
            assert ret.neg_correct_logit is not None
            neg_correct_logit = ret.neg_correct_logit
            (
                objective_loss,
                patch_unweighted_ce,
                mile_weight,
                patch_target_weight,
            ) = patch_loss_forward(
                lse,
                mean_logit,
                neg_correct_logit,
                targets,
                c.size(0),
                params.mile_gamma,
                params.return_loss_metrics,
            )
            nll = None
            token_loss = None
            unweighted_nll_mean = None
        else:
            assert ret.neg_correct_logit is not None
            neg_correct_logit = ret.neg_correct_logit

            if params.vocab_parallel_options is not None:
                vp_lse = lse
                lse = vp_reduce_lse(lse, pg=params.vocab_parallel_options.group)

                if mean_logit is not None:
                    mean_logit = vp_reduce_mean_logit(
                        vp_lse,
                        mean_logit,
                        lse,
                        pg=params.vocab_parallel_options.group,
                    )

                neg_correct_logit = vp_reduce_correct_logit(
                    neg_correct_logit, pg=params.vocab_parallel_options.group, dtype=lse.dtype
                )

            nll = neg_correct_logit.add_(lse).clamp_min_(0.0)
            patch_unweighted_ce = None
            patch_target_weight = None
            if params.mile_gamma is not None:
                assert mean_logit is not None
                mile_weight, token_loss, unweighted_nll_mean = cce_mile_forward_kernel(
                    lse,
                    mean_logit,
                    nll,
                    params.mile_gamma,
                    return_unweighted_nll_mean=params.return_loss_metrics,
                    mean_reduction=params.reduction == "mean",
                    group_mask=params.mile_group_mask,
                    max_entropy=math.log(
                        c.size(0)
                        * (
                            1
                            if params.vocab_parallel_options is None
                            else torch.distributed.get_world_size(
                                params.vocab_parallel_options.group
                            )
                        )
                    ),
                )
            else:
                token_loss = nll
                mile_weight = None
                unweighted_nll_mean = None

        ctx.save_for_backward(
            e,
            c,
            bias,
            lse,
            params.targets,
            params.valids,
            mile_weight,
            patch_target_weight,
            mu,
            mu_vocab_size,
        )
        ctx.params = params
        ctx.e_info = e_info
        ctx.c_info = c_info
        ctx.bias_info = bias_info

        if not params.return_lse:
            ret_lse = None
        else:
            ret_lse = handle_reduction_none(params.batch_shape, params.valids, params.shift, lse)

        reduction = params.reduction
        if not params.patch_training_enabled:
            assert token_loss is not None
            if reduction == "mean":
                objective_loss = (
                    token_loss.sum() if params.mile_gamma is not None else token_loss.mean()
                )
            elif reduction == "sum":
                objective_loss = token_loss.sum()
            elif reduction == "none":
                objective_loss = handle_reduction_none(
                    params.batch_shape, params.valids, params.shift, token_loss
                )
            else:
                raise ValueError(f"Unknown reduction {reduction}")

        if params.return_loss_metrics:
            assert reduction == "mean"
            if patch_unweighted_ce is not None:
                unweighted_ce = patch_unweighted_ce
            elif unweighted_nll_mean is None:
                unweighted_ce = objective_loss
            else:
                unweighted_ce = unweighted_nll_mean
            mu_loss_metric = mu_loss if mu_loss is not None else objective_loss.new_zeros(())
            loss_metrics = torch.stack(
                (
                    unweighted_ce,
                    objective_loss - unweighted_ce,
                    mu_loss_metric,
                )
            ).detach()
            ctx.mark_non_differentiable(loss_metrics)
        else:
            loss_metrics = None

        loss = objective_loss
        if mu_loss is not None:
            loss = loss + mu_loss

        return loss, ret_lse, loss_metrics

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        grad_out: torch.Tensor,
        grad_lse_out: torch.Tensor | None,
        grad_loss_metrics_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        (
            e,
            c,
            bias,
            lse,
            targets,
            valids,
            mile_weight,
            patch_target_weight,
            mu,
            mu_vocab_size,
        ) = ctx.saved_tensors

        params = cast(CCEParams, ctx.params)
        reduction = params.reduction
        if reduction == "mean":
            grad_scale = 1 / max(lse.numel(), 1)
        elif reduction == "sum":
            grad_scale = 1.0
        elif reduction == "none":
            grad_scale = 1.0
            grad_out = grad_out.view(-1)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        if grad_lse_out is not None:
            grad_lse_out = grad_lse_out.view(-1)

        reduce_e_grad = False
        pg = None
        if (vp_opts := params.vocab_parallel_options) is not None:
            is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
            targets = torch.where(
                is_my_target,
                targets - vp_opts.start,
                ## NB
                # The backward kernel already uses
                # c.size(0) + 1 as the padding value to ensure that
                # (targets.size(0) % block_size) == 0, so for targets
                # that aren't in this VP rank's range, we can just consider
                # them as padded and all work work as expected.
                targets.new_full((), c.size(0) + 1),
            )

            reduce_e_grad = vp_opts.reduce_e_grad
            pg = vp_opts.group

        de, dc, dbias = cce_backward_kernel(
            do=grad_out,
            dlse=grad_lse_out,
            e=e,
            e_info=ctx.e_info,
            c=c,
            c_info=ctx.c_info,
            bias=bias,
            bias_info=ctx.bias_info,
            lse=lse,
            mile_weight=mile_weight,
            patch_target_weight=patch_target_weight,
            mu=mu,
            mu_vocab_size=mu_vocab_size,
            mu_loss_lambda=params.mu_loss_lambda,
            mile_gamma=None if params.patch_training_enabled else params.mile_gamma,
            valids=valids,
            softcap=params.softcap,
            filter_eps=params.filter_eps,
            targets=targets,
            shift=params.shift,
            vocab_ordering=None,
            grad_scale=grad_scale,
            accum_e_fp32=params.accum_e_fp32,
            accum_c_fp32=params.accum_c_fp32,
            auto_mixed_grad_accum=params.auto_mixed_grad_accum,
            filter_e_grad=params.filter_e_grad,
            filter_c_grad=params.filter_c_grad,
            reduce_e_grad=reduce_e_grad,
            pg=pg,
        )

        return de, dc, dbias, None


def linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    params: CCEParams,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    loss, lse, loss_metrics = cast(
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
        LinearCrossEntropyFunction.apply(e, c, bias, params),
    )

    if params.shift != 0 and params.reduction == "none":
        loss = loss[..., params.shift :]

    if params.return_lse and params.shift != 0:
        assert lse is not None
        lse = lse[..., params.shift :]

    return loss, lse, loss_metrics


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
@add_doc_start(*(doc_str + "\n" for doc_str in CCE_OPTS_DOC))
def cce_linear_cross_entropy(
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
    _auto_mixed_grad_accum: bool = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    vocab_parallel_options: VocabParallelOptions | None = None,
    mile_enabled: bool = False,
    mile_gamma: float = 1.0,
    mile_group_mask: torch.Tensor | None = None,
    mu_loss_enabled: bool = False,
    mu_loss_lambda: float = 1e-4,
    patch_training_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
        mile_group_mask,
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

    e = e.contiguous()
    targets = targets.contiguous()
    shift = int(shift)
    if patch_training_enabled:
        batch_shape = targets.size()[:-1]
        e = e.flatten(0, -2)
        targets = targets.reshape(-1, targets.size(-1))
        # Normalize an arbitrary ignore_index once so every indexed kernel can
        # use the same one-sided, out-of-range-safe target predicate.
        targets = torch.where(targets == ignore_index, -1, targets)
        valids = None
    else:
        batch_shape = targets.size()
        global_vocab_size = c.size(0) * (
            1
            if vocab_parallel_options is None
            else torch.distributed.get_world_size(vocab_parallel_options.group)
        )
        valids = _build_flat_valids(targets, ignore_index, shift, global_vocab_size)
        if mile_group_mask is not None:
            group_mask = mile_group_mask.contiguous().flatten()
            if valids is not None:
                group_mask = group_mask.index_select(0, valids.to(torch.long))
            mile_group_mask = group_mask
        e = e.flatten(0, -2)
        targets = targets.flatten()
        if (targets.data_ptr() % 16) != 0:
            targets = torch.nn.functional.pad(targets, (0, 1))[:-1]
        assert (targets.data_ptr() % 16) == 0
    cce_params = CCEParams(
        targets,
        valids,
        softcap,
        reduction,
        _handle_eps(
            filter_eps, torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else e.dtype
        ),
        shift,
        batch_shape,
        accum_e_fp32,
        accum_c_fp32,
        filter_e_grad=filter_e_grad and filter_eps is not None,
        filter_c_grad=filter_c_grad and filter_eps is not None,
        vocab_parallel_options=vocab_parallel_options,
        return_lse=return_lse,
        return_loss_metrics=return_loss_metrics,
        auto_mixed_grad_accum=_auto_mixed_grad_accum,
        mile_gamma=mile_gamma if mile_enabled else None,
        mile_group_mask=mile_group_mask,
        mu_loss_lambda=mu_loss_lambda if mu_loss_enabled else None,
        patch_training_enabled=patch_training_enabled,
    )

    return linear_cross_entropy_apply(e, c, bias, cce_params)
