"""Device-side finalization of accumulated CCE/MiLe/Top-k telemetry.

Call after summing scalar sufficient statistics over the reporting interval
and, when applicable, all distributed workers. Averaging per-batch perplexities
does not produce token-weighted perplexity. This API never evaluates logits.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = tl = libdevice = None


if triton is not None:

    @triton.jit
    def _finalize_lm_metrics_kernel(totals, output):
        i = tl.arange(0, 32)
        raw = tl.load(totals + i, i < 11, 0)
        tl.store(output + i, raw, i < 11)
        for phase in tl.static_range(2):
            count = tl.load(totals + phase * 3 + 2)
            ce = tl.load(totals + phase * 3) / tl.maximum(count, 1.0)
            mile = tl.load(totals + phase * 3 + 1) / tl.maximum(count, 1.0)
            start = 11 + phase * 4
            tl.store(output + start, tl.where(count > 0, ce, 0.0))
            tl.store(output + start + 1, tl.where(count > 0, libdevice.exp(ce), 0.0))
            tl.store(output + start + 2, tl.where(count > 0, mile, 0.0))
            tl.store(output + start + 3, tl.where(count > 0, libdevice.exp(mile), 0.0))
        k = tl.arange(0, 4)
        hits = tl.load(totals + 6 + k, k < 3, 0)
        count = tl.load(totals + 9)
        tl.store(output + 19 + k, hits / tl.maximum(count, 1.0), k < 3)


@torch.library.custom_op("cut_cross_entropy::finalize_lm_metrics", mutates_args=())
def finalize_lm_metrics(totals: torch.Tensor) -> torch.Tensor:
    """Finalize detached, reduced scalar totals without reading them on the host.

    Input is a contiguous FP64 vector of length 11, in this order:
    ``[token_CE_sum, token_MiLe_sum, token_N, patch_CE_sum, patch_MiLe_sum,
    patch_N, hits1, hits3, hits5, topk_N, nonfinite_topk_targets]``.
    CCE exposes per-microbatch mean CE and a MiLe reweighting delta; sum
    ``N * CE`` and ``N * (CE + delta)`` across microbatches, excluding mu-loss
    and auxiliary objectives. Top-k uses the existing CCE hit counts. MiLe
    changes loss weights, not rankings; do not create a post-MiLe Top-k.

    Return a new FP64 vector of length 22 on the same device:
    ``[the original 11 totals, token_CE, token_PPL, token_MiLe, token_exp_MiLe,
    patch_CE, patch_exp_CE, patch_MiLe, patch_exp_MiLe, acc1, acc3, acc5]``.
    Empty phases produce zero placeholders. The caller must omit their log
    fields and validate raw counts/nonfinite flags when publishing metrics.
    Patch exp-CE and exp-MiLe are not ordinary autoregressive perplexity.

    CUDA uses one warp in one Triton kernel; other devices use a Torch
    reference. Only 11 input and 22 output scalars use FP64, preserving large
    counts and the range of Python-float exponentials. There is no collective,
    host transfer, scalar read, vocabulary projection, or backward. This is an
    explicit logging-time API: ordinary CCE forwards incur no additional work.
    The shape is fixed for telemetry, independent of model/batch geometry.
    """
    if totals.shape != (11,) or totals.dtype != torch.float64 or not totals.is_contiguous():
        raise ValueError("LM finalization requires 11 contiguous FP64 scalar totals")
    if totals.requires_grad:
        raise ValueError("LM finalization requires detached scalar totals")
    if totals.is_cuda and triton is not None:
        output = totals.new_empty(22)
        _finalize_lm_metrics_kernel[(1,)](totals, output, num_warps=1)
        return output
    phases = totals[:6].reshape(2, 3)
    means = phases[:, :2] / phases[:, 2:].clamp_min(1)
    metrics = torch.stack((means[:, 0], means[:, 0].exp(), means[:, 1], means[:, 1].exp()), -1)
    metrics = torch.where(phases[:, 2:] > 0, metrics, 0.0)
    return torch.cat((totals, metrics.flatten(), totals[6:9] / totals[9].clamp_min(1)))


@finalize_lm_metrics.register_fake
def _finalize_lm_metrics_fake(totals):
    return totals.new_empty(22)
