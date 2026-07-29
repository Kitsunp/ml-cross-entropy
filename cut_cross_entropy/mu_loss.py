# Copyright (C) 2026. All Rights Reserved.
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _output_embedding_sum_kernel(
    C,
    EmbeddingSum,
    V,
    D,
    stride_cv,
    stride_cd,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(axis=0)
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
    offs_v = tl.arange(0, BLOCK_V).to(tl.int64)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_v in range(0, tl.cdiv(V, BLOCK_V)):
        rows = start_v * BLOCK_V + offs_v
        values = tl.load(
            C + rows[:, None] * stride_cv + offs_d[None, :] * stride_cd,
            mask=(rows[:, None] < V) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(values, axis=0)

    tl.store(EmbeddingSum + offs_d, accumulator, mask=offs_d < D)


@triton.jit
def _mu_loss_finalize_kernel(
    EmbeddingSum,
    VocabSize,
    Mu,
    Loss,
    D,
    mu_loss_lambda,
    BLOCK_D: tl.constexpr,
):
    offs_d = tl.arange(0, BLOCK_D)
    embedding_sum = tl.load(EmbeddingSum + offs_d, mask=offs_d < D, other=0.0)
    vocab_size = tl.load(VocabSize)
    mu = embedding_sum / vocab_size
    tl.store(Mu + offs_d, mu, mask=offs_d < D)
    tl.store(Loss, mu_loss_lambda * tl.sum(mu * mu, axis=0))


@triton.jit
def _add_mu_loss_gradient_kernel(
    dC,
    Mu,
    VocabSize,
    dOut,
    V,
    D,
    stride_dcv,
    stride_dcd,
    mu_loss_lambda,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_v = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
    mask = (offs_v[:, None] < V) & (offs_d[None, :] < D)
    dc_ptrs = dC + offs_v[:, None] * stride_dcv + offs_d[None, :] * stride_dcd
    dc = tl.load(dc_ptrs, mask=mask, other=0.0).to(tl.float32)
    mu = tl.load(Mu + offs_d, mask=offs_d < D, other=0.0)
    vocab_size = tl.load(VocabSize)
    d_out = tl.load(dOut)
    gradient = d_out * (2.0 * mu_loss_lambda / vocab_size) * mu
    tl.store(dc_ptrs, dc + gradient[None, :], mask=mask)


def mu_loss_forward_kernel(
    c: torch.Tensor,
    mu_loss_lambda: float,
    pg: torch.distributed.ProcessGroup | None = None,
    vocab_parallel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(loss, mean_embedding, global_vocab_size)`` for output embeddings."""
    assert c.ndim == 2
    assert c.is_cuda
    vocab_size, embedding_dim = c.shape
    embedding_sum = torch.empty(embedding_dim, device=c.device, dtype=torch.float32)
    _output_embedding_sum_kernel[(triton.cdiv(embedding_dim, 32),)](
        c,
        embedding_sum,
        vocab_size,
        embedding_dim,
        c.stride(0),
        c.stride(1),
        BLOCK_V=128,
        BLOCK_D=32,
    )

    global_vocab_size = torch.tensor(float(vocab_size), device=c.device, dtype=torch.float32)
    if vocab_parallel:
        torch.distributed.all_reduce(embedding_sum, group=pg)
        torch.distributed.all_reduce(global_vocab_size, group=pg)

    mu = torch.empty_like(embedding_sum)
    loss = torch.empty((), device=c.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(embedding_dim)
    _mu_loss_finalize_kernel[(1,)](
        embedding_sum,
        global_vocab_size,
        mu,
        loss,
        embedding_dim,
        mu_loss_lambda,
        BLOCK_D=block_d,
    )
    return loss, mu, global_vocab_size


def add_mu_loss_gradient_kernel(
    dc: torch.Tensor,
    mu: torch.Tensor,
    global_vocab_size: torch.Tensor,
    d_out: torch.Tensor,
    mu_loss_lambda: float,
) -> None:
    """Add the direct mu-loss gradient to an existing classifier gradient in-place."""
    assert dc.ndim == 2
    assert mu.ndim == 1 and mu.numel() == dc.size(1)
    assert global_vocab_size.numel() == 1
    assert d_out.numel() == 1
    grid = (triton.cdiv(dc.size(0), 32), triton.cdiv(dc.size(1), 32))
    _add_mu_loss_gradient_kernel[grid](
        dc,
        mu,
        global_vocab_size,
        d_out,
        dc.size(0),
        dc.size(1),
        dc.stride(0),
        dc.stride(1),
        mu_loss_lambda,
        BLOCK_V=32,
        BLOCK_D=32,
    )
