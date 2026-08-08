"""Standalone Leviathan LEV layer — reference implementation (oracle/fallback).

Extracted from the NeoLLM modeling file (LeviathanGenerator) so kernel
development does not depend on the full transformers stack.  This module
is the SEMANTIC REFERENCE: the exact math of the paper (arXiv:2601.22040,
Sec. 3.1).  Custom kernels must reproduce this transformation.

Math (per token i):
    b = ceil(V^(1/k))
    i -> (i_1, ..., i_k)                      base-b decomposition
    z(i) = sum_r C_r[i_r]                     codebook lookup + accumulation
    for each head l (h heads):
        z~_l = sigmoid(1/2 * LN(W_seed,l z))  in [0,1]^d_seed
        B[n,d,g] = quadratic B-spline basis of z~_l on kappa knots [0,1],
                   normalized across g
        phi[n,d,r] = sum_g B[n,d,g] * S_l[d,g,r]
        M_l[n,r]   = prod_d phi[n,d,r]        (sign-parity product)
    E_i = sum_l M_l[n,:] @ W_out,l            W_out,l in R^{r x D}

Only torch is required.  Everything is derived from LeviathanConfig at
runtime — no hardcoded architecture constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class LeviathanConfig:
    """All LEV hyperparameters.  Defaults follow the paper (Sec. 3.1/4)."""

    vocab_size: int = 200_376          # V  (o200k_base)
    hidden_size: int = 512             # D
    generator_d_seed: int = 128        # d_seed
    generator_num_modes: int = 8       # h heads
    generator_num_knots: int = 16      # kappa
    generator_spline_degree: int = 2   # quadratic (fixed by the kernel form)
    generator_k: int = 3               # k compositional coordinates
    generator_krank: int = 64          # r tensor-product rank
    initializer_range: float = 0.02
    dtype: torch.dtype = torch.bfloat16

    @property
    def b(self) -> int:
        """Base of the compositional decomposition: b = ceil(V^(1/k))."""
        return math.ceil(self.vocab_size ** (1.0 / self.generator_k))

    @property
    def representable_vocab(self) -> int:
        return self.b ** self.generator_k


class LeviathanGenerator(nn.Module):
    """LEV layer.  Identical math to modeling_neollm.LeviathanGenerator."""

    def __init__(self, config: LeviathanConfig):
        super().__init__()
        self.config = config
        self.d_seed = config.generator_d_seed
        self.num_modes = config.generator_num_modes
        self.num_knots = config.generator_num_knots
        self.spline_degree = config.generator_spline_degree
        self.k = config.generator_k
        self.krank = config.generator_krank
        self.hidden_size = config.hidden_size
        b = config.b
        self.b = b
        if b ** self.k < config.vocab_size:
            raise ValueError(
                f"base-b cannot represent vocab: b={b}, k={self.k}, "
                f"b^k={b ** self.k} < vocab_size={config.vocab_size}"
            )

        # Stage 1: shared codebooks C_1..C_k in R^{b x d_seed}
        self.codebooks = nn.Parameter(torch.empty(self.k, b, self.d_seed))

        # Fixed knot grid over [0, 1] (kappa points), not learned.
        self.register_buffer(
            "knot_grid", torch.linspace(0.0, 1.0, self.num_knots), persistent=False
        )

        # Stage 2: per-head seed projections W_seed,l in R^{d_seed x d_seed}
        self.head_proj_weight = nn.Parameter(
            torch.empty(self.num_modes, self.d_seed, self.d_seed)
        )
        self.head_norm_weight = nn.Parameter(torch.ones(self.num_modes, self.d_seed))
        self.head_norm_bias = nn.Parameter(torch.zeros(self.num_modes, self.d_seed))
        self.head_norm_eps = 1e-5

        # Stage 3: spline coefficients S_l in R^{d_seed x kappa x r};
        # effective coefficient is (1 + head_spline_delta).
        self.head_spline_delta = nn.Parameter(
            torch.empty(self.num_modes, self.d_seed, self.num_knots, self.krank)
        )

        # Stage 4: per-head output projections W_out,l in R^{r x D}
        self.head_out_weight = nn.Parameter(
            torch.empty(self.num_modes, self.krank, self.hidden_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ir = self.config.initializer_range
        nn.init.normal_(self.codebooks, mean=0.0, std=ir)
        nn.init.normal_(self.head_proj_weight, mean=0.0, std=ir)
        nn.init.ones_(self.head_norm_weight)
        nn.init.zeros_(self.head_norm_bias)
        nn.init.normal_(self.head_spline_delta, mean=0.0, std=0.1)
        out_std = ir / math.sqrt(self.num_modes)
        nn.init.normal_(self.head_out_weight, mean=0.0, std=out_std)

    # ------------------------------------------------------------------
    # Reference math (float32 internals where the current impl uses them)
    # ------------------------------------------------------------------
    def _base_k_decompose(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.long().clone()
        coords = torch.empty(*token_ids.shape, self.k, dtype=torch.long, device=token_ids.device)
        for r in range(self.k - 1, -1, -1):
            coords[..., r] = ids % self.b
            ids = ids // self.b
        return coords

    @staticmethod
    def _normalize_bspline_basis(B: torch.Tensor) -> torch.Tensor:
        denom = B.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return B / denom

    def _bspline_basis(self, x_flat: torch.Tensor) -> torch.Tensor:
        scale = float(self.num_knots - 1)
        x32 = x_flat.float()
        x_e = x32.unsqueeze(-1)
        grid = self.knot_grid.float().view(1, 1, -1)
        d = (x_e - grid).abs() * scale
        B = torch.where(
            d < 0.5,
            0.75 - d**2,
            torch.where(d < 1.5, 0.5 * (1.5 - d) ** 2, torch.zeros_like(d)),
        )
        return self._normalize_bspline_basis(B)

    def _tensor_product(self, B: torch.Tensor, spline_coeff: torch.Tensor) -> torch.Tensor:
        """Rank-r separable tensor product over d_seed, sign-parity (KHRONOS)."""
        phi = torch.einsum("ndg,dgk->ndk", B, spline_coeff)
        log_mag = torch.log(phi.abs() + 1e-9).sum(dim=1)
        num_neg = (phi < 0).to(torch.int32).sum(dim=1)
        prod_sign = 1.0 - 2.0 * (num_neg % 2).float()
        return prod_sign * torch.exp(log_mag)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """embeds [*ids.shape, hidden_size] — the semantic reference."""
        target_dtype = self.codebooks.dtype
        orig_shape = token_ids.shape
        N = token_ids.numel()

        coords = self._base_k_decompose(token_ids)
        coords_flat = coords.reshape(N, self.k)
        z = torch.zeros(N, self.d_seed, device=token_ids.device, dtype=target_dtype)
        for r in range(self.k):
            z = z + self.codebooks[r][coords_flat[:, r]]

        e = torch.zeros(N, self.hidden_size, device=token_ids.device, dtype=target_dtype)
        for m in range(self.num_modes):
            proj_w = self.head_proj_weight[m]
            zh = torch.nn.functional.linear(
                z.to(dtype=proj_w.dtype, device=proj_w.device), proj_w
            ).float()
            norm_w = self.head_norm_weight[m].float()
            norm_b = self.head_norm_bias[m].float()
            mean = zh.mean(dim=-1, keepdim=True)
            var = zh.var(dim=-1, keepdim=True, unbiased=False)
            zh = (zh - mean) / (var + self.head_norm_eps).sqrt()
            zh = zh * norm_w + norm_b
            zh = torch.sigmoid(zh / 2.0).clamp(0.0, 1.0)

            B = self._bspline_basis(zh)
            modes = self._tensor_product(B, 1.0 + self.head_spline_delta[m].float())
            e = e + modes.to(self.head_out_weight.dtype) @ self.head_out_weight[m]

        return e.reshape(*orig_shape, self.hidden_size)


def build_generator(
    vocab_size: int = 200_376,
    hidden_size: int = 512,
    d_seed: int = 128,
    num_modes: int = 8,
    num_knots: int = 16,
    k: int = 3,
    krank: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> LeviathanGenerator:
    """Deterministic construction with the reference initialization."""
    torch.manual_seed(seed)
    cfg = LeviathanConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        generator_d_seed=d_seed,
        generator_num_modes=num_modes,
        generator_num_knots=num_knots,
        generator_k=k,
        generator_krank=krank,
        dtype=dtype,
    )
    gen = LeviathanGenerator(cfg).to(device=device, dtype=dtype)
    return gen
