from __future__ import annotations

import torch


def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _exclusive_branch(
    branch: torch.Tensor,
    reference: torch.Tensor,
    alpha: torch.Tensor,
    reference_fp32: torch.Tensor,
    reference_norm_sq: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    branch_fp32 = branch.float()
    dot = (branch_fp32 * reference_fp32).sum(dim=-1, keepdim=True)
    projection = (dot / reference_norm_sq).to(branch.dtype)
    residual = branch - alpha.to(branch.dtype) * projection * reference
    return _normalize(residual, eps)


def polynorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    proj_eps: float = 1.0e-6,
    exclusive_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference PolyNorm with an operation order matching the CuTe path."""
    x_sq = x.pow(2)
    x_cu = x * x_sq

    x1 = x * x_sq.mean(-1, keepdim=True).add(eps).rsqrt()
    x2 = x_sq * (x_sq * x_sq).mean(-1, keepdim=True).add(eps).rsqrt()
    x3 = x_cu * (x_cu * x_cu).mean(-1, keepdim=True).add(eps).rsqrt()

    if exclusive_logits is not None:
        alpha2, alpha3 = torch.sigmoid(exclusive_logits).unbind()
        x1_fp32 = x1.float()
        reference_norm_sq = x1_fp32.pow(2).sum(-1, keepdim=True).clamp_min(proj_eps)
        x2 = _exclusive_branch(x2, x1, alpha2, x1_fp32, reference_norm_sq, eps)
        x3 = _exclusive_branch(x3, x1, alpha3, x1_fp32, reference_norm_sq, eps)

    return weight[0] * x3 + weight[1] * x2 + weight[2] * x1 + bias


__all__ = ["polynorm_reference"]
