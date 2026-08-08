"""LeviathanFunction: torch.autograd.Function integrado para la capa LEV.

Puerta de entrada ÚNICA integrable (contrato): ejecuta el forward del paquete
de kernels (`forward.forward_impl.leviathan_forward`) si está disponible y
cae al forward de referencia CPU (`backward.backward_impl.leviathan_forward_ref`)
en caso contrario. El backward delega en `leviathan_backward`.

Uso:
    embeds = LeviathanFunction.apply(
        ids, codebooks, head_proj_weight, head_norm_weight, head_norm_bias,
        head_spline_delta, head_out_weight, cfg,
    )

Contrato de intermedios (ver README.md de backward/):
    El forward se invoca con save_intermediates=True. El dict `saved` que
    devuelva se consume en el backward:
      - Modo lean (recomendado): z, coords, x_hat_por_head, mean_por_head,
        rsqrt_por_head  (~300 MB a N=65536).
      - Modo fat (debug/N pequeño): además B_por_head, phi_por_head,
        modes_por_head (OOM a N>=16384, NO usar en entrenamiento).
      - saved=None: el backward re-ejecuta el forward de referencia.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

try:  # import relativo (paquete backward/) o absoluto (layout plano del contrato)
    from .backward_impl import leviathan_backward, leviathan_forward_ref
except ImportError:  # pragma: no cover
    from backward_impl import leviathan_backward, leviathan_forward_ref

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_forward_impl():
    """Resolve the kernel forward across layouts (package or flat).

    Priority: the package's OWN forward_impl (relative import — this file's
    sibling, always current) -> top-level forward_impl -> forward.forward_impl
    (original subpackage layout) -> CPU reference.  The legacy layouts can
    hold a STALE copy (a sync bug caused the autotune wrapper to run from
    forward/forward_impl.py while the package had the raw launch — fixed
    by preferring the package module, 2026-08-07).
    """
    if _BASE not in sys.path:
        sys.path.insert(0, _BASE)
    import importlib

    try:
        from .forward_impl import leviathan_forward  # type: ignore

        return leviathan_forward
    except ImportError:
        pass
    for mod_name in ("forward_impl", "forward.forward_impl"):
        try:
            mod = importlib.import_module(mod_name)
            return mod.leviathan_forward
        except Exception:
            continue
    return leviathan_forward_ref


def _load_triton_backward():
    """backward_kernels.leviathan_backward_triton si existe; si no, None."""
    if _BASE not in sys.path:
        sys.path.insert(0, _BASE)
    try:
        from .backward_kernels import leviathan_backward_triton  # type: ignore
    except ImportError:
        try:
            from backward_kernels import leviathan_backward_triton  # type: ignore
        except Exception:
            return None
    return leviathan_backward_triton


_leviathan_backward_triton = _load_triton_backward()


def leviathan_backward_triton_or_torch(grad_out, params, cfg, saved, ids,
                                       chunk):
    """Triton backward when supported, else the verified torch fallback."""
    if _leviathan_backward_triton is not None:
        try:
            grads = _leviathan_backward_triton(grad_out, params, cfg, saved,
                                               ids)
            if grads is not None:
                return grads
        except (RuntimeError, TypeError, ValueError):
            pass
    return leviathan_backward(
        grad_out,
        params,
        cfg,
        saved=saved,
        ids=ids,
        chunk=chunk,
    )


_leviathan_forward = _load_forward_impl()
#: True si el backward está conectado al paquete de kernels forward (no al fallback).
_HAS_KERNEL_FORWARD: bool = _leviathan_forward is not leviathan_forward_ref

_PARAM_KEYS = (
    "codebooks",
    "head_proj_weight",
    "head_norm_weight",
    "head_norm_bias",
    "head_spline_delta",
    "head_out_weight",
)


class LeviathanFunction(torch.autograd.Function):
    """Forward/backward LEV con autograd. Entrada discreta (ids) sin gradiente."""

    @staticmethod
    def forward(
        ctx,
        ids: torch.Tensor,
        codebooks: torch.Tensor,
        head_proj_weight: torch.Tensor,
        head_norm_weight: torch.Tensor,
        head_norm_bias: torch.Tensor,
        head_spline_delta: torch.Tensor,
        head_out_weight: torch.Tensor,
        cfg: Any,
    ) -> torch.Tensor:
        params: Dict[str, torch.Tensor] = {
            "codebooks": codebooks,
            "head_proj_weight": head_proj_weight,
            "head_norm_weight": head_norm_weight,
            "head_norm_bias": head_norm_bias,
            "head_spline_delta": head_spline_delta,
            "head_out_weight": head_out_weight,
        }
        try:
            embeds, saved = _leviathan_forward(
                ids, params, cfg, save_intermediates=True)
        except (RuntimeError, TypeError, ValueError):
            # unsupported config / wrong dtype / no CUDA -> reference forward
            # (never wrong results: fallback is the oracle)
            embeds, saved = leviathan_forward_ref(
                ids, params, cfg, save_intermediates=True)

        ctx.cfg = cfg
        ctx.saved_intermediates = saved
        ctx.save_for_backward(
            ids,
            codebooks,
            head_proj_weight,
            head_norm_weight,
            head_norm_bias,
            head_spline_delta,
            head_out_weight,
        )
        return embeds

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        (
            ids,
            codebooks,
            head_proj_weight,
            head_norm_weight,
            head_norm_bias,
            head_spline_delta,
            head_out_weight,
        ) = ctx.saved_tensors
        params: Dict[str, torch.Tensor] = {
            "codebooks": codebooks,
            "head_proj_weight": head_proj_weight,
            "head_norm_weight": head_norm_weight,
            "head_norm_bias": head_norm_bias,
            "head_spline_delta": head_spline_delta,
            "head_out_weight": head_out_weight,
        }
        # chunk the backward over tokens so the torch fallback never OOMs at
        # large N (B/phi are recomputed per chunk; 8192 keeps the working set
        # ~1-2 GB).  The Triton backward will not need this.
        chunk = getattr(ctx.cfg, "backward_chunk", None) or 8192
        grads = leviathan_backward_triton_or_torch(
            grad_out, params, ctx.cfg, ctx.saved_intermediates, ids, chunk)
        return (
            None,  # ids: entrada discreta, sin gradiente
            grads["codebooks"],
            grads["head_proj_weight"],
            grads["head_norm_weight"],
            grads["head_norm_bias"],
            grads["head_spline_delta"],
            grads["head_out_weight"],
            None,  # cfg
        )


def leviathan_apply(
    ids: torch.Tensor,
    params: Dict[str, torch.Tensor],
    cfg: Any,
) -> torch.Tensor:
    """Conveniencia: apply() con dict de params (mismas claves que el módulo)."""
    return LeviathanFunction.apply(ids, *[params[k] for k in _PARAM_KEYS], cfg)
