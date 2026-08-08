"""Backward de la capa LEV (LeviathanGenerator) + forward CPU de referencia.

Este módulo es la implementación de REFERENCIA (CPU, torch puro) del backward
de la capa LEV operando sobre un dict de parámetros. Los kernels Triton de
producción deben reproducir exactamente estas fórmulas (ver README.md).

La forward de referencia (`leviathan_forward_ref`) es un espejo EXACTO de
`leviathan_core.LeviathanGenerator.forward` (mismas operaciones, mismo orden,
mismos epsilones) pero toma los parámetros de un dict en vez de un nn.Module.
Se usa para: (a) sanity-check del backward, (b) fallback cuando el paquete de
kernels forward (forward/forward_impl.py) todavía no existe o no soporta el
guardado de intermedios.

Matemática (paper arXiv:2601.22040 §3.1, ver leviathan_core.py):

  por token n:  coords[n,r] = dígitos base-b de ids[n];  z[n] = sum_r C[r, coords[n,r]]
  por head l:
    zh_l   = z @ W_l^T                        (W_l = head_proj_weight[l])
    mu_l, var_l = mean/var poblacional sobre d de zh_l
    rho_l  = 1/sqrt(var_l + eps_ln)
    xh_l   = (zh_l - mu_l) * rho_l            (LayerNorm sin affine)
    y_l    = xh_l * gamma_l + beta_l          (affine LN)
    t_l    = clamp(sigmoid(y_l / 2), 0, 1)
    w[n,d,g]  = B-spline cuadrática de t sobre kappa knots (grid linspace(0,1,kappa))
                dg = |t - g|*(kappa-1);  w = 0.75-dg^2 (dg<.5); 0.5*(1.5-dg)^2 (dg<1.5); 0
    D[n,d]    = sum_g w[n,d,g]                (>= 0.5 siempre; clamp_min(1e-12) inactivo)
    B[n,d,g]  = w[n,d,g] / D[n,d]             (normalización sobre g)
    S[d,g,r]  = 1 + delta_l[d,g,r]
    phi[n,d,r]  = sum_g B[n,d,g] * S[d,g,r]
    M[n,r]      = sign-parity product sobre d:
                  L = sum_d log(|phi| + 1e-9);  P = 1 - 2*((#d: phi<0) mod 2);  M = P*exp(L)
  E[n,:] = sum_l M_l[n,:] @ W_out_l

Gradientes (cadena completa, dado G = dE/dE_out):

  dE/dM_l        = G @ W_out_l^T                        [N, r]
  dE/dW_out_l    = M_l^T @ G                            [r, D]
  dE/dphi[n,d,r] = dE/dM[n,r] * M[n,r] * sign(phi)/(|phi| + 1e-9)
                   (derivada exacta de P*exp(sum_d log(|phi|+eps)): P constante en phi!=0)
  dE/dS_l[d,g,r] = sum_n B[n,d,g] * dE/dphi[n,d,r]      (= dE/ddelta_l)
  dE/dB[n,d,g]   = sum_r dE/dphi[n,d,r] * S[d,g,r]
  dE/dw[n,d,g]   = (D[n,d]*dE/dB[n,d,g] - w[n,d,g]*sum_g' dE/dB[n,d,g']) / D[n,d]^2
  dE/dt[n,d]     = sum_g dE/dw[n,d,g] * dw_dd(n,d,g) * sign(t[n,d]-g) * (kappa-1)
                   dw_dd = -2*dg (dg<.5); -(1.5-dg) (.5<=dg<1.5); 0 (else)
  dE/dy[n,d]     = dE/dt[n,d] * 0.5*t*(1-t) * mask(t in (0,1))     (sigmoid/2 + clamp)
  dE/dxh[n,d]    = dE/dy[n,d] * gamma_l[d]
  dE/dgamma_l[d] = sum_n dE/dy[n,d] * xh[n,d]
  dE/dbeta_l[d]  = sum_n dE/dy[n,d]
  dE/dzh[n,d]    = rho_l[n] * (dE/dxh - mean_d(dE/dxh) - xh * mean_d(dE/dxh * xh))
  dE/dW_l[j,i]   = sum_n dE/dzh[n,j] * z[n,i]
  dE/dz_l[n,i]   = sum_j dE/dzh[n,j] * W_l[j,i]
  dE/dz          = sum_l dE/dz_l
  dE/dC[r,i,:]   = sum_{n: coords[n,r]=i} dE/dz[n,:]     (scatter-add)

Estrategia de memoria (ver README.md): se guarda z + por head (xh, mu, rho)
en fp32 (~300 MB a N=65536) y se RECOMPUTAN B/phi/M en el backward con
chunking sobre N. Nunca se materializan B/phi completos en HBM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Epsilones idénticos a leviathan_core.py (NO cambiar: la derivada debe ser la
# derivada exacta de la implementación de referencia).
EPS_LOG = 1e-9   # dentro de log(|phi| + eps) del producto sign-parity
EPS_LN = 1e-5    # LayerNorm (LeviathanGenerator.head_norm_eps)

_PARAM_KEYS: Tuple[str, ...] = (
    "codebooks",
    "head_proj_weight",
    "head_norm_weight",
    "head_norm_bias",
    "head_spline_delta",
    "head_out_weight",
)

# Alias de claves aceptadas en `saved` (español e inglés) -> clave interna.
_SAVE_KEY_MAP: Dict[str, str] = {
    "x_hat_por_head": "x_hat",
    "mean_por_head": "mean",
    "rsqrt_por_head": "rsqrt",
    "B_por_head": "B",
    "phi_por_head": "phi",
    "modes_por_head": "M",
    "x_hat_por_head": "x_hat",      # keys emitted by the Triton forward (lean)
    "mean_por_head": "mean",
    "rsqrt_por_head": "rsqrt",
    "x_hat_per_head": "x_hat",      # legacy aliases (reference forward)
    "mean_per_head": "mean",
    "rsqrt_per_head": "rsqrt",
    "B_per_head": "B",
    "phi_per_head": "phi",
    "modes_per_head": "M",
}


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """fp64 si ALGÚN tensor es fp64 (gradcheck), si no fp32 (acumulación)."""
    if any(t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


def _promote(x: torch.Tensor) -> torch.Tensor:
    """Promueve a fp32 salvo que ya sea fp64 (la referencia usa .float() siempre;
    aquí respetamos fp64 para poder hacer gradcheck exacto)."""
    if x.dtype in (torch.float32, torch.float64):
        return x
    return x.float()


def _to_wd(x: torch.Tensor, wd: torch.dtype) -> torch.Tensor:
    return x if x.dtype == wd else x.to(wd)


def base_k_decompose(ids: torch.Tensor, b: int, k: int) -> torch.Tensor:
    """Descomposición base-b de k dígitos (espejo de _base_k_decompose)."""
    ids = ids.long().clone()
    coords = torch.empty(*ids.shape, k, dtype=torch.long, device=ids.device)
    for r in range(k - 1, -1, -1):
        coords[..., r] = ids % b
        ids = ids // b
    return coords


def _bspline(
    t: torch.Tensor,
    cfg: Any,
    knot_grid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """B-spline cuadrática normalizada sobre kappa knots.

    Returns (B, w, Dsum, dw_dt) con
      B     [..., kappa]   base normalizada (sum_g B = 1)
      w     [..., kappa]   base sin normalizar
      Dsum  [..., 1]       sum_g w  (clamp_min 1e-12, inactivo en la práctica)
      dw_dt [..., kappa]   dw/d(t) analítico (para el backward)
    """
    kappa = int(cfg.generator_num_knots)
    scale = float(kappa - 1)
    wd = torch.float64 if t.dtype == torch.float64 else torch.float32
    t = _to_wd(t, wd)
    if knot_grid is None:
        grid = torch.linspace(0.0, 1.0, kappa, dtype=wd, device=t.device)
    else:
        if knot_grid.numel() != kappa:
            raise ValueError(f"knot_grid: expected {kappa} elements, got {knot_grid.numel()}")
        grid = knot_grid.to(device=t.device, dtype=wd).reshape(-1)
    grid = grid.view(1, 1, -1)
    d = (t.unsqueeze(-1) - grid).abs() * scale
    w = torch.where(
        d < 0.5,
        0.75 - d * d,
        torch.where(d < 1.5, 0.5 * (1.5 - d) ** 2, torch.zeros_like(d)),
    )
    Dsum = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    B = w / Dsum
    # derivadas de la B-spline respecto a t
    dw_dd = torch.where(
        d < 0.5,
        -2.0 * d,
        torch.where(d < 1.5, -(1.5 - d), torch.zeros_like(d)),
    )
    dw_dt = dw_dd * torch.sign(t.unsqueeze(-1) - grid) * scale
    return B, w, Dsum, dw_dt


def _sign_parity_product(phi: torch.Tensor) -> torch.Tensor:
    """Producto sobre d con sign-parity (espejo exacto de _tensor_product)."""
    log_mag = torch.log(phi.abs() + EPS_LOG).sum(dim=1)
    num_neg = (phi < 0).to(torch.int32).sum(dim=1)
    prod_sign = 1.0 - 2.0 * (num_neg % 2).float()
    return prod_sign * torch.exp(log_mag)


# ---------------------------------------------------------------------------
# forward de referencia (dict de params) — espejo de LeviathanGenerator.forward
# ---------------------------------------------------------------------------
def leviathan_forward_ref(
    ids: torch.Tensor,
    params: Dict[str, torch.Tensor],
    cfg: Any,
    save_intermediates: bool = False,
    save_full: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Forward LEV sobre dict de params -> (embeds [*ids.shape, D], saved).

    Espejo EXACTO de LeviathanGenerator.forward (mismas ops, mismo orden).
    `saved` (dict) contiene:
      z                  [N, d_seed]  dtype del modelo
      coords             [N, k]       int64 (dígitos base-b)
      x_hat_por_head     [h, N, d_seed]  salida LayerNorm (sin affine)
      mean_por_head      [h, N, 1]    media poblacional de zh
      rsqrt_por_head     [h, N, 1]    1/sqrt(var + eps)
    y con `save_full=True` además B/phi/modes por head (solo debug / N pequeño).
    """
    C = params["codebooks"]
    Wp = params["head_proj_weight"]
    gamma = params["head_norm_weight"]
    beta = params["head_norm_bias"]
    delta = params["head_spline_delta"]
    Wout = params["head_out_weight"]
    target_dtype = C.dtype
    k, b, d_seed = C.shape
    h = Wp.shape[0]
    krank, D = Wout.shape[1], Wout.shape[2]
    kappa = delta.shape[2]
    eps_ln = float(getattr(cfg, "head_norm_eps", EPS_LN))

    orig_shape = ids.shape
    ids_l = ids.long().reshape(-1)
    N = ids_l.numel()

    coords = base_k_decompose(ids_l, b, k)  # [N, k]

    # Stage 1: codebooks (acumulación en dtype del modelo, como la referencia)
    z = torch.zeros(N, d_seed, dtype=target_dtype, device=ids.device)
    for r in range(k):
        z = z + C[r][coords[:, r]]

    saved: Optional[Dict[str, Any]] = None
    knot_grid = params.get("knot_grid")
    if knot_grid is not None:
        knot_grid = knot_grid.to(device=ids.device).contiguous()

    if save_intermediates:
        saved = {
            "z": z,
            "coords": coords,
            "knot_grid": knot_grid,
            "x_hat_por_head": [],
            "mean_por_head": [],
            "rsqrt_por_head": [],
        }
        if save_full:
            saved["B_por_head"] = []
            saved["phi_por_head"] = []
            saved["modes_por_head"] = []

    e = torch.zeros(N, D, dtype=target_dtype, device=ids.device)
    for l in range(h):
        W_l, gamma_l, beta_l, delta_l, Wout_l = Wp[l], gamma[l], beta[l], delta[l], Wout[l]

        # Stage 2: proyección seed + LayerNorm + sigmoid(x/2)
        zh = F.linear(z.to(W_l.dtype), W_l)  # matmul en dtype de W (referencia)
        zh = _promote(zh)                    # .float() de la referencia (fp64 si procede)
        mean = zh.mean(dim=-1, keepdim=True)
        var = zh.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (zh - mean) / (var + eps_ln).sqrt()
        y = x_hat * _promote(gamma_l) + _promote(beta_l)
        t = torch.sigmoid(y / 2.0).clamp(0.0, 1.0)

        # Stage 3: spline -> phi -> M
        B, _, _, _ = _bspline(t, cfg, knot_grid)
        S = 1.0 + _promote(delta_l)
        phi = torch.einsum("ndg,dgr->ndr", B, S)
        M = _sign_parity_product(phi)

        # Stage 4: salida (matmul en dtype de W_out, como la referencia)
        e = e + M.to(Wout_l.dtype) @ Wout_l

        if saved is not None:
            wd_f = _work_dtype(zh)
            saved["x_hat_por_head"].append(x_hat.to(wd_f))
            saved["mean_por_head"].append(mean.to(wd_f))
            saved["rsqrt_por_head"].append((1.0 / (var + eps_ln).sqrt()).to(wd_f))
            if save_full:
                saved["B_por_head"].append(B)
                saved["phi_por_head"].append(phi)
                saved["modes_por_head"].append(M)

    e = e.reshape(*orig_shape, D)
    if saved is not None:
        for key in ("x_hat_por_head", "mean_por_head", "rsqrt_por_head",
                    "B_por_head", "phi_por_head", "modes_por_head"):
            if key in saved:
                saved[key] = torch.stack(saved[key])
    return e, saved


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
def _per_head_saves(saved: Dict[str, Any], h: int) -> List[Dict[str, torch.Tensor]]:
    """Extrae los saves por head (apilados [h, ...] o listas) a una lista de dicts."""
    shs: List[Dict[str, torch.Tensor]] = [{} for _ in range(h)]
    for src_key, dst_key in _SAVE_KEY_MAP.items():
        v = saved.get(src_key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for l in range(h):
                if l < len(v):
                    shs[l][dst_key] = v[l]
        else:
            for l in range(h):
                shs[l][dst_key] = v[l]
    return shs


def _get_xhat(
    z: torch.Tensor,
    W_l: torch.Tensor,
    cfg: Any,
    sh: Dict[str, torch.Tensor],
    wd: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x_hat, mean, rsqrt) guardados por el forward o recomputados desde z.

    Recomputar desde z es numéricamente consistente SOLO si el forward calculó
    zh con la misma precisión (misma matmul, mismo dtype); el modo recomendado
    de entrenamiento guarda (x_hat, mean, rsqrt) — ver README.md.
    """
    if sh.get("x_hat") is not None and sh.get("mean") is not None and sh.get("rsqrt") is not None:
        return (
            _to_wd(sh["x_hat"], wd),
            _to_wd(sh["mean"], wd),
            _to_wd(sh["rsqrt"], wd),
        )
    zh = F.linear(z.to(W_l.dtype), W_l)
    zh = _promote(zh)
    eps_ln = float(getattr(cfg, "head_norm_eps", EPS_LN))
    mean = zh.mean(dim=-1, keepdim=True)
    var = zh.var(dim=-1, keepdim=True, unbiased=False)
    s = (var + eps_ln).sqrt()
    x_hat = (zh - mean) / s
    rsqrt = 1.0 / s
    return x_hat, mean, rsqrt


def _head_backward(
    l: int,
    G: torch.Tensor,
    params: Dict[str, torch.Tensor],
    cfg: Any,
    z: torch.Tensor,
    sh: Dict[str, torch.Tensor],
    chunk: Optional[int],
    wd: torch.dtype,
    knot_grid: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradientes de la head l. Devuelve (dS, dW_l, dgamma_l, dbeta_l, dWout_l, dz_l).

    G [N, D] es el gradiente de salida. `z` [N, d] en wd. Chunking sobre N para
    acotar memoria (nunca se materializan B/phi completos).
    """
    W_l = params["head_proj_weight"][l]
    gamma_l = params["head_norm_weight"][l]
    beta_l = params["head_norm_bias"][l]
    delta_l = params["head_spline_delta"][l]
    Wout_l = params["head_out_weight"][l]
    N, d_seed = z.shape
    kappa = delta_l.shape[1]
    krank, D = Wout_l.shape

    W_l_wd, gamma_wd, beta_wd = _to_wd(W_l, wd), _to_wd(gamma_l, wd), _to_wd(beta_l, wd)
    S_l = 1.0 + _to_wd(delta_l, wd)          # [d, kappa, r]
    Wout_wd = _to_wd(Wout_l, wd)

    # dE/dM completo (barato: [N, r])
    dM = G @ Wout_wd.t()                      # [N, r]

    # (x_hat, mean, rsqrt): guardados o recomputados
    x_hat, _, rsqrt = _get_xhat(z, W_l, cfg, sh, wd)
    t = torch.sigmoid((x_hat * gamma_wd + beta_wd) / 2.0).clamp(0.0, 1.0)

    # B/phi guardados (modo fat/debug) o None -> recomputar; M siempre se
    # recompone en fp32 desde phi para no reutilizar el redondeo bf16 del kernel.
    B_saved = sh.get("B")
    phi_saved = sh.get("phi")
    # acumuladores
    dS = torch.zeros(d_seed, kappa, krank, dtype=wd, device=G.device)
    dW_l = torch.zeros(d_seed, d_seed, dtype=wd, device=G.device)
    dgamma_l = torch.zeros(d_seed, dtype=wd, device=G.device)
    dbeta_l = torch.zeros(d_seed, dtype=wd, device=G.device)
    dWout_l = torch.zeros(krank, D, dtype=wd, device=G.device)
    dz_l = torch.zeros(N, d_seed, dtype=wd, device=G.device)

    if chunk is None or chunk >= N:
        bounds = [(0, N)]
    else:
        bounds = [(s, min(s + chunk, N)) for s in range(0, N, chunk)]

    for s, e in bounds:
        z_c = z[s:e]
        t_c = t[s:e]
        x_hat_c = x_hat[s:e]
        rsqrt_c = rsqrt[s:e]

        # --- 1) B, phi, M (recomputados desde t salvo modo fat) ---
        if B_saved is not None:
            B_c = _to_wd(B_saved[s:e], wd)
            _, w_c, Dsum_c, dw_dt_c = _bspline(t_c, cfg, knot_grid)
        else:
            B_c, w_c, Dsum_c, dw_dt_c = _bspline(t_c, cfg, knot_grid)
        if phi_saved is not None:
            phi_c = _to_wd(phi_saved[s:e], wd)
        else:
            phi_c = torch.einsum("ndg,dgr->ndr", B_c, S_l)
        # Recompute M from the fp32 phi chain.  The Triton forward stores M in
        # bf16 to keep the saved state lean; reusing that rounded tensor here
        # amplifies error in dphi when sign-parity stress makes |M| large.
        # Fat/reference saves are also safe to recompute and stay consistent
        # with the exact phi used above.
        M_c = _sign_parity_product(phi_c)

        # --- 2) gradiente de M respecto a phi ---
        dphi_c = dM[s:e].unsqueeze(1) * M_c.unsqueeze(1) * torch.sign(phi_c) / (phi_c.abs() + EPS_LOG)  # [C,d,r]

        # --- 3) dS (== ddelta) y dB ---
        dS += torch.einsum("ndg,ndr->dgr", B_c, dphi_c)
        dB_c = torch.einsum("ndr,dgr->ndg", dphi_c, S_l)

        # --- 4) spline backward: dB -> dw -> dt ---
        # B_g = w_g / D  (D = sum_g' w_g')  =>  dE/dw_g = (D*dB_g - sum_g' dB_g'*w_g') / D^2
        wsum_c = (dB_c * w_c).sum(dim=-1, keepdim=True)          # [C,d,1] const. en g
        dw_c = (Dsum_c * dB_c - wsum_c) / (Dsum_c * Dsum_c)
        dt_c = torch.einsum("ndg,ndg->nd", dw_c, dw_dt_c)        # [C,d]

        # --- 5) sigmoid(x/2) + clamp backward ---
        mask = (t_c > 0.0) & (t_c < 1.0)
        dy_c = dt_c * (0.5 * t_c * (1.0 - t_c)) * mask           # dE/dy

        # --- 6) LayerNorm backward ---
        dx_hat_c = dy_c * gamma_wd                                # dE/dx_hat
        dgamma_l += (dy_c * x_hat_c).sum(dim=0)
        dbeta_l += dy_c.sum(dim=0)
        m1 = dx_hat_c.mean(dim=-1, keepdim=True)
        m2 = (dx_hat_c * x_hat_c).mean(dim=-1, keepdim=True)
        dzh_c = rsqrt_c * (dx_hat_c - m1 - x_hat_c * m2)          # dE/dzh

        # --- 7) z-side: dW_l y dz_l ---
        dW_l += torch.einsum("nj,ni->ji", dzh_c, z_c)
        dz_l[s:e] = torch.einsum("nj,ji->ni", dzh_c, W_l_wd)

        # --- 8) dW_out (necesita M del chunk) ---
        dWout_l += torch.einsum("nr,nd->rd", M_c, G[s:e])

    return dS, dW_l, dgamma_l, dbeta_l, dWout_l, dz_l


def leviathan_backward(
    grad_out: torch.Tensor,
    params: Dict[str, torch.Tensor],
    cfg: Any,
    saved: Optional[Dict[str, Any]] = None,
    ids: Optional[torch.Tensor] = None,
    chunk: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Gradientes de todos los parámetros LEV respecto a grad_out [..., D].

    Args:
        grad_out: gradiente de la pérdida respecto a la salida [*ids.shape, D].
        params:   dict con las mismas claves que los parámetros del módulo.
        cfg:      LeviathanConfig.
        saved:    intermedios del forward (recomendado) o None para recomputar:
                  - None -> re-ejecuta leviathan_forward_ref(save_intermediates=True)
                    (requiere `ids`)
                  - dict con 'z' (+ opcional 'coords') -> recomputa el resto
                  - dict con 'x_hat_por_head'/'mean_por_head'/'rsqrt_por_head'
                    (modo lean recomendado, ver README) o
                    'B_por_head'/'phi_por_head'/'modes_por_head' (modo fat/debug)
        ids:      necesario solo si saved=None o si saved no trae 'coords'.
        chunk:    nº de tokens por bloque para acotar memoria (None = todo N).

    Returns:
        dict con las mismas claves que params; gradientes del mismo shape,
        en el dtype de cada parámetro (acumulación interna fp32/fp64).
    """
    for key in _PARAM_KEYS:
        if key not in params:
            raise KeyError(f"leviathan_backward: falta '{key}' en params")

    C = params["codebooks"]
    Wp = params["head_proj_weight"]
    gamma = params["head_norm_weight"]
    beta = params["head_norm_bias"]
    delta = params["head_spline_delta"]
    Wout = params["head_out_weight"]
    k, b, d_seed = C.shape
    h = Wp.shape[0]
    krank, D = Wout.shape[1], Wout.shape[2]

    wd = _work_dtype(grad_out, C, Wp, gamma, beta, delta, Wout)
    G = grad_out.reshape(-1, D)
    G = _to_wd(G, wd)
    N = G.shape[0]

    if saved is None:
        if ids is None:
            raise ValueError(
                "leviathan_backward: saved=None requiere `ids` para recomputar el forward"
            )
        _, saved = leviathan_forward_ref(ids, params, cfg, save_intermediates=True)
    if not isinstance(saved, dict):
        raise TypeError("leviathan_backward: `saved` debe ser un dict o None")

    z = _to_wd(saved["z"], wd)
    coords = saved.get("coords")
    if coords is None:
        if ids is None:
            raise ValueError(
                "leviathan_backward: `saved` no trae 'coords' y falta `ids` "
                "para derivar los gradientes de los codebooks"
            )
        coords = base_k_decompose(ids.long().reshape(-1), b, k)

    # Normalize the forward-kernel saved layout: the Triton forward stores
    # modes as a single [N, h, krank] tensor (N first), while the backward
    # expects head-first [h, N, r] (or per-head lists).  Detect by shape.
    modes = saved.get("modes_por_head")
    if modes is not None and not isinstance(modes, (list, tuple)):
        if modes.ndim == 3 and modes.shape[0] == N and modes.shape[1] == h:
            saved["modes_por_head"] = modes.permute(1, 0, 2).contiguous()

    shs = _per_head_saves(saved, h)

    knot_grid = saved.get("knot_grid")
    if knot_grid is None:
        knot_grid = params.get("knot_grid")
    if knot_grid is not None:
        knot_grid = knot_grid.to(device=G.device, dtype=wd).reshape(-1)
        if knot_grid.numel() != delta.shape[2]:
            raise ValueError(
                f"knot_grid: expected {delta.shape[2]} elements, got {knot_grid.numel()}"
            )

    dC = torch.zeros_like(C, dtype=wd)
    dWp = torch.zeros_like(Wp, dtype=wd)
    dgamma = torch.zeros_like(gamma, dtype=wd)
    dbeta = torch.zeros_like(beta, dtype=wd)
    ddelta = torch.zeros_like(delta, dtype=wd)
    dWout = torch.zeros_like(Wout, dtype=wd)
    dz = torch.zeros(N, d_seed, dtype=wd, device=G.device)

    for l in range(h):
        dS_l, dW_l, dg_l, db_l, dWo_l, dz_l = _head_backward(
            l, G, params, cfg, z, shs[l], chunk, wd, knot_grid
        )
        ddelta[l] = dS_l
        dWp[l] = dW_l
        dgamma[l] = dg_l
        dbeta[l] = db_l
        dWout[l] = dWo_l
        dz += dz_l

    # codebooks: scatter-add sobre los dígitos base-b
    coords = coords.to(G.device)
    for r in range(k):
        dC[r].index_add_(0, coords[:, r], dz)

    # gradientes en el dtype de cada parámetro (mismo shape)
    return {
        "codebooks": dC.to(C.dtype),
        "head_proj_weight": dWp.to(Wp.dtype),
        "head_norm_weight": dgamma.to(gamma.dtype),
        "head_norm_bias": dbeta.to(beta.dtype),
        "head_spline_delta": ddelta.to(delta.dtype),
        "head_out_weight": dWout.to(Wout.dtype),
    }
