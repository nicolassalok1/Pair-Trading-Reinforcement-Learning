#!/usr/bin/env python3
"""
Calibration Heston autonome.

Ce script reprend la logique de calibration utilisée dans l'app Streamlit :
- lecture d'une chaîne d'options call au format cache CBOE (colonnes S0, K, T, C_mkt, iv_market)
- filtrage des points valides (T > 0.1, prix positifs, IV renseignée)
- calibration des paramètres Heston via un Adam maison sur les paramètres non contraints
- sauvegarde/affichage des paramètres calibrés.

Usage minimal :
    python3 calibrate_heston.py
    python3 calibrate_heston.py --csv heston_dummy_calls.csv --iters 300 --lr 0.05 --output params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch

# Rendez les modules internes importables sans lancer toute l'app Streamlit.
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts" / "scriptsGPT" / "pricing_scripts"
HESTON_DIR = SCRIPTS_DIR / "Heston"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(HESTON_DIR))

from Heston.heston_torch import HestonParams, carr_madan_call_torch  # noqa: E402

torch.set_default_dtype(torch.float64)
MIN_IV_MATURITY = 0.1
DEFAULT_ITERS = 400
DEFAULT_LR = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prices_from_unconstrained(
    u: torch.Tensor,
    S0_t: torch.Tensor,
    K_t: torch.Tensor,
    T_t: torch.Tensor,
    r: float,
    q: float,
) -> torch.Tensor:
    """Calcule les prix Heston (Carr–Madan) pour un vecteur de strikes."""
    params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    prices: list[torch.Tensor] = []
    for S0_i, K_i, T_i in zip(S0_t, K_t, T_t):
        price_i = carr_madan_call_torch(S0_i, r, q, T_i, params, K_i)
        prices.append(price_i)
    return torch.stack(prices)


def _heston_nn_loss(
    u: torch.Tensor,
    S0_t: torch.Tensor,
    K_t: torch.Tensor,
    T_t: torch.Tensor,
    C_mkt_t: torch.Tensor,
    r: float,
    q: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perte quadratique moyenne entre prix modèle et prix marché."""
    model_prices = _prices_from_unconstrained(u, S0_t, K_t, T_t, r, q)
    diff = model_prices - C_mkt_t
    if weights is not None:
        return 0.5 * (weights * diff**2).mean()
    return 0.5 * (diff**2).mean()


def calibrate_heston(
    df: pd.DataFrame,
    *,
    r: float,
    q: float,
    max_iters: int = DEFAULT_ITERS,
    lr: float = DEFAULT_LR,
    device: torch.device | None = None,
    progress: bool = False,
) -> Tuple[HestonParams, list[float], pd.DataFrame]:
    """
    Calibre Heston sur une chaîne d'options call.

    Args:
        df: DataFrame contenant S0, K, T, C_mkt et idéalement iv_market.
        r: taux sans risque.
        q: dividende continu.
        max_iters: itérations de l'optimiseur Adam maison.
        lr: pas de l'optimiseur.
        device: torch.device cible.
        progress: affiche les pertes intermédiaires.
    Returns:
        params calibrés, historique des pertes, dataframe filtrée utilisée.
    """
    device = device or DEVICE
    df_clean = df.dropna(subset=["S0", "K", "T", "C_mkt"]).copy()
    df_clean = df_clean[(df_clean["T"] > MIN_IV_MATURITY) & (df_clean["C_mkt"] > 0.05)]
    if "iv_market" in df_clean.columns:
        df_clean = df_clean[df_clean["iv_market"] > 0]
    if df_clean.empty:
        raise ValueError("Pas de points exploitables pour la calibration.")

    S0_ref = float(df_clean["S0"].median())
    moneyness = df_clean["K"].values / S0_ref

    S0_t = torch.tensor(df_clean["S0"].values, dtype=torch.float64, device=device)
    K_t = torch.tensor(df_clean["K"].values, dtype=torch.float64, device=device)
    T_t = torch.tensor(df_clean["T"].values, dtype=torch.float64, device=device)
    C_mkt_t = torch.tensor(df_clean["C_mkt"].values, dtype=torch.float64, device=device)

    # Pondère plus fortement l'ATM pour stabiliser la calibration.
    weights_np = 1.0 / (np.abs(moneyness - 1.0) + 1e-3)
    weights_np = np.clip(weights_np / weights_np.mean(), 0.5, 5.0)
    weights_t = torch.tensor(weights_np, dtype=torch.float64, device=device)

    u = torch.zeros(5, dtype=torch.float64, device=device, requires_grad=True)
    m = torch.zeros_like(u)
    v = torch.zeros_like(u)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    loss_history: list[float] = []
    for iteration in range(max_iters):
        if u.grad is not None:
            u.grad.zero_()
        loss_val = _heston_nn_loss(u, S0_t, K_t, T_t, C_mkt_t, r, q, weights=weights_t)
        loss_val.backward()
        with torch.no_grad():
            grad = u.grad
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))
            u -= lr * m_hat / (torch.sqrt(v_hat) + eps)
        loss_item = float(loss_val.detach().cpu())
        loss_history.append(loss_item)
        if progress and (iteration == 0 or (iteration + 1) % max(1, max_iters // 10) == 0):
            params_dbg = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
            print(
                f"[{iteration + 1:4d}/{max_iters}] "
                f"loss={loss_item:.6f} | "
                f"kappa={float(params_dbg.kappa):.4f}, "
                f"theta={float(params_dbg.theta):.4f}, "
                f"sigma={float(params_dbg.sigma):.4f}, "
                f"rho={float(params_dbg.rho):.4f}, "
                f"v0={float(params_dbg.v0):.4f}"
            )

    params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    return params, loss_history, df_clean


def _params_to_dict(params: HestonParams) -> dict:
    """Convertit HestonParams torch -> dictionnaire float JSON-serialisable."""
    return {
        "kappa": float(params.kappa.detach().cpu()),
        "theta": float(params.theta.detach().cpu()),
        "sigma": float(params.sigma.detach().cpu()),
        "rho": float(params.rho.detach().cpu()),
        "v0": float(params.v0.detach().cpu()),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration Heston (Carr–Madan) sur un cache CSV.")
    default_csv = Path(__file__).resolve().parent / "heston_dummy_calls.csv"
    parser.add_argument("--csv", type=Path, default=default_csv, help="Chemin vers le CSV call (format cache_CSV).")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Nombre d'itérations de calibration.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate de l'optimiseur Adam.")
    parser.add_argument("--rate", type=float, default=0.02, help="Taux sans risque r.")
    parser.add_argument("--dividend", type=float, default=0.0, help="Dividende continu q.")
    parser.add_argument("--output", type=Path, help="Chemin JSON de sortie pour stocker les paramètres calibrés.")
    parser.add_argument("--progress", action="store_true", help="Affiche l'évolution de la perte pendant la calibration.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.csv.exists():
        print(f"[!] CSV introuvable : {args.csv}", file=sys.stderr)
        return 1

    df_calls = pd.read_csv(args.csv)
    try:
        params, losses, df_used = calibrate_heston(
            df_calls,
            r=float(args.rate),
            q=float(args.dividend),
            max_iters=int(args.iters),
            lr=float(args.lr),
            device=DEVICE,
            progress=bool(args.progress),
        )
    except ValueError as exc:
        print(f"[!] Calibration impossible : {exc}", file=sys.stderr)
        return 2

    params_dict = _params_to_dict(params)
    final_loss = losses[-1] if losses else float("nan")
    rmse = float(np.sqrt(2 * final_loss)) if losses else float("nan")

    print("\nCalibration terminée.")
    print(f"  Device         : {DEVICE}")
    print(f"  Points utilisés: {len(df_used)}")
    print(f"  Perte finale   : {final_loss:.6f} (RMSE ~ {rmse:.6f})")
    print(
        "  Paramètres     : "
        f"kappa={params_dict['kappa']:.4f}, "
        f"theta={params_dict['theta']:.4f}, "
        f"sigma={params_dict['sigma']:.4f}, "
        f"rho={params_dict['rho']:.4f}, "
        f"v0={params_dict['v0']:.4f}"
    )

    if args.output:
        payload = {
            "params": params_dict,
            "loss": final_loss,
            "rmse": rmse,
            "meta": {
                "csv": str(Path(args.csv).resolve()),
                "n_points": len(df_used),
                "r": float(args.rate),
                "q": float(args.dividend),
                "iterations": int(args.iters),
                "lr": float(args.lr),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nParamètres sauvegardés dans {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
