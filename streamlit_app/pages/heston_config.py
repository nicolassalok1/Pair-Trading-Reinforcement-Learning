"""Heston configuration page."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from streamlit_app.utils.config_loader import load_config, save_config  # noqa: E402


def _parse_float_list(raw: str) -> List[float]:
    """Parse comma separated float list."""
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def render() -> None:
    """Render Heston config form."""
    st.subheader("Heston configuration")
    with st.expander("Guide des champs (Heston)", expanded=False):
        st.markdown(
            "- **Enable Heston module** : active ou saute totalement le calcul de features Heston.\n"
            "- **CSV path** : fichier d'options vanille (S0, K, T, C_mkt) pour calibrer le modele.\n"
            "- **Rolling window size** : nombre de lignes considerees par fenetre pour les stats/calibration.\n"
            "- **Maturities / Strikes** : grilles d'echeances et strikes lues dans le CSV ou imposees pour la calibration.\n"
            "- **Normalization** : choix du pre-traitement des features (minmax, standard ou aucun).\n"
            "- **Pricer parameters** : hyperparametres du solveur (iters, lr, taux r/q, device CPU/CUDA).\n"
            "- **Output CSV path** : destination des features Heston exportees pour la suite."
        )
    cfg = load_config("heston")
    pricer_cfg = cfg.get("pricer", {}) or {}

    with st.form(key="heston_config_form"):
        enabled = st.checkbox("Enable Heston module", value=bool(cfg.get("enabled", True)))
        csv_path = st.text_input("CSV path", value=cfg.get("csv_path", "heston_dummy_calls.csv"))
        window_size = st.number_input("Rolling window size", min_value=1, value=int(cfg.get("window_size", 50)))

        maturities_raw = st.text_input(
            "Maturities (comma separated)", value=",".join(map(str, cfg.get("maturities", [0.25, 0.5, 1.0])))
        )
        strikes_raw = st.text_input(
            "Strikes (comma separated)", value=",".join(map(str, cfg.get("strikes", [0.8, 1.0, 1.2])))
        )
        norm_options = ["minmax", "standard", "none"]
        norm_value = cfg.get("normalization", "minmax")
        if norm_value not in norm_options:
            norm_value = "minmax"
        normalization = st.selectbox("Normalization", options=norm_options, index=norm_options.index(norm_value))

        st.markdown("**Pricer parameters**")
        max_iters = st.number_input("Max iterations", min_value=1, value=int(pricer_cfg.get("max_iters", 5)))
        lr = st.number_input("Learning rate", min_value=0.0001, value=float(pricer_cfg.get("lr", 0.05)), format="%.5f")
        r = st.number_input("Risk-free rate (r)", value=float(pricer_cfg.get("r", 0.02)), format="%.4f")
        q = st.number_input("Dividend yield (q)", value=float(pricer_cfg.get("q", 0.0)), format="%.4f")
        device = st.selectbox("Device", options=["cpu", "cuda"], index=0 if pricer_cfg.get("device", "cpu") == "cpu" else 1)

        output_csv = st.text_input("Output CSV path", value=cfg.get("output_csv", "STATICS/Heston/features.csv"))

        submitted = st.form_submit_button("Save Heston config")

    if submitted:
        new_cfg = {
            "enabled": enabled,
            "csv_path": csv_path,
            "window_size": int(window_size),
            "maturities": _parse_float_list(maturities_raw),
            "strikes": _parse_float_list(strikes_raw),
            "normalization": normalization,
            "pricer": {"max_iters": int(max_iters), "lr": float(lr), "r": float(r), "q": float(q), "device": device},
            "output_csv": output_csv,
        }
        path, persisted = save_config("heston", new_cfg)
        st.success(f"Heston config saved to {path}")
        st.json(persisted)


if __name__ == "__main__":
    render()
