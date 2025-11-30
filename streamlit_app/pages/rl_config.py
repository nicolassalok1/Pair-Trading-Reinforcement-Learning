"""RL configuration page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from streamlit_app.utils.config_loader import load_config, save_config  # noqa: E402


def render() -> None:
    """Render RL config form."""
    st.subheader("RL configuration")
    with st.expander("Guide des champs (RL)", expanded=False):
        st.markdown(
            "- **Learning rate / Batch size** : pas de descente de gradient et taille des minibatchs.\n"
            "- **Gamma (discount)** : importance donnee aux recompenses futures.\n"
            "- **Number of episodes** : iterations d'entrainement du bandit contextuel.\n"
            "- **Exploration rate** : proportion d'actions prises au hasard (epsilon-greedy ou boltzmann).\n"
            "- **Policy** : strategie d'exploration (epsilon_greedy ou boltzmann).\n"
            "- **Device** : execution CPU ou GPU pour le calcul.\n"
            "- **Use Heston / Use NLP** : injecte ou non les features des autres modules.\n"
            "- **Dry-run** : saute l'entrainement lourd et renvoie un resultat de test."
        )
    st.caption("Commencez avec peu d'episodes et un exploration_rate > 0 pour tester. Activez CPU si vous n'avez pas CUDA.")
    cfg = load_config("rl")

    with st.form(key="rl_config_form"):
        learning_rate = st.number_input(
            "Learning rate", min_value=1e-6, max_value=1.0, value=float(cfg.get("learning_rate", 0.001)), format="%.6f"
        )
        batch_size = st.number_input("Batch size", min_value=1, value=int(cfg.get("batch_size", 32)))
        gamma = st.slider("Gamma (discount)", min_value=0.0, max_value=1.0, value=float(cfg.get("gamma", 0.99)), step=0.01)
        episodes = st.number_input("Number of episodes", min_value=1, value=int(cfg.get("episodes", 3)))
        exploration_rate = st.slider(
            "Exploration rate", min_value=0.0, max_value=1.0, value=float(cfg.get("exploration_rate", 0.1)), step=0.01
        )
        policy = st.selectbox("Policy", options=["epsilon_greedy", "boltzmann"], index=0 if cfg.get("policy") == "epsilon_greedy" else 1)
        device = st.selectbox("Device", options=["cpu", "gpu"], index=0 if cfg.get("device", "cpu") == "cpu" else 1)
        use_heston = st.checkbox("Use Heston features", value=bool(cfg.get("use_heston", True)))
        use_nlp = st.checkbox("Use NLP features", value=bool(cfg.get("use_nlp", True)))
        dry_run = st.checkbox("Dry-run (skip heavy training)", value=bool(cfg.get("dry_run", False)))

        submitted = st.form_submit_button("Save RL config")

    if submitted:
        new_cfg = {
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "gamma": float(gamma),
            "episodes": int(episodes),
            "exploration_rate": float(exploration_rate),
            "policy": policy,
            "device": device,
            "use_heston": use_heston,
            "use_nlp": use_nlp,
            "dry_run": dry_run,
        }
        path, persisted = save_config("rl", new_cfg)
        st.success(f"RL config saved to {path}")
        st.json(persisted)


if __name__ == "__main__":
    render()
