"""NLP configuration page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from streamlit_app.utils.config_loader import load_config, save_config  # noqa: E402


def render() -> None:
    """Render NLP config form."""
    st.subheader("NLP configuration")
    with st.expander("Guide des champs (NLP)", expanded=False):
        st.markdown(
            "- **Enable NLP module** : active ou coupe toute la brique NLP.\n"
            "- **OpenAI API key / model** : identifiants pour generer du sentiment via GPT.\n"
            "- **Telegram bot token / channel ID** : recuperation des messages Telegram a scorer.\n"
            "- **Sentiment threshold** : met a zero les signaux trop faibles (|score| < seuil).\n"
            "- **Normalize sentiment features** : recentre et normalise les 4 features avant export.\n"
            "- **Mock GPT / Mock Telegram** : evite tout appel reseau et renvoie des valeurs factices.\n"
            "- **Output CSV path** : ou ecrire le fichier de features pour les etapes suivantes."
        )
    st.caption("Renseignez vos cles si vous voulez interroger Telegram + GPT en mode real-run. En dry-run, des valeurs factices sont generees.")
    cfg = load_config("nlp")

    with st.form(key="nlp_config_form"):
        enabled = st.checkbox("Enable NLP module", value=bool(cfg.get("enabled", True)))
        openai_api_key = st.text_input("OpenAI API key", value=cfg.get("openai_api_key", ""))
        openai_model = st.text_input("OpenAI model", value=cfg.get("openai_model", "gpt-4o"))
        telegram_api_token = st.text_input("Telegram bot token", value=cfg.get("telegram_api_token", ""))
        telegram_channel_id = st.text_input("Telegram channel ID", value=cfg.get("telegram_channel_id", ""))
        sentiment_threshold = st.slider(
            "Sentiment threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(cfg.get("sentiment_threshold", 0.25)),
            step=0.05,
        )
        normalize_features = st.checkbox("Normalize sentiment features", value=bool(cfg.get("normalize_features", True)))
        mock_gpt = st.checkbox("Mock GPT (no external call)", value=bool(cfg.get("mock_gpt", True)))
        mock_telegram = st.checkbox("Mock Telegram (no network)", value=bool(cfg.get("mock_telegram", True)))
        output_csv = st.text_input("Output CSV path", value=cfg.get("output_csv", "STATICS/NLP/sentiment_features.csv"))

        submitted = st.form_submit_button("Save NLP config")

    if submitted:
        new_cfg = {
            "enabled": enabled,
            "openai_api_key": openai_api_key,
            "openai_model": openai_model,
            "telegram_api_token": telegram_api_token,
            "telegram_channel_id": telegram_channel_id,
            "sentiment_threshold": float(sentiment_threshold),
            "normalize_features": normalize_features,
            "mock_gpt": mock_gpt,
            "mock_telegram": mock_telegram,
            "output_csv": output_csv,
        }
        path, persisted = save_config("nlp", new_cfg)
        st.success(f"NLP config saved to {path}")
        st.json(persisted)


if __name__ == "__main__":
    render()
