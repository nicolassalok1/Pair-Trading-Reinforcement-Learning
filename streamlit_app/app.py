"""Main Streamlit app for Pair-Trading-Reinforcement-Learning."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
try:  # TensorFlow is optional for UI startup; RL stage will fail gracefully if missing.
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.get_logger().setLevel("ERROR")
    TF_OK = True
except ImportError as exc:  # pragma: no cover - import guard
    tf = None
    TF_OK = False
    TF_IMPORT_ERROR = exc

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from streamlit_app.pages import heston_config, logs as logs_page, nlp_config, rl_config  # noqa: E402
from streamlit_app.utils.config_loader import load_config, save_config  # noqa: E402
from streamlit_app.utils.log_handler import append_log  # noqa: E402

from MAIN import Basics, Reinforcement  # noqa: E402
from STRATEGY.Cointegration import EGCointegration  # noqa: E402
from UTIL import FileIO  # noqa: E402
from nlp_module import sentiment_processor  # noqa: E402
from heston_model.calibrate_heston import get_heston_features_from_csv  # noqa: E402

STYLE_PATH = ROOT / "utils" / "style.css"


def load_css() -> None:
    """Inject custom CSS."""
    if STYLE_PATH.exists():
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _generate_synthetic_pair(
    base_a: float = 30000.0,
    base_b: float = 1.3,
    vol_a: float = 0.6,
    vol_b: float = 0.25,
    corr: float = 0.7,
    n_obs: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a synthetic price pair to keep the demo runnable."""
    rng = np.random.default_rng(seed=123)
    cov = np.array([[1.0, corr], [corr, 1.0]])
    L = np.linalg.cholesky(cov)
    shocks = rng.standard_normal(size=(n_obs, 2)) @ L.T

    dt = 1 / 1440
    mu = 0.02
    ret_a = np.exp((mu - 0.5 * vol_a**2) * dt + vol_a * shocks[:, 0] * np.sqrt(dt))
    ret_b = np.exp((mu - 0.5 * vol_b**2) * dt + vol_b * shocks[:, 1] * np.sqrt(dt))
    prices_a = base_a * np.cumprod(ret_a)
    prices_b = base_b * np.cumprod(ret_b)

    dates = pd.date_range("2020-01-01", periods=n_obs, freq="T")
    df_a = pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "close": prices_a})
    df_b = pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "close": prices_b})
    return df_a, df_b


def run_nlp(cfg: Dict, mode: str) -> Dict:
    """Execute NLP stage; supports dry-run and mocks."""
    append_log("nlp", f"Starting NLP stage (mode={mode})")
    output_csv = Path(cfg.get("output_csv", "STATICS/NLP/sentiment_features.csv"))
    _ensure_parent(output_csv)

    if not cfg.get("enabled", True):
        append_log("nlp", "NLP module disabled; skipping.")
        return {"status": "skipped", "output_csv": str(output_csv)}

    try:
        if mode == "dry-run" or cfg.get("mock_gpt") or cfg.get("mock_telegram"):
            feats = np.array([0.1, 0.2, 0.0, 0.3], dtype=np.float32)
        else:
            feats = sentiment_processor.get_sentiment_features(
                api_token=cfg.get("telegram_api_token", ""),
                channel_id=cfg.get("telegram_channel_id", ""),
                gpt_api_key=cfg.get("openai_api_key", ""),
                gpt_model=cfg.get("openai_model", "gpt-4o"),
            )

        feats = np.asarray(feats, dtype=np.float32).flatten()
        threshold = float(cfg.get("sentiment_threshold", 0.0))
        if threshold > 0:
            feats = np.where(np.abs(feats) >= threshold, feats, 0.0)
        if cfg.get("normalize_features", True):
            feats = (feats - feats.mean()) / (feats.std() + 1e-9)

        pd.DataFrame([feats], columns=["sentiment_score", "confidence", "direction", "strength"]).to_csv(
            output_csv, index=False
        )
        status = "mock" if mode == "dry-run" or cfg.get("mock_gpt") or cfg.get("mock_telegram") else "ok"
        append_log("nlp", f"Sentiment features saved to {output_csv} (status={status})")
        return {"status": status, "output_csv": str(output_csv), "features": feats.tolist()}
    except Exception as exc:
        append_log("nlp", f"ERROR: NLP stage failed - {exc}")
        return {"status": "error", "error": str(exc), "output_csv": str(output_csv)}


def run_heston(cfg: Dict, mode: str) -> Dict:
    """Execute Heston stage (feature extraction)."""
    append_log("heston", f"Starting Heston stage (mode={mode})")
    output_csv = Path(cfg.get("output_csv", "STATICS/Heston/features.csv"))
    _ensure_parent(output_csv)

    if not cfg.get("enabled", True):
        append_log("heston", "Heston module disabled; skipping.")
        return {"status": "skipped", "output_csv": str(output_csv)}

    try:
        csv_path = Path(cfg.get("csv_path", "heston_dummy_calls.csv"))
        if not csv_path.exists() and mode == "dry-run":
            append_log("heston", f"CSV {csv_path} missing; generating synthetic grid for dry-run.")
            df = pd.DataFrame(
                {
                    "S0": [100.0, 100.0, 100.0, 100.0],
                    "K": [95, 100, 105, 110],
                    "T": [0.25, 0.5, 0.75, 1.0],
                    "C_mkt": [11.0, 9.5, 8.2, 7.0],
                }
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)

        if mode == "dry-run":
            feats = np.array([1.0, 0.04, 0.3, -0.5, 0.05], dtype=np.float32)
            pd.DataFrame([feats], columns=["kappa", "theta", "sigma", "rho", "v0"]).to_csv(output_csv, index=False)
            append_log("heston", f"Mock Heston features saved to {output_csv}")
            return {"status": "mock", "output_csv": str(output_csv), "features": feats.tolist()}

        features, metrics = get_heston_features_from_csv(
            csv_path=csv_path,
            r=float(cfg.get("pricer", {}).get("r", 0.02)),
            q=float(cfg.get("pricer", {}).get("q", 0.0)),
            max_iters=int(cfg.get("pricer", {}).get("max_iters", 5)),
            lr=float(cfg.get("pricer", {}).get("lr", 0.05)),
            device=None if cfg.get("pricer", {}).get("device", "cpu") == "cpu" else cfg.get("pricer", {}).get("device"),
        )
        pd.DataFrame([features], columns=["kappa", "theta", "sigma", "rho", "v0"]).to_csv(output_csv, index=False)
        append_log("heston", f"Heston features saved to {output_csv} | metrics={metrics}")
        return {"status": "ok", "output_csv": str(output_csv), "features": features.tolist(), "metrics": metrics}
    except Exception as exc:
        append_log("heston", f"ERROR: Heston stage failed - {exc}")
        return {"status": "error", "error": str(exc), "output_csv": str(output_csv)}


def _build_network(n_state: int, n_action: int) -> Basics.Network:
    if not TF_OK:
        raise ImportError(f"TensorFlow missing: {TF_IMPORT_ERROR}")
    tf.reset_default_graph()
    state_in = tf.placeholder(shape=[1], dtype=tf.int32)
    network = Basics.Network(state_in)

    one_hot = {
        "one_hot": {"func_name": "one_hot", "input_arg": "indices", "layer_para": {"indices": None, "depth": n_state}}
    }
    output_layer = {
        "final": {
            "func_name": "fully_connected",
            "input_arg": "inputs",
            "layer_para": {
                "inputs": None,
                "num_outputs": n_action,
                "biases_initializer": None,
                "activation_fn": tf.nn.relu,
                "weights_initializer": tf.ones_initializer(),
            },
        }
    }
    network.build_layers(one_hot)
    network.add_layer_duplicates(output_layer, 1)
    return network


def run_rl(cfg: Dict, mode: str, nlp_out: Dict | None = None, heston_out: Dict | None = None) -> Dict:
    """Run a lightweight RL training loop."""
    append_log("rl", f"Starting RL stage (mode={mode})")
    if not TF_OK:
        msg = f"TensorFlow not installed: {TF_IMPORT_ERROR}"
        append_log("rl", f"ERROR: {msg}")
        return {"status": "error", "error": msg}
    if cfg.get("dry_run") or mode == "dry-run":
        append_log("rl", "Dry-run enabled; skipping heavy training.")
        return {"status": "dry-run"}

    try:
        df_x, df_y = _generate_synthetic_pair()
        df_x, df_y = EGCointegration.clean_data(df_x, df_y, "date", "close")
        train_len = int(len(df_x) * 0.7)
        x_train, y_train = df_x.iloc[:train_len, :], df_y.iloc[:train_len, :]

        actions = {
            "n_hist": [60, 120],
            "n_forward": [120, 240],
            "trade_th": [1.0, 2.0],
            "stop_loss": [1.0],
            "cl": [0.05, 0.1],
        }
        n_action = int(np.prod([len(actions[k]) for k in actions]))
        n_state = 1

        network = _build_network(n_state, n_action)
        base_cfg = FileIO.read_yaml(str(REPO_ROOT / "CONFIG" / "config_train.yml"))
        base_cfg["ActionSpaceAction"] = actions
        base_cfg["StateSpaceState"] = {"transaction_cost": [0.001]}
        base_cfg["AgentLearningRate"] = float(cfg.get("learning_rate", 0.001))
        base_cfg["Counter"]["Counter_1"]["end_num"] = int(cfg.get("episodes", 3))
        base_cfg["Counter"]["Counter_2"]["end_num"] = 200
        base_cfg["Counter"]["Counter_2"]["n_buffer"] = 0

        eg_train = EGCointegration(x_train, y_train, "date", "close")
        agent = Reinforcement.ContextualBandit(network, base_cfg, eg_train)
        with tf.Session() as sess:
            agent.process(sess, save=False, restore=False)
            rewards = agent.recorder.record.get("ENGINE_REWARD", [])
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            append_log("rl", f"RL training done. Mean reward={mean_reward:.4f}")
            return {"status": "ok", "mean_reward": mean_reward, "n_rewards": len(rewards)}
    except Exception as exc:
        append_log("rl", f"ERROR: RL stage failed - {exc}")
        return {"status": "error", "error": str(exc)}


def run_pipeline(mode: str) -> Dict:
    """Run NLP -> Heston -> RL with centralized logging."""
    start = time.time()
    append_log("rl", f"Pipeline triggered (mode={mode})")
    nlp_cfg = load_config("nlp")
    heston_cfg = load_config("heston")
    rl_cfg = load_config("rl")

    nlp_res = run_nlp(nlp_cfg, mode)
    heston_res = run_heston(heston_cfg, mode)
    rl_res = run_rl(rl_cfg, mode, nlp_out=nlp_res, heston_out=heston_res)
    elapsed = time.time() - start
    append_log("rl", f"Pipeline finished in {elapsed:.2f}s")
    return {"nlp": nlp_res, "heston": heston_res, "rl": rl_res, "elapsed": elapsed}


def _config_tabs(default_tab: str | None = None) -> None:
    tabs = st.tabs(["Config NLP", "Config Heston", "Config RL"])
    if default_tab:
        st.caption(f"Focused on: {default_tab.upper()}")
    with tabs[0]:
        nlp_config.render()
    with tabs[1]:
        heston_config.render()
    with tabs[2]:
        rl_config.render()


def main() -> None:
    st.set_page_config(page_title="Pair Trading RL UI", layout="wide")
    load_css()
    st.title("Pair Trading Reinforcement Learning UI")
    st.caption("Configure NLP, Heston, and RL pipelines, run them, and monitor logs.")
    st.info(
        "Bienvenue ! Cette interface vous guide pour enchainer trois briques : "
        "1) extraire un sentiment Telegram/LLM (NLP), 2) calibrer Heston et produire des features, "
        "3) lancer un entrainement RL (bandit contextuel) sur des prix synthetiques. "
        "Passez sur chaque onglet pour regler les hyperparametres, puis declenchez le pipeline depuis la sidebar."
    )
    if not TF_OK:
        st.warning(f"TensorFlow non installe dans cet environnement ({TF_IMPORT_ERROR}). "
                   "L'onglet RL fonctionnera uniquement après installation (pip install tensorflow==2.10.1).")

    with st.sidebar:
        st.header("Controls")
        selected_module = st.selectbox("Focus config tab", options=["nlp", "heston", "rl"], format_func=str.upper)
        run_mode = st.radio("Execution mode", options=["dry-run", "real-run"], index=0)
        st.write(
            "- **dry-run** : utilise des donnees/valeurs factices pour aller vite.\n"
            "- **real-run** : lance les modules reels (verifiez vos tokens/chemins et la dispo GPU)."
        )
        if st.button("Run pipeline"):
            st.session_state["last_run"] = run_pipeline(run_mode)
            st.success(f"Pipeline launched in {run_mode} mode.")
            st.toast("Pipeline en cours : surveillez les logs.")

    tabs = st.tabs(["Configuration", "Execution Logs"])
    with tabs[0]:
        st.markdown(
            "Choisissez un onglet pour regler la configuration de chaque module. "
            "Les boutons **Save config** ecrivent les fichiers YAML dans `CONFIG/`."
        )
        _config_tabs(default_tab=selected_module)
    with tabs[1]:
        st.markdown("Suivez l'avancement ou les erreurs dans les logs. Activez l'auto-refresh pour un suivi en direct.")
        logs_page.render()

    if st.session_state.get("last_run"):
        st.divider()
        st.subheader("Last pipeline run summary")
        run_res = st.session_state["last_run"]
        st.write({"elapsed_sec": round(run_res.get("elapsed", 0), 2)})
        col1, col2, col3 = st.columns(3)
        col1.metric("NLP status", run_res["nlp"].get("status", "n/a"))
        col2.metric("Heston status", run_res["heston"].get("status", "n/a"))
        col3.metric("RL status", run_res["rl"].get("status", "n/a"))
        st.json(run_res)


if __name__ == "__main__":
    main()
