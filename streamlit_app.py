import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow.compat.v1 as tf

from MAIN import Basics, Reinforcement
from STRATEGY.Cointegration import EGCointegration
from UTIL import FileIO

tf.disable_v2_behavior()
tf.get_logger().setLevel("ERROR")

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_TRAIN_PATH = REPO_ROOT / "CONFIG" / "config_train.yml"
PRICE_DIR = REPO_ROOT / "STATICS" / "PRICE"


@st.cache_data
def _load_config(path: Path = CONFIG_TRAIN_PATH) -> Dict:
    return FileIO.read_yaml(str(path))


@st.cache_data
def _list_price_files() -> List[str]:
    if not PRICE_DIR.exists():
        return []
    return sorted([p.stem for p in PRICE_DIR.glob("*.csv")])


@st.cache_data
def _read_price_csv(path: Path, n_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if n_rows:
        df = df.iloc[:n_rows, :]
    return df


@st.cache_data
def _generate_synthetic_pair(
    base_a: float,
    base_b: float,
    vol_a: float,
    vol_b: float,
    corr: float,
    n_obs: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def _normalize_price_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError(f"Le fichier {label} doit contenir les colonnes 'date' et 'close'.")
    return df[[cols["date"], cols["close"]]].rename(columns={cols["date"]: "date", cols["close"]: "close"})


def _prepare_datasets(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    train_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_x, df_y = EGCointegration.clean_data(df_x.copy(), df_y.copy(), "date", "close")
    n_total = len(df_x)
    n_train = max(2, int(n_total * train_ratio))
    df_x_train = df_x.iloc[:n_train, :].reset_index(drop=True)
    df_y_train = df_y.iloc[:n_train, :].reset_index(drop=True)
    df_x_test = df_x.iloc[n_train:, :].reset_index(drop=True)
    df_y_test = df_y.iloc[n_train:, :].reset_index(drop=True)
    return df_x_train, df_y_train, df_x_test, df_y_test


def _build_spaces(
    n_hist_list: List[int],
    n_forward_list: List[int],
    trade_th_list: List[float],
    stop_loss_list: List[float],
    cl_list: List[float],
) -> Tuple[Dict, int]:
    actions = {
        "n_hist": n_hist_list,
        "n_forward": n_forward_list,
        "trade_th": trade_th_list,
        "stop_loss": stop_loss_list,
        "cl": cl_list,
    }
    n_action = int(np.prod([len(actions[key]) for key in actions.keys()]))
    return actions, n_action


def _build_network(n_state: int, n_action: int) -> Tuple[Basics.Network, Dict]:
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
    return network, {"state_in": state_in}


def _run_training(
    EG_train: EGCointegration,
    config_base: Dict,
    actions: Dict,
    transaction_cost: float,
    epochs: int,
    iterations: int,
    learning_rate: float,
    experience_replay: bool,
) -> Dict:
    config = copy.deepcopy(config_base)
    config["StateSpaceState"] = {"transaction_cost": [transaction_cost]}
    config["ActionSpaceAction"] = actions
    config["AgentLearningRate"] = float(learning_rate)
    config["ExperienceReplay"] = experience_replay
    config["Counter"]["Counter_1"]["end_num"] = int(epochs)
    config["Counter"]["Counter_2"]["end_num"] = int(iterations)
    config["Counter"]["Counter_2"]["n_buffer"] = 0
    config["Counter"]["Counter_2"]["print_freq"] = None
    config["Counter"]["Counter_4"]["n_buffer"] = max(iterations // 2, 1)

    n_state = len(config["StateSpaceState"])
    n_action = int(np.prod([len(actions[key]) for key in actions.keys()]))
    network, placeholders = _build_network(n_state, n_action)
    agent = Reinforcement.ContextualBandit(network, config, EG_train)

    with tf.Session() as sess:
        agent.process(sess, save=False, restore=False)
        actions_log = agent.recorder.record.get("NETWORK_ACTION", [])
        rewards_log = agent.recorder.record.get("ENGINE_REWARD", [])
        rewards_df = pd.DataFrame({"action": actions_log, "reward": rewards_log})
        logits = sess.run(agent.output, feed_dict=agent.feed_dict)
        opt_action = int(np.argmax(logits))
        action_dict = agent.action_space.convert(opt_action, "index_to_dict")

    mean_reward = float(rewards_df["reward"].mean()) if len(rewards_df) > 0 else 0.0
    return {
        "rewards_df": rewards_df,
        "mean_reward": mean_reward,
        "opt_action": opt_action,
        "action_dict": action_dict,
        "n_action": n_action,
    }


def _run_backtest(
    EG_test: EGCointegration,
    action_dict: Dict,
    transaction_cost: float,
    max_steps: int,
) -> Dict:
    n_hist = int(action_dict["n_hist"])
    n_forward = int(action_dict["n_forward"])
    start_idx = n_hist + 1
    end_idx = len(EG_test.x) - n_forward
    if end_idx <= start_idx:
        return {"net": [], "returns": [], "cum_net": []}

    indices = list(range(start_idx, min(end_idx, start_idx + max_steps)))
    net_profits: List[float] = []
    returns: List[float] = []

    for i in indices:
        EG_test.process(index=i, transaction_cost=transaction_cost, **action_dict)
        trade_record = EG_test.record
        if not trade_record:
            continue
        trade_df = pd.DataFrame(trade_record)
        net = trade_df["profit"] - trade_df["trade_cost"] - trade_df["close_cost"]
        trade_price = trade_df["trade_price"].replace(0, np.nan)
        net = net.replace([np.inf, -np.inf], np.nan)
        mask = net.notnull() & trade_price.notnull()
        if mask.any():
            net = net[mask]
            trade_price = trade_price[mask]
        net_list = net.tolist()
        net_profits.extend(net_list)
        ret = (net / trade_price.abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        returns.extend(ret.tolist())

    cum_net = list(np.cumsum(net_profits)) if net_profits else []
    return {"net": net_profits, "returns": returns, "cum_net": cum_net}


def _display_action_summary(action_dict: Dict):
    cols = st.columns(len(action_dict))
    for (name, value), col in zip(action_dict.items(), cols):
        col.metric(name, value)


def main():
    st.set_page_config(page_title="Pair Trading RL - Streamlit", layout="wide")
    st.title("Pair Trading Reinforcement Learning")
    st.markdown(
        "Interface Streamlit pour tester rapidement l'agent contextual bandit "
        "sur des paires de prix. Charge une paire de CSV (`date`, `close`) "
        "ou genere une paire synthetic, puis entraine et backteste la strategie cointegration."
    )

    with st.sidebar:
        st.header("Donnees")
        data_mode = st.radio("Source", ["Fichiers locaux", "Upload", "Synthetic"], index=0)
        max_rows = st.slider("Limiter le nombre de lignes", 1000, 20000, 8000, step=1000)
        train_ratio = st.slider("Partie entrainement", 0.5, 0.9, 0.7, step=0.05)

        available = _list_price_files()
        ticker_x = st.selectbox("Serie A", available, index=available.index("BTC") if "BTC" in available else 0)
        ticker_y = st.selectbox("Serie B", available, index=available.index("USD") if "USD" in available else 0)
        upload_x = st.file_uploader("CSV Serie A", type="csv", key="upload_x") if data_mode == "Upload" else None
        upload_y = st.file_uploader("CSV Serie B", type="csv", key="upload_y") if data_mode == "Upload" else None
        base_a = st.number_input("Base A (synthetic)", value=30000.0) if data_mode == "Synthetic" else None
        base_b = st.number_input("Base B (synthetic)", value=1.3) if data_mode == "Synthetic" else None
        vol_a = st.number_input("Vol A", value=0.6, format="%.3f") if data_mode == "Synthetic" else None
        vol_b = st.number_input("Vol B", value=0.25, format="%.3f") if data_mode == "Synthetic" else None
        corr = st.slider("Correlation", -0.95, 0.95, 0.7) if data_mode == "Synthetic" else None

        st.header("Espace d'action")
        n_hist_list = st.multiselect("n_hist (historique)", options=list(range(60, 601, 60)), default=[60, 180, 360])
        n_forward_list = st.multiselect(
            "n_forward (horizon)", options=list(range(120, 1201, 120)), default=[240, 480, 720]
        )
        trade_th_list = st.multiselect("trade_th (ecart std)", options=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0], default=[1.0, 2.0])
        stop_loss_list = st.multiselect("stop_loss", options=[1.0, 1.5, 2.0], default=[1.0, 1.5])
        cl_list = st.multiselect("p-value seuil (cl)", options=[0.05, 0.1, 0.15], default=[0.05, 0.1])

        st.header("Entrainement")
        epochs = st.slider("Epochs", 1, 20, 4)
        iterations = st.slider("Iterations / epoch", 50, 3000, 800, step=50)
        learning_rate = st.number_input("Learning rate", value=0.001, format="%.4f")
        transaction_cost = st.number_input("Transaction cost", value=0.001, format="%.4f")
        experience_replay = st.checkbox("Experience replay", value=True)
        max_backtest_steps = st.slider("Pas de backtest", 50, 2000, 400, step=50)

    try:
        if data_mode == "Upload":
            if upload_x is None or upload_y is None:
                st.info("Charge deux CSV pour lancer l'entrainement.")
                return
            df_x = _normalize_price_df(pd.read_csv(upload_x), "A")
            df_y = _normalize_price_df(pd.read_csv(upload_y), "B")
        elif data_mode == "Synthetic":
            df_x, df_y = _generate_synthetic_pair(
                base_a=base_a or 30000.0,
                base_b=base_b or 1.3,
                vol_a=vol_a or 0.6,
                vol_b=vol_b or 0.25,
                corr=corr or 0.7,
                n_obs=max_rows,
            )
        else:
            if not available:
                st.error("Aucun fichier trouve dans STATICS/PRICE.")
                return
            df_x = _normalize_price_df(_read_price_csv(PRICE_DIR / f"{ticker_x}.csv", max_rows), ticker_x)
            df_y = _normalize_price_df(_read_price_csv(PRICE_DIR / f"{ticker_y}.csv", max_rows), ticker_y)
    except Exception as exc:
        st.error(f"Erreur de chargement des donnees: {exc}")
        return

    st.subheader("Apercu des donnees")
    preview_cols = st.columns(2)
    preview_cols[0].dataframe(df_x.head(5))
    preview_cols[1].dataframe(df_y.head(5))

    df_x_train, df_y_train, df_x_test, df_y_test = _prepare_datasets(df_x, df_y, train_ratio)
    st.caption(
        f"Train: {len(df_x_train)} points | Test: {len(df_x_test)} points | "
        f"Action space: {len(n_hist_list)}x{len(n_forward_list)}x{len(trade_th_list)}x{len(stop_loss_list)}x{len(cl_list)}"
    )

    max_hist = max(n_hist_list) if n_hist_list else 0
    max_forward = max(n_forward_list) if n_forward_list else 0
    if len(df_x_train) < max_hist + max_forward + 5:
        st.warning("Dataset d'entrainement trop court pour les fenetres choisies.")
        return
    if len(df_x_test) < max_hist + max_forward + 5:
        st.warning("Dataset de test trop court pour les fenetres choisies.")
        return

    actions, n_action = _build_spaces(n_hist_list, n_forward_list, trade_th_list, stop_loss_list, cl_list)
    if n_action == 0:
        st.error("Choisis au moins une valeur par parametre d'action.")
        return
    if n_action > 800:
        st.warning(f"Espace d'action tres large ({n_action}). Reduis le nombre de valeurs pour garder un entrainement rapide.")

    train_clicked = st.button("Lancer l'entrainement")
    if train_clicked:
        with st.spinner("Entrainement en cours..."):
            try:
                config_base = _load_config()
                EG_train = EGCointegration(df_x_train, df_y_train, "date", "close")
                train_result = _run_training(
                    EG_train,
                    config_base,
                    actions,
                    transaction_cost,
                    epochs,
                    iterations,
                    learning_rate,
                    experience_replay,
                )
                st.session_state["train_result"] = train_result
            except Exception as exc:
                st.error(f"Echec de l'entrainement: {exc}")
                return

    train_result = st.session_state.get("train_result")
    if train_result:
        st.subheader("Resultats entrainement")
        st.metric("Reward moyenne", f"{train_result['mean_reward']:.4f}")
        _display_action_summary(train_result["action_dict"])

        rewards_df = train_result["rewards_df"]
        if not rewards_df.empty:
            summary = rewards_df.groupby("action")["reward"].agg(["mean", "count"]).reset_index()
            st.dataframe(summary.rename(columns={"mean": "reward_moyenne", "count": "occurences"}))
            st.line_chart(rewards_df["reward"], height=180)

        st.divider()
        backtest_clicked = st.button("Backtest sur le meilleur set d'actions")
        if backtest_clicked:
            with st.spinner("Backtest en cours..."):
                try:
                    EG_test = EGCointegration(df_x_test, df_y_test, "date", "close")
                    backtest_result = _run_backtest(
                        EG_test,
                        train_result["action_dict"],
                        transaction_cost,
                        max_steps=max_backtest_steps,
                    )
                    st.session_state["backtest_result"] = backtest_result
                except Exception as exc:
                    st.error(f"Backtest impossible: {exc}")
                    return

    backtest_result = st.session_state.get("backtest_result")
    if backtest_result:
        st.subheader("Backtest")
        net = backtest_result.get("net", [])
        returns = backtest_result.get("returns", [])
        cum_net = backtest_result.get("cum_net", [])
        st.metric("Trades simules", len(net))
        if net:
            st.metric("PNL total (unite prix)", f"{sum(net):.4f}")
            st.metric("PNL cumule max", f"{max(cum_net):.4f}")
            st.metric("Sharpe proxy (mean/std ret)", f"{np.mean(returns) / (np.std(returns)+1e-9):.2f}" if returns else "n/a")
            st.line_chart(pd.DataFrame({"PnL cumulee": cum_net}), height=200)
        else:
            st.info("Aucun trade genere sur le jeu de test pour ces parametres.")


if __name__ == "__main__":
    main()
