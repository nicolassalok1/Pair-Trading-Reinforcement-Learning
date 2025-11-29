import numpy as np
import pandas as pd
import pytest
from pathlib import Path

env_mod = pytest.importorskip("envs.pairtrading_env")
PairTradingEnv = getattr(env_mod, "PairTradingEnv", None)
if PairTradingEnv is None:
    pytest.skip("PairTradingEnv not available.", allow_module_level=True)

heston_mod = pytest.importorskip("heston_model.calibrate_heston")
HestonParams = getattr(heston_mod, "HestonParams", None)
if HestonParams is None:
    pytest.skip("HestonParams not available; heston_torch missing.", allow_module_level=True)


def _make_dummy_calls(csv_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "S0": [100.0, 100.0, 100.0, 100.0],
            "K": [95, 100, 105, 110],
            "T": [0.25, 0.5, 0.75, 1.0],
            "C_mkt": [11.0, 9.5, 8.2, 7.0],
            "iv_market": [0.29, 0.28, 0.27, 0.26],
        }
    )
    df.to_csv(csv_path, index=False)
    print("\n[RL Fusion] Heston CSV header & first 3 rows:")
    print(df.head(3))
    return csv_path


def test_state_fusion_with_heston_and_sentiment(monkeypatch, tmp_path: Path):
    csv_file = _make_dummy_calls(tmp_path / "dummy_calls.csv")
    # Compute Heston features quickly
    feats, metrics = heston_mod.get_heston_features_from_csv(
        csv_path=csv_file,
        r=0.02,
        q=0.0,
        max_iters=3,
        lr=0.05,
        device=None,
    )
    print(f"[RL Fusion] Heston features: {feats}")
    print(f"[RL Fusion] Heston metrics: {metrics}")

    sentiment_vec = np.array([0.1, 0.2, 1.0, 0.3], dtype=np.float32)

    env = PairTradingEnv(
        market_feature_dim=3,
        heston_feature_dim=5,
        use_sentiment=True,
        heston_csv_path=None,  # we will inject manually
    )
    # Inject precomputed Heston features and sentiment
    env._heston_features = feats.astype(np.float32)
    monkeypatch.setattr(env, "_fetch_sentiment", lambda: sentiment_vec)

    market_features = np.array([1.0, -0.5, 0.25], dtype=np.float32)
    merged_state = env.build_state(market_features)
    print(f"[RL Fusion] Market features: {market_features}")
    print(f"[RL Fusion] Sentiment features: {sentiment_vec}")
    print(f"[RL Fusion] Merged state (len={len(merged_state)}): {merged_state}")

    # Dummy RL output (order) based on simple rule
    order = "buy" if merged_state.sum() > 0 else "sell"
    print(f"[RL Fusion] RL order: {order}")

    assert merged_state.shape[0] == env.observation_space.shape[0]
    assert np.all(np.isfinite(merged_state))
