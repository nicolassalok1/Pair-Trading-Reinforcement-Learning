import numpy as np
import pandas as pd
import torch

from envs.pairtrading_env import PairTradingEnv
from heston_model.calibrate_heston import get_heston_features_from_market_df


def test_env_state_simple_no_sentiment():
    env = PairTradingEnv(
        market_feature_dim=3,
        heston_feature_dim=5,
        use_sentiment=False,  # avoids config dependency
    )
    env._heston_features = np.arange(5, dtype=np.float32)
    market = np.array([1.0, -0.5, 0.25], dtype=np.float32)
    state = env.build_state(market)
    assert state.shape == (8,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()


def test_env_state_stress_large_features():
    market_dim = 256
    env = PairTradingEnv(
        market_feature_dim=market_dim,
        heston_feature_dim=5,
        use_sentiment=False,
    )
    env._heston_features = np.linspace(0.1, 0.5, 5, dtype=np.float32)
    market = np.random.randn(market_dim).astype(np.float32)
    state = env.build_state(market)
    assert state.shape[0] == market_dim + 5
    assert state.dtype == np.float32
    assert np.isfinite(state).all()


def test_heston_calibration_stress_small_iters():
    n = 50
    df = pd.DataFrame(
        {
            "S0": np.full(n, 100.0),
            "K": np.linspace(80, 120, n),
            "T": np.linspace(0.2, 1.5, n),
            "C_mkt": np.linspace(5.0, 15.0, n),
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats, metrics = get_heston_features_from_market_df(
        df_options=df,
        r=0.02,
        q=0.0,
        max_iters=8,  # keep fast; just a stability check
        lr=0.05,
        device=device,
    )
    assert feats.shape == (5,)
    assert np.isfinite(feats).all()
    assert {"loss", "rmse", "n_points"}.issubset(metrics.keys())
