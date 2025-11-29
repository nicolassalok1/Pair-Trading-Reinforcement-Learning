"""
Gym-compatible environment skeleton integrating sentiment features.

Only input shape/state concatenation is managed here; reward/policy logic is unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import gym
import numpy as np
from gym import spaces

from nlp_module.sentiment_processor import get_sentiment_features
from utils.config_loader import load_nlp_config

logger = logging.getLogger(__name__)


class PairTradingEnv(gym.Env):
    """
    Minimal environment skeleton that appends NLP sentiment features to the state vector.
    The user is expected to supply market_features (and optional heston_features)
    when building the state; only the concatenation and observation space sizing are handled.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        market_feature_dim: int,
        action_space_n: int = 2,
        heston_feature_dim: int = 0,
        use_sentiment: bool = True,
        nlp_config_path: Optional[str] = None,
    ):
        super().__init__()
        self.market_feature_dim = int(market_feature_dim)
        self.heston_feature_dim = int(heston_feature_dim)
        self.sentiment_dim = 4 if use_sentiment else 0
        self.use_sentiment = use_sentiment

        self.action_space = spaces.Discrete(action_space_n)
        obs_dim = self.market_feature_dim + self.heston_feature_dim + self.sentiment_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._nlp_cfg = load_nlp_config(nlp_config_path) if use_sentiment else {}

    def _fetch_sentiment(self) -> np.ndarray:
        if not self.use_sentiment:
            return np.zeros(0, dtype=np.float32)
        try:
            return get_sentiment_features(
                api_token=self._nlp_cfg.get("telegram_api_token", ""),
                channel_id=self._nlp_cfg.get("telegram_channel_id", ""),
                gpt_api_key=self._nlp_cfg.get("gpt_api_key", ""),
                gpt_model=self._nlp_cfg.get("gpt_model", "gpt-4o"),
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Sentiment fetch failed: %s", exc)
            return np.zeros(4, dtype=np.float32)

    def build_state(self, market_features: np.ndarray, heston_features: Optional[np.ndarray] = None) -> np.ndarray:
        market_features = np.asarray(market_features, dtype=np.float32).flatten()
        heston_features = np.asarray(heston_features, dtype=np.float32).flatten() if heston_features is not None else np.zeros(self.heston_feature_dim, dtype=np.float32)
        sentiment_features = self._fetch_sentiment() if self.use_sentiment else np.zeros(0, dtype=np.float32)
        return np.concatenate([market_features, heston_features, sentiment_features]).astype(np.float32)

    def reset(self, **kwargs):
        # Placeholder reset; actual logic should populate market/heston features externally.
        empty_market = np.zeros(self.market_feature_dim, dtype=np.float32)
        empty_heston = np.zeros(self.heston_feature_dim, dtype=np.float32)
        state = self.build_state(empty_market, empty_heston)
        return state

    def step(self, action):
        raise NotImplementedError("PairTradingEnv step logic not defined; integrate with existing RL loop.")
