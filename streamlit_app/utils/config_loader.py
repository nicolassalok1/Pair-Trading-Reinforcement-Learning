"""Config loader for Streamlit UI.

Provides read/write helpers with defaults and light validation.
Config files are stored under repo_root/config/<module_name>_config.yaml.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Tuple

import yaml

DEFAULT_CONFIG: Dict[str, Dict] = {
    "nlp": {
        "enabled": True,
        "openai_api_key": "",
        "openai_model": "gpt-4o",
        "telegram_api_token": "",
        "telegram_channel_id": "",
        "sentiment_threshold": 0.25,
        "normalize_features": True,
        "mock_gpt": True,
        "mock_telegram": True,
        "output_csv": "STATICS/NLP/sentiment_features.csv",
    },
    "heston": {
        "enabled": True,
        "csv_path": "heston_dummy_calls.csv",
        "window_size": 50,
        "maturities": [0.25, 0.5, 1.0],
        "strikes": [0.8, 1.0, 1.2],
        "normalization": "minmax",
        "pricer": {"max_iters": 5, "lr": 0.05, "r": 0.02, "q": 0.0, "device": "cpu"},
        "output_csv": "STATICS/Heston/features.csv",
    },
    "rl": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "gamma": 0.99,
        "episodes": 3,
        "exploration_rate": 0.1,
        "policy": "epsilon_greedy",
        "device": "cpu",
        "use_heston": True,
        "use_nlp": True,
        "dry_run": False,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_dir() -> Path:
    env_dir = os.environ.get("PAIR_RL_CONFIG_DIR")
    base = Path(env_dir) if env_dir else _repo_root() / "config"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _config_path(module_name: str) -> Path:
    module = module_name.lower()
    return _config_dir() / f"{module}_config.yaml"


def _merge_defaults(module: str, cfg: Dict) -> Dict:
    """Merge user cfg with defaults, preserving user overrides."""
    default_cfg = copy.deepcopy(DEFAULT_CONFIG[module])
    merged = copy.deepcopy(cfg) if cfg else {}

    def _deep_merge(src: Dict, default: Dict) -> Dict:
        out = copy.deepcopy(default)
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(v, out[k])
            else:
                out[k] = v
        return out

    return _deep_merge(merged, default_cfg)


def load_config(module_name: str) -> Dict:
    """Load config for a module, creating it with defaults if missing."""
    module = module_name.lower()
    if module not in DEFAULT_CONFIG:
        raise ValueError(f"Unknown module: {module}")
    path = _config_path(module)
    if not path.exists():
        save_config(module, DEFAULT_CONFIG[module])
        return copy.deepcopy(DEFAULT_CONFIG[module])

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = _merge_defaults(module, data)
    # If merged differs (missing keys), resave to keep file consistent.
    if merged != data:
        save_config(module, merged)
    return merged


def save_config(module_name: str, cfg: Dict) -> Tuple[Path, Dict]:
    """Persist config to disk after merging defaults; returns path and merged cfg."""
    module = module_name.lower()
    if module not in DEFAULT_CONFIG:
        raise ValueError(f"Unknown module: {module}")
    merged = _merge_defaults(module, cfg or {})
    path = _config_path(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return path, merged


__all__ = ["load_config", "save_config", "DEFAULT_CONFIG"]
