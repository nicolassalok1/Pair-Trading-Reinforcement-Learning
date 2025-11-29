"""File-based log helpers for Streamlit UI."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List

LOG_ENV = "PAIR_RL_LOG_DIR"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _log_dir() -> Path:
    env_dir = os.environ.get(LOG_ENV)
    base = Path(env_dir) if env_dir else _repo_root() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _log_path(module_name: str) -> Path:
    return _log_dir() / f"{module_name.lower()}.log"


def append_log(module_name: str, message: str) -> Path:
    """Append a timestamped line to the module log."""
    path = _log_path(module_name)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} - {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def get_log(module_name: str, tail: int | None = 500) -> List[str]:
    """Return log lines (tail-first if specified)."""
    path = _log_path(module_name)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if tail is not None and tail > 0:
        return lines[-tail:]
    return lines


def clear_log(module_name: str) -> None:
    """Remove the log file for a module."""
    path = _log_path(module_name)
    if path.exists():
        path.unlink()


__all__ = ["append_log", "get_log", "clear_log"]
