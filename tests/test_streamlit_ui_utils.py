import os
from pathlib import Path

from streamlit_app.utils import config_loader, log_handler


def test_config_loader_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PAIR_RL_CONFIG_DIR", str(tmp_path))
    cfg = config_loader.load_config("nlp")
    assert cfg["openai_model"]
    cfg["openai_model"] = "gpt-test"
    path, persisted = config_loader.save_config("nlp", cfg)
    assert Path(path).exists()
    cfg2 = config_loader.load_config("nlp")
    assert cfg2["openai_model"] == "gpt-test"
    assert persisted["openai_model"] == "gpt-test"


def test_log_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("PAIR_RL_LOG_DIR", str(tmp_path))
    log_handler.clear_log("nlp")
    log_handler.append_log("nlp", "hello")
    log_handler.append_log("nlp", "ERROR something bad")
    lines = log_handler.get_log("nlp", tail=None)
    assert len(lines) == 2
    assert any("ERROR" in l for l in lines)
    log_handler.clear_log("nlp")
    assert log_handler.get_log("nlp") == []
