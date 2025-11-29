import numpy as np
import pandas as pd
import pytest
from pathlib import Path

nlp_mod = pytest.importorskip("nlp_module.sentiment_processor")


def _make_dummy_messages(csv_path: Path) -> tuple[Path, str]:
    df = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 10:00:00", "2025-01-01 10:01:00", "2025-01-01 10:02:00"],
            "message": [
                "BTC/USD looks strong, expecting upside.",
                "Some consolidation expected.",
                "Bias turning neutral.",
            ],
        }
    )
    df.to_csv(csv_path, index=False)
    print("\n[NLP] CSV header & first 3 rows:")
    print(df.head(3))
    return csv_path, df.iloc[0]["message"]


def test_sentiment_pipeline_monkeypatch(monkeypatch, tmp_path: Path):
    csv_file, sample_msg = _make_dummy_messages(tmp_path / "dummy_msgs.csv")

    def fake_listen(api_token, channel_id, timeout=10):
        print(f"[NLP] listen_telegram_channel input token={api_token}, channel={channel_id}")
        return sample_msg

    def fake_analyze(message: str, api_key: str, model: str = "gpt-4o"):
        print(f"[NLP] analyze_sentiment_with_gpt input: {message}")
        return {
            "pair": "BTC/USD",
            "sentiment_score": 0.7,
            "direction": "long",
            "confidence": 0.8,
            "strength": 0.6,
        }

    monkeypatch.setattr(nlp_mod, "listen_telegram_channel", fake_listen)
    monkeypatch.setattr(nlp_mod, "analyze_sentiment_with_gpt", fake_analyze)

    feats = nlp_mod.get_sentiment_features(
        api_token="TOKEN",
        channel_id="CHANNEL",
        gpt_api_key="OPENAI_KEY",
        gpt_model="gpt-4o",
    )
    print(f"[NLP] Output features: {feats}")
    assert feats.shape == (4,)
    assert np.allclose(feats, [0.7, 0.8, 1.0, 0.6])
