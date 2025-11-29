import numpy as np
import pandas as pd
import pytest
from pathlib import Path

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch unavailable or failed to load; skipping Heston tests.", allow_module_level=True)

heston_module = pytest.importorskip("heston_model.calibrate_heston")
HestonParams = getattr(heston_module, "HestonParams", None)
if HestonParams is None:
    pytest.skip("HestonParams not available; heston_torch missing.", allow_module_level=True)


def test_heston_params_to_features_roundtrip():
    params = HestonParams(
        kappa=torch.tensor(1.0),
        theta=torch.tensor(0.04),
        sigma=torch.tensor(0.3),
        rho=torch.tensor(-0.5),
        v0=torch.tensor(0.05),
    )
    feats = heston_module.heston_params_to_features(params)
    assert feats.shape == (5,)
    assert np.allclose(feats, [1.0, 0.04, 0.3, -0.5, 0.05])


def test_get_heston_features_from_csv(tmp_path: Path):
    csv_src = Path(__file__).resolve().parents[1] / "heston_dummy_calls.csv"
    if not csv_src.exists():
        pytest.skip("heston_dummy_calls.csv not found.")
    # Copy to temp to avoid modifying originals
    csv_local = tmp_path / "heston_dummy_calls.csv"
    csv_local.write_bytes(csv_src.read_bytes())

    features, metrics = heston_module.get_heston_features_from_csv(
        csv_path=csv_local,
        r=0.02,
        q=0.0,
        max_iters=5,  # keep it fast for tests
        lr=0.05,
        device=torch.device("cpu"),
    )
    assert features.shape == (5,)
    assert np.all(np.isfinite(features))
    assert "loss" in metrics and "rmse" in metrics and "n_points" in metrics
