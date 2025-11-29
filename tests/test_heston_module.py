import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Avoid hard crashes when torch DLLs are broken: only run if explicitly forced
# with a very explicit value to prevent accidental crashes. Log the reason.
force_flag = os.environ.get("RUN_TORCH_TESTS_FORCE", "").lower()
if force_flag != "force_import":
    pytest.skip(
        f"Skipping torch-dependent Heston tests: RUN_TORCH_TESTS_FORCE={force_flag!r} "
        "(set RUN_TORCH_TESTS_FORCE=force_import to attempt at your own risk).",
        allow_module_level=True,
    )

try:
    import torch  # noqa: E402
except BaseException as exc:  # catch broad to avoid fatal DLL crashes
    pytest.skip(f"Skipping Heston tests: torch failed to load ({exc}).", allow_module_level=True)

heston_mod = pytest.importorskip("heston_model.calibrate_heston")
HestonParams = getattr(heston_mod, "HestonParams", None)
if HestonParams is None:
    pytest.skip("HestonParams not available; heston_torch missing.", allow_module_level=True)


def _make_dummy_calls(csv_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "S0": [100.0, 100.0, 100.0, 100.0, 100.0],
            "K": [90, 95, 100, 105, 110],
            "T": [0.25, 0.5, 0.75, 1.0, 1.25],
            "C_mkt": [12.0, 10.5, 9.0, 8.0, 7.5],
            "iv_market": [0.3, 0.28, 0.27, 0.26, 0.25],
        }
    )
    df.to_csv(csv_path, index=False)
    print("\n[Heston] CSV header & first 3 rows:")
    print(df.head(3))
    return csv_path


def test_get_heston_features_from_csv(tmp_path: Path):
    csv_file = _make_dummy_calls(tmp_path / "dummy_calls.csv")
    features, metrics = heston_mod.get_heston_features_from_csv(
        csv_path=csv_file,
        r=0.02,
        q=0.0,
        max_iters=5,  # keep fast
        lr=0.05,
        device=torch.device("cpu"),
    )
    print(f"[Heston] Input CSV: {csv_file}")
    print(f"[Heston] Features: {features}")
    print(f"[Heston] Metrics: {metrics}")
    assert features.shape == (5,)
    assert np.all(np.isfinite(features))
    assert set(["loss", "rmse", "n_points"]).issubset(metrics.keys())


def test_heston_params_to_features_roundtrip():
    params = HestonParams(
        kappa=torch.tensor(1.0),
        theta=torch.tensor(0.04),
        sigma=torch.tensor(0.3),
        rho=torch.tensor(-0.5),
        v0=torch.tensor(0.05),
    )
    feats = heston_mod.heston_params_to_features(params)
    print(f"[Heston] Params -> features: {feats}")
    assert feats.shape == (5,)
    assert np.allclose(feats, [1.0, 0.04, 0.3, -0.5, 0.05])
