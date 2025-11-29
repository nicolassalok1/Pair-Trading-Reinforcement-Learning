"""
Lightweight package init to avoid hard failures when optional data-fetch
dependencies are missing. Calibration utilities can still be imported directly
via `heston_model.calibrate_heston`.
"""

__all__ = ["get_heston_dataset"]

try:
    from .heston_data_builder import get_heston_dataset
except Exception:
    # Provide a stub that raises a clear error if the data builder is requested.
    def get_heston_dataset(*args, **kwargs):  # type: ignore
        raise ImportError("get_heston_dataset unavailable: optional data dependencies are missing.")
