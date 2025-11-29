# Expose get_heston_dataset when imported as a package; tolerate direct execution contexts.
try:
    from heston_model.heston_data_builder import get_heston_dataset  # type: ignore
    __all__ = ["get_heston_dataset"]
except Exception:  # pragma: no cover - allow non-package import contexts
    __all__ = []
