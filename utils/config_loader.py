import yaml
from pathlib import Path
from typing import Dict


def load_nlp_config(path: str = None) -> Dict:
    """
    Load NLP config from YAML.
    """
    default_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "nlp_config.yaml"
    with open(default_path, "r") as f:
        return yaml.safe_load(f)
