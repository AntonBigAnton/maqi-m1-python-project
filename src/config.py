# config.yaml loader shared by every module
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

REQUIRED_KEYS = ("symbol", "horizon_minutes", "start_date", "end_date", "lags", "seed", "data_dir")
PATH_KEYS = ("data_dir", "figures_dir", "tables_dir")


def load_config(path=DEFAULT_CONFIG, **overrides):
    """Read the YAML config, apply overrides and resolve relative paths from the repo root.

    Overrides let a smoke test run another coin or horizon without a second file,
    e.g. load_config(symbol="ETHUSDT", horizon_minutes=5).
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(overrides)

    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise KeyError(f"{path} is missing: {', '.join(missing)}")

    for key in PATH_KEYS:
        if key in cfg:
            cfg[key] = REPO_ROOT / cfg[key]
    return cfg