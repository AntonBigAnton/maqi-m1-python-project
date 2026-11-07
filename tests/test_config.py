import re
from pathlib import Path

import pytest

from src.config import REPO_ROOT, load_config


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg["symbol"], str)
    assert isinstance(cfg["horizon_minutes"], int)
    assert cfg["start_date"] <= cfg["end_date"]
    assert all(isinstance(lag, int) for lag in cfg["lags"])


def test_paths_resolve_from_repo_root():
    cfg = load_config()
    assert cfg["data_dir"] == REPO_ROOT / "data"
    assert Path(cfg["figures_dir"]).is_absolute()


def test_overrides_replace_values():
    cfg = load_config(symbol="ETHUSDT", horizon_minutes=5)
    assert cfg["symbol"] == "ETHUSDT"
    assert cfg["horizon_minutes"] == 5


def test_missing_key_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("symbol: BTCUSDT\n", encoding="utf-8")
    with pytest.raises(KeyError, match="horizon_minutes"):
        load_config(path)


def test_src_hardcodes_no_symbol_or_date():
    """Issue #3: the coin and the sample window come from config.yaml only (docstrings excepted)."""
    pattern = re.compile(r"""["'](?:BTC|ETH)\w*["']|["']\d{4}-\d{2}-\d{2}["']""")
    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in sorted((REPO_ROOT / "src").glob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith(("#", '"""'))
    ]
    assert not offenders, "\n".join(offenders)