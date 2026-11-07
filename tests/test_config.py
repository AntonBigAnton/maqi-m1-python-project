import re
import ast
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


def string_literals(path):
    """Yield (line, text) for every string in a module, docstrings and comments excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and ast.get_docstring(node) is not None
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def test_src_hardcodes_no_symbol_or_date():
    """Issue #3: the coin and the sample window come from config.yaml only."""
    pattern = re.compile(r"\b(?:BTC|ETH)[A-Z]*\b|\b\d{4}-\d{2}-\d{2}\b")
    offenders = [
        f"{path.name}:{line}: {text!r}"
        for path in sorted((REPO_ROOT / "src").glob("*.py"))
        for line, text in string_literals(path)
        if pattern.search(text)
    ]
    assert not offenders, "\n".join(offenders)