# small checks: parquet cache, output naming, sample window
from datetime import date

import pandas as pd

from src import clean, panel
from src.config import REPO_ROOT, load_config
from src.pipeline import sample_window


def test_load_or_build_saves_then_reads_from_cache(tmp_path):
    path = tmp_path / "clean" / "table.parquet"
    table = pd.DataFrame({"a": [1, 2]})
    built = []

    def build():
        built.append(True)
        return table

    pd.testing.assert_frame_equal(clean.load_or_build(path, build), table)
    pd.testing.assert_frame_equal(clean.load_or_build(path, build), table)
    assert len(built) == 1


def test_quick_and_full_runs_use_different_files():
    full = panel.panel_path("data", "BTCUSDT", date(2026, 1, 1), date(2026, 6, 30))
    quick = panel.panel_path("data", "BTCUSDT", date(2026, 1, 1), date(2026, 1, 31))
    assert full != quick


def test_quick_window_is_the_first_month():
    cfg = {"start_date": date(2026, 1, 1), "end_date": date(2026, 6, 30)}
    assert sample_window(cfg) == (date(2026, 1, 1), date(2026, 6, 30))
    assert sample_window(cfg, quick=True) == (date(2026, 1, 1), date(2026, 1, 31))


def test_data_dir_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config(REPO_ROOT / "config.yaml")["data_dir"] == REPO_ROOT / "data"
