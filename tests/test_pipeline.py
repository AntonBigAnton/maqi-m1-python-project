# small checks: the one-command entry point
from datetime import date

from src import pipeline
from src.config import load_config
from src.features import BLOCK_A, BLOCK_B


def test_quick_run_never_writes_to_the_report_folders():
    cfg = load_config()
    quick = pipeline.quick_outputs(cfg)
    assert quick["figures_dir"] == cfg["data_dir"] / "quick" / "figures"
    assert quick["tables_dir"] == cfg["data_dir"] / "quick" / "tables"
    assert cfg["figures_dir"].name == "figures" and cfg["figures_dir"].parent.name == "report"  # left unchanged


def test_months_in_window():
    assert pipeline.months_in_window(date(2026, 1, 1), date(2026, 1, 31)) == 1
    assert pipeline.months_in_window(date(2026, 1, 1), date(2026, 6, 30)) == 6
    assert pipeline.months_in_window(date(2026, 1, 15), date(2026, 3, 1)) == 3


def test_quick_window_is_too_short_for_the_walk_forward():
    cfg = load_config()
    start, end = pipeline.sample_window(cfg, quick=True)
    assert pipeline.months_in_window(start, end) < cfg["walk_forward"]["first_test_month"]


def test_feature_blocks_follow_the_config():
    block_a, block_b = pipeline.feature_blocks(load_config())
    assert block_a == BLOCK_A
    assert block_b == BLOCK_B


def test_feature_blocks_follow_the_config_lags():
    block_a, _ = pipeline.feature_blocks(load_config(lags=[2, 10]))
    assert block_a[:2] == ["a_ret_2", "a_ret_10"]
