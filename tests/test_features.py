# block A features: hand-checked values, no look-ahead, NaN after a gap, nothing hardcoded
import ast
import math

import numpy as np
import pandas as pd
import pytest

from src.config import REPO_ROOT, load_config
from src.features import BLOCK_A, block_a, block_a_columns, feature_lags

JAN_1_6AM_MS = 1767247200000  # 2026-01-01 06:00 UTC
MINUTE = 60_000
LAGS = [1, 5]

# Log VWAPs chosen so the returns are easy to check by hand
LOG_VWAP = [0.00, 0.01, 0.03, 0.02, 0.02, 0.05, 0.04, 0.06, 0.07, 0.05]
SIGNED_VOLUME = [2, -4, 6, 0, 4, -2, 8, 2, -6, 4]


def toy_panel(rows=10, missing_rows=()):
    """A 1-min spot panel: volume 10 and 5 trades every minute; listed rows are missing."""
    log_vwap = (LOG_VWAP * (rows // 10 + 1))[:rows]
    signed = (SIGNED_VOLUME * (rows // 10 + 1))[:rows]
    panel = pd.DataFrame(
        {
            "open_time": [JAN_1_6AM_MS + i * MINUTE for i in range(rows)],
            "spot_vwap": np.exp(log_vwap),
            "spot_signed_volume": np.array(signed, dtype=float),
            "spot_volume": 10.0,
            "spot_count": pd.array([5] * rows, dtype="Int64"),
            "spot_missing": False,
        }
    )
    for i in missing_rows:
        panel.loc[i, ["spot_vwap", "spot_signed_volume", "spot_volume"]] = np.nan
        panel.loc[i, "spot_count"] = pd.NA
        panel.loc[i, "spot_missing"] = True
    return panel


def test_hand_checked_values():
    # Each feature on row t is computed from rows up to t - 1.
    out = block_a(toy_panel(), LAGS)
    assert out.loc[3, "a_ret_1"] == pytest.approx(0.03 - 0.01)          # rows 1..2
    assert out.loc[6, "a_ret_5"] == pytest.approx(0.05 - 0.00)          # rows 0..5
    assert out.loc[2, "a_flow_1"] == pytest.approx(-4 / 10)             # row 1
    assert out.loc[5, "a_flow_5"] == pytest.approx((2 - 4 + 6 + 0 + 4) / 50)  # rows 0..4
    # returns of rows 1..5 are 0.01, 0.02, -0.01, 0, 0.03: mean 0.01, squared deviations sum to 0.001
    assert out.loc[6, "a_rv_5"] == pytest.approx(math.sqrt(0.001 / 4))
    # 06:00 is a quarter of the way round the day
    assert out.loc[0, "a_hour_sin"] == pytest.approx(1.0)
    assert out.loc[0, "a_hour_cos"] == pytest.approx(0.0, abs=1e-12)


def test_features_at_t_ignore_minute_t():
    """Issue #16: every feature at t uses only data before t."""
    panel = toy_panel()
    changed = panel.copy()
    changed.loc[5, ["spot_vwap", "spot_signed_volume", "spot_volume"]] = [999.0, 50.0, 60.0]
    before, after = block_a(panel, LAGS), block_a(changed, LAGS)
    pd.testing.assert_frame_equal(before.loc[:5], after.loc[:5])  # rows up to 5 are unchanged
    assert not before.loc[6].equals(after.loc[6])                  # row 6 is the first to see it


def test_not_enough_history_is_nan():
    out = block_a(toy_panel(), LAGS)
    assert out.loc[0:1, "a_ret_1"].isna().all()
    assert np.isnan(out.loc[5, "a_ret_5"])
    assert np.isnan(out.loc[5, "a_rv_5"])
    assert out["a_count_rel"].isna().all()  # needs 60 earlier minutes


def test_activity_relative_to_rolling_mean():
    out = block_a(toy_panel(rows=70), LAGS)
    assert np.isnan(out.loc[59, "a_count_rel"])
    assert out.loc[60, "a_count_rel"] == pytest.approx(1.0)
    assert out.loc[69, "a_vol_rel"] == pytest.approx(1.0)


def test_nan_after_a_gap_never_filled():
    out = block_a(toy_panel(rows=12, missing_rows=[4]), LAGS)
    assert out.loc[5:6, "a_ret_1"].isna().all()
    assert out.loc[7, "a_ret_1"] == pytest.approx(0.04 - 0.05)  # rows 5..6 are clean again
    assert out.loc[5:10, "a_ret_5"].isna().all()                # every window still contains row 4
    assert out.loc[5:9, "a_flow_5"].isna().all()
    assert out.loc[10, "a_flow_5"] == pytest.approx((-2 + 8 + 2 - 6 + 4) / 50)  # rows 5..9
    assert out.loc[5:10, "a_rv_5"].isna().all()


def test_columns_match_block_a_list():
    out = block_a(toy_panel(), LAGS)
    assert list(out.columns) == block_a_columns(LAGS)
    assert list(out.columns) == [
        "a_ret_1", "a_ret_5", "a_flow_1", "a_flow_5", "a_rv_5",
        "a_count_rel", "a_vol_rel", "a_hour_sin", "a_hour_cos",
    ]
    assert BLOCK_A == block_a_columns(load_config()["lags"])


def test_feature_lags_follow_the_horizon():
    cfg = {"horizon_minutes": 15, "lags": [1, 5, 15, 60]}
    assert feature_lags(15, cfg) == [1, 5, 15, 60]
    assert feature_lags(5, cfg) == [1, 2, 5, 20]
    real = load_config()
    assert feature_lags(real["horizon_minutes"], real) == real["lags"]


def test_features_hardcode_no_symbol_or_horizon():
    source = (REPO_ROOT / "src" / "features.py").read_text(encoding="utf-8")
    assert "BTC" not in source
    numbers = [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant)]
    assert 15 not in numbers