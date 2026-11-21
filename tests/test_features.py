# block A and B features: hand-checked values, no look-ahead, NaN after a gap, nothing hardcoded
import ast
import math

import numpy as np
import pandas as pd
import pytest

from src.config import REPO_ROOT, load_config
from src.features import (
    BLOCK_A, BLOCK_B, block_a, block_a_columns, block_b, block_b_columns, build_features, feature_lags,
)

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


# ---------- Block B ----------

B_LAGS = [1, 5, 15]  # block B keeps the middle lag only: 5


def perp_toy_panel():
    """toy_panel plus perp candles, metrics and funding, with values easy to check by hand."""
    panel = toy_panel()
    # perp trades 5 bps above spot, 10 bps on row 3
    premium = [1.0005] * 10
    premium[3] = 1.001
    panel["perp_vwap"] = panel["spot_vwap"] * premium
    panel["perp_signed_volume"] = 6.0
    panel["perp_volume"] = 20.0
    # two metrics readings: rows 0-4 and rows 5-9; open interest goes from 1000 to 1010
    panel["metrics_time"] = pd.array([JAN_1_6AM_MS] * 5 + [JAN_1_6AM_MS + 5 * MINUTE] * 5, dtype="Int64")
    panel["sum_open_interest"] = [1000.0] * 5 + [1010.0] * 5
    panel["sum_toptrader_long_short_ratio"] = 2.0
    panel["count_long_short_ratio"] = 0.5
    panel["sum_taker_long_short_vol_ratio"] = 1.0
    panel["funding_rate"] = 0.0001
    return panel


def test_block_b_hand_checked_values():
    out = block_b(perp_toy_panel(), B_LAGS)
    assert out.loc[1, "b_basis_bps"] == pytest.approx(5.0)    # row 0
    assert out.loc[4, "b_basis_bps"] == pytest.approx(10.0)   # row 3
    assert out.loc[9, "b_basis_chg_5"] == pytest.approx(5.0 - 10.0)  # row 8 minus row 3
    assert out.loc[5, "b_flow_5"] == pytest.approx(30 / 100)
    assert out.loc[5, "b_flow_div_5"] == pytest.approx(30 / 100 - (2 - 4 + 6 + 0 + 4) / 50)
    assert out.loc[6, "b_lead_5"] == pytest.approx(0.0)       # same premium at rows 0 and 5
    assert out.loc[9, "b_lead_5"] == pytest.approx(math.log(1.0005) - math.log(1.001))  # rows 3..8
    assert out.loc[0, "b_toptrader_ls"] == pytest.approx(math.log(2.0))
    assert out.loc[0, "b_ls"] == pytest.approx(math.log(0.5))
    assert out.loc[0, "b_taker_ratio"] == pytest.approx(0.0)
    assert out.loc[0, "b_funding_bps"] == pytest.approx(1.0)


def test_open_interest_change_only_between_readings():
    out = block_b(perp_toy_panel(), B_LAGS)
    assert out.loc[0:4, "b_oi_chg"].isna().all()  # first reading: nothing to compare with
    assert np.allclose(out.loc[5:9, "b_oi_chg"], 1.0)  # 1000 -> 1010 is +1%, on every minute of the reading


def test_open_interest_change_is_empty_after_a_missing_reading():
    panel = perp_toy_panel()
    panel.loc[5:9, "metrics_time"] = JAN_1_6AM_MS + 10 * MINUTE  # the reading at +5 min never came
    out = block_b(panel, B_LAGS)
    assert out.loc[5:9, "b_oi_chg"].isna().all()


def test_block_b_columns_and_size():
    out = block_b(perp_toy_panel(), B_LAGS)
    assert list(out.columns) == block_b_columns(B_LAGS)
    assert BLOCK_B == block_b_columns(load_config()["lags"])
    assert len(BLOCK_A) <= 15 and len(BLOCK_B) <= 15
    assert not set(BLOCK_A) & set(BLOCK_B)


def random_panel(rows, seed):
    """A panel with every column block A and B read, filled with random positive values."""
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame({"open_time": [JAN_1_6AM_MS + i * MINUTE for i in range(rows)]})
    for column in [
        "spot_vwap", "spot_volume", "perp_vwap", "perp_volume", "sum_open_interest",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ]:
        panel[column] = rng.uniform(1, 2, rows)
    panel["spot_signed_volume"] = rng.uniform(-1, 1, rows)
    panel["perp_signed_volume"] = rng.uniform(-1, 1, rows)
    panel["spot_count"] = pd.array(rng.integers(1, 10, rows), dtype="Int64")
    panel["metrics_time"] = pd.array([JAN_1_6AM_MS + (i // 5) * 5 * MINUTE for i in range(rows)], dtype="Int64") # a new metrics reading every 5 minutes
    panel["funding_rate"] = rng.uniform(-0.001, 0.001, rows)
    panel["spot_missing"] = False
    return panel


# Candle columns of row t are only complete at t + 1, so they are changed from row t on.
# Metrics and funding on row t are already known at t, so they are changed after row t only.
CANDLE_COLUMNS = [
    "spot_vwap", "spot_volume", "spot_signed_volume", "spot_count",
    "perp_vwap", "perp_volume", "perp_signed_volume",
]
KNOWN_AT_T_COLUMNS = [
    "open_time", "metrics_time", "sum_open_interest", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio", "funding_rate",
]


def randomise(panel, column, first_row, rng):
    """Replace a column with random values from first_row to the end."""
    rows = len(panel) - first_row
    if column in ("open_time", "metrics_time", "spot_count"):
        panel.loc[first_row:, column] = rng.integers(1, 10**9, rows)
    else:
        panel.loc[first_row:, column] = rng.uniform(1, 2, rows)


@pytest.mark.parametrize("t", [30, 70, 100])
def test_changing_data_after_t_never_changes_features_up_to_t(t):
    """Issue #19: features up to row t are identical whatever happens later, for both blocks."""
    cfg = load_config()
    panel = random_panel(rows=120, seed=0)
    changed = panel.copy()
    rng = np.random.default_rng(1)
    for column in CANDLE_COLUMNS:
        randomise(changed, column, t, rng)
    for column in KNOWN_AT_T_COLUMNS:
        randomise(changed, column, t + 1, rng)

    columns = BLOCK_A + BLOCK_B
    before = build_features(panel, cfg)[columns]
    after = build_features(changed, cfg)[columns]
    pd.testing.assert_frame_equal(before.loc[:t], after.loc[:t])
    assert not before.loc[t + 1:].equals(after.loc[t + 1:])  # the change did reach later rows
