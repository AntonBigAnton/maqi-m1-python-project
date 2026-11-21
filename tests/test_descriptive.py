# descriptive statistics: lead-lag (cross-correlation and Granger) on synthetic data
import numpy as np
import pandas as pd
import pytest

from src.descriptive import (
    cross_correlation, granger, granger_pair, intraday_profile, lead_lag_signals, lead_lag_table,
    missing_minutes, tracking_stats, volume_share,
)

MINUTE = 60_000
LEAD = 3  # x moves 3 minutes before y


def leading_pair(rows=5000, seed=0):
    """x is random; y at minute t is x at minute t - 3 plus noise, so x leads y by 3 minutes."""
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(size=rows))
    y = x.shift(LEAD) + 0.5 * pd.Series(rng.normal(size=rows))
    return x, y


def test_cross_correlation_peaks_at_the_lead():
    x, y = leading_pair()
    table = cross_correlation(x, y, max_lag=10)
    best = table.loc[table["corr"].idxmax()]
    assert best["lag"] == LEAD
    assert best["corr"] > best["band"]


def test_swapping_the_series_moves_the_peak_to_minus_the_lead():
    x, y = leading_pair()
    table = cross_correlation(y, x, max_lag=10)
    assert table.loc[table["corr"].idxmax(), "lag"] == -LEAD


def test_band_is_1_96_over_root_n():
    x, y = leading_pair()
    table = cross_correlation(x, y, max_lag=2)
    assert list(table["lag"]) == [-2, -1, 0, 1, 2]
    assert np.allclose(table["band"], 1.96 / np.sqrt(table["n"]))


def test_granger_finds_the_leader_and_reports_the_effect_size():
    x, y = leading_pair()
    forward = granger_pair(x, y, max_lag=5)
    backward = granger_pair(y, x, max_lag=5)
    assert forward["p_value"] < 0.01
    assert forward["delta_r2"] > 0.5  # x's past explains most of y
    assert backward["delta_r2"] < 0.01  # y's past adds almost nothing to x


def synthetic_panel(rows=600, seed=0):
    """A panel where the perp leads spot by 3 minutes."""
    rng = np.random.default_rng(seed)
    perp_ret = rng.normal(0, 1e-3, rows)
    spot_ret = np.roll(perp_ret, LEAD) + rng.normal(0, 2e-4, rows)
    return pd.DataFrame({
        "open_time": np.arange(rows) * MINUTE,
        "perp_vwap": 100 * np.exp(np.cumsum(perp_ret)),
        "spot_vwap": 100 * np.exp(np.cumsum(spot_ret)),
        "perp_signed_volume": rng.uniform(-1, 1, rows),
        "perp_volume": 2.0,
    })


def test_lead_lag_table_covers_every_perp_signal():
    table = lead_lag_table(synthetic_panel(), max_lag=5)
    assert set(table["signal"]) == {"perp_ret", "perp_flow", "basis_bps"}
    perp = table[table["signal"] == "perp_ret"]
    assert perp.loc[perp["corr"].idxmax(), "lag"] == LEAD


def test_granger_runs_both_directions():
    table = granger(synthetic_panel(), max_lag=5)
    assert len(table) == 6
    row = table[(table["cause"] == "perp_ret") & (table["effect"] == "spot_ret")].iloc[0]
    assert row["p_value"] < 0.01


def test_missing_minute_stays_missing():
    panel = synthetic_panel(rows=10)
    panel.loc[4, ["perp_vwap", "spot_vwap"]] = np.nan
    signals = lead_lag_signals(panel)
    assert signals.loc[4:5, ["spot_ret", "perp_ret"]].isna().all().all()  # returns into and out of minute 4
    assert np.isnan(signals.loc[4, "basis_bps"]) and not np.isnan(signals.loc[5, "basis_bps"])


# ---------- Blocks A-B: data quality and market structure (#20) ----------

JAN_1_MS = 1767225600000  # 2026-01-01 00:00 UTC


def market_panel(days=2, seed=0):
    """A complete 1-minute spot/perp panel from 2026-01-01: random walk prices, perp 5 bps above spot."""
    rng = np.random.default_rng(seed)
    rows = days * 1440
    spot = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, rows)))
    perp = spot * 1.0005
    return pd.DataFrame({
        "open_time": JAN_1_MS + np.arange(rows) * MINUTE,
        "spot_close": spot, "perp_close": perp,
        "spot_vwap": spot, "perp_vwap": perp,
        "spot_volume": 1.0, "perp_volume": 3.0,
        "spot_quote_volume": spot * 1.0, "perp_quote_volume": perp * 3.0,
        "spot_signed_volume": rng.uniform(-1, 1, rows), "perp_signed_volume": rng.uniform(-3, 3, rows),
        "gap": False,
    })


def test_missing_minutes_counts_gaps_by_day_and_hour():
    panel = market_panel()
    panel.loc[1440 + 3 * 60 + 10: 1440 + 3 * 60 + 14, "gap"] = True  # 5 minutes on Jan 2, 03:10-03:14
    table = missing_minutes(panel)
    assert table.shape == (2, 24)
    assert table.iloc[1, 3] == 5
    assert table.to_numpy().sum() == 5


def test_tracking_stats_in_bps():
    stats = tracking_stats(market_panel()).iloc[0]
    assert stats["median_abs_bps"] == pytest.approx(5.0)
    assert stats["minutes"] == 2 * 1440


def test_volume_share_is_perp_over_total():
    share = volume_share(market_panel())
    assert len(share) == 2
    # perp trades 3 units at a price 5 bps above spot's 1 unit
    assert np.allclose(share["perp_share"], 3 * 1.0005 / (3 * 1.0005 + 1))


def test_intraday_profile_has_one_row_per_hour():
    profile = intraday_profile(market_panel())
    assert list(profile.index) == list(range(24))
    assert (profile["spot_vol_bps"] > 0).all()