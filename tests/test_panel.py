# small checks: minute alignment, VWAP and signed volume, gap mask, as-of joins never look ahead
import numpy as np
import pandas as pd
import pytest

from src import clean, schemas
from src.panel import build_panel, check_spot_perp_tracking, drop_unknown_start, make_panel, panel_path

START = END = "2026-01-01"
JAN_1_MS = 1767225600000
MINUTE = 60_000


def candles(minutes, close=100.0, volume=2.0, taker_buy_volume=1.5):
    """A clean klines table with one candle per listed minute of 2026-01-01."""
    open_time = [JAN_1_MS + m * MINUTE for m in minutes]
    return pd.DataFrame(
        {
            "open_time": open_time,
            "close_time": [t + MINUTE - 1 for t in open_time],
            "open": close, "high": close, "low": close, "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "count": 5,
            "taker_buy_volume": taker_buy_volume,
            "taker_buy_quote_volume": taker_buy_volume * close,
        }
    )


def metrics(minutes, values):
    return pd.DataFrame(
        {
            "create_time": [JAN_1_MS + m * MINUTE for m in minutes],
            **{name: values for name in schemas.METRICS_COLUMNS[2:]},
        }
    )


def funding(times_ms, rates):
    return pd.DataFrame({"funding_time": times_ms, "funding_rate": rates, "mark_price": 100.0})


NO_FUNDING = funding([JAN_1_MS - 8 * 60 * MINUTE], [0.0001])


def row(panel, minute):
    return panel.loc[panel["open_time"] == JAN_1_MS + minute * MINUTE].iloc[0]


def test_every_minute_is_a_row_and_gaps_are_flagged_not_filled():
    panel = make_panel(START, END, candles([0, 1, 2]), candles([0, 2]), NO_FUNDING)
    assert len(panel) == 1440
    assert panel["open_time"].diff().dropna().eq(MINUTE).all()
    assert row(panel, 1)["perp_missing"] and row(panel, 1)["gap"]
    assert np.isnan(row(panel, 1)["perp_close"])
    assert not row(panel, 2)["gap"]
    assert row(panel, 3)["spot_missing"] and row(panel, 3)["perp_missing"]
    assert panel["perp_count"].dtype == "Int64"


def test_vwap_and_signed_volume_per_market():
    spot = pd.concat([candles([0], close=100.0), candles([1], volume=0.0, taker_buy_volume=0.0)])
    panel = make_panel(START, END, spot, candles([0], close=101.0), NO_FUNDING)
    assert row(panel, 0)["spot_vwap"] == 100.0
    assert row(panel, 0)["perp_vwap"] == 101.0
    assert row(panel, 0)["spot_signed_volume"] == 2 * 1.5 - 2.0
    assert np.isnan(row(panel, 1)["spot_vwap"])


def test_premium_index_is_aligned_on_the_same_minute():
    premium = candles([0, 1])[["open_time", "close_time", "open", "high", "low", "close"]]
    premium["close"] = [0.0001, 0.0002]
    panel = make_panel(START, END, candles([0, 1]), candles([0, 1]), NO_FUNDING, premium=premium)
    assert row(panel, 1)["premium_index"] == 0.0002


def test_metrics_are_used_only_once_known():
    panel = make_panel(
        START, END, candles([0]), candles([0]), NO_FUNDING,
        metrics=metrics([0, 5], [1.0, 2.0]), metrics_known_after_minutes=5,
    )
    assert np.isnan(row(panel, 4)["sum_open_interest"])
    assert row(panel, 5)["sum_open_interest"] == 1.0
    assert row(panel, 9)["sum_open_interest"] == 1.0
    assert row(panel, 10)["sum_open_interest"] == 2.0
    assert row(panel, 10)["metrics_time"] == JAN_1_MS + 5 * MINUTE


def test_stale_metrics_are_left_missing():
    panel = make_panel(
        START, END, candles([0]), candles([0]), NO_FUNDING,
        metrics=metrics([0], [1.0]), metrics_known_after_minutes=5, metrics_max_age_minutes=15,
    )
    assert row(panel, 20)["sum_open_interest"] == 1.0
    assert np.isnan(row(panel, 21)["sum_open_interest"])


def test_funding_settled_a_few_ms_after_a_minute_starts_appears_the_next_minute():
    settled = funding([JAN_1_MS - 8 * 60 * MINUTE, JAN_1_MS + 8], [0.0001, 0.0003])
    panel = make_panel(START, END, candles([0]), candles([0]), settled)
    assert row(panel, 0)["funding_rate"] == 0.0001
    assert row(panel, 1)["funding_rate"] == 0.0003


def test_changing_data_after_t_never_changes_the_panel_up_to_t():
    rng = np.random.default_rng(0)
    minutes = range(0, 1440, 5)
    full_metrics = metrics(minutes, rng.normal(size=len(minutes)))
    full_funding = funding([JAN_1_MS + h * 60 * MINUTE + 8 for h in (0, 8, 16)], [0.1, 0.2, 0.3])
    kwargs = {"metrics_known_after_minutes": 5, "metrics_max_age_minutes": 15}
    panel = make_panel(START, END, candles(range(1440)), candles(range(1440)), full_funding,
                             metrics=full_metrics, **kwargs)

    cutoff = JAN_1_MS + 600 * MINUTE
    changed_metrics = full_metrics.copy()
    changed_metrics.loc[changed_metrics["create_time"] + 5 * MINUTE > cutoff, "sum_open_interest"] = 999.0
    changed_funding = full_funding.copy()
    changed_funding.loc[changed_funding["funding_time"] > cutoff, "funding_rate"] = 999.0
    changed = make_panel(START, END, candles(range(1440)), candles(range(1440)), changed_funding,
                               metrics=changed_metrics, **kwargs)

    up_to_cutoff = panel["open_time"] <= cutoff
    pd.testing.assert_frame_equal(panel[up_to_cutoff], changed[up_to_cutoff])
    for time_column, delay in (("metrics_time", 5 * MINUTE), ("funding_time", 0)):
        joined = panel.dropna(subset=[time_column])
        assert len(joined) > 0
        assert (joined[time_column] + delay <= joined["open_time"]).all()


def test_missing_optional_datasets_keep_the_same_columns():
    full = make_panel(
        START, END, candles([0]), candles([0]), NO_FUNDING,
        premium=candles([0])[["open_time", "close_time", "open", "high", "low", "close"]],
        metrics=metrics([0], [1.0]),
    )
    bare = make_panel(START, END, candles([0]), candles([0]), NO_FUNDING)
    assert list(bare.columns) == list(full.columns)
    assert bare["premium_index"].isna().all() and bare["sum_open_interest"].isna().all()


def test_minutes_before_metrics_and_funding_are_known_are_dropped():
    settled = funding([JAN_1_MS + 2 * MINUTE], [0.0001])
    panel = make_panel(
        START, END, candles(range(10)), candles(range(10)), settled,
        metrics=metrics([0, 5], [1.0, 2.0]), metrics_known_after_minutes=5,
    )
    trimmed = drop_unknown_start(panel)
    assert trimmed["open_time"].iloc[0] == JAN_1_MS + 5 * MINUTE
    assert trimmed[["metrics_time", "funding_time"]].notna().all().all()
    assert trimmed.index[0] == 0


def test_later_missing_readings_are_kept():
    panel = make_panel(
        START, END, candles([0]), candles([0]), NO_FUNDING,
        metrics=metrics([0], [1.0]), metrics_known_after_minutes=5, metrics_max_age_minutes=15,
    )
    trimmed = drop_unknown_start(panel)
    assert trimmed["open_time"].iloc[0] == JAN_1_MS + 5 * MINUTE
    assert len(trimmed) == 1440 - 5
    assert trimmed["sum_open_interest"].isna().any()


def test_a_dataset_that_was_not_downloaded_does_not_empty_the_panel():
    panel = make_panel(START, END, candles([0]), candles([0]), NO_FUNDING)
    assert len(drop_unknown_start(panel)) == 1440


def test_spot_and_perp_that_do_not_track_stop_the_run():
    panel = make_panel(START, END, candles([0]), candles([0], close=100_000.0), NO_FUNDING)
    with pytest.raises(ValueError, match="median"):
        check_spot_perp_tracking(panel)
    check_spot_perp_tracking(make_panel(START, END, candles([0]), candles([0], close=100.05), NO_FUNDING))


def test_build_panel_reads_clean_tables_and_saves_one_parquet(tmp_path):
    spot_spec = {"dataset": "klines", "market": "spot", "frequency": "monthly", "interval": "1m", "required": True}
    perp_spec = {**spot_spec, "market": "perp"}
    metrics_spec = {"dataset": "metrics", "market": "perp", "frequency": "daily", "required": False,
                    "known_after_minutes": 5, "max_age_minutes": 15}
    cfg = {"data_dir": tmp_path, "archive_datasets": [spot_spec, perp_spec, metrics_spec]}
    tables = {
        ("spot", "klines"): candles(range(10)),
        ("perp", "klines"): candles(range(10)),
        ("perp", "metrics"): metrics([0], [1.0]),
        ("perp", "fundingRate"): NO_FUNDING,
    }
    for (market, dataset), table in tables.items():
        path = clean.clean_path(tmp_path, market, dataset, "BTCUSDT", START, END)
        path.parent.mkdir(parents=True)
        table.to_parquet(path)

    panel = build_panel(cfg, "BTCUSDT", START, END)

    assert panel_path(tmp_path, "BTCUSDT", START, END).exists()
    # Metrics stamped 00:00 are known at 00:05, so the panel starts there.
    assert panel["open_time"].iloc[0] == JAN_1_MS + 5 * MINUTE
    assert row(panel, 5)["sum_open_interest"] == 1.0
    assert len(panel) == 1440 - 5
    assert panel["gap"].sum() == 1440 - 10
