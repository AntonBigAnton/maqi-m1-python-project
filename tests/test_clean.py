# small checks: header and unit unification, window trimming, duplicates, one parquet per dataset
import json

import pandas as pd
import pytest

from src import clean, fetch

SPOT_KLINES = {"dataset": "klines", "market": "spot", "frequency": "monthly", "interval": "1m", "required": True}
PERP_KLINES = {"dataset": "klines", "market": "perp", "frequency": "monthly", "interval": "1m", "required": True}
PREMIUM = {"dataset": "premiumIndexKlines", "market": "perp", "frequency": "monthly", "interval": "1m", "required": True}
METRICS = {"dataset": "metrics", "market": "perp", "frequency": "daily", "required": False}

# 2026-01-01 00:00 UTC and the last minute of 2025
JAN_1_MS = 1767225600000
DEC_31_LAST_MS = JAN_1_MS - 60_000

PERP_HEADER = ",".join(clean.KLINE_COLUMNS)
METRICS_HEADER = ",".join(clean.METRICS_COLUMNS)


def make_cfg(tmp_path, datasets):
    return {
        "data_dir": tmp_path,
        "archive_url": "https://archive.test/data",
        "markets": {"spot": {"archive_path": "spot"}, "perp": {"archive_path": "futures/um"}},
        "archive_datasets": datasets,
    }


def kline_row(open_ms, unit=1):
    """One raw kline line; unit=1000 writes the timestamps in microseconds like spot files since 2025."""
    close_ms = open_ms + 59_999
    return f"{open_ms * unit},100.0,101.0,99.0,100.5,2.0,{close_ms * unit + unit - 1},201.0,7,1.0,100.5,0"


def write_raw(cfg, spec, period, lines):
    """Place a CSV and its zip marker where the fetch step leaves them."""
    zip_path = fetch.archive_file_path(cfg, spec, "BTCUSDT", period)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.touch()
    zip_path.with_suffix(".csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_to_milliseconds_converts_only_microsecond_values():
    values = pd.Series([JAN_1_MS, JAN_1_MS * 1000, JAN_1_MS * 1000 + 59_999_999])
    assert clean.to_milliseconds(values).tolist() == [JAN_1_MS, JAN_1_MS, JAN_1_MS + 59_999]


def test_spot_and_perp_klines_share_one_schema_and_unit(tmp_path):
    cfg = make_cfg(tmp_path, [SPOT_KLINES, PERP_KLINES])
    write_raw(cfg, SPOT_KLINES, "2026-01", [kline_row(JAN_1_MS, unit=1000), kline_row(JAN_1_MS + 60_000, unit=1000)])
    write_raw(cfg, PERP_KLINES, "2026-01", [PERP_HEADER, kline_row(JAN_1_MS), kline_row(JAN_1_MS + 60_000)])

    spot = clean.clean_archive_dataset(cfg, SPOT_KLINES, "BTCUSDT", "2026-01-01", "2026-01-31")
    perp = clean.clean_archive_dataset(cfg, PERP_KLINES, "BTCUSDT", "2026-01-01", "2026-01-31")

    assert spot.dtypes.equals(perp.dtypes)
    assert spot["open_time"].tolist() == perp["open_time"].tolist() == [JAN_1_MS, JAN_1_MS + 60_000]
    assert spot["close_time"].iloc[0] == JAN_1_MS + 59_999
    assert spot["count"].dtype == "int64"
    assert "ignore" not in spot


def test_klines_are_trimmed_deduplicated_and_sorted(tmp_path):
    cfg = make_cfg(tmp_path, [PERP_KLINES])
    write_raw(cfg, PERP_KLINES, "2026-01", [
        PERP_HEADER,
        kline_row(JAN_1_MS + 60_000),
        kline_row(JAN_1_MS),
        kline_row(JAN_1_MS),
        kline_row(DEC_31_LAST_MS),
    ])
    table = clean.clean_archive_dataset(cfg, PERP_KLINES, "BTCUSDT", "2026-01-01", "2026-01-31")
    assert table["open_time"].tolist() == [JAN_1_MS, JAN_1_MS + 60_000]


def test_conflicting_duplicates_keep_the_last_row():
    raw = pd.DataFrame(
        {"funding_time": [JAN_1_MS, JAN_1_MS], "funding_rate": [0.0001, 0.0002], "mark_price": [1.0, 1.0]}
    )
    table = clean.to_clean_table(raw, "fundingRate", "2026-01-01", "2026-01-31", "test")
    assert table["funding_rate"].tolist() == [0.0002]


def test_unexpected_header_stops_the_run(tmp_path):
    cfg = make_cfg(tmp_path, [PERP_KLINES])
    write_raw(cfg, PERP_KLINES, "2026-01", ["open_time,open,high,low,price", kline_row(JAN_1_MS)])
    with pytest.raises(ValueError, match="unexpected header"):
        clean.clean_archive_dataset(cfg, PERP_KLINES, "BTCUSDT", "2026-01-01", "2026-01-31")


def test_candles_off_the_minute_grid_stop_the_run():
    raw = pd.DataFrame([kline_row(JAN_1_MS + 1).split(",")], columns=clean.KLINE_COLUMNS)
    with pytest.raises(ValueError, match="off the 1-minute grid"):
        clean.to_clean_table(raw, "klines", "2026-01-01", "2026-01-31", "test")


def test_premium_index_keeps_only_prices(tmp_path):
    cfg = make_cfg(tmp_path, [PREMIUM])
    write_raw(cfg, PREMIUM, "2026-01", [PERP_HEADER, kline_row(JAN_1_MS)])
    table = clean.clean_archive_dataset(cfg, PREMIUM, "BTCUSDT", "2026-01-01", "2026-01-31")
    assert list(table) == ["open_time", "close_time", "open", "high", "low", "close"]


def test_metrics_times_become_utc_milliseconds(tmp_path):
    cfg = make_cfg(tmp_path, [METRICS])
    write_raw(cfg, METRICS, "2026-01-01", [
        METRICS_HEADER,
        "2026-01-01 00:05:00,BTCUSDT,2.0,3.0,1.1,1.2,1.3,0.9",
        "2026-01-01 00:00:00,BTCUSDT,1.0,2.0,1.1,1.2,1.3,0.9",
    ])
    table = clean.clean_archive_dataset(cfg, METRICS, "BTCUSDT", "2026-01-01", "2026-01-01")
    assert table["create_time"].tolist() == [JAN_1_MS, JAN_1_MS + 5 * 60_000]
    assert "symbol" not in table
    assert table["sum_open_interest"].dtype == "float64"


def test_missing_optional_files_are_skipped_and_required_ones_raise(tmp_path):
    cfg = make_cfg(tmp_path, [METRICS, PERP_KLINES])
    assert clean.clean_archive_dataset(cfg, METRICS, "BTCUSDT", "2026-01-01", "2026-01-02") is None
    with pytest.raises(FileNotFoundError):
        clean.clean_archive_dataset(cfg, PERP_KLINES, "BTCUSDT", "2026-01-01", "2026-01-31")


def test_clean_all_writes_one_parquet_per_dataset(tmp_path):
    cfg = make_cfg(tmp_path, [SPOT_KLINES, PERP_KLINES])
    write_raw(cfg, SPOT_KLINES, "2026-01", [kline_row(JAN_1_MS, unit=1000)])
    write_raw(cfg, PERP_KLINES, "2026-01", [PERP_HEADER, kline_row(JAN_1_MS)])
    funding = [{"symbol": "BTCUSDT", "fundingTime": JAN_1_MS + 8, "fundingRate": "0.0001", "markPrice": ""}]
    funding_path = fetch.funding_path(tmp_path, "BTCUSDT", "2026-01-01", "2026-01-31")
    funding_path.parent.mkdir(parents=True)
    funding_path.write_text(json.dumps(funding), encoding="utf-8")

    summary = clean.clean_all(cfg, "BTCUSDT", "2026-01-01", "2026-01-31")

    assert summary["rows"].tolist() == [1, 1, 1]
    written = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.glob("clean/**/*.parquet"))
    assert written == [
        "clean/perp/fundingRate/BTCUSDT_2026-01-01_2026-01-31.parquet",
        "clean/perp/klines/BTCUSDT_2026-01-01_2026-01-31.parquet",
        "clean/spot/klines/BTCUSDT_2026-01-01_2026-01-31.parquet",
    ]
    funding_table = pd.read_parquet(tmp_path / written[0])
    assert funding_table["mark_price"].isna().all()
