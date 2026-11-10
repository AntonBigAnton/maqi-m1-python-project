# raw → clean → 1-min panel
import json
import logging
from pathlib import Path

import pandas as pd

from src import fetch

log = logging.getLogger(__name__)

MS_PER_MINUTE = 60_000
# Epoch values above this are microseconds: 1e14 ms is in the year 5138, 1e14 µs is in 1973.
MICROSECOND_THRESHOLD = 10**14

# Column order of each archive CSV. Spot files have no header row, perp files do,
# so columns are always named by position and any header present is checked against this list.
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
METRICS_COLUMNS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]
# Perp files stop after is_buyer_maker; the missing column is read as empty and then dropped.
AGG_TRADE_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker", "is_best_match",
]

PRICE_COLUMNS = {"open": "float64", "high": "float64", "low": "float64", "close": "float64"}

# One clean schema per dataset: raw column order, output columns and dtypes,
# epoch columns to convert to UTC ms, the time column used to trim and sort, and the unique key.
SCHEMAS = {
    "klines": {
        "raw_columns": KLINE_COLUMNS,
        "columns": {
            "open_time": "int64", "close_time": "int64", **PRICE_COLUMNS,
            "volume": "float64", "quote_volume": "float64", "count": "int64",
            "taker_buy_volume": "float64", "taker_buy_quote_volume": "float64",
        },
        "epoch_columns": ["open_time", "close_time"],
        "time": "open_time",
        "key": "open_time",
    },
    # Same file layout as klines, but the volume and count columns are placeholders (zeros), so they are dropped.
    "premiumIndexKlines": {
        "raw_columns": KLINE_COLUMNS,
        "columns": {"open_time": "int64", "close_time": "int64", **PRICE_COLUMNS},
        "epoch_columns": ["open_time", "close_time"],
        "time": "open_time",
        "key": "open_time",
    },
    "metrics": {
        "raw_columns": METRICS_COLUMNS,
        "columns": {"create_time": "int64", **{name: "float64" for name in METRICS_COLUMNS[2:]}},
        "datetime_columns": ["create_time"],
        "time": "create_time",
        "key": "create_time",
    },
    "aggTrades": {
        "raw_columns": AGG_TRADE_COLUMNS,
        "columns": {
            "agg_trade_id": "int64", "price": "float64", "quantity": "float64",
            "first_trade_id": "int64", "last_trade_id": "int64",
            "transact_time": "int64", "is_buyer_maker": "bool",
        },
        "epoch_columns": ["transact_time"],
        "time": "transact_time",
        "key": "agg_trade_id",
    },
    "fundingRate": {
        "columns": {"funding_time": "int64", "funding_rate": "float64", "mark_price": "float64"},
        "epoch_columns": ["funding_time"],
        "time": "funding_time",
        "key": "funding_time",
    },
}


def clean_path(data_dir, market, dataset, symbol, start, end):
    """Parquet path of one clean dataset; named by window so --quick never overwrites a full run."""
    return Path(data_dir) / "clean" / market / dataset / f"{symbol}_{start}_{end}.parquet"


def panel_path(data_dir, symbol, start, end):
    """Parquet path of a symbol's 1-min panel."""
    return Path(data_dir) / "panel" / f"{symbol}_{start}_{end}.parquet"


def load_or_build(path, build):
    """Return the parquet at path, calling build() and saving the result on a miss."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    table = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(path.name + ".part")
    table.to_parquet(part_path)
    part_path.replace(path)
    return table


def to_milliseconds(values):
    """Convert epoch integers to UTC ms, value by value.

    Spot archive files switched from ms to µs on 2025-01-01, so a window spanning
    that date mixes both units; each value is converted on its own magnitude.
    """
    values = values.astype("int64")
    return values.where(values < MICROSECOND_THRESHOLD, values // 1000)


def datetime_to_milliseconds(values):
    """Convert UTC datetime strings such as '2026-01-01 00:05:00' to epoch ms."""
    return pd.to_datetime(values, utc=True, format="ISO8601").dt.as_unit("ms").astype("int64")


def read_archive_csv(path, raw_columns):
    """Read one archive CSV with the dataset's column names, whether or not it has a header row."""
    with open(path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    has_header = not first_line[:1].isdigit()
    if has_header:
        header = [name.strip().lower() for name in first_line.split(",")]
        if header != raw_columns[: len(header)]:
            raise ValueError(f"{path.name}: unexpected header {header}, expected {raw_columns}")
    return pd.read_csv(
        path,
        header=None,
        names=raw_columns,
        skiprows=1 if has_header else 0,
        true_values=["true", "True"],
        false_values=["false", "False"],
    )


def drop_duplicates(table, key, name):
    """Drop repeated rows; if two rows share a key but differ, keep the last one and warn."""
    n_rows = len(table)
    table = table.drop_duplicates()
    exact = n_rows - len(table)
    conflicting = table.duplicated(subset=key, keep="last")
    if conflicting.any():
        log.warning("%s: %d rows share a %s with different values; keeping the last", name, conflicting.sum(), key)
        table = table[~conflicting]
    if exact:
        log.info("%s: dropped %d duplicate rows", name, exact)
    return table


def check_minute_grid(table, name):
    """Fail if candles are not on whole minutes, the symptom of a timestamp unit bug."""
    off_grid = (table["open_time"] % MS_PER_MINUTE != 0) | (
        table["close_time"] != table["open_time"] + MS_PER_MINUTE - 1
    )
    if off_grid.any():
        example = table.loc[off_grid, ["open_time", "close_time"]].iloc[0].tolist()
        raise ValueError(f"{name}: {off_grid.sum()} candles are off the 1-minute grid, e.g. {example}")


def to_clean_table(raw, dataset, start, end, name):
    """Apply a dataset's schema to raw rows: UTC ms timestamps, types, window, no duplicates, sorted."""
    schema = SCHEMAS[dataset]
    table = raw.copy()
    for column in schema.get("epoch_columns", []):
        table[column] = to_milliseconds(table[column])
    for column in schema.get("datetime_columns", []):
        table[column] = datetime_to_milliseconds(table[column])
    table = table[list(schema["columns"])].astype(schema["columns"])

    # Monthly files extend past the window; keep only rows inside it.
    start_ms, end_ms = fetch.window_to_ms(start, end)
    table = table[table[schema["time"]].between(start_ms, end_ms)]

    table = drop_duplicates(table, schema["key"], name)
    table = table.sort_values(schema["time"]).reset_index(drop=True)
    if "open_time" in table and "close_time" in table:
        check_minute_grid(table, name)
    return table


def read_archive_dataset(cfg, spec, symbol, start, end):
    """Concatenate the extracted CSVs of one archive dataset; return None if none are on disk."""
    raw_columns = SCHEMAS[spec["dataset"]]["raw_columns"]
    frames = []
    for period in fetch.archive_periods(spec["frequency"], start, end):
        zip_path = fetch.archive_file_path(cfg, spec, symbol, period)
        csv_path = zip_path.with_suffix(".csv")
        if not (zip_path.exists() and csv_path.exists()):
            if spec["required"]:
                raise FileNotFoundError(f"{csv_path} is missing; run the fetch step first")
            log.warning("Skipping missing optional file %s", csv_path.name)
            continue
        frames.append(read_archive_csv(csv_path, raw_columns))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def read_funding(data_dir, symbol, start, end):
    """Load the cached funding JSON as a table with snake_case columns."""
    path = fetch.funding_path(data_dir, symbol, start, end)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the fetch step first")
    records = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(
        {
            "funding_time": records["fundingTime"],
            "funding_rate": pd.to_numeric(records["fundingRate"]),
            # Old records can have an empty mark price; keep them with a missing value.
            "mark_price": pd.to_numeric(records["markPrice"], errors="coerce"),
        }
    )


def clean_archive_dataset(cfg, spec, symbol, start, end):
    """Return one archive dataset's clean table, building and saving it on the first call."""
    start = spec.get("start_date", start)
    end = spec.get("end_date", end)
    name = f"{spec['market']} {spec['dataset']}"
    path = clean_path(cfg["data_dir"], spec["market"], spec["dataset"], symbol, start, end)
    if path.exists():
        return pd.read_parquet(path)
    raw = read_archive_dataset(cfg, spec, symbol, start, end)
    if raw is None:
        log.warning("%s: no files on disk, no clean table written", name)
        return None
    return load_or_build(path, lambda: to_clean_table(raw, spec["dataset"], start, end, name))


def clean_funding(cfg, symbol, start, end):
    """Return the clean funding history, building and saving it on the first call."""
    path = clean_path(cfg["data_dir"], "perp", "fundingRate", symbol, start, end)

    def build():
        raw = read_funding(cfg["data_dir"], symbol, start, end)
        return to_clean_table(raw, "fundingRate", start, end, "perp fundingRate")

    return load_or_build(path, build)


def clean_all(cfg, symbol, start, end):
    """Write one clean parquet per downloaded dataset; return a row-count summary."""
    tables = {}
    for spec in cfg["archive_datasets"]:
        table = clean_archive_dataset(cfg, spec, symbol, start, end)
        if table is not None:
            tables[(spec["market"], spec["dataset"])] = table
    tables[("perp", "fundingRate")] = clean_funding(cfg, symbol, start, end)

    summary = pd.DataFrame(
        [
            {
                "market": market,
                "dataset": dataset,
                "rows": len(table),
                "first_utc": pd.to_datetime(table[SCHEMAS[dataset]["time"]].min(), unit="ms", utc=True),
                "last_utc": pd.to_datetime(table[SCHEMAS[dataset]["time"]].max(), unit="ms", utc=True),
            }
            for (market, dataset), table in tables.items()
        ]
    )
    log.info("Clean tables:\n%s", summary.to_string(index=False))
    return summary
