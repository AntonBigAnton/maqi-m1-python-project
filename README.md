# maqi-m1-python-project
Python For Data Science project

## Install

Requires Python 3.11 or newer. From the repo root:

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install .` is enough to run the pipeline; the `dev` extra adds `pytest` and `nb-clean`.

## Run

```
python -m src.pipeline
```

Add `--quick` to run on one month, or `--config path/to/config.yaml` to use another symbol, window or data folder.
All data comes from public Binance URLs (no API key) and is stored in the gitignored `data/` folder. Files already there are not downloaded again.

## Data collection size and runtime

Measured for BTCUSDT, 2026-01-01 to 2026-06-30, on one Windows laptop:

| | |
| --- | --- |
| Download size | 31 MB |
| Size on disk (zips and extracted CSVs) | 131 MB |
| First run | about 4.5 minutes |
| Re-run with data already present | about 3 seconds |

Every download is recorded in `data/raw/download_log.csv` (file, URL, size, SHA-256, checksum status).

Before downloading, the pipeline checks that spot and perp candles cover the whole sample window and stops with an error if they do not. The result is saved to `data/availability_<symbol>.csv` for the report.

## Datasets: coverage and timing

| Dataset | Market | Resolution | Timestamp | Known at |
| --- | --- | --- | --- | --- |
| `klines` (1m) | spot, perp | 1 min | `open_time` = start of the minute | end of the minute |
| `premiumIndexKlines` (1m) | perp | 1 min | `open_time` = start of the minute | end of the minute |
| `metrics` | perp | 5 min | `create_time` = start of the 5-minute window | `create_time` + 5 min |
| funding rate (REST) | perp | 8 h | `fundingTime` = settlement time | `fundingTime` |

**`metrics` coverage.** All 181 daily files for 2026-01-01 to 2026-06-30 are published, each with 288 rows (one every 5 minutes, 00:00 to 23:55 UTC) and no missing values (checked 2026-11-07). Columns: `sum_open_interest`, `sum_open_interest_value`, `count_toptrader_long_short_ratio`, `sum_toptrader_long_short_ratio`, `count_long_short_ratio` and `sum_taker_long_short_vol_ratio`.

**What `create_time` means.** We rebuilt `sum_taker_long_short_vol_ratio` (taker buy volume over taker sell volume) from 1-minute perp klines for 2026-01-01. It matches the 5 minutes *starting* at `create_time` (correlation 0.995, median relative error 3%). The 5 minutes ending at `create_time` don't match (correlation 0.04). So a row stamped 12:00 describes 12:00–12:05 and is only known at 12:05. The panel joins `metrics` as of `create_time + known_after_minutes` (5, set in `config.yaml`), so no feature uses a value before it exists. Open interest is a snapshot that we couldn't check the same way, so it gets the same conservative 5-minute delay.

## Configuration

Every run parameter (symbol, horizon, sample window, lags, seed, paths) lives in `config.yaml`. Code reads it with `src.config.load_config()`, so nothing in `src/` hardcodes the coin, the horizon or the dates.

## Contributing

- One branch per task, merged into `main` through a pull request with one approving review. Direct pushes to `main` are blocked.
- Run `pytest` before opening a pull request.
- Strip notebook outputs before committing. Run this once per clone to do it automatically:

  ```
  nb-clean add-filter --remove-empty-cells
  ```