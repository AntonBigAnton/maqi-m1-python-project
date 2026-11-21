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

This downloads the data, cleans it into one table per dataset and builds the 1-minute spot/perp panel (see [Cleaning and the 1-minute panel](#cleaning-and-the-1-minute-panel)).

Add `--quick` to run on one month, or `--config path/to/config.yaml` to use another symbol, window or data folder.
All data comes from public Binance URLs (no API key) and is stored in the gitignored `data/` folder. Files already there are not downloaded again, and clean tables and panels already there are not rebuilt: after changing the cleaning or panel code, delete `data/clean/` and `data/panel/` to rebuild them.

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

## Cleaning and the 1-minute panel

The data goes through three layers, each saved in `data/` so later steps never redo earlier ones:

| Layer | Code | Output |
| --- | --- | --- |
| Raw | `src/fetch.py` | `data/raw/`: files exactly as downloaded |
| Clean | `src/clean.py` | `data/clean/<market>/<dataset>/<symbol>_<start>_<end>.parquet`: one typed table per dataset |
| Panel | `src/panel.py` | `data/panel/<symbol>_<start>_<end>.parquet`: one row per minute, spot and perp side by side |

Files are named by symbol and window, so a `--quick` run never overwrites a full one. The column layouts of the Binance files and of the clean tables are in `src/schemas.py`.

### Cleaning choices

- **Headers.** Spot archive files have no header row and perp files do. Columns are always named by position, and a header that does not match the expected layout stops the run.
- **Timestamps.** Every timestamp is stored as UTC milliseconds (`int64`). Spot files switched from milliseconds to microseconds on 2025-01-01; each value is converted on its own magnitude, so a window spanning that date works. `metrics` datetime strings and funding times are converted to the same unit.
- **Window.** Monthly files extend past the sample window; rows outside it are dropped.
- **Duplicates.** Repeated rows are dropped. If two rows share a timestamp but differ, the last one is kept and a warning is logged.
- **Types.** Prices, volumes and ratios are `float64`, trade counts `int64`. The unused `ignore` and `symbol` columns are dropped, and so are the volume and count columns of `premiumIndexKlines`, which are placeholders.
- **Checks.** A candle not on a whole minute stops the run, as it would mean a timestamp unit bug.

### Panel

Row `open_time = t` holds the candles of the minute starting at *t*. Columns:

| Columns | Content |
| --- | --- |
| `open_time` | Start of the minute, UTC ms |
| `spot_*`, `perp_*` | Each market's candle (`open`, `high`, `low`, `close`, `volume`, `quote_volume`, `count`, `taker_buy_volume`, `taker_buy_quote_volume`), plus `vwap` (quote volume / volume, missing when nothing traded) and `signed_volume` (2 × taker buy volume − volume) |
| `premium_index` | Close of the perp premium index for the same minute |
| `metrics_time` and the six `metrics` columns | Last `metrics` reading known at or before *t* |
| `funding_time`, `funding_rate` | Last funding rate settled at or before *t* |
| `spot_missing`, `perp_missing`, `gap` | Gap mask: the market had no candle that minute (`gap` = either) |

- **No look-ahead.** `metrics` and funding are joined backward, as of the time each value is known (see [the timing table above](#datasets-coverage-and-timing)), never forward. `metrics_time` and `funding_time` are kept so this can be checked.
- **No silent filling.** Every minute of the window is a row. A missing minute stays empty and is flagged in the gap mask. A `metrics` reading older than `max_age_minutes` (15, in `config.yaml`) is left missing rather than carried through an outage.
- **Start of the window.** The panel starts at the first minute where `metrics` and funding are both known: 00:05 on the first day, since the first `metrics` reading is only known 5 minutes in.
- **Same columns on every run.** If an optional dataset was not downloaded, its columns are still there, left empty.
- **Check.** The run stops if the median gap between perp and spot closes is above 1%, which would mean a join or unit bug (for BTCUSDT in January 2026 it is 4.5 bps).

## Features

`build_features(panel, cfg)` in `src/features.py` adds both blocks to the panel. `BLOCK_A` and `BLOCK_B` list their column names.

- **Lags.** `feature_lags(horizon)` gives `[1, 5, 15, 60]` for 15 minutes. Block A uses all of them; block B uses the middle ones (5, 15) to stay near 15 features per block.
- **No look-ahead.** Candle features on row t only use minutes up to t − 1. Metrics and funding are attached by the panel once known, so they are used as is.
- **No filling.** Any window containing a missing minute gives NaN.

### Block A: spot only (the baseline)

| Name | Formula | Unit | Source columns | Why |
| --- | --- | --- | --- | --- |
| `a_ret_{L}` | log VWAP(t−1) − log VWAP(t−1−L) | log return | `spot_vwap` | Spot momentum or reversal |
| `a_flow_{L}` | Σ signed volume / Σ volume over L | fraction (−1 to 1) | `spot_signed_volume`, `spot_volume` | Spot buying pressure |
| `a_rv_{L}` | std of 1-min returns over L (L ≥ 5) | log return | `spot_vwap` | Volatility regime |
| `a_count_rel`, `a_vol_rel` | value / its 60-min mean | ratio | `spot_count`, `spot_volume` | Unusual activity |
| `a_hour_sin`, `a_hour_cos` | sin, cos(2π · hour / 24) | – | `open_time` | Intraday cycle |

### Block B: perp (its added value answers H1)

| Name | Formula | Unit | Source columns | Why |
| --- | --- | --- | --- | --- |
| `b_flow_{L}` | as `a_flow_{L}`, on the perp | fraction (−1 to 1) | `perp_signed_volume`, `perp_volume` | Perp buying pressure may lead spot |
| `b_flow_div_{L}` | `b_flow_{L}` − spot flow over L | fraction | perp and spot flow columns | Perp pressure spot has not shown yet |
| `b_basis_bps` | (perp VWAP / spot VWAP − 1) × 10⁴ | bps | `perp_vwap`, `spot_vwap` | Spot tends to close the gap |
| `b_basis_chg_{L}` | change in `b_basis_bps` over L | bps | `perp_vwap`, `spot_vwap` | Fresh perp pressure |
| `b_lead_{L}` | perp log return − spot log return over L | log return | `perp_vwap`, `spot_vwap` | Perp moved first, spot may follow |
| `b_oi_chg` | % change in open interest since the previous reading | % | `sum_open_interest`, `metrics_time` | Positions opened or closed |
| `b_toptrader_ls`, `b_ls`, `b_taker_ratio` | log of the ratio | log ratio | `metrics` ratio columns | Positioning skew |
| `b_funding_bps` | funding rate × 10⁴ | bps | `funding_rate` | Slow context on the crowded side |
| `b_delev_long_{L}` | potential long liquidations that became known in the last L minutes | count | `sum_open_interest`, `metrics_time`, `perp_close`, `perp_signed_volume`, `perp_volume` | Forced selling on the perp may spill into spot |
| `b_delev_short_{L}` | same, for potential short liquidations | count | same | Forced buying on the perp may spill into spot |

### Potential liquidations (deleveraging proxy)

Binance does not publish liquidations for USD-M perps, so they are inferred. A 5-minute bar is a potential liquidation when open interest falls to its 1% quantile or lower **and** either |perp return| or |perp flow share| reaches its 95% quantile (spec section 6). The side is long when the price fell, short when it rose.

- `deleveraging_events(panel, oi_q)` lists them, with the open interest drop in % and USD. `cluster_events(events)` merges those at most 30 minutes apart, dated at the first.
- `sensitivity_table(panel)` gives the counts at the 0.5%, 1% and 2% cut-offs.
- **Cut-offs.** Full sample for the descriptive analysis; `"past"` (earlier bars only) for the model features by default; or fixed values from `deleveraging_thresholds` on training data.
- **Timing.** A bar's open interest change runs to the reading stamped at its end, as in the spec's table. It is only used once that reading is known, 10 minutes after the bar starts.

## Configuration

Every run parameter (symbol, horizon, sample window, lags, seed, paths) lives in `config.yaml`. Code reads it with `src.config.load_config()`, so nothing in `src/` hardcodes the coin, the horizon or the dates.

## Figures

Every figure uses the shared style in `src/plots.py`, so they all look the same in the report (spec §10).

| Name | What it does |
| --- | --- |
| `SPOT_COLOUR`, `PERP_COLOUR` | Fixed colours: spot is always blue (`#1f77b4`), perp always orange (`#ff7f0e`) |
| `CLUSTER_MARKER` | How suspected deleveraging clusters are marked: a red ✕, used as `ax.scatter(x, y, **CLUSTER_MARKER)` |
| `apply_style()` | Sets the matplotlib theme, figure size, font sizes and a light grid. Call once before plotting |
| `label_axes(ax, x, y, unit)` | Labels both axes with their units: `unit=("min", "bps")` gives `Time [min]` and `Basis [bps]`. Raises an error if a unit is missing |
| `save_figure(fig, name, cfg)` | Saves the figure as `<figures_dir>/<name>.png` (`report/figures` in `config.yaml`), creates the folder if needed, then closes the figure |

## Contributing

- One branch per task, merged into `main` through a pull request with one approving review. Direct pushes to `main` are blocked.
- Run `pytest` before opening a pull request.
- Strip notebook outputs before committing. Run this once per clone to do it automatically:

  ```
  nb-clean add-filter --remove-empty-cells
  ```