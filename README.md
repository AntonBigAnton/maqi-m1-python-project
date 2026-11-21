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

## Descriptive analysis

`src/descriptive.py` computes the tables behind the descriptive figures (spec section 7); `src/plots.py` draws them.

### Lead-lag (block E)

Does the perp move before spot (H2)?

| Function | What it gives | Reading |
| --- | --- | --- |
| `lead_lag_table(panel, max_lag=30)` | Correlation of perp return, perp flow share and basis at minute *t* with the spot return at *t* + lag, lags −30 to +30 | A peak at a positive lag means the perp leads spot; how fast it falls to zero shows how long the lead lasts |
| `granger(panel, max_lag=15)` | Granger F-tests in both directions (perp signal → spot return and back), with statsmodels OLS | `delta_r2` is the share of variance the other series' past adds: the size of the effect |

- **Confidence band.** Figure 8 shades ±1.96 / √N, the 95% band of zero correlation for independent data. Returns and flows are autocorrelated, so the real band is wider.
- **Significance vs size.** With about 260,000 minutes almost any effect is significant, so report `delta_r2`, not only the p-value.
- **No filling.** A missing minute leaves its returns empty, and pairs with an empty value are skipped.

**Event study around suspected deleveraging clusters (Figure 9 ★).** `event_study(panel, clusters, window=(-30, 60))` averages the spot return and the basis change around each cluster, by side (long: price fell; short: price rose), with 95% bootstrap bands that resample whole clusters.

- **Time 0** is the end of the cluster's first 5-minute bar. The bar itself (shaded) contains both the perp and the spot move, so only the path after 0 is evidence of a lead.
- **Paths** use minute closes: spot return and basis change since time 0, in bps.
- **Placebo.** `placebo_events(panel, clusters)` picks bars with an equally big perp move but no fall in open interest, away from any cluster. If the cluster paths look like the placebo, a big move alone explains them.
- Figure 9 uses the 1% open interest threshold; the appendix repeats it at 0.5% and 2%. Clusters are at least 30 minutes apart, but their windows can overlap.

### Data quality and market structure (blocks A–B)

| Function | What it gives | Figure |
| --- | --- | --- |
| `missing_minutes(panel)` | Missing minutes (spot or perp) per day and UTC hour | Figure 1 |
| `tracking_stats(panel)` | Gap between perp and spot closes in bps: mean, median, 99th percentile, max | Table (the check in `panel.py`, as numbers) |
| `volume_share(panel)` | Daily perp share of traded value (quote volume) | Figure 2 |
| `intraday_profile(panel)` | Traded value per minute and std of 1-min returns by UTC hour, spot and perp | Figure 3 |

### Returns and volatility (block C)

| Function | What it gives | Output |
| --- | --- | --- |
| `return_samples(panel, horizon)` and `return_moments(samples)` | 1-min and horizon-min spot returns (non-overlapping), with mean, std, skew, excess kurtosis, min and max | Figure 4, table |
| `acf_table(panel, max_lag=60)` | Autocorrelation of 1-min returns and of absolute returns | Figure 5 |
| `five_minute_series(panel)` and `stationarity_tests(series)` | ADF and KPSS tests on 5-minute spot return, basis and flows (statsmodels) | Table in `report/tables` |

- **Reading the tests.** ADF's null is a unit root, so a small p-value means stationary; KPSS's null is stationary, so a small p-value means non-stationary. KPSS p-values are only tabulated between 0.01 and 0.1.
- **Why 5 minutes.** The tests on 260,000 one-minute values are slow; 5-minute bars keep the same behaviour with 52,000 values.
- `save_table(table, name, cfg)` writes a table as `report/tables/<name>.csv`.

### Perp-specific signals (block D)

| Function | What it gives | Output |
| --- | --- | --- |
| `basis_over_time(panel)`, `deleveraging_clusters(panel)` and `basis_half_life(panel)` | The 1-min basis in bps, the suspected deleveraging clusters (full sample), and an AR(1) fit with the basis half-life in minutes | Figure 6, table |
| `leverage_series(panel)` | Funding rate at each settlement (bps) and open interest at each metrics reading (billion USDT) | Figure 7 |
| `flow_correlation(panel, lags)` | Correlation of spot and perp signed flow share over each window | Table |
| `redundant_pairs(corr, threshold=0.9)` | Pairs of block A and B features with \|correlation\| ≥ 0.9, to drop before the models | Figure 10, table |

- **Half-life.** With basis_t = a + φ · basis_{t−1}, a deviation halves in ln(0.5) / ln(φ) minutes.

## Targets and walk-forward validation

Targets are in `src/models.py` and the splitter is in `src/evaluate.py` (spec sections 5 and 9). The feature columns are passed in as two lists, `block_a` (spot only) and `block_b` (perp), so this code does not depend on how `src/features.py` builds them.

### Targets

| Target | Definition |
| --- | --- |
| Regression `y` | Forward spot log return over the horizon *h*: log VWAP of minute *t + h* minus log VWAP of minute *t* |
| Classification | Up / flat / down: flat if \|y\| ≤ *k* × `vol_scale`, otherwise the sign of `y` |

- **Volatility scale.** `vol_scale` is the standard deviation of 1-minute spot VWAP returns over the previous `vol_window_minutes` (60), times √*h*. It uses only minutes before *t*, so the class threshold is known when the forecast is made, and it widens in volatile periods instead of calling every move "up" or "down".
- **Threshold fitted on training data only.** *k* is chosen on each fold's training rows so that a share `flat_share` (one third) of them is flat, then applied unchanged to the test rows.
- **No filling.** A missing VWAP at *t* or *t + h* (no trade, gap, end of the window) leaves the target missing. The future price is looked up by time, not by row position.
- **Same rows for both models.** `model_frame` drops any minute that is missing the target, the scale or any feature of block A or B. The baseline (A) and the full model (A + B) are then scored on the same minutes, so any difference between them comes from block B.

### Walk-forward splits

The window is cut into calendar months (UTC). Each fold tests on one month and trains on every month before it:

| Fold | Fit (tuning) | Tune | Train (final fit) | Test |
| --- | --- | --- | --- | --- |
| 1 | month 1 | month 2 | months 1–2 | month 3 |
| 2 | months 1–2 | month 3 | months 1–3 | month 4 |
| 3 | months 1–3 | month 4 | months 1–4 | month 5 |
| 4 | months 1–4 | month 5 | months 1–5 | month 6 |

- **Tuning.** Hyperparameters are chosen by fitting on the *fit* months and scoring on the *tune* month (the last training month). The model is then refitted on all training months and scored once on the test month. The test month is never used to choose anything.
- **Fitted on training rows only.** `fold_data` fits the feature scaler (mean 0, standard deviation 1) and the class threshold on the rows it trains on, then applies them unchanged to the rows it scores. Tree models skip the scaler with `scale=False`.
- **Embargo.** A label at *t* uses prices up to minute *t + h*. So the last `embargo_minutes` before each scored period are dropped from the rows fitted on it: before the test month, and before the tune month. The default is the horizon *h*, and anything shorter is rejected.
- **Cut by time.** Folds are defined by timestamps, so they stay correct after `model_frame` drops rows.
- **Run length.** At least `first_test_month` months of data are needed, so a `--quick` (one-month) run cannot produce folds.

```python
from src.evaluate import fold_data, walk_forward_splits
from src.models import feature_sets, model_frame

frame = model_frame(data, BLOCK_A, BLOCK_B, cfg["horizon_minutes"], cfg["target"]["vol_window_minutes"])
for fold in walk_forward_splits(frame["open_time"], cfg["horizon_minutes"], cfg["walk_forward"]["first_test_month"],
                                cfg["walk_forward"]["embargo_minutes"]):
    for name, columns in feature_sets(BLOCK_A, BLOCK_B).items():
        tune = fold_data(frame, fold.fit, fold.tune, columns, cfg["target"]["flat_share"])
        test = fold_data(frame, fold.train, fold.test, columns, cfg["target"]["flat_share"])
```

`fold_table(folds, frame["open_time"])` gives the size and date range of every set, for the report.

### Metrics

Models save their test-month predictions with one row per model, feature set (`A` or `A+B`), fold and minute: `y_true` and `y_pred` for regression, and `class_true` and `p_down`, `p_flat`, `p_up` for classification. `fold_metrics(predictions, task)` scores each fold, and `results_table` gives the mean, standard deviation, min and max across folds for every model and feature set.

| Task | Metric | Reading |
| --- | --- | --- |
| Regression | Out-of-sample R² = 1 − SSE(model) / SSE(zero forecast) | 0 = no better than forecasting no move; negative = worse |
| Classification | Balanced accuracy, macro F1, log loss | Accuracy alone is not used: when "flat" dominates, always predicting it scores well |

### Models (levels 0–2)

Each model is fitted on block A and on A + B, on the same folds and rows (spec section 8):

| Task | Model | Hyperparameter tuned on the tune month |
| --- | --- | --- |
| Regression | `zero`: forecasts no move (random walk) | none |
| Regression | `ridge`, `lasso` | Penalty (scale-free: ridge as a multiple of the training rows, lasso of the target's standard deviation), chosen by out-of-sample R² |
| Classification | `majority`: the training class shares | none |
| Classification | `logit`: multinomial logistic regression | `C`, chosen by log loss |
| Both | `boosting`: histogram gradient boosting (regressor or classifier) | Tree depth and learning rate, chosen by R² or log loss |

`walk_forward_predictions(frame, folds, BLOCK_A, BLOCK_B, flat_share, task)` runs every model × feature set × fold. For each fold it scores every grid value by fitting on the fit months and scoring on the tune month, then refits the best on all training months and forecasts the test month. It returns the test-month predictions in the format above, plus the chosen hyperparameters. `save_predictions` writes them to `data/predictions/<symbol>_<start>_<end>_h<horizon>_<task>.parquet`.

Boosting checks for non-linear effects and interactions that the linear models cannot capture. Its other settings are fixed for noisy minute data: 200 trees, leaves of at least 500 rows, and no early stopping, so a fit depends only on its training rows and `seed` from `config.yaml`. The simplest grid point (depth 1, learning rate 0.003) is close to a constant forecast: if tuning picks it, boosting found nothing to use.

On the real six-month BTCUSDT panel, a full run (both tasks, all models, both feature sets, four folds) takes about 2 minutes on an Apple-silicon laptop. Boosting takes almost all of that; the level 0–1 models take about 10 seconds.

### Significance tests

Forecasts are made every minute for a 15-minute horizon, so consecutive errors overlap. Every test therefore uses Newey-West (HAC) standard errors, with the horizon as the number of lags.

| Function | Test | Used for |
| --- | --- | --- |
| `clark_west` | Clark-West, one-sided | **H1**: does A + B forecast better than A? A is nested in A + B, so the test corrects for the noise added by estimating B's coefficients, which are zero if H1 is false |
| `diebold_mariano` | Diebold-Mariano, two-sided | Non-nested comparisons, such as linear vs boosting, on squared errors or log losses |
| `hac_coefficients` | OLS with Newey-West standard errors | Level-1 coefficient table: which perp features matter |
| `holm` | Holm correction | Adjusting the pooled Clark-West p-values across model levels |

`clark_west_table(predictions, models, lags)` gives the H1 answer: one Clark-West test per model and fold (for Figure 11), plus one pooled over all test months. The pooled p-values are Holm-adjusted across models, and `h1_supported` is true when the adjusted p-value is below 5%.

### Running the full ablation

`run_ablation(cfg, data, BLOCK_A, BLOCK_B, symbol, start, end)` runs everything above in one call: the modelling table, the folds, every model on A and A + B for both tasks, and every test. `data` is the panel with the feature columns added. It saves:

| Output | Content |
| --- | --- |
| `data/predictions/<symbol>_<start>_<end>_h<horizon>_<task>.parquet` | Test-month predictions of every model, feature set and fold |
| `report/tables/<symbol>_h<horizon>_folds.csv` | Size and date range of each fold's sets |
| `report/tables/<symbol>_h<horizon>_tuning_<task>.csv` | Hyperparameters chosen on each tune month |
| `report/tables/<symbol>_h<horizon>_fold_metrics_<task>.csv` | Metrics of each model, feature set and fold |
| `report/tables/<symbol>_h<horizon>_results_<task>.csv` | Mean and spread of the metrics across folds |
| `report/tables/<symbol>_h<horizon>_clark_west.csv` | **H1**: Clark-West of A + B vs A for every regression model except the zero forecast, per fold and pooled, with Holm |
| `report/tables/<symbol>_h<horizon>_diebold_mariano.csv` | Diebold-Mariano of each linear model vs boosting, per feature set |
| `report/tables/<symbol>_h<horizon>_coefficients.csv` | Figure 12: OLS coefficients of the A + B model on standardised features, with Newey-West standard errors and 95% intervals |
| `report/tables/<symbol>_h<horizon>_permutation_importance.csv` | Figure 12: R² lost when each block-B feature is shuffled in the test month, in the A + B boosting regression, per fold and pooled |

**Reading the figure 12 tables.** Both are descriptive, not causal:
- **Coefficients** are fitted in-sample on every usable minute. A coefficient is the change in the forecast 15-minute return for a one-standard-deviation move in the feature.
- **Permutation importance** refits boosting on each fold's training months with its tuned hyperparameters. It then measures how much test-month R² drops when one feature is shuffled. A feature the model doesn't use scores about 0. Correlated features share their importance, because shuffling one leaves the information in the others.

Tables are named by symbol and horizon, so a smoke test on another symbol never overwrites them. Nothing is cached: every call rebuilds the results from the data passed in. On the real BTCUSDT panel it takes about 2 minutes.

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
| `figure_8(lead_lag, cfg)` | Figure 8 ★: cross-correlation of each perp signal with future spot returns (from `lead_lag_table`), with the ±1.96 / √N band; saved as `fig08_lead_lag_<symbol>.png` |
| `figure_1`, `figure_2`, `figure_3` | Missing minutes heatmap (day × hour), perp share of traded value over time, intraday volume and volatility; saved as `fig01_…`, `fig02_…`, `fig03_…` |
| `figure_4`, `figure_5` | Return histograms with a normal overlay and QQ plots; ACF of returns vs absolute returns; saved as `fig04_…`, `fig05_…` |
| `figure_6`, `figure_7`, `figure_10` | Basis with deleveraging clusters marked; funding and open interest on aligned panels; feature correlation heatmap; saved as `fig06_…`, `fig07_…`, `fig10_…` |
| `figure_9(study, placebo, cfg, suffix="")` | Figure 9 ★: event study by side, spot return and basis change since the end of the event bar, with bootstrap bands and the placebo path; saved as `fig09_event_study_<symbol><suffix>.png` |

## Contributing

- One branch per task, merged into `main` through a pull request with one approving review. Direct pushes to `main` are blocked.
- Run `pytest` before opening a pull request.
- Strip notebook outputs before committing. Run this once per clone to do it automatically:

  ```
  nb-clean add-filter --remove-empty-cells
  ```