# clean → 1-min panel: spot and perp aligned on one minute grid
import logging
from pathlib import Path

import pandas as pd

from src import fetch
from src.clean import clean_archive_dataset, clean_funding, load_or_build
from src.schemas import MARKET_COLUMNS, MS_PER_MINUTE, SCHEMAS

log = logging.getLogger(__name__)

# Panel: one row per minute of the window. Row open_time = t holds the candles of [t, t + 1 min);
# values joined from slower datasets are the last ones known at or before t. The panel starts
# at the first minute where metrics and funding are both known (a few minutes into the window).

# Spot and perp closes of the same minute differ by the basis, a few bps for liquid pairs.
# A median gap above 1% means a join or unit bug, not a market effect.
MAX_MEDIAN_CLOSE_GAP = 0.01


def panel_path(data_dir, symbol, start, end):
    """Parquet path of a symbol's 1-min panel."""
    return Path(data_dir) / "panel" / f"{symbol}_{start}_{end}.parquet"


def empty_table(dataset):
    """A clean table with a dataset's columns and dtypes but no rows."""
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in SCHEMAS[dataset]["columns"].items()})


def minute_grid(start, end):
    """Every minute of the window, including minutes no market traded in."""
    start_ms, end_ms = fetch.window_to_ms(start, end)
    return pd.DataFrame({"open_time": pd.Series(range(start_ms, end_ms + 1, MS_PER_MINUTE), dtype="int64")})


def market_columns(klines, market):
    """Prefix a market's candle columns with its name and add VWAP and signed volume."""
    table = klines[["open_time", *MARKET_COLUMNS]].copy()
    # A minute without trades has no VWAP; it stays missing rather than borrowing a price.
    table["vwap"] = (table["quote_volume"] / table["volume"]).where(table["volume"] > 0)
    table["signed_volume"] = 2 * table["taker_buy_volume"] - table["volume"]
    return table.rename(columns={name: f"{market}_{name}" for name in table.columns if name != "open_time"})


def asof_join(panel, table, time_column, known_after_minutes=0, max_age_minutes=None):
    """Attach to each minute the last row of table known at or before its open, never a later one.

    A row stamped time_column = s is known at s + known_after_minutes. Readings older than
    max_age_minutes are left missing instead of being carried forward through an outage.
    """
    right = table.copy()
    right["known_at"] = right[time_column] + known_after_minutes * MS_PER_MINUTE
    tolerance = None if max_age_minutes is None else max_age_minutes * MS_PER_MINUTE
    joined = pd.merge_asof(
        panel,
        right.sort_values("known_at"),
        left_on="open_time",
        right_on="known_at",
        direction="backward",
        tolerance=tolerance,
    )
    # Minutes with no known reading leave the source time missing; keep it an integer ms column.
    joined[time_column] = joined[time_column].astype("Int64")
    return joined.drop(columns="known_at")


def make_panel(start, end, spot, perp, funding, premium=None, metrics=None,
               metrics_known_after_minutes=0, metrics_max_age_minutes=None):
    """Align clean tables on the minute grid: candles by minute, metrics and funding as of each minute."""
    panel = minute_grid(start, end)
    for market, klines in (("spot", spot), ("perp", perp)):
        panel = panel.merge(market_columns(klines, market), on="open_time", how="left", validate="one_to_one")
        # Keep trade counts integer whether or not the window has missing minutes.
        panel[f"{market}_count"] = panel[f"{market}_count"].astype("Int64")

    # Optional datasets that were not downloaded still get their columns, left empty,
    # so the panel has the same schema on every run.
    if premium is None:
        premium = empty_table("premiumIndexKlines")
    premium = premium[["open_time", "close"]].rename(columns={"close": "premium_index"})
    panel = panel.merge(premium, on="open_time", how="left", validate="one_to_one")

    if metrics is None:
        metrics = empty_table("metrics")
    panel = asof_join(
        panel,
        metrics.rename(columns={"create_time": "metrics_time"}),
        "metrics_time",
        metrics_known_after_minutes,
        metrics_max_age_minutes,
    )
    panel = asof_join(panel, funding[["funding_time", "funding_rate"]], "funding_time")

    # Gap mask: missing minutes stay as empty rows, flagged here, never forward-filled.
    panel["spot_missing"] = panel["spot_close"].isna()
    panel["perp_missing"] = panel["perp_close"].isna()
    panel["gap"] = panel["spot_missing"] | panel["perp_missing"]
    return panel


def drop_unknown_start(panel, time_columns=("metrics_time", "funding_time")):
    """Drop the first minutes of the window, before every joined dataset has a known value.

    Metrics are first known known_after_minutes into the window, and funding at the first
    settlement after it starts. Only the start is trimmed: a later missing reading (an outage)
    stays a missing value. A joined dataset with no value at all (an optional dataset that was
    not downloaded) is ignored, so it does not empty the panel.
    """
    known = [panel[column].notna() for column in time_columns if panel[column].notna().any()]
    if not known:
        return panel
    all_known = pd.concat(known, axis=1).all(axis=1).to_numpy()
    if not all_known.any():
        raise ValueError(f"No minute has a known value for all of {', '.join(time_columns)}")
    first = int(all_known.argmax())
    if first:
        log.info("Dropped the first %d minutes, before %s were all known", first, ", ".join(time_columns))
    return panel.iloc[first:].reset_index(drop=True)


def check_spot_perp_tracking(panel):
    """Fail if spot and perp closes do not track each other, the symptom of a join or unit bug."""
    close_gap = (panel["perp_close"] / panel["spot_close"] - 1).abs().dropna()
    if close_gap.empty:
        raise ValueError("No minute has both a spot and a perp close")
    log.info(
        "Perp vs spot close gap: median %.1f bps, 99th percentile %.1f bps",
        close_gap.median() * 1e4,
        close_gap.quantile(0.99) * 1e4,
    )
    if close_gap.median() > MAX_MEDIAN_CLOSE_GAP:
        raise ValueError(f"Spot and perp closes differ by a median {close_gap.median():.2%}")


def dataset_spec(cfg, market, dataset):
    """Return the config entry of one archive dataset, or None if it is not configured."""
    for spec in cfg["archive_datasets"]:
        if spec["market"] == market and spec["dataset"] == dataset:
            return spec
    return None


def build_panel(cfg, symbol, start, end):
    """Return the symbol's 1-min spot/perp panel, building and saving it on the first call."""

    def load(market, dataset):
        spec = dataset_spec(cfg, market, dataset)
        return None if spec is None else clean_archive_dataset(cfg, spec, symbol, start, end)

    def build():
        spot, perp = load("spot", "klines"), load("perp", "klines")
        if spot is None or perp is None:
            raise ValueError("config.yaml must list both spot and perp klines")
        metrics_spec = dataset_spec(cfg, "perp", "metrics") or {}
        panel = make_panel(
            start,
            end,
            spot,
            perp,
            funding=clean_funding(cfg, symbol, start, end),
            premium=load("perp", "premiumIndexKlines"),
            metrics=load("perp", "metrics"),
            metrics_known_after_minutes=metrics_spec.get("known_after_minutes", 0),
            metrics_max_age_minutes=metrics_spec.get("max_age_minutes"),
        )
        panel = drop_unknown_start(panel)
        check_spot_perp_tracking(panel)
        log.info(
            "Panel: %d minutes, %d with spot missing, %d with perp missing",
            len(panel),
            panel["spot_missing"].sum(),
            panel["perp_missing"].sum(),
        )
        return panel

    return load_or_build(panel_path(cfg["data_dir"], symbol, start, end), build)
