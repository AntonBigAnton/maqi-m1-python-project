# blocks A and B, as-of joins
"""Model features, built from the 1-min panel.

Timing: row t holds the candle of minute [t, t + 1 min), which is only complete at t + 1 min.
So a feature on row t only uses minutes up to t - 1.
Missing minutes are empty (NaN) in the panel, so any window that contains one gives NaN.
Nothing is ever filled in.
"""
import numpy as np
import pandas as pd

from src.config import load_config
from src.schemas import MS_PER_MINUTE

ACTIVITY_WINDOW = 60  # activity is compared with its average over the last hour
MIN_RV_WINDOW = 5     # volatility needs at least 5 returns to mean something
MS_PER_HOUR = 60 * MS_PER_MINUTE
BPS = 10_000          # 1 basis point = 0.01%
METRICS_EVERY_MINUTES = 5  # Binance publishes one metrics reading every 5 minutes

# Deleveraging proxy : potential liquidations
DELEV_OI_QUANTILE = 0.01    # open interest drop at or below its 1% quantile ...
DELEV_MOVE_QUANTILE = 0.95  # ... and |perp return| or |flow share| at or above its 95% quantile
CLUSTER_GAP_MINUTES = 30    # potential liquidations at most 30 minutes apart form one cluster
MIN_HISTORY_DAYS = 1        # past-only thresholds start after one day of 5-minute bars
BAR_MS = METRICS_EVERY_MINUTES * MS_PER_MINUTE


# metrics column behind each block B ratio feature
RATIO_COLUMNS = {
    "b_toptrader_ls": "sum_toptrader_long_short_ratio",
    "b_ls": "count_long_short_ratio",
    "b_taker_ratio": "sum_taker_long_short_vol_ratio",
}


def feature_lags(horizon, cfg=None):
    """Lags for a horizon: the config lags, rescaled if horizon differs from the config's.

    config.yaml gives the lags for its own horizon ([1, 5, 15, 60] for 15 minutes).
    Another horizon gets the same lags scaled in proportion, at least 1 minute each.
    """
    if cfg is None:
        cfg = load_config()
    scale = horizon / cfg["horizon_minutes"]
    lags = []
    for lag in cfg["lags"]:
        new_lag = max(1, round(lag * scale))
        if new_lag not in lags:
            lags.append(new_lag)
    return lags


def block_a_columns(lags):
    """Names of the block A columns, in the order block_a returns them."""
    names = []
    for lag in lags:
        names.append(f"a_ret_{lag}")
    for lag in lags:
        names.append(f"a_flow_{lag}")
    for lag in lags:
        if lag >= MIN_RV_WINDOW:
            names.append(f"a_rv_{lag}")
    names += ["a_count_rel", "a_vol_rel", "a_hour_sin", "a_hour_cos"]
    return names


def block_a(panel, lags):
    """Spot-only features (block A), one row per panel row."""
    log_price = np.log(panel["spot_vwap"])
    ret_1min = log_price.diff()  # log return of each minute
    signed_volume = panel["spot_signed_volume"]
    volume = panel["spot_volume"]
    count = panel["spot_count"].astype(float)

    features = pd.DataFrame(index=panel.index)

    # Each rolling(...) window must be full of values, so one missing minute gives NaN.
    for lag in lags:
        features[f"a_ret_{lag}"] = ret_1min.rolling(lag).sum()
    for lag in lags:
        features[f"a_flow_{lag}"] = signed_volume.rolling(lag).sum() / volume.rolling(lag).sum()
    for lag in lags:
        if lag >= MIN_RV_WINDOW:
            features[f"a_rv_{lag}"] = ret_1min.rolling(lag).std()
    features["a_count_rel"] = count / count.rolling(ACTIVITY_WINDOW).mean()
    features["a_vol_rel"] = volume / volume.rolling(ACTIVITY_WINDOW).mean()

    # Move everything down one row, so row t only uses minutes up to t - 1.
    features = features.shift(1)

    # Hour of day (UTC) placed on a circle, so 23:00 and 00:00 are close. The clock is known at t.
    hour = (panel["open_time"] // MS_PER_HOUR) % 24
    angle = 2 * np.pi * hour / 24
    features["a_hour_sin"] = np.sin(angle)
    features["a_hour_cos"] = np.cos(angle)
    return features


def block_b_lags(lags):
    """Block B uses the middle lags only (drops the shortest and the longest), to stay near 15 features."""
    if len(lags) <= 2:
        return lags
    return lags[1:-1]


def block_b_columns(lags):
    """Names of the block B columns, in the order block_b returns them."""
    b_lags = block_b_lags(lags)
    names = []
    for lag in b_lags:
        names.append(f"b_flow_{lag}")
    for lag in b_lags:
        names.append(f"b_flow_div_{lag}")
    names.append("b_basis_bps")
    for lag in b_lags:
        names.append(f"b_basis_chg_{lag}")
    for lag in b_lags:
        names.append(f"b_lead_{lag}")
    names += ["b_oi_chg", "b_toptrader_ls", "b_ls", "b_taker_ratio", "b_funding_bps"]
    for side in ("long", "short"):
        for lag in b_lags:
            names.append(f"b_delev_{side}_{lag}")
    return names


def open_interest_change(panel):
    """% change in open interest between each metrics reading and the one before it.

    The panel repeats a reading on every minute until the next one, so the change is computed
    once per reading (when metrics_time changes), then copied to every minute of that reading.
    """
    readings = panel[["metrics_time", "sum_open_interest"]].dropna()
    readings = readings.drop_duplicates(subset="metrics_time")
    change = (readings["sum_open_interest"] / readings["sum_open_interest"].shift(1) - 1) * 100
    # If a reading is missing, the previous one is more than 5 minutes back: leave the change empty.
    minutes_since_previous = readings["metrics_time"].astype(float).diff() / MS_PER_MINUTE
    change = change.where(minutes_since_previous == METRICS_EVERY_MINUTES)
    change_by_time = pd.Series(change.to_numpy(), index=readings["metrics_time"].to_numpy())
    return panel["metrics_time"].map(change_by_time).astype(float)


def block_b(panel, lags, delev_thresholds="past"):
    """Perp features (block B), one row per panel row.

    delev_thresholds sets the cut-offs of the potential liquidation counts (see deleveraging_counts).
    """
    spot_ret_1min = np.log(panel["spot_vwap"]).diff()
    perp_ret_1min = np.log(panel["perp_vwap"]).diff()
    basis = (panel["perp_vwap"] / panel["spot_vwap"] - 1) * BPS
    b_lags = block_b_lags(lags)

    features = pd.DataFrame(index=panel.index)

    # From the candles: like block A, a window with a missing minute gives NaN.
    for lag in b_lags:
        features[f"b_flow_{lag}"] = panel["perp_signed_volume"].rolling(lag).sum() / panel["perp_volume"].rolling(lag).sum()
    for lag in b_lags:
        spot_flow = panel["spot_signed_volume"].rolling(lag).sum() / panel["spot_volume"].rolling(lag).sum()
        features[f"b_flow_div_{lag}"] = features[f"b_flow_{lag}"] - spot_flow
    features["b_basis_bps"] = basis
    for lag in b_lags:
        features[f"b_basis_chg_{lag}"] = basis.diff().rolling(lag).sum()
    for lag in b_lags:
        features[f"b_lead_{lag}"] = perp_ret_1min.rolling(lag).sum() - spot_ret_1min.rolling(lag).sum()

    # Move the candle features down one row, so row t only uses minutes up to t - 1.
    features = features.shift(1)

    # From metrics and funding: the panel only attaches a reading once it is known,
    # so the value on row t is already known at t and is not shifted.
    features["b_oi_chg"] = open_interest_change(panel)
    for name, column in RATIO_COLUMNS.items():
        features[name] = np.log(panel[column])
    features["b_funding_bps"] = panel["funding_rate"] * BPS

    # Potential liquidations: counted once their 5-minute bar has closed, so not shifted either.
    counts = deleveraging_counts(panel, lags, delev_thresholds)
    for column in counts.columns:
        features[column] = counts[column]

    return features


def build_features(panel, cfg, delev_thresholds="past"):
    """The panel with the block A and block B feature columns added."""
    lags = feature_lags(cfg["horizon_minutes"], cfg)
    return pd.concat([panel, block_a(panel, lags), block_b(panel, lags, delev_thresholds)], axis=1)


# ---------- Deleveraging proxy: potential liquidations (#18) ----------

def five_minute_bars(panel):
    """Perp 5-minute bars with the three measures of the proxy.

    The bar starting at s covers minutes s to s + 4. Its open interest change is the one between
    the readings stamped s and s + 5, as in the spec's sensitivity table. It is known (known_at)
    once the bar has closed and the panel shows the reading stamped s + 5. A bar with a missing
    minute or a missing reading gets empty (NaN) measures.
    """
    minutes = pd.DataFrame({
        "bar_time": panel["open_time"] // BAR_MS * BAR_MS,
        "perp_ret": np.log(panel["perp_close"]).diff(),  # close-to-close 1-min log return
        "signed_volume": panel["perp_signed_volume"],
        "volume": panel["perp_volume"],
    })
    grouped = minutes.groupby("bar_time")
    full = METRICS_EVERY_MINUTES  # a bar needs all of its minutes
    bars = pd.DataFrame({
        "perp_ret": grouped["perp_ret"].sum(min_count=full),
        "flow_share": grouped["signed_volume"].sum(min_count=full) / grouped["volume"].sum(min_count=full),
    })

    # One row per metrics reading, kept at the first minute the panel shows it (when it is known)
    readings = panel[["open_time", "metrics_time", "sum_open_interest", "sum_open_interest_value"]].dropna()
    readings = readings.assign(stamp=readings["metrics_time"].astype("int64") // BAR_MS * BAR_MS)
    readings = readings.drop_duplicates(subset="stamp")
    stamps = readings["stamp"].to_numpy()
    oi = pd.Series(readings["sum_open_interest"].to_numpy(), index=stamps)
    oi_value = pd.Series(readings["sum_open_interest_value"].to_numpy(), index=stamps)
    reading_known = pd.Series(readings["open_time"].to_numpy(), index=stamps)

    # Change between the reading stamped s and the one 5 minutes earlier; empty if either is missing
    oi_chg = oi / oi.reindex(oi.index - BAR_MS).to_numpy() - 1
    oi_value_chg = oi_value - oi_value.reindex(oi_value.index - BAR_MS).to_numpy()

    # The change up to the reading stamped s goes with the bar starting at s - 5 min
    bar_end = bars.index + BAR_MS
    bars["oi_chg"] = oi_chg.reindex(bar_end).to_numpy()
    bars["oi_value_chg_usd"] = oi_value_chg.reindex(bar_end).to_numpy()
    # Known once the bar has closed and its closing reading is known (fmax ignores a missing reading)
    bars["known_at"] = np.fmax(bar_end, reading_known.reindex(bar_end).to_numpy()).astype("int64")
    return bars


def deleveraging_thresholds(bars, oi_q=DELEV_OI_QUANTILE, ret_q=DELEV_MOVE_QUANTILE, flow_q=DELEV_MOVE_QUANTILE):
    """Fixed cut-offs from a set of bars: the full sample, or one fold's training months."""
    return {
        "oi_chg": bars["oi_chg"].quantile(oi_q),
        "abs_ret": bars["perp_ret"].abs().quantile(ret_q),
        "abs_flow": bars["flow_share"].abs().quantile(flow_q),
    }


def past_thresholds(bars, oi_q=DELEV_OI_QUANTILE, ret_q=DELEV_MOVE_QUANTILE, flow_q=DELEV_MOVE_QUANTILE):
    """Cut-offs for each bar from the bars before it only, so no future data is used."""
    min_bars = MIN_HISTORY_DAYS * 24 * 60 // METRICS_EVERY_MINUTES

    def past_quantile(values, q):
        return values.expanding(min_periods=min_bars).quantile(q).shift(1)

    return pd.DataFrame({
        "oi_chg": past_quantile(bars["oi_chg"], oi_q),
        "abs_ret": past_quantile(bars["perp_ret"].abs(), ret_q),
        "abs_flow": past_quantile(bars["flow_share"].abs(), flow_q),
    }, index=bars.index)


def choose_thresholds(bars, thresholds, oi_q, ret_q=DELEV_MOVE_QUANTILE, flow_q=DELEV_MOVE_QUANTILE):
    """None: full sample. "past": from earlier bars only. A dict: fixed cut-offs given by the caller."""
    if thresholds is None:
        return deleveraging_thresholds(bars, oi_q, ret_q, flow_q)
    if isinstance(thresholds, str) and thresholds == "past":
        return past_thresholds(bars, oi_q, ret_q, flow_q)
    return thresholds


def flag_bars(bars, thresholds):
    """Mark each bar: was it checked, is it a potential liquidation, and on which side."""
    flagged = bars.copy()
    oi_drop = bars["oi_chg"] <= thresholds["oi_chg"]
    big_move = bars["perp_ret"].abs() >= thresholds["abs_ret"]
    one_sided = bars["flow_share"].abs() >= thresholds["abs_flow"]
    flagged["event"] = oi_drop & (big_move | one_sided)

    # Checked = every measure and every cut-off is known for this bar
    checked = bars[["oi_chg", "perp_ret", "flow_share"]].notna().all(axis=1)
    for name in ("oi_chg", "abs_ret", "abs_flow"):
        checked &= pd.Series(thresholds[name], index=bars.index).notna()
    flagged["checked"] = checked

    # Long positions are flushed when the price falls, short ones when it rises.
    # If the price did not move, the direction of the flow decides.
    direction = np.sign(bars["perp_ret"]).where(bars["perp_ret"] != 0, np.sign(bars["flow_share"]))
    flagged["side"] = np.where(direction < 0, "long", "short")
    return flagged


def deleveraging_events(panel, oi_q, ret_q=DELEV_MOVE_QUANTILE, flow_q=DELEV_MOVE_QUANTILE, thresholds=None):
    """Potential liquidations: one row per 5-minute bar where open interest dropped sharply
    while the perp price moved a lot or the flow was very one-sided.

    thresholds: None for full-sample cut-offs (descriptive analysis only), "past" for cut-offs from
    earlier bars only, or a dict of fixed cut-offs (e.g. from deleveraging_thresholds on training data).
    """
    bars = five_minute_bars(panel)
    flagged = flag_bars(bars, choose_thresholds(bars, thresholds, oi_q, ret_q, flow_q))
    events = flagged[flagged["event"]]
    return pd.DataFrame({
        "bar_time": events.index.to_numpy(),
        "known_at": events["known_at"].to_numpy(),
        "side": events["side"].to_numpy(),
        "oi_chg_pct": (events["oi_chg"] * 100).to_numpy(),
        "perp_ret_bps": (events["perp_ret"] * BPS).to_numpy(),
        "flow_share": events["flow_share"].to_numpy(),
        "oi_value_chg_usd": events["oi_value_chg_usd"].to_numpy(),
    })


def cluster_events(events, gap_minutes=CLUSTER_GAP_MINUTES):
    """Group potential liquidations at most gap_minutes after the previous one into clusters.

    A cascade often spans several bars; each cluster is dated at its first potential liquidation.
    """
    gap_ms = gap_minutes * MS_PER_MINUTE
    cluster_ids, cluster, previous = [], 0, None
    for t in events["bar_time"]:
        if previous is not None and t - previous > gap_ms:
            cluster += 1
        cluster_ids.append(cluster)
        previous = t
    grouped = events.assign(cluster=cluster_ids).groupby("cluster")
    return pd.DataFrame({
        "start": grouped["bar_time"].first(),
        "known_at": grouped["known_at"].first(),
        "end": grouped["bar_time"].last(),
        "events": grouped.size(),
        "side": grouped["side"].first(),
        "oi_value_chg_usd": grouped["oi_value_chg_usd"].sum(),
    }).reset_index(drop=True)


def sensitivity_table(panel, oi_quantiles=(0.005, 0.01, 0.02), gap_minutes=CLUSTER_GAP_MINUTES):
    """Number of potential liquidations and clusters at several open interest cut-offs (full sample)."""
    bars = five_minute_bars(panel)
    rows = []
    for oi_q in oi_quantiles:
        cut = deleveraging_thresholds(bars, oi_q)
        events = deleveraging_events(panel, oi_q, thresholds=cut)
        rows.append({
            "oi_quantile_pct": oi_q * 100,
            "oi_cut_pct": round(cut["oi_chg"] * 100, 3),
            "ret_cut_bps": round(cut["abs_ret"] * BPS, 1),
            "flow_cut": round(cut["abs_flow"], 3),
            "events": len(events),
            "falling": int((events["side"] == "long").sum()),
            "rising": int((events["side"] == "short").sum()),
            "clusters": len(cluster_events(events, gap_minutes)),
        })
    return pd.DataFrame(rows)


def deleveraging_counts(panel, lags, thresholds="past", oi_q=DELEV_OI_QUANTILE):
    """Block B features: potential liquidations by side that became known in the last L minutes.

    Each bar is counted from the minute it is known (known_at): closed, with its reading published.
    A bar that could not be checked (missing data, or too little history for "past" cut-offs)
    makes the counts of every window containing it empty (NaN): nothing is filled in.
    """
    bars = five_minute_bars(panel)
    flagged = flag_bars(bars, choose_thresholds(bars, thresholds, oi_q))
    counts = pd.DataFrame(index=panel.index)
    for side in ("long", "short"):
        # On the minute a bar becomes known: 1 if it is a potential liquidation on this side, 0 if not,
        # NaN if it could not be checked. Every other minute: 0.
        hits = (flagged["event"] & (flagged["side"] == side)).astype(float).where(flagged["checked"])
        hits = pd.Series(hits.to_numpy(), index=flagged["known_at"].to_numpy())
        # Two bars can become known on the same minute: add them up, NaN if either was not checked
        not_checked = hits.isna().groupby(level=0).any()
        hits = hits.fillna(0.0).groupby(level=0).sum().mask(not_checked)
        per_minute = pd.Series(0.0, index=panel["open_time"].to_numpy())
        closes = hits.index.intersection(per_minute.index)
        per_minute[closes] = hits[closes]
        for lag in block_b_lags(lags):
            counts[f"b_delev_{side}_{lag}"] = per_minute.rolling(lag).sum().to_numpy()
    return counts

# Block column names for the default config, for the ablation (A vs A + B).
BLOCK_A = block_a_columns(load_config()["lags"])
BLOCK_B = block_b_columns(load_config()["lags"])
