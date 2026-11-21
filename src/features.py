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


def block_b(panel, lags):
    """Perp features (block B), one row per panel row."""
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
    return features


def build_features(panel, cfg):
    """The panel with the block A and block B feature columns added."""
    lags = feature_lags(cfg["horizon_minutes"], cfg)
    return pd.concat([panel, block_a(panel, lags), block_b(panel, lags)], axis=1)


# Block column names for the default config, for the ablation (A vs A + B).
BLOCK_A = block_a_columns(load_config()["lags"])
BLOCK_B = block_b_columns(load_config()["lags"])
