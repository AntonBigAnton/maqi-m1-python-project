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


# Block A columns for the default config
BLOCK_A = block_a_columns(load_config()["lags"])