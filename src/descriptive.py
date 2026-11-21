# descriptive statistics (spec section 7): the numbers behind the descriptive figures
"""Descriptive analysis. Functions here compute tables; plots.py draws them.

Lead-lag (block E): does the perp move before spot, and how long does the effect last?
Rows of the panel are consecutive minutes, so shifting a column by k rows moves it by k minutes.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features import BPS, MS_PER_HOUR

Z_95 = 1.96  # two-sided 95% normal quantile
PERP_SIGNALS = ["perp_ret", "perp_flow", "basis_bps"]


def lead_lag_signals(panel):
    """1-minute series compared in the lead-lag analysis, one row per panel row.

    spot_ret and perp_ret are 1-minute log VWAP returns, perp_flow is perp signed volume / volume,
    basis_bps is (perp VWAP / spot VWAP - 1) x 10,000. A missing minute stays NaN.
    """
    return pd.DataFrame({
        "spot_ret": np.log(panel["spot_vwap"]).diff(),
        "perp_ret": np.log(panel["perp_vwap"]).diff(),
        "perp_flow": (panel["perp_signed_volume"] / panel["perp_volume"]).where(panel["perp_volume"] > 0),
        "basis_bps": (panel["perp_vwap"] / panel["spot_vwap"] - 1) * BPS,
    }, index=panel.index)


def cross_correlation(x, y, max_lag=30):
    """Correlation of x at minute t with y at minute t + lag, for lag = -max_lag .. max_lag.

    A positive lag means x moves first (x leads y). Pairs with a missing value are skipped.
    band is the 95% band of zero correlation for independent data, 1.96 / sqrt(N): returns
    and flows are autocorrelated, so the real band is wider (say so in the report).
    """
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        pair = pd.DataFrame({"x": x.to_numpy(), "y": y.shift(-lag).to_numpy()}).dropna()
        n = len(pair)
        rows.append({"lag": lag, "corr": pair["x"].corr(pair["y"]), "n": n, "band": Z_95 / np.sqrt(n)})
    return pd.DataFrame(rows)


def lead_lag_table(panel, max_lag=30):
    """Cross-correlation of each perp signal with future spot returns, stacked in one table."""
    signals = lead_lag_signals(panel)
    tables = []
    for name in PERP_SIGNALS:
        table = cross_correlation(signals[name], signals["spot_ret"], max_lag)
        tables.append(table.assign(signal=name))
    return pd.concat(tables, ignore_index=True)[["signal", "lag", "corr", "n", "band"]]


def granger_pair(x, y, max_lag=15):
    """Does the past of x help predict y, beyond y's own past? (Granger test, statsmodels OLS)

    Restricted model: y_t on a constant and y_{t-1} .. y_{t-max_lag}.
    Full model: the same plus x_{t-1} .. x_{t-max_lag}. The F-test compares them.
    delta_r2 is the extra share of y's variance explained by x's past: the size of the effect.
    """
    y_lags = {f"y_lag_{k}": y.shift(k) for k in range(1, max_lag + 1)}
    x_lags = {f"x_lag_{k}": x.shift(k) for k in range(1, max_lag + 1)}
    data = pd.DataFrame({"y": y, **y_lags, **x_lags}).dropna()
    restricted = sm.OLS(data["y"], sm.add_constant(data[list(y_lags)])).fit()
    full = sm.OLS(data["y"], sm.add_constant(data[list(y_lags) + list(x_lags)])).fit()
    f_stat, p_value, _ = full.compare_f_test(restricted)
    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "r2_restricted": restricted.rsquared,
        "r2_full": full.rsquared,
        "delta_r2": full.rsquared - restricted.rsquared,
        "n": len(data),
    }


def granger(panel, max_lag=15):
    """Granger tests in both directions between spot returns and each perp signal.

    With about 260,000 minutes almost any effect is significant, so read delta_r2 (the size of
    the effect) rather than the p-value alone.
    """
    signals = lead_lag_signals(panel)
    rows = []
    for name in PERP_SIGNALS:
        for cause, effect in ((name, "spot_ret"), ("spot_ret", name)):
            result = granger_pair(signals[cause], signals[effect], max_lag)
            rows.append({"cause": cause, "effect": effect, **result})
    return pd.DataFrame(rows)


# ---------- Blocks A-B: data quality and market structure (#20) ----------

def missing_minutes(panel):
    """Missing minutes (spot or perp, the panel's gap mask) per day and UTC hour: a day x hour table."""
    table = pd.DataFrame({
        "day": pd.to_datetime(panel["open_time"], unit="ms", utc=True).dt.date,
        "hour": (panel["open_time"] // MS_PER_HOUR) % 24,
        "missing": panel["gap"].astype(int),
    })
    return table.pivot_table(index="day", columns="hour", values="missing", aggfunc="sum", fill_value=0)


def tracking_stats(panel):
    """How closely spot and perp closes track each other: the gap |perp / spot - 1| in bps.

    The same measure as panel.check_spot_perp_tracking, returned as a table instead of a log line.
    """
    gap_bps = ((panel["perp_close"] / panel["spot_close"] - 1) * BPS).dropna()
    return pd.DataFrame([{
        "minutes": len(gap_bps),
        "mean_bps": gap_bps.mean(),
        "median_abs_bps": gap_bps.abs().median(),
        "p99_abs_bps": gap_bps.abs().quantile(0.99),
        "max_abs_bps": gap_bps.abs().max(),
    }])


def volume_share(panel):
    """Daily perp share of the traded value: perp / (perp + spot) quote volume, between 0 and 1."""
    day = pd.to_datetime(panel["open_time"], unit="ms", utc=True).dt.floor("D")
    daily = pd.DataFrame({
        "spot": panel["spot_quote_volume"],
        "perp": panel["perp_quote_volume"],
    }).groupby(day.rename("day")).sum()
    daily["perp_share"] = daily["perp"] / (daily["perp"] + daily["spot"])
    return daily


def intraday_profile(panel):
    """Average activity by UTC hour: quote volume per minute and the std of 1-minute log returns (bps)."""
    hour = ((panel["open_time"] // MS_PER_HOUR) % 24).rename("hour")
    table = pd.DataFrame({
        "spot_volume": panel["spot_quote_volume"],
        "perp_volume": panel["perp_quote_volume"],
        "spot_vol_bps": np.log(panel["spot_vwap"]).diff() * BPS,
        "perp_vol_bps": np.log(panel["perp_vwap"]).diff() * BPS,
    })
    grouped = table.groupby(hour)
    return pd.DataFrame({
        "spot_volume": grouped["spot_volume"].mean(),
        "perp_volume": grouped["perp_volume"].mean(),
        "spot_vol_bps": grouped["spot_vol_bps"].std(),
        "perp_vol_bps": grouped["perp_vol_bps"].std(),
    })