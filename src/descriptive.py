# descriptive statistics (spec section 7): the numbers behind the descriptive figures
"""Descriptive analysis. Functions here compute tables; plots.py draws them.

Lead-lag (block E): does the perp move before spot, and how long does the effect last?
Rows of the panel are consecutive minutes, so shifting a column by k rows moves it by k minutes.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features import BPS

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