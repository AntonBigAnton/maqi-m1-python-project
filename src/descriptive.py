# descriptive statistics (spec section 7): the numbers behind the descriptive figures
"""Descriptive analysis. Functions here compute tables; plots.py draws them.

Lead-lag (block E): does the perp move before spot, and how long does the effect last?
Rows of the panel are consecutive minutes, so shifting a column by k rows moves it by k minutes.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

from src.features import (
    BAR_MS, BPS, DELEV_MOVE_QUANTILE, DELEV_OI_QUANTILE, MS_PER_HOUR, cluster_events, deleveraging_events,
    five_minute_bars,
)
from src.schemas import MS_PER_MINUTE

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


# ---------- Block C: returns and volatility (#21) ----------

def save_table(table, name, cfg):
    """Save a results table as <tables_dir>/<name>.csv (report/tables in config.yaml) and return the path."""
    folder = Path(cfg["tables_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    table.to_csv(path)
    return path


def horizon_returns(panel, horizon):
    """Spot log VWAP returns over `horizon` minutes, one every `horizon` minutes (no overlap)."""
    log_price = np.log(panel["spot_vwap"])
    returns = log_price - log_price.shift(horizon)
    on_grid = (panel["open_time"] // MS_PER_MINUTE) % horizon == 0
    return returns[on_grid].dropna()


def return_samples(panel, horizon):
    """The two return samples of Figure 4: 1-minute and horizon-minute spot returns, in bps."""
    return {
        "1 min": horizon_returns(panel, 1) * BPS,
        f"{horizon} min": horizon_returns(panel, horizon) * BPS,
    }


def return_moments(samples):
    """Moments of each return sample: a normal distribution has skew 0 and excess kurtosis 0."""
    rows = []
    for label, returns in samples.items():
        rows.append({
            "returns": label,
            "n": len(returns),
            "mean_bps": returns.mean(),
            "std_bps": returns.std(),
            "skew": returns.skew(),
            "excess_kurtosis": returns.kurt(),
            "min_bps": returns.min(),
            "max_bps": returns.max(),
        })
    return pd.DataFrame(rows).set_index("returns")


def acf_table(panel, max_lag=60):
    """Autocorrelation of 1-minute spot returns and of their absolute value, lags 1 to max_lag.

    Returns barely autocorrelate (little linear predictability); absolute returns do, for a long
    time (volatility clustering). band is 1.96 / sqrt(N), the 95% band for independent data.
    """
    returns = np.log(panel["spot_vwap"]).diff()
    n = returns.notna().sum()
    return pd.DataFrame({
        "lag": range(1, max_lag + 1),
        "acf_returns": [returns.autocorr(lag) for lag in range(1, max_lag + 1)],
        "acf_abs_returns": [returns.abs().autocorr(lag) for lag in range(1, max_lag + 1)],
        "band": Z_95 / np.sqrt(n),
    })


def five_minute_series(panel):
    """Spot return, basis and flows on 5-minute bars, for tests too slow on 1-minute data.

    VWAP = quote volume / volume over the 5 minutes. A bar with a missing minute is empty, and so is
    the return of a bar whose previous bar is empty.
    """
    bar = (panel["open_time"] // BAR_MS * BAR_MS).rename("bar_time")
    full = BAR_MS // MS_PER_MINUTE  # a bar needs all of its minutes
    columns = ["spot_quote_volume", "spot_volume", "perp_quote_volume", "perp_volume",
               "spot_signed_volume", "perp_signed_volume"]
    sums = panel[columns].groupby(bar).sum(min_count=full)
    spot_vwap = sums["spot_quote_volume"] / sums["spot_volume"]
    perp_vwap = sums["perp_quote_volume"] / sums["perp_volume"]
    log_spot = np.log(spot_vwap)
    return pd.DataFrame({
        "spot_ret": log_spot - log_spot.reindex(log_spot.index - BAR_MS).to_numpy(),
        "basis_bps": (perp_vwap / spot_vwap - 1) * BPS,
        "spot_flow": sums["spot_signed_volume"] / sums["spot_volume"],
        "perp_flow": sums["perp_signed_volume"] / sums["perp_volume"],
    })


def stationarity_tests(series):
    """ADF and KPSS tests on each column (statsmodels).

    ADF: the null is a unit root (non-stationary), so a small p-value means stationary.
    KPSS: the null is stationary, so a small p-value means non-stationary. KPSS p-values are only
    tabulated between 0.01 and 0.1, so values at those bounds mean "at most" / "at least".
    """
    rows = []
    for name in series.columns:
        values = series[name].dropna().to_numpy()
        adf_stat, adf_p, *_ = adfuller(values, autolag="AIC", result_object=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # KPSS warns when its p-value is at a table bound
            kpss_stat, kpss_p, *_ = kpss(values, regression="c", nlags="auto")
        if adf_p < 0.05 and kpss_p > 0.05:
            verdict = "stationary"
        elif adf_p >= 0.05 and kpss_p <= 0.05:
            verdict = "unit root"
        else:
            verdict = "mixed"
        rows.append({"series": name, "n": len(values), "adf_stat": adf_stat, "adf_p": adf_p,
                     "kpss_stat": kpss_stat, "kpss_p": kpss_p, "verdict": verdict})
    return pd.DataFrame(rows).set_index("series")


# ---------- Block D: perp-specific signals (#22) ----------

def basis_over_time(panel):
    """The 1-minute basis in bps, (perp VWAP / spot VWAP - 1) x 10,000, indexed by UTC time."""
    time = pd.to_datetime(panel["open_time"], unit="ms", utc=True)
    basis = (panel["perp_vwap"] / panel["spot_vwap"] - 1) * BPS
    return pd.Series(basis.to_numpy(), index=time, name="basis_bps")


def basis_half_life(panel):
    """AR(1) fit of the 1-minute basis: basis_t = a + phi x basis_{t-1} + noise.

    The half-life ln(0.5) / ln(phi) is how many minutes a deviation from the mean basis takes to
    halve. It is only defined for 0 < phi < 1 (a basis that reverts to its mean).
    """
    basis = (panel["perp_vwap"] / panel["spot_vwap"] - 1) * BPS
    pairs = pd.DataFrame({"now": basis, "before": basis.shift(1)}).dropna()
    before = pairs["before"] - pairs["before"].mean()
    now = pairs["now"] - pairs["now"].mean()
    # OLS slope with an intercept; undefined if the basis never moves
    phi = (before * now).sum() / (before ** 2).sum() if (before ** 2).sum() > 0 else np.nan
    half_life = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan
    return pd.DataFrame([{"n": len(pairs), "mean_bps": basis.mean(), "phi": phi, "half_life_min": half_life}])


def deleveraging_clusters(panel, oi_q=DELEV_OI_QUANTILE):
    """Clusters of potential liquidations on the full sample (descriptive use), from features.py."""
    return cluster_events(deleveraging_events(panel, oi_q))


def leverage_series(panel):
    """Funding rate at each settlement (bps) and open interest at each metrics reading (billion USDT)."""
    funding = panel[["funding_time", "funding_rate"]].dropna().drop_duplicates("funding_time")
    readings = panel[["metrics_time", "sum_open_interest_value"]].dropna().drop_duplicates("metrics_time")
    return {
        "funding_bps": pd.Series(
            (funding["funding_rate"] * BPS).to_numpy(),
            index=pd.to_datetime(funding["funding_time"].astype("int64"), unit="ms", utc=True),
        ),
        "open_interest_bn": pd.Series(
            (readings["sum_open_interest_value"] / 1e9).to_numpy(),
            index=pd.to_datetime(readings["metrics_time"].astype("int64"), unit="ms", utc=True),
        ),
    }


def flow_correlation(panel, lags):
    """Correlation of spot and perp signed flow share over the same window, for each window L."""
    rows = []
    for lag in lags:
        spot = panel["spot_signed_volume"].rolling(lag).sum() / panel["spot_volume"].rolling(lag).sum()
        perp = panel["perp_signed_volume"].rolling(lag).sum() / panel["perp_volume"].rolling(lag).sum()
        pair = pd.DataFrame({"spot": spot, "perp": perp}).dropna()
        rows.append({"window_min": lag, "corr": pair["spot"].corr(pair["perp"]), "n": len(pair)})
    return pd.DataFrame(rows)


def redundant_pairs(corr, threshold=0.9):
    """Pairs of features whose |correlation| is at least threshold: candidates to drop (for Arya)."""
    rows = []
    names = list(corr.columns)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            value = corr.loc[first, second]
            if abs(value) >= threshold:
                rows.append({"feature_1": first, "feature_2": second, "corr": value})
    return pd.DataFrame(rows, columns=["feature_1", "feature_2", "corr"])


# ---------- Block E: event study around suspected deleveraging clusters (#23) ----------

def event_paths(panel, starts, window=(-30, 60)):
    """Spot and basis paths around each event, one row per event and one column per minute offset.

    Time 0 is the end of the event's 5-minute bar (start + 5 min): only what happens after it
    counts as evidence of a lead. Prices are minute closes, so the price at offset k is the close
    of minute 0 + k - 1. spot_bps is the cumulative spot log return since time 0, basis_bps the
    change in basis since time 0, both in bps. An event whose window leaves the panel is dropped.
    """
    offsets = np.arange(window[0], window[1] + 1)
    position = pd.Series(np.arange(len(panel)), index=panel["open_time"].to_numpy())
    log_spot = np.log(panel["spot_close"].to_numpy())
    basis = ((panel["perp_close"] / panel["spot_close"] - 1) * BPS).to_numpy()
    spot_rows, basis_rows = [], []
    for start in starts:
        last_minute_of_bar = start + BAR_MS - MS_PER_MINUTE  # its close is the price at time 0
        if last_minute_of_bar not in position.index:
            continue
        rows = position[last_minute_of_bar] + offsets
        if rows[0] < 0 or rows[-1] >= len(panel):
            continue
        spot_rows.append((log_spot[rows] - log_spot[rows[offsets == 0]]) * BPS)
        basis_rows.append(basis[rows] - basis[rows[offsets == 0]])
    return offsets, np.array(spot_rows).reshape(-1, len(offsets)), np.array(basis_rows).reshape(-1, len(offsets))


def bootstrap_mean(paths, n_boot=1000, seed=0):
    """Mean path and its 95% bootstrap interval, resampling whole events (rows) with replacement."""
    n = len(paths)
    valid = ~np.isnan(paths)
    filled = np.where(valid, paths, 0.0)
    mean = filled.sum(axis=0) / valid.sum(axis=0)
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot)  # n_boot resamples
    boot = (counts @ filled) / (counts @ valid)
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return mean, low, high


def event_study(panel, clusters, window=(-30, 60), n_boot=1000, seed=0):
    """Mean spot and basis paths around clusters, by side, with 95% bootstrap intervals (Figure 9).

    clusters needs start (ms) and side ("long" = price fell, "short" = price rose), as given by
    deleveraging_clusters or placebo_events. Returns one row per side, series and minute offset.
    """
    tables = []
    for side in ("long", "short"):
        starts = clusters.loc[clusters["side"] == side, "start"]
        offsets, spot, basis = event_paths(panel, starts, window)
        if len(spot) == 0:
            continue
        for series, paths in (("spot_bps", spot), ("basis_bps", basis)):
            mean, low, high = bootstrap_mean(paths, n_boot, seed)
            tables.append(pd.DataFrame({"side": side, "series": series, "offset_min": offsets,
                                        "mean": mean, "low": low, "high": high, "n": len(paths)}))
    return pd.concat(tables, ignore_index=True)


def placebo_events(panel, clusters, ret_q=DELEV_MOVE_QUANTILE, exclusion_minutes=60):
    """Placebo events: 5-minute bars with an equally big perp move but no fall in open interest.

    Same move cut-off as the proxy, open interest flat or rising, and more than exclusion_minutes
    away from any cluster. If the cluster paths look like these, a big move alone explains them.
    """
    bars = five_minute_bars(panel)
    big_move = bars["perp_ret"].abs() >= bars["perp_ret"].abs().quantile(ret_q)
    chosen = bars[big_move & (bars["oi_chg"] >= 0)]
    starts = clusters["start"].to_numpy()
    exclusion_ms = exclusion_minutes * MS_PER_MINUTE
    far = np.array([len(starts) == 0 or np.abs(starts - t).min() > exclusion_ms for t in chosen.index], dtype=bool)
    chosen = chosen.loc[far]
    return pd.DataFrame({
        "start": chosen.index.to_numpy(),
        "side": np.where(chosen["perp_ret"] < 0, "long", "short"),
    })