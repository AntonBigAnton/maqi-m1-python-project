"""Shared plotting style: every figure in the project uses these."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Fixed colours
SPOT_COLOUR = "#1f77b4"  # blue
PERP_COLOUR = "#ff7f0e"  # orange

# Style for marking liquidation clusters on a chart: ax.scatter(x, y, **CLUSTER_MARKER)
CLUSTER_MARKER = {"marker": "x", "color": "#d62728", "s": 40, "zorder": 3}


def apply_style():
    """Set the same look for all figures. Call once before plotting."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_figure(fig, name, cfg):
    """Save fig as <figures_dir>/<name>.png and return the file path."""
    folder = Path(cfg["figures_dir"])
    folder.mkdir(parents=True, exist_ok=True)  # create the folder if missing
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # free memory
    return path


def label_axes(ax, x, y, unit):
    """Label both axes with units, e.g. label_axes(ax, "Time", "Price", ("min", "USDT")).

    Raises an error if a unit is missing, so no figure goes out without units.
    """
    if not isinstance(unit, (tuple, list)) or len(unit) != 2:
        raise ValueError("unit must be a pair (x_unit, y_unit), e.g. ('min', 'bps').")

    x_unit, y_unit = unit
    if not x_unit or not y_unit:
        raise ValueError("Every axis needs a unit, e.g. ('min', 'USDT').")
    ax.set_xlabel(f"{x} [{x_unit}]")
    ax.set_ylabel(f"{y} [{y_unit}]")


# ---------- Figures (spec section 10) ----------

SIGNAL_TITLES = {
    "perp_ret": "Perp return",
    "perp_flow": "Perp flow share",
    "basis_bps": "Basis",
}


def figure_8(lead_lag, cfg):
    """Figure 8: cross-correlation of perp signals with future spot returns, lags -30 to +30 min.

    lead_lag is the table from descriptive.lead_lag_table. One panel per perp signal; the grey
    band is ±1.96 / sqrt(N), too narrow because the series are autocorrelated.
    """
    apply_style()
    signals = list(dict.fromkeys(lead_lag["signal"]))
    fig, axes = plt.subplots(len(signals), 1, sharex=True, squeeze=False, figsize=(10, 3 * len(signals)))
    axes = axes[:, 0]
    for ax, signal in zip(axes, signals):
        table = lead_lag[lead_lag["signal"] == signal]
        ax.fill_between(table["lag"], -table["band"], table["band"], color="grey", alpha=0.3,
                        label="95% band if independent")
        ax.plot(table["lag"], table["corr"], color=PERP_COLOUR, marker="o", markersize=3,
                label=f"{SIGNAL_TITLES.get(signal, signal)} at t, spot return at t + lag")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(SIGNAL_TITLES.get(signal, signal))
        ax.legend(loc="best")
        label_axes(ax, "Lag (positive: perp leads spot)", "Correlation", ("min", "–"))
    for ax in axes[:-1]:
        ax.set_xlabel("")  # the panels share the x axis: label it once, at the bottom
    fig.suptitle(f"{cfg['symbol']}: perp signals vs future spot returns")
    fig.tight_layout()
    return save_figure(fig, f"fig08_lead_lag_{cfg['symbol']}", cfg)


def figure_1(missing, cfg):
    """Figure 1: missing minutes per day (rows) and UTC hour (columns), from descriptive.missing_minutes."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # 0 is always white, even when no minute is missing
    worst = max(1, int(missing.to_numpy().max()))
    image = ax.imshow(missing.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=worst, interpolation="nearest")

    ax.set_xticks(range(0, 24, 3))
    days = list(missing.index)
    step = max(1, len(days) // 12)
    ax.set_yticks(range(0, len(days), step))
    ax.set_yticklabels([str(d) for d in days[::step]])
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Missing minutes [count]")
    label_axes(ax, "Hour of day", "Day", ("UTC", "date"))
    ax.set_title(f"{cfg['symbol']}: missing minutes (spot or perp)")
    return save_figure(fig, f"fig01_missing_minutes_{cfg['symbol']}", cfg)


def figure_2(share, cfg):
    """Figure 2: daily perp share of traded value, from descriptive.volume_share."""
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(share.index, share["perp_share"] * 100, color=PERP_COLOUR, label="Perp share")
    ax.axhline(50, color="black", linewidth=0.8, linestyle="--", label="Equal split")
    ax.legend(loc="best")
    label_axes(ax, "Day", "Perp share of traded value", ("UTC", "%"))
    ax.set_title(f"{cfg['symbol']}: perp vs spot traded value")
    return save_figure(fig, f"fig02_volume_share_{cfg['symbol']}", cfg)


def figure_3(profile, cfg):
    """Figure 3: volume and volatility by UTC hour for spot and perp, from descriptive.intraday_profile."""
    apply_style()
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    top.plot(profile.index, profile["spot_volume"] / 1e6, color=SPOT_COLOUR, marker="o", label="Spot")
    top.plot(profile.index, profile["perp_volume"] / 1e6, color=PERP_COLOUR, marker="s", label="Perp")
    label_axes(top, "Hour of day", "Traded value per minute", ("UTC", "million USDT"))
    top.set_xlabel("")
    top.legend(loc="best")
    bottom.plot(profile.index, profile["spot_vol_bps"], color=SPOT_COLOUR, marker="o", label="Spot")
    bottom.plot(profile.index, profile["perp_vol_bps"], color=PERP_COLOUR, marker="s", label="Perp")
    label_axes(bottom, "Hour of day", "Std of 1-min returns", ("UTC", "bps"))
    bottom.legend(loc="best")
    fig.suptitle(f"{cfg['symbol']}: intraday activity")
    fig.tight_layout()
    return save_figure(fig, f"fig03_intraday_{cfg['symbol']}", cfg)


def figure_4(samples, cfg):
    """Figure 4: histograms with a normal overlay (top) and QQ plots (bottom), one column per return sample.

    samples comes from descriptive.return_samples. The histograms show the central 99.8% of returns,
    so the fat tails are visible in the QQ plots; the moments table uses every return.
    """
    apply_style()
    fig, axes = plt.subplots(2, len(samples), squeeze=False, figsize=(5 * len(samples), 8))
    for column, (label, returns) in enumerate(samples.items()):
        hist, qq = axes[0, column], axes[1, column]
        low, high = returns.quantile([0.001, 0.999])
        hist.hist(returns, bins=100, range=(low, high), density=True,
                  color=SPOT_COLOUR, alpha=0.6, label="Spot returns")
        grid = np.linspace(low, high, 400)
        hist.plot(grid, stats.norm.pdf(grid, returns.mean(), returns.std()), color="black",
                  label="Normal, same mean and std")
        hist.set_title(f"{label} returns")
        hist.legend(loc="best")
        label_axes(hist, "Return", "Density", ("bps", "1/bps"))
        (theoretical, ordered), (slope, intercept, _) = stats.probplot(returns, dist="norm")
        qq.scatter(theoretical, ordered, s=3, color=SPOT_COLOUR, label="Spot returns")
        qq.plot(theoretical, slope * theoretical + intercept, color="black", label="Normal")
        qq.legend(loc="best")
        label_axes(qq, "Normal quantile", "Return quantile", ("std", "bps"))
    fig.suptitle(f"{cfg['symbol']}: return distributions")
    fig.tight_layout()
    return save_figure(fig, f"fig04_returns_{cfg['symbol']}", cfg)


def figure_5(acf, cfg):
    """Figure 5: autocorrelation of 1-minute returns vs absolute returns, from descriptive.acf_table."""
    apply_style()
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for ax, column, title in ((top, "acf_returns", "Returns"), (bottom, "acf_abs_returns", "Absolute returns")):
        ax.bar(acf["lag"], acf[column], color=SPOT_COLOUR, label=title)
        ax.fill_between(acf["lag"], -acf["band"], acf["band"], color="grey", alpha=0.3,
                        label="95% band if independent")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.legend(loc="best")
        label_axes(ax, "Lag", "Autocorrelation", ("min", "–"))
    top.set_xlabel("")
        # Lag 1 of returns is large and mechanical (VWAP averaging): zoom on the other lags and label lag 1
    limit = 1.5 * max(acf.loc[acf["lag"] > 1, "acf_returns"].abs().max(), acf["band"].max())
    first = acf.loc[acf["lag"] == 1, "acf_returns"].iloc[0]
    if abs(first) > limit:
        top.set_ylim(-limit, limit)
        top.annotate(f"lag 1: {first:.2f} (off scale)", xy=(1, np.sign(first) * limit),
                     xytext=(5, 0.7 * np.sign(first) * limit), arrowprops={"arrowstyle": "->"})
    fig.suptitle(f"{cfg['symbol']}: autocorrelation of 1-minute spot returns")
    fig.tight_layout()
    return save_figure(fig, f"fig05_acf_{cfg['symbol']}", cfg)


def figure_6(basis, clusters, half_life, cfg):
    """Figure 6: basis over time (5-minute means) with suspected deleveraging clusters marked.

    basis comes from descriptive.basis_over_time, clusters from descriptive.deleveraging_clusters,
    half_life from descriptive.basis_half_life.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    smooth = basis.resample("5min").mean()
    ax.plot(smooth.index, smooth, color=PERP_COLOUR, linewidth=0.6, label="Basis, 5-min mean")
    starts = pd.to_datetime(clusters["start"], unit="ms", utc=True)
    marked = smooth.reindex(starts.dt.floor("5min"))
    ax.scatter(marked.index, marked, **CLUSTER_MARKER, label="Suspected deleveraging cluster")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="best")
    label_axes(ax, "Time", "Basis", ("UTC", "bps"))
    row = half_life.iloc[0]
    ax.set_title(f"{cfg['symbol']}: basis over time (AR(1) half-life {row['half_life_min']:.1f} min)")
    return save_figure(fig, f"fig06_basis_{cfg['symbol']}", cfg)


def figure_7(leverage, cfg):
    """Figure 7: funding rate (top) and open interest (bottom) on the same time axis."""
    apply_style()
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    funding = leverage["funding_bps"]
    top.step(funding.index, funding, where="post", color=PERP_COLOUR, label="Funding rate")
    top.axhline(0, color="black", linewidth=0.8)
    top.legend(loc="best")
    label_axes(top, "Time", "Funding rate", ("UTC", "bps per settlement"))
    top.set_xlabel("")
    open_interest = leverage["open_interest_bn"]
    bottom.plot(open_interest.index, open_interest, color=PERP_COLOUR, linewidth=0.8, label="Open interest")
    bottom.legend(loc="best")
    label_axes(bottom, "Time", "Open interest", ("UTC", "billion USDT"))
    fig.suptitle(f"{cfg['symbol']}: funding and open interest")
    fig.tight_layout()
    return save_figure(fig, f"fig07_funding_oi_{cfg['symbol']}", cfg)


def figure_10(corr, cfg):
    """Figure 10: correlation matrix of the block A and B features (from features.build_features)."""
    apply_style()
    size = max(8, 0.35 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Correlation [–]")
    ax.set_title(f"{cfg['symbol']}: correlation of block A and B features")
    fig.tight_layout()
    return save_figure(fig, f"fig10_feature_corr_{cfg['symbol']}", cfg)