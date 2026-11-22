"""Shared plotting style: every figure in the project uses these."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.features import METRICS_EVERY_MINUTES

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


def figure_9(study, placebo, cfg, suffix=""):
    """Figure 9: mean spot return and basis change around suspected deleveraging clusters.

    study and placebo come from descriptive.event_study. Columns: long and short clusters; rows:
    spot return and basis change since time 0 (end of the event bar), with 95% bootstrap bands.
    The grey band before 0 is the event bar itself; only the path after 0 is evidence of a lead.
    suffix tells apart the appendix versions at other thresholds (e.g. "_q0.5").
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
    rows = (("spot_bps", "Spot return since time 0", SPOT_COLOUR), ("basis_bps", "Basis change since time 0", PERP_COLOUR))
    for column, side in enumerate(("long", "short")):
        for row, (series, ylabel, colour) in enumerate(rows):
            ax = axes[row, column]
            table = study[(study["side"] == side) & (study["series"] == series)]
            if table.empty:
                ax.set_title(f"No {side} clusters")
                continue
            ax.axvspan(-METRICS_EVERY_MINUTES, 0, color="grey", alpha=0.15, label="Event bar")
            ax.fill_between(table["offset_min"], table["low"], table["high"], color=colour, alpha=0.25,
                            label="95% bootstrap band")
            ax.plot(table["offset_min"], table["mean"], color=colour,
                    label=f"Clusters (n = {table['n'].iloc[0]})")
            reference = placebo[(placebo["side"] == side) & (placebo["series"] == series)]
            if not reference.empty:
                ax.plot(reference["offset_min"], reference["mean"], color="grey", linestyle="--",
                        label=f"Placebo: big move, no OI drop (n = {reference['n'].iloc[0]})")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.legend(loc="best", fontsize=8)
            label_axes(ax, "Minutes from the end of the event bar", ylabel, ("min", "bps"))
            if row == 0:
                ax.set_title(f"Suspected {side} deleveraging" + (" (price fell)" if side == "long" else " (price rose)"))
                ax.set_xlabel("")
    fig.suptitle(f"{cfg['symbol']}: event study around suspected deleveraging clusters")
    fig.tight_layout()
    return save_figure(fig, f"fig09_event_study_{cfg['symbol']}{suffix}", cfg)

# ---------- Results figures (spec section 10, H1) ----------

# Each model keeps its colour and marker in every results figure, so colour is never the only cue.
# Blue, orange and red are left to spot, perp and clusters.
MODEL_STYLES = {
    "ridge": {"color": "#2ca02c", "marker": "o", "label": "Ridge"},
    "lasso": {"color": "#9467bd", "marker": "s", "label": "Lasso"},
    "boosting": {"color": "#8c564b", "marker": "^", "label": "Boosting"},
}
R2_PP = 100  # R² is shown in percentage points: an R² gain of 0.001 is 0.1 pp
RETURN_BPS = 1e4  # log returns are shown in basis points


def model_style(model):
    """Colour, marker and label of a model; models not in MODEL_STYLES get grey circles."""
    return MODEL_STYLES.get(model, {"color": "grey", "marker": "o", "label": model})


def fold_labels(folds):
    """'Fold n: YYYY-MM' for each fold of evaluate.fold_table (or its saved CSV), keyed by fold number as text."""
    if folds is None:
        return {}
    months = pd.to_datetime(folds["test_first"]).dt.strftime("%Y-%m")
    return {str(fold): f"Fold {fold}: {month}" for fold, month in zip(folds["fold"], months)}


def figure_11(clark_west, cfg, folds=None, alpha=0.05):
    """Figure 11 ★: out-of-sample R² gain of A + B over A per walk-forward fold, with Clark-West p-values.

    clark_west is evaluate.clark_west_table (or its saved CSV): one row per model and fold plus a
    pooled row (fold "all"). Each dot is one model on one test month; right of 0 means the perp
    features helped. The printed p-value is the one-sided Clark-West test of that fold; on the
    pooled row it is the Holm-adjusted p-value that answers H1. Filled markers are below alpha.
    folds (evaluate.fold_table) labels each fold with its test month.
    """
    apply_style()
    table = clark_west.assign(fold=clark_west["fold"].astype(str))
    per_fold = sorted((f for f in table["fold"].unique() if f != "all"), key=int)
    rows = per_fold + ["all"]
    y = {fold: i for i, fold in enumerate(per_fold)}
    y["all"] = len(per_fold) + 0.5  # a small gap sets the pooled answer apart
    labels = {**{f: f"Fold {f}" for f in per_fold}, **fold_labels(folds), "all": "All folds (pooled)"}

    models = list(dict.fromkeys(table["model"]))
    offsets = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else [0.0]
    fig, ax = plt.subplots(figsize=(10, 1.1 * len(rows) + 1.5))
    for offset, model in zip(offsets, models):
        style = model_style(model)
        for row in table[table["model"] == model].itertuples():
            pooled = row.fold == "all"
            p = row.p_holm if pooled else row.p_value
            x, height = row.r2_gain * R2_PP, y[row.fold] + offset
            ax.scatter(x, height, s=90 if pooled else 60, marker=style["marker"], color=style["color"],
                       facecolors=style["color"] if p < alpha else "white", linewidths=1.5, zorder=3)
            text = f"Holm p = {p:.2f}" if pooled else f"p = {p:.2f}"
            ax.annotate(text, (x, height), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(len(per_fold) - 0.25, color="grey", linewidth=0.6, linestyle=":")
    ax.set_yticks([y[f] for f in rows])
    ax.set_yticklabels([labels[f] for f in rows])
    ax.set_ylim(y["all"] + 0.6, -0.6)  # first fold at the top, pooled row at the bottom
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.25 * (right - left))  # room for the p-values right of the dots

    handles = [plt.Line2D([], [], linestyle="", marker=model_style(m)["marker"], color=model_style(m)["color"],
                          markersize=8, label=model_style(m)["label"]) for m in models]
    handles += [plt.Line2D([], [], linestyle="", marker="o", color="black", markersize=8, label=f"p < {alpha:g}"),
                plt.Line2D([], [], linestyle="", marker="o", color="black", markerfacecolor="white", markersize=8,
                           label=f"p ≥ {alpha:g}")]
    ax.legend(handles=handles, loc="best")
    label_axes(ax, "Out-of-sample R² of A + B minus A", "Walk-forward fold", ("pp", "test month"))
    ax.set_title(f"{cfg['symbol']}, {cfg['horizon_minutes']}-min horizon: do perp features improve the forecast?")
    return save_figure(fig, f"fig11_forecast_gain_{cfg['symbol']}_h{cfg['horizon_minutes']}", cfg)


def figure_12(coefficients, importance, cfg, alpha=0.05):
    """Figure 12 ★: which perp features drive the forecast, as coefficients (left) and permutation importance (right).

    coefficients is evaluate.coefficient_table and importance evaluate.permutation_importance_table
    (or their saved CSVs). One row per block B feature, the same order in both panels, sorted by
    the size of the coefficient. Left: in-sample coefficient of the standardised feature (forecast
    15-minute return per standard deviation) with its 95% Newey-West interval and Holm p-value.
    Right: drop in test-month R² when the feature is shuffled, pooled over folds, with the range
    across folds as the whisker, one bar per model. Both are descriptive, not causal.
    """
    apply_style()
    coefs = coefficients[coefficients["block"] == "B"].copy()
    coefs = coefs.reindex(coefs["coef"].abs().sort_values().index)  # largest at the top of a barh
    features = list(coefs["feature"])
    pooled = importance[(importance["fold"].astype(str) == "all") & (importance["block"] == "B")]
    models = list(dict.fromkeys(pooled["model"]))
    positions = np.arange(len(features))

    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(13, 0.45 * len(features) + 2))
    coef = coefs["coef"].to_numpy() * RETURN_BPS
    whiskers = np.vstack([coef - coefs["ci_low"].to_numpy() * RETURN_BPS, coefs["ci_high"].to_numpy() * RETURN_BPS - coef])
    left.barh(positions, coef, xerr=whiskers, color=PERP_COLOUR, alpha=0.8, capsize=3,
              error_kw={"linewidth": 1}, label="Coefficient, 95% HAC interval")
    for position, row in zip(positions, coefs.itertuples()):
        star = " *" if row.p_holm < alpha else ""
        left.annotate(f"p_Holm = {row.p_holm:.2f}{star}", (1.01, position), xycoords=("axes fraction", "data"),
                      va="center", fontsize=8)
    left.axvline(0, color="black", linewidth=0.8)
    left.set_yticks(positions)
    left.set_yticklabels(features)
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)  # below the panel, never over a bar
    label_axes(left, "Forecast return per 1 std of the feature", "Perp feature (block B)", ("bps", "name"))
    left.set_title("OLS coefficients, in-sample on all minutes")

    height = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        style = model_style(model)
        table = pooled[pooled["model"] == model].set_index("feature").reindex(features)
        value = table["importance"].to_numpy() * R2_PP
        spread = np.vstack([value - table["importance_min"].to_numpy() * R2_PP,
                            table["importance_max"].to_numpy() * R2_PP - value])
        shift = (i - (len(models) - 1) / 2) * height
        right.barh(positions + shift, value, height=height, xerr=spread, color=style["color"], alpha=0.8,
                   hatch="//" if i % 2 else None, capsize=2, error_kw={"linewidth": 1},
                   label=f"{style['label']}, range across folds")
    right.axvline(0, color="black", linewidth=0.8)
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)  # below the panel, never over a bar
    label_axes(right, "Drop in test-month R² when shuffled", "", ("pp", "name"))
    right.set_ylabel("")
    right.set_title("Permutation importance, mean over folds")

    fig.suptitle(f"{cfg['symbol']}, {cfg['horizon_minutes']}-min horizon: which perp features carry the signal?")
    fig.tight_layout()
    return save_figure(fig, f"fig12_perp_features_{cfg['symbol']}_h{cfg['horizon_minutes']}", cfg)


def figures_11_12(cfg, symbol=None):
    """Draw figures 11 and 12 from the tables evaluate.run_ablation saved; returns both paths.

    symbol defaults to the config's; the horizon is the config's, as in the table names.
    """
    from src.evaluate import table_path  # here, so the descriptive figures never load the models

    symbol = symbol or cfg["symbol"]
    read = lambda name: pd.read_csv(table_path(cfg["tables_dir"], symbol, cfg["horizon_minutes"], name))
    cfg = {**cfg, "symbol": symbol}
    return (figure_11(read("clark_west"), cfg, folds=read("folds")),
            figure_12(read("coefficients"), read("permutation_importance"), cfg))
