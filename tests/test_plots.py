import matplotlib

matplotlib.use("Agg")  # draw without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import numpy as np

from src import plots


def test_save_figure_writes_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    path = plots.save_figure(fig, "test_fig", {"figures_dir": tmp_path})
    assert path.exists()
    assert path.name == "test_fig.png"


def test_colours_are_fixed():
    assert plots.SPOT_COLOUR == "#1f77b4"
    assert plots.PERP_COLOUR == "#ff7f0e"


def test_label_axes_requires_units():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plots.label_axes(ax, "Time", "Price", ("min", ""))

def test_label_axes_rejects_a_single_string():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plots.label_axes(ax, "Time", "Price", "ms")
        

def test_figure_8_writes_the_file(tmp_path):
    lead_lag = pd.DataFrame({
        "signal": ["perp_ret"] * 3 + ["perp_flow"] * 3,
        "lag": [-1, 0, 1] * 2,
        "corr": [0.0, 0.1, 0.3, 0.0, 0.05, 0.02],
        "n": 1000,
        "band": 0.062,
    })
    path = plots.figure_8(lead_lag, {"figures_dir": tmp_path, "symbol": "TESTUSDT"})
    assert path.exists()
    assert path.name == "fig08_lead_lag_TESTUSDT.png"


def test_figures_1_to_3_write_their_files(tmp_path):
    from src import descriptive
    from tests.test_descriptive import market_panel

    cfg = {"figures_dir": tmp_path, "symbol": "TESTUSDT"}
    panel = market_panel()
    paths = [
        plots.figure_1(descriptive.missing_minutes(panel), cfg),
        plots.figure_2(descriptive.volume_share(panel), cfg),
        plots.figure_3(descriptive.intraday_profile(panel), cfg),
    ]
    assert [p.name for p in paths] == [
        "fig01_missing_minutes_TESTUSDT.png", "fig02_volume_share_TESTUSDT.png", "fig03_intraday_TESTUSDT.png",
    ]
    assert all(p.exists() for p in paths)


def test_figures_4_and_5_write_their_files(tmp_path):
    from src import descriptive
    from tests.test_descriptive import market_panel

    cfg = {"figures_dir": tmp_path, "symbol": "TESTUSDT"}
    panel = market_panel()
    paths = [
        plots.figure_4(descriptive.return_samples(panel, 15), cfg),
        plots.figure_5(descriptive.acf_table(panel, max_lag=10), cfg),
    ]
    assert [p.name for p in paths] == ["fig04_returns_TESTUSDT.png", "fig05_acf_TESTUSDT.png"]
    assert all(p.exists() for p in paths)


def test_figures_6_7_and_10_write_their_files(tmp_path):
    from src import descriptive
    from tests.test_descriptive import JAN_1_MS, market_panel

    cfg = {"figures_dir": tmp_path, "symbol": "TESTUSDT"}
    panel = market_panel()
    panel["perp_vwap"] = panel["spot_vwap"] * (1 + np.linspace(0, 1e-3, len(panel)))  # a moving basis
    panel["funding_time"] = pd.array([JAN_1_MS] * len(panel), dtype="Int64")
    panel["funding_rate"] = 0.0001
    panel["metrics_time"] = pd.array(panel["open_time"] // 300_000 * 300_000, dtype="Int64")
    panel["sum_open_interest_value"] = 8e9
    clusters = pd.DataFrame({"start": [JAN_1_MS + 600 * 60_000]})
    features = pd.DataFrame({"a_ret_1": [0.1, 0.2, 0.4], "b_flow_5": [0.3, 0.1, 0.0]})
    paths = [
        plots.figure_6(descriptive.basis_over_time(panel), clusters, descriptive.basis_half_life(panel), cfg),
        plots.figure_7(descriptive.leverage_series(panel), cfg),
        plots.figure_10(features.corr(), cfg),
    ]
    assert [p.name for p in paths] == [
        "fig06_basis_TESTUSDT.png", "fig07_funding_oi_TESTUSDT.png", "fig10_feature_corr_TESTUSDT.png",
    ]
    assert all(p.exists() for p in paths)


def test_figure_9_writes_its_file(tmp_path):
    rows = []
    for side in ("long", "short"):
        for series in ("spot_bps", "basis_bps"):
            for offset in range(-5, 11):
                rows.append({"side": side, "series": series, "offset_min": offset,
                             "mean": 0.1 * offset, "low": 0.1 * offset - 1, "high": 0.1 * offset + 1, "n": 40})
    study = pd.DataFrame(rows)
    path = plots.figure_9(study, study.assign(n=500), {"figures_dir": tmp_path, "symbol": "TESTUSDT"}, suffix="_q0.5")
    assert path.name == "fig09_event_study_TESTUSDT_q0.5.png"
    assert path.exists()

# ---------- Figures 11 and 12 (results) ----------

RESULTS_CFG = {"symbol": "TESTUSDT", "horizon_minutes": 15}


def clark_west_example():
    """Shaped like evaluate.clark_west_table: two models, two folds and the pooled row."""
    rows = []
    for model, gains in (("ridge", [-0.002, 0.001, -0.0005]), ("boosting", [0.003, -0.001, 0.001])):
        for fold, gain in zip([1, 2, "all"], gains):
            rows.append({"model": model, "fold": fold, "rows": 1000, "r2_gain": gain, "cw_stat": 1.0,
                         "p_value": 0.01 if fold == "all" else 0.03 if gain > 0.002 else 0.4, "p_holm": 0.5 if fold == "all" else np.nan})
    return pd.DataFrame(rows)


IMPORTANCE = {"b_small": 1e-5, "b_mid": 2e-5, "b_big": 3e-5}


def interpretation_example():
    """Shaped like evaluate.coefficient_table and permutation_importance_table."""
    coefficients = pd.DataFrame({
        "feature": ["const", "a_ret_5", "b_small", "b_big", "b_mid"],
        "block": [np.nan, "A", "B", "B", "B"],
        "coef": [1e-5, 2e-4, 1e-5, -3e-4, 1e-4],
        "hac_se": 5e-5, "t": 1.0, "p_value": 0.3,
        "ci_low": [0, 1e-4, -9e-5, -4e-4, 0], "ci_high": [2e-5, 3e-4, 1.1e-4, -2e-4, 2e-4],
        "p_holm": [np.nan, np.nan, 0.9, 0.01, 0.4],
    })
    rows = []
    for model in ("ridge", "boosting"):
        for fold in (1, 2, "all"):
            for feature in ("b_small", "b_big", "b_mid", "a_ret_5"):
                rows.append({"model": model, "task": "regression", "fold": fold, "feature": feature,
                             "importance": IMPORTANCE.get(feature, 1e-5), "importance_std": 1e-6,
                             "importance_min": 0.0, "importance_max": 4e-5, "block": "A" if feature.startswith("a_") else "B"})
    return coefficients, pd.DataFrame(rows)


def capture_figure(monkeypatch):
    """Keep the figure plots.save_figure receives, instead of closing it, to check what it shows."""
    kept = {}

    def keep(fig, name, cfg):
        kept["fig"], kept["name"] = fig, name
        return name
    monkeypatch.setattr(plots, "save_figure", keep)
    return kept


def test_figure_11_writes_the_file_named_by_symbol_and_horizon(tmp_path):
    path = plots.figure_11(clark_west_example(), {**RESULTS_CFG, "figures_dir": tmp_path})
    assert path.exists()
    assert path.name == "fig11_forecast_gain_TESTUSDT_h15.png"


def test_figure_11_lists_folds_by_test_month_with_the_pooled_row_last(monkeypatch):
    kept = capture_figure(monkeypatch)
    folds = pd.DataFrame({"fold": [1, 2], "test_first": ["2026-03-01 00:20", "2026-04-01 00:20"]})
    saved = clark_west_example().astype({"fold": str})  # a CSV read back gives the folds as text
    plots.figure_11(saved, RESULTS_CFG, folds=folds)
    labels = [t.get_text() for t in kept["fig"].axes[0].get_yticklabels()]
    assert labels == ["Fold 1: 2026-03", "Fold 2: 2026-04", "All folds (pooled)"]


def test_figure_11_fills_only_the_markers_below_alpha(monkeypatch):
    kept = capture_figure(monkeypatch)
    plots.figure_11(clark_west_example(), RESULTS_CFG)
    filled = [tuple(c.get_facecolor()[0][:3]) != (1.0, 1.0, 1.0) for c in kept["fig"].axes[0].collections]
    assert sum(filled) == 1  # only boosting on fold 1: the pooled rows use Holm p = 0.5, not the raw 0.01
    texts = [t.get_text() for t in kept["fig"].axes[0].texts]
    assert texts.count("Holm p = 0.50") == 2 and "p = 0.01" not in texts


def test_figure_12_writes_the_file_named_by_symbol_and_horizon(tmp_path):
    path = plots.figure_12(*interpretation_example(), {**RESULTS_CFG, "figures_dir": tmp_path})
    assert path.exists()
    assert path.name == "fig12_perp_features_TESTUSDT_h15.png"


def test_figure_12_shows_only_perp_features_largest_coefficient_on_top_in_both_panels(monkeypatch):
    kept = capture_figure(monkeypatch)
    plots.figure_12(*interpretation_example(), RESULTS_CFG)
    left, right = kept["fig"].axes
    labels = [t.get_text() for t in left.get_yticklabels()]
    assert labels == ["b_small", "b_mid", "b_big"]  # barh draws from the bottom up
    assert left.get_shared_y_axes().joined(left, right)  # same rows, labelled once on the left
    assert len(right.patches) == 2 * 3  # one bar per model and perp feature, block A left out
    ridge = sorted(right.patches[:3], key=lambda bar: bar.get_y())  # bottom row first
    assert [round(bar.get_width() / plots.R2_PP, 8) for bar in ridge] == [IMPORTANCE[f] for f in labels]


def test_figures_11_12_draw_from_the_tables_run_ablation_saved(tmp_path):
    from src.config import load_config
    from src.evaluate import run_ablation
    from tests.test_ablation import GRIDS, MODELS, synthetic_panel

    cfg = load_config(data_dir=tmp_path / "data", tables_dir=tmp_path / "tables", figures_dir=tmp_path / "figures")
    run_ablation(cfg, synthetic_panel(), ["a1"], ["b1"], "SYM", "s", "e", models=MODELS, grids=GRIDS)
    paths = plots.figures_11_12(cfg, symbol="SYM")
    assert [p.name for p in paths] == ["fig11_forecast_gain_SYM_h15.png", "fig12_perp_features_SYM_h15.png"]
    assert all(p.exists() for p in paths)
