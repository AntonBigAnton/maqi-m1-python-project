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