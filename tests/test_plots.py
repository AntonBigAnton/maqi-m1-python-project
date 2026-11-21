import matplotlib

matplotlib.use("Agg")  # draw without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import pytest

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