"""Shared plotting style: every figure in the project uses these."""
from pathlib import Path

import matplotlib.pyplot as plt

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