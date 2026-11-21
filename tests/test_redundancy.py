# duplicate block B features: dropped before the models, and caught by the coefficient table
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evaluate import coefficient_table
from src.features import BLOCK_A, BLOCK_B, build_features
from src.models import TARGET, VOL_SCALE, drop_redundant, redundant_features

MINUTE = 60_000


def realistic_panel(rows=3000, seed=0):
    """Minute panel with the columns features.py reads: spot random walk, perp a few bps away."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01 00:05").value // 10**6
    spot = 100 * np.exp(np.cumsum(rng.normal(0, 3e-4, rows)))
    volume, perp_volume = rng.uniform(1, 3, rows), rng.uniform(2, 6, rows)
    return pd.DataFrame({
        "open_time": start + np.arange(rows) * MINUTE,
        "spot_vwap": spot,
        "perp_vwap": spot * (1 + rng.normal(-4e-4, 1e-4, rows)),  # basis around -4 bps
        "spot_volume": volume,
        "spot_signed_volume": volume * rng.uniform(-1, 1, rows),
        "spot_count": pd.array(rng.integers(50, 500, rows), dtype="Int64"),
        "perp_volume": perp_volume,
        "perp_signed_volume": perp_volume * rng.uniform(-1, 1, rows),
        "metrics_time": pd.array(start + (np.arange(rows) // 5) * 5 * MINUTE, dtype="Int64"),
        "sum_open_interest": 1e5 * np.exp(np.cumsum(rng.normal(0, 1e-3, rows))),
        "sum_toptrader_long_short_ratio": rng.uniform(0.8, 1.5, rows),
        "count_long_short_ratio": rng.uniform(0.8, 1.5, rows),
        "sum_taker_long_short_vol_ratio": rng.uniform(0.5, 2, rows),
        "funding_rate": 1e-4,
    })


def standardised(frame, columns):
    values = frame[columns].dropna()
    return (values - values.mean()) / values.std()


def test_drop_redundant_matches_every_look_back_window():
    block_b = ["b_flow_2", "b_flow_div_2", "b_basis_bps", "b_basis_chg_20", "b_lead_20", "b_flow_div_5"]
    assert drop_redundant(block_b) == ["b_flow_2", "b_basis_bps", "b_lead_20"]  # order kept
    assert drop_redundant(["a_ret_1"]) == ["a_ret_1"]


def test_redundant_features_catch_exact_and_near_exact_copies_only():
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=5000), rng.normal(size=5000)
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "correlated": 0.95 * x1 + 0.3 * rng.normal(size=5000),  # strongly correlated but genuine (R² ≈ 0.9)
        "combination": x1 - x2,  # exact copy of other features
        # leaves ~6e-8 of its variance unexplained, like b_lead_L given b_basis_chg_L on the real panel
        "near_copy": 1e4 * (x2 + rng.normal(0, 2.5e-4, 5000)),
        "constant": 3.0,
    })
    assert redundant_features(X) == ["combination", "near_copy", "constant"]


def test_tinas_features_contain_exactly_the_duplicates_we_drop():
    """On a realistic panel, A + B has redundant features; after drop_redundant it has none."""
    data = build_features(realistic_panel(), load_config())
    flagged = redundant_features(standardised(data, BLOCK_A + BLOCK_B))
    # flow divergence is an exact copy; lead return and basis change differ only at second order,
    # so the guard flags whichever of each pair comes second (b_lead_L, after b_basis_chg_L)
    assert set(flagged) == {"b_flow_div_5", "b_flow_div_15", "b_lead_5", "b_lead_15"}
    assert redundant_features(standardised(data, BLOCK_A + drop_redundant(BLOCK_B))) == []
    assert len(drop_redundant(BLOCK_B)) == len(BLOCK_B) - 4  # two windows x two duplicate kinds


def test_coefficient_table_names_duplicates_instead_of_returning_empty_errors():
    rng = np.random.default_rng(2)
    a1, b1 = rng.normal(size=3000), rng.normal(size=3000)
    frame = pd.DataFrame({"a1": a1, "b1": b1, "b_dup": a1 + b1, TARGET: 0.3 * b1 + rng.normal(size=3000),
                          VOL_SCALE: 1.0})
    with pytest.raises(ValueError, match=r"not identified: \['b_dup'\]"):
        coefficient_table(frame, ["a1"], ["b1", "b_dup"], lags=15)
    assert coefficient_table(frame, ["a1"], ["b1"], lags=15)["hac_se"].notna().all()
