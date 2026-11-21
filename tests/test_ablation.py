# full ablation run: tables and predictions saved, named by symbol and horizon
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from src.config import load_config
from src.evaluate import (
    clark_west_table, diebold_mariano_table, predictions_path, row_losses, run_ablation, table_path,
)
from src.models import DOWN, FLAT, UP

H = 15
MODELS = {"regression": ["zero", "ridge", "boosting"], "classification": ["majority", "logit", "boosting"]}
GRIDS = {"ridge": [{"penalty": 1e-3}], "logit": [{"C": 1.0}],
         "boosting": [{"max_depth": 2, "learning_rate": 0.1, "max_iter": 10}]}  # small grids keep the test fast
TABLES = ["folds", "permutation_importance", "tuning_regression", "fold_metrics_regression", "results_regression",
          "clark_west", "tuning_classification", "fold_metrics_classification", "results_classification",
          "diebold_mariano", "coefficients"]


def synthetic_panel(seed=0):
    """Three months of minutes (one test fold). b1 at t shifts the next 15 minutes of spot VWAP."""
    times = pd.date_range("2026-01-01 00:05", "2026-03-31 23:59", freq="min").as_unit("ms").asi8
    rng = np.random.default_rng(seed)
    n = len(times)
    a1, b1 = rng.normal(size=n), rng.normal(size=n)
    drift = 2e-4 / H * np.convolve(b1, np.ones(H))[:n]  # minute s carries b1 of the 15 minutes before it
    returns = rng.normal(0, 1e-4, n) + np.concatenate([[0], drift[:-1]])
    return pd.DataFrame({"open_time": times, "spot_vwap": 100 * np.exp(np.cumsum(returns)), "a1": a1, "b1": b1})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = load_config(data_dir=out / "data", tables_dir=out / "tables")
    tables = run_ablation(cfg, synthetic_panel(), ["a1"], ["b1"], "SYM", "s", "e", models=MODELS, grids=GRIDS)
    return cfg, tables


def test_every_table_is_returned_and_saved_by_symbol_and_horizon(run):
    cfg, tables = run
    assert list(tables) == TABLES
    for name in TABLES:
        path = table_path(cfg["tables_dir"], "SYM", H, name)
        assert path.name == f"SYM_h15_{name}.csv"
        assert len(pd.read_csv(path)) == len(tables[name]) > 0


def test_predictions_of_both_tasks_are_saved(run):
    cfg, _ = run
    for task in ("regression", "classification"):
        saved = pd.read_parquet(predictions_path(cfg["data_dir"], "SYM", "s", "e", H, task))
        assert set(saved["model"]) == set(MODELS[task])
        assert set(saved["feature_set"]) == {"A", "A+B"}


def test_h1_table_tests_every_regression_model_but_the_zero_forecast(run):
    _, tables = run
    cw = tables["clark_west"]
    assert set(cw["model"]) == {"ridge", "boosting"}
    pooled = cw[cw["fold"] == "all"].set_index("model")
    assert pooled.loc["ridge", "h1_supported"]  # b1 carries a real effect
    assert pooled["p_holm"].notna().all()


def test_results_and_tuning_tables_cover_every_model_and_feature_set(run):
    _, tables = run
    for task in ("regression", "classification"):
        results = tables[f"results_{task}"]
        assert len(results) == len(MODELS[task]) * 2
        params = tables[f"tuning_{task}"]["params"].map(json.loads)  # saved as readable JSON
        assert params.map(lambda p: isinstance(p, dict)).all()
    assert tables["folds"]["test_month"].tolist() == ["2026-03"]


def test_diebold_mariano_compares_linear_models_with_boosting(run):
    _, tables = run
    dm = tables["diebold_mariano"]
    pairs = set(zip(dm["task"], dm["model_1"], dm["model_2"], dm["feature_set"]))
    assert pairs == {("regression", "ridge", "boosting", "A"), ("regression", "ridge", "boosting", "A+B"),
                     ("classification", "logit", "boosting", "A"), ("classification", "logit", "boosting", "A+B")}
    assert dm["p_value"].between(0, 1).all()


def test_figure_12_tables_cover_the_perp_features(run):
    _, tables = run
    pooled = tables["permutation_importance"].query("fold == 'all'")
    assert pooled["feature"].tolist() == ["b1"] and (pooled["model"] == "boosting").all()
    assert tables["coefficients"]["feature"].tolist() == ["const", "a1", "b1"]


def test_duplicate_block_b_features_are_dropped_before_the_models(tmp_path, caplog):
    panel = synthetic_panel()
    panel["b_flow_div_5"] = panel["b1"] - panel["a1"]  # an exact duplicate, named like features.py does
    cfg = load_config(data_dir=tmp_path / "data", tables_dir=tmp_path / "tables")
    with caplog.at_level("INFO"):
        tables = run_ablation(cfg, panel, ["a1"], ["b1", "b_flow_div_5"], "SYM", "s", "e",
                              models={"regression": ["ridge"], "classification": ["majority"]}, grids=GRIDS)
    assert "b_flow_div_5" in caplog.text and "Dropped" in caplog.text
    assert tables["coefficients"]["feature"].tolist() == ["const", "a1", "b1"]  # identified without the duplicate


def test_another_symbol_does_not_overwrite_the_tables(run):
    cfg, _ = run
    first = table_path(cfg["tables_dir"], "SYM", H, "results_regression").read_text()
    run_ablation(cfg, synthetic_panel(seed=1), ["a1"], ["b1"], "OTHER", "s", "e",
                 models={"regression": ["zero"], "classification": ["majority"]}, grids=GRIDS)
    assert table_path(cfg["tables_dir"], "OTHER", H, "results_regression").exists()
    assert pd.read_csv(table_path(cfg["tables_dir"], "OTHER", H, "permutation_importance")).empty  # no boosting
    assert table_path(cfg["tables_dir"], "SYM", H, "results_regression").read_text() == first


def test_clark_west_table_is_empty_but_well_formed_without_models_to_test():
    table = clark_west_table(pd.DataFrame(columns=["model", "feature_set", "fold", "open_time", "y_true", "y_pred"]),
                             [], lags=H)
    assert table.empty and {"model", "fold", "p_holm", "h1_supported"} <= set(table.columns)


def test_row_log_losses_average_to_sklearns_log_loss():
    predictions = pd.DataFrame({
        "model": "logit", "feature_set": "A", "fold": 1, "open_time": [0, 1, 2],
        "class_true": [DOWN, FLAT, UP], "p_down": [0.7, 0.2, 0.1], "p_flat": [0.2, 0.5, 0.3], "p_up": [0.1, 0.3, 0.6],
    })
    losses = row_losses(predictions, "classification")["loss"]
    assert losses.tolist() == pytest.approx([-np.log(0.7), -np.log(0.5), -np.log(0.6)])
    expected = log_loss(predictions["class_true"], predictions[["p_down", "p_flat", "p_up"]], labels=[DOWN, FLAT, UP])
    assert losses.mean() == pytest.approx(expected)


def test_diebold_mariano_table_skips_pairs_with_a_missing_model():
    predictions = pd.DataFrame({"model": "ridge", "feature_set": "A", "fold": 1, "open_time": [0, 1],
                                "y_true": [0.1, -0.1], "y_pred": [0.0, 0.0]})
    assert diebold_mariano_table(predictions, "regression", lags=H).empty
