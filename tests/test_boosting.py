# level 2 gradient boosting: non-linear effects, same folds and prediction format as level 1
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    REGRESSION_COLUMNS, CLASSIFICATION_COLUMNS, PREDICTION_KEYS, diebold_mariano, fold_metrics, results_table,
    walk_forward_predictions, walk_forward_splits,
)
from src.models import CLASSIFICATION_MODELS, DEFAULT_GRIDS, REGRESSION_MODELS, TARGET, VOL_SCALE

H = 15
FLAT_SHARE = 1 / 3
GRID = {"boosting": [{"max_depth": 3, "learning_rate": 0.1}]}  # one candidate keeps the tests fast


def interaction_frame(seed=0):
    """Six months, one row every 10 min: b1 moves y in the direction of a1's sign (an effect that flips with
    funding), so b1 has no linear effect on average and only a model with interactions can use it."""
    times = pd.date_range("2026-01-01 00:05", "2026-06-30 23:55", freq="10min").as_unit("ms").asi8
    rng = np.random.default_rng(seed)
    a1, b1 = rng.normal(size=len(times)), rng.normal(size=len(times))
    y = b1 * np.sign(a1) + rng.normal(0, 0.5, len(times))
    return pd.DataFrame({"open_time": times, "a1": a1, "b1": b1, TARGET: y, VOL_SCALE: 1.0})


@pytest.fixture(scope="module")
def frame():
    return interaction_frame()


@pytest.fixture(scope="module")
def folds(frame):
    return walk_forward_splits(frame["open_time"], H)


@pytest.fixture(scope="module")
def regression(frame, folds):
    return walk_forward_predictions(frame, folds, ["a1"], ["b1"], FLAT_SHARE, "regression",
                                    models=["ridge", "boosting"], grids=GRID, seed=42)


def test_boosting_is_part_of_both_tasks_with_depth_and_learning_rate_tuned():
    assert "boosting" in REGRESSION_MODELS and "boosting" in CLASSIFICATION_MODELS
    assert {tuple(sorted(p)) for p in DEFAULT_GRIDS["boosting"]} == {("learning_rate", "max_depth")}
    assert len(DEFAULT_GRIDS["boosting"]) > 1


def test_boosting_finds_an_interaction_that_ridge_misses(regression):
    predictions, _ = regression
    table = results_table(fold_metrics(predictions, "regression")).set_index(["model", "feature_set"])
    assert table.loc[("ridge", "A+B"), "oos_r2_mean"] < 0.02
    assert table.loc[("boosting", "A+B"), "oos_r2_mean"] > 0.5
    assert table.loc[("boosting", "A"), "oos_r2_mean"] < 0.02  # a1 alone says nothing about the sign


def test_diebold_mariano_compares_boosting_with_ridge(regression):
    """Non-nested comparison: same feature set, different model."""
    predictions, _ = regression
    rows = predictions[predictions["feature_set"] == "A+B"]
    ridge, boosting = (rows[rows["model"] == m].sort_values(["fold", "open_time"]) for m in ("ridge", "boosting"))
    stat, p = diebold_mariano((boosting["y_true"] - boosting["y_pred"]) ** 2,
                              (ridge["y_true"] - ridge["y_pred"]) ** 2, lags=H)
    assert stat < 0 and p < 0.001  # boosting has the lower loss


def test_predictions_have_the_level_1_format_and_folds(regression, frame, folds):
    predictions, tuning = regression
    assert list(predictions.columns) == PREDICTION_KEYS + REGRESSION_COLUMNS
    ridge = predictions[predictions["model"] == "ridge"].reset_index(drop=True)
    boosting = predictions[predictions["model"] == "boosting"].reset_index(drop=True)
    pd.testing.assert_frame_equal(ridge[["feature_set", "fold", "open_time", "y_true"]],
                                  boosting[["feature_set", "fold", "open_time", "y_true"]])
    assert (tuning.loc[tuning["model"] == "boosting", "params"] == GRID["boosting"][0]).all()


def test_boosting_classifier_beats_the_majority_class(frame, folds):
    predictions, _ = walk_forward_predictions(frame, folds[:2], ["a1"], ["b1"], FLAT_SHARE, "classification",
                                              models=["majority", "boosting"], grids=GRID, seed=42)
    assert list(predictions.columns) == PREDICTION_KEYS + CLASSIFICATION_COLUMNS
    assert np.allclose(predictions[["p_down", "p_flat", "p_up"]].sum(axis=1), 1)
    table = results_table(fold_metrics(predictions, "classification")).set_index(["model", "feature_set"])
    assert table.loc[("boosting", "A+B"), "log_loss_mean"] < table.loc[("majority", "A+B"), "log_loss_mean"] - 0.05
    assert table.loc[("boosting", "A+B"), "balanced_accuracy_mean"] > 0.45


def test_boosting_is_reproducible_and_never_sees_the_test_targets(frame, folds):
    run = lambda data: walk_forward_predictions(data, folds[:1], ["a1"], ["b1"], FLAT_SHARE, "regression",
                                                models=["boosting"], grids=GRID, seed=42)[0]
    base = run(frame)
    pd.testing.assert_frame_equal(run(frame), base)  # same seed, same forecasts
    shocked = frame.copy()
    shocked.loc[folds[0].test, TARGET] = 100.0  # a wildly different March outcome
    np.testing.assert_array_equal(run(shocked)["y_pred"], base["y_pred"])
