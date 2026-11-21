# per-fold metrics: out-of-sample R² vs zero forecast, classification scores, results table
import numpy as np
import pandas as pd
import pytest

from src.evaluate import classification_scores, fold_metrics, oos_r2, results_table
from src.models import DOWN, FLAT, UP


def test_oos_r2_is_zero_for_the_zero_forecast_and_one_when_perfect():
    y = np.array([0.01, -0.02, 0.005, 0.0])
    assert oos_r2(y, np.zeros(4)) == 0
    assert oos_r2(y, y) == 1


def test_oos_r2_is_negative_when_worse_than_doing_nothing():
    y = np.array([0.01, -0.01])
    assert oos_r2(y, -y) == pytest.approx(1 - 4 * 2e-4 / 2e-4)  # = -3
    assert oos_r2(y, [0.005, -0.005]) == pytest.approx(0.75)


def test_perfect_class_probabilities_score_one():
    true = [DOWN, FLAT, UP, UP]
    proba = np.eye(3)[[0, 1, 2, 2]] * 0.97 + 0.01  # rows (0.98, 0.01, 0.01): columns down, flat, up
    scores = classification_scores(true, proba)
    assert scores["balanced_accuracy"] == 1
    assert scores["macro_f1"] == 1
    assert scores["log_loss"] == pytest.approx(-np.log(0.98))


def test_uniform_probabilities_give_log_loss_of_log_3():
    true = [DOWN, FLAT, UP]
    assert classification_scores(true, np.full((3, 3), 1 / 3))["log_loss"] == pytest.approx(np.log(3))


def test_always_predicting_the_biggest_class_is_not_rewarded():
    """80% flat: always 'flat' has 80% accuracy, but balanced accuracy 1/3."""
    true = [FLAT] * 8 + [UP, DOWN]
    proba = np.tile([0.1, 0.8, 0.1], (10, 1))
    scores = classification_scores(true, proba)
    assert scores["balanced_accuracy"] == pytest.approx(1 / 3)
    assert scores["macro_f1"] == pytest.approx((2 * 0.8 / 1.8) / 3)  # F1 of flat = 0.889, others 0


def regression_predictions():
    rng = np.random.default_rng(0)
    rows = []
    for model in ("ridge", "zero"):
        for feature_set in ("A", "A+B"):
            for fold in (1, 2, 3):
                y = rng.normal(size=50)
                pred = 0.0 if model == "zero" else 0.5 * y
                rows.append(pd.DataFrame({
                    "model": model, "feature_set": feature_set, "fold": fold,
                    "open_time": np.arange(50) + 1000 * fold, "y_true": y, "y_pred": pred,
                }))
    return pd.concat(rows, ignore_index=True)


def test_fold_metrics_scores_every_model_feature_set_and_fold():
    predictions = regression_predictions()
    metrics = fold_metrics(predictions, "regression")
    assert len(metrics) == 2 * 2 * 3
    assert (metrics["rows"] == 50).all()
    assert np.allclose(metrics.loc[metrics["model"] == "zero", "oos_r2"], 0)
    assert np.allclose(metrics.loc[metrics["model"] == "ridge", "oos_r2"], 0.75)  # 1 - 0.5²


def test_results_table_has_one_row_per_model_and_feature_set_with_fold_spread():
    metrics = pd.DataFrame({
        "model": "ridge", "feature_set": ["A", "A", "A+B", "A+B"], "fold": [1, 2, 1, 2],
        "rows": 10, "oos_r2": [-0.01, 0.01, 0.02, 0.04],
    })
    table = results_table(metrics).set_index("feature_set")
    assert list(table.index) == ["A", "A+B"]
    assert table.loc["A+B", "oos_r2_mean"] == pytest.approx(0.03)
    assert table.loc["A+B", "oos_r2_std"] == pytest.approx(np.std([0.02, 0.04], ddof=1))
    assert table.loc["A", "oos_r2_min"] == -0.01 and table.loc["A", "oos_r2_max"] == 0.01
    assert (table["folds"] == 2).all()


def test_classification_fold_metrics_and_bad_input():
    predictions = pd.DataFrame({
        "model": "logit", "feature_set": "A", "fold": 1, "open_time": [0, 1, 2],
        "class_true": [DOWN, FLAT, UP], "p_down": 1 / 3, "p_flat": 1 / 3, "p_up": 1 / 3,
    })
    metrics = fold_metrics(predictions, "classification")
    assert metrics.loc[0, "log_loss"] == pytest.approx(np.log(3))
    with pytest.raises(KeyError, match="y_pred"):
        fold_metrics(predictions, "regression")
    with pytest.raises(ValueError, match="task"):
        fold_metrics(predictions, "ranking")
