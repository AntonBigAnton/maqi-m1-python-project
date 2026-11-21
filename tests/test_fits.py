# level 0-1 models and the walk-forward runner: benchmarks, ridge / lasso / logit, tuning, no leakage
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    clark_west_table, fold_data, fold_metrics, predictions_path, results_table, save_predictions,
    walk_forward_predictions, walk_forward_splits,
)
from src.models import DEFAULT_GRIDS, DOWN, FLAT, TARGET, UP, VOL_SCALE, fit_predict

H = 15
FLAT_SHARE = 1 / 3


def synthetic_frame(n_minutes_step=10, beta=0.3, seed=0):
    """A model_frame over six months (one row every 10 min): a1 is noise, b1 predicts y with slope beta."""
    times = pd.date_range("2026-01-01 00:05", "2026-06-30 23:55", freq=f"{n_minutes_step}min").as_unit("ms").asi8
    rng = np.random.default_rng(seed)
    a1, b1 = rng.normal(size=len(times)), rng.normal(size=len(times))
    y = beta * b1 + rng.normal(size=len(times))
    return pd.DataFrame({"open_time": times, "a1": a1, "b1": b1, TARGET: y, VOL_SCALE: 1.0})


@pytest.fixture(scope="module")
def frame():
    return synthetic_frame()


@pytest.fixture(scope="module")
def folds(frame):
    return walk_forward_splits(frame["open_time"], H)


@pytest.fixture(scope="module")
def regression(frame, folds):
    return walk_forward_predictions(frame, folds, ["a1"], ["b1"], FLAT_SHARE, "regression")


@pytest.fixture(scope="module")
def classification(frame, folds):
    return walk_forward_predictions(frame, folds, ["a1"], ["b1"], FLAT_SHARE, "classification")


def small_fold(beta=0.5, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    data = pd.DataFrame({"x": x, TARGET: beta * x + rng.normal(0, 0.5, n), VOL_SCALE: 1.0})
    return fold_data(data, np.arange(n // 2), np.arange(n // 2, n), ["x"], FLAT_SHARE)


def test_benchmarks_forecast_no_move_and_the_training_class_shares():
    data = small_fold()
    assert (fit_predict("zero", {}, data) == 0).all()
    proba = fit_predict("majority", {}, data)
    shares = pd.Series(data.class_train).value_counts(normalize=True)
    assert proba.shape == (len(data.X_eval), 3)
    assert proba[0] == pytest.approx([shares[DOWN], shares[FLAT], shares[UP]])


def test_ridge_and_lasso_recover_the_slope_and_shrink_with_the_penalty():
    data = small_fold()
    slope = lambda forecast: np.polyfit(data.X_eval["x"], forecast, 1)[0] / data.scaler.scale_[0]
    assert slope(fit_predict("ridge", {"penalty": 1e-6}, data)) == pytest.approx(0.5, abs=0.05)
    assert slope(fit_predict("lasso", {"penalty": 1e-6}, data)) == pytest.approx(0.5, abs=0.05)
    assert abs(slope(fit_predict("ridge", {"penalty": 100.0}, data))) < 0.05
    assert np.ptp(fit_predict("lasso", {"penalty": 10.0}, data)) == 0  # every coefficient set to zero


def test_logit_probabilities_follow_the_signal():
    data = small_fold()
    proba = fit_predict("logit", {"C": 1.0}, data)
    assert np.allclose(proba.sum(axis=1), 1)
    high = data.X_eval["x"].to_numpy() > 1
    assert proba[high, 2].mean() > 0.6  # high x: mostly "up"
    assert proba[~high, 2].mean() < proba[high, 2].mean()


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        fit_predict("svm", {}, small_fold())


def test_every_model_feature_set_and_fold_forecasts_its_test_month(regression, frame, folds):
    predictions, tuning = regression
    assert set(predictions["model"]) == {"zero", "ridge", "lasso"}
    assert set(predictions["feature_set"]) == {"A", "A+B"}
    for (model, feature_set), rows in predictions.groupby(["model", "feature_set"]):
        assert list(rows["fold"].unique()) == [1, 2, 3, 4]
        expected = np.concatenate([frame["open_time"].to_numpy()[f.test] for f in folds])
        assert (rows["open_time"].to_numpy() == expected).all()
    assert len(tuning) == 3 * 2 * 4
    assert set(tuning.loc[tuning["model"] == "ridge", "params"].map(lambda p: p["penalty"])) <= {
        p["penalty"] for p in DEFAULT_GRIDS["ridge"]}


def test_perp_signal_shows_up_as_a_gain_from_a_to_a_plus_b(regression):
    predictions, _ = regression
    table = results_table(fold_metrics(predictions, "regression")).set_index(["model", "feature_set"])
    assert table.loc[("zero", "A"), "oos_r2_mean"] == 0
    assert table.loc[("ridge", "A"), "oos_r2_mean"] < 0.01  # a1 is noise
    assert table.loc[("ridge", "A+B"), "oos_r2_mean"] > 0.05  # b1 carries beta = 0.3
    cw = clark_west_table(predictions, ["ridge", "lasso"], lags=H)
    assert cw.loc[cw["fold"] == "all", "h1_supported"].all()


def test_no_signal_means_no_h1_support(folds):
    noise = synthetic_frame(beta=0.0, seed=3)
    predictions, _ = walk_forward_predictions(noise, folds, ["a1"], ["b1"], FLAT_SHARE, "regression", models=["ridge"])
    cw = clark_west_table(predictions, ["ridge"], lags=H)
    assert not cw.loc[cw["fold"] == "all", "h1_supported"].any()


def test_classification_predictions_beat_the_majority_class(classification):
    predictions, tuning = classification
    assert np.allclose(predictions[["p_down", "p_flat", "p_up"]].sum(axis=1), 1)
    table = results_table(fold_metrics(predictions, "classification")).set_index(["model", "feature_set"])
    assert table.loc[("logit", "A+B"), "log_loss_mean"] < table.loc[("majority", "A+B"), "log_loss_mean"]
    assert table.loc[("logit", "A+B"), "balanced_accuracy_mean"] > 0.4
    assert set(tuning["model"]) == {"majority", "logit"}


def test_changing_the_test_month_never_changes_its_forecasts(frame, folds):
    """No leakage: scaler, threshold, hyperparameters and coefficients only see training months."""
    first = folds[0]
    base, _ = walk_forward_predictions(frame, folds[:1], ["a1"], ["b1"], FLAT_SHARE, "regression", models=["ridge"])
    shocked = frame.copy()
    shocked.loc[first.test, ["a1", "b1", TARGET]] *= 5  # a different March
    after, _ = walk_forward_predictions(shocked, folds[:1], ["a1"], ["b1"], FLAT_SHARE, "regression", models=["ridge"])
    # the fitted models are unchanged: each feature set's forecast is the same linear function of
    # March's features, so scaling them by 5 gives after = 5 * base + a constant per feature set
    assert (after["y_true"] == 5 * base["y_true"]).all()
    for feature_set in ("A", "A+B"):
        rows = base["feature_set"] == feature_set
        gap = after.loc[rows, "y_pred"] - 5 * base.loc[rows, "y_pred"]
        assert np.ptp(gap) < 1e-12


def test_tuning_picks_the_better_penalty(frame, folds):
    grids = {"ridge": [{"penalty": 1e3}, {"penalty": 1e-6}]}  # crushing vs nearly none
    _, tuning = walk_forward_predictions(frame, folds, ["a1"], ["b1"], FLAT_SHARE, "regression",
                                         models=["ridge"], grids=grids)
    assert (tuning.loc[tuning["feature_set"] == "A+B", "params"].map(lambda p: p["penalty"]) == 1e-6).all()


def test_hyperparameters_are_chosen_on_the_tune_month_not_the_test_month(folds):
    """Fold 1 tunes on February. With no signal in February, the heavy penalty must win there,
    even though March (the test month) has the signal and would favour the light one."""
    frame = synthetic_frame(seed=4)
    february = pd.to_datetime(frame["open_time"], unit="ms").dt.month == 2
    frame.loc[february, TARGET] = np.random.default_rng(5).normal(size=february.sum())
    grids = {"ridge": [{"penalty": 1e3}, {"penalty": 1e-6}]}
    _, tuning = walk_forward_predictions(frame, folds[:1], ["a1"], ["b1"], FLAT_SHARE, "regression",
                                         models=["ridge"], grids=grids)
    assert tuning.set_index("feature_set").loc["A+B", "params"] == {"penalty": 1e3}


def test_predictions_are_saved_and_read_back(regression, tmp_path):
    predictions, _ = regression
    path = predictions_path(tmp_path, "SYM", "a", "b", H, "regression")
    assert path.name == "SYM_a_b_h15_regression.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(save_predictions(predictions, path)), predictions)


def test_bad_task_is_rejected(frame, folds):
    with pytest.raises(ValueError, match="task"):
        walk_forward_predictions(frame, folds, ["a1"], ["b1"], FLAT_SHARE, "ranking")
