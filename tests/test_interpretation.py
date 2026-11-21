# Figure 12 data: HAC coefficient table and permutation importance of the perp features
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    coefficient_table, holm, fold_data, permutation_importance_table, tuned_params, walk_forward_predictions,
    walk_forward_splits,
)
from src.models import TARGET, VOL_SCALE, fit_model, fit_predict

H = 15
FLAT_SHARE = 1 / 3
BOOSTING = {"boosting": [{"max_depth": 2, "learning_rate": 0.1, "max_iter": 30}]}


def synthetic_frame(seed=0):
    """Six months, one row every 10 min: a1 (block A) and b2 are noise, b1 predicts y with slope 0.3."""
    times = pd.date_range("2026-01-01 00:05", "2026-06-30 23:55", freq="10min").as_unit("ms").asi8
    rng = np.random.default_rng(seed)
    a1, b1, b2 = (rng.normal(size=len(times)) for _ in range(3))
    y = 0.3 * b1 + rng.normal(size=len(times))
    return pd.DataFrame({"open_time": times, "a1": a1, "b1": b1, "b2": b2, TARGET: y, VOL_SCALE: 1.0})


@pytest.fixture(scope="module")
def frame():
    return synthetic_frame()


@pytest.fixture(scope="module")
def folds(frame):
    return walk_forward_splits(frame["open_time"], H)


@pytest.fixture(scope="module")
def tuning(frame, folds):
    return walk_forward_predictions(frame, folds, ["a1"], ["b1", "b2"], FLAT_SHARE, "regression",
                                    models=["ridge", "boosting"], grids=BOOSTING)[1]


@pytest.fixture(scope="module")
def importance(frame, folds, tuning):
    return permutation_importance_table(frame, folds, ["a1"], ["b1", "b2"], FLAT_SHARE, tuning, repeats=3)


def test_fit_model_returns_the_forecast_function_used_by_fit_predict(frame, folds):
    data = fold_data(frame, folds[0].train, folds[0].test, ["a1", "b1"], FLAT_SHARE)
    predict = fit_model("ridge", {"penalty": 1e-3}, data, "regression")
    np.testing.assert_array_equal(predict(data.X_eval), fit_predict("ridge", {"penalty": 1e-3}, data, "regression"))


def test_coefficients_find_the_perp_signal_with_hac_intervals(frame):
    table = coefficient_table(frame, ["a1"], ["b1", "b2"], lags=H).set_index("feature")
    assert list(table.index) == ["const", "a1", "b1", "b2"]
    assert pd.isna(table.loc["const", "block"]) and table["block"].iloc[1:].tolist() == ["A", "B", "B"]
    assert table.loc["b1", "coef"] == pytest.approx(0.3, abs=0.03)  # standardised: slope × sd(b1) = 0.3
    assert table.loc["b1", "ci_low"] > 0 and table.loc["b1", "p_value"] < 1e-6
    assert abs(table.loc["b2", "t"]) < 3  # noise: no strong effect (a 95% interval misses 0 one time in 20)


def test_block_b_p_values_are_holm_adjusted_for_testing_every_block_b_coefficient(frame):
    table = coefficient_table(frame, ["a1"], ["b1", "b2"], lags=H).set_index("feature")
    assert table.loc[["const", "a1"], "p_holm"].isna().all()
    b = table.loc[["b1", "b2"]]
    assert b["p_holm"].tolist() == pytest.approx(list(holm(b["p_value"])))
    assert (b["p_holm"] >= b["p_value"]).all()


def test_coefficients_can_be_restricted_to_some_rows(frame, folds):
    table = coefficient_table(frame, ["a1"], ["b1", "b2"], lags=H, rows=folds[0].train).set_index("feature")
    full = coefficient_table(frame, ["a1"], ["b1", "b2"], lags=H).set_index("feature")
    assert table.loc["b1", "hac_se"] > full.loc["b1", "hac_se"]  # fewer rows, wider interval


def test_permuting_the_perp_signal_hurts_and_permuting_noise_does_not(importance):
    pooled = importance[importance["fold"] == "all"].set_index("feature")
    assert set(pooled.index) == {"b1", "b2"}  # block B by default
    assert pooled.loc["b1", "importance"] > 0.03  # R² lost when b1 is shuffled
    assert abs(pooled.loc["b2", "importance"]) < 0.005
    assert (pooled["block"] == "B").all() and (importance["model"] == "boosting").all()


def test_importance_has_one_row_per_fold_and_feature_plus_pooled(importance):
    per_fold = importance[importance["fold"] != "all"]
    assert len(per_fold) == 4 * 2
    pooled = importance[importance["fold"] == "all"].set_index("feature")
    for feature in ("b1", "b2"):
        values = per_fold.loc[per_fold["feature"] == feature, "importance"]
        assert pooled.loc[feature, "importance"] == pytest.approx(values.mean())
        assert pooled.loc[feature, "importance_std"] == pytest.approx(values.std(ddof=1))


def test_importance_is_reproducible_and_works_for_ridge_and_classes(frame, folds, tuning):
    run = lambda **kw: permutation_importance_table(frame, folds[:1], ["a1"], ["b1", "b2"], FLAT_SHARE, tuning,
                                                    repeats=2, seed=7, **kw)
    pd.testing.assert_frame_equal(run(), run())
    ridge = run(model="ridge", features=["a1", "b1"]).set_index("feature")
    assert ridge.loc["b1", "importance"].iloc[-1] > ridge.loc["a1", "importance"].iloc[-1]

    classes = walk_forward_predictions(frame, folds[:1], ["a1"], ["b1", "b2"], FLAT_SHARE, "classification",
                                       models=["boosting"], grids=BOOSTING)[1]
    table = permutation_importance_table(frame, folds[:1], ["a1"], ["b1", "b2"], FLAT_SHARE, classes,
                                         task="classification", repeats=2)
    pooled = table[table["fold"] == "all"].set_index("feature")
    assert pooled.loc["b1", "importance"] > pooled.loc["b2", "importance"]  # log loss rises most for b1


def test_tuned_params_read_dicts_and_saved_json(tuning):
    params = tuned_params(tuning, "boosting", "A+B", 1)
    saved = tuning.assign(params=tuning["params"].map(json.dumps))
    assert tuned_params(saved, "boosting", "A+B", 1) == params
    with pytest.raises(ValueError, match="No tuned hyperparameters"):
        tuned_params(tuning, "lasso", "A+B", 1)


def test_unknown_features_are_rejected(frame, folds, tuning):
    with pytest.raises(ValueError, match="Not features"):
        permutation_importance_table(frame, folds, ["a1"], ["b1", "b2"], FLAT_SHARE, tuning, features=["oi"])
