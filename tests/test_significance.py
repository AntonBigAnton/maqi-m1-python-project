# significance tests: Clark-West (H1), Diebold-Mariano, Newey-West coefficients, Holm correction
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.evaluate import (
    clark_west, clark_west_table, diebold_mariano, hac_coefficients, hac_mean_test, holm, paired_forecasts,
)

H = 15


def overlapping_noise(n, h, rng):
    """Sum of the next h shocks: autocorrelated like overlapping h-minute returns."""
    shocks = rng.normal(size=n + h)
    return np.convolve(shocks, np.ones(h), mode="valid")[:n]


def newey_west_t(x, lags):
    """Hand-written Newey-West t-statistic of the mean (Bartlett weights, no small-sample correction)."""
    n, u = len(x), x - x.mean()
    s = u @ u + 2 * sum((1 - j / (lags + 1)) * (u[j:] @ u[:-j]) for j in range(1, lags + 1))
    return x.mean() / np.sqrt(s / n**2)


def test_hac_mean_test_matches_the_newey_west_formula():
    x = overlapping_noise(2000, H, np.random.default_rng(0)) + 0.3
    t, p = hac_mean_test(x, H)
    assert t == pytest.approx(newey_west_t(x, H), rel=1e-6)
    assert 0 <= p <= 1


def test_hac_mean_test_is_wider_than_the_naive_test_on_overlapping_data():
    x = overlapping_noise(5000, H, np.random.default_rng(1)) + 0.5
    naive_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(hac_mean_test(x, H)[0]) < abs(naive_t) / 2  # overlap shrinks the effective sample ~h-fold


def test_clark_west_rejects_when_block_b_really_predicts():
    rng = np.random.default_rng(2)
    b = rng.normal(size=3000)
    y = 0.2 * b + rng.normal(size=3000)
    stat, p = clark_west(y, np.zeros(3000), 0.2 * b, lags=0)
    assert stat > 3 and p < 0.001


def test_clark_west_has_about_the_right_size_when_block_b_is_noise():
    """With B useless (its estimated coefficient is pure noise), CW should reject about 5% of the time."""
    rng = np.random.default_rng(3)
    rejections = []
    for _ in range(400):
        b, y = rng.normal(size=500), rng.normal(size=500)
        coef = rng.normal(0, 1 / np.sqrt(500))  # estimation noise of a zero coefficient
        rejections.append(clark_west(y, np.zeros(500), coef * b, lags=0)[1] < 0.05)
    assert 0.01 <= np.mean(rejections) <= 0.10


def test_clark_west_corrects_for_the_noise_of_useless_extra_coefficients():
    """B is pure noise: A + B has a clearly higher MSE, but Clark-West's correction makes the test neutral."""
    rng = np.random.default_rng(9)
    y, noise_forecast = rng.normal(size=5000), rng.normal(0, 0.5, 5000)
    mse_t, _ = hac_mean_test(y**2 - (y - noise_forecast) ** 2, lags=0)
    assert mse_t < -10  # a plain MSE comparison says B hurts
    stat, p = clark_west(y, np.zeros(5000), noise_forecast, lags=0)
    assert abs(stat) < 3 and 0.001 < p < 0.999


def test_clark_west_p_value_is_one_sided():
    rng = np.random.default_rng(4)
    b = rng.normal(size=2000)
    y = 0.3 * b + rng.normal(size=2000)
    _, p_good = clark_west(y, np.zeros(2000), 0.3 * b, lags=0)  # B adds the true effect
    _, p_bad = clark_west(y, np.zeros(2000), -0.3 * b, lags=0)  # B adds it with the wrong sign
    assert p_good < 0.01 and p_bad > 0.99


def test_diebold_mariano_sign_symmetry_and_identical_forecasts():
    rng = np.random.default_rng(5)
    loss_good, loss_bad = rng.exponential(1.0, 3000), rng.exponential(1.3, 3000)
    stat, p = diebold_mariano(loss_good, loss_bad, H)
    assert stat < 0 and p < 0.01
    stat_swapped, p_swapped = diebold_mariano(loss_bad, loss_good, H)
    assert stat_swapped == pytest.approx(-stat) and p_swapped == pytest.approx(p)
    assert diebold_mariano(loss_good, loss_good, H) == (0.0, 1.0)  # identical forecasts: no evidence either way
    with pytest.raises(ValueError, match="non-zero constant"):
        diebold_mariano(np.full(100, 2.0), np.ones(100), H)


def test_clark_west_with_identical_forecasts_is_neutral():
    """A lasso can drop every feature of both sets: same forecasts, so no evidence for H1."""
    y = np.random.default_rng(10).normal(size=100)
    assert clark_west(y, np.full(100, 0.1), np.full(100, 0.1), lags=H) == (0.0, 0.5)


def test_tiny_loss_differences_are_not_mistaken_for_constants():
    """Returns are ~1e-3 and A vs A + B forecasts can differ by ~1e-7: CW terms are ~1e-10 but still tested."""
    rng = np.random.default_rng(11)
    b = rng.normal(0, 1e-7, 5000)  # what block B adds to the forecast
    y = rng.normal(0, 1e-3, 5000) + 1000 * b
    stat, p = clark_west(y, np.zeros(5000), b, lags=H)
    assert stat > 3 and p < 0.01


def test_hac_coefficients_recover_the_effect_with_wider_errors_than_ols():
    rng = np.random.default_rng(6)
    n = 5000
    X = pd.DataFrame({"basis": overlapping_noise(n, H, rng) / np.sqrt(H), "noise": rng.normal(size=n)})
    y = 0.5 * X["basis"] + overlapping_noise(n, H, rng) / np.sqrt(H)
    table = hac_coefficients(X, y, H).set_index("feature")
    assert list(table.index) == ["const", "basis", "noise"]
    assert table.loc["basis", "coef"] == pytest.approx(0.5, abs=0.1)
    assert table.loc["basis", "p_value"] < 0.001
    ols_se = sm.OLS(y, sm.add_constant(X)).fit().bse["basis"]
    assert table.loc["basis", "hac_se"] > 2 * ols_se
    assert (table["ci_low"] < table["coef"]).all() and (table["coef"] < table["ci_high"]).all()


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def predictions_for(model, beta, rng, folds=(1, 2, 3, 4), n=1500):
    """A and A + B regression forecasts of one model, B carrying a true effect beta."""
    frames = []
    for fold in folds:
        a, b = rng.normal(size=n), rng.normal(size=n)
        y = 0.1 * a + beta * b + rng.normal(size=n)
        common = {"model": model, "fold": fold, "open_time": np.arange(n), "y_true": y}
        frames.append(pd.DataFrame({**common, "feature_set": "A", "y_pred": 0.1 * a}))
        coef_b = beta + rng.normal(0, 1 / np.sqrt(n))  # estimated with noise
        frames.append(pd.DataFrame({**common, "feature_set": "A+B", "y_pred": 0.1 * a + coef_b * b}))
    return pd.concat(frames, ignore_index=True)


def test_clark_west_table_answers_h1_per_model_with_holm():
    rng = np.random.default_rng(7)
    predictions = pd.concat([predictions_for("ridge", 0.15, rng), predictions_for("boosting", 0.0, rng)])
    table = clark_west_table(predictions, ["ridge", "boosting"], lags=H)
    assert len(table) == 2 * (4 + 1)  # four folds and the pooled row, per model
    pooled = table[table["fold"] == "all"].set_index("model")
    assert pooled.loc["ridge", "h1_supported"] and not pooled.loc["boosting", "h1_supported"]
    assert pooled.loc["ridge", "r2_gain"] > 0
    assert (pooled["p_holm"] >= pooled["p_value"]).all()
    assert table.loc[table["fold"] != "all", "p_holm"].isna().all()
    assert (pooled["rows"] == 4 * 1500).all()


def test_paired_forecasts_reject_mismatched_tables():
    rng = np.random.default_rng(8)
    predictions = predictions_for("ridge", 0.1, rng, folds=(1,), n=10)
    assert len(paired_forecasts(predictions, "ridge")) == 10
    with pytest.raises(ValueError, match="No minute"):
        paired_forecasts(predictions, "lasso")
    broken = predictions.copy()
    broken.loc[broken["feature_set"] == "A+B", "y_true"] += 1
    with pytest.raises(ValueError, match="different targets"):
        paired_forecasts(broken, "ridge")
