# walk-forward, metrics, Clark-West, DM
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from src.models import (
    CLASSIFICATION_MODELS, DEFAULT_GRIDS, DOWN, FLAT, REGRESSION_MODELS, TARGET, UP, VOL_SCALE,
    REDUNDANT_FEATURE_PREFIXES, classify, drop_redundant, feature_sets, fit_flat_threshold, fit_model, fit_predict,
    model_frame, redundant_features,
)
from src.schemas import MS_PER_MINUTE

log = logging.getLogger(__name__)

# Walk-forward validation (spec §9). The window is cut into calendar months (UTC). Fold k tests
# on month k and trains on every month before it (expanding window). Hyperparameters are tuned
# by fitting on the training months but the last and scoring on that last month. Between any
# rows used for fitting and the rows they are scored on, an embargo of at least the horizon
# drops the training rows whose label would reach into the scored period.


@dataclass(frozen=True)
class Fold:
    """Row positions of one walk-forward fold.

    fit / tune: tuning stage (fit on fit, score hyperparameters on tune = last training month).
    train / test: final stage (refit on train = fit + tune months, score on the test month).
    """

    number: int
    test_month: str
    fit: np.ndarray
    tune: np.ndarray
    train: np.ndarray
    test: np.ndarray


def month_starts(open_time):
    """Start (UTC ms) of each calendar month the data covers, plus the start of the month after."""
    times = pd.to_datetime(np.asarray(open_time, dtype="int64"), unit="ms")
    months = pd.period_range(times.min().to_period("M"), times.max().to_period("M") + 1, freq="M")
    starts = months.to_timestamp().as_unit("ms").asi8
    return months[:-1].strftime("%Y-%m").tolist(), starts


def walk_forward_splits(open_time, horizon_minutes, first_test_month=3, embargo_minutes=None):
    """Expanding monthly folds: one per month from first_test_month (1 = first month of the data).

    A training row t has a label that uses prices up to the minute t + h, so it is kept only if
    t + h ends before the scored period starts: rows in the embargo_minutes before it (default
    and minimum h) are dropped. The same rule separates the fit months from the tune month.
    """
    if embargo_minutes is None:
        embargo_minutes = horizon_minutes
    if embargo_minutes < horizon_minutes:
        raise ValueError(f"Embargo ({embargo_minutes} min) must be at least the horizon ({horizon_minutes} min)")
    if first_test_month < 3:
        raise ValueError("first_test_month must be at least 3: one month to fit and one to tune")

    times = np.asarray(open_time, dtype="int64")
    names, starts = month_starts(times)
    if len(names) < first_test_month:
        raise ValueError(f"Walk-forward needs at least {first_test_month} months of data, got {len(names)}")
    embargo = embargo_minutes * MS_PER_MINUTE

    def rows(mask):
        return np.flatnonzero(mask)

    folds = []
    for month in range(first_test_month - 1, len(names)):
        tune_start, test_start, test_end = starts[month - 1], starts[month], starts[month + 1]
        fold = Fold(
            number=len(folds) + 1,
            test_month=names[month],
            fit=rows(times < tune_start - embargo),
            tune=rows((times >= tune_start) & (times < test_start - embargo)),
            train=rows(times < test_start - embargo),
            test=rows((times >= test_start) & (times < test_end)),
        )
        if min(len(fold.fit), len(fold.tune), len(fold.test)) == 0:
            raise ValueError(f"Fold {fold.number} (test {fold.test_month}) has an empty fit, tune or test set")
        folds.append(fold)
    return folds


def fold_table(folds, open_time):
    """One row per fold: sizes and date ranges of each set, for the README and the report."""
    times = pd.to_datetime(np.asarray(open_time, dtype="int64"), unit="ms")
    records = []
    for fold in folds:
        record = {"fold": fold.number, "test_month": fold.test_month}
        for name in ("fit", "tune", "train", "test"):
            rows = getattr(fold, name)
            record[f"{name}_rows"] = len(rows)
            record[f"{name}_first"] = times[rows[0]]
            record[f"{name}_last"] = times[rows[-1]]
        records.append(record)
    return pd.DataFrame(records)


@dataclass
class FoldData:
    """Features and targets of one fitting stage; scaler and class threshold fitted on its training rows."""

    X_train: pd.DataFrame
    y_train: pd.Series
    class_train: np.ndarray
    X_eval: pd.DataFrame
    y_eval: pd.Series
    class_eval: np.ndarray
    flat_threshold: float
    scaler: StandardScaler | None


def fold_data(frame, train_rows, eval_rows, feature_columns, flat_share, scale=True):
    """Split a model_frame into training and scored rows, fitting everything learned from data on training rows only.

    The flat threshold and, if scale is True, a standard scaler (mean 0, std 1 per feature) are
    fitted on the training rows and applied unchanged to the scored rows. Tree models do not need
    scaling: pass scale=False. Tuning stage: fold_data(frame, fold.fit, fold.tune, ...).
    Final stage: fold_data(frame, fold.train, fold.test, ...).
    """
    columns = list(feature_columns)
    train, scored = frame.iloc[train_rows], frame.iloc[eval_rows]
    X_train, X_eval = train[columns], scored[columns]
    scaler = None
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=columns)
        X_eval = pd.DataFrame(scaler.transform(X_eval), index=X_eval.index, columns=columns)
    threshold = fit_flat_threshold(train[TARGET], train[VOL_SCALE], flat_share)
    return FoldData(
        X_train=X_train,
        y_train=train[TARGET],
        class_train=classify(train[TARGET], train[VOL_SCALE], threshold),
        X_eval=X_eval,
        y_eval=scored[TARGET],
        class_eval=classify(scored[TARGET], scored[VOL_SCALE], threshold),
        flat_threshold=threshold,
        scaler=scaler,
    )


# Per-fold metrics (spec §9). Models save their test-month predictions as one row per
# (model, feature set, fold, minute), with these columns:
PREDICTION_KEYS = ["model", "feature_set", "fold", "open_time"]
REGRESSION_COLUMNS = ["y_true", "y_pred"]  # forward return and its forecast
CLASS_PROBABILITIES = {DOWN: "p_down", FLAT: "p_flat", UP: "p_up"}
CLASSIFICATION_COLUMNS = ["class_true", *CLASS_PROBABILITIES.values()]  # label and forecast probabilities


def oos_r2(y_true, y_pred):
    """Out-of-sample R² against the zero-return forecast: 1 - SSE(model) / SSE(predicting 0).

    0 means no better than forecasting no move; negative means worse than doing nothing.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true**2))


def classification_scores(class_true, probabilities):
    """Balanced accuracy, macro F1 and log loss of up / flat / down forecasts.

    probabilities has one column per class in the order down, flat, up. The forecast class is the
    most likely one. Accuracy alone is not reported: it rewards always predicting the biggest class.
    """
    labels = list(CLASS_PROBABILITIES)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    return {
        "balanced_accuracy": balanced_accuracy_score(class_true, predicted),
        "macro_f1": f1_score(class_true, predicted, labels=labels, average="macro", zero_division=0),
        "log_loss": log_loss(class_true, probabilities, labels=labels),
    }


def check_columns(predictions, columns):
    if missing := [c for c in [*PREDICTION_KEYS, *columns] if c not in predictions.columns]:
        raise KeyError(f"Prediction table is missing: {missing}")


def fold_metrics(predictions, task):
    """One row per model, feature set and fold with that fold's test metrics. task is "regression" or "classification"."""
    if task == "regression":
        check_columns(predictions, REGRESSION_COLUMNS)
        score = lambda g: {"oos_r2": oos_r2(g["y_true"], g["y_pred"])}
    elif task == "classification":
        check_columns(predictions, CLASSIFICATION_COLUMNS)
        score = lambda g: classification_scores(g["class_true"], g[list(CLASS_PROBABILITIES.values())])
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
    records = [
        {"model": model, "feature_set": feature_set, "fold": fold, "rows": len(group), **score(group)}
        for (model, feature_set, fold), group in predictions.groupby(["model", "feature_set", "fold"], sort=True)
    ]
    return pd.DataFrame(records)


def results_table(metrics):
    """Mean, standard deviation, min and max of each metric across folds, one row per model and feature set."""
    names = [c for c in metrics.columns if c not in ("model", "feature_set", "fold", "rows")]
    table = metrics.groupby(["model", "feature_set"], sort=True)[names].agg(["mean", "std", "min", "max"])
    table.columns = [f"{name}_{stat}" for name, stat in table.columns]
    table.insert(0, "folds", metrics.groupby(["model", "feature_set"], sort=True)["fold"].nunique())
    return table.reset_index()


# Significance tests (spec §9). Forecasts are issued every minute for a 15-minute horizon, so
# consecutive errors overlap and are autocorrelated: every test uses Newey-West (HAC) standard
# errors with at least h lags.


def hac_mean_test(series, lags):
    """t-statistic of the mean of series and its two-sided normal p-value, with a Newey-West standard error.

    A series of exact zeros (identical forecasts, e.g. a lasso that dropped every feature of both
    feature sets) is no evidence either way: t = 0, p = 1. Exact comparisons only, since loss
    differences of minute returns are of order 1e-7 and a tolerance would erase them.
    """
    series = np.asarray(series, dtype=float)
    if np.isnan(series).any():
        raise ValueError("Series contains missing values")
    if (series == 0).all():
        return 0.0, 1.0
    if np.ptp(series) == 0:
        raise ValueError("Series is a non-zero constant: its standard error is zero")
    fit = sm.OLS(series, np.ones(len(series))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0]), float(fit.pvalues[0])


def clark_west(y_true, pred_small, pred_big, lags):
    """Clark-West test that the bigger of two nested models (A + B) forecasts better than the smaller (A).

    f_t = e_small² - (e_big² - (pred_small - pred_big)²). The correction term removes the noise the
    big model adds by estimating extra coefficients that are zero under the null. One-sided:
    a small p-value means block B improves the forecast. Returns (statistic, p-value).
    """
    y_true, pred_small, pred_big = (np.asarray(a, dtype=float) for a in (y_true, pred_small, pred_big))
    f = (y_true - pred_small) ** 2 - ((y_true - pred_big) ** 2 - (pred_small - pred_big) ** 2)
    t, p_two_sided = hac_mean_test(f, lags)
    return t, p_two_sided / 2 if t > 0 else 1 - p_two_sided / 2


def diebold_mariano(loss_1, loss_2, lags):
    """Diebold-Mariano test of equal forecast accuracy for non-nested models (e.g. linear vs boosting).

    Takes per-row losses (squared errors, or log losses for the classes). Two-sided: a negative
    statistic means model 1 has the lower loss. Returns (statistic, p-value).
    """
    return hac_mean_test(np.asarray(loss_1, dtype=float) - np.asarray(loss_2, dtype=float), lags)


def hac_coefficients(X, y, lags):
    """OLS coefficients with Newey-West standard errors, for the level-1 coefficient table.

    Fit it on standardised training features so coefficients are comparable across features.
    """
    fit = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    ci = fit.conf_int()
    return pd.DataFrame(
        {"coef": fit.params, "hac_se": fit.bse, "t": fit.tvalues, "p_value": fit.pvalues,
         "ci_low": ci.iloc[:, 0], "ci_high": ci.iloc[:, 1]}
    ).rename_axis("feature").reset_index()


def holm(p_values):
    """Holm-adjusted p-values (family-wise error control across model levels), in the input order."""
    return multipletests(np.asarray(p_values, dtype=float), method="holm")[1]


def paired_forecasts(predictions, model, small="A", big="A+B"):
    """One row per fold and minute with the true return and both feature sets' forecasts of one model."""
    check_columns(predictions, REGRESSION_COLUMNS)
    rows = predictions[predictions["model"] == model]
    pairs = rows[rows["feature_set"] == small].merge(
        rows[rows["feature_set"] == big], on=["fold", "open_time"], suffixes=("_small", "_big"), validate="one_to_one"
    )
    if len(pairs) == 0:
        raise ValueError(f"No minute has both {small} and {big} forecasts for model {model!r}")
    if not np.allclose(pairs["y_true_small"], pairs["y_true_big"]):
        raise ValueError("The two feature sets were scored on different targets")
    return pairs.rename(columns={"y_true_small": "y_true"}).sort_values(["fold", "open_time"]).reset_index(drop=True)


def clark_west_table(predictions, models, lags, alpha=0.05):
    """The H1 answer: Clark-West test of A + B against A for each model, per fold and pooled over folds.

    Pooled p-values are Holm-adjusted across models; H1 is supported for a model if its adjusted
    p-value is below alpha. Per-fold rows (for Figure 11) are not adjusted.
    """
    records = []
    for model in models:
        pairs = paired_forecasts(predictions, model)
        groups = [(fold, group) for fold, group in pairs.groupby("fold")] + [("all", pairs)]
        for fold, group in groups:
            stat, p = clark_west(group["y_true"], group["y_pred_small"], group["y_pred_big"], lags)
            gain = oos_r2(group["y_true"], group["y_pred_big"]) - oos_r2(group["y_true"], group["y_pred_small"])
            records.append({"model": model, "fold": fold, "rows": len(group), "r2_gain": gain,
                            "cw_stat": stat, "p_value": p})
    table = pd.DataFrame(records, columns=["model", "fold", "rows", "r2_gain", "cw_stat", "p_value"])
    pooled = table["fold"] == "all"
    table["p_holm"] = np.nan
    table["h1_supported"] = pd.Series(pd.NA, index=table.index, dtype="boolean")
    if pooled.any():  # no model to test (e.g. only the zero forecast) leaves an empty table
        table.loc[pooled, "p_holm"] = holm(table.loc[pooled, "p_value"])
        table.loc[pooled, "h1_supported"] = table.loc[pooled, "p_holm"] < alpha
    return table


# Walk-forward fits (spec §8-9): every model on every feature set and fold, tuned on the tune
# month, refitted on the whole training window and scored on the test month.


def tuning_score(task, data, forecast):
    """Score used to choose hyperparameters on the tune month: higher is better."""
    if task == "regression":
        return oos_r2(data.y_eval, forecast)
    return -log_loss(data.class_eval, forecast, labels=list(CLASS_PROBABILITIES))


def walk_forward_predictions(frame, folds, block_a, block_b, flat_share, task, models=None, grids=None, seed=0):
    """Test-month predictions of every model x feature set x fold, and the hyperparameters chosen.

    frame is a models.model_frame and folds come from walk_forward_splits on its open_time. For
    each fold, every candidate in a model's grid is fitted on the fit months and scored on the
    tune month (R² for regression, log loss for classes); the best is refitted on all training
    months and forecasts the test month. Scaler and class threshold are fitted on the rows each
    model is trained on (fold_data). Returns (predictions, tuning), predictions in the
    PREDICTION_KEYS + REGRESSION_COLUMNS / CLASSIFICATION_COLUMNS format. seed is the config seed.
    """
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
    models = list(models or (REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS))
    grids = {**DEFAULT_GRIDS, **(grids or {})}
    predictions, tuning = [], []
    for feature_set, columns in feature_sets(block_a, block_b).items():
        for fold in folds:
            tune_data = fold_data(frame, fold.fit, fold.tune, columns, flat_share)
            test_data = fold_data(frame, fold.train, fold.test, columns, flat_share)
            keys = {"feature_set": feature_set, "fold": fold.number}
            for model in models:
                scores = [
                    tuning_score(task, tune_data, fit_predict(model, params, tune_data, task, seed))
                    for params in grids[model]
                ]
                best = grids[model][int(np.argmax(scores))]
                tuning.append({"model": model, **keys, "params": best, "tune_score": max(scores)})
                forecast = fit_predict(model, best, test_data, task, seed)
                open_time = frame["open_time"].iloc[fold.test].to_numpy()
                rows = pd.DataFrame({"model": model, **keys, "open_time": open_time})
                if task == "regression":
                    rows["y_true"], rows["y_pred"] = test_data.y_eval.to_numpy(), forecast
                else:
                    rows["class_true"] = test_data.class_eval
                    rows[list(CLASS_PROBABILITIES.values())] = forecast
                predictions.append(rows)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(tuning)


def predictions_path(data_dir, symbol, start, end, horizon_minutes, task):
    """Parquet path of one run's test-month predictions."""
    return Path(data_dir) / "predictions" / f"{symbol}_{start}_{end}_h{horizon_minutes}_{task}.parquet"


def save_predictions(predictions, path):
    """Save a prediction table, creating its folder; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    return path


# Interpretation (spec §8, Figure 12): which perp features drive the gain. Both outputs are
# descriptive, not causal.


def coefficient_table(frame, block_a, block_b, lags, rows=None):
    """OLS coefficients of the A + B model with Newey-West standard errors and 95% intervals.

    Features are standardised on the same rows, so a coefficient is the change in the forecast
    return for a one-standard-deviation move of the feature. In-sample, on every row of the
    model_frame by default (rows restricts it): this is inference on the coefficients, not a
    forecast evaluation. lags should be at least the horizon (overlapping labels).
    """
    columns = feature_sets(block_a, block_b)["A+B"]
    data = frame if rows is None else frame.iloc[rows]
    X = pd.DataFrame(StandardScaler().fit_transform(data[columns]), index=data.index, columns=columns)
    if redundant := redundant_features(X):
        raise ValueError(f"Coefficients are not identified: {redundant} duplicate other features of A + B")
    table = hac_coefficients(X, data[TARGET], lags)
    table.insert(1, "block", table["feature"].map({**{c: "A" for c in block_a}, **{c: "B" for c in block_b}}))
    return table


def tuned_params(tuning, model, feature_set, fold):
    """Hyperparameters chosen for one model, feature set and fold (a dict, or its JSON string in saved tables)."""
    match = tuning[(tuning["model"] == model) & (tuning["feature_set"] == feature_set) & (tuning["fold"] == fold)]
    if len(match) != 1:
        raise ValueError(f"No tuned hyperparameters for {model!r}, {feature_set!r}, fold {fold}")
    params = match["params"].iloc[0]
    return json.loads(params) if isinstance(params, str) else params


def permutation_importance_table(frame, folds, block_a, block_b, flat_share, tuning, task="regression",
                                 model="boosting", features=None, repeats=5, seed=0):
    """Permutation importance of features (block B by default) in the A + B model, per fold and pooled.

    For each fold the model is refitted on the training months with the hyperparameters chosen in
    the walk-forward run (tuning), then each feature is shuffled in the test month, repeats times,
    and the drop in the test score is recorded: out-of-sample R² for regression, log loss (as an
    increase) for classification. A feature the model does not use scores about 0. Shuffling also
    breaks the feature's autocorrelation and its correlation with other features, so correlated
    features share their importance.
    """
    columns = feature_sets(block_a, block_b)["A+B"]
    features = list(block_b if features is None else features)
    if missing := [f for f in features if f not in columns]:
        raise ValueError(f"Not features of A + B: {missing}")
    rng = np.random.default_rng(seed)
    records = []
    for fold in folds:
        data = fold_data(frame, fold.train, fold.test, columns, flat_share)
        predict = fit_model(model, tuned_params(tuning, model, "A+B", fold.number), data, task, seed)
        base = tuning_score(task, data, predict(data.X_eval))
        for feature in features:
            drops = []
            for _ in range(repeats):
                shuffled = data.X_eval.copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                drops.append(base - tuning_score(task, data, predict(shuffled)))
            records.append({"fold": fold.number, "feature": feature, "importance": float(np.mean(drops))})
    per_fold = pd.DataFrame(records, columns=["fold", "feature", "importance"])
    pooled = per_fold.groupby("feature", sort=False)["importance"].agg(["mean", "std", "min", "max"])
    pooled = pooled.reset_index().rename(columns={"mean": "importance", "std": "importance_std",
                                                  "min": "importance_min", "max": "importance_max"})
    pooled["fold"] = "all"
    table = pd.concat([per_fold, pooled], ignore_index=True)
    table.insert(0, "model", model)
    table.insert(1, "task", task)
    table["block"] = table["feature"].map({**{c: "A" for c in block_a}, **{c: "B" for c in block_b}})
    return table


# Full ablation run: one call from the panel with features to every modelling table of the report.

# Non-nested comparisons tested with Diebold-Mariano: each linear model against boosting.
DM_PAIRS = {"regression": [("ridge", "boosting"), ("lasso", "boosting")], "classification": [("logit", "boosting")]}


def row_losses(predictions, task):
    """Loss of every prediction row: squared error for regression, log loss of the true class for classification."""
    keys = predictions[PREDICTION_KEYS].copy()
    if task == "regression":
        keys["loss"] = (predictions["y_true"] - predictions["y_pred"]) ** 2
    else:
        probabilities = predictions[list(CLASS_PROBABILITIES.values())].to_numpy(dtype=float)
        true_column = predictions["class_true"].map({c: i for i, c in enumerate(CLASS_PROBABILITIES)}).to_numpy()
        p_true = probabilities[np.arange(len(predictions)), true_column]
        keys["loss"] = -np.log(np.clip(p_true, 1e-15, 1))  # same clipping as a zero probability in log_loss
    return keys


def diebold_mariano_table(predictions, task, lags, pairs=None):
    """Diebold-Mariano test of each (model 1, model 2) pair on each feature set, pooled over folds.

    A negative statistic means model 1 has the lower loss. Pairs with a model missing from the
    predictions are skipped.
    """
    losses = row_losses(predictions, task)
    present = set(losses["model"])
    records = []
    for model_1, model_2 in pairs if pairs is not None else DM_PAIRS[task]:
        if not {model_1, model_2} <= present:
            continue
        for feature_set in sorted(losses["feature_set"].unique()):
            rows = losses[losses["feature_set"] == feature_set]
            pair = rows[rows["model"] == model_1].merge(
                rows[rows["model"] == model_2], on=["fold", "open_time"], suffixes=("_1", "_2"), validate="one_to_one"
            )
            stat, p = diebold_mariano(pair["loss_1"], pair["loss_2"], lags)
            records.append({"task": task, "model_1": model_1, "model_2": model_2, "feature_set": feature_set,
                            "rows": len(pair), "mean_loss_diff": float((pair["loss_1"] - pair["loss_2"]).mean()),
                            "dm_stat": stat, "p_value": p})
    return pd.DataFrame(records, columns=["task", "model_1", "model_2", "feature_set", "rows", "mean_loss_diff",
                                          "dm_stat", "p_value"])


def table_path(tables_dir, symbol, horizon_minutes, name):
    """CSV path of one results table, named by symbol and horizon so other runs never overwrite it."""
    return Path(tables_dir) / f"{symbol}_h{horizon_minutes}_{name}.csv"


def run_ablation(cfg, data, block_a, block_b, symbol, start, end, models=None, grids=None):
    """Run the whole ablation (spec §8-9) and save its predictions and tables; returns the tables by name.

    data is the panel with the feature columns of block_a and block_b added. Every model runs on
    A and A + B over the walk-forward folds, for both tasks. Predictions go to
    data/predictions/ and these tables to tables_dir as <symbol>_h<horizon>_<name>.csv:
    folds, tuning_<task>, fold_metrics_<task>, results_<task>, clark_west (the H1 test, on the
    regression models other than the zero forecast), diebold_mariano (linear vs boosting),
    coefficients (HAC coefficients of the A + B model, in-sample) and permutation_importance
    (block B features in the A + B boosting regression; empty if boosting was not run).
    models and grids ({task: [model names]}, {model: grid}) restrict the run, e.g. in tests.
    Block B features that duplicate others (models.REDUNDANT_FEATURE_PREFIXES) are dropped first.
    Nothing is cached: results are always rebuilt from the data passed in.
    """
    horizon = cfg["horizon_minutes"]
    if dropped := [f for f in block_b if f not in drop_redundant(block_b)]:
        log.info("Dropped block B features that duplicate others (%s): %s", ", ".join(REDUNDANT_FEATURE_PREFIXES),
                 ", ".join(dropped))
        block_b = drop_redundant(block_b)
    frame = model_frame(data, block_a, block_b, horizon, cfg["target"]["vol_window_minutes"])
    folds = walk_forward_splits(
        frame["open_time"], horizon, cfg["walk_forward"]["first_test_month"], cfg["walk_forward"]["embargo_minutes"]
    )
    log.info("Ablation: %d usable minutes of %d, %d folds", len(frame), len(data), len(folds))
    tables = {"folds": fold_table(folds, frame["open_time"])}
    comparisons = []
    for task in ("regression", "classification"):
        predictions, tuning = walk_forward_predictions(
            frame, folds, block_a, block_b, cfg["target"]["flat_share"], task,
            models=(models or {}).get(task), grids=grids, seed=cfg["seed"],
        )
        save_predictions(predictions, predictions_path(cfg["data_dir"], symbol, start, end, horizon, task))
        if task == "regression":
            # without boosting in the run there is nothing to permute: no folds give an empty table
            boosting_folds = folds if "boosting" in set(predictions["model"]) else []
            tables["permutation_importance"] = permutation_importance_table(
                frame, boosting_folds, block_a, block_b, cfg["target"]["flat_share"], tuning, seed=cfg["seed"]
            )
        tuning["params"] = tuning["params"].map(lambda params: json.dumps(params, sort_keys=True))
        metrics = fold_metrics(predictions, task)
        tables.update({f"tuning_{task}": tuning, f"fold_metrics_{task}": metrics,
                       f"results_{task}": results_table(metrics)})
        if task == "regression":
            nested = [m for m in predictions["model"].unique() if m != "zero"]
            tables["clark_west"] = clark_west_table(predictions, nested, lags=horizon)
        comparisons.append(diebold_mariano_table(predictions, task, lags=horizon))
    tables["diebold_mariano"] = pd.concat(comparisons, ignore_index=True)
    tables["coefficients"] = coefficient_table(frame, block_a, block_b, lags=horizon)

    for name, table in tables.items():
        path = table_path(cfg["tables_dir"], symbol, horizon, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    for row in tables["clark_west"].query("fold == 'all'").itertuples():
        log.info("H1 (Clark-West, A + B vs A), %s: Holm p = %.4f, supported: %s",
                 row.model, row.p_holm, row.h1_supported)
    return tables
