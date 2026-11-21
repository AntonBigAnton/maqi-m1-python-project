# walk-forward, metrics, Clark-West, DM
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from src.models import DOWN, FLAT, TARGET, UP, VOL_SCALE, classify, fit_flat_threshold
from src.schemas import MS_PER_MINUTE

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
