# walk-forward, metrics, Clark-West, DM
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models import TARGET, VOL_SCALE, classify, fit_flat_threshold
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
