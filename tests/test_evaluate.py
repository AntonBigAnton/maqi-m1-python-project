# walk-forward splitter: monthly expanding folds, embargo, tuning month, train-only threshold
import numpy as np
import pandas as pd
import pytest

from src.evaluate import fold_data, fold_table, walk_forward_splits
from src.models import TARGET, VOL_SCALE

MINUTE = 60_000
H = 15


def minute_grid(start="2026-01-01 00:05", end="2026-06-30 23:59"):
    """open_time of every minute of a six-month window, starting 5 minutes in like the panel."""
    return pd.date_range(start, end, freq="min").as_unit("ms").asi8


def month_of(times_ms):
    return pd.to_datetime(times_ms, unit="ms").strftime("%Y-%m")


@pytest.fixture(scope="module")
def times():
    return minute_grid()


@pytest.fixture(scope="module")
def folds(times):
    return walk_forward_splits(times, H)


def test_four_folds_testing_months_three_to_six(folds, times):
    assert [f.test_month for f in folds] == ["2026-03", "2026-04", "2026-05", "2026-06"]
    for fold in folds:
        assert set(month_of(times[fold.test])) == {fold.test_month}
    # the test sets cover months 3 to 6 completely, with no row tested twice
    tested = np.concatenate([f.test for f in folds])
    assert len(tested) == len(set(tested)) == (times >= pd.Timestamp("2026-03-01").value // 10**6).sum()


def test_training_window_expands_and_starts_at_the_first_month(folds):
    for fold in folds:
        assert fold.train[0] == 0
    assert all(len(a.train) < len(b.train) for a, b in zip(folds, folds[1:]))


def test_no_training_label_reaches_the_scored_period(folds, times):
    """A label at t uses the VWAP of minute t + h, which ends at t + h + 1 min."""
    for fold in folds:
        for train, scored in ((fold.train, fold.test), (fold.fit, fold.tune)):
            last_label_end = times[train].max() + (H + 1) * MINUTE
            assert last_label_end <= times[scored].min()


def test_embargo_drops_exactly_h_minutes_and_no_more(folds, times):
    for fold in folds:
        gap = times[fold.test].min() - times[fold.train].max()
        assert gap == (H + 1) * MINUTE  # H dropped minutes between the last train and first test row


def test_tuning_uses_the_last_training_month_only(folds, times):
    for fold in folds:
        test_month = pd.Period(fold.test_month)
        assert set(month_of(times[fold.tune])) == {str(test_month - 1)}
        assert set(month_of(times[fold.fit])) <= {str(m) for m in pd.period_range("2026-01", test_month - 2)}
        # fit and tune are inside train, disjoint, and nothing from them is in test
        assert set(fold.fit) | set(fold.tune) <= set(fold.train)
        assert not set(fold.fit) & set(fold.tune)
        assert not set(fold.train) & set(fold.test)


def test_first_fold_fits_on_month_one_and_tunes_on_month_two(folds, times):
    first = folds[0]
    assert set(month_of(times[first.fit])) == {"2026-01"}
    assert set(month_of(times[first.tune])) == {"2026-02"}


def test_longer_embargo_and_other_horizons(times):
    fold = walk_forward_splits(times, 5, embargo_minutes=60)[0]
    assert times[fold.test].min() - times[fold.train].max() == 61 * MINUTE


def test_splitter_rejects_short_embargo_and_short_data(times):
    with pytest.raises(ValueError, match="at least the horizon"):
        walk_forward_splits(times, H, embargo_minutes=H - 1)
    with pytest.raises(ValueError, match="at least 3 months"):
        walk_forward_splits(minute_grid(end="2026-01-31 23:59"), H)  # a --quick run
    with pytest.raises(ValueError, match="first_test_month"):
        walk_forward_splits(times, H, first_test_month=2)


def test_splitter_works_on_a_table_with_dropped_rows(times):
    """model_frame drops minutes with missing values; folds are cut by time, not by position."""
    kept = times[np.random.default_rng(0).random(len(times)) > 0.1]
    for fold in walk_forward_splits(kept, H):
        assert kept[fold.train].max() + (H + 1) * MINUTE <= kept[fold.test].min()


def test_fold_table_reports_sizes_and_ranges(folds, times):
    table = fold_table(folds, times)
    assert list(table["fold"]) == [1, 2, 3, 4]
    assert (table["fit_rows"] + table["tune_rows"] <= table["train_rows"]).all()
    assert table.loc[0, "test_first"] == pd.Timestamp("2026-03-01")


def test_fold_data_fits_the_threshold_on_training_rows_only():
    rng = np.random.default_rng(2)
    n = 1000
    frame = pd.DataFrame(
        {"a1": rng.normal(size=n), "b1": rng.normal(size=n), TARGET: rng.normal(0, 1, n), VOL_SCALE: 1.0}
    )
    train, scored = np.arange(600), np.arange(600, n)
    base = fold_data(frame, train, scored, ["a1"], 1 / 3)

    # scaling the scored targets changes their labels but never the threshold
    shocked = frame.copy()
    shocked.loc[600:, TARGET] *= 10
    after = fold_data(shocked, train, scored, ["a1"], 1 / 3)
    assert after.flat_threshold == base.flat_threshold
    assert (after.class_eval != 0).mean() > (base.class_eval != 0).mean()

    assert list(base.X_train.columns) == ["a1"]
    assert base.X_train["a1"].mean() == pytest.approx(0, abs=1e-12)
    assert base.X_train["a1"].std(ddof=0) == pytest.approx(1)
    assert len(base.X_train) == len(base.class_train) == 600
    assert len(base.X_eval) == len(base.y_eval) == 400
    assert (base.class_train == 0).mean() == pytest.approx(1 / 3, abs=0.01)


def test_fold_data_fits_the_scaler_on_training_rows_only():
    rng = np.random.default_rng(3)
    n = 1000
    frame = pd.DataFrame({"a1": rng.normal(5, 2, n), TARGET: rng.normal(size=n), VOL_SCALE: 1.0})
    train, scored = np.arange(600), np.arange(600, n)
    base = fold_data(frame, train, scored, ["a1"], 1 / 3)

    # shifting the scored features moves their scaled values but not the scaler or the training rows
    shocked = frame.copy()
    shocked.loc[600:, "a1"] += 100
    after = fold_data(shocked, train, scored, ["a1"], 1 / 3)
    assert after.scaler.mean_ == pytest.approx(base.scaler.mean_)
    pd.testing.assert_frame_equal(after.X_train, base.X_train)
    np.testing.assert_allclose(after.X_eval["a1"] - base.X_eval["a1"], 100 / base.scaler.scale_[0])

    raw = fold_data(frame, train, scored, ["a1"], 1 / 3, scale=False)
    assert raw.scaler is None
    pd.testing.assert_frame_equal(raw.X_eval, frame.iloc[scored][["a1"]])
