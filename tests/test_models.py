# targets: forward return, past volatility (no look-ahead), flat threshold, feature blocks
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models import (
    DOWN, FLAT, TARGET, UP, VOL_SCALE,
    classify, feature_sets, fit_flat_threshold, forward_return, make_targets, model_frame, past_volatility,
)

JAN_1_MS = 1767225600000
MINUTE = 60_000
H = 15


def panel(log_prices, minutes=None):
    """A panel with spot VWAP = exp(log_prices), one row per listed minute (default 0, 1, 2, ...)."""
    minutes = range(len(log_prices)) if minutes is None else minutes
    return pd.DataFrame(
        {
            "open_time": [JAN_1_MS + m * MINUTE for m in minutes],
            "spot_vwap": np.exp(np.asarray(log_prices, dtype=float)),
        }
    )


def random_panel(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return panel(np.log(100) + np.cumsum(rng.normal(0, 1e-3, n)))


def test_forward_return_spans_vwap_of_minute_t_to_minute_t_plus_h():
    data = random_panel()
    y = forward_return(data, H)
    log_vwap = np.log(data["spot_vwap"])
    assert y[10] == pytest.approx(log_vwap[10 + H] - log_vwap[10])
    assert y.iloc[-H:].isna().all()  # no future price at the end of the window
    assert y.iloc[: -H].notna().all()


def test_forward_return_looks_up_the_future_by_time_not_position():
    # minute 20 is missing from the grid: row 5 (target minute 20) has no target,
    # and row 6 must use minute 21, not the row 15 positions later.
    minutes = [m for m in range(60) if m != 20]
    data = panel(np.linspace(0, 1, 60)[minutes], minutes=minutes)
    y = forward_return(data, H)
    assert np.isnan(y[5])
    assert y[6] == pytest.approx(np.log(data["spot_vwap"][20]) - np.log(data["spot_vwap"][6]))


def test_missing_vwap_gives_a_missing_target_not_a_filled_one():
    data = random_panel()
    data.loc[40, "spot_vwap"] = np.nan
    y = forward_return(data, H)
    assert np.isnan(y[40]) and np.isnan(y[40 - H])
    assert not np.isnan(y[41])


def test_past_volatility_matches_std_of_previous_returns():
    data = random_panel()
    vol = past_volatility(data, 60)
    returns = np.diff(np.log(data["spot_vwap"]))  # returns[s - 1] is the return of minute s
    # at t = 100: returns of minutes 40..99
    assert vol[100] == pytest.approx(np.std(returns[39:99], ddof=1))
    assert np.isnan(vol[20])  # fewer than half the window observed


def test_volatility_scale_does_not_look_ahead():
    """Changing prices from minute t on must not change the threshold scale at t."""
    data = random_panel()
    t = 300
    before = make_targets(data, H, 60)[VOL_SCALE]
    shocked = data.copy()
    shocked.loc[t:, "spot_vwap"] *= 1.5
    after = make_targets(shocked, H, 60)[VOL_SCALE]
    pd.testing.assert_series_equal(before[: t + 1], after[: t + 1])
    assert not np.isclose(before[t + 1], after[t + 1])  # minute t's return enters at t + 1


def test_volatility_scale_is_volatility_times_sqrt_horizon_and_missing_if_zero():
    data = random_panel()
    targets = make_targets(data, H, 60)
    assert targets[VOL_SCALE][200] == pytest.approx(past_volatility(data, 60)[200] * np.sqrt(H))
    frozen = panel(np.zeros(200))
    assert make_targets(frozen, H, 60)[VOL_SCALE].isna().all()


def test_flat_threshold_puts_the_requested_share_of_training_rows_in_flat():
    rng = np.random.default_rng(1)
    y, scale = rng.normal(0, 1, 30_000), rng.uniform(0.5, 2, 30_000)
    k = fit_flat_threshold(y, scale, 1 / 3)
    labels = classify(y, scale, k)
    shares = pd.Series(labels).value_counts(normalize=True)
    assert shares[FLAT] == pytest.approx(1 / 3, abs=0.01)
    assert shares[UP] == pytest.approx(shares[DOWN], abs=0.02)


def test_classify_labels_and_boundary():
    labels = classify([0.3, -0.3, 0.1, -0.2, 0.2], [1.0] * 5, 0.2)
    assert labels.tolist() == [UP, DOWN, FLAT, FLAT, FLAT]


def test_threshold_and_classify_reject_bad_input():
    with pytest.raises(ValueError, match="flat_share"):
        fit_flat_threshold([0.1], [1.0], 1.0)
    with pytest.raises(ValueError, match="No training row"):
        fit_flat_threshold([np.nan], [1.0], 0.3)
    with pytest.raises(ValueError, match="missing"):
        classify([np.nan], [1.0], 0.2)


def test_feature_sets_nest_a_in_a_plus_b():
    sets = feature_sets(["ret_1", "rv_15"], ["basis", "oi_change"])
    assert sets == {"A": ["ret_1", "rv_15"], "A+B": ["ret_1", "rv_15", "basis", "oi_change"]}


@pytest.mark.parametrize(
    "block_a, block_b, message",
    [
        ([], ["basis"], "at least one"),
        (["ret_1"], ["ret_1"], "both blocks"),
        (["ret_1", "ret_1"], [], "twice"),
        (["ret_1"], [TARGET], "Target"),
        (["open_time"], [], "Target or time"),
    ],
)
def test_feature_sets_reject_bad_blocks(block_a, block_b, message):
    with pytest.raises(ValueError, match=message):
        feature_sets(block_a, block_b)


def test_model_frame_keeps_the_same_rows_for_both_blocks():
    data = random_panel()
    data["a1"] = 1.0
    data["b1"] = 2.0
    data.loc[100, "b1"] = np.nan  # a perp gap: the minute is dropped for the baseline too
    frame = model_frame(data, ["a1"], ["b1"], H, 60)
    assert list(frame.columns) == ["open_time", "a1", "b1", TARGET, VOL_SCALE]
    assert JAN_1_MS + 100 * MINUTE not in frame["open_time"].values
    assert frame.notna().all().all()
    with pytest.raises(KeyError, match="b2"):
        model_frame(data, ["a1"], ["b2"], H, 60)


def test_config_has_target_and_walk_forward_settings():
    cfg = load_config()
    assert 0 < cfg["target"]["flat_share"] < 1
    assert cfg["target"]["vol_window_minutes"] > 0
    assert cfg["walk_forward"]["first_test_month"] >= 3
