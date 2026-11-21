# ablation fits (block A vs A + B): targets, class threshold, feature sets
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

from src.schemas import MS_PER_MINUTE

# Target columns added to the modelling table (spec §5). Feature columns must not reuse these names.
TARGET = "y"  # forward spot log return over the horizon
VOL_SCALE = "vol_scale"  # past volatility of that return, known when the forecast is made
TARGET_COLUMNS = (TARGET, VOL_SCALE)

# Classification labels
DOWN, FLAT, UP = -1, 0, 1


def forward_return(panel, horizon_minutes, price="spot_vwap"):
    """Forward log return y_t = log VWAP_{t+h} - log VWAP_t, one value per panel row.

    Row t holds the minute [t, t + 1 min), so the target spans the VWAP of minute t to the
    VWAP of minute t + h. The future minute is looked up by time, not by position, so a row
    missing from the grid can never shift a later price into the wrong row. A missing VWAP at
    either end (no trade, gap, end of the window) leaves the target missing.
    """
    log_price = pd.Series(np.log(panel[price].to_numpy()), index=panel["open_time"].to_numpy())
    if not log_price.index.is_unique:
        raise ValueError("open_time must be unique to compute forward returns")
    future = log_price.reindex(log_price.index + horizon_minutes * MS_PER_MINUTE).to_numpy()
    return pd.Series(future - log_price.to_numpy(), index=panel.index, name=TARGET)


def past_volatility(panel, window_minutes, price="spot_vwap"):
    """Standard deviation of 1-min log returns over the window_minutes before each row.

    The value at t only uses VWAPs of minutes before t (the last one being [t - 1 min, t)),
    so it is known when the forecast at t is made. Returns are put on a complete minute grid
    first, so the window always covers window_minutes of clock time, and missing returns are
    skipped rather than filled. Fewer than half the window observed gives a missing value.
    """
    times = panel["open_time"].to_numpy()
    grid = np.arange(times.min(), times.max() + MS_PER_MINUTE, MS_PER_MINUTE)
    log_price = pd.Series(np.log(panel[price].to_numpy()), index=times).reindex(grid)
    returns = log_price.diff()  # return of minute s: VWAP of s vs VWAP of s - 1 min
    vol = returns.rolling(window_minutes, min_periods=window_minutes // 2).std().shift(1)
    return pd.Series(vol.reindex(times).to_numpy(), index=panel.index, name="past_volatility")


def make_targets(panel, horizon_minutes, vol_window_minutes):
    """Regression target and the volatility scale of the class threshold, one row per minute.

    vol_scale is the past 1-min volatility times sqrt(h): the typical size of an h-minute
    return in the current regime. A zero volatility (prices frozen) gives no scale.
    """
    vol = past_volatility(panel, vol_window_minutes)
    return pd.DataFrame(
        {
            "open_time": panel["open_time"],
            TARGET: forward_return(panel, horizon_minutes),
            VOL_SCALE: (vol * np.sqrt(horizon_minutes)).where(vol > 0),
        }
    )


def fit_flat_threshold(y, vol_scale, flat_share):
    """Multiplier k such that a share flat_share of these (training) returns satisfy |y| <= k * vol_scale.

    Fit on the training rows of a fold only; the same k is then applied to its test rows.
    flat_share = 1/3 gives three classes of about equal size on the training data.
    """
    if not 0 < flat_share < 1:
        raise ValueError(f"flat_share must be between 0 and 1, got {flat_share}")
    ratio = np.abs(np.asarray(y, dtype=float) / np.asarray(vol_scale, dtype=float))
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        raise ValueError("No training row has both a target and a volatility scale")
    return float(np.quantile(ratio, flat_share))


def classify(y, vol_scale, threshold):
    """Up / flat / down labels: flat if |y| <= threshold * vol_scale, else the sign of y."""
    y = np.asarray(y, dtype=float)
    band = threshold * np.asarray(vol_scale, dtype=float)
    if np.isnan(y).any() or np.isnan(band).any():
        raise ValueError("Cannot classify rows with a missing target or volatility scale")
    return np.select([y > band, y < -band], [UP, DOWN], default=FLAT).astype("int64")


def feature_sets(block_a, block_b):
    """The two specifications of the ablation: spot-only baseline (A) and A plus perp features (B).

    Feature columns are passed in (block lists from features.py) so this works before the
    features are final. Blocks must be disjoint, A non-empty, and no feature may be a target.
    """
    block_a, block_b = list(block_a), list(block_b)
    if not block_a:
        raise ValueError("Block A (the baseline) needs at least one feature")
    for name, block in (("A", block_a), ("B", block_b)):
        if len(set(block)) != len(block):
            raise ValueError(f"Block {name} lists a feature twice")
    if overlap := set(block_a) & set(block_b):
        raise ValueError(f"Features in both blocks: {sorted(overlap)}")
    if leaked := (set(block_a) | set(block_b)) & {*TARGET_COLUMNS, "open_time"}:
        raise ValueError(f"Target or time columns used as features: {sorted(leaked)}")
    return {"A": block_a, "A+B": block_a + block_b}


def model_frame(data, block_a, block_b, horizon_minutes, vol_window_minutes):
    """The modelling table: open_time, every feature of A and B, the target and its volatility scale.

    data is the panel with the feature columns added. Rows missing the target, the scale or any
    feature of either block are dropped, so the baseline and the full model are fitted and
    scored on exactly the same minutes and their difference is due to block B only.
    """
    columns = feature_sets(block_a, block_b)["A+B"]
    if missing := [c for c in columns if c not in data.columns]:
        raise KeyError(f"Feature columns not in the data: {missing}")
    targets = make_targets(data, horizon_minutes, vol_window_minutes)
    frame = pd.concat([data[["open_time", *columns]], targets[list(TARGET_COLUMNS)]], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame.sort_values("open_time").reset_index(drop=True)


# Level 0-2 models (spec §8). Each is fitted on block A and on A + B with the same folds.
# Penalties are scale-free so one grid suits any symbol or horizon: features are standardised
# (fold_data), the ridge penalty is a multiple of the number of training rows, and the lasso
# penalty a multiple of the standard deviation of the training target.
# Boosting (level 2) tunes tree depth and learning rate. Its other settings are fixed for noisy
# minute data: large leaves (BOOSTING_MIN_LEAF rows) and a fixed number of trees, without early
# stopping, so a fit only depends on its training rows and the seed. The simplest grid point is
# close to a constant forecast: if tuning picks it, boosting found nothing to use.
REGRESSION_MODELS = ("zero", "ridge", "lasso", "boosting")
CLASSIFICATION_MODELS = ("majority", "logit", "boosting")
BOOSTING_TREES = 200
BOOSTING_MIN_LEAF = 500
DEFAULT_GRIDS = {
    "zero": [{}],
    "majority": [{}],
    "ridge": [{"penalty": p} for p in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)],
    "lasso": [{"penalty": p} for p in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)],
    "logit": [{"C": c} for c in (1e-3, 1e-2, 1e-1, 1.0)],
    # from almost constant (stumps, learning rate 0.003 over 200 trees) to moderately flexible
    "boosting": [{"max_depth": d, "learning_rate": r} for d in (1, 2, 3) for r in (0.003, 0.03)],
}
CLASSES = [DOWN, FLAT, UP]  # column order of class probabilities


def class_probabilities(estimator, X):
    """Down / flat / up probabilities in CLASSES order; a class absent from the training rows gets 0."""
    proba = np.zeros((len(X), len(CLASSES)))
    proba[:, [CLASSES.index(c) for c in estimator.classes_]] = estimator.predict_proba(X)
    return proba


def fit_predict(model, params, data, task, seed=0):
    """Fit one model on data.X_train and forecast data.X_eval.

    data is a FoldData (evaluate.fold_data). Regression models return the forecast returns;
    classification models return an (n, 3) array of down / flat / up probabilities.
    zero: forecasts no move (the random walk). majority: the training class shares, so its
    forecast class is the most common one. ridge / lasso: penalised linear regressions.
    logit: multinomial logistic regression. boosting: histogram gradient boosting, a regressor
    on returns or a classifier on the classes depending on task. Trees ignore the feature scaling.
    seed fixes boosting's only random step (the sample used to bin features on large data).
    """
    if model not in (REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS):
        raise ValueError(f"Unknown model {model!r} for task {task!r}")
    n_eval = len(data.X_eval)
    if model == "zero":
        return np.zeros(n_eval)
    if model == "majority":
        shares = pd.Series(data.class_train).value_counts(normalize=True).reindex(CLASSES, fill_value=0.0)
        return np.tile(shares.to_numpy(), (n_eval, 1))
    if model == "ridge":
        estimator = Ridge(alpha=params["penalty"] * len(data.X_train))
        return estimator.fit(data.X_train, data.y_train).predict(data.X_eval)
    if model == "lasso":
        estimator = Lasso(alpha=params["penalty"] * float(np.std(data.y_train)), max_iter=10_000)
        return estimator.fit(data.X_train, data.y_train).predict(data.X_eval)
    if model == "logit":
        estimator = LogisticRegression(C=params["C"], max_iter=2_000).fit(data.X_train, data.class_train)
        return class_probabilities(estimator, data.X_eval)
    if model == "boosting":
        settings = dict(
            max_iter=BOOSTING_TREES, min_samples_leaf=BOOSTING_MIN_LEAF, early_stopping=False,
            random_state=seed, **params,
        )
        if task == "regression":
            estimator = HistGradientBoostingRegressor(**settings)
            return estimator.fit(data.X_train, data.y_train).predict(data.X_eval)
        estimator = HistGradientBoostingClassifier(**settings).fit(data.X_train, data.class_train)
        return class_probabilities(estimator, data.X_eval)
    raise ValueError(f"No fitting code for model {model!r}")
