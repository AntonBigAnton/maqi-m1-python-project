# entry point: fetch → clean → features → models → figures
import argparse
import logging
import time

import matplotlib
import pandas as pd

from src import clean, descriptive, fetch, panel, plots
from src.config import DEFAULT_CONFIG, load_config
from src.features import block_a_columns, block_b_columns, build_features, feature_lags, sensitivity_table

log = logging.getLogger(__name__)

APPENDIX_OI_QUANTILES = ((0.005, "_q0.5"), (0.02, "_q2"))  # event study robustness (appendix figures 9)


def sample_window(cfg, quick=False):
    """Return the configured (start, end) dates; quick mode keeps only the first month."""
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    if quick:
        end = min(end, start + pd.offsets.MonthEnd(0))
    return start.date(), end.date()


def quick_outputs(cfg):
    """Send quick-run figures and tables to data/quick/, so they never overwrite the report's."""
    quick_dir = cfg["data_dir"] / "quick"
    return {**cfg, "figures_dir": quick_dir / "figures", "tables_dir": quick_dir / "tables"}


def months_in_window(start, end):
    """Number of calendar months the window touches."""
    return len(pd.period_range(start, end, freq="M"))


def feature_blocks(cfg):
    """Block A and block B column names for the config's horizon and lags."""
    lags = feature_lags(cfg["horizon_minutes"], cfg)
    return block_a_columns(lags), block_b_columns(lags)


def run_descriptive(cfg, data, features, block_a, block_b):
    """Every descriptive table and figure of the report (spec §7, figures 1 to 10), as in the notebook.

    data is the 1-min panel and features the same panel with the feature columns added.
    """
    symbol = cfg["symbol"]
    save = lambda table, name: descriptive.save_table(table, f"{name}_{symbol}", cfg)

    # 1. Data quality and market structure
    plots.figure_1(descriptive.missing_minutes(data), cfg)
    save(descriptive.tracking_stats(data), "tracking")
    plots.figure_2(descriptive.volume_share(data), cfg)
    plots.figure_3(descriptive.intraday_profile(data), cfg)

    # 2. Returns and volatility
    samples = descriptive.return_samples(data, cfg["horizon_minutes"])
    save(descriptive.return_moments(samples), "return_moments")
    plots.figure_4(samples, cfg)
    plots.figure_5(descriptive.acf_table(data), cfg)
    save(descriptive.stationarity_tests(descriptive.five_minute_series(data)), "stationarity")

    # 3. Perp-specific signals
    save(sensitivity_table(data), "deleveraging_sensitivity")
    half_life = descriptive.basis_half_life(data)
    save(half_life, "basis_half_life")
    clusters = descriptive.deleveraging_clusters(data)
    plots.figure_6(descriptive.basis_over_time(data), clusters, half_life, cfg)
    plots.figure_7(descriptive.leverage_series(data), cfg)
    save(descriptive.flow_correlation(data, cfg["lags"]), "flow_correlation")
    correlation = features[block_a + block_b].corr()
    plots.figure_10(correlation, cfg)
    save(descriptive.redundant_pairs(correlation), "redundant_features")

    # 4. Lead-lag
    plots.figure_8(descriptive.lead_lag_table(data), cfg)
    save(descriptive.granger(data), "granger")
    study = descriptive.event_study(data, clusters)
    save(study, "event_study")
    plots.figure_9(study, descriptive.event_study(data, descriptive.placebo_events(data, clusters)), cfg)
    for oi_q, suffix in APPENDIX_OI_QUANTILES:
        appendix_clusters = descriptive.deleveraging_clusters(data, oi_q)
        appendix_placebo = descriptive.placebo_events(data, appendix_clusters)
        plots.figure_9(descriptive.event_study(data, appendix_clusters),
                       descriptive.event_study(data, appendix_placebo), cfg, suffix)


def run_models(cfg, features, block_a, block_b, start, end):
    """The ablation (block A vs A + B, spec §8-9) and its figures 11 and 12."""
    from src.evaluate import run_ablation  # here, so a run that skips the models never loads them

    results = run_ablation(cfg, features, block_a, block_b, cfg["symbol"], start, end)
    plots.figure_11(results["clark_west"], cfg, folds=results["folds"])
    plots.figure_12(results["coefficients"], results["permutation_importance"], cfg)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to the config file")
    parser.add_argument("--quick", action="store_true", help="run on one month for a fast check")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matplotlib.use("Agg")  # save figures without opening windows
    began = time.perf_counter()
    cfg = load_config(args.config)
    if args.quick:
        cfg = quick_outputs(cfg)
    symbol = cfg["symbol"]
    start, end = sample_window(cfg, args.quick)

    log.info("Step 1/5: fetch")
    fetch.fetch_all(cfg, symbol, start, end)
    log.info("Step 2/5: clean")
    clean.clean_all(cfg, symbol, start, end)
    data = panel.build_panel(cfg, symbol, start, end)

    log.info("Step 3/5: features")
    block_a, block_b = feature_blocks(cfg)
    features = build_features(data, cfg)

    log.info("Step 4/5: descriptive tables and figures")
    run_descriptive(cfg, data, features, block_a, block_b)

    log.info("Step 5/5: models and figures 11-12")
    needed = cfg["walk_forward"]["first_test_month"]
    if months_in_window(start, end) < needed:
        log.warning("Skipping the models: walk-forward needs at least %d months of data, the window has %d",
                    needed, months_in_window(start, end))
    else:
        run_models(cfg, features, block_a, block_b, start, end)

    log.info("Done in %.1f min. Figures in %s, tables in %s",
             (time.perf_counter() - began) / 60, cfg["figures_dir"], cfg["tables_dir"])


if __name__ == "__main__":
    main()
