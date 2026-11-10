# entry point: fetch → clean → features → models → figures
import argparse
import logging

import pandas as pd

from src import clean, fetch
from src.config import DEFAULT_CONFIG, load_config


def sample_window(cfg, quick=False):
    """Return the configured (start, end) dates; quick mode keeps only the first month."""
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    if quick:
        end = min(end, start + pd.offsets.MonthEnd(0))
    return start.date(), end.date()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to the config file")
    parser.add_argument("--quick", action="store_true", help="run on one month for a fast check")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    start, end = sample_window(cfg, args.quick)
    fetch.fetch_all(cfg, cfg["symbol"], start, end)
    clean.clean_all(cfg, cfg["symbol"], start, end)


if __name__ == "__main__":
    main()
