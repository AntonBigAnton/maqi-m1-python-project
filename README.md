# maqi-m1-python-project
Python For Data Science project

## Install

Requires Python 3.11 or newer. From the repo root:

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install .` is enough to run the pipeline; the `dev` extra adds `pytest` and `nb-clean`.

## Run

```
python -m src.pipeline
```

Add `--quick` to run on one month, or `--config path/to/config.yaml` to use another symbol, window or data folder.
All data comes from public Binance URLs (no API key) and is stored in the gitignored `data/` folder. Files already there are not downloaded again.

## Data collection size and runtime

Measured for BTCUSDT, 2026-01-01 to 2026-06-30, on one Windows laptop:

| | |
| --- | --- |
| Download size | 31 MB |
| Size on disk (zips and extracted CSVs) | 131 MB |
| First run | about 4.5 minutes |
| Re-run with data already present | about 3 seconds |

Every download is recorded in `data/raw/download_log.csv` (file, URL, size, SHA-256, checksum status).

## Configuration

Every run parameter (symbol, horizon, sample window, lags, seed, paths) lives in `config.yaml`. Code reads it with `src.config.load_config()`, so nothing in `src/` hardcodes the coin, the horizon or the dates.

## Contributing

- One branch per task, merged into `main` through a pull request with one approving review. Direct pushes to `main` are blocked.
- Run `pytest` before opening a pull request.
- Strip notebook outputs before committing. Run this once per clone to do it automatically:

  ```
  nb-clean add-filter --remove-empty-cells
  ```