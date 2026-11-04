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

## Configuration

Every run parameter (symbol, horizon, sample window, lags, seed, paths) lives in `config.yaml`. Code reads it with `src.config.load_config()`, so nothing in `src/` hardcodes the coin, the horizon or the dates.

## Contributing

- One branch per task, merged into `main` through a pull request with one approving review. Direct pushes to `main` are blocked.
- Run `pytest` before opening a pull request.
- Strip notebook outputs before committing. Run this once per clone to do it automatically:

  ```
  nb-clean add-filter --remove-empty-cells
  ```