# raw → clean → 1-min panel
from pathlib import Path

import pandas as pd


def clean_path(data_dir, market, dataset, symbol, start, end):
    """Parquet path of one clean dataset; named by window so --quick never overwrites a full run."""
    return Path(data_dir) / "clean" / market / dataset / f"{symbol}_{start}_{end}.parquet"


def panel_path(data_dir, symbol, start, end):
    """Parquet path of a symbol's 1-min panel."""
    return Path(data_dir) / "panel" / f"{symbol}_{start}_{end}.parquet"


def load_or_build(path, build):
    """Return the parquet at path, calling build() and saving the result on a miss."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    table = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(path.name + ".part")
    table.to_parquet(part_path)
    part_path.replace(path)
    return table
