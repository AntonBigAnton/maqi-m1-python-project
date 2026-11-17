# column layouts of the Binance archive files and the clean and panel tables
# These are facts about the data, the same for every symbol; run settings live in config.yaml.

MS_PER_MINUTE = 60_000
# Epoch values above this are microseconds: 1e14 ms is in the year 5138, 1e14 µs is in 1973.
MICROSECOND_THRESHOLD = 10**14

# Column order of each archive CSV. Spot files have no header row, perp files do,
# so columns are always named by position and any header present is checked against this list.
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
METRICS_COLUMNS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]
# Perp files stop after is_buyer_maker; the missing column is read as empty and then dropped.
AGG_TRADE_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker", "is_best_match",
]

PRICE_COLUMNS = {"open": "float64", "high": "float64", "low": "float64", "close": "float64"}

# One clean schema per dataset: raw column order, output columns and dtypes,
# epoch columns to convert to UTC ms, the time column used to trim and sort, and the unique key.
SCHEMAS = {
    "klines": {
        "raw_columns": KLINE_COLUMNS,
        "columns": {
            "open_time": "int64", "close_time": "int64", **PRICE_COLUMNS,
            "volume": "float64", "quote_volume": "float64", "count": "int64",
            "taker_buy_volume": "float64", "taker_buy_quote_volume": "float64",
        },
        "epoch_columns": ["open_time", "close_time"],
        "time": "open_time",
        "key": "open_time",
    },
    # Same file layout as klines, but the volume and count columns are placeholders (zeros), so they are dropped.
    "premiumIndexKlines": {
        "raw_columns": KLINE_COLUMNS,
        "columns": {"open_time": "int64", "close_time": "int64", **PRICE_COLUMNS},
        "epoch_columns": ["open_time", "close_time"],
        "time": "open_time",
        "key": "open_time",
    },
    "metrics": {
        "raw_columns": METRICS_COLUMNS,
        "columns": {"create_time": "int64", **{name: "float64" for name in METRICS_COLUMNS[2:]}},
        "datetime_columns": ["create_time"],
        "time": "create_time",
        "key": "create_time",
    },
    "aggTrades": {
        "raw_columns": AGG_TRADE_COLUMNS,
        "columns": {
            "agg_trade_id": "int64", "price": "float64", "quantity": "float64",
            "first_trade_id": "int64", "last_trade_id": "int64",
            "transact_time": "int64", "is_buyer_maker": "bool",
        },
        "epoch_columns": ["transact_time"],
        "time": "transact_time",
        "key": "agg_trade_id",
    },
    "fundingRate": {
        "columns": {"funding_time": "int64", "funding_rate": "float64", "mark_price": "float64"},
        "epoch_columns": ["funding_time"],
        "time": "funding_time",
        "key": "funding_time",
    },
}

# Candle columns each market contributes to the panel (prefixed spot_ or perp_ there).
MARKET_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume",
]
