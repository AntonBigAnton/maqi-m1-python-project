# archive download, checksums, REST calls + cache
import csv
import hashlib
import json
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import truststore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1024 * 1024
FUNDING_PAGE_LIMIT = 1000
LOG_FIELDS = ["logged_at_utc", "file", "url", "size_bytes", "sha256", "status"]


class ChecksumError(Exception):
    """A downloaded archive file does not match its published .CHECKSUM."""


class CoverageError(Exception):
    """A market's candles do not cover the whole sample window."""


def make_session():
    """Return a session that retries transient errors and honours Retry-After."""
    # Use the OS trust store: antivirus TLS scanning breaks requests' bundled CA list.
    truststore.inject_into_ssl()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def raw_dir(data_dir):
    return Path(data_dir) / "raw"


def log_download(log_path, file, url, size_bytes, sha256, status):
    """Append one row to the provenance log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LOG_FIELDS)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        writer.writerow([now, Path(file).name, url, size_bytes, sha256, status])


def archive_periods(frequency, start, end):
    """List the archive periods covering a window, e.g. '2026-01' or '2026-01-31'."""
    if frequency == "monthly":
        return list(pd.period_range(start, end, freq="M").strftime("%Y-%m"))
    if frequency == "daily":
        return list(pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"))
    raise ValueError(f"Unknown archive frequency: {frequency}")


def archive_relative_path(cfg, spec, symbol, period):
    """Build a file's path inside the archive; klines files are named by interval."""
    market_path = cfg["markets"][spec["market"]]["archive_path"]
    parts = [market_path, spec["frequency"], spec["dataset"], symbol]
    if "interval" in spec:
        parts.append(spec["interval"])
    label = spec.get("interval", spec["dataset"])
    parts.append(f"{symbol}-{label}-{period}.zip")
    return "/".join(parts)


def download_archive_file(session, url, zip_path, log_path):
    """Download, verify and unzip one archive file; return its status.

    The zip is moved to its final name last, so if it exists, the checksum passed
    and the CSV was extracted. An interrupted run simply redoes that file.
    """
    if zip_path.exists():
        return "skipped"

    checksum = session.get(url + ".CHECKSUM", timeout=TIMEOUT_SECONDS)
    if checksum.status_code == 404:
        log_download(log_path, zip_path, url, 0, "", "missing")
        return "missing"
    checksum.raise_for_status()
    expected = checksum.text.split()[0]

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = zip_path.with_name(zip_path.name + ".part")
    sha256 = hashlib.sha256()
    size_bytes = 0
    with session.get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with part_path.open("wb") as f:
            for chunk in response.iter_content(CHUNK_BYTES):
                f.write(chunk)
                sha256.update(chunk)
                size_bytes += len(chunk)

    actual = sha256.hexdigest()
    if actual != expected:
        part_path.unlink()
        log_download(log_path, zip_path, url, size_bytes, actual, "checksum_mismatch")
        raise ChecksumError(f"{url}: expected {expected}, got {actual}")

    with zipfile.ZipFile(part_path) as archive:
        archive.extractall(zip_path.parent)
    zip_path.with_name(zip_path.name + ".CHECKSUM").write_text(checksum.text, encoding="utf-8")
    part_path.replace(zip_path)
    log_download(log_path, zip_path, url, size_bytes, actual, "verified")
    log.info("Downloaded %s (%.1f MB)", zip_path.name, size_bytes / 1e6)
    return "downloaded"


def fetch_archive_dataset(session, cfg, spec, symbol, start, end):
    """Download every file of one dataset; return a list of per-file statuses.

    Monthly files can extend past the window; the clean step trims them to the exact dates.
    """
    start = spec.get("start_date", start)
    end = spec.get("end_date", end)
    raw = raw_dir(cfg["data_dir"])
    results = []
    for period in archive_periods(spec["frequency"], start, end):
        relative_path = archive_relative_path(cfg, spec, symbol, period)
        url = f"{cfg['archive_url']}/{relative_path}"
        zip_path = raw / spec["market"] / spec["dataset"] / symbol / Path(relative_path).name
        status = download_archive_file(session, url, zip_path, raw / "download_log.csv")
        if status == "missing":
            if spec["required"]:
                raise FileNotFoundError(f"Required archive file not found: {url}")
            log.warning("Optional archive file not found: %s", url)
        results.append(
            {"market": spec["market"], "dataset": spec["dataset"], "period": period, "status": status}
        )
    return results


def get_json(session, url, params):
    response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def load_or_fetch_json(path, log_path, url, fetch):
    """Return the JSON cached at path, calling fetch() and saving the result on a miss."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(path.name + ".part")
    part_path.write_text(json.dumps(payload), encoding="utf-8")
    part_path.replace(path)
    log_download(log_path, path, url, path.stat().st_size, "", "rest_cached")
    return payload


def fetch_market_info(session, rest_url, symbol):
    """Return the raw first and latest 1-min candles and exchange info entry for a symbol."""
    klines_url = f"{rest_url}/klines"
    first_candle = get_json(
        session, klines_url, {"symbol": symbol, "interval": "1m", "startTime": 0, "limit": 1}
    )
    # Without startTime, the endpoint returns the most recent candles.
    last_candle = get_json(session, klines_url, {"symbol": symbol, "interval": "1m", "limit": 1})
    # The perp endpoint ignores the symbol filter and returns every listed contract.
    exchange_info = get_json(session, f"{rest_url}/exchangeInfo", {"symbol": symbol})
    entries = [entry for entry in exchange_info["symbols"] if entry["symbol"] == symbol]
    return {"first_candle": first_candle, "last_candle": last_candle, "symbol_info": entries}


def load_market_info(session, raw, market, rest_url, symbol, end_ms):
    """Return the cached market info, refetching it if it predates the window end.

    The latest candle changes over time, so a cache written before the window
    ended (or by an older version without last_candle) is refreshed once.
    """
    path = raw / market / "availability" / f"{symbol}.json"

    def load():
        return load_or_fetch_json(
            path, raw / "download_log.csv", rest_url, lambda: fetch_market_info(session, rest_url, symbol)
        )

    was_cached = path.exists()
    info = load()
    last_candle = info.get("last_candle")
    if was_cached and (not last_candle or last_candle[0][0] + 60_000 <= end_ms):
        path.unlink()
        info = load()
    return info


def check_availability(session, cfg, symbol, start, end):
    """Save and return a table of each market's coverage; fail if the window is not fully covered.

    A market covers the window if its first 1-min candle opens at or before the
    window start and its latest candle opens at or after the window's last minute.
    """
    raw = raw_dir(cfg["data_dir"])
    start_ms, end_ms = window_to_ms(start, end)
    last_minute_ms = end_ms + 1 - 60_000
    rows = []
    for market, market_cfg in cfg["markets"].items():
        info = load_market_info(session, raw, market, market_cfg["rest_url"], symbol, end_ms)
        if not info["first_candle"] or not info["symbol_info"]:
            raise CoverageError(f"{symbol} is not listed on the {market} market")
        first_ms = info["first_candle"][0][0]
        last_ms = info["last_candle"][0][0]
        rows.append(
            {
                "market": market,
                "symbol": symbol,
                "status": info["symbol_info"][0]["status"],
                "first_candle_utc": pd.to_datetime(first_ms, unit="ms", utc=True),
                "last_candle_utc": pd.to_datetime(last_ms, unit="ms", utc=True),
                "window_start": start,
                "window_end": end,
                "covers_window_start": first_ms <= start_ms,
                "covers_window_end": last_ms >= last_minute_ms,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(Path(cfg["data_dir"]) / f"availability_{symbol}.csv", index=False)

    gaps = table[~(table["covers_window_start"] & table["covers_window_end"])]
    if not gaps.empty:
        details = "; ".join(
            f"{row.market} has candles from {row.first_candle_utc} to {row.last_candle_utc}"
            for row in gaps.itertuples()
        )
        raise CoverageError(f"{symbol} does not cover {start} to {end}: {details}")
    return table


def window_to_ms(start, end):
    """Convert an inclusive date window to UTC millisecond bounds."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") + timedelta(days=1)).timestamp() * 1000) - 1
    return start_ms, end_ms


def fetch_funding_pages(session, url, symbol, start_ms, end_ms):
    """Collect the full funding history, following startTime across pages."""
    records = []
    cursor = start_ms
    while True:
        page = get_json(
            session,
            url,
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": FUNDING_PAGE_LIMIT},
        )
        records.extend(page)
        if len(page) < FUNDING_PAGE_LIMIT:
            return records
        cursor = page[-1]["fundingTime"] + 1


def fetch_funding(session, cfg, symbol, start, end):
    """Return the raw funding history for a window, cached as JSON."""
    raw = raw_dir(cfg["data_dir"])
    url = f"{cfg['markets']['perp']['rest_url']}/fundingRate"
    start_ms, end_ms = window_to_ms(start, end)
    records = load_or_fetch_json(
        raw / "perp" / "fundingRate" / symbol / f"{symbol}-fundingRate-{start}-{end}.json",
        raw / "download_log.csv",
        url,
        lambda: fetch_funding_pages(session, url, symbol, start_ms, end_ms),
    )
    if not records:
        raise ValueError(f"No funding history for {symbol} between {start} and {end}")
    return records


def fetch_all(cfg, symbol, start, end):
    """Run all collection steps for a symbol and window; return the per-file statuses."""
    session = make_session()
    Path(cfg["data_dir"]).mkdir(parents=True, exist_ok=True)

    availability = check_availability(session, cfg, symbol, start, end)
    log.info("Availability:\n%s", availability.to_string(index=False))

    results = []
    for spec in cfg["archive_datasets"]:
        results.extend(fetch_archive_dataset(session, cfg, spec, symbol, start, end))
    statuses = pd.DataFrame(results)

    funding = fetch_funding(session, cfg, symbol, start, end)
    log.info("Funding history: %d records", len(funding))

    summary = statuses.groupby(["market", "dataset", "status"]).size()
    log.info("Archive files:\n%s", summary.to_string())
    return statuses
