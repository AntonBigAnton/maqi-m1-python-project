# small checks: archive URLs, checksums, skip-if-present, funding pagination
import hashlib
import io
import zipfile

import pytest

from src import fetch

CFG = {
    "archive_url": "https://archive.test/data",
    "markets": {
        "spot": {"archive_path": "spot", "rest_url": "https://spot.test"},
        "perp": {"archive_path": "futures/um", "rest_url": "https://perp.test"},
    },
}


class FakeResponse:
    def __init__(self, status_code=200, content=b"", payload=None):
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="ignore")
        self.payload = payload

    def raise_for_status(self):
        assert self.status_code < 400

    def iter_content(self, _chunk_size):
        yield self.content

    def json(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeSession:
    """Serves queued responses by URL and records every request made."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, params=None, **_kwargs):
        self.calls.append((url, params))
        response = self.responses[url]
        return response.pop(0) if isinstance(response, list) else response


def make_zip(name="data.csv", body=b"a,b\n1,2\n"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def archive_session(url, payload, published_sha=None):
    sha = published_sha or hashlib.sha256(payload).hexdigest()
    checksum = f"{sha}  file.zip".encode()
    return FakeSession(
        {url: FakeResponse(content=payload), url + ".CHECKSUM": FakeResponse(content=checksum)}
    )


def test_archive_periods_cover_the_window():
    assert fetch.archive_periods("monthly", "2026-01-15", "2026-03-01") == ["2026-01", "2026-02", "2026-03"]
    assert fetch.archive_periods("daily", "2026-01-31", "2026-02-01") == ["2026-01-31", "2026-02-01"]


def test_archive_paths_match_the_binance_layout():
    klines = {"dataset": "klines", "market": "spot", "frequency": "monthly", "interval": "1m"}
    assert (
        fetch.archive_relative_path(CFG, klines, "BTCUSDT", "2026-01")
        == "spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-01.zip"
    )
    metrics = {"dataset": "metrics", "market": "perp", "frequency": "daily"}
    assert (
        fetch.archive_relative_path(CFG, metrics, "ETHUSDT", "2026-01-31")
        == "futures/um/daily/metrics/ETHUSDT/ETHUSDT-metrics-2026-01-31.zip"
    )


def test_download_verifies_extracts_then_skips(tmp_path):
    url = "https://archive.test/file.zip"
    session = archive_session(url, make_zip())
    zip_path = tmp_path / "file.zip"
    log_path = tmp_path / "log.csv"

    assert fetch.download_archive_file(session, url, zip_path, log_path) == "downloaded"
    assert (tmp_path / "data.csv").read_bytes() == b"a,b\n1,2\n"
    assert "verified" in log_path.read_text()

    calls_after_download = len(session.calls)
    assert fetch.download_archive_file(session, url, zip_path, log_path) == "skipped"
    assert len(session.calls) == calls_after_download


def test_checksum_mismatch_fails_loudly_and_leaves_nothing(tmp_path):
    url = "https://archive.test/file.zip"
    session = archive_session(url, make_zip(), published_sha="0" * 64)
    log_path = tmp_path / "log.csv"

    with pytest.raises(fetch.ChecksumError):
        fetch.download_archive_file(session, url, tmp_path / "file.zip", log_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["log.csv"]
    assert "checksum_mismatch" in log_path.read_text()


@pytest.mark.parametrize("required", [True, False])
def test_missing_file_stops_only_required_datasets(tmp_path, required):
    url = "https://archive.test/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip"
    session = FakeSession({url + ".CHECKSUM": FakeResponse(status_code=404)})
    cfg = {**CFG, "data_dir": tmp_path}
    spec = {"dataset": "metrics", "market": "perp", "frequency": "daily", "required": required}

    if required:
        with pytest.raises(FileNotFoundError):
            fetch.fetch_archive_dataset(session, cfg, spec, "BTCUSDT", "2026-01-01", "2026-01-01")
    else:
        results = fetch.fetch_archive_dataset(session, cfg, spec, "BTCUSDT", "2026-01-01", "2026-01-01")
        assert results[0]["status"] == "missing"


def test_funding_follows_pages_then_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "FUNDING_PAGE_LIMIT", 2)
    url = "https://perp.test/fundingRate"
    pages = [
        FakeResponse(payload=[{"fundingTime": 100}, {"fundingTime": 200}]),
        FakeResponse(payload=[{"fundingTime": 300}]),
    ]
    session = FakeSession({url: pages})
    cfg = {**CFG, "data_dir": tmp_path}

    records = fetch.fetch_funding(session, cfg, "BTCUSDT", "2026-01-01", "2026-01-31")
    assert [r["fundingTime"] for r in records] == [100, 200, 300]
    assert session.calls[1][1]["startTime"] == 201

    assert fetch.fetch_funding(session, cfg, "BTCUSDT", "2026-01-01", "2026-01-31") == records
    assert len(session.calls) == 2


def test_window_to_ms_includes_the_whole_last_day():
    start_ms, end_ms = fetch.window_to_ms("2026-01-01", "2026-01-01")
    assert start_ms == 1767225600000
    assert end_ms == start_ms + 86_400_000 - 1
