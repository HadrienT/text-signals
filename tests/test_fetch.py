from pathlib import Path

import httpx
import pytest

from text_signals.fetch import Fetcher, cache_path_for, sha256_bytes


def _fetcher(tmp_path: Path, calls: list[str], sleeps: list[float]) -> Fetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/missing":
            return httpx.Response(404)
        return httpx.Response(200, content=b"body of " + request.url.path.encode())

    now = iter(float(i) for i in range(100))
    return Fetcher(
        tmp_path,
        delay_s=2.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        clock=lambda: next(now),
    )


def test_cache_path_for() -> None:
    assert cache_path_for("https://WWW.x.gov/a/b.htm") == Path("www.x.gov/a/b.htm")
    assert cache_path_for("https://x.gov/a/") == Path("x.gov/a/index.html")
    assert cache_path_for("https://x.gov/a?q=1").name.startswith("a__")


def test_second_get_is_served_from_cache_and_manifest_persists(tmp_path: Path) -> None:
    calls: list[str] = []
    f = _fetcher(tmp_path, calls, [])
    body, rec = f.get("https://x.gov/a.htm")
    assert body == b"body of /a.htm"
    assert rec.sha256 == sha256_bytes(body)
    assert (tmp_path / rec.path).read_bytes() == body

    again, _ = f.get("https://x.gov/a.htm")
    assert again == body
    assert len(calls) == 1

    # A new fetcher on the same root reads the manifest and makes no request either.
    f2 = _fetcher(tmp_path, calls, [])
    assert f2.get("https://x.gov/a.htm")[0] == body
    assert len(calls) == 1


def test_refresh_refetches(tmp_path: Path) -> None:
    calls: list[str] = []
    f = _fetcher(tmp_path, calls, [])
    f.get("https://x.gov/a.htm")
    f.get("https://x.gov/a.htm", refresh=True)
    assert len(calls) == 2


def test_throttle_waits_between_network_requests(tmp_path: Path) -> None:
    sleeps: list[float] = []
    f = _fetcher(tmp_path, [], sleeps)
    f.get("https://x.gov/a.htm")
    f.get("https://x.gov/b.htm")
    assert sleeps and all(s > 0 for s in sleeps)


def test_http_error_is_raised_and_not_cached(tmp_path: Path) -> None:
    f = _fetcher(tmp_path, [], [])
    with pytest.raises(httpx.HTTPStatusError):
        f.get("https://x.gov/missing")
    assert f.record("https://x.gov/missing") is None


def test_tampered_cache_is_detected(tmp_path: Path) -> None:
    f = _fetcher(tmp_path, [], [])
    _, rec = f.get("https://x.gov/a.htm")
    (tmp_path / rec.path).write_bytes(b"edited")
    with pytest.raises(ValueError, match="sha256"):
        f.get("https://x.gov/a.htm")


def test_unexpected_body_is_rejected_before_caching(tmp_path: Path) -> None:
    f = _fetcher(tmp_path, [], [])
    with pytest.raises(ValueError, match="expected"):
        f.get("https://x.gov/a.htm", expected_sha256="0" * 64)
    assert f.record("https://x.gov/a.htm") is None
    assert not (tmp_path / "x.gov" / "a.htm").exists()
    good = sha256_bytes(b"body of /a.htm")
    assert f.get("https://x.gov/a.htm", expected_sha256=good)[1].sha256 == good


def test_import_file_checks_the_pin(tmp_path: Path) -> None:
    src = tmp_path / "manual.csv"
    src.write_bytes(b"word\n")
    f = _fetcher(tmp_path / "cache", [], [])
    with pytest.raises(ValueError):
        f.import_file("https://x.gov/d.csv", src, expected_sha256="0" * 64)
    rec = f.import_file("https://x.gov/d.csv", src, expected_sha256=sha256_bytes(b"word\n"))
    assert rec.note is not None and "manual" in rec.note
    # Served from cache afterwards, with no network request.
    calls: list[str] = []
    assert _fetcher(tmp_path / "cache", calls, []).get("https://x.gov/d.csv")[0] == b"word\n"
    assert calls == []
