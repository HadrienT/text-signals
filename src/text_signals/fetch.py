"""Polite, cached HTTP fetching with a provenance manifest.

Every response body is stored as fetched under ``<root>/<host>/<path>``, and every fetch appends
one line to ``<root>/manifest.jsonl`` with the URL, the fetch time and the sha256 of the body. A
URL that is already in the manifest is served from disk, so a rerun makes no network request.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

USER_AGENT = "text-signals-research/0.1 (+https://github.com/HadrienT/text-signals)"


class FetchRecord(BaseModel):
    """One manifest line: where a raw file came from and what it contained."""

    url: str
    final_url: str
    fetched_at: datetime
    status_code: int
    content_type: str | None
    sha256: str
    path: str  # relative to the cache root
    note: str | None = None  # e.g. how a file that could not be fetched by code was obtained


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_path_for(url: str) -> Path:
    """Relative on-disk path for a URL: ``<host>/<path>``, with ``index.html`` for directories."""
    parts = urlsplit(url)
    path = parts.path.lstrip("/") or "index.html"
    if path.endswith("/"):
        path += "index.html"
    if parts.query:
        path += "__" + hashlib.sha256(parts.query.encode()).hexdigest()[:12]
    return Path(parts.netloc.lower()) / path


class Fetcher:
    """HTTP GET with a minimum delay between network requests, a disk cache and a manifest."""

    def __init__(
        self,
        root: Path,
        *,
        delay_s: float = 2.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = root
        self.delay_s = delay_s
        self._client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=30.0,
        )
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.manifest_path = root / "manifest.jsonl"
        self._records: dict[str, FetchRecord] = self._load_manifest()

    def _load_manifest(self) -> dict[str, FetchRecord]:
        records: dict[str, FetchRecord] = {}
        if self.manifest_path.exists():
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = FetchRecord.model_validate_json(line)
                    records[rec.url] = rec
        return records

    def record(self, url: str) -> FetchRecord | None:
        return self._records.get(url)

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.delay_s - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def get(
        self, url: str, *, refresh: bool = False, expected_sha256: str | None = None
    ) -> tuple[bytes, FetchRecord]:
        """Return the body of ``url`` and its manifest record, from cache unless ``refresh``.

        Raises ``httpx.HTTPStatusError`` on a non-2xx response, and ``ValueError`` when
        ``expected_sha256`` is given and the body does not match it (some hosts answer 200 with
        an error page). Failed fetches are neither cached nor written to the manifest.
        """
        cached = self._records.get(url)
        if cached is not None and not refresh:
            body = (self.root / cached.path).read_bytes()
            if sha256_bytes(body) != cached.sha256:
                raise ValueError(f"cached file for {url} does not match its manifest sha256")
            return body, cached

        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        body = response.content
        if expected_sha256 is not None and sha256_bytes(body) != expected_sha256:
            raise ValueError(
                f"{url}: sha256 {sha256_bytes(body)} differs from the expected {expected_sha256} "
                f"(content-type {response.headers.get('content-type')})"
            )
        rel = cache_path_for(url)
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        rec = FetchRecord(
            url=url,
            final_url=str(response.url),
            fetched_at=datetime.now(UTC),
            status_code=response.status_code,
            content_type=response.headers.get("content-type"),
            sha256=sha256_bytes(body),
            path=rel.as_posix(),
        )
        self._append(rec)
        return body, rec

    def _append(self, rec: FetchRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.model_dump(mode="json")) + "\n")
        self._records[rec.url] = rec

    def import_file(self, url: str, source: Path, *, expected_sha256: str) -> FetchRecord:
        """Put a manually downloaded copy of ``url`` into the cache, if it is the expected file.

        For hosts that block scripted downloads at times (Google Drive quotas). The manifest
        line says so in ``note``; the sha256 check guarantees the bytes are the pinned ones.
        """
        body = source.read_bytes()
        if sha256_bytes(body) != expected_sha256:
            raise ValueError(f"{source}: sha256 {sha256_bytes(body)} != {expected_sha256}")
        rel = cache_path_for(url)
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        rec = FetchRecord(
            url=url,
            final_url=url,
            fetched_at=datetime.now(UTC),
            status_code=0,
            content_type=None,
            sha256=expected_sha256,
            path=rel.as_posix(),
            note=f"imported from a manual download ({source.name}); sha256 matches the pin",
        )
        self._append(rec)
        return rec

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
