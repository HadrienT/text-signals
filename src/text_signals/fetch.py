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

    def get(self, url: str, *, refresh: bool = False) -> tuple[bytes, FetchRecord]:
        """Return the body of ``url`` and its manifest record, from cache unless ``refresh``.

        Raises ``httpx.HTTPStatusError`` on a non-2xx response; failed fetches are not cached.
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
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.model_dump(mode="json")) + "\n")
        self._records[url] = rec
        return body, rec

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
