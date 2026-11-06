"""Score every statement from 2000 with the ChronoGPT vintage of the year before (§6.7).

One vintage at a time: download its weights (pinned revision, sha256 checked), score the
statements it is assigned, append the results to the cache, delete the weights. Resumable: a
vintage already in the cache is skipped. CPU only.

Usage: ``python -m text_signals.sources.fomc.chrono_scores``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from text_signals.encoders import SPLITTER_VERSION, split_sentences
from text_signals.sources.fomc.extract import SampleDoc, all_docs

REPO = "manelalab/chrono-gpt-instruct-v1-{vintage}1231"
FIRST_VINTAGE, LAST_VINTAGE = 1999, 2024
WEIGHTS_DIR = Path("data/raw/chrono")
CACHE = Path("data/interim/fomc/chrono/scores.jsonl")
OUT = Path("data/processed/fomc/chrono_scores.csv")


def vintage_for(year: int) -> int | None:
    """The newest vintage strictly before the release year; None before 2000."""
    if year <= FIRST_VINTAGE:
        return None
    return min(year - 1, LAST_VINTAGE)


def assignments(docs: list[SampleDoc]) -> dict[int, list[SampleDoc]]:
    out: dict[int, list[SampleDoc]] = defaultdict(list)
    for d in docs:
        v = vintage_for(d.release_date.year)
        if v is not None:
            out[v].append(d)
    return dict(out)


def _api(repo: str) -> dict[str, object]:
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "text-signals research"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data: dict[str, object] = json.load(r)
    return data


def download(vintage: int) -> tuple[Path, str, str]:
    """The weights of one vintage at its current revision; returns (path, revision, sha256)."""
    repo = REPO.format(vintage=vintage)
    meta = _api(repo)
    revision = str(meta["sha"])
    files = {str(f["rfilename"]): f for f in meta["siblings"]}  # type: ignore[attr-defined]
    expected = str(files["pytorch_model.bin"]["lfs"]["sha256"])
    path = WEIGHTS_DIR / f"{vintage}.bin"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/pytorch_model.bin"
    req = urllib.request.Request(url, headers={"User-Agent": "text-signals research"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=600) as r, path.open("wb") as fh:
        while chunk := r.read(1 << 22):
            digest.update(chunk)
            fh.write(chunk)
    if digest.hexdigest() != expected:
        path.unlink()
        raise ValueError(f"{repo}: sha256 mismatch")
    return path, revision, expected


def done_vintages() -> set[int]:
    if not CACHE.exists():
        return set()
    return {int(json.loads(line)["vintage"]) for line in CACHE.read_text().splitlines() if line}


def score_vintage(vintage: int, docs: list[SampleDoc]) -> None:
    import tiktoken
    import torch

    from text_signals import chrono_gpt

    torch.set_num_threads(48)
    path, revision, sha = download(vintage)
    try:
        model = chrono_gpt.load(str(path), torch.float32)
        encode = tiktoken.get_encoding("gpt2").encode
        offset = chrono_gpt.anchor_offset(model, encode)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as fh:
            for d in docs:
                sentences = split_sentences(d.text)
                raw = [chrono_gpt.log_odds(model, encode, s) for s in sentences]
                record = {
                    "doc_id": d.doc_id,
                    "release_date": d.release_date.isoformat(),
                    "vintage": vintage,
                    "repo": REPO.format(vintage=vintage),
                    "revision": revision,
                    "weights_sha256": sha,
                    "prompt_version": chrono_gpt.PROMPT_VERSION,
                    "splitter_version": SPLITTER_VERSION,
                    "anchor_offset": offset,
                    "sentence_log_odds": raw,
                    "chrono_tone": sum(raw) / len(raw) - offset,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                fh.write(json.dumps(record) + "\n")
    finally:
        path.unlink(missing_ok=True)


def write_table() -> pd.DataFrame:
    rows = [json.loads(line) for line in CACHE.read_text().splitlines() if line]
    t = pd.DataFrame(rows)[["doc_id", "release_date", "vintage", "chrono_tone"]]
    t = t.drop_duplicates("doc_id", keep="last").sort_values("release_date")
    t.to_csv(OUT, index=False)
    return t


def main() -> None:
    plan = assignments(all_docs())
    done = done_vintages()
    for vintage in sorted(plan):
        if vintage in done:
            continue
        start = time.monotonic()
        score_vintage(vintage, plan[vintage])
        print(
            f"vintage {vintage}: {len(plan[vintage])} statements, {time.monotonic() - start:.0f}s",
            flush=True,
        )
    t = write_table()
    print(f"{len(t)} statements scored -> {OUT}")


if __name__ == "__main__":
    main()
