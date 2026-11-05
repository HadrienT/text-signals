"""Loughran-McDonald (LM) sentiment dictionary and the dictionary baseline score.

The master dictionary is downloaded from the Software Repository for Accounting and Finance
(University of Notre Dame), which hosts it on Google Drive. It is free for academic research;
it is fetched by code and never committed.

In the CSV, each sentiment column holds 0 when the word is not in the list, the year the word
was added (e.g. 2009) when it is, and minus a year (e.g. -2020) when it was removed that year.
A word belongs to a list when the value is positive.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from text_signals.fetch import Fetcher

LM_VERSION = "1993-2025"
# Google Drive id of Loughran-McDonald_MasterDictionary_1993-2025.csv, as linked from
# https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (updated March 2026).
LM_DRIVE_ID = "1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
LM_URL = f"https://drive.usercontent.google.com/download?id={LM_DRIVE_ID}&export=download"
# sha256 of that file when first fetched (2026-11-04). A different file means the authors
# published a new version: update LM_VERSION and this hash deliberately.
LM_SHA256 = "e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d"

CATEGORIES = (
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Strong_Modal",
    "Weak_Modal",
    "Constraining",
)

_TOKEN_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class LMDictionary:
    version: str
    sha256: str
    words: frozenset[str]  # every word of the master dictionary, upper case
    lists: dict[str, frozenset[str]]  # category -> words currently in that list


def parse_lm_csv(data: bytes, *, sha256: str, version: str = LM_VERSION) -> LMDictionary:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    missing = {"Word", *CATEGORIES} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"LM dictionary without columns {sorted(missing)}")
    words: set[str] = set()
    lists: dict[str, set[str]] = {c: set() for c in CATEGORIES}
    for row in reader:
        word = row["Word"].strip().upper()
        if not word:
            continue
        words.add(word)
        for category in CATEGORIES:
            if int(row[category]) > 0:
                lists[category].add(word)
    return LMDictionary(
        version=version,
        sha256=sha256,
        words=frozenset(words),
        lists={c: frozenset(ws) for c, ws in lists.items()},
    )


def fetch_lm_dictionary(fetcher: Fetcher, *, expected_sha256: str = LM_SHA256) -> LMDictionary:
    """Download (or read from cache) the pinned LM master dictionary.

    Google Drive answers 200 with an HTML page when its download quota is exceeded; the sha256
    check rejects that page, as it would a new version of the dictionary (in which case update
    ``LM_VERSION`` and ``LM_SHA256`` on purpose).
    """
    try:
        body, rec = fetcher.get(LM_URL, expected_sha256=expected_sha256)
    except ValueError as exc:
        raise ValueError(
            f"{exc}\nIf Google Drive refuses scripted downloads (quota exceeded), download the "
            f"CSV from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ in a browser and "
            f"run `just import-lm <file>`."
        ) from exc
    return parse_lm_csv(body, sha256=rec.sha256)


def tokenize(text: str) -> list[str]:
    """Upper-case alphabetic tokens; numbers and punctuation are dropped."""
    return [t.upper() for t in _TOKEN_RE.findall(text)]


class LMScore(BaseModel):
    n_tokens: int
    n_negative: int
    n_positive: int
    n_uncertainty: int
    net_negative: float  # (negative - positive) / n_tokens
    uncertainty: float  # uncertainty / n_tokens


def lm_score(text: str, lm: LMDictionary) -> LMScore:
    """Dictionary baseline of the protocol (§4.1): (negative − positive) / total tokens.

    The denominator counts every alphabetic token, in the dictionary or not.
    """
    tokens = tokenize(text)
    n = len(tokens)
    neg = sum(t in lm.lists["Negative"] for t in tokens)
    pos = sum(t in lm.lists["Positive"] for t in tokens)
    unc = sum(t in lm.lists["Uncertainty"] for t in tokens)
    return LMScore(
        n_tokens=n,
        n_negative=neg,
        n_positive=pos,
        n_uncertainty=unc,
        net_negative=(neg - pos) / n if n else float("nan"),
        uncertainty=unc / n if n else float("nan"),
    )


def main() -> None:
    """``python -m text_signals.lexicon import-lm FILE``: cache a manually downloaded LM CSV."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=["import-lm"])
    parser.add_argument("file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    with Fetcher(args.data_dir / "raw") as fetcher:
        rec = fetcher.import_file(LM_URL, args.file, expected_sha256=LM_SHA256)
    print(f"cached {rec.path} (sha256 {rec.sha256})")


if __name__ == "__main__":
    main()
