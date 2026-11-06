"""Look-ahead tests of the LLM score (protocol §6.7, H3).

A model trained on text written after a statement may know what followed it. Three checks:

- **cutoff split**: does the score's effect differ after the model's training cutoff?
- **anonymisation**: remove dates and names, read again, and compare the scores;
- **recognition probe**: can the model date an anonymised statement?

``anonymise`` rewrites a statement: dates with a month name become ``[date]``, a month alone
``[month]``, four-digit years ``[year]``, and the names of FOMC members (read from the
statements' own vote paragraphs and "Chairman …" phrases) ``[member]``. Policy wording, numbers
and rates stay as they are.

Usage: ``python -m text_signals.sources.fomc.lookahead {anonymised-run,probe-run,report}``.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from text_signals.llm import Decoding, ExtractionRecord, LLMClient, Prompt, extract_all
from text_signals.sources.fomc.extract import SampleDoc, all_docs, run_local

MONTHS = "January|February|March|April|June|July|August|September|October|November|December"
_NAME = r"[A-Z][a-z]+(?: [A-Z]\.)?(?: [A-Z][a-z]*[A-Z]?[a-z]+(?:-[A-Z][a-z]+)?)+"
_NOT_NAMES = {"Vice Chairman", "Vice Chair", "Federal Reserve", "Open Market"}
# a vote sentence ends at a period not preceded by a lone capital (an initial: "William J.")
_VOTE = re.compile(r"Voting (?:for|against).*?(?:(?<!\b[A-Z])\.(?=\s|$)|$)")
_CHAIR = re.compile(r"Chairman (" + _NAME + ")")


def member_names(texts: Iterable[str]) -> set[str]:
    """Full names of FOMC members as the statements write them in their vote paragraphs."""
    names: set[str] = set()
    for text in texts:
        flat = re.sub(r"\s+", " ", text).replace(", Jr.", "")
        for sentence in _VOTE.findall(flat):
            names |= {n for n in re.findall(_NAME, sentence) if n not in _NOT_NAMES}
        names |= set(_CHAIR.findall(flat))
    return names


def _surname(name: str) -> str:
    return name.split()[-1]


def anonymise(text: str, names: set[str]) -> str:
    out = text
    # names: full names first, then surnames alone; "Jr." goes with its name
    for name in sorted(names, key=len, reverse=True):
        out = re.sub(re.escape(name) + r"(?:,? Jr\.)?", "[member]", out)
    for surname in sorted({_surname(n) for n in names}, key=len, reverse=True):
        out = re.sub(r"\b" + re.escape(surname) + r"\b", "[member]", out)
    # dates with a month: "January 28, 2015", "January 28", "January 2015"
    out = re.sub(
        r"\b(?:" + MONTHS + r"|May)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?", "[date]", out
    )
    out = re.sub(r"\b(?:" + MONTHS + r"|May),?\s+\d{4}\b", "[date]", out)
    out = re.sub(r"\b(?:" + MONTHS + r")\b", "[month]", out)
    out = re.sub(r"\b(in|since|of|late|early|mid-) May\b", r"\1 [month]", out, flags=re.I)
    out = re.sub(r"\b(?:19|20)\d{2}\b", "[year]", out)
    return out


PROBE_VERSION = "fomc-date-probe-v1"


class DateGuess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    year: int = Field(
        ge=1990, le=2030, description="The year the statement was most likely released."
    )


DATE_PROBE_V1 = Prompt(
    version=PROBE_VERSION,
    system=(
        "You are shown a statement of the Federal Open Market Committee (FOMC) from which dates "
        "and names have been removed. Give the year in which it was most likely released, as "
        "JSON with one field, 'year'. Answer with your best single guess."
    ),
    user_template="Statement:\n\n{text}",
)

OUT = Path("data/interim/fomc/lookahead")


def anonymised_docs() -> list[SampleDoc]:
    docs = all_docs()
    names = member_names(d.text for d in docs)
    return [SampleDoc(d.code, d.doc_id, d.release_date, anonymise(d.text, names)) for d in docs]


def run_anonymised(base_url: str) -> list[ExtractionRecord]:
    """The chosen procedure's values (a single pass equals the first answer) on the anonymised
    texts, cached apart from the original reading."""
    return run_local(base_url, anonymised_docs(), OUT / "anonymised")


def run_probe(base_url: str) -> list[ExtractionRecord]:
    client = LLMClient(base_url)
    model, model_path = client.model_info()
    docs = anonymised_docs()

    def run(doc_id: str, text: str) -> ExtractionRecord:
        return client.extract(
            doc_id=doc_id,
            text=text,
            prompt=DATE_PROBE_V1,
            schema=DateGuess,
            schema_version=PROBE_VERSION,
            decoding=Decoding(max_tokens=32),
            model=model,
            model_path=model_path,
        )

    return extract_all(
        ((d.doc_id, d.text) for d in docs),
        run,
        model=model,
        prompt_version=DATE_PROBE_V1.version,
        schema_version=PROBE_VERSION,
        cache_path=OUT / "date_probe" / f"{model}.jsonl",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["anonymised-run", "probe-run", "show"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.cmd == "anonymised-run":
        records = run_anonymised(args.base_url)
        print(f"{len(records)} records, {sum(r.error is not None for r in records)} invalid")
    elif args.cmd == "probe-run":
        records = run_probe(args.base_url)
        print(f"{len(records)} records, {sum(r.error is not None for r in records)} invalid")
    else:
        docs = anonymised_docs()
        print(len(member_names(d.text for d in all_docs())), "member names")
        for d in (docs[0], docs[len(docs) // 2], docs[-1]):
            print(f"--- {d.doc_id}\n{d.text}\n")


if __name__ == "__main__":
    main()
