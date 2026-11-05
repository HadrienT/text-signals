"""Local labelling page: listen to a statement with the sentence being read highlighted, and
fill the label form next to it (protocol §6.6).

The audio and its sentence timings come from the AudioBook workspace that read the listening PDF
(``write_listening_pdf``) aloud: ``out/NN - <title>.m4a``, ``out/NN.timings.json`` and
``sentences.jsonl``. The page reads them as files; AudioBook itself is not imported.

Labels are written to ``data/labels/fomc_statements_v1/<pass>/<code>.json``. Standard library
only: ``python -m text_signals.sources.fomc.label_app`` (see ``just label``).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from text_signals.sources.fomc.extraction import StatementExtraction
from text_signals.sources.fomc.labelling import (
    GUIDE_VERSION,
    HELP_FR,
    HumanLabel,
    chapter_title,
    load_label,
    save_label,
)

_SAFE = re.compile(r"^[A-Z0-9]{2,4}$")
PASSES = ("practice", "first", "second")


@dataclass(frozen=True)
class Chapter:
    title: str
    audio: Path
    sentences: list[dict[str, Any]]  # text, start, end (seconds)


def load_chapters(work: Path) -> dict[str, Chapter]:
    """AudioBook output, by chapter title. Chapters not assembled yet are absent."""
    by_chapter: dict[int, list[dict[str, Any]]] = {}
    for line in (work / "sentences.jsonl").read_text("utf-8").splitlines():
        if line.strip():
            s = json.loads(line)
            by_chapter.setdefault(int(s["chapter"]), []).append(s)
    out: dict[str, Chapter] = {}
    for path in sorted((work / "out").glob("*.timings.json")):
        t = json.loads(path.read_text("utf-8"))
        at = {x["id"]: x for x in t["sentences"]}
        sentences = [
            {"text": s["text"], "start": at[s["id"]]["start"], "end": at[s["id"]]["end"]}
            for s in by_chapter.get(int(t["chapter"]), [])
            if s["id"] in at
        ]
        out[str(t["title"])] = Chapter(str(t["title"]), work / "out" / t["file"], sentences)
    return out


def field_definitions() -> list[dict[str, str]]:
    """Each field with its binding English definition (the schema) and the French help."""
    schema = StatementExtraction.model_json_schema()
    defs = schema["$defs"]
    names = {
        "policy_action": "PolicyAction",
        "forward_guidance": "ForwardGuidance",
        "inflation_assessment": "InflationAssessment",
        "labor_market_assessment": "LaborMarketAssessment",
        "growth_assessment": "GrowthAssessment",
        "balance_sheet": "BalanceSheet",
        "risk_balance": "RiskBalance",
    }
    out = [
        {"field": f, "en": defs[d]["properties"]["value"]["description"], "fr": HELP_FR[f]}
        for f, d in names.items()
    ]
    out.append(
        {
            "field": "dissents",
            "en": defs["Dissent"]["properties"]["preferred"]["description"],
            "fr": HELP_FR["dissents"],
        }
    )
    return out


class App:
    def __init__(self, plan: pd.DataFrame, work: Path, labels: Path) -> None:
        self.plan = plan
        self.work = work
        self.labels = labels

    def codes(self, pass_: str) -> list[str]:
        p = self.plan
        if pass_ == "practice":
            rows = p[p["role"] == "practice"]
        elif pass_ == "second":
            rows = p[(p["role"] == "label") & p["second_pass"]]
        else:
            rows = p[p["role"] == "label"]
        return list(rows["code"])

    def status(self) -> dict[str, Any]:
        chapters = load_chapters(self.work)
        return {
            "guide_version": GUIDE_VERSION,
            "passes": {
                pass_: [
                    {
                        "code": c,
                        "audio": chapter_title(c) in chapters,
                        "done": load_label(self.labels, pass_, c) is not None,
                    }
                    for c in self.codes(pass_)
                ]
                for pass_ in PASSES
            },
        }

    def statement(self, code: str) -> dict[str, Any] | None:
        ch = load_chapters(self.work).get(chapter_title(code))
        if ch is None:
            return None
        return {"code": code, "title": ch.title, "sentences": ch.sentences}

    def audio(self, code: str) -> Path | None:
        ch = load_chapters(self.work).get(chapter_title(code))
        return None if ch is None else ch.audio


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    page = (Path(__file__).parent / "label_page.html").read_text("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _parts(self) -> list[str]:
            return [p for p in self.path.split("?")[0].split("/") if p]

        def do_GET(self) -> None:  # noqa: N802
            parts = self._parts()
            if not parts:
                body = page.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parts == ["api", "status"]:
                self._json(app.status())
            elif parts == ["api", "fields"]:
                self._json(field_definitions())
            elif len(parts) == 3 and parts[:2] == ["api", "statement"] and _SAFE.match(parts[2]):
                found = app.statement(parts[2])
                self._json(found) if found else self._json({}, HTTPStatus.NOT_FOUND)
            elif (
                len(parts) == 4
                and parts[:2] == ["api", "labels"]
                and parts[2] in PASSES
                and _SAFE.match(parts[3])
            ):
                label = load_label(app.labels, parts[2], parts[3])
                self._json(label.model_dump(mode="json", by_alias=True) if label else None)
            elif len(parts) == 2 and parts[0] == "audio" and _SAFE.match(parts[1]):
                path = app.audio(parts[1])
                if path is None or not path.is_file():
                    self._json({}, HTTPStatus.NOT_FOUND)
                else:
                    self._file(path)
            else:
                self._json({}, HTTPStatus.NOT_FOUND)

        def _file(self, path: Path) -> None:
            """Serve the audio with byte ranges, which browsers need to seek."""
            size = path.stat().st_size
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else end
                else:
                    start = size - int(m.group(2))
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                self.wfile.write(fh.read(end - start + 1))

        def do_PUT(self) -> None:  # noqa: N802
            parts = self._parts()
            ok = (
                len(parts) == 4
                and parts[:2] == ["api", "labels"]
                and parts[2] in PASSES
                and parts[3] in app.codes(parts[2])
            )
            if not ok:
                self._json({"error": "unknown pass or code"}, HTTPStatus.NOT_FOUND)
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                data = json.loads(raw)
                data.update(code=parts[3], labelled_at=datetime.now(UTC).isoformat())
                data["pass"] = parts[2]
                label = HumanLabel.model_validate(data)
            except (ValueError, ValidationError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
                return
            save_label(app.labels, label)
            self._json({"saved": True})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("data/labels/fomc_statements_sample_v1.csv")
    )
    parser.add_argument(
        "--work", type=Path, default=Path.home() / "AudioBook" / "work" / "fomc-labelling-v1"
    )
    parser.add_argument("--labels", type=Path, default=Path("data/labels/fomc_statements_v1"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    app = App(pd.read_csv(args.plan), args.work, args.labels)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"labelling page on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
