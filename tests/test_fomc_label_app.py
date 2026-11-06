import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from text_signals.sources.fomc.label_app import App, make_handler
from text_signals.sources.fomc.labelling import load_label

PLAN = pd.DataFrame(
    {
        "code": ["S01", "S02", "P1"],
        "doc_id": ["a", "b", "c"],
        "role": ["label", "label", "practice"],
        "second_pass": [True, False, False],
    }
)

LABEL = {
    "policy_action": "raise",
    "change_bp": 25,
    "forward_guidance": 1,
    "inflation_assessment": 1,
    "labor_market_assessment": 0,
    "growth_assessment": None,
    "balance_sheet": None,
    "risk_balance": "inflation",
    "dissents_tighter": 0,
    "dissents_easier": 1,
    "dissents_other": 0,
    "unsure": ["forward_guidance"],
    "notes": "",
}


def _workspace(root: Path) -> Path:
    """A fake AudioBook workspace with chapter 1 = Statement S01 assembled."""
    (root / "out").mkdir(parents=True)
    sentences = [
        {"id": "01-0001", "chapter": 1, "text": "Statement S01"},
        {"id": "01-0002", "chapter": 1, "text": "Inflation remains elevated."},
    ]
    (root / "sentences.jsonl").write_text("\n".join(json.dumps(s) for s in sentences))
    timings = {
        "chapter": 1,
        "title": "Statement S01",
        "file": "01 - Statement S01.m4a",
        "duration": 4.0,
        "sentences": [
            {"id": "01-0001", "start": 0.5, "end": 1.5},
            {"id": "01-0002", "start": 1.9, "end": 3.6},
        ],
    }
    (root / "out" / "01.timings.json").write_text(json.dumps(timings))
    (root / "out" / "01 - Statement S01.m4a").write_bytes(bytes(range(100)))
    return root


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    labels = tmp_path / "labels"
    app = App(PLAN, _workspace(tmp_path / "work"), labels)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", labels
    httpd.shutdown()


def _get(url: str) -> object:
    with urlopen(url) as r:
        return json.loads(r.read())


def _put(url: str, body: object) -> int:
    req = Request(url, data=json.dumps(body).encode(), method="PUT")
    try:
        with urlopen(req) as r:
            return int(r.status)
    except HTTPError as exc:
        return exc.code


def test_status_lists_passes_and_audio(server: tuple[str, Path]) -> None:
    base, _ = server
    status = _get(f"{base}/api/status")
    assert isinstance(status, dict)
    first = status["passes"]["first"]
    assert [x["code"] for x in first] == ["S01", "S02"]
    assert [x["audio"] for x in first] == [True, False]
    assert [x["code"] for x in status["passes"]["second"]] == ["S01"]
    assert [x["code"] for x in status["passes"]["practice"]] == ["P1"]


def test_statement_has_sentence_timings(server: tuple[str, Path]) -> None:
    base, _ = server
    s = _get(f"{base}/api/statement/S01")
    assert isinstance(s, dict)
    assert s["sentences"][1] == {"text": "Inflation remains elevated.", "start": 1.9, "end": 3.6}


def test_label_round_trip(server: tuple[str, Path]) -> None:
    base, labels = server
    assert _put(f"{base}/api/labels/first/S01", LABEL) == 200
    saved = load_label(labels, "first", "S01")
    assert (
        saved is not None and saved.forward_guidance == 1 and saved.unsure == ["forward_guidance"]
    )
    got = _get(f"{base}/api/labels/first/S01")
    assert isinstance(got, dict) and got["pass"] == "first" and got["code"] == "S01"
    # The second pass is stored apart and starts empty.
    assert _get(f"{base}/api/labels/second/S01") is None


def test_invalid_labels_and_codes_are_refused(server: tuple[str, Path]) -> None:
    base, _ = server
    assert _put(f"{base}/api/labels/first/S01", {**LABEL, "forward_guidance": 3}) == 422
    assert _put(f"{base}/api/labels/first/S01", {**LABEL, "unsure": ["tone"]}) == 422
    assert _put(f"{base}/api/labels/second/S02", LABEL) == 404  # not in the second pass
    assert _put(f"{base}/api/labels/first/S99", LABEL) == 404


def test_audio_supports_byte_ranges(server: tuple[str, Path]) -> None:
    base, _ = server
    with urlopen(Request(f"{base}/audio/S01", headers={"Range": "bytes=10-19"})) as r:
        assert r.status == 206
        assert r.headers["Content-Range"] == "bytes 10-19/100"
        assert r.read() == bytes(range(10, 20))
