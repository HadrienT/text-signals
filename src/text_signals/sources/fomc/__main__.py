"""``python -m text_signals.sources.fomc``: fetch and parse the FOMC statements, then join
them with Treasury yields and score them with the LM dictionary."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from text_signals.sources.fomc import events
from text_signals.sources.fomc.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    parser.add_argument(
        "--refresh", action="store_true", help="refetch the calendar pages and FRED series"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    docs = run(args.data_dir, delay_s=args.delay, refresh=args.refresh)
    events.run(docs, args.data_dir, delay_s=args.delay, refresh=args.refresh)


if __name__ == "__main__":
    main()
