from __future__ import annotations

import argparse
from pathlib import Path

from aac.cli import debug, explain, record, suggest
from aac.cli.app import build_engine
from aac.presets import available_presets, describe_presets
from aac.storage.json_store import JsonHistoryStore

DEFAULT_HISTORY_PATH = Path(".aac_history.json")
DEFAULT_LIMIT = 10


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aac",
        description="Adaptive autocomplete engine with learning and explainability",
    )

    parser.add_argument(
        "--preset",
        default="production",
        choices=available_presets(),
        help="Autocomplete engine preset (controls predictors, ranking, learning). Default: production",
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Path to persisted autocomplete history",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "presets",
        help="List available presets and their behavior",
    )

    suggest_p = subparsers.add_parser("suggest", help="Get autocomplete suggestions")
    suggest_p.add_argument("text")
    suggest_p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)

    explain_p = subparsers.add_parser("explain", help="Explain why suggestions were ranked")
    explain_p.add_argument("text")
    explain_p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)

    record_p = subparsers.add_parser("record", help="Record a user selection")
    record_p.add_argument("text")
    record_p.add_argument("value")

    debug_p = subparsers.add_parser("debug", help="Run the debug pipeline")
    debug_p.add_argument("text")

    args = parser.parse_args()

    if args.command == "presets":
        print(describe_presets())
        return

    # Load persisted history
    store = JsonHistoryStore(args.history_path)
    persisted_history = store.load()

    # Build engine from preset and attach history
    engine = build_engine(
        preset=args.preset,
        history=persisted_history,
    )

    dispatch = {
        "suggest": lambda: suggest.run(
            engine=engine,
            text=args.text,
            limit=args.limit,
        ),
        "explain": lambda: explain.run(
            engine=engine,
            text=args.text,
            limit=args.limit,
        ),
        "record": lambda: record.run(
            engine=engine,
            store=store,
            text=args.text,
            value=args.value,
        ),
        "debug": lambda: debug.run(
            engine=engine,
            text=args.text,
        ),
    }

    dispatch[args.command]()
