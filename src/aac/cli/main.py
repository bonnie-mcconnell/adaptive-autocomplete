from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aac.cli import debug, explain, record, suggest
from aac.cli.app import build_engine
from aac.presets import PRESETS, available_presets, describe_presets
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

    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a custom vocabulary file. "
            "Replaces the bundled 48k English vocabulary. "
            "Format: one word per line (wordlist) or free-form text (see --vocab-format). "
            "Example: aac --vocab-path commands.txt suggest git"
        ),
    )

    parser.add_argument(
        "--vocab-format",
        default="wordlist",
        choices=["wordlist", "text"],
        help=(
            "Format of --vocab-path file. "
            "'wordlist': one word or phrase per line (default). "
            "'text': free-form text, words weighted by frequency."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "presets",
        help="List available presets and their behavior",
    ).add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output preset descriptions as JSON",
    )

    suggest_p = subparsers.add_parser("suggest", help="Get autocomplete suggestions")
    suggest_p.add_argument(
        "text",
        help="Input prefix to complete (e.g. 'prog' -> suggestions starting with 'prog')",
    )
    suggest_p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help=f"Maximum number of suggestions to return (default: {DEFAULT_LIMIT})",
    )

    explain_p = subparsers.add_parser("explain", help="Explain why suggestions were ranked")
    explain_p.add_argument(
        "text",
        help="Input prefix to explain (e.g. 'prog')",
    )
    explain_p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help=f"Maximum number of suggestions to explain (default: {DEFAULT_LIMIT})",
    )

    record_p = subparsers.add_parser("record", help="Record a user selection")
    record_p.add_argument(
        "text",
        help="Input prefix the user was completing (e.g. 'prog')",
    )
    record_p.add_argument(
        "value",
        help="Completion the user selected (e.g. 'programming')",
    )

    debug_p = subparsers.add_parser(
        "debug",
        help="Run the debug pipeline (verbose internal output, not for end users)",
    )
    debug_p.add_argument(
        "text",
        help="Input prefix to debug",
    )

    args = parser.parse_args()

    try:
        _run(args)
    except BrokenPipeError:
        # Consumer closed the pipe (e.g. `aac suggest he | head -5`).
        # Suppress the traceback and exit cleanly.
        sys.exit(0)


def _run(args: argparse.Namespace) -> None:
    if args.command == "presets":
        if args.json:
            print(json.dumps(
                [
                    {
                        "name": p.name,
                        "description": p.description,
                        "predictors": list(p.predictors),
                        "ranking": p.ranking,
                        "learning": p.learning,
                    }
                    for p in PRESETS.values()
                ],
                indent=2,
            ))
        else:
            print(describe_presets())
        return

    # Load persisted history
    store = JsonHistoryStore(args.history_path)
    persisted_history = store.load()

    # Load custom vocabulary if provided
    vocabulary = None
    if args.vocab_path is not None:
        from aac.vocabulary import vocabulary_from_file
        try:
            vocabulary = vocabulary_from_file(args.vocab_path, format=args.vocab_format)
        except FileNotFoundError:
            print(
                f"aac: error: vocabulary file not found: {args.vocab_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not vocabulary:
            print(
                f"aac: error: vocabulary file is empty or contains no valid words: {args.vocab_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Build engine from preset and attach history
    engine = build_engine(
        preset=args.preset,
        history=persisted_history,
        vocabulary=vocabulary,
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
