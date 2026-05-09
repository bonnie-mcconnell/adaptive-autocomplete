"""Command-line interface entry point. Routes subcommands to their handler functions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aac.cli import debug, demo, explain, history, record, suggest
from aac.cli.app import build_engine
from aac.presets import PRESETS, available_presets, compare_presets, describe_presets
from aac.storage.json_store import JsonHistoryStore

DEFAULT_HISTORY_PATH = Path("~/.aac_history.json").expanduser()
DEFAULT_LIMIT = 10

_PRESET_HELP = (
    "Engine preset (controls predictors, ranking, learning). "
    "Default: production. "
    "Choices: stateless, default, recency, production, robust. "
    "Run 'aac presets' to see what each preset does."
)


def _add_preset_arg(p: argparse.ArgumentParser) -> None:
    """Add --preset to a subcommand parser (standard position: after subcommand)."""
    p.add_argument(
        "--preset",
        default="production",
        choices=available_presets(),
        help=_PRESET_HELP,
    )


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    """Add --preset, --history-path, --vocab-path, --vocab-format to a subparser."""
    _add_preset_arg(p)
    p.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        metavar="PATH",
        help=f"Path to persisted history file (default: {DEFAULT_HISTORY_PATH})",
    )
    p.add_argument(
        "--vocab-path",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a custom vocabulary file. "
            "Replaces the bundled 48k English vocabulary. "
            "Format controlled by --vocab-format."
        ),
    )
    p.add_argument(
        "--vocab-format",
        default="wordlist",
        choices=["wordlist", "text"],
        help=(
            "'wordlist': one word per line (default). "
            "'text': free-form text, words weighted by frequency."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aac",
        description="Adaptive autocomplete engine with learning and explainability.",
        epilog=(
            "Examples:\n"
            "  aac suggest programing\n"
            "  aac suggest prog --preset stateless --limit 5\n"
            "  aac explain prog\n"
            "  aac explain prog --json\n"
            "  aac record prog programming\n"
            "  aac compare recieve\n"
            "  aac compare recieve --presets stateless production\n"
            "  aac history prog\n"
            "  aac presets\n"
            "  aac demo\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global --history-path so it can be placed before the subcommand.
    # Each subcommand also accepts --history-path via _add_shared_args() for
    # users who prefer the subcommand-local form. argparse resolves the
    # conflict with parse_known_args; we use set_defaults to let the
    # subcommand value win when both are supplied.
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        metavar="PATH",
        dest="history_path_global",
        help=f"Path to persisted history file (default: {DEFAULT_HISTORY_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── presets ───────────────────────────────────────────────────────
    presets_p = subparsers.add_parser(
        "presets",
        help="List available presets and what each one does",
    )
    presets_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON",
    )

    # ── suggest ───────────────────────────────────────────────────────
    suggest_p = subparsers.add_parser(
        "suggest",
        help="Get ranked completion suggestions for a prefix",
        description=(
            "Return ranked completion suggestions for a prefix.\n\n"
            "Examples:\n"
            "  aac suggest prog\n"
            "  aac suggest programing               # typo recovery\n"
            "  aac suggest prog --limit 5\n"
            "  aac suggest prog --preset stateless  # no learning\n"
            "  aac suggest prog --json              # pipe-friendly output\n"
            "  aac suggest prog --confidence        # show confidence scores\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    suggest_p.add_argument("text", help="Input prefix to complete")
    suggest_p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, metavar="N",
        help=f"Maximum suggestions to return (default: {DEFAULT_LIMIT})",
    )
    suggest_p.add_argument(
        "--json", action="store_true", default=False,
        help="Output as JSON array (pipe-friendly)",
    )
    suggest_p.add_argument(
        "--confidence", action="store_true", default=False,
        help="Include normalised confidence scores (0–1) alongside each suggestion",
    )
    _add_shared_args(suggest_p)

    # ── explain ───────────────────────────────────────────────────────
    explain_p = subparsers.add_parser(
        "explain",
        help="Show score breakdowns explaining why each suggestion ranked where it did",
        description=(
            "Show per-suggestion score breakdowns.\n\n"
            "Columns: final score, percentage of top score, base predictor score, history boost.\n\n"
            "Examples:\n"
            "  aac explain prog\n"
            "  aac explain prog --json     # full breakdown as JSON\n"
            "  aac explain prog --limit 5\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    explain_p.add_argument("text", help="Input prefix to explain")
    explain_p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, metavar="N",
        help=f"Maximum suggestions to explain (default: {DEFAULT_LIMIT})",
    )
    explain_p.add_argument(
        "--json", action="store_true", default=False,
        help=(
            "Output full per-predictor breakdown as JSON. "
            "Includes base_components, history_components, contribution_pct. "
            "Useful for debugging weight tuning."
        ),
    )
    _add_shared_args(explain_p)

    # ── batch ─────────────────────────────────────────────────────────
    batch_p = subparsers.add_parser(
        "batch",
        help="Get suggestions for multiple prefixes at once (JSON output)",
        description=(
            "Return suggestions for multiple prefixes in one call.\n\n"
            "Always outputs JSON (a dict mapping each prefix to its suggestion list).\n\n"
            "Examples:\n"
            "  aac batch prog hel wor\n"
            "  aac batch prog hel --limit 5\n"
            "  aac batch prog hel | jq '.prog'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    batch_p.add_argument(
        "texts", nargs="+",
        help="One or more input prefixes to complete",
    )
    batch_p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, metavar="N",
        help=f"Maximum suggestions per prefix (default: {DEFAULT_LIMIT})",
    )
    _add_shared_args(batch_p)

    # ── eval ──────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser(
        "eval",
        help="Evaluate an engine against a labelled query log",
        description=(
            "Run precision@k, MRR, NDCG, and hit-rate evaluation against\n"
            "a labelled query log.\n\n"
            "Examples:\n"
            "  aac eval --from-history                          # use ~/.aac_history.json\n"
            "  aac eval --query-log queries.jsonl               # labelled JSONL file\n"
            "  aac eval --from-history --preset stateless       # compare preset\n"
            "  aac eval --from-history --k 5                    # evaluate at k=5\n"
            "  aac eval --from-history --markdown               # table for README\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_group = eval_p.add_mutually_exclusive_group(required=True)
    eval_group.add_argument(
        "--from-history", action="store_true",
        help="Build query log from ~/.aac_history.json (uses recorded selections as ground truth)",
    )
    eval_group.add_argument(
        "--query-log", type=Path, metavar="PATH",
        help="Path to a JSONL file with labelled queries ({prefix, relevant, grades?})",
    )
    eval_p.add_argument(
        "--k", type=int, default=10, metavar="K",
        help="Evaluation depth - only top-K results scored (default: 10)",
    )
    eval_p.add_argument(
        "--min-count", type=int, default=1, metavar="N",
        help="Minimum selection count for a word to be relevant (--from-history only, default: 1)",
    )
    eval_p.add_argument(
        "--markdown", action="store_true",
        help="Output a Markdown table (paste into README)",
    )
    eval_p.add_argument(
        "--worst", type=int, default=0, metavar="N",
        help="Show the N worst-ranked queries (useful for debugging)",
    )
    _add_shared_args(eval_p)

    # ── tune ──────────────────────────────────────────────────────────
    tune_p = subparsers.add_parser(
        "tune",
        help="Find optimal predictor weights using coordinate descent",
        description=(
            "Automated weight optimisation. Finds the predictor weight\n"
            "combination that maximises MRR@10 over your query log.\n\n"
            "Examples:\n"
            "  aac tune --from-history\n"
            "  aac tune --from-history --metric ndcg\n"
            "  aac tune --from-history --strategy grid\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tune_group = tune_p.add_mutually_exclusive_group(required=True)
    tune_group.add_argument(
        "--from-history", action="store_true",
        help="Build query log from ~/.aac_history.json",
    )
    tune_group.add_argument(
        "--query-log", type=Path, metavar="PATH",
        help="Path to a JSONL file with labelled queries",
    )
    tune_p.add_argument(
        "--metric", default="mrr",
        choices=["mrr", "ndcg", "precision", "recall", "ap", "hit_rate"],
        help="Metric to maximise (default: mrr)",
    )
    tune_p.add_argument(
        "--strategy", default="coordinate",
        choices=["coordinate", "grid"],
        help="Optimisation strategy (default: coordinate)",
    )
    tune_p.add_argument(
        "--k", type=int, default=10, help="Evaluation depth (default: 10)",
    )
    _add_shared_args(tune_p)

    # ── compare ───────────────────────────────────────────────────────
    compare_p = subparsers.add_parser(
        "compare",
        help="Compare how different presets rank the same prefix side-by-side",
        description=(
            "Compare suggestion rankings across multiple presets.\n\n"
            "Shows which presets recover typos, how scores differ, and where history\n"
            "boost applies. Engines are cached after the first build.\n\n"
            "Examples:\n"
            "  aac compare recieve                      # all presets\n"
            "  aac compare recieve --presets stateless production\n"
            "  aac compare prog --limit 5\n"
            "  aac compare recieve --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_p.add_argument("text", help="Input prefix to compare across presets")
    compare_p.add_argument(
        "--presets", nargs="+", default=None, metavar="PRESET",
        choices=available_presets(),
        help="Presets to compare (default: all). E.g. --presets stateless production",
    )
    compare_p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, metavar="N",
        help=f"Maximum suggestions per preset (default: {DEFAULT_LIMIT})",
    )
    compare_p.add_argument(
        "--json", action="store_true", default=False,
        help="Output raw comparison rows as JSON",
    )
    compare_p.add_argument(
        "--vocab-path", type=Path, default=None, metavar="PATH",
        help="Custom vocabulary shared across all preset engines",
    )
    compare_p.add_argument(
        "--vocab-format", default="wordlist", choices=["wordlist", "text"],
    )

    # ── record ────────────────────────────────────────────────────────
    record_p = subparsers.add_parser(
        "record",
        help="Record a user selection so the engine learns from it",
        description=(
            "Record that a user selected a completion for a given prefix.\n"
            "The engine learns immediately; next suggest call will reflect the update.\n\n"
            "Example:\n"
            "  aac record prog programming\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    record_p.add_argument("text", help="Input prefix the user was completing")
    record_p.add_argument("value", help="Completion the user selected")
    _add_shared_args(record_p)

    # ── history ───────────────────────────────────────────────────────
    history_p = subparsers.add_parser(
        "history",
        help="Show what the engine has learned (selection counts and timestamps)",
    )
    history_p.add_argument(
        "prefix", nargs="?", default=None,
        help="Prefix to inspect. Omit to show a summary of all prefixes.",
    )
    history_p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Maximum entries to show (default: {DEFAULT_LIMIT})",
    )
    history_p.add_argument(
        "--history-path", type=Path, default=DEFAULT_HISTORY_PATH, metavar="PATH",
        help=f"Path to persisted history file (default: {DEFAULT_HISTORY_PATH})",
    )
    history_p.add_argument(
        "--json", action="store_true", default=False,
        help="Output history as JSON",
    )

    # ── debug ─────────────────────────────────────────────────────────
    debug_p = subparsers.add_parser(
        "debug",
        help="Show verbose internal pipeline state (pre-ranking scores, ranker steps)",
    )
    debug_p.add_argument("text", help="Input prefix to debug")
    _add_shared_args(debug_p)

    # ── demo ──────────────────────────────────────────────────────────
    demo_p = subparsers.add_parser(
        "demo",
        help="Start an interactive browser demo",
        description=(
            "Open a local browser UI showing live suggestions with confidence bars,\n"
            "per-suggestion score breakdowns, and a preset comparison table.\n"
            "No account, API key, or internet required.\n\n"
            "Example:\n"
            "  aac demo\n"
            "  aac demo --port 9000 --no-browser\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    demo_p.add_argument("--host", default="127.0.0.1",
                        help=(
                            "Interface to bind to (default: 127.0.0.1 - localhost only). "
                            "Use 0.0.0.0 when running inside Docker so the container "
                            "port is reachable from the host machine."
                        ))
    demo_p.add_argument("--port", type=int, default=8421,
                        help="Local port for the demo server (default: 8421)")
    demo_p.add_argument("--no-browser", action="store_true", default=False,
                        help="Print the URL but do not open the browser automatically")
    _add_shared_args(demo_p)

    args = parser.parse_args()

    # Resolve history_path: subcommand-local --history-path wins; global fallback
    # lets users write `aac --history-path /path record he hello` (global form).
    if not hasattr(args, "history_path"):
        args.history_path = args.history_path_global

    try:
        _run(args)
    except BrokenPipeError:
        # Consumer closed the pipe (e.g. `aac suggest he | head -5`).
        # Suppress the traceback; exit cleanly.
        sys.exit(0)


def _load_vocabulary(args: argparse.Namespace) -> dict[str, int] | None:
    """Load custom vocabulary from --vocab-path if provided."""
    vocab_path = getattr(args, "vocab_path", None)
    if vocab_path is None:
        return None
    from aac.vocabulary import vocabulary_from_file
    try:
        vocab = vocabulary_from_file(vocab_path, format=getattr(args, "vocab_format", "wordlist"))
    except FileNotFoundError:
        print(f"aac: error: vocabulary file not found: {vocab_path}", file=sys.stderr)
        sys.exit(1)
    if not vocab:
        print(
            f"aac: error: vocabulary file is empty or contains no valid words: {vocab_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    return vocab


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
                    for name in available_presets()
                    for p in [PRESETS[name]]
                ],
                indent=2,
            ))
        else:
            print(describe_presets())
        return

    if args.command == "compare":
        _run_compare(args)
        return

    if args.command == "history":
        store = JsonHistoryStore(args.history_path)
        loaded = store.load()
        if getattr(args, "json", False):
            _run_history_json(loaded, args)
        else:
            history.run(history=loaded, prefix=args.prefix, limit=args.limit)
        return

    # All other commands need a full engine
    vocabulary = _load_vocabulary(args)
    history_path = getattr(args, "history_path", DEFAULT_HISTORY_PATH)
    store = JsonHistoryStore(history_path)
    persisted_history = store.load()

    engine = build_engine(
        preset=args.preset,
        history=persisted_history,
        vocabulary=vocabulary,
    )

    if args.command == "eval":
        _run_eval(args, engine, store)
        return
    elif args.command == "tune":
        _run_tune(args, engine)
        return

    if args.command == "suggest":
        _run_suggest(args, engine)
    elif args.command == "explain":
        _run_explain(args, engine)
    elif args.command == "batch":
        results = engine.batch_suggest(args.texts, limit=args.limit)
        print(json.dumps(results, indent=2))
    elif args.command == "record":
        record.run(engine=engine, store=store, text=args.text, value=args.value)
    elif args.command == "debug":
        debug.run(engine=engine, text=args.text)
    elif args.command == "demo":
        demo.run(
            engine=engine,
            host=getattr(args, "host", "127.0.0.1"),
            port=args.port,
            preset=args.preset,
            no_browser=args.no_browser,
        )


def _run_suggest(args: argparse.Namespace, engine: object) -> None:
    from aac.engine.engine import AutocompleteEngine
    if not isinstance(engine, AutocompleteEngine):
        raise TypeError(f"Expected AutocompleteEngine, got {type(engine).__name__}")

    if args.confidence:
        results = engine.suggest_with_confidence(args.text, limit=args.limit)
        if args.json:
            print(json.dumps(
                [{"suggestion": w, "confidence": round(c, 4)} for w, c in results],
                indent=2,
            ))
        else:
            if not results:
                print("(no suggestions available)")
                return
            for word, conf in results:
                bar = "█" * int(conf * 10)
                print(f"{word:<20s}  {conf:.0%}  {bar}")
    else:
        suggestions = engine.suggest(args.text, limit=args.limit)
        if args.json:
            print(json.dumps(suggestions, indent=2))
        else:
            suggest.run(engine=engine, text=args.text, limit=args.limit)


def _run_explain(args: argparse.Namespace, engine: object) -> None:
    from aac.engine.engine import AutocompleteEngine
    if not isinstance(engine, AutocompleteEngine):
        raise TypeError(f"Expected AutocompleteEngine, got {type(engine).__name__}")

    if args.json:
        dicts = engine.explain_as_dicts(args.text)[:args.limit]
        print(json.dumps(dicts, indent=2))
    else:
        explain.run(engine=engine, text=args.text, limit=args.limit)


def _run_compare(args: argparse.Namespace) -> None:
    vocabulary = _load_vocabulary(args)

    presets_arg: list[str] | None = getattr(args, "presets", None)

    print("Building engines for comparison (first run may take a few seconds)...",
          file=sys.stderr, end="\r")

    result = compare_presets(
        args.text,
        presets=presets_arg,
        vocabulary=vocabulary,
        limit=args.limit,
    )

    # Clear the "building..." message
    print(" " * 70, file=sys.stderr, end="\r")

    if args.json:
        print(json.dumps({
            "text": result.text,
            "presets": result.presets,
            "rows": result.rows,
        }, indent=2))
    else:
        table = result.to_table(limit=args.limit)
        if not table.strip():
            print(f"No suggestions for {args.text!r} in any preset.")
        else:
            print(table)


def _run_history_json(loaded_history: object, args: argparse.Namespace) -> None:
    from aac.domain.history import History
    if not isinstance(loaded_history, History):
        raise TypeError(f"Expected History, got {type(loaded_history).__name__}")

    prefix = args.prefix
    limit = args.limit

    if prefix:
        counts = loaded_history.counts_for_prefix(prefix)
        entries = [
            {"prefix": prefix, "value": v, "count": c}
            for v, c in sorted(counts.items(), key=lambda x: -x[1])[:limit]
        ]
    else:
        snap = loaded_history.snapshot_counts()
        entries = []
        for pfx, counts in sorted(snap.items()):
            for val, cnt in sorted(counts.items(), key=lambda x: -x[1])[:limit]:
                entries.append({"prefix": pfx, "value": val, "count": cnt})

    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()


def _run_eval(
    args: argparse.Namespace,
    engine: object,
    store: object,
) -> None:
    from aac.engine.engine import AutocompleteEngine
    from aac.evaluation import EvaluationHarness
    from aac.evaluation.datasets import load_jsonl
    if not isinstance(engine, AutocompleteEngine):
        raise TypeError(f"Expected AutocompleteEngine, got {type(engine).__name__}")

    if getattr(args, "from_history", False):
        from aac.storage.json_store import JsonHistoryStore
        if not isinstance(store, JsonHistoryStore):
            raise TypeError(f"Expected JsonHistoryStore, got {type(store).__name__}")
        history = store.load()
        try:
            harness = EvaluationHarness.from_history(
                history, k=args.k, min_count=args.min_count
            )
        except ValueError as e:
            print(f"aac eval: {e}", file=sys.stderr)
            print(
                "Tip: record some selections with 'aac record PREFIX WORD' first,\n"
                "     or use --query-log with a labelled JSONL file.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        try:
            log = load_jsonl(args.query_log)
            harness = EvaluationHarness(log, k=args.k)
        except FileNotFoundError:
            print(f"aac eval: query log not found: {args.query_log}", file=sys.stderr)
            sys.exit(1)

    result = harness.run(engine)

    if args.markdown:
        print(result.to_markdown_table())
    else:
        print(result.summary())
        if result.by_prefix_length:
            print("\nBy prefix length:")
            for length, metrics in sorted(result.by_prefix_length.items()):
                bar = "█" * int(metrics["mrr"] * 20)
                print(
                    f"  len={length}  n={int(metrics['n']):<4}  "
                    f"MRR={metrics['mrr']:.3f}  {bar}"
                )

    if getattr(args, "worst", 0) > 0:
        print(f"\nWorst {args.worst} queries:")
        for qr in result.worst_queries(args.worst):
            print(
                f"  {qr.entry.prefix!r:<15}  MRR={qr.mrr:.3f}  "
                f"got={qr.ranked[:3]}  expected={sorted(qr.entry.relevant)[:3]}"
            )


def _run_tune(args: argparse.Namespace, engine: object) -> None:
    from aac.engine.engine import AutocompleteEngine
    from aac.evaluation import EvaluationHarness, WeightOptimiser
    from aac.evaluation.datasets import load_jsonl
    if not isinstance(engine, AutocompleteEngine):
        raise TypeError(f"Expected AutocompleteEngine, got {type(engine).__name__}")

    preset = getattr(args, "preset", "production")

    if getattr(args, "from_history", False):
        history = engine.history
        try:
            harness = EvaluationHarness.from_history(history, k=args.k)
        except ValueError as e:
            print(f"aac tune: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            log = load_jsonl(args.query_log)
            harness = EvaluationHarness(log, k=args.k)
        except FileNotFoundError:
            print(f"aac tune: query log not found: {args.query_log}", file=sys.stderr)
            sys.exit(1)

    opt = WeightOptimiser(harness, metric=args.metric, verbose=True)

    # Default weight search space for built-in predictors.
    # Key names must match predictor.name as returned by engine.describe().
    # AdaptiveSymSpellPredictor.name == "symspell" (not "adaptive_symspell") so
    # that explain() base_components and EngineConfig serialisation are consistent
    # whether the engine uses SymSpellPredictor or AdaptiveSymSpellPredictor.
    # Filtered below to only include predictors present in the target preset.
    full_weight_grid = {
        "frequency": [0.5, 0.8, 1.0, 1.5, 2.0],
        "history":   [0.8, 1.0, 1.2, 1.5, 2.0],
        "symspell":  [0.2, 0.35, 0.5, 0.7],
        "trigram":   [0.2, 0.4, 0.6, 0.8],
    }

    # Build the engine template once to discover which predictors exist in this preset.
    # This also pre-warms the optimiser's predictor cache.
    opt._get_base_weighted_predictors(preset)
    preset_predictor_names = {
        wp.predictor.name
        for wp in opt._predictor_cache[preset]
    }

    # Filter: only tune predictors that exist in the preset.
    # Tuning 'symspell' weight on 'stateless' (which has no symspell predictor)
    # would produce confusing output showing 0-impact weight changes.
    weight_grid = {
        name: vals
        for name, vals in full_weight_grid.items()
        if name in preset_predictor_names
    }

    if not weight_grid:
        print(
            f"aac tune: no tunable predictors found in preset {preset!r}. "
            f"Preset predictors: {sorted(preset_predictor_names)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.strategy == "grid":
        # Grid search with coarser grid to keep it tractable
        coarse_grid = {k: [v[0], v[len(v)//2], v[-1]] for k, v in weight_grid.items()}
        result = opt.grid_search(base_preset=preset, weight_grid=coarse_grid)
    else:
        result = opt.coordinate_descent(base_preset=preset, weight_grid=weight_grid)

    print()
    print(result.report())
    print()
    print("To use these weights:")
    print("  from aac.domain.types import WeightedPredictor")
    print("  from aac.engine.engine import AutocompleteEngine")
    for name, weight in sorted(result.best_weights.items()):
        print(f"  # {name}: {weight:.3f}")
