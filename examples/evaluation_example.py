"""
Offline evaluation and weight optimisation example.

    python examples/evaluation_example.py
"""
from __future__ import annotations

import time

from aac.data import load_english_frequencies
from aac.evaluation import EvaluationHarness, WeightOptimiser
from aac.evaluation.datasets import QueryLogEntry, make_synthetic_query_log, save_jsonl
from aac.presets import create_engine


def main() -> None:
    print("=" * 60)
    print("adaptive-autocomplete: Offline Evaluation Example")
    print("=" * 60)

    # ---------------------------------------------------------------------------
    # 1. Build a synthetic query log
    # ---------------------------------------------------------------------------
    print("\n[1] Building synthetic query log from English vocabulary...")
    vocab = list(load_english_frequencies().keys())[:600]

    log = make_synthetic_query_log(
        vocab,
        prefix_lengths=[2, 3, 4],
        include_typos=True,
        seed=42,
    )
    print(f"    Query log: {len(log)} entries")
    print(f"    Sample prefixes: {[e.prefix for e in log[:5]]}")
    print(f"    Sample relevant: {list(log[0].relevant)[:3]}")

    harness = EvaluationHarness(log, k=10)

    # ---------------------------------------------------------------------------
    # 2. Compare presets on the same log
    # ---------------------------------------------------------------------------
    print("\n[2] Comparing presets (same query log, same k=10)...")
    print(f"\n{'Preset':<14}  {'P@10':>6}  {'MRR':>6}  {'NDCG':>6}  {'MAP':>6}  {'Hit%':>6}  {'Time':>6}")
    print("-" * 62)

    preset_results = {}
    for preset in ["stateless", "default", "production"]:
        engine = create_engine(preset)
        t0 = time.perf_counter()
        result = harness.run(engine)
        elapsed = (time.perf_counter() - t0) * 1000
        preset_results[preset] = result
        print(
            f"{preset:<14}  "
            f"{result.mean_precision:>6.3f}  "
            f"{result.mean_mrr:>6.3f}  "
            f"{result.mean_ndcg:>6.3f}  "
            f"{result.mean_ap:>6.3f}  "
            f"{result.hit_rate:>5.1%}  "
            f"{elapsed:>5.0f}ms"
        )

    print("""
Note: 'stateless' may score higher than 'production' on a synthetic prefix-match log.
This is expected: the synthetic log's relevant sets are built from exact prefix matches,
which is exactly what frequency sorting optimises for. SymSpell/trigram sometimes surface
near-miss words that score above the exact-prefix candidates (e.g. 'bus' → 'busy' before
'business'). On a real query log built from user history, production beats stateless
because it recovers typos (MRR=0.0 → MRR=0.79) that dominate real-world queries.
    """.strip())

    # Show typo-recovery comparison where production's advantage is clear
    typo_log = [
        QueryLogEntry("programing", {"programming"}),
        QueryLogEntry("recieve", {"receive"}),
        QueryLogEntry("definitly", {"definitely"}),
        QueryLogEntry("occured", {"occurred"}),
        QueryLogEntry("seperate", {"separate"}),
    ]
    typo_harness = EvaluationHarness(typo_log, k=5)
    print("\nTypo-recovery only (stateless has no SymSpell):")
    for preset in ["stateless", "robust", "production"]:
        r = typo_harness.run(create_engine(preset))
        print(f"  {preset:<14}  MRR={r.mean_mrr:.3f}  Hit={r.hit_rate:.0%}")

    # ---------------------------------------------------------------------------
    # 3. Show per-prefix-length breakdown for production preset
    # ---------------------------------------------------------------------------
    print("\n[3] Per-prefix-length MRR for 'production' preset:")
    print(f"\n{'Length':<10}  {'Queries':>8}  {'MRR':>6}  {'NDCG':>6}  {'Hit%':>6}  Bar")
    print("-" * 55)
    breakdown = preset_results["production"].by_prefix_length
    for length, metrics in sorted(breakdown.items()):
        bar = "█" * int(metrics["mrr"] * 25)
        print(
            f"len={length:<6}  "
            f"{int(metrics['n']):>8}  "
            f"{metrics['mrr']:>6.3f}  "
            f"{metrics['ndcg']:>6.3f}  "
            f"{metrics['hit_rate']:>5.1%}  "
            f"{bar}"
        )

    # ---------------------------------------------------------------------------
    # 4. Show worst queries - where does the engine fail?
    # ---------------------------------------------------------------------------
    print("\n[4] 5 hardest queries for 'production' preset:")
    for qr in preset_results["production"].worst_queries(5):
        got = qr.ranked[:3] if qr.ranked else ["(none)"]
        expected = sorted(qr.entry.relevant)[:2]
        print(
            f"  {qr.entry.prefix!r:<12}  MRR={qr.mrr:.3f}  "
            f"got={got}  expected={expected}"
        )

    # ---------------------------------------------------------------------------
    # 5. Markdown table for README
    # ---------------------------------------------------------------------------
    print("\n[5] Markdown table (paste into README):")
    print()
    print(preset_results["production"].to_markdown_table())

    # ---------------------------------------------------------------------------
    # 6. Weight optimisation on the default preset
    # ---------------------------------------------------------------------------
    print("\n[6] Coordinate descent weight optimisation (default preset, metric=MRR)...")
    opt = WeightOptimiser(harness, metric="mrr", verbose=False)
    opt_result = opt.coordinate_descent(
        base_preset="default",
        weight_grid={
            "frequency": [0.5, 1.0, 1.5, 2.0],
            "history":   [0.8, 1.0, 1.2, 1.6],
        },
        max_rounds=3,
    )
    print()
    print(opt_result.report())

    improvement = opt_result.improvement_pct
    if improvement > 0.1:
        print(f"\n✓ Optimisation improved MRR by {improvement:.1f}%")
    else:
        print(f"\n✓ Preset weights already near-optimal for this query log (Δ={improvement:.2f}%)")

    # ---------------------------------------------------------------------------
    # 7. Save query log for reuse
    # ---------------------------------------------------------------------------
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "query_log.jsonl"
        save_jsonl(log, path)
        print(f"\n[7] Query log saved to {path} ({path.stat().st_size} bytes)")
        print("    Load with: from aac.evaluation.datasets import load_jsonl")
        print(f"    First line: {path.read_text().splitlines()[0]}")

    print("\n" + "=" * 60)
    print("Done. See aac.evaluation for full API docs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
