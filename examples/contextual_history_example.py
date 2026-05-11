"""
contextual_history_example.py - domain-partitioned learning

Demonstrates ContextualHistory: the same prefix produces different
rankings depending on which domain the user is typing in.

The use-case is a code editor or multi-pane application where the same
prefix "python" means something different in:
  - a shell context (python commands, scripts)
  - a documentation search (Python language topics)
  - a package manager (Python packages)

Without domain partitioning, selections from all three contexts pollute
each other. ContextualHistory isolates them.

Run:
    python examples/contextual_history_example.py
"""
from __future__ import annotations

from aac.domain.contextual_history import ContextualHistory
from aac.presets import create_engine


def main() -> None:
    ctx = ContextualHistory()

    # Build domain-specific engines sharing the same ContextualHistory
    shell_engine = create_engine("default", history=ctx.for_domain("shell"))
    docs_engine  = create_engine("default", history=ctx.for_domain("docs"))
    pkg_engine   = create_engine("default", history=ctx.for_domain("packages"))

    # Simulate user interactions in each domain
    print("=== Simulating user selections across domains ===\n")

    # Shell domain: user frequently runs Python scripts
    for _ in range(5):
        ctx.record("py", "python3",      domain="shell")
    for _ in range(3):
        ctx.record("py", "python",       domain="shell")
    ctx.record("py", "pytest",           domain="shell")

    # Docs domain: user searches Python language topics
    for _ in range(6):
        ctx.record("py", "python syntax",    domain="docs")
    for _ in range(4):
        ctx.record("py", "python tutorial",  domain="docs")
    ctx.record("py", "python stdlib",        domain="docs")

    # Packages domain: user installs packages
    for _ in range(4):
        ctx.record("py", "pyarrow",      domain="packages")
    for _ in range(3):
        ctx.record("py", "pydantic",     domain="packages")
    for _ in range(2):
        ctx.record("py", "pytest",       domain="packages")

    # Now compare suggestions for "py" across domains
    prefix = "py"
    print(f'Prefix: "{prefix}"\n')

    print("Shell context (expects python3, python, pytest at top):")
    print(" ", shell_engine.suggest(prefix, limit=5))

    print("\nDocs context (expects python syntax, python tutorial at top):")
    print(" ", docs_engine.suggest(prefix, limit=5))

    print("\nPackages context (expects pyarrow, pydantic at top):")
    print(" ", pkg_engine.suggest(prefix, limit=5))

    print("\n=== Domain statistics ===")
    for domain, hist in sorted(ctx.domains()):
        entries = len(hist)
        top = list(hist.counts_for_prefix(prefix).items())[:3]
        print(f"  {domain}: {entries} entries, top for {prefix!r}: {top}")

    print("\n=== Explanation for shell domain ===")
    explanations = shell_engine.explain(prefix)
    for e in explanations[:3]:
        print(
            f"  {e.value:<25s}  base={e.base_score:.3f}  "
            f"boost=+{e.history_boost:.3f}  final={e.final_score:.3f}"
        )

    # Demonstrate that domains are truly isolated: recording in shell
    # doesn't affect docs suggestions
    print("\n=== Isolation check ===")
    for _ in range(20):
        ctx.record("py", "python3", domain="shell")

    docs_top = docs_engine.suggest(prefix, limit=3)
    shell_top = shell_engine.suggest(prefix, limit=3)
    assert "python3" == shell_top[0], "Shell top should be python3 after 25 selections"
    # Docs engine has never seen "python3" - it should not be in top results.
    # Shell and docs histories are fully isolated.
    assert "python3" not in docs_top, (
        f"Domain isolation failure: 'python3' appears in docs suggestions {docs_top} "
        f"despite only being recorded in the shell domain"
    )

    print("  Shell top:", shell_top[:3])
    print("  Docs top: ", docs_top[:3])
    print("  ✓ Domain isolation confirmed: 25 shell selections did not corrupt docs rankings")


if __name__ == "__main__":
    main()
