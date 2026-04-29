"""
CLI subcommand: aac history [prefix]

Shows what the engine has learned - selection counts and recency
per prefix, sorted by count descending.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from aac.domain.history import History, HistoryEntry


def run(*, history: History, prefix: str | None, limit: int) -> None:
    """
    Print a summary of recorded selections.

    With no prefix: shows the top-level prefix summary (total selections
    per prefix, most-selected value for each).

    With a prefix: shows per-value selection counts for that prefix,
    sorted by count descending, with recency information.
    """
    entries = list(history.entries())

    if not entries:
        print("No history recorded yet.")
        print("Use 'aac record <text> <value>' to record selections.")
        return

    if prefix is None:
        _show_summary(entries, limit)
    else:
        _show_prefix(entries, prefix.lower(), limit)


def _show_summary(entries: list[HistoryEntry], limit: int) -> None:
    """Show top-level summary: counts per prefix."""
    from collections import Counter

    prefix_counts: Counter[str] = Counter()
    prefix_top: dict[str, str] = {}
    prefix_value_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for e in entries:
        prefix_counts[e.prefix] += 1
        prefix_value_counts[e.prefix][e.value] += 1

    for pfx in prefix_counts:
        prefix_top[pfx] = prefix_value_counts[pfx].most_common(1)[0][0]

    print(f"{'Prefix':<20} {'Selections':>10}  {'Top completion'}")
    print("─" * 55)
    for pfx, count in prefix_counts.most_common(limit):
        top = prefix_top[pfx]
        print(f"  {pfx:<18} {count:>10}  {top!r}")

    total = sum(prefix_counts.values())
    unique_prefixes = len(prefix_counts)
    print()
    print(f"Total: {total} selections across {unique_prefixes} prefix(es).")
    print("Run 'aac history <prefix>' to see per-value breakdown for a prefix.")


def _show_prefix(entries: list[HistoryEntry], prefix: str, limit: int) -> None:
    """Show per-value breakdown for a specific prefix."""
    now = datetime.now(tz=timezone.utc)

    value_counts: dict[str, int] = defaultdict(int)
    value_last_seen: dict[str, datetime] = {}

    for e in entries:
        if e.prefix == prefix:
            value_counts[e.value] += 1
            if e.value not in value_last_seen or e.timestamp > value_last_seen[e.value]:
                value_last_seen[e.value] = e.timestamp

    if not value_counts:
        print(f"No history recorded for prefix {prefix!r}.")
        return

    print(f"History for prefix {prefix!r}:")
    print(f"  {'Value':<24} {'Count':>6}  {'Last selected'}")
    print("  " + "─" * 50)

    for value, count in sorted(value_counts.items(), key=lambda kv: -kv[1])[:limit]:
        last = value_last_seen.get(value)
        if last:
            elapsed = now - last
            secs = int(elapsed.total_seconds())
            if secs < 60:
                ago = f"{secs}s ago"
            elif secs < 3600:
                ago = f"{secs // 60}m ago"
            elif secs < 86400:
                ago = f"{secs // 3600}h ago"
            else:
                ago = f"{secs // 86400}d ago"
        else:
            ago = "unknown"
        print(f"  {value:<24} {count:>6}  {ago}")

    total = sum(value_counts.values())
    print(f"\n  Total: {total} selections for {prefix!r}.")
