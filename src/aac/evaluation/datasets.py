"""
Query log structures and dataset generators.

A QueryLog is a list of (query_text, relevant_completions) pairs that
represent ground-truth relevance for evaluation. Three sources:

1. From a History object  - uses recorded selections as ground truth.
   The most realistic source: it reflects actual user behaviour.

2. Synthetic              - generates query prefixes from a vocabulary
   with known relevant completions (prefix-match + typo variants).
   Useful for unit testing the evaluation harness itself.

3. From a JSONL file      - loads a human-labelled query log for
   production-quality evaluation. Format described in load_jsonl().
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aac.domain.history import History


@dataclass
class QueryLogEntry:
    """A single evaluation query: a prefix and the set of relevant completions (ground truth)."""
    prefix: str
    relevant: set[str]
    grades: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prefix:
            raise ValueError("QueryLogEntry prefix must be non-empty")
        if not self.relevant:
            raise ValueError(
                f"QueryLogEntry for prefix {self.prefix!r} has no relevant completions. "
                "An entry with no relevant completions cannot contribute to any metric."
            )


#: A query log is a list of entries.
QueryLog = list[QueryLogEntry]


def make_query_log_from_history(
    history: History,
    *,
    min_count: int = 1,
    max_entries: int | None = None,
    seed: int = 42,
) -> QueryLog:
    """
    Build a QueryLog from a History object, using recorded selections as ground truth.

    Produces one QueryLogEntry per prefix. relevant = values selected >= min_count times.
    Grade for each completion = selection_count / max_selection_count
    for that prefix (so the most-selected word gets grade 1.0).

    Parameters:
        history:      History instance to build the log from.
        min_count:    Minimum selection count for a value to be
                      considered relevant. Default: 1.
        max_entries:  Cap on the number of entries. If the history has
                      more prefixes than this, a deterministic subsample
                      is taken using ``seed``.
        seed:         Random seed for reproducible subsampling.
                      Only used when ``max_entries`` is set. Default: 42.

    Returns:
        QueryLog with one entry per prefix that has at least one
        relevant completion.

    Example::

        from aac.presets import create_engine
        from aac.evaluation import make_query_log_from_history, EvaluationHarness

        engine = create_engine("production")
        # ... record_selection() calls or load from JsonHistoryStore ...

        log = make_query_log_from_history(engine.history, min_count=2)
        harness = EvaluationHarness(log)
        result = harness.run(engine)
        print(result.summary())
    """
    counts = history.snapshot_counts()

    entries: list[QueryLogEntry] = []
    for prefix, word_counts in counts.items():
        relevant_words = {
            word: count
            for word, count in word_counts.items()
            if count >= min_count
        }
        if not relevant_words:
            continue

        max_count = max(relevant_words.values())
        grades = {
            word: count / max_count
            for word, count in relevant_words.items()
        }

        entries.append(QueryLogEntry(
            prefix=prefix,
            relevant=set(relevant_words.keys()),
            grades=grades,
        ))

    if max_entries is not None and len(entries) > max_entries:
        # Use a seeded RNG so the same history always produces the same
        # subsample. Without seeding, evaluation results vary between runs
        # even with identical inputs, making metric comparisons meaningless.
        rng = random.Random(seed)
        entries = rng.sample(entries, max_entries)

    return entries


def make_synthetic_query_log(
    vocabulary: list[str],
    *,
    prefix_lengths: list[int] | None = None,
    include_typos: bool = True,
    seed: int = 42,
) -> QueryLog:
    """
    Generate a synthetic QueryLog from a vocabulary.

    For each (word, prefix_length) combination, the relevant completion
    is the word itself (exact prefix match). If ``include_typos`` is True,
    adds typo variants of some prefixes where the relevant set contains
    the correctly-spelled word.

    Useful for:
    - Unit testing the EvaluationHarness without needing real user data
    - Benchmarking different presets on a controlled vocabulary
    - Establishing a baseline before collecting real query logs

    Parameters:
        vocabulary:      List of vocabulary words.
        prefix_lengths:  Prefix lengths to generate. Default: [2, 3, 4].
        include_typos:   Whether to add typo-prefix entries. Default: True.
        seed:            Random seed for reproducibility.

    Returns:
        QueryLog with entries for prefix-match and (optionally) typo cases.
    """
    rng = random.Random(seed)
    if prefix_lengths is None:
        prefix_lengths = [2, 3, 4]

    entries: dict[str, QueryLogEntry] = {}

    for word in vocabulary:
        for length in prefix_lengths:
            if len(word) <= length:
                continue
            prefix = word[:length]
            if prefix in entries:
                entries[prefix].relevant.add(word)
            else:
                entries[prefix] = QueryLogEntry(
                    prefix=prefix,
                    relevant={word},
                )

    if include_typos:
        # Add a small number of typo entries from a random sample
        sample = rng.sample(vocabulary, min(50, len(vocabulary)))
        for word in sample:
            if len(word) < 4:
                continue
            # Single character deletion typo
            pos = rng.randint(1, len(word) - 2)
            typo = word[:pos] + word[pos + 1:]
            if typo not in entries:
                entries[typo] = QueryLogEntry(
                    prefix=typo,
                    relevant={word},
                )

    return list(entries.values())


def load_jsonl(path: Path) -> QueryLog:
    """
    Load a query log from a JSONL file (one JSON object per line).

    Expected format per line::

        {"prefix": "prog", "relevant": ["programming", "program"], "grades": {"programming": 1.0, "program": 0.8}}

    The ``grades`` field is optional. If absent, binary relevance is assumed.

    Parameters:
        path: Path to the JSONL file.

    Returns:
        QueryLog.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line has invalid format.
    """
    entries = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                entries.append(QueryLogEntry(
                    prefix=obj["prefix"],
                    relevant=set(obj["relevant"]),
                    grades=obj.get("grades", {}),
                ))
            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError(
                    f"Invalid query log entry at line {i} in {path}: {e}\n"
                    f"Expected: {{\"prefix\": \"...\", \"relevant\": [...], \"grades\": {{...}}}}"
                ) from e
    return entries


def save_jsonl(log: QueryLog, path: Path) -> None:
    """Save a QueryLog to JSONL. Creates parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in log:
            obj: dict[str, object] = {
                "prefix": entry.prefix,
                "relevant": sorted(entry.relevant),
            }
            if entry.grades:
                obj["grades"] = entry.grades
            f.write(json.dumps(obj) + "\n")
