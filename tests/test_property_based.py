"""
Property-based tests using Hypothesis.

These tests verify invariants that are explicitly stated in the codebase
but cannot be exhaustively covered by example-based tests alone. The four
invariants under test are:

1. RankingExplanation arithmetic: ``final_score == base_score + history_boost``
   for all finite, non-negative score combinations - including after ``merge()``
   and ``apply_history_boost()``. Two floating-point precision bugs in these
   methods were found by Hypothesis and would not have been caught by
   example-based tests.

2. Ranker candidate-set preservation: no ranker may add or remove suggestions.
   Verified for LearningRanker and DecayRanker across arbitrary suggestion
   lists and history states.

3. History prefix-index consistency: ``entries_for_prefix()`` via the
   incremental index must always match a brute-force full scan, regardless
   of insertion order or prefix distribution.

4. History count accuracy: ``count(value)`` must equal the exact number of
   times that value was recorded, regardless of noise in the history.

Why Hypothesis here?
    Each invariant is enforced as a hard runtime contract (``__post_init__``,
    ``RuntimeError``, index consistency). Hypothesis generates thousands of
    inputs including degenerate cases that manual test authorship rarely
    reaches: zero values, large floats near representability limits, unicode
    prefixes, interleaved insertions across many prefixes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aac.domain.history import History
from aac.domain.types import ScoredSuggestion, Suggestion
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.explanation import RankingExplanation
from aac.ranking.learning import LearningRanker

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=_UTC)

# Finite, non-negative floats that won't cause overflow in arithmetic.
# NaN and infinity are excluded: the engine rejects them at its own boundary.
_finite_score = st.floats(
    min_value=0.0,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)

# Non-empty alphanumeric strings, deduplicated across lists.
_suggestion_value = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
    min_size=1,
    max_size=20,
)

# Non-empty list of unique suggestion values.
_unique_values = st.lists(
    _suggestion_value,
    min_size=1,
    max_size=15,
    unique=True,
)


# ---------------------------------------------------------------------------
# Invariant 1: RankingExplanation arithmetic
# ---------------------------------------------------------------------------


class TestRankingExplanationInvariant:
    """
    final_score == base_score + history_boost must hold for all valid inputs,
    including after merge() and apply_history_boost().

    Two floating-point precision bugs were found here by Hypothesis:
    - merge() was summing four terms independently; the four-way sum rounds
      differently from the two-way sum used by __post_init__ at ~1e8.
    - apply_history_boost() had the same issue: base + old_boost + new_boost
      rounds differently from base + (old_boost + new_boost).
    Both are fixed; these tests guard against regression.
    """

    @given(base_score=_finite_score, history_boost=_finite_score)
    def test_valid_explanation_satisfies_invariant(
        self, base_score: float, history_boost: float
    ) -> None:
        """Any finite non-negative (base, boost) pair must construct without error."""
        exp = RankingExplanation(
            value="word",
            base_score=base_score,
            history_boost=history_boost,
            final_score=base_score + history_boost,
            source="test",
        )
        assert abs(exp.final_score - (exp.base_score + exp.history_boost)) < 1e-9

    @given(
        base_score=_finite_score,
        history_boost=_finite_score,
        delta=st.floats(
            min_value=1e-6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    def test_inconsistent_final_score_raises(
        self, base_score: float, history_boost: float, delta: float
    ) -> None:
        """A final_score that diverges from base + boost by more than 1e-9 must raise."""
        with pytest.raises(ValueError, match="final_score"):
            RankingExplanation(
                value="word",
                base_score=base_score,
                history_boost=history_boost,
                final_score=base_score + history_boost + delta,
                source="test",
            )

    @given(base_score=_finite_score, boost_a=_finite_score, boost_b=_finite_score)
    def test_merge_preserves_invariant(
        self, base_score: float, boost_a: float, boost_b: float
    ) -> None:
        """merge() must produce an explanation that satisfies the invariant.

        Guards the floating-point fix: four-way sum (base + boost_a + 0 + boost_b)
        rounds differently from two-way sum (merged_base + merged_boost) at ~1e8.
        """
        exp_a = RankingExplanation(
            value="word",
            base_score=base_score,
            history_boost=boost_a,
            final_score=base_score + boost_a,
            source="ranker_a",
        )
        exp_b = RankingExplanation(
            value="word",
            base_score=0.0,
            history_boost=boost_b,
            final_score=boost_b,
            source="ranker_b",
        )
        merged = exp_a.merge(exp_b)
        assert abs(merged.final_score - (merged.base_score + merged.history_boost)) < 1e-9

    @given(
        base_score=_finite_score,
        existing_boost=_finite_score,
        new_boost=_finite_score,
    )
    def test_apply_history_boost_preserves_invariant(
        self, base_score: float, existing_boost: float, new_boost: float
    ) -> None:
        """apply_history_boost() must satisfy the invariant after application.

        Guards the three-way float sum bug: base + old_boost + new_boost
        rounds differently from base + (old_boost + new_boost) at ~1e8.
        """
        exp = RankingExplanation(
            value="word",
            base_score=base_score,
            history_boost=existing_boost,
            final_score=base_score + existing_boost,
            source="test",
        )
        boosted = exp.apply_history_boost(boost=new_boost, source="ranker")
        assert abs(boosted.final_score - (boosted.base_score + boosted.history_boost)) < 1e-9

    @given(base_score=_finite_score)
    def test_zero_boost_explanation(self, base_score: float) -> None:
        """Zero history_boost is valid (pure frequency ranking with no learning)."""
        exp = RankingExplanation(
            value="word",
            base_score=base_score,
            history_boost=0.0,
            final_score=base_score,
            source="score",
        )
        assert exp.history_boost == 0.0
        assert abs(exp.final_score - exp.base_score) < 1e-9


# ---------------------------------------------------------------------------
# Invariant 2: Ranker candidate-set preservation
# ---------------------------------------------------------------------------


class TestRankerCandidateSetPreservation:
    """
    LearningRanker and DecayRanker must never add or remove suggestions.

    The engine enforces this with a RuntimeError at runtime. These tests
    verify it holds across arbitrary suggestion lists and history states,
    catching any regression where a ranker might conditionally drop or
    duplicate candidates.
    """

    @given(
        values=_unique_values,
        scores=st.lists(_finite_score, min_size=1, max_size=15),
        history_events=st.lists(
            st.tuples(_suggestion_value, _suggestion_value),
            max_size=20,
        ),
        boost=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        dominance_ratio=st.floats(
            min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_learning_ranker_preserves_candidate_set(
        self,
        values: list[str],
        scores: list[float],
        history_events: list[tuple[str, str]],
        boost: float,
        dominance_ratio: float,
    ) -> None:
        """LearningRanker output values must equal input values as a set."""
        suggestions = [
            ScoredSuggestion(
                suggestion=Suggestion(value=v),
                score=scores[i % len(scores)],
            )
            for i, v in enumerate(values)
        ]

        history = History()
        for prefix, value in history_events:
            history.record(prefix, value, timestamp=_FIXED_NOW)

        ranker = LearningRanker(history, boost=boost, dominance_ratio=dominance_ratio)
        ranked = ranker.rank(values[0], suggestions)

        assert {s.suggestion.value for s in ranked} == {s.suggestion.value for s in suggestions}
        assert len(ranked) == len(suggestions)

    @given(
        values=_unique_values,
        scores=st.lists(_finite_score, min_size=1, max_size=15),
        history_events=st.lists(
            st.tuples(_suggestion_value, _suggestion_value),
            max_size=20,
        ),
        half_life=st.floats(
            min_value=60.0, max_value=86400.0, allow_nan=False, allow_infinity=False
        ),
        weight=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_decay_ranker_preserves_candidate_set(
        self,
        values: list[str],
        scores: list[float],
        history_events: list[tuple[str, str]],
        half_life: float,
        weight: float,
    ) -> None:
        """DecayRanker output values must equal input values as a set."""
        suggestions = [
            ScoredSuggestion(
                suggestion=Suggestion(value=v),
                score=scores[i % len(scores)],
            )
            for i, v in enumerate(values)
        ]

        history = History()
        for prefix, value in history_events:
            history.record(prefix, value, timestamp=_FIXED_NOW)

        ranker = DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=half_life),
            weight=weight,
            now=_FIXED_NOW,
        )
        ranked = ranker.rank(values[0], suggestions)

        assert {s.suggestion.value for s in ranked} == {s.suggestion.value for s in suggestions}
        assert len(ranked) == len(suggestions)

    @given(
        values=_unique_values,
        scores=st.lists(_finite_score, min_size=1, max_size=15),
    )
    def test_learning_ranker_no_history_preserves_order(
        self,
        values: list[str],
        scores: list[float],
    ) -> None:
        """With no history, LearningRanker must return suggestions in their original order."""
        suggestions = [
            ScoredSuggestion(
                suggestion=Suggestion(value=v),
                score=scores[i % len(scores)],
            )
            for i, v in enumerate(values)
        ]
        ranker = LearningRanker(History())
        ranked = ranker.rank("query", suggestions)
        assert [s.suggestion.value for s in ranked] == [s.suggestion.value for s in suggestions]


# ---------------------------------------------------------------------------
# Invariant 3 & 4: History prefix-index consistency and count accuracy
# ---------------------------------------------------------------------------


class TestHistoryIndexConsistency:
    """
    The incremental prefix index must always agree with a brute-force full scan.

    History maintains ``_by_prefix`` alongside ``_entries``. These tests verify
    the two representations stay consistent under arbitrary insertion sequences,
    including mixed prefixes, duplicate values, and unicode text.
    """

    @given(
        events=st.lists(
            st.tuples(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=20)),
            min_size=1,
            max_size=100,
        ),
        query_prefix=st.text(min_size=1, max_size=10),
    )
    def test_entries_for_prefix_matches_brute_force(
        self,
        events: list[tuple[str, str]],
        query_prefix: str,
    ) -> None:
        """entries_for_prefix() must match brute-force scan for any query prefix."""
        history = History()
        for prefix, value in events:
            history.record(prefix, value, timestamp=_FIXED_NOW)

        indexed = list(history.entries_for_prefix(query_prefix))
        brute = [e for e in history.entries() if e.prefix == query_prefix]

        assert len(indexed) == len(brute)
        assert all(a == b for a, b in zip(indexed, brute, strict=False))

    @given(
        events=st.lists(
            st.tuples(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=20)),
            min_size=1,
            max_size=100,
        ),
        query_prefix=st.text(min_size=1, max_size=10),
    )
    def test_counts_for_prefix_matches_brute_force(
        self,
        events: list[tuple[str, str]],
        query_prefix: str,
    ) -> None:
        """counts_for_prefix() must produce the same totals as a brute-force aggregation."""
        history = History()
        for prefix, value in events:
            history.record(prefix, value, timestamp=_FIXED_NOW)

        indexed_counts = history.counts_for_prefix(query_prefix)

        brute_counts: dict[str, int] = {}
        for e in history.entries():
            if e.prefix == query_prefix:
                brute_counts[e.value] = brute_counts.get(e.value, 0) + 1

        assert indexed_counts == brute_counts

    @given(
        prefixes=st.lists(
            st.text(min_size=1, max_size=5),
            min_size=2,
            max_size=6,
            unique=True,
        ),
        values=st.lists(
            st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        n_records=st.integers(min_value=10, max_value=200),
    )
    def test_index_consistent_across_all_prefixes_simultaneously(
        self,
        prefixes: list[str],
        values: list[str],
        n_records: int,
    ) -> None:
        """After many insertions, every prefix in the index must match brute-force."""
        import random

        rng = random.Random(42)
        history = History()
        for _ in range(n_records):
            p = rng.choice(prefixes)
            v = rng.choice(values)
            history.record(p, v, timestamp=_FIXED_NOW)

        for prefix in prefixes:
            indexed = list(history.entries_for_prefix(prefix))
            brute = [e for e in history.entries() if e.prefix == prefix]
            assert indexed == brute, (
                f"Index mismatch for prefix {prefix!r}: "
                f"indexed={len(indexed)}, brute={len(brute)}"
            )

    @given(
        value=st.text(min_size=1, max_size=20),
        n=st.integers(min_value=1, max_value=50),
    )
    def test_count_matches_exact_insertion_total(
        self,
        value: str,
        n: int,
    ) -> None:
        """count(value) must equal the exact number of times value was recorded."""
        history = History()
        prefix = "test"
        for _ in range(n):
            history.record(prefix, value, timestamp=_FIXED_NOW)
        for _ in range(3):
            history.record(prefix, value + "_noise", timestamp=_FIXED_NOW)

        assert history.count(value) == n
