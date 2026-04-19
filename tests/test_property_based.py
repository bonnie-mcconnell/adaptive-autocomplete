"""
Property-based tests using Hypothesis.

These tests verify invariants that are explicitly stated in the codebase
but cannot be exhaustively covered by example-based tests alone.  The
three invariants under test are:

1. RankingExplanation: ``final_score == base_score + history_boost``
   for all finite, non-negative score combinations - including edge cases
   like zero boost, zero base, and very large values.

2. Ranker candidate-set preservation: the engine's ``_apply_ranking``
   contract guarantees that no ranker can add or remove suggestions.
   We verify this holds for every LearningRanker and DecayRanker call
   across arbitrary suggestion lists and history states.

3. History prefix-index consistency: ``entries_for_prefix()`` via the
   incremental index must always return the same result as a brute-force
   full scan, regardless of insertion order or prefix distribution.

Why Hypothesis here?
    The invariants are stated as hard contracts enforced at runtime
    (``__post_init__`` for explanations, ``RuntimeError`` for rankers,
    and the index is relied on throughout the codebase).  Hypothesis
    generates thousands of inputs including degenerate cases that manual
    test authorship rarely reaches - zero values, NaN-adjacent floats,
    unicode prefixes, interleaved insertions across many prefixes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aac.domain.history import History, HistoryEntry
from aac.domain.types import (
    CompletionContext,
    ScoredSuggestion,
    Suggestion,
)
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.explanation import RankingExplanation
from aac.ranking.learning import LearningRanker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=_UTC)

# Strategy for finite, non-negative floats that won't cause overflow in
# arithmetic.  We exclude NaN and infinity because the engine rejects them
# at the validation boundary (math.isfinite check in _apply_ranking).
_finite_score = st.floats(
    min_value=0.0,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)

# Strategy for non-empty ASCII suggestion values, deduplicated.
_suggestion_value = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
    min_size=1,
    max_size=20,
)

# Strategy for a non-empty list of unique suggestion values.
_unique_values = st.lists(
    _suggestion_value,
    min_size=1,
    max_size=15,
    unique=True,
)

# Strategy for a scored suggestion with a finite, non-negative score.
def _scored_suggestion(value: str) -> st.SearchStrategy[ScoredSuggestion]:
    return _finite_score.map(
        lambda score: ScoredSuggestion(suggestion=Suggestion(value=value), score=score)
    )


# ---------------------------------------------------------------------------
# Invariant 1: RankingExplanation arithmetic
# ---------------------------------------------------------------------------

class TestRankingExplanationInvariant:
    """
    final_score == base_score + history_boost must hold for all valid inputs.

    The invariant is enforced in ``__post_init__``, so a violation raises
    ``ValueError`` at construction time.  We verify both that well-formed
    inputs construct without error and that the arithmetic identity is
    provably satisfied.
    """

    @given(
        base_score=_finite_score,
        history_boost=_finite_score,
    )
    def test_valid_explanation_satisfies_invariant(
        self, base_score: float, history_boost: float
    ) -> None:
        """Any finite non-negative (base, boost) pair produces a valid explanation."""
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
        delta=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_inconsistent_final_score_raises(
        self, base_score: float, history_boost: float, delta: float
    ) -> None:
        """A final_score that differs from base + boost by more than 1e-9 must raise."""
        with pytest.raises(ValueError, match="final_score"):
            RankingExplanation(
                value="word",
                base_score=base_score,
                history_boost=history_boost,
                final_score=base_score + history_boost + delta,  # intentionally wrong
                source="test",
            )

    @given(
        base_score=_finite_score,
        boost_a=_finite_score,
        boost_b=_finite_score,
    )
    def test_merge_preserves_invariant(
        self, base_score: float, boost_a: float, boost_b: float
    ) -> None:
        """RankingExplanation.merge() must produce an explanation that still satisfies the invariant."""
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

    @given(base_score=_finite_score)
    def test_zero_boost_explanation(self, base_score: float) -> None:
        """Zero history_boost is a degenerate but valid case (pure frequency ranking)."""
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

    The engine enforces this at runtime with a ``RuntimeError``.  Here we
    verify it holds across arbitrary suggestion lists and history states,
    catching any regression where a ranker might conditionally drop or
    duplicate candidates.
    """

    @given(
        values=_unique_values,
        scores=st.lists(
            _finite_score,
            min_size=1,
            max_size=15,
        ),
        history_events=st.lists(
            st.tuples(_suggestion_value, _suggestion_value),
            max_size=20,
        ),
        boost=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        dominance_ratio=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
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
        """LearningRanker.rank() output values must equal input values as a set."""
        # Build scored suggestions, cycling scores if fewer than values
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

        # Use the first value as the query prefix (arbitrary but deterministic)
        prefix = values[0]
        ranked = ranker.rank(prefix, suggestions)

        assert {s.suggestion.value for s in ranked} == {s.suggestion.value for s in suggestions}
        assert len(ranked) == len(suggestions)

    @given(
        values=_unique_values,
        scores=st.lists(
            _finite_score,
            min_size=1,
            max_size=15,
        ),
        history_events=st.lists(
            st.tuples(_suggestion_value, _suggestion_value),
            max_size=20,
        ),
        half_life=st.floats(min_value=60.0, max_value=86400.0, allow_nan=False, allow_infinity=False),
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
        """DecayRanker.rank() output values must equal input values as a set."""
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

        prefix = values[0]
        ranked = ranker.rank(prefix, suggestions)

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
        # No history => no reordering => original order preserved
        assert [s.suggestion.value for s in ranked] == [s.suggestion.value for s in suggestions]


# ---------------------------------------------------------------------------
# Invariant 3: History prefix-index consistency
# ---------------------------------------------------------------------------

class TestHistoryPrefixIndexConsistency:
    """
    The incremental prefix index must always agree with a brute-force full scan.

    History maintains ``_by_prefix`` as an incrementally-updated dict
    alongside the main ``_entries`` list.  These tests verify the two
    representations stay consistent under arbitrary insertion sequences,
    including mixed prefixes, duplicate values, and unicode.
    """

    @given(
        events=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),   # prefix
                st.text(min_size=1, max_size=20),   # value
            ),
            min_size=1,
            max_size=100,
        ),
        query_prefix=st.text(min_size=1, max_size=10),
    )
    def test_index_matches_brute_force_for_arbitrary_prefixes(
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
        assert all(a == b for a, b in zip(indexed, brute))

    @given(
        events=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.text(min_size=1, max_size=20),
            ),
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
        """After many insertions, every prefix in the index must match a brute-force scan."""
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
    def test_count_matches_manual_total(
        self,
        value: str,
        n: int,
    ) -> None:
        """History.count(value) must equal the number of times value was recorded."""
        history = History()
        prefix = "test"
        for _ in range(n):
            history.record(prefix, value, timestamp=_FIXED_NOW)
        # Record some noise under a different value so count() is tested
        # against a non-trivial entry list.
        for _ in range(3):
            history.record(prefix, value + "_other", timestamp=_FIXED_NOW)

        assert history.count(value) == n
