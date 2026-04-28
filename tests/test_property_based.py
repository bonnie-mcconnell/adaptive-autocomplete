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


# ---------------------------------------------------------------------------
# Invariant 5: All predictors emit scores in (0, 1] for any valid input
# ---------------------------------------------------------------------------

class TestNormalisedScoreInvariant:
    """
    All predictors must emit scores in (0, 1] for any valid vocabulary
    and prefix combination.

    This invariant is the foundation of the weighted-aggregation model:
    if predictor scores are not in a common bounded space, weights become
    meaningless and dominant predictors (by raw scale, not by signal
    quality) crowd out the rest.

    Invariant: for every ScoredSuggestion s returned by any predictor,
        0 < s.score <= 1.0

    Why property-based?
        Manual tests can only cover a handful of (vocab, freq, prefix)
        combinations. Hypothesis generates degenerate cases that manual
        authorship misses: single-word vocabularies, all-equal frequencies,
        max_freq=1, high-count histories, unicode prefixes, frequencies
        that differ by many orders of magnitude.
    """

    @given(
        freq_pairs=st.lists(
            st.tuples(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=10),
                st.integers(min_value=1, max_value=1_000_000),
            ),
            min_size=1,
            max_size=30,
            unique_by=lambda x: x[0],
        ),
        prefix_len=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_frequency_predictor_scores_in_unit_interval(
        self,
        freq_pairs: list[tuple[str, int]],
        prefix_len: int,
    ) -> None:
        """FrequencyPredictor scores must lie in (0, 1] for all valid inputs."""
        from aac.domain.types import CompletionContext
        from aac.predictors.frequency import FrequencyPredictor

        frequencies = dict(freq_pairs)
        predictor = FrequencyPredictor(frequencies)

        # Generate prefixes from actual vocabulary words so we get real hits
        for word, _ in freq_pairs[:5]:
            if len(word) > prefix_len:
                prefix = word[:prefix_len]
                for s in predictor.predict(CompletionContext(prefix)):
                    assert 0.0 < s.score <= 1.0 + 1e-9, (
                        f"FrequencyPredictor score {s.score!r} out of (0, 1] "
                        f"for word={s.value!r}, prefix={prefix!r}"
                    )

    @given(
        prefix=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6),
        selections=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=10),
            min_size=1,
            max_size=20,
        ),
        counts=st.lists(
            st.integers(min_value=1, max_value=500),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_history_predictor_scores_in_unit_interval(
        self,
        prefix: str,
        selections: list[str],
        counts: list[int],
    ) -> None:
        """HistoryPredictor scores must lie in (0, 1] for all valid inputs."""
        from aac.domain.history import History
        from aac.predictors.history import HistoryPredictor

        history = History()
        for value, count in zip(selections, counts, strict=False):
            for _ in range(count):
                history.record(prefix, value, timestamp=_FIXED_NOW)

        predictor = HistoryPredictor(history)
        for s in predictor.predict(prefix):
            assert 0.0 < s.score <= 1.0 + 1e-9, (
                f"HistoryPredictor score {s.score!r} out of (0, 1] "
                f"for value={s.value!r}, prefix={prefix!r}"
            )

    @given(
        freq_pairs=st.lists(
            st.tuples(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=10),
                st.integers(min_value=1, max_value=1_000_000),
            ),
            min_size=2,
            max_size=20,
            unique_by=lambda x: x[0],
        ),
        prefix_len=st.integers(min_value=1, max_value=4),
        history_count=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_combined_engine_scores_remain_finite_and_positive(
        self,
        freq_pairs: list[tuple[str, int]],
        prefix_len: int,
        history_count: int,
    ) -> None:
        """
        The combined weighted score (FrequencyPredictor + HistoryPredictor)
        must be finite and positive for all valid inputs.

        This is a system-level invariant: the engine already enforces
        finite scores via _apply_ranking(), but testing it here at the
        predictor aggregation level catches bugs before they hit the
        engine's invariant check.
        """
        import math

        from aac.domain.history import History
        from aac.domain.types import CompletionContext, WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.predictors.frequency import FrequencyPredictor
        from aac.predictors.history import HistoryPredictor

        frequencies = dict(freq_pairs)
        history = History()

        # Record some selections so HistoryPredictor has signal
        for word, _ in freq_pairs[:3]:
            if len(word) > prefix_len:
                for _ in range(history_count):
                    history.record(word[:prefix_len], word, timestamp=_FIXED_NOW)

        engine = AutocompleteEngine(
            predictors=[
                WeightedPredictor(FrequencyPredictor(frequencies), weight=1.0),
                WeightedPredictor(HistoryPredictor(history), weight=1.5),
            ],
            history=history,
        )

        for word, _ in freq_pairs[:5]:
            if len(word) > prefix_len:
                prefix = word[:prefix_len]
                ctx = CompletionContext(prefix)
                for s in engine.predict_scored(ctx):
                    assert math.isfinite(s.score), (
                        f"Non-finite combined score {s.score!r} for {s.value!r}"
                    )
                    assert s.score > 0, (
                        f"Non-positive combined score {s.score!r} for {s.value!r}"
                    )
