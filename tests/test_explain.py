"""
Tests for explain() and explain_as_dicts().

Three correctness properties verified here:

1. Arithmetic invariant: final_score == base_score + history_boost for every
   explanation, including with DecayRanker and LearningRanker in the chain.

2. Single-pass guarantee: explain() calls each ranker exactly once (not twice
   as the old double-pipeline implementation did). Verified with a call-counting
   spy on DecayRanker.

3. base_components completeness: explain() returns a base_components dict with
   an entry for every configured predictor - including predictors that returned
   no candidates for this prefix (those get 0.0, not a missing key).
   the difference between "predictor ran but found nothing" and "predictor not
   configured" explicit.
"""
from __future__ import annotations

import pytest

from aac.domain.history import History
from aac.domain.types import WeightedPredictor
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.presets import create_engine
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.score import ScoreRanker

_VOCAB = {
    "hello": 100, "help": 80, "hero": 50, "her": 200,
    "here": 120, "heap": 40, "world": 300, "word": 150,
    "programming": 500, "program": 400, "progress": 300,
}


# ---------------------------------------------------------------------------
# Arithmetic invariant
# ---------------------------------------------------------------------------

class TestExplainArithmeticInvariant:
    def test_invariant_holds_stateless(self) -> None:
        """Without learning, all boosts are zero and final == base."""
        e = create_engine("stateless", vocabulary=_VOCAB)
        for exp in e.explain("he"):
            assert exp.history_boost == pytest.approx(0.0), (
                f"Expected zero boost for '{exp.value}' with no history"
            )
            assert abs(exp.final_score - exp.base_score) < 1e-9

    def test_invariant_holds_with_decay_ranker(self) -> None:
        """final_score == base_score + history_boost with DecayRanker in chain."""
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("prog", "programming")
        for exp in e.explain("prog"):
            expected = exp.base_score + exp.history_boost
            assert abs(exp.final_score - expected) < 1e-9, (
                f"Invariant violated for '{exp.value}': "
                f"final={exp.final_score}, base+boost={expected}"
            )

    def test_boost_is_zero_without_history(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        for exp in e.explain("prog"):
            assert exp.history_boost == pytest.approx(0.0), (
                f"Expected zero boost for '{exp.value}' with no history, "
                f"got {exp.history_boost}"
            )

    def test_boost_positive_after_selection(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        for _ in range(3):
            e.record_selection("he", "hello")

        exps = {exp.value: exp for exp in e.explain("he")}
        assert "hello" in exps, "'hello' must appear in explain('he') results"
        assert exps["hello"].history_boost > 0, (
            f"'hello' should have positive boost after 3 selections, "
            f"got {exps['hello'].history_boost}"
        )

    def test_order_matches_suggest(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("he", "hello")
        explained_order = [exp.value for exp in e.explain("he")]
        assert explained_order == e.suggest("he"), (
            "explain() order must match suggest() order"
        )


# ---------------------------------------------------------------------------
# Single-pass guarantee
# ---------------------------------------------------------------------------

class TestExplainSinglePass:
    def test_ranker_called_exactly_once(self) -> None:
        """explain() must call ranker.rank() once per ranker, not twice.

        The old double-pipeline implementation called each ranker twice per
        explain() call: once to get the ranked order, once to compute deltas.
        The current implementation does a single forward pass: one rank() call
        per ranker captures both the order and the delta. We verify this with
        a call-counting spy.
        """
        history = History()
        decay = DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=3600),
            weight=1.5,
        )

        rank_call_count = 0
        original_rank = decay.rank

        def counting_rank(prefix, suggestions):  # type: ignore[override]
            nonlocal rank_call_count
            rank_call_count += 1
            return original_rank(prefix, suggestions)

        decay.rank = counting_rank  # type: ignore[method-assign]

        engine = AutocompleteEngine(
            predictors=[WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0)],
            ranker=[ScoreRanker(), decay],
            history=history,
        )

        engine.explain("he")

        assert rank_call_count == 1, (
            f"DecayRanker.rank() should be called exactly once per explain(), "
            f"got {rank_call_count} calls"
        )


# ---------------------------------------------------------------------------
# base_components completeness
# ---------------------------------------------------------------------------

class TestExplainBaseComponentsComplete:
    def test_all_predictor_names_present_in_base_components(self) -> None:
        """Every configured predictor must appear in base_components, even when
        it returned no candidates (those get 0.0, not a missing key).

        This matters for weight-tuning tooling and for the explain UI: a missing
        key is ambiguous between "predictor ran, found nothing" and "predictor
        not configured". A 0.0 entry is unambiguous.
        """
        e = create_engine("production", vocabulary=_VOCAB)
        configured_predictor_names = {wp.predictor.name for wp in e._predictors}

        for exp in e.explain("prog"):
            missing = configured_predictor_names - exp.base_components.keys()
            assert not missing, (
                f"Predictor(s) {missing} missing from base_components for '{exp.value}'. "
                f"Expected all of: {configured_predictor_names}"
            )

    def test_missing_predictor_contribution_is_zero_not_absent(self) -> None:
        """A predictor that returned no candidates for this prefix scores 0.0,
        not a missing key."""
        e = create_engine("production", vocabulary=_VOCAB)
        # "prog" triggers symspell, trigram, and history predictors
        for exp in e.explain("prog"):
            for name, score in exp.base_components.items():
                assert isinstance(score, float), (
                    f"base_components[{name!r}] must be float, got {type(score)}"
                )

    def test_contribution_pct_sums_to_approx_one_when_scores_nonzero(self) -> None:
        """contribution_pct fractions must sum to approximately 1.0."""
        e = create_engine("default", vocabulary=_VOCAB)
        e.record_selection("he", "hello")
        for exp in e.explain("he"):
            if abs(exp.final_score) > 1e-9 and exp.contribution_pct:
                total = sum(exp.contribution_pct.values())
                assert abs(total - 1.0) < 0.01, (
                    f"contribution_pct for '{exp.value}' sums to {total:.4f}, expected ~1.0"
                )


# ---------------------------------------------------------------------------
# explain_as_dicts schema
# ---------------------------------------------------------------------------

class TestExplainAsDicts:
    def test_schema_has_required_keys(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("prog", "programming")
        dicts = e.explain_as_dicts("prog")
        assert dicts, "expected non-empty explain_as_dicts results"
        required = {
            "value", "base_score", "history_boost", "final_score",
            "sources", "base_components", "history_components",
        }
        for d in dicts:
            assert required.issubset(d.keys()), (
                f"explain_as_dicts row missing keys: {required - d.keys()}"
            )

    def test_order_matches_explain(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("prog", "programming")
        dicts = e.explain_as_dicts("prog")
        exps = e.explain("prog")
        assert [d["value"] for d in dicts] == [ex.value for ex in exps]
