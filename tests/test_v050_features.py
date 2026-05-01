"""
Tests for v0.5.0 features:

  - suggest_with_confidence(): division guard, ordering, limit
  - suggest_with_history(): counts, ordering, post-selection
  - explain() single-pass: no longer re-runs rankers
  - compare_presets(): PresetComparison structure and to_table()
  - bktree hidden from available_presets()
  - PredictorLearnsFromHistory protocol in reset_history()
  - reset_history() full propagation verified

Each test is self-contained.  No shared fixtures that obscure what is
being tested.  Assertions include failure messages so a CI failure is
immediately actionable without reading the source.
"""
from __future__ import annotations

import math

import pytest

from aac.domain.history import History
from aac.domain.types import WeightedPredictor
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.presets import (
    PresetComparison,
    available_presets,
    compare_presets,
    create_engine,
    get_preset,
)
from aac.ranking.contracts import PredictorLearnsFromHistory
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.score import ScoreRanker

_VOCAB = {
    "hello": 100, "help": 80, "hero": 50, "her": 200,
    "here": 120, "heap": 40, "world": 300, "word": 150,
    "programming": 500, "program": 400, "progress": 300,
}


# ---------------------------------------------------------------------------
# suggest_with_confidence
# ---------------------------------------------------------------------------

class TestSuggestWithConfidence:
    def _engine(self) -> AutocompleteEngine:
        return create_engine("stateless", vocabulary=_VOCAB)

    def test_top_result_confidence_is_one(self) -> None:
        e = self._engine()
        results = e.suggest_with_confidence("he")
        assert results, "expected non-empty results for 'he'"
        _, top_conf = results[0]
        assert top_conf == pytest.approx(1.0), (
            f"top confidence must be exactly 1.0, got {top_conf}"
        )

    def test_confidences_descending(self) -> None:
        e = self._engine()
        confs = [c for _, c in e.suggest_with_confidence("he")]
        assert confs == sorted(confs, reverse=True), (
            f"confidences must be non-increasing: {confs}"
        )

    def test_confidences_in_unit_interval(self) -> None:
        e = self._engine()
        for word, conf in e.suggest_with_confidence("he"):
            assert 0.0 < conf <= 1.0, (
                f"confidence for {word!r} out of (0, 1]: {conf}"
            )

    def test_order_matches_suggest(self) -> None:
        e = self._engine()
        words_conf = [w for w, _ in e.suggest_with_confidence("he")]
        words_sug = e.suggest("he")
        assert words_conf == words_sug, (
            "suggest_with_confidence order must match suggest()"
        )

    def test_limit_respected(self) -> None:
        e = self._engine()
        assert len(e.suggest_with_confidence("he", limit=3)) <= 3

    def test_limit_matches_suggest(self) -> None:
        e = self._engine()
        words_conf = [w for w, _ in e.suggest_with_confidence("he", limit=4)]
        words_sug = e.suggest("he", limit=4)
        assert words_conf == words_sug

    def test_empty_prefix_returns_empty(self) -> None:
        assert self._engine().suggest_with_confidence("") == []

    def test_no_match_returns_empty(self) -> None:
        assert self._engine().suggest_with_confidence("zzzzqqqq") == []

    def test_near_zero_top_score_does_not_explode(self) -> None:
        """
        With a tiny-frequency vocabulary and no history, top_score may be
        very small.  The old `or 1.0` guard only catches exactly zero -
        a score of 1e-12 would divide every confidence by 1e-12 and push
        them all toward infinity or 1.0 falsely.

        Use a single-word vocab with frequency=1 (produces the smallest
        valid score) and verify confidences stay in (0, 1].
        """
        e = create_engine("stateless", vocabulary={"zephyr": 1})
        results = e.suggest_with_confidence("ze")
        # May be empty if "zephyr" is not suggested for "ze" - that's fine.
        # If there are results they must be in range.
        for word, conf in results:
            assert math.isfinite(conf), f"confidence for {word!r} is not finite: {conf}"
            assert 0.0 < conf <= 1.0, f"confidence for {word!r} out of (0, 1]: {conf}"

    def test_single_candidate_confidence_is_one(self) -> None:
        """When only one suggestion is returned, its confidence must be exactly 1.0."""
        # Use a partial prefix that matches exactly one word.
        # FrequencyPredictor returns completions that *start with* the prefix,
        # so we need a prefix, not the full word.
        e = create_engine("stateless", vocabulary={"zephyr": 100})
        results = e.suggest_with_confidence("ze")
        assert results, "expected 'zephyr' to match prefix 'ze'"
        assert len(results) == 1, f"expected exactly one result, got {len(results)}"
        assert results[0][1] == pytest.approx(1.0), (
            f"single result confidence must be 1.0, got {results[0][1]}"
        )

    def test_learning_increases_confidence(self) -> None:
        """After recording selections, the boosted word's confidence must rise."""
        e = create_engine("default", vocabulary=_VOCAB)
        before = {w: c for w, c in e.suggest_with_confidence("he")}

        for _ in range(5):
            e.record_selection("he", "heap")

        after = {w: c for w, c in e.suggest_with_confidence("he")}
        heap_before = before.get("heap", 0.0)
        heap_after = after.get("heap", 0.0)
        assert heap_after > heap_before, (
            f"'heap' confidence should increase after 5 selections: "
            f"{heap_before:.4f} → {heap_after:.4f}"
        )


# ---------------------------------------------------------------------------
# suggest_with_history
# ---------------------------------------------------------------------------

class TestSuggestWithHistory:
    def _engine(self) -> AutocompleteEngine:
        return create_engine("default", vocabulary=_VOCAB)

    def test_no_history_all_counts_zero(self) -> None:
        e = self._engine()
        results = e.suggest_with_history("he", limit=10)
        assert results, "expected results for 'he'"
        for word, count in results:
            assert count == 0, (
                f"'{word}' should have count=0 with no history, got {count}"
            )

    def test_recorded_selection_shows_count(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")
        e.record_selection("he", "hello")

        results = dict(e.suggest_with_history("he"))
        assert results.get("hello") == 2, (
            f"'hello' should have count=2, got {results.get('hello')}"
        )

    def test_unselected_words_have_count_zero(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")

        results = dict(e.suggest_with_history("he"))
        for word, count in results.items():
            if word != "hello":
                assert count == 0, (
                    f"'{word}' should have count=0, got {count}"
                )

    def test_order_matches_suggest(self) -> None:
        e = self._engine()
        e.record_selection("he", "help")
        words_with_hist = [w for w, _ in e.suggest_with_history("he")]
        words_suggest = e.suggest("he")
        assert words_with_hist == words_suggest, (
            "suggest_with_history order must match suggest()"
        )

    def test_limit_respected(self) -> None:
        e = self._engine()
        assert len(e.suggest_with_history("he", limit=3)) <= 3

    def test_limit_matches_suggest(self) -> None:
        e = self._engine()
        words_hist = [w for w, _ in e.suggest_with_history("he", limit=4)]
        words_sug = e.suggest("he", limit=4)
        assert words_hist == words_sug

    def test_empty_prefix_returns_empty(self) -> None:
        e = self._engine()
        assert e.suggest_with_history("") == []

    def test_no_match_returns_empty(self) -> None:
        e = self._engine()
        assert e.suggest_with_history("zzzzqqqq") == []

    def test_multiple_prefixes_independent(self) -> None:
        """Selections for 'he' must not affect counts for 'hel'."""
        e = self._engine()
        e.record_selection("he", "hello")
        e.record_selection("he", "hello")

        results_he = dict(e.suggest_with_history("he"))
        results_hel = dict(e.suggest_with_history("hel"))

        assert results_he.get("hello", 0) == 2
        # 'hel' prefix has a different normalised prefix key - no bleed-over
        assert results_hel.get("hello", 0) == 0, (
            "counts for 'he' prefix must not bleed into 'hel' prefix results"
        )

    def test_count_reflects_multiple_different_selections(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")
        e.record_selection("he", "help")
        e.record_selection("he", "hello")

        results = dict(e.suggest_with_history("he"))
        assert results.get("hello") == 2
        assert results.get("help") == 1

    def test_reset_clears_counts(self) -> None:
        e = self._engine()
        e.record_selection("he", "hello")
        e.reset_history()

        results = dict(e.suggest_with_history("he"))
        assert all(c == 0 for c in results.values()), (
            "all counts must be 0 after reset_history()"
        )


# ---------------------------------------------------------------------------
# explain() single-pass correctness
# ---------------------------------------------------------------------------

class TestExplainSinglePass:
    """
    Verifies that explain() produces correct results via single forward pass.
    """

    def test_explain_invariant_holds_with_decay_ranker(self) -> None:
        """final_score == base_score + history_boost for every explanation."""
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("prog", "programming")
        for exp in e.explain("prog"):
            expected = exp.base_score + exp.history_boost
            assert abs(exp.final_score - expected) < 1e-9, (
                f"Invariant violated for '{exp.value}': "
                f"final={exp.final_score}, base+boost={expected}"
            )

    def test_explain_order_matches_suggest(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("he", "hello")
        explained_order = [exp.value for exp in e.explain("he")]
        suggested_order = e.suggest("he")
        assert explained_order == suggested_order, (
            "explain() order must match suggest() order"
        )

    def test_explain_boost_is_zero_without_history(self) -> None:
        """Without any recorded selections, all boosts must be zero."""
        e = create_engine("production", vocabulary=_VOCAB)
        for exp in e.explain("prog"):
            assert exp.history_boost == pytest.approx(0.0), (
                f"Expected zero boost for '{exp.value}' with no history, "
                f"got {exp.history_boost}"
            )

    def test_explain_boost_positive_after_selection(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        for _ in range(3):
            e.record_selection("he", "hello")

        exps = {exp.value: exp for exp in e.explain("he")}
        assert "hello" in exps, "'hello' must appear in explain('he') results"
        assert exps["hello"].history_boost > 0, (
            f"'hello' should have positive boost after 3 selections, "
            f"got {exps['hello'].history_boost}"
        )

    def test_explain_ranker_called_once_not_twice(self) -> None:
        """
        The new single-pass explain() must call ranker.rank() exactly N times
        (once per ranker), not 2N times as the old implementation did.

        We verify this by counting rank() calls via a spy on a DecayRanker.
        """
        history = History()
        decay = DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=3600),
            weight=1.5,
        )

        rank_call_count = 0
        original_rank = decay.rank

        def counting_rank(prefix, suggestions):
            nonlocal rank_call_count
            rank_call_count += 1
            return original_rank(prefix, suggestions)

        decay.rank = counting_rank  # type: ignore[method-assign]

        engine = AutocompleteEngine(
            predictors=[
                WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0),
            ],
            ranker=[ScoreRanker(), decay],
            history=history,
        )

        engine.explain("he")

        # With the old double-pipeline: ScoreRanker.rank called 2×, DecayRanker.rank 2×
        # With the new single-pass:   ScoreRanker.rank called 1×, DecayRanker.rank 1×
        assert rank_call_count == 1, (
            f"DecayRanker.rank() should be called exactly once per explain(), "
            f"got {rank_call_count} calls"
        )

    def test_explain_as_dicts_schema(self) -> None:
        e = create_engine("production", vocabulary=_VOCAB)
        e.record_selection("prog", "programming")
        dicts = e.explain_as_dicts("prog")
        assert dicts, "expected non-empty explain_as_dicts results"
        required_keys = {"value", "base_score", "history_boost", "final_score",
                         "sources", "base_components", "history_components"}
        for d in dicts:
            assert required_keys.issubset(d.keys()), (
                f"explain_as_dicts row missing keys: {required_keys - d.keys()}"
            )


# ---------------------------------------------------------------------------
# compare_presets
# ---------------------------------------------------------------------------

class TestComparePresets:
    def test_returns_preset_comparison(self) -> None:
        cmp = compare_presets("he", ["stateless", "default"], vocabulary=_VOCAB)
        assert isinstance(cmp, PresetComparison)

    def test_presets_list_correct(self) -> None:
        cmp = compare_presets("he", ["stateless", "default"], vocabulary=_VOCAB)
        assert cmp.presets == ["stateless", "default"]

    def test_text_preserved(self) -> None:
        cmp = compare_presets("hello", ["stateless"], vocabulary=_VOCAB)
        assert cmp.text == "hello"

    def test_rows_contain_all_preset_keys(self) -> None:
        cmp = compare_presets("he", ["stateless", "default"], vocabulary=_VOCAB)
        for row in cmp.rows:
            assert set(row["ranks"].keys()) == {"stateless", "default"}  # type: ignore[union-attr]
            assert set(row["finals"].keys()) == {"stateless", "default"}  # type: ignore[union-attr]

    def test_stateless_has_no_boosts(self) -> None:
        """Stateless preset has no learning, all boosts must be zero."""
        cmp = compare_presets("he", ["stateless"], vocabulary=_VOCAB)
        for row in cmp.rows:
            boost = row["boosts"]["stateless"]  # type: ignore[index]
            if boost is not None:
                assert boost == pytest.approx(0.0), (
                    f"stateless should have zero boost for '{row['value']}', got {boost}"
                )

    def test_none_for_missing_suggestions(self) -> None:
        """A suggestion absent from a preset appears as None in that preset's columns."""
        # stateless has no typo recovery; production does.
        # "recieve" (typo) should appear in production but not stateless.
        cmp = compare_presets("recieve", ["stateless", "production"])
        # Find a word that production returns but stateless doesn't.
        for row in cmp.rows:
            if row["ranks"]["production"] is not None and row["ranks"]["stateless"] is None:  # type: ignore[index]
                # Verify the stateless columns are all None.
                assert row["base_scores"]["stateless"] is None  # type: ignore[index]
                assert row["finals"]["stateless"] is None  # type: ignore[index]
                break

    def test_ranks_start_at_one(self) -> None:
        cmp = compare_presets("he", ["stateless"], vocabulary=_VOCAB)
        non_none_ranks = [
            row["ranks"]["stateless"]  # type: ignore[index]
            for row in cmp.rows
            if row["ranks"]["stateless"] is not None  # type: ignore[index]
        ]
        assert min(non_none_ranks) == 1, "ranks must start at 1"

    def test_ranks_are_contiguous(self) -> None:
        cmp = compare_presets("he", ["stateless"], vocabulary=_VOCAB)
        non_none = sorted(
            row["ranks"]["stateless"]  # type: ignore[index]
            for row in cmp.rows
            if row["ranks"]["stateless"] is not None  # type: ignore[index]
        )
        assert non_none == list(range(1, len(non_none) + 1)), (
            f"ranks must be contiguous: {non_none}"
        )

    def test_limit_respected_per_preset(self) -> None:
        limit = 3
        cmp = compare_presets("he", ["stateless"], vocabulary=_VOCAB, limit=limit)
        non_none = [
            r for r in cmp.rows if r["ranks"]["stateless"] is not None  # type: ignore[index]
        ]
        assert len(non_none) <= limit

    def test_to_table_returns_string(self) -> None:
        cmp = compare_presets("he", ["stateless", "default"], vocabulary=_VOCAB)
        table = cmp.to_table()
        assert isinstance(table, str)
        assert len(table) > 0

    def test_to_table_contains_preset_names(self) -> None:
        cmp = compare_presets("he", ["stateless", "default"], vocabulary=_VOCAB)
        table = cmp.to_table()
        assert "stateless" in table
        assert "default" in table

    def test_to_table_limit_truncates_rows(self) -> None:
        cmp = compare_presets("he", ["stateless"], vocabulary=_VOCAB)
        full = cmp.to_table()
        limited = cmp.to_table(limit=2)
        assert len(limited.splitlines()) <= len(full.splitlines())

    def test_repr_shows_preset_and_row_count(self) -> None:
        cmp = compare_presets("he", ["stateless"], vocabulary=_VOCAB)
        r = repr(cmp)
        assert "stateless" in r
        assert "rows=" in r

    def test_default_presets_uses_all_public(self) -> None:
        """compare_presets() with no presets arg uses available_presets()."""
        # This is a small-vocab fast test - just verify structure, not content
        cmp = compare_presets("he", vocabulary=_VOCAB, limit=3)
        assert set(cmp.presets) == set(available_presets())

    def test_typo_recovery_visible_in_comparison(self) -> None:
        """production recovers typos; stateless does not. Table shows this."""
        cmp = compare_presets("recieve", ["stateless", "production"], limit=10)
        # At least one word should appear in production but not stateless
        recovery_visible = any(
            row["ranks"]["production"] is not None  # type: ignore[index]
            and row["ranks"]["stateless"] is None  # type: ignore[index]
            for row in cmp.rows
        )
        assert recovery_visible, (
            "compare_presets should show typo-recovered words absent from stateless"
        )

    def test_history_shared_across_presets(self) -> None:
        """A shared History is passed to all preset engines."""
        history = History()
        history.record("he", "hello")
        history.record("he", "hello")

        cmp = compare_presets(
            "he", ["default", "recency"],
            vocabulary=_VOCAB,
            history=history,
            limit=10,
        )
        # Both presets should rank "hello" highly with 2 recorded selections.
        for name in ["default", "recency"]:
            hello_row = next(
                (r for r in cmp.rows if r["value"] == "hello"), None
            )
            assert hello_row is not None, f"'hello' should appear in '{name}' results"
            rank = hello_row["ranks"][name]  # type: ignore[index]
            assert rank is not None and rank <= 3, (
                f"'hello' should rank in top 3 for '{name}' with 2 selections, got rank {rank}"
            )


# ---------------------------------------------------------------------------
# bktree hidden from available_presets()
# ---------------------------------------------------------------------------

class TestBktreeHidden:
    def test_bktree_not_in_available_presets(self) -> None:
        assert "bktree" not in available_presets(), (
            "bktree must not appear in available_presets() - it degrades to O(n) "
            "at 48k+ words and must not be accidentally deployed"
        )

    def test_bktree_still_accessible_via_create_engine(self) -> None:
        """bktree is hidden, not removed - create_engine('bktree') must still work."""
        e = create_engine("bktree", vocabulary=_VOCAB)
        assert e.suggest("he"), "bktree engine must return results"

    def test_bktree_still_accessible_via_get_preset(self) -> None:
        preset = get_preset("bktree")
        assert preset.name == "bktree"

    def test_available_presets_contains_expected_public(self) -> None:
        expected = {"stateless", "default", "recency", "production", "robust"}
        actual = set(available_presets())
        assert expected == actual, (
            f"available_presets() mismatch. expected={expected}, got={actual}"
        )


# ---------------------------------------------------------------------------
# PredictorLearnsFromHistory protocol
# ---------------------------------------------------------------------------

class TestPredictorLearnsFromHistoryProtocol:
    def test_history_predictor_satisfies_protocol(self) -> None:
        history = History()
        predictor = HistoryPredictor(history)
        assert isinstance(predictor, PredictorLearnsFromHistory), (
            "HistoryPredictor must satisfy PredictorLearnsFromHistory"
        )

    def test_frequency_predictor_does_not_satisfy_protocol(self) -> None:
        predictor = FrequencyPredictor(_VOCAB)
        assert not isinstance(predictor, PredictorLearnsFromHistory), (
            "FrequencyPredictor must not satisfy PredictorLearnsFromHistory - "
            "it has no history attribute"
        )

    def test_reset_history_uses_protocol_not_hasattr(self) -> None:
        """
        A predictor with no 'history' attribute must not be updated by reset_history().

        runtime_checkable Protocol checks for attribute *presence*, not type.
        The important protection is against predictors that have no 'history'
        attribute at all - they are not updated.  A predictor with a wrongly-typed
        'history' attribute would still be updated (Python protocol limitation),
        but such a predictor would be incorrectly implemented regardless.

        This test verifies the case that matters: a predictor without any
        'history' attribute is left untouched.
        """

        class _StatelessPredictor:
            """Predictor with no history attribute - must not be touched by reset."""
            name = "stateless_fake"
            _marker = "original"

            def predict(self, ctx):
                return []

        history = History()
        fake = _StatelessPredictor()

        engine = AutocompleteEngine(
            predictors=[
                WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0),
                WeightedPredictor(fake, weight=1.0),  # type: ignore[arg-type]
            ],
            ranker=ScoreRanker(),
            history=history,
        )

        engine.reset_history()

        # Predictor with no 'history' attribute must not have one added.
        assert not hasattr(fake, "history"), (
            "reset_history() must not inject a 'history' attribute into a predictor "
            "that does not implement PredictorLearnsFromHistory"
        )
        assert fake._marker == "original", "predictor state must be unchanged"

    def test_reset_history_updates_history_predictor_via_protocol(self) -> None:
        """reset_history() must update HistoryPredictor via the protocol."""
        history = History()
        history.record("he", "hello")
        hist_predictor = HistoryPredictor(history)

        engine = AutocompleteEngine(
            predictors=[
                WeightedPredictor(FrequencyPredictor(_VOCAB), weight=1.0),
                WeightedPredictor(hist_predictor, weight=1.5),
            ],
            ranker=ScoreRanker(),
            history=history,
        )

        engine.reset_history()

        # After reset, the predictor must read from the new empty history.
        assert hist_predictor.history is engine.history, (
            "HistoryPredictor.history must point to the new History after reset"
        )
        assert len(list(hist_predictor.history.entries())) == 0, (
            "HistoryPredictor must see the new empty history after reset"
        )
