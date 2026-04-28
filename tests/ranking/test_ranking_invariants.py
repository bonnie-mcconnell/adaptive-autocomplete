from __future__ import annotations

import math

from aac.domain.types import CompletionContext, ScoredSuggestion, Suggestion
from aac.engine.engine import AutocompleteEngine
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.ranking.score import score_and_rank


def _scores(results: list[ScoredSuggestion]) -> list[float]:
    return [r.score for r in results]


def test_ranking_is_deterministic() -> None:
    engine = AutocompleteEngine(
        predictors=[
            StaticPrefixPredictor(vocabulary=["hello", "help", "helium"])
        ]
    )
    ctx = CompletionContext(text="he")
    first = engine.predict_scored(ctx)
    second = engine.predict_scored(ctx)
    assert first == second


def test_scores_are_finite_and_non_negative() -> None:
    results = score_and_rank(
        [
            ScoredSuggestion(suggestion=Suggestion("test"), score=1.0, explanation=None),
            ScoredSuggestion(suggestion=Suggestion("testing"), score=0.5, explanation=None),
        ]
    )
    for r in results:
        assert isinstance(r.score, float)
        assert math.isfinite(r.score)
        assert r.score >= 0.0


def test_results_are_sorted_by_score_descending() -> None:
    results = score_and_rank(
        [
            ScoredSuggestion(Suggestion("a"), 0.2, None),
            ScoredSuggestion(Suggestion("b"), 0.9, None),
            ScoredSuggestion(Suggestion("c"), 0.5, None),
        ]
    )
    scores = _scores(results)
    assert scores == sorted(scores, reverse=True)


def test_ranking_is_idempotent() -> None:
    initial = [
        ScoredSuggestion(Suggestion("a"), 0.8, None),
        ScoredSuggestion(Suggestion("b"), 0.4, None),
    ]
    once = score_and_rank(initial)
    twice = score_and_rank(once)
    assert once == twice


def test_predictors_do_not_mutate_each_other() -> None:
    engine = AutocompleteEngine(
        predictors=[
            StaticPrefixPredictor(vocabulary=["hello", "help"]),
            StaticPrefixPredictor(vocabulary=["helium"]),
        ]
    )
    ctx = CompletionContext(text="he")
    results = engine.predict_scored(ctx)
    values = [r.suggestion.value for r in results]
    assert "hello" in values
    assert "help" in values
    assert "helium" in values

# ------------------------------------------------------------------
# LearningRanker returns boosted scores
# ------------------------------------------------------------------

def test_learning_ranker_rank_returns_boosted_scores() -> None:
    """rank() must return ScoredSuggestion objects with the boosted score."""
    from aac.domain.history import History
    from aac.domain.types import ScoredSuggestion, Suggestion
    from aac.ranking.learning import LearningRanker

    history = History()
    history.record("he", "hello")
    history.record("he", "hello")

    ranker = LearningRanker(history, boost=1.0)
    suggestions = [
        ScoredSuggestion(Suggestion("hello"), score=10.0),
        ScoredSuggestion(Suggestion("help"), score=8.0),
    ]

    ranked = ranker.rank("he", suggestions)
    hello_result = next(s for s in ranked if s.suggestion.value == "hello")
    help_result = next(s for s in ranked if s.suggestion.value == "help")

    # hello: boost = min(2*1.0, 1.0*10.0) = 2.0 -> score = 12.0
    assert hello_result.score == 12.0
    assert help_result.score == 8.0


# ------------------------------------------------------------------
# LearningRanker.explain() uses pre-boost scores (not post-boost)
# ------------------------------------------------------------------

def test_learning_ranker_explain_uses_pre_boost_base_score() -> None:
    """
    LearningRanker.explain() is a boost-only method: it returns base_score=0
    and history_boost equal to the boost it applied.

    The full pre-ranking base score is surfaced by engine.explain(), which
    derives it directly from _score() and is immune to ranker composition.
    This test verifies:
      1. ranker.explain() produces boost=5.0 (capped at 0.5 * pre-boost base)
      2. ranker.explain() base_score=0.0 (boost ranker, not a base ranker)
      3. engine.explain() surfaces base_score=10.0 (true predictor base)
    """
    from aac.domain.history import History
    from aac.domain.types import ScoredSuggestion, Suggestion
    from aac.ranking.learning import LearningRanker

    history = History()
    history.record("he", "hello")
    history.record("he", "hello")

    # dominance_ratio=0.5: boost capped at 0.5 * base_score
    ranker = LearningRanker(history, boost=5.0, dominance_ratio=0.5)

    original = [
        ScoredSuggestion(Suggestion("hello"), score=10.0),
        ScoredSuggestion(Suggestion("help"), score=8.0),
    ]

    # rank() computes boost against pre-boost base_score=10.0:
    # raw_boost = 2 * 5.0 = 10.0; capped at 0.5 * 10.0 = 5.0
    ranked = ranker.rank("he", original)
    hello_ranked = next(s for s in ranked if s.suggestion.value == "hello")
    assert hello_ranked.score == 15.0

    # ranker.explain() is a boost-only method: base=0, boost=delta
    exps = ranker.explain("he", original)
    hello_exp = next(e for e in exps if e.value == "hello")

    assert hello_exp.base_score == 0.0, (
        "LearningRanker is a boost ranker; base_score must be 0 "
        f"(full base is in engine.explain()), got {hello_exp.base_score}"
    )
    assert hello_exp.history_boost == 5.0, (
        f"boost must be capped at 0.5*10.0=5.0, got {hello_exp.history_boost}"
    )
    assert hello_exp.final_score == 5.0, (
        f"ranker explain final = boost only = 5.0, got {hello_exp.final_score}"
    )


def test_engine_explain_shows_correct_base_score_with_learning_ranker() -> None:
    """
    engine.explain() must show the true pre-ranking predictor score as base_score,
    regardless of which rankers are composed.
    """
    from aac.domain.history import History
    from aac.engine.engine import AutocompleteEngine
    from aac.predictors.static_prefix import StaticPrefixPredictor
    from aac.ranking.learning import LearningRanker

    history = History()
    history.record("he", "help")

    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help", "hero"])],
        ranker=LearningRanker(history, boost=1.0),
        history=history,
    )

    exps = {e.value: e for e in engine.explain("he")}

    # help has a history boost; its base_score must be the predictor score (1.0),
    # not 0.0 (what LearningRanker.explain() would return in isolation).
    assert exps["help"].base_score == 1.0, (
        f"engine.explain() base_score must be predictor score 1.0, got {exps['help'].base_score}"
    )
    assert exps["help"].history_boost == 1.0, (
        f"history_boost must equal the LearningRanker boost, got {exps['help'].history_boost}"
    )
    assert exps["help"].final_score == 2.0

    # hello has no history; base_score is the predictor score, boost=0
    assert exps["hello"].base_score == 1.0
    assert exps["hello"].history_boost == 0.0


# ---------------------------------------------------------------------------
# DecayRanker stability and trace
# ---------------------------------------------------------------------------

class TestDecayRankerStabilityAndTrace:
    """
    DecayRanker must produce stable output and record trace contributions.
    """

    def test_equal_score_order_is_stable(self) -> None:
        """Equal-score suggestions must preserve original insertion order."""
        from datetime import datetime, timezone

        from aac.domain.history import History
        from aac.domain.types import ScoredSuggestion, Suggestion
        from aac.ranking.decay import DecayFunction, DecayRanker

        history = History()  # no history → all boosts are 0 → all scores equal
        ranker = DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=3600),
            weight=1.0,
            now=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        suggestions = [
            ScoredSuggestion(suggestion=Suggestion(value=v), score=1.0)
            for v in ["alpha", "beta", "gamma", "delta"]
        ]

        result = ranker.rank("prefix", suggestions)
        assert [s.value for s in result] == ["alpha", "beta", "gamma", "delta"]

    def test_trace_entry_added_when_boost_nonzero(self) -> None:
        """DecayRanker must add a trace entry when it applies a non-zero boost."""
        from datetime import datetime, timezone

        from aac.domain.history import History
        from aac.domain.types import ScoredSuggestion, Suggestion
        from aac.ranking.decay import DecayFunction, DecayRanker

        now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        history = History()
        history.record("he", "hero", timestamp=now)

        ranker = DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=3600),
            weight=2.0,
            now=now,
        )

        suggestions = [
            ScoredSuggestion(suggestion=Suggestion(value="hero"), score=1.0),
        ]

        result = ranker.rank("he", suggestions)
        assert len(result) == 1
        trace = result[0].trace
        assert any("DecayRanker" in entry for entry in trace), (
            f"Expected DecayRanker trace entry, got: {trace}"
        )

    def test_trace_not_added_when_boost_zero(self) -> None:
        """DecayRanker must not add a trace entry when boost is zero."""
        from datetime import datetime, timezone

        from aac.domain.history import History
        from aac.domain.types import ScoredSuggestion, Suggestion
        from aac.ranking.decay import DecayFunction, DecayRanker

        history = History()
        ranker = DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=3600),
            now=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        suggestions = [
            ScoredSuggestion(suggestion=Suggestion(value="hello"), score=1.0),
        ]

        result = ranker.rank("he", suggestions)
        assert len(result[0].trace) == 0


def test_score_ranker_explain_handles_missing_explanation() -> None:
    """
    ScoreRanker.explain() must not crash when ScoredSuggestion.explanation is None.
    This branch fires when suggestions are constructed manually without a predictor
    explanation — for example, in tests or custom predictor integrations.
    """
    from aac.domain.types import ScoredSuggestion, Suggestion
    from aac.ranking.score import ScoreRanker

    s = ScoredSuggestion(Suggestion("hello"), score=0.8)
    assert s.explanation is None  # confirm the setup

    exps = ScoreRanker().explain("he", [s])
    assert len(exps) == 1
    assert exps[0].value == "hello"
    assert exps[0].base_score == 0.8
    assert exps[0].source == "score"  # fallback label when explanation is absent
