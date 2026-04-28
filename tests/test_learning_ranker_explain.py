"""
Tests for LearningRanker ranking and explanation behaviour.

Covers: ordering with/without history, boost calculation, explain invariants,
shared history contract with engine, and mutation safety.
"""
from __future__ import annotations

from aac.domain.history import History
from aac.domain.types import ScoredSuggestion, Suggestion
from aac.engine.engine import AutocompleteEngine
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.ranking.learning import LearningRanker


def _suggestions(*values: str, score: float = 0.0) -> list[ScoredSuggestion]:
    return [ScoredSuggestion(Suggestion(v), score) for v in values]


# ------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------

def test_ranker_preserves_order_without_history() -> None:
    ranker = LearningRanker(History())
    ranked = ranker.rank("x", _suggestions("a", "b"))
    assert [s.value for s in ranked] == ["a", "b"]


def test_ranker_boosts_previously_selected_value() -> None:
    history = History()
    history.record("he", "hello")
    history.record("he", "hello")
    ranker = LearningRanker(history, boost=1.0)
    suggestions = [
        ScoredSuggestion(Suggestion("hello"), 1.0),
        ScoredSuggestion(Suggestion("help"), 1.0),
    ]
    ranked = ranker.rank("he", suggestions)
    assert ranked[0].value == "hello"


def test_ranker_empty_suggestions_returns_empty() -> None:
    assert LearningRanker(History()).rank("he", []) == []


# ------------------------------------------------------------------
# Explanation
# ------------------------------------------------------------------

def test_explain_zero_boost_without_history() -> None:
    ranker = LearningRanker(History())
    for exp in ranker.explain("he", _suggestions("a", "b", score=1.0)):
        assert exp.history_boost == 0.0
        assert exp.final_score == exp.base_score


def test_explain_reflects_selection_count() -> None:
    history = History()
    history.record("he", "help")
    history.record("he", "help")
    ranker = LearningRanker(history, boost=1.0)
    by_value = {e.value: e for e in ranker.explain("he", _suggestions("hello", "help"))}
    assert by_value["help"].history_boost == 2.0
    assert by_value["hello"].history_boost == 0.0


def test_explain_final_score_invariant() -> None:
    """final_score == base_score + history_boost for every suggestion."""
    history = History()
    history.record("he", "help")
    ranker = LearningRanker(history, boost=1.5)
    for exp in ranker.explain("he", _suggestions("hello", "help", score=5.0)):
        assert abs(exp.final_score - (exp.base_score + exp.history_boost)) < 1e-9


def test_explain_does_not_mutate_history() -> None:
    history = History()
    LearningRanker(history).explain("he", _suggestions("hello"))
    assert len(history.entries()) == 0


def test_explain_as_dicts_schema() -> None:
    """
    LearningRanker.explain_as_dicts() reflects the boost-ranker contract:
    base_score=0.0 (the ranker contributes boost, not base),
    history_boost=0.0 (no history for this prefix), final_score=0.0.
    Use engine.explain_as_dicts() for full base+boost breakdowns.
    """
    ranker = LearningRanker(History())
    rows = ranker.explain_as_dicts("he", _suggestions("hello", score=0.5))
    assert rows == [{"value": "hello", "base_score": 0.0, "history_boost": 0.0, "final_score": 0.0}]


# ------------------------------------------------------------------
# Shared history contract
# ------------------------------------------------------------------

def test_engine_and_learning_ranker_share_history() -> None:
    """record_selection on the engine must be visible to the ranker's history."""
    history = History()
    ranker = LearningRanker(history)
    engine = AutocompleteEngine(
        predictors=[StaticPrefixPredictor(["hello", "help"])],
        ranker=ranker,
    )
    engine.record_selection("he", "help")
    assert history.count("help") == 1


def test_history_learning_boosts_selected_value() -> None:
    """End-to-end: recording via history must improve suggestion rank.

    Uses a controlled vocabulary so the test is independent of the bundled
    word list and its frequency ordering. With the global 48k vocabulary,
    low-frequency words like 'hero' may not appear in the top-N results,
    making the assertion vacuously impossible.
    """
    from aac.domain.types import WeightedPredictor
    from aac.predictors.frequency import FrequencyPredictor
    from aac.predictors.history import HistoryPredictor
    from aac.ranking.score import ScoreRanker

    # Controlled vocabulary: hero and help are close in frequency.
    # 3 selections × weight 1.5 = +4.5 boost → hero (50+4.5=54.5) > help (52).
    vocab = {"her": 200, "hello": 100, "help": 52, "hero": 50}
    history = History()

    engine = AutocompleteEngine(
        predictors=[
            WeightedPredictor(predictor=FrequencyPredictor(vocab), weight=1.0),
            WeightedPredictor(predictor=HistoryPredictor(history), weight=1.5),
        ],
        ranker=ScoreRanker(),
        history=history,
    )

    before_values = engine.suggest("he")
    assert "hero" in before_values, "hero must appear in results before recording"
    before_pos = before_values.index("hero")

    # 3 selections × weight 1.5 = +4.5 → hero rises above help
    history.record("he", "hero")
    history.record("he", "hero")
    history.record("he", "hero")

    after_values = engine.suggest("he")
    assert "hero" in after_values
    assert after_values.index("hero") < before_pos, (
        f"Expected hero to rise from position {before_pos}, "
        f"got position {after_values.index('hero')}"
    )