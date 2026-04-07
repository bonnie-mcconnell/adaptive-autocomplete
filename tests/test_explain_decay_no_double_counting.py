"""
Regression test: DecayRanker must not double-count history boost in explanations.

Root cause of the original bug: engine.explain() passed post-ranking suggestions
(with decay-boosted scores) into each ranker's explain(). ScoreRanker then treated
the already-boosted score as base, and DecayRanker added its boost again. This
produced final_score > base_score + history_boost, breaking the explanation invariant.

Fix: engine.explain() now passes pre-ranking scores to each ranker's explain(),
so each ranker explains only its own contribution from a clean baseline.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aac.domain.history import History
from aac.domain.types import WeightedPredictor
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.score import ScoreRanker

_NOW = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_HALF_LIFE = 3600.0  # 1 hour


def _engine_with_decay_history(selection_age_seconds: float) -> AutocompleteEngine:
    history = History()
    history.record(
        "he",
        "hero",
        timestamp=_NOW - timedelta(seconds=selection_age_seconds),
    )

    return AutocompleteEngine(
        predictors=[
            WeightedPredictor(
                predictor=FrequencyPredictor(
                    {"hello": 100, "help": 80, "hero": 50}
                ),
                weight=1.0,
            )
        ],
        ranker=[
            ScoreRanker(),
            DecayRanker(
                history=history,
                decay=DecayFunction(half_life_seconds=_HALF_LIFE),
                weight=1.0,
                now=_NOW,
            ),
        ],
        history=history,
    )


def test_explanation_invariant_holds_with_decay_ranker() -> None:
    """final_score == base_score + history_boost for every suggestion."""
    engine = _engine_with_decay_history(selection_age_seconds=1800)  # 30 min ago
    explanations = engine.explain("he")

    assert explanations, "Expected at least one explanation"

    for exp in explanations:
        assert abs(exp.final_score - (exp.base_score + exp.history_boost)) < 1e-9, (
            f"Explanation invariant broken for '{exp.value}': "
            f"base={exp.base_score} + boost={exp.history_boost} "
            f"!= final={exp.final_score}"
        )


def test_decay_boost_reflects_recency() -> None:
    """A selection made recently produces a higher boost than one made long ago."""
    recent = _engine_with_decay_history(selection_age_seconds=60)     # 1 min ago
    stale = _engine_with_decay_history(selection_age_seconds=86400)   # 24 hours ago

    recent_expl = {e.value: e for e in recent.explain("he")}
    stale_expl = {e.value: e for e in stale.explain("he")}

    assert recent_expl["hero"].history_boost > stale_expl["hero"].history_boost, (
        "Recent selection should produce higher boost than stale selection"
    )


def test_no_history_means_zero_boost() -> None:
    """Suggestions with no history signal must have zero history boost."""
    history = History()  # empty

    engine = AutocompleteEngine(
        predictors=[
            WeightedPredictor(
                predictor=FrequencyPredictor({"hello": 100, "help": 80}),
                weight=1.0,
            )
        ],
        ranker=[
            ScoreRanker(),
            DecayRanker(
                history=history,
                decay=DecayFunction(half_life_seconds=_HALF_LIFE),
                weight=1.0,
                now=_NOW,
            ),
        ],
        history=history,
    )

    for exp in engine.explain("he"):
        assert exp.history_boost == 0.0, (
            f"Expected zero boost with no history, got {exp.history_boost} for '{exp.value}'"
        )


def test_unaffected_suggestions_have_zero_boost() -> None:
    """Only the selected suggestion gets a boost; others must have zero."""
    engine = _engine_with_decay_history(selection_age_seconds=0)
    explanations = {e.value: e for e in engine.explain("he")}

    assert explanations["hero"].history_boost > 0.0
    assert explanations["hello"].history_boost == 0.0
    assert explanations["help"].history_boost == 0.0