"""
adaptive-autocomplete: a composable ranking and suggestion engine.

Quick start:

    from aac.presets import create_engine

    engine = create_engine("production")
    engine.suggest("programing")        # → ['programming']  (typo recovered)
    engine.record_selection("programing", "programming")
    engine.explain("programing")        # → per-suggestion score breakdown

Or compose a custom engine directly:

    from aac import AutocompleteEngine
    from aac.predictors import FrequencyPredictor, HistoryPredictor
    from aac.domain.types import WeightedPredictor
    from aac.domain.history import History
    from aac.ranking.score import ScoreRanker

    history = History()
    engine = AutocompleteEngine(
        predictors=[
            WeightedPredictor(FrequencyPredictor({"hello": 100, "help": 80}), weight=1.0),
            WeightedPredictor(HistoryPredictor(history), weight=1.5),
        ],
        ranker=ScoreRanker(),
        history=history,
    )
"""
from __future__ import annotations

from aac.domain.history import History
from aac.domain.types import (
    CompletionContext,
    Predictor,
    ScoredSuggestion,
    Suggestion,
    WeightedPredictor,
)
from aac.engine.engine import AutocompleteEngine
from aac.presets import create_engine, get_preset
from aac.ranking.explanation import RankingExplanation

__all__ = [
    "AutocompleteEngine",
    "CompletionContext",
    "History",
    "Predictor",
    "RankingExplanation",
    "ScoredSuggestion",
    "Suggestion",
    "WeightedPredictor",
    "create_engine",
    "get_preset",
]
