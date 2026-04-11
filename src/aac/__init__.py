"""
adaptive-autocomplete: a composable ranking and suggestion engine.

Quick start:

    from aac.presets import get_preset
    from aac.domain.history import History

    engine = get_preset("production").build(History(), None)
    engine.suggest("helo")           # typo-tolerant completions
    engine.record_selection("helo", "hello")
    engine.explain("helo")           # per-suggestion score breakdown

Or compose directly:

    from aac.engine import AutocompleteEngine
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

from aac.engine.engine import AutocompleteEngine

__all__ = ["AutocompleteEngine"]
