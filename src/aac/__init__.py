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

from aac.domain.contextual_history import ContextualHistory
from aac.domain.history import History
from aac.domain.thread_safe_history import ThreadSafeHistory
from aac.domain.types import (
    CompletionContext,
    Predictor,
    ScoredSuggestion,
    WeightedPredictor,
)
from aac.engine.config import EngineConfig
from aac.engine.engine import AutocompleteEngine
from aac.predictors.adaptive_symspell import AdaptiveSymSpellPredictor
from aac.presets import PresetComparison, compare_presets, create_engine, get_preset
from aac.ranking.explanation import RankingExplanation
from aac.storage.json_store import JsonHistoryStore
from aac.vocabulary import (
    vocabulary_from_file,
    vocabulary_from_text,
    vocabulary_from_wordlist,
)

__all__ = [
    "AdaptiveSymSpellPredictor",
    "AutocompleteEngine",
    "CompletionContext",
    "ContextualHistory",
    "EngineConfig",
    "History",
    "JsonHistoryStore",
    "Predictor",
    "PresetComparison",
    "RankingExplanation",
    "ScoredSuggestion",
    "ThreadSafeHistory",
    "WeightedPredictor",
    "compare_presets",
    "create_engine",
    "get_preset",
    "vocabulary_from_file",
    "vocabulary_from_text",
    "vocabulary_from_wordlist",
]
