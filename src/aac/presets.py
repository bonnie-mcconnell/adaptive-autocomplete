from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from aac.data import load_english_frequencies
from aac.domain.history import History
from aac.domain.types import WeightedPredictor
from aac.engine.engine import AutocompleteEngine
from aac.predictors.edit_distance import EditDistancePredictor
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.predictors.trigram import TrigramPredictor
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.score import ScoreRanker

# ---------------------------------------------------------------------
# Preset definition
# ---------------------------------------------------------------------

PresetBuilder: TypeAlias = Callable[
    [History | None, Mapping[str, int] | None],
    AutocompleteEngine,
]


@dataclass(frozen=True)
class EnginePreset:
    """
    Named, validated engine composition.

    A preset represents intent, not configuration detail.
    Metadata fields drive describe_presets() output - update them
    here when adding or changing a preset, not in describe_presets().
    """

    name: str
    description: str
    build: PresetBuilder
    predictors: tuple[str, ...]
    ranking: str
    learning: str


# ---------------------------------------------------------------------
# Default vocabulary
#
# Loaded once at import time and shared across all preset builders.
# Using the cached load function means the JSON file is read exactly
# once regardless of how many presets are constructed.
# ---------------------------------------------------------------------

_DEFAULT_VOCABULARY: Mapping[str, int] = load_english_frequencies()


# ---------------------------------------------------------------------
# Preset builders
# ---------------------------------------------------------------------


def _stateless_engine(
    _: History | None,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """Pure frequency ranking. No learning, no history, fully reproducible."""
    frequencies = vocabulary or _DEFAULT_VOCABULARY

    predictors = [
        WeightedPredictor(
            predictor=FrequencyPredictor(frequencies=frequencies),
            weight=1.0,
        ),
    ]

    return AutocompleteEngine(
        predictors=predictors,
        ranker=ScoreRanker(),
        history=History(),
    )


def _default_engine(
    history: History | None,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """
    Frequency + history learning at the prediction layer.

    Learning happens by weighting HistoryPredictor alongside
    FrequencyPredictor. No ranking-layer boost; history influence
    appears in the base score rather than as a separate column in
    explain() output.
    """
    history = history or History()
    frequencies = vocabulary or _DEFAULT_VOCABULARY

    predictors = [
        WeightedPredictor(
            predictor=FrequencyPredictor(frequencies=frequencies),
            weight=1.0,
        ),
        WeightedPredictor(
            predictor=HistoryPredictor(history),
            weight=1.5,
        ),
    ]

    return AutocompleteEngine(
        predictors=predictors,
        ranker=[ScoreRanker()],
        history=history,
    )


def _recency_boosted_engine(
    history: History | None,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """
    Frequency + history with exponential recency decay at ranking time.

    DecayRanker applies a time-weighted boost at ranking time so recent
    selections outweigh old ones. The decay half-life is 1 hour: a
    selection made 1 hour ago contributes half the boost of one made now.
    History boost appears separately in explain() output.
    """
    history = history or History()
    frequencies = vocabulary or _DEFAULT_VOCABULARY

    predictors = [
        WeightedPredictor(
            predictor=FrequencyPredictor(frequencies=frequencies),
            weight=1.0,
        ),
        WeightedPredictor(
            predictor=HistoryPredictor(history),
            weight=1.0,
        ),
    ]

    rankers = [
        ScoreRanker(),
        DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=3600),
            weight=2.0,
        ),
    ]

    return AutocompleteEngine(
        predictors=predictors,
        ranker=rankers,
        history=history,
    )


def _robust_engine(
    history: History | None,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """
    BK-tree approximate matching. Suitable for small vocabularies only.

    Uses EditDistancePredictor (BK-tree) for typo recovery. The BK-tree
    degrades to O(n) at max_distance=2 with short prefixes over large
    vocabularies: ~60ms/call at 48k words. Use the 'production' preset
    for typo recovery at full vocabulary scale.
    """
    history = history or History()
    frequencies = vocabulary or _DEFAULT_VOCABULARY

    predictors = [
        WeightedPredictor(
            predictor=FrequencyPredictor(frequencies=frequencies),
            weight=1.0,
        ),
        WeightedPredictor(
            predictor=HistoryPredictor(history),
            weight=1.2,
        ),
        WeightedPredictor(
            predictor=EditDistancePredictor(
                vocabulary=frequencies.keys(),
                max_distance=2,
            ),
            weight=0.4,  # intentionally weak fallback signal
        ),
    ]

    rankers = [
        ScoreRanker(),
        DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=3600),
            weight=1.5,
        ),
    ]

    return AutocompleteEngine(
        predictors=predictors,
        ranker=rankers,
        history=history,
    )


def _production_engine(
    history: History | None,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """
    Trigram approximate matching. Full vocabulary scale, ~600µs/call.

    Solves the BK-tree scalability problem: at max_distance=2 over 48k
    words, the BK-tree's triangle inequality pruning becomes ineffective
    (search ball covers most of the metric space) and search degrades to
    O(n) at ~60ms/call.

    TrigramPredictor pre-filters to a shortlist of ~20-100 words using
    trigram overlap before running exact Levenshtein, giving ~600µs/call
    at the same vocabulary size - a 100x improvement.

    Constraint: trigram matching requires prefix length >= 4. For 1-3
    character prefixes only frequency and history signals apply. Use the
    'robust' preset with a curated small vocabulary if short-prefix typo
    recovery is required.
    """
    history = history or History()
    frequencies = vocabulary or _DEFAULT_VOCABULARY

    predictors = [
        WeightedPredictor(
            predictor=FrequencyPredictor(frequencies=frequencies),
            weight=1.0,
        ),
        WeightedPredictor(
            predictor=HistoryPredictor(history),
            weight=1.2,
        ),
        WeightedPredictor(
            predictor=TrigramPredictor(
                vocabulary=frequencies.keys(),
                max_distance=2,
            ),
            weight=0.4,  # intentionally weak fallback signal
        ),
    ]

    rankers = [
        ScoreRanker(),
        DecayRanker(
            history=history,
            decay=DecayFunction(half_life_seconds=3600),
            weight=1.5,
        ),
    ]

    return AutocompleteEngine(
        predictors=predictors,
        ranker=rankers,
        history=history,
    )


# ---------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------

PRESETS: dict[str, EnginePreset] = {
    "default": EnginePreset(
        name="default",
        description="Frequency + history learning (no typo recovery)",
        build=_default_engine,
        predictors=("frequency", "history"),
        ranking="score-based",
        learning="enabled",
    ),
    "production": EnginePreset(
        name="production",
        description="Trigram typo recovery + recency decay. Recommended for full-scale use.",
        build=_production_engine,
        predictors=("frequency", "history", "trigram"),
        ranking="score + recency decay",
        learning="enabled (time-aware)",
    ),
    "recency": EnginePreset(
        name="recency",
        description="History-aware autocomplete with exponential time decay",
        build=_recency_boosted_engine,
        predictors=("frequency", "history"),
        ranking="score + recency decay",
        learning="enabled (time-aware)",
    ),
    "robust": EnginePreset(
        name="robust",
        description="BK-tree typo recovery. Small vocabularies only (degrades at 48k+ words).",
        build=_robust_engine,
        predictors=("frequency", "history", "edit-distance"),
        ranking="score + recency decay",
        learning="enabled (time-aware)",
    ),
    "stateless": EnginePreset(
        name="stateless",
        description="Pure frequency ranking - no learning, fully reproducible",
        build=_stateless_engine,
        predictors=("frequency",),
        ranking="score-based",
        learning="disabled",
    ),
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def available_presets() -> list[str]:
    return sorted(PRESETS.keys())


def get_preset(name: str) -> EnginePreset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown preset '{name}'. "
            f"Available presets: {', '.join(available_presets())}"
        ) from None


def create_engine(
    preset: str,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """
    Backwards-compatible factory.
    Prefer build_engine(...) in the app layer.
    """
    return get_preset(preset).build(None, vocabulary)


def describe_presets() -> str:
    """
    Human-readable description of all available presets.

    Intended for CLI and documentation output.
    Descriptions are derived from EnginePreset metadata fields,
    not hardcoded - update the PRESETS registry to change output.
    """
    lines: list[str] = []

    for name in available_presets():
        preset = PRESETS[name]
        lines.append(preset.name)
        lines.append(f"  {preset.description}")
        lines.append(f"  predictors: {', '.join(preset.predictors)}")
        lines.append(f"  ranking: {preset.ranking}")
        lines.append(f"  learning: {preset.learning}")
        lines.append("")

    return "\n".join(lines).rstrip()
