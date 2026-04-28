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
from aac.predictors.symspell import SymSpellPredictor
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
# Loaded lazily on first use rather than at import time.  Importing
# ``aac.presets`` would otherwise trigger a 717 KB JSON parse as a
# side effect, which is surprising for a library module.  The lru_cache
# on load_english_frequencies() ensures the file is read exactly once
# regardless of how many preset builders call _get_default_vocabulary().
# ---------------------------------------------------------------------

def _get_default_vocabulary() -> Mapping[str, int]:
    """Return the bundled vocabulary, loading it on first call."""
    return load_english_frequencies()


# ---------------------------------------------------------------------
# Preset builders
# ---------------------------------------------------------------------


def _stateless_engine(
    _: History | None,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """Pure frequency ranking. No learning, no history, fully reproducible."""
    frequencies = vocabulary or _get_default_vocabulary()

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

    Weight rationale:
        Both predictors emit log-normalised scores in (0, 1].
        FrequencyPredictor(weight=1.0) contributes up to 1.0 to the
        combined score.  HistoryPredictor(weight=1.5) contributes up to
        1.5.  A word selected even once scores log(2)/log(2) = 1.0 from
        HistoryPredictor, which after the 1.5× weight gives 1.5 - enough
        to overcome the frequency signal for all but the most common words.
        After a few selections the history signal dominates, which is the
        intended behaviour.
    """
    history = history or History()
    frequencies = vocabulary or _get_default_vocabulary()

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

    Weight rationale:
        FrequencyPredictor and HistoryPredictor are given equal weight (1.0)
        so neither dominates at the prediction layer - the DecayRanker
        (weight=2.0) provides the recency signal that breaks ties.  The 2.0
        weight means a very-recent single selection contributes a boost of
        up to 2.0, enough to push a history-matched word above any
        frequency-only candidate.
    """
    history = history or History()
    frequencies = vocabulary or _get_default_vocabulary()

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
    SymSpell-based typo recovery. Fast at any vocabulary size.

    Uses a delete-neighbourhood index (SymSpell algorithm) for O(1) average
    typo recovery. At 48k words and max_distance=2: ~0.4ms/call - 150x
    faster than the previous BK-tree implementation.

    Construction cost: ~1.5s at 48k words (one-time, at engine creation).
    Memory: ~50MB for the delete-neighbourhood map at 48k words.

    Handles 1-3 character prefixes correctly (unlike TrigramPredictor which
    requires prefix length >= 4).

    Weight rationale:
        All predictors emit log-normalised scores in (0, 1].
        SymSpellPredictor(weight=0.4): intentionally weak - it fires on
        every prefix (including exact matches) and would swamp the frequency
        signal if given a high weight.  0.4 means a perfect typo-corrected
        match contributes 0.4 to the combined score, enough to promote a
        corrected word above a low-frequency exact match, but not above
        a high-frequency one.  HistoryPredictor(weight=1.2) is slightly
        higher than frequency to give user selection history a mild edge.
    """
    history = history or History()
    frequencies = vocabulary or _get_default_vocabulary()

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
            predictor=SymSpellPredictor(
                vocabulary=frequencies.keys(),
                max_distance=2,
                frequencies=frequencies,
            ),
            weight=0.4,
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


def _bktree_engine(
    history: History | None,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """
    BK-tree approximate matching. Retained for benchmarking comparison.

    Degrades to O(n) at max_distance=2 over 48k+ words (~60ms/call).
    Use the 'robust' preset (SymSpell) for production typo recovery.
    """
    history = history or History()
    frequencies = vocabulary or _get_default_vocabulary()

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
            weight=0.4,
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
    frequencies = vocabulary or _get_default_vocabulary()

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
                frequencies=frequencies,
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
        description="SymSpell typo recovery - O(1) queries at any vocabulary size, works on 1-3 char prefixes",
        build=_robust_engine,
        predictors=("frequency", "history", "symspell"),
        ranking="score + recency decay",
        learning="enabled (time-aware)",
    ),
    "bktree": EnginePreset(
        name="bktree",
        description="BK-tree typo recovery - retained for benchmarking. Degrades to O(n) at 48k+ words.",
        build=_bktree_engine,
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
    history: History | None = None,
) -> AutocompleteEngine:
    """
    Build an engine from a named preset with the bundled vocabulary.

    Parameters:
        preset:     Name of the preset (see ``available_presets()``).
        vocabulary: Custom word-frequency mapping.  If omitted, the bundled
                    48 k-word English corpus is used.
        history:    Existing ``History`` instance to attach for learning and
                    persistence.  Pass the result of
                    ``JsonHistoryStore(path).load()`` here so the engine reads
                    and writes the same history that will be saved to disk.
                    If omitted, a fresh in-memory history is created.

    Example - persistent engine across restarts::

        from aac.presets import create_engine
        from aac.storage.json_store import JsonHistoryStore
        from pathlib import Path

        store = JsonHistoryStore(Path("~/.aac_history.json").expanduser())
        engine = create_engine("production", history=store.load())

        suggestions = engine.suggest("prog")
        engine.record_selection("prog", suggestions[0])
        store.save(engine.history)   # persist learning to disk
    """
    return get_preset(preset).build(history, vocabulary)


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
