"""Named engine presets (stateless, default, recency, production, robust) and the compare_presets() tool."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from aac.data import load_english_frequencies
from aac.domain.history import History
from aac.domain.types import WeightedPredictor
from aac.engine.engine import AutocompleteEngine
from aac.predictors.adaptive_symspell import AdaptiveSymSpellPredictor
from aac.predictors.edit_distance import EditDistancePredictor
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.predictors.symspell import SymSpellPredictor
from aac.predictors.trigram import TrigramPredictor
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.explanation import RankingExplanation
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
    """A named, validated engine configuration. Metadata fields populate describe_presets() output."""

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
    """Frequency + history at the prediction layer. No decay; history appears in the base score."""
    history = history if history is not None else History()
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
    """Frequency + history with 1-hour exponential recency decay at ranking time."""
    history = history if history is not None else History()
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
    """SymSpell typo recovery (~0.4ms/query, works on 1-3 char prefixes) + recency decay."""
    history = history if history is not None else History()
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
    history = history if history is not None else History()
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


def _production_engine(
    history: History | None,
    vocabulary: Mapping[str, int] | None = None,
) -> AutocompleteEngine:
    """
    Hybrid typo recovery: SymSpell for 1-3 char prefixes, trigram for 4+.

    TrigramPredictor returns nothing below 4 chars; SymSpell covers the gap.
    Both run on longer prefixes - SymSpell's additive signal rarely conflicts
    with trigram and is consistent enough to leave active.
    """
    history = history if history is not None else History()
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
            predictor=AdaptiveSymSpellPredictor(
                vocabulary=frequencies.keys(),
                max_distance=2,
                short_prefix_len=4,
                short_max_distance=1,
                frequencies=frequencies,
            ),
            weight=0.35,
        ),
        WeightedPredictor(
            predictor=TrigramPredictor(
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
        description="Hybrid typo recovery (SymSpell 1-3 chars, trigram 4+) + recency decay. Recommended default.",
        build=_production_engine,
        predictors=("frequency", "history", "symspell", "trigram"),
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
    # bktree is excluded from the public listing - it degrades to O(n) at scale.
    return sorted(name for name in PRESETS if name != "bktree")


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
    *,
    thread_safe: bool = False,
) -> AutocompleteEngine:
    """
    Build an engine from a named preset.

    vocabulary: custom word->freq mapping; defaults to bundled 48k English corpus.
    history: pass JsonHistoryStore(path).load() to persist learning across runs.
    thread_safe: wraps history in ThreadSafeHistory for concurrent use.
    """
    from aac.domain.thread_safe_history import ThreadSafeHistory

    if thread_safe and not isinstance(history, ThreadSafeHistory):
        history = ThreadSafeHistory(history)

    return get_preset(preset).build(history, vocabulary)


def describe_presets() -> str:
    """Return a human-readable description of all available presets."""
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


class PresetComparison:
    """Side-by-side ranking data for one query across multiple presets."""

    def __init__(
        self,
        text: str,
        presets: list[str],
        rows: list[dict[str, object]],
    ) -> None:
        self.text = text
        self.presets = presets
        self.rows = rows

    def to_table(self, *, limit: int | None = None) -> str:
        """Return a plain-text comparison table, one suggestion per row."""
        rows = self.rows[:limit] if limit is not None else self.rows
        if not rows:
            return f"No suggestions for {self.text!r} in any preset."

        col_w = max(len(p) for p in self.presets) + 2
        val_w = max(len(str(r["value"])) for r in rows) + 2

        header_parts = [f"{'suggestion':<{val_w}}"]
        for p in self.presets:
            header_parts.append(f"{p:^{col_w * 4}}")
        header = "  ".join(header_parts)

        sub_header_parts = [" " * val_w]
        for _ in self.presets:
            sub_header_parts.append(
                f"{'rank':>{col_w}}{'base':>{col_w}}{'boost':>{col_w}}{'final':>{col_w}}"
            )
        sub_header = "  ".join(sub_header_parts)

        separator = "-" * len(sub_header)

        lines = [header, sub_header, separator]
        for row in rows:
            parts = [f"{row['value']:<{val_w}}"]
            # row structure is guaranteed by compare_presets() above:
            # {"value": str, "ranks": dict, "base_scores": dict, ...}
            ranks = cast(dict[str, int | None], row["ranks"])
            bases = cast(dict[str, float | None], row["base_scores"])
            boosts = cast(dict[str, float | None], row["boosts"])
            finals = cast(dict[str, float | None], row["finals"])
            for p in self.presets:
                rank = ranks.get(p)
                base = bases.get(p)
                boost = boosts.get(p)
                final = finals.get(p)
                rank_s = f"#{rank}" if rank is not None else "-"
                base_s = f"{base:.3f}" if base is not None else "-"
                boost_s = f"{boost:+.3f}" if boost is not None else "-"
                final_s = f"{final:.3f}" if final is not None else "-"
                parts.append(
                    f"{rank_s:>{col_w}}{base_s:>{col_w}}{boost_s:>{col_w}}{final_s:>{col_w}}"
                )
            lines.append("  ".join(parts))

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PresetComparison(text={self.text!r}, "
            f"presets={self.presets!r}, "
            f"rows={len(self.rows)})"
        )


# Module-level engine cache for compare_presets().
# Keyed by preset name. Only populated for the default vocabulary (no custom vocab).
# Building SymSpell indexes takes ~2s per preset; caching makes repeated
# compare_presets() calls instant after the first warm-up.
_ENGINE_CACHE: dict[str, AutocompleteEngine] = {}


def _get_cached_engine(preset_name: str) -> AutocompleteEngine:
    """Return a module-level cached engine for compare_presets(). Always has empty history."""
    if preset_name not in _ENGINE_CACHE:
        _ENGINE_CACHE[preset_name] = create_engine(preset_name)
    return _ENGINE_CACHE[preset_name]


def warm_cache(preset_names: list[str] | None = None) -> None:
    """Pre-build preset engines to avoid first-call latency in compare_presets() (~8s for all presets)."""
    names = preset_names if preset_names is not None else available_presets()
    for name in names:
        _get_cached_engine(name)


def compare_presets(
    text: str,
    presets: list[str] | None = None,
    *,
    vocabulary: Mapping[str, int] | None = None,
    history: History | None = None,
    limit: int = 10,
) -> PresetComparison:
    """
    Run explain() across multiple presets and return a side-by-side PresetComparison.

    First call per preset takes ~1-2s to build indexes; results are cached.
    Intended for offline analysis, not hot-path use.
    """
    preset_names = presets if presets is not None else available_presets()

    # Build one engine per preset.
    #
    # Caching strategy: when no custom vocabulary or history is provided,
    # reuse module-level cached engines. Building 5 preset engines from scratch
    # (including two SymSpell indexes for production) takes ~8s - unacceptable
    # for an interactive comparison tool. The cache is keyed by preset name;
    # custom vocabulary/history always gets a fresh engine.
    #
    # All engines get an independent copy of the History so that engine
    # operations during comparison cannot corrupt each other's state.
    engines: dict[str, AutocompleteEngine] = {}
    for name in preset_names:
        if vocabulary is None and history is None:
            engines[name] = _get_cached_engine(name)
        else:
            engines[name] = create_engine(
                name,
                vocabulary=vocabulary,
                history=history.copy() if history is not None else None,
            )

    # Collect explanations per preset.
    explanations_by_preset: dict[str, list[RankingExplanation]] = {}
    for name, engine in engines.items():
        explanations_by_preset[name] = engine.explain(text)[:limit]

    # Union of all suggestions that appeared in any preset, in order of
    # first appearance (preserves the ordering of the first preset listed).
    seen: dict[str, None] = {}
    for name in preset_names:
        for exp in explanations_by_preset[name]:
            seen.setdefault(exp.value, None)
    all_values = list(seen)

    # Build lookup: preset -> {value: (rank, explanation)}
    lookup: dict[str, dict[str, tuple[int, RankingExplanation]]] = {}
    for name in preset_names:
        lookup[name] = {
            exp.value: (i + 1, exp)
            for i, exp in enumerate(explanations_by_preset[name])
        }

    rows: list[dict[str, object]] = []
    for value in all_values:
        ranks: dict[str, int | None] = {}
        bases: dict[str, float | None] = {}
        boosts: dict[str, float | None] = {}
        finals: dict[str, float | None] = {}

        for name in preset_names:
            entry = lookup[name].get(value)
            if entry is not None:
                rank, exp = entry[0], entry[1]
                ranks[name] = rank
                bases[name] = exp.base_score
                boosts[name] = exp.history_boost
                finals[name] = exp.final_score
            else:
                ranks[name] = None
                bases[name] = None
                boosts[name] = None
                finals[name] = None

        rows.append({
            "value": value,
            "ranks": ranks,
            "base_scores": bases,
            "boosts": boosts,
            "finals": finals,
        })

    return PresetComparison(text=text, presets=preset_names, rows=rows)

