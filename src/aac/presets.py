"""Named engine presets (stateless, default, recency, production, robust) and the compare_presets() tool."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from aac.data import load_english_frequencies
from aac.domain.history import History
from aac.domain.thread_safe_history import ThreadSafeHistory
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
    # Production engines are likely used in concurrent server environments.
    # Use a ThreadSafeHistory by default when the caller does not supply one.
    history = history or ThreadSafeHistory()
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

    Neither approach alone covers the full input range well:

    - TrigramPredictor: ~600µs/call, excellent discrimination at 4+ chars,
      but returns nothing for 1-3 char prefixes where trigrams are too sparse.

    - SymSpellPredictor: ~400µs/call, O(1) average query, correct at any
      prefix length - but at max_distance=2 over 48k words it generates a
      large candidate shortlist for very short queries, adding noise.

    The hybrid uses each where it performs best: SymSpell fills the 1-3 char
    gap where trigram is silent, trigram takes over at 4+ chars where it's
    fast and precise. Both run all the time; at 4+ chars SymSpell contributes
    an additive signal that rarely conflicts with trigram - it's consistent
    enough to leave active rather than switching between strategies.

    Weight rationale:
        FrequencyPredictor(weight=1.0) and HistoryPredictor(weight=1.2) set
        the base signal. SymSpellPredictor(weight=0.35) and
        TrigramPredictor(weight=0.4) are intentionally weak - they are typo
        recovery signals, not primary ranking signals. At short prefixes,
        SymSpell provides the only typo signal and its 0.35 weight is enough
        to surface a corrected word above a low-frequency exact match.
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
    # bktree is retained in PRESETS for internal benchmarking but excluded
    # from the public listing.  It degrades to O(n) at 48k+ words and is
    # not suitable for production use.  Users who need it can call
    # create_engine("bktree") directly.
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


class PresetComparison:
    """
    Side-by-side explanation comparison across multiple presets for a query.

    Side-by-side explanation data across multiple presets for one query.
    See ``compare_presets()`` to build one.

    Attributes:
        text:     The query prefix that was compared.
        presets:  The preset names that were compared, in the order given.
        rows:     One row per unique suggestion, in order of first appearance
                  across all preset results.  Each row is a dict:

                    {
                      "value":        str,          # the suggestion
                      "ranks":        {preset: int | None},
                      "base_scores":  {preset: float | None},
                      "boosts":       {preset: float | None},
                      "finals":       {preset: float | None},
                    }

                  A value of ``None`` means the suggestion was not returned
                  by that preset for this query.

    Use ``to_table()`` for a plain-text summary, or access ``rows`` directly
    for structured output (CLI flags, JSON APIs, Jupyter notebooks).
    """

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
        """
        Return a plain-text comparison table.

        Each row is one suggestion; columns show rank, base score, boost,
        and final score for each preset side-by-side.

        Parameters:
            limit: Maximum rows to include.  Defaults to all rows.
        """
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
            # cast avoids type: ignore while staying explicit about assumptions.
            ranks = row["ranks"]
            bases = row["base_scores"]
            boosts = row["boosts"]
            finals = row["finals"]
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
    """
    Return a cached engine for the given preset, building it if needed.

    The cached engine always has an empty history (no learning state),
    making it suitable for compare_presets() which needs consistent,
    isolated engines. Do not use the cached engine directly for recording
    selections - it is shared across all compare_presets() calls.
    """
    if preset_name not in _ENGINE_CACHE:
        _ENGINE_CACHE[preset_name] = create_engine(preset_name)
    return _ENGINE_CACHE[preset_name]


def warm_cache(preset_names: list[str] | None = None) -> None:
    """
    Pre-build engines for the given presets (or all public presets).

    Call this at application startup to avoid the first-call latency spike
    in compare_presets(). Building all 5 preset engines takes ~8 seconds.

    Example::

        import threading
        threading.Thread(target=warm_cache, daemon=True).start()
    """
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
    Compare suggestion rankings and score breakdowns across multiple presets.

    Builds one engine per preset, runs ``explain()`` on each, and returns a
    ``PresetComparison`` with side-by-side scores for every suggestion that
    appears in any preset's results.

    Useful for questions like: "Would switching to ``production`` change my
    top-5?" or "Which preset surfaces 'receive' for the typo 'recieve'?"

    Parameters:
        text:       The query prefix to compare.
        presets:    Preset names to compare.  Defaults to all public presets
                    (``available_presets()``).
        vocabulary: Custom vocabulary shared across all preset engines.
                    If None, the bundled 48k English corpus is used.
        history:    History shared across all preset engines.  If None, each
                    engine starts with an empty history - so rankings reflect
                    only frequency and typo-recovery signals, not learning.
                    Pass a loaded History to compare presets under real usage.
        limit:      Maximum suggestions to include per preset.  Default: 10.

    Returns:
        A ``PresetComparison`` with ``rows`` (structured data) and
        ``to_table()`` (human-readable summary).

    Example::

        cmp = compare_presets("recieve", ["stateless", "production"])
        print(cmp.to_table())

        # Check whether 'receive' is recovered by all presets
        for row in cmp.rows:
            if row["value"] == "receive":
                print(row["finals"])  # {preset: score or None}

    Notes:
        Building engines with full-scale SymSpell or trigram indexes takes
        ~1–2 seconds per engine.  ``compare_presets()`` is intended for
        offline analysis, not hot-path request handling.  Cache the results
        or run comparisons in a background job.
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
    lookup: dict[str, dict[str, tuple[int, object]]] = {}
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

