"""AutocompleteEngine: the main public API. Combines predictors, rankers, and history into a single object."""
from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

from aac.domain.history import History
from aac.domain.types import (
    CompletionContext,
    Predictor,
    ScoredSuggestion,
    WeightedPredictor,
)
from aac.ranking.base import Ranker
from aac.ranking.contracts import (
    LearnsFromHistory,
    PredictorAcceptsRecord,
    PredictorLearnsFromHistory,
)
from aac.ranking.explanation import RankingExplanation
from aac.ranking.score import ScoreRanker

if TYPE_CHECKING:
    from aac.engine.config import EngineConfig

# ---------------------------------------------------------------------------
# Confidence scoring constants
# ---------------------------------------------------------------------------

# When the top-ranked candidate's score exceeds this multiple of the second
# candidate's score, we consider it "dominant" - a heavily-learned selection.
# Raw score normalisation in that regime makes alternatives look nearly worthless
# (e.g. 6% confidence) even when they are excellent completions. Above this
# threshold we switch to rank-based confidence so alternatives remain meaningful.
_DOMINANCE_THRESHOLD: float = 4.0

# Rank-decay rate used when the top candidate is dominant.
# Position k receives confidence 1 / (1 + k * _RANK_DECAY_RATE), capped at 1.0.
# k=0 → 1.00, k=1 → 0.71, k=2 → 0.56, k=3 → 0.47, k=4 → 0.41
_RANK_DECAY_RATE: float = 0.4


class DebugState(TypedDict):
    """
    Internal debug surface.

    WARNING:
        Values reference live internal objects.
        Returned data MUST NOT be mutated.
    """

    input: str
    scored: list[ScoredSuggestion]
    ranked: list[ScoredSuggestion]
    suggestions: list[str]


class _PredictorInfo(TypedDict):
    name: str
    weight: float


class DescribeState(TypedDict):
    """
    Return type of AutocompleteEngine.describe().

    Fully typed so callers (tests, CLI, tooling) get precise
    type information rather than dict[str, object].
    """

    predictors: list[_PredictorInfo]
    rankers: list[str]
    history_entries: int


class AutocompleteEngine:
    """
    Orchestrates prediction, ranking, learning, and explanation.

    This is the public entrypoint for the autocomplete system.
    Only documented methods are considered stable.

    Architectural invariants:
    - Engine owns the CompletionContext lifecycle
    - Internally everything operates on ScoredSuggestion
    - Rankers must not add or remove suggestions
    - Scores must remain finite
    - Explanation final scores must reconcile with ranking scores
    - Projection to Suggestion happens only at API boundaries
    - History has a single source of truth
    """

    def __init__(
        self,
        predictors: Sequence[Predictor | WeightedPredictor],
        ranker: Ranker | Sequence[Ranker] | None = None,
        history: History | None = None,
    ) -> None:
        # Normalise predictors to WeightedPredictor
        self._predictors: list[WeightedPredictor] = []
        for p in predictors:
            if isinstance(p, WeightedPredictor):
                self._predictors.append(p)
            else:
                self._predictors.append(WeightedPredictor(predictor=p, weight=1.0))

        # Normalise rankers
        if ranker is None:
            self._rankers: list[Ranker] = [ScoreRanker()]
        elif isinstance(ranker, Ranker):
            self._rankers = [ranker]
        else:
            self._rankers = list(ranker)

        # Resolve history source of truth.
        #
        # All learning rankers must share the same History object as the engine.
        # If histories diverge, record_selection() writes to engine history but
        # rankers read their own instance and never see updates - silent breakage.
        # We enforce consistency at construction time and fail fast.
        learning_rankers = [r for r in self._rankers if isinstance(r, LearnsFromHistory)]

        if history is not None:
            for ranker in learning_rankers:
                if ranker.history is not history:
                    raise ValueError(
                        f"{ranker.__class__.__name__} owns a different History instance "
                        f"than the one passed to AutocompleteEngine. "
                        f"All learning rankers must share the engine's History. "
                        f"Pass the same History object to both the ranker and the engine, "
                        f"or omit the engine-level history and let the engine adopt the "
                        f"ranker's history automatically."
                    )
            self._history = history
        else:
            owned = next(
                (r.history for r in learning_rankers),
                None,
            )
            self._history = owned if owned is not None else History()
            for ranker in learning_rankers:
                if ranker.history is not self._history:
                    raise ValueError(
                        f"{ranker.__class__.__name__} owns a different History instance "
                        f"than {self._rankers[0].__class__.__name__}. "
                        f"All learning rankers must share the same History object."
                    )

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _score(
        self,
        ctx: CompletionContext,
    ) -> list[ScoredSuggestion]:
        """
        Collect and aggregate scored suggestions from all predictors.

        Aggregation is additive across predictors and weights.
        Predictor explanations are preserved but not interpreted here.
        """
        aggregated, _ = self._score_with_breakdown(ctx)
        return aggregated

    def _score_with_breakdown(
        self,
        ctx: CompletionContext,
    ) -> tuple[list[ScoredSuggestion], dict[str, dict[str, float]]]:
        """
        Like _score(), but also returns a per-predictor weighted contribution map.

        Returns:
            (suggestions, breakdown) where breakdown[value][predictor_name]
            is the weighted score that predictor contributed for that value.

        Used by explain() to build base_components without parsing trace strings.
        Trace strings are human-readable and must not be treated as structured data -
        predictor names are arbitrary and may contain any characters.
        """
        aggregated: dict[str, ScoredSuggestion] = {}
        breakdown: dict[str, dict[str, float]] = {}

        for weighted in self._predictors:
            results = weighted.predictor.predict(ctx)
            predictor_name = weighted.predictor.name

            for scored in results:
                key = scored.suggestion.value
                weighted_score = scored.score * weighted.weight
                trace_entry = (
                    f"Predictor={predictor_name}, "
                    f"weight={weighted.weight}, raw_score={scored.score}"
                )

                if key not in aggregated:
                    aggregated[key] = ScoredSuggestion(
                        suggestion=scored.suggestion,
                        score=weighted_score,
                        explanation=scored.explanation,
                        trace=(trace_entry,),
                    )
                    breakdown[key] = {predictor_name: weighted_score}
                else:
                    prev = aggregated[key]
                    aggregated[key] = ScoredSuggestion(
                        suggestion=prev.suggestion,
                        score=prev.score + weighted_score,
                        explanation=prev.explanation,
                        trace=prev.trace + (trace_entry,),
                    )
                    breakdown[key][predictor_name] = (
                        breakdown[key].get(predictor_name, 0.0) + weighted_score
                    )

        return list(aggregated.values()), breakdown

    @staticmethod
    def _check_ranker_invariant(
        ranker_name: str,
        before: list[ScoredSuggestion],
        after: list[ScoredSuggestion],
    ) -> None:
        """
        Enforce the ranker set-preservation invariant.

        Rankers may reorder and rescore suggestions but must not add or
        remove entries.  Calling this after every ranker step ensures any
        violation is caught immediately and names the offending ranker.

        Extracted as a static method so both ``_apply_ranking()`` and
        ``explain()`` share exactly the same check with no code duplication.
        If the invariant description ever changes it changes in one place.

        Raises:
            RuntimeError: If the ranker added or removed any suggestion.
        """
        before_values = {s.suggestion.value for s in before}
        after_values = {s.suggestion.value for s in after}
        if after_values != before_values:
            added = after_values - before_values
            removed = before_values - after_values
            raise RuntimeError(
                f"Ranker {ranker_name} modified the suggestion set. "
                f"Added: {added or 'none'}. Removed: {removed or 'none'}."
            )

    def _apply_ranking(
        self,
        ctx: CompletionContext,
        scored: list[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        """
        Apply rankers while enforcing engine invariants.

        Rankers may reorder or rescore suggestions, but must not add or
        remove entries.

        Raises:
            RuntimeError: If a ranker adds or removes suggestions.
            ValueError: If a ranker produces a non-finite score.
        """
        ranked = scored

        for ranker in self._rankers:
            before = ranked
            ranked = ranker.rank(ctx.text, ranked)
            self._check_ranker_invariant(ranker.__class__.__name__, before, ranked)

        for s in ranked:
            if not math.isfinite(s.score):
                raise ValueError(
                    f"Non-finite score for '{s.suggestion.value}': {s.score}"
                )

        return ranked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, text: str, *, limit: int | None = None) -> list[str]:
        """
        Return ranked suggestion strings for user-facing consumption.

        Parameters:
            text:  The input prefix to complete.
            limit: Maximum number of suggestions to return.  If omitted,
                   all suggestions from the predictors are returned.
                   Equivalent to ``engine.suggest(text)[:limit]`` but
                   avoids constructing the full list when only the top-N
                   are needed.

        Scores and explanations are intentionally hidden.
        Use explain() or predict_scored() for introspection.
        """
        ctx = CompletionContext(text)
        ranked = self._apply_ranking(ctx, self._score(ctx))
        values = [s.suggestion.value for s in ranked]
        return values[:limit] if limit is not None else values

    def predict_scored(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        """
        Return ranked scored suggestions.

        Intended for testing, benchmarking, and engine-level inspection.
        Guarantees: ranking invariants enforced, deterministic ordering,
        finite scores.
        """
        return self._apply_ranking(ctx, self._score(ctx))

    def _predict_scored_unranked(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        """
        INTERNAL: Return scored suggestions WITHOUT ranking.

        Does not apply rankers or enforce ranking invariants.
        Intended for diagnostics and internal inspection only.
        """
        return self._score(ctx)

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(self, text: str) -> list[RankingExplanation]:
        """
        Return per-suggestion ranking explanations in final ranked order.

        Architecture:
            Explanations are built from two sources of ground truth:

            1. **Base score** - the aggregated predictor score from
               ``_score_with_breakdown()``, before any ranker touches it.
               This is always the correct base regardless of which rankers
               are present or how many.

            2. **Ranker deltas** - captured in a single forward pass through
               the ranker chain.  For each ranker, the score delta it applied
               is ``post_score - pre_score`` per suggestion.  This is computed
               once, not by re-running the pipeline a second time.

            This approach is immune to the double-pipeline problem: explain()
            costs the same as suggest(), not 2×.  It is also immune to the
            double-counting problem that plagues explanation-by-accumulation:
            there is no way for one ranker's explanation to overwrite or add
            to another's base score.

        Returns explanations ordered by final ranked position.
        """
        ctx = CompletionContext(text)

        # Ground truth 1: pre-ranking predictor scores with per-predictor breakdown.
        pre_ranking, predictor_breakdown = self._score_with_breakdown(ctx)

        # Single forward pass through the ranker chain.
        # After each ranker we record the score delta it applied, then pass
        # the re-scored list into the next ranker.  This gives us both the
        # final ranked order AND each ranker's individual contribution in
        # exactly one pipeline run.
        ranker_deltas: list[tuple[str, dict[str, float]]] = []
        running = pre_ranking

        for ranker in self._rankers:
            pre_scores = {s.suggestion.value: s.score for s in running}
            after = ranker.rank(ctx.text, running)
            self._check_ranker_invariant(ranker.__class__.__name__, running, after)

            post_scores = {s.suggestion.value: s.score for s in after}
            source_name = ranker.__class__.__name__.replace("Ranker", "").lower()

            deltas: dict[str, float] = {
                v: post_scores[v] - pre_scores[v]
                for v in pre_scores
                if abs(post_scores[v] - pre_scores[v]) > 1e-12
            }
            if deltas:
                ranker_deltas.append((source_name, deltas))

            running = after

        # `running` is now in final ranked order.
        # Build explanations using pre-ranking breakdown (base) and
        # accumulated ranker deltas (boost).
        pre_by_value = {s.suggestion.value: s for s in pre_ranking}
        post_by_value = {s.suggestion.value: s.score for s in running}

        explanations: list[RankingExplanation] = []
        # All predictor names configured in this engine - used to populate
        # base_components with 0.0 for predictors that didn't fire for this
        # word.  Without this, a word that ranks beyond FrequencyPredictor's
        # max_results cutoff would show no "frequency" key in base_components,
        # which is indistinguishable from "no FrequencyPredictor configured".
        all_predictor_names = [wp.predictor.name for wp in self._predictors]
        all_ranker_names = [
            r.__class__.__name__.replace("Ranker", "").lower()
            for r in self._rankers
        ]

        for s in running:
            value = s.suggestion.value
            pre = pre_by_value[value]
            base_score = pre.score
            predictor_source = pre.explanation.source if pre.explanation else "unknown"

            # Start with zeros for every configured predictor, then overwrite
            # with actual contributions.  This makes the breakdown complete:
            # a 0.0 means "predictor ran, word was below its threshold",
            # not "predictor not configured".
            base_components: dict[str, float] = {
                name: 0.0 for name in all_predictor_names
            }
            base_components.update(predictor_breakdown.get(value, {}))

            total_boost = post_by_value[value] - base_score
            history_components: dict[str, float] = {
                name: 0.0 for name in all_ranker_names
                if name not in ("score",)  # ScoreRanker applies no boost
            }
            for ranker_name, deltas in ranker_deltas:
                if value in deltas:
                    history_components[ranker_name] = deltas[value]

            # contribution_pct: what fraction of the final score came from
            # each source.  Useful for weight-tuning decisions.
            final = base_score + total_boost
            if abs(final) > 1e-12:
                contribution_pct: dict[str, float] = {
                    k: round(v / final, 4)
                    for k, v in {**base_components, **history_components}.items()
                    if abs(v) > 1e-12
                }
            else:
                contribution_pct = {}

            explanations.append(
                RankingExplanation(
                    value=value,
                    base_score=base_score,
                    history_boost=total_boost,
                    final_score=base_score + total_boost,
                    source=predictor_source,
                    base_components=base_components,
                    history_components=history_components,
                    contribution_pct=contribution_pct,
                )
            )

        return explanations

    def explain_as_dicts(self, text: str) -> list[dict[str, object]]:
        """
        Convenience adapter for CLI and serialisation layers.

        Returns per-suggestion dicts with top-level score fields and, when
        multiple rankers contributed, a ``components`` breakdown showing each
        ranker's individual contribution.  The ``source`` field names the
        composite when more than one ranker contributed, rather than silently
        preserving the name of the first ranker.

        Schema::

            {
                "value":               str,
                "base_score":          float,
                "history_boost":       float,
                "final_score":         float,
                "sources":             [str, ...],
                "base_components":     {str: float},
                "history_components":  {str: float},
                "contribution_pct":    {str: float},
            }
        """
        results: list[dict[str, object]] = []
        for e in self.explain(text):
            all_sources = list(e.base_components.keys()) + [
                k for k in e.history_components if k not in e.base_components
            ]
            results.append(
                {
                    "value": e.value,
                    "base_score": e.base_score,
                    "history_boost": e.history_boost,
                    "final_score": e.final_score,
                    "sources": all_sources if len(all_sources) > 1 else [e.source],
                    "base_components": dict(e.base_components),
                    "history_components": dict(e.history_components),
                    "contribution_pct": dict(e.contribution_pct),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def suggest_with_history(
        self,
        text: str,
        *,
        limit: int | None = None,
    ) -> list[tuple[str, int]]:
        """
        Return ranked suggestions paired with their raw selection counts.

        Each suggestion is paired with the number of times the user has
        selected it for this prefix.  A count of 0 means the suggestion
        comes from frequency or typo-recovery signals, not from recorded
        history.

        This is the right API when you want to show a count badge or
        "recently used" marker next to suggestions in a UI.  Calling
        ``suggest()`` and ``history.counts_for_prefix()`` separately and
        zipping the results produces the same data but requires two calls
        and risks the rankings diverging if history is updated between them.

        Parameters:
            text:  The input prefix to complete.
            limit: Maximum number of suggestions to return.

        Returns:
            List of ``(suggestion, count)`` pairs in ranked order.

        Example::

            for word, count in engine.suggest_with_history("prog", limit=5):
                badge = f"({count})" if count > 0 else ""
                print(f"{word} {badge}")
        """
        ctx = CompletionContext(text)
        ranked = self._apply_ranking(ctx, self._score(ctx))
        if limit is not None:
            ranked = ranked[:limit]

        prefix = ctx.prefix()
        counts = self._history.counts_for_prefix(prefix)

        return [
            (s.suggestion.value, counts.get(s.suggestion.value, 0))
            for s in ranked
        ]

    def suggest_full(
        self,
        text: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        """
        Return ranked suggestions with count and confidence in a single pipeline pass.

        Combines ``suggest_with_history()`` and ``suggest_with_confidence()``
        into one call so the pipeline runs exactly once. Each result is a dict
        with keys ``word`` (str), ``count`` (int), and ``confidence`` (float).

        Use this when you need all three signals - for example in a demo UI
        that shows a count badge and a confidence bar alongside each suggestion.
        Calling ``suggest_with_history`` and ``suggest_with_confidence``
        separately runs the full score → rank pipeline twice, which doubles
        SymSpell lookups, trigram lookups, and decay calculations per request.

        Parameters:
            text:  The input prefix to complete.
            limit: Maximum number of suggestions to return.

        Returns:
            List of ``{"word": str, "count": int, "confidence": float}`` dicts
            in ranked order.

        Example::

            for item in engine.suggest_full("prog", limit=10):
                print(f"{item['word']}  ({item['count']})  {item['confidence']:.0%}")
        """
        ctx = CompletionContext(text)
        ranked = self._apply_ranking(ctx, self._score(ctx))
        if limit is not None:
            ranked = ranked[:limit]

        prefix = ctx.prefix()
        counts = self._history.counts_for_prefix(prefix)

        # Confidence computation (same logic as suggest_with_confidence,
        # but applied to the already-ranked list - no second pipeline run).
        if not ranked:
            return []

        top_score = ranked[0].score
        second_score = ranked[1].score if len(ranked) > 1 else 0.0
        safe_second = max(abs(second_score), 1e-9)
        is_dominant = top_score / safe_second > _DOMINANCE_THRESHOLD
        normaliser = max(abs(top_score), 1e-9)

        results = []
        for k, s in enumerate(ranked):
            value = s.suggestion.value
            if is_dominant:
                confidence = 1.0 / (1.0 + k * _RANK_DECAY_RATE)
            else:
                confidence = s.score / normaliser
            results.append({
                "word": value,
                "count": counts.get(value, 0),
                "confidence": float(min(1.0, max(0.0, confidence))),
            })

        return results

    def suggest_with_confidence(
        self,
        text: str,
        *,
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return ranked suggestions with normalised confidence scores.

        Each suggestion is paired with a confidence value in (0, 1] where
        1.0 means the top-ranked candidate.

        **Normalisation design:**

        Raw score division (score / top_score) becomes misleading when
        history learning creates large score gaps - e.g. after 5 selections,
        the top result may score 9.1 while the second scores 0.57, giving
        the second a 6% confidence even though it's an excellent match.

        This method uses a hybrid approach:
          - If the top score dominates by more than ``_DOMINANCE_THRESHOLD`` (4×),
            rank-based weighting is applied: position k gets
            ``1 / (1 + k * _RANK_DECAY_RATE)``, capped at 1.0.
            This reflects that a heavily-selected word IS the right answer
            (high confidence) while preserving meaningful separation for
            the remaining alternatives.
          - If scores are in a normal range (no strong dominance), raw
            normalised scores are used - they're meaningful and stable.

        Parameters:
            text:  The input prefix to complete.
            limit: Maximum number of suggestions to return.

        Returns:
            List of ``(suggestion, confidence)`` pairs in ranked order.

        Example::

            results = engine.suggest_with_confidence("prog", limit=5)
            for word, conf in results:
                label = "\u2605 " if conf > 0.8 else "  "
                print(f"{label}{word}  ({conf:.0%})")
        """
        ctx = CompletionContext(text)
        ranked = self._apply_ranking(ctx, self._score(ctx))
        if limit is not None:
            ranked = ranked[:limit]

        if not ranked:
            return []

        if len(ranked) == 1:
            return [(ranked[0].suggestion.value, 1.0)]

        top_score = ranked[0].score
        second_score = ranked[1].score if len(ranked) > 1 else 0.0

        # Detect strong dominance: if the top candidate has been heavily
        # boosted by learning (top > _DOMINANCE_THRESHOLD × second), raw score
        # normalisation would make all other results look nearly worthless.
        # Use rank-based weighting instead: position k gets
        # 1/(1 + k * _RANK_DECAY_RATE). See module constants for rationale.
        safe_second = max(abs(second_score), 1e-9)
        is_dominant = top_score / safe_second > _DOMINANCE_THRESHOLD

        normaliser = max(abs(top_score), 1e-9)

        results = []
        for k, s in enumerate(ranked):
            if is_dominant:
                conf = 1.0 / (1.0 + k * _RANK_DECAY_RATE)
            else:
                conf = s.score / normaliser
            results.append((s.suggestion.value, min(1.0, max(0.0, conf))))

        return results

    def reset_history(self) -> None:
        """
        Clear all recorded history from the engine's in-memory state.

        Replaces the internal History object with a fresh empty one and
        propagates the change to all learning rankers and any predictors
        that expose a ``history`` attribute (e.g. ``HistoryPredictor``).

        This does not modify any persisted store - if you have a
        ``JsonHistoryStore``, call ``store.save(engine.history)`` after
        resetting to write the empty history to disk.  Otherwise the next
        ``store.load()`` will restore the old history.

        Example - reset and persist the cleared state::

            engine.reset_history()
            store.save(engine.history)

        Typical use cases: testing, user-initiated "forget everything",
        or switching to a new domain without restarting the process.
        """
        self._history = History()

        # Propagate to learning rankers.
        for ranker in self._rankers:
            if isinstance(ranker, LearnsFromHistory):
                ranker.history = self._history

        # Propagate to predictors that implement PredictorLearnsFromHistory.
        # Using the typed protocol rather than hasattr() means only predictors
        # that explicitly opt in (by implementing the property) are updated.
        # A predictor with an unrelated attribute named 'history' is not
        # affected - it must satisfy the full protocol (readable + settable
        # property returning History) to be updated here.
        for weighted in self._predictors:
            if isinstance(weighted.predictor, PredictorLearnsFromHistory):
                weighted.predictor.history = self._history

    def record_selection(self, text: str, value: str) -> None:
        """
        Record a user selection for learning.

        Writes to engine history directly, then calls record(ctx, value)
        on any predictor that implements that method. This hook exists for
        predictors that maintain private state beyond the shared History.

        Note: HistoryPredictor reads from the shared History and does not
        implement a record hook - the engine's direct write is sufficient.
        Adding a hook to HistoryPredictor would record each selection twice.

        Key invariant:
            History is keyed by ``ctx.prefix()``, not ``ctx.text``.
            Lookups in ``counts_for_prefix()`` and ``entries_for_prefix()``
            use the normalised prefix as the key. Recording under ``ctx.text``
            (the raw input string) would produce keys that never match lookups,
            silently disabling the learning signal.
        """
        ctx = CompletionContext(text)
        self._history.record(ctx.prefix(), value)

        for weighted in self._predictors:
            # runtime_checkable Protocol checks attribute *presence* only, not
            # type. A predictor with a non-callable .record attribute (e.g. a
            # string) will pass isinstance() but must not be called.
            if isinstance(weighted.predictor, PredictorAcceptsRecord) and callable(
                weighted.predictor.record
            ):
                weighted.predictor.record(ctx, value)

    # ------------------------------------------------------------------
    # Debug (INTENTIONALLY UNSTABLE)
    # ------------------------------------------------------------------

    def debug(self, text: str) -> DebugState:
        """
        Developer-only debug surface. NOT a stable API.
        Returned objects MUST NOT be mutated.
        """
        ctx = CompletionContext(text)
        scored = self._score(ctx)
        ranked = self._apply_ranking(ctx, scored)
        return {
            "input": text,
            "scored": scored,
            "ranked": ranked,
            "suggestions": [s.suggestion.value for s in ranked],
        }

    def batch_suggest(
        self,
        texts: list[str],
        *,
        limit: int | None = None,
    ) -> dict[str, list[str]]:
        """
        Return suggestions for multiple prefixes in one call.

        Equivalent to calling ``suggest()`` on each prefix individually,
        but expressed as a single API call so callers don't need a loop.
        Useful for: evaluation pipelines, search dashboards pre-fetching
        completions for multiple fields, test harnesses.

        Parameters:
            texts: List of input prefixes to complete.
            limit: Maximum number of suggestions per prefix.

        Returns:
            Dict mapping each prefix to its suggestion list.
            Preserves input order. If a prefix produces no suggestions,
            its value is an empty list.

        Example::

            results = engine.batch_suggest(
                ["prog", "hel", "wor"],
                limit=5,
            )
            # {
            #   "prog": ["programming", "program", ...],
            #   "hel":  ["help", "hello", "held", ...],
            #   "wor":  ["word", "world", "work", ...],
            # }
        """
        return {text: self.suggest(text, limit=limit) for text in texts}

    def batch_explain(
        self,
        texts: list[str],
        *,
        limit: int | None = None,
    ) -> dict[str, list[RankingExplanation]]:
        """
        Return score explanations for multiple prefixes in one call.

        Equivalent to calling ``explain()`` on each prefix individually.
        Useful for offline analysis of ranking behaviour across a query set.

        Parameters:
            texts: List of input prefixes to explain.
            limit: Maximum number of explanations per prefix.

        Returns:
            Dict mapping each prefix to its explanation list.

        Example::

            explanations = engine.batch_explain(["prog", "hel"])
            for prefix, exps in explanations.items():
                print(f"{prefix}: top={exps[0].value} final={exps[0].final_score:.3f}")
        """
        result = {}
        for text in texts:
            exps = self.explain(text)
            result[text] = exps[:limit] if limit is not None else exps
        return result

    async def batch_suggest_async(
        self,
        texts: list[str],
        *,
        limit: int | None = None,
    ) -> dict[str, list[str]]:
        """
        Async version of batch_suggest().

        Schedules each query on the thread pool executor via suggest_async()
        and awaits all results with asyncio.gather(). This keeps the event
        loop unblocked while queries run - the primary use case is async web
        frameworks (FastAPI, Starlette) where blocking the event loop degrades
        all concurrent request handlers.

        **GIL note**: on CPython, pure-Python CPU-bound work does not run in
        true parallel across threads - threads take turns holding the GIL.
        batch_suggest_async improves *event-loop responsiveness* but does not
        reduce *wall-clock latency* for the batch itself compared to a
        sequential loop.  If you need genuine parallelism for large batches,
        use ProcessPoolExecutor or run the engine in a worker process.

        Example::

            @app.get("/batch")
            async def batch(q: list[str]) -> dict[str, list[str]]:
                return await engine.batch_suggest_async(q, limit=5)
        """
        tasks = [self.suggest_async(text, limit=limit) for text in texts]
        results = await asyncio.gather(*tasks)
        return dict(zip(texts, results, strict=False))

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def suggest_async(
        self,
        text: str,
        *,
        limit: int | None = None,
    ) -> list[str]:
        """
        Async wrapper around suggest().

        Runs the synchronous suggest() in the default thread pool executor
        so it does not block the event loop.  Use this in async frameworks
        (FastAPI, Starlette, aiohttp) to keep request handlers non-blocking.

        ``asyncio.get_running_loop()`` is used rather than the deprecated
        ``get_event_loop()``.  ``get_running_loop()`` raises ``RuntimeError``
        if called outside a running coroutine, making misuse explicit rather
        than silently returning or creating a new loop.

        Example::

            @app.get("/suggest")
            async def suggest(q: str) -> list[str]:
                return await engine.suggest_async(q, limit=10)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.suggest(text, limit=limit))

    async def explain_async(self, text: str) -> list[RankingExplanation]:
        """Async wrapper around explain(). See suggest_async() for rationale."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.explain(text))

    async def record_selection_async(self, text: str, value: str) -> None:
        """Async wrapper around record_selection(). See suggest_async() for rationale."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.record_selection(text, value))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> DescribeState:
        """
        Return a typed description of the engine configuration.

        Intended for CLI inspection, debugging, and documentation.
        ``history_entries`` is the total number of recorded selections across
        all prefixes.  Reading ``len(self._history._entries)`` directly is
        intentional: ``entries()`` returns a full tuple copy (O(n) allocation)
        while ``_entries`` is the underlying list (O(1) len).
        """
        return {
            "predictors": [
                {"name": wp.predictor.name, "weight": wp.weight}
                for wp in self._predictors
            ],
            "rankers": [r.__class__.__name__ for r in self._rankers],
            "history_entries": len(self._history._entries),
        }

    @property
    def history(self) -> History:
        """Return the engine's history source of truth."""
        return self._history

    def to_config(
        self,
        *,
        preset: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EngineConfig:
        """
        Serialise this engine's configuration to an ``EngineConfig``.

        The config captures predictor names, weights, and ranker parameters
        in a JSON-serialisable form.  Use it to:

        - Deploy the same engine to multiple servers without repeating
          Python constructor calls.
        - Audit and diff two engine configurations (``config_a.diff(config_b)``).
        - Store the engine config alongside vocabulary and history for
          full operational reproducibility.

        Parameters:
            preset:   If this engine was built via ``create_engine()``,
                      pass the preset name here so ``config.build()`` can
                      use the fast preset reconstruction path.  If None,
                      ``config.build()`` uses the ``PredictorRegistry``
                      to reconstruct each predictor by name.
            metadata: Arbitrary caller-supplied key-value pairs stored
                      alongside the config (e.g. vocabulary path, deploy
                      timestamp, git SHA).

        Returns:
            An ``EngineConfig`` that can be serialised via ``to_json()``
            and reconstructed via ``EngineConfig.from_json(...).build()``.

        Example::

            config = engine.to_config(
                preset="production",
                metadata={"vocabulary": "~/.aac_vocab.json"},
            )
            with open("engine_config.json", "w") as f:
                f.write(config.to_json())

            # On another server:
            with open("engine_config.json") as f:
                engine2 = EngineConfig.from_json(f.read()).build()
        """
        from aac.engine.config import EngineConfig, PredictorConfig, RankerConfig
        from aac.ranking.decay import DecayRanker

        predictors = [
            PredictorConfig(name=wp.predictor.name, weight=wp.weight)
            for wp in self._predictors
        ]

        rankers: list[RankerConfig] = []
        for r in self._rankers:
            if isinstance(r, DecayRanker):
                rankers.append(RankerConfig(
                    name="decay",
                    params={
                        "half_life_seconds": r._decay.half_life_seconds,
                        "weight": r._weight,
                    },
                ))
            else:
                rankers.append(RankerConfig(
                    name=r.__class__.__name__.replace("Ranker", "").lower(),
                ))

        return EngineConfig(
            preset=preset,
            predictors=predictors,
            rankers=rankers,
            metadata=metadata or {},
        )