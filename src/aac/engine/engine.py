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
    """Internal debug surface. Do not mutate returned objects."""

    input: str
    scored: list[ScoredSuggestion]
    ranked: list[ScoredSuggestion]
    suggestions: list[str]


class _PredictorInfo(TypedDict):
    name: str
    weight: float


class DescribeState(TypedDict):
    """Return type of AutocompleteEngine.describe()."""

    predictors: list[_PredictorInfo]
    rankers: list[str]
    history_entries: int


class AutocompleteEngine:
    """Orchestrates prediction, ranking, learning, and explanation."""

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
        """Aggregate scored suggestions from all predictors (additive weighting)."""
        aggregated, _ = self._score_with_breakdown(ctx)
        return aggregated

    def _score_with_breakdown(
        self,
        ctx: CompletionContext,
    ) -> tuple[list[ScoredSuggestion], dict[str, dict[str, float]]]:
        """Like _score(), but also returns breakdown[value][predictor_name] = weighted_score."""
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

        Raises RuntimeError naming the offender if the ranker changed the candidate set.
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
        """Apply all rankers in sequence, enforcing the set-preservation invariant."""
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
        """Return ranked suggestion strings for a prefix. Use explain() for scores."""
        ctx = CompletionContext(text)
        ranked = self._apply_ranking(ctx, self._score(ctx))
        values = [s.suggestion.value for s in ranked]
        return values[:limit] if limit is not None else values

    def predict_scored(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        """Ranked ScoredSuggestion list. For testing and inspection."""
        return self._apply_ranking(ctx, self._score(ctx))

    def _predict_scored_unranked(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        """Return scored suggestions without ranking. Diagnostics only."""
        return self._score(ctx)

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(self, text: str) -> list[RankingExplanation]:
        """Return per-suggestion RankingExplanation objects in final ranked order."""
        ctx = CompletionContext(text)

        # Pre-ranking predictor scores with per-predictor breakdown.
        pre_ranking, predictor_breakdown = self._score_with_breakdown(ctx)

        # Forward pass through the ranker chain. After each ranker, record
        # the score delta it applied, then feed the re-scored list into the
        # next ranker. One pass gives both final ranked order and per-ranker
        # contribution.
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
            # with actual contributions. A 0.0 means "predictor ran, word
            # was below its threshold" - not "predictor not configured".
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
        """explain() as plain dicts - for CLI and JSON serialisation."""
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
        """Return (suggestion, selection_count) pairs. Count 0 means no history for this value."""
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
        """suggest_with_history + suggest_with_confidence in one pipeline pass."""
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
        Return (suggestion, confidence) pairs where confidence is in (0, 1].

        Uses raw score normalisation normally. When the top result dominates
        by more than _DOMINANCE_THRESHOLD (heavy learning), switches to
        rank-based decay so alternatives don't look nearly worthless.
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
        """Replace in-memory History with a fresh empty one. Does not touch persisted stores."""
        self._history = History()

        # Propagate to learning rankers.
        for ranker in self._rankers:
            if isinstance(ranker, LearnsFromHistory):
                ranker.history = self._history

        # Propagate to predictors with a reassignable history property.
        for weighted in self._predictors:
            if isinstance(weighted.predictor, PredictorLearnsFromHistory):
                weighted.predictor.history = self._history

    def record_selection(self, text: str, value: str) -> None:
        """Record a selection into History and notify any predictor record hooks."""
        ctx = CompletionContext(text)
        self._history.record(ctx.prefix(), value)

        for weighted in self._predictors:
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
        """suggest() for multiple prefixes. Returns {prefix: suggestions}."""
        return {text: self.suggest(text, limit=limit) for text in texts}

    def batch_explain(
        self,
        texts: list[str],
        *,
        limit: int | None = None,
    ) -> dict[str, list[RankingExplanation]]:
        """explain() for multiple prefixes. Returns {prefix: explanations}."""
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
        """Async wrapper: runs batch_suggest() with asyncio.gather()."""
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
        """Async wrapper: runs suggest() in the thread pool executor."""
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
        """Return a summary of the engine configuration for inspection."""
        return {
            "predictors": [
                {"name": wp.predictor.name, "weight": wp.weight}
                for wp in self._predictors
            ],
            "rankers": [r.__class__.__name__ for r in self._rankers],
            "history_entries": len(self._history),
        }

    @property
    def history(self) -> History:
        """Return the engine's history source of truth."""
        return self._history

    @property
    def predictors(self) -> list[WeightedPredictor]:
        """Copy of the engine's weighted predictors. Predictor objects are the live instances."""
        return list(self._predictors)

    def to_config(
        self,
        *,
        preset: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EngineConfig:
        """Serialise the engine to an EngineConfig for JSON export and reconstruction."""
        from aac.engine.config import EngineConfig, PredictorConfig, RankerConfig
        from aac.ranking.decay import DecayRanker
        from aac.ranking.learning import LearningRanker

        predictors = [
            PredictorConfig(name=wp.predictor.name, weight=wp.weight)
            for wp in self._predictors
        ]

        rankers: list[RankerConfig] = []
        for r in self._rankers:
            if isinstance(r, DecayRanker):
                rankers.append(RankerConfig(
                    name="decay",
                    params=r.ranker_config(),
                ))
            elif isinstance(r, LearningRanker):
                rankers.append(RankerConfig(
                    name="learning",
                    params=r.ranker_config(),
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