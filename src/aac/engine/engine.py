from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypedDict

from aac.domain.history import History
from aac.domain.types import (
    CompletionContext,
    Predictor,
    ScoredSuggestion,
    Suggestion,
    WeightedPredictor,
)
from aac.ranking.base import Ranker
from aac.ranking.contracts import LearnsFromHistory
from aac.ranking.explanation import RankingExplanation
from aac.ranking.score import ScoreRanker


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
        # Normalize predictors to WeightedPredictor
        self._predictors: list[WeightedPredictor] = []
        for p in predictors:
            if isinstance(p, WeightedPredictor):
                self._predictors.append(p)
            else:
                self._predictors.append(
                    WeightedPredictor(predictor=p, weight=1.0)
                )

        # Normalize rankers
        if ranker is None:
            self._rankers: list[Ranker] = [ScoreRanker()]
        elif isinstance(ranker, Ranker):
            self._rankers = [ranker]
        else:
            self._rankers = list(ranker)

        # Resolve history source of truth
        if history is not None:
            self._history = history
        elif any(isinstance(r, LearnsFromHistory) for r in self._rankers):
            for r in self._rankers:
                if isinstance(r, LearnsFromHistory):
                    self._history = r.history
                    break
        else:
            self._history = History()

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _score(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        """
        Collect and aggregate scored suggestions from all predictors.

        Notes:
            - Predictor explanations are preserved but not interpreted here.
            - Aggregation is additive across predictors and weights.
        """
        aggregated: dict[str, ScoredSuggestion] = {}

        for weighted in self._predictors:
            results = weighted.predictor.predict(ctx)

            for scored in results:
                key = scored.suggestion.value
                weighted_score = scored.score * weighted.weight

                trace_entry = (
                    f"Predictor={weighted.predictor.name}, "
                    f"weight={weighted.weight}, raw_score={scored.score}"
                )

                if key not in aggregated:
                    aggregated[key] = ScoredSuggestion(
                        suggestion=scored.suggestion,
                        score=weighted_score,
                        explanation=scored.explanation,
                        trace=[trace_entry],
                    )
                else:
                    prev = aggregated[key]
                    aggregated[key] = ScoredSuggestion(
                        suggestion=prev.suggestion,
                        score=prev.score + weighted_score,
                        explanation=prev.explanation,
                        trace=prev.trace + [trace_entry],
                    )

        return list(aggregated.values())

    def _apply_ranking(
        self,
        ctx: CompletionContext,
        scored: list[ScoredSuggestion],
    ) -> list[ScoredSuggestion]:
        """
        Apply rankers while enforcing engine invariants.

        Rankers may reorder or rescore suggestions,
        but must not add or remove entries.
        """
        ranked = scored
        original_values = {s.suggestion.value for s in ranked}

        for ranker in self._rankers:
            ranked = ranker.rank(ctx.text, ranked)

            assert {s.suggestion.value for s in ranked} == original_values, (
                f"Ranker {ranker.__class__.__name__} modified suggestion set"
            )
            
        for s in ranked:
            if not math.isfinite(s.score):
                raise ValueError(
                    f"Non-finite score for '{s.suggestion.value}': {s.score}"
                )

        return ranked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, text: str) -> list[Suggestion]:
        """
        Return ranked suggestions for user-facing consumption.

        This API intentionally hides scores and explanations.
        Use explain() or debug() for introspection.
        """
        ctx = CompletionContext(text)
        ranked = self._apply_ranking(ctx, self._score(ctx))
        return [s.suggestion for s in ranked]

    def predict_scored(self, ctx: CompletionContext) -> list[ScoredSuggestion]:
        """
        Return ranked scored suggestions.

        This is the primary scored-output API intended for:
        - testing
        - benchmarking
        - engine-level inspection

        Guarantees:
        - ranking invariants enforced
        - deterministic ordering
        - finite scores
        """
        return self._apply_ranking(ctx, self._score(ctx))

    def _predict_scored_unranked(
        self, ctx: CompletionContext
    ) -> list[ScoredSuggestion]:
        """
        INTERNAL: Return scored suggestions WITHOUT ranking.

        WARNING:
            - Does not apply rankers
            - Does not enforce ranking invariants
            - Intended for diagnostics / internal inspection only
        """
        return self._score(ctx)


    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(self, text: str) -> list[RankingExplanation]:
        """
        Return per-suggestion ranking explanations.

        Notes:
            - Explanations are ranker-driven.
            - Predictor explanations are treated as upstream signal
              and may be incorporated by rankers if desired.
        """
        ctx = CompletionContext(text)
        scored = self._score(ctx)
        ranked = self._apply_ranking(ctx, scored)

        aggregated: dict[str, RankingExplanation] = {}

        for ranker in self._rankers:
            for exp in ranker.explain(ctx.text, ranked):
                if exp.value not in aggregated:
                    aggregated[exp.value] = exp
                else:
                    aggregated[exp.value] = aggregated[exp.value].merge(exp)

        return [
            aggregated[s.suggestion.value]
            for s in ranked
            if s.suggestion.value in aggregated
        ]

    def explain_as_dicts(self, text: str) -> list[dict[str, float | str]]:
        """
        Convenience adapter for CLI / serialization layers.
        """
        return [
            {
                "value": e.value,
                "base_score": e.base_score,
                "history_boost": e.history_boost,
                "final_score": e.final_score,
            }
            for e in self.explain(text)
        ]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def record_selection(self, text: str, value: str) -> None:
        """
        Record a user selection for learning.

        Predictors may optionally implement a `record(...)` hook.
        This is intentionally duck-typed to avoid forcing
        learning behavior on all predictors.
        """
        ctx = CompletionContext(text)
        self._history.record(ctx.text, value)

        for weighted in self._predictors:
            record = getattr(weighted.predictor, "record", None)
            if callable(record):
                record(ctx, value)

    # ------------------------------------------------------------------
    # Developer/debug API (INTENTIONALLY UNSTABLE)
    # ------------------------------------------------------------------

    def debug(self, text: str) -> DebugState:
        """
        Developer-only debug surface.

        NOT a stable API.
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


    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """
        Return a human-readable description of the engine configuration.

        Intended for:
        - CLI inspection
        - debugging
        - documentation
        """
        return {
            "predictors": [
                {
                    "name": wp.predictor.name,
                    "weight": wp.weight,
                }
                for wp in self._predictors
            ],
            "rankers": [
                r.__class__.__name__
                for r in self._rankers
            ],
            "history_enabled": self._history is not None,
        }

    # ------------------------------------------------------------------

    @property
    def history(self) -> History:
        """Return the engine's history source of truth."""
        return self._history
