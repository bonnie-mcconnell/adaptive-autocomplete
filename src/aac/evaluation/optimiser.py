"""WeightOptimiser: grid search and coordinate descent over predictor weights."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aac.domain.history import History
    from aac.domain.types import WeightedPredictor
    from aac.engine.engine import AutocompleteEngine
    from aac.evaluation.harness import EvaluationHarness, EvaluationResult
    from aac.ranking.base import Ranker

_VALID_METRICS: frozenset[str] = frozenset(
    {"mrr", "ndcg", "precision", "recall", "ap", "hit_rate"}
)


@dataclass
class OptimisationResult:
    """Result of a weight optimisation run."""
    best_weights: dict[str, float]
    baseline_score: float
    best_score: float
    metric: str
    strategy: str
    history: list[tuple[dict[str, float], float]] = field(default_factory=list)
    n_evaluations: int = 0

    @property
    def improvement(self) -> float:
        """Absolute improvement in the metric."""
        return self.best_score - self.baseline_score

    @property
    def improvement_pct(self) -> float:
        """Relative improvement as a percentage."""
        if self.baseline_score == 0:
            return 0.0
        return (self.best_score - self.baseline_score) / self.baseline_score * 100

    def report(self) -> str:
        """Human-readable optimisation summary."""
        lines = [
            f"WeightOptimiser - {self.strategy}",
            f"Metric:       {self.metric}",
            f"Baseline:     {self.baseline_score:.4f}",
            f"Optimised:    {self.best_score:.4f}  "
            f"(+{self.improvement:.4f}, +{self.improvement_pct:.1f}%)",
            f"Evaluations:  {self.n_evaluations}",
            "Best weights:",
        ]
        if self.best_weights:
            for name, weight in sorted(self.best_weights.items()):
                lines.append(f"  {name:<24s}  {weight:.3f}")
        else:
            lines.append("  (no weights in search space changed from baseline)")
        return "\n".join(lines)


class WeightOptimiser:
    """
    Automated predictor weight tuning.

    Predictor indexes are built once at construction; only weights change
    between evaluations, so tuning the production preset costs ~2ms/eval
    rather than 10s.
    """

    def __init__(
        self,
        harness: EvaluationHarness,
        *,
        metric: str = "mrr",
        verbose: bool = True,
    ) -> None:
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Valid metrics: {sorted(_VALID_METRICS)}"
            )
        self._harness = harness
        self._metric = metric
        self._verbose = verbose
        # Cache of built WeightedPredictor lists keyed by preset name.
        # Predictors are expensive to build (SymSpell: ~10s); weights are not.
        self._predictor_cache: dict[str, list[WeightedPredictor]] = {}
        # Ranker template cache: preset name + ":rankers" -> Ranker list.
        self._ranker_cache: dict[str, list[Ranker]] = {}

    def _get_metric(self, result: EvaluationResult) -> float:
        """Extract the chosen metric from an EvaluationResult."""
        mapping: dict[str, float] = {
            "mrr": result.mean_mrr,
            "ndcg": result.mean_ndcg,
            "precision": result.mean_precision,
            "recall": result.mean_recall,
            "ap": result.mean_ap,
            "hit_rate": result.hit_rate,
        }
        return float(mapping[self._metric])

    def _get_base_weighted_predictors(self, base_preset: str) -> list[WeightedPredictor]:
        """Return cached predictors for a preset, building indexes on first call."""
        if base_preset not in self._predictor_cache:
            if self._verbose:
                print(f"Building {base_preset!r} preset indexes (once)...", end=" ", flush=True)
            from aac.domain.history import History
            from aac.presets import get_preset
            engine = get_preset(base_preset).build(History(), None)
            self._predictor_cache[base_preset] = list(engine._predictors)
            if self._verbose:
                print("done.")
        return self._predictor_cache[base_preset]

    def _get_cached_ranker_templates(self, base_preset: str) -> list[Ranker]:
        """Cached ranker templates for a preset. Re-instantiated with fresh history per evaluation."""
        if base_preset not in self._ranker_cache:
            from aac.domain.history import History
            from aac.presets import get_preset
            tmp_engine = get_preset(base_preset).build(History(), None)
            self._ranker_cache[base_preset] = list(tmp_engine._rankers)
        return self._ranker_cache[base_preset]

    def _rebuild_rankers_for_history(
        self,
        base_preset: str,
        history: History,
    ) -> list[Ranker]:
        """Bind ranker templates to a fresh History for one evaluation run."""
        from aac.ranking.decay import DecayFunction, DecayRanker
        from aac.ranking.learning import LearningRanker
        from aac.ranking.score import ScoreRanker

        template_rankers = self._get_cached_ranker_templates(base_preset)
        fresh_rankers: list[Ranker] = []

        for template in template_rankers:
            if isinstance(template, ScoreRanker):
                fresh_rankers.append(ScoreRanker())
            elif isinstance(template, DecayRanker):
                cfg = template.ranker_config()
                fresh_rankers.append(DecayRanker(
                    history=history,
                    decay=DecayFunction(
                        half_life_seconds=cfg["half_life_seconds"]
                    ),
                    weight=cfg["weight"],
                ))
            elif isinstance(template, LearningRanker):  # pragma: no cover
                # No current built-in preset uses LearningRanker as its ranker;
                # all presets use ScoreRanker + optional DecayRanker. This branch
                # is retained for forward-compatibility: if a future preset adds
                # LearningRanker, it will work correctly without a code change.
                cfg = template.ranker_config()
                fresh_rankers.append(LearningRanker(
                    history=history,
                    boost=cfg["boost"],
                    dominance_ratio=cfg["dominance_ratio"],
                ))
            else:  # pragma: no cover
                # Unknown ranker type - fall back to reusing the template.
                # Stateless rankers (e.g. custom scoring functions) are safe
                # to share across evaluations; History-bound ones will read
                # stale data, but users of custom rankers accept this risk.
                fresh_rankers.append(template)

        return fresh_rankers

    def _build_engine_with_weights(
        self,
        base_preset: str,
        weights: dict[str, float],
    ) -> AutocompleteEngine:
        """Build an engine with cached indexes but new weights (~2ms vs 5s for production)."""
        from aac.domain.history import History
        from aac.domain.types import WeightedPredictor
        from aac.engine.engine import AutocompleteEngine

        base_wps = self._get_base_weighted_predictors(base_preset)

        new_predictors: list[WeightedPredictor] = []
        for wp in base_wps:
            name = wp.predictor.name
            new_weight = weights.get(name, wp.weight)
            new_predictors.append(WeightedPredictor(
                predictor=wp.predictor,
                weight=new_weight,
            ))

        # A fresh History per evaluation prevents history state from bleeding
        # across evaluations. Rankers must share the same History instance as
        # the engine - AutocompleteEngine enforces this at construction time.
        shared_history = History()
        fresh_rankers = self._rebuild_rankers_for_history(base_preset, shared_history)

        return AutocompleteEngine(
            predictors=new_predictors,
            ranker=fresh_rankers or None,
            history=shared_history,
        )

    def _get_baseline_weights(self, base_preset: str) -> dict[str, float]:
        """Return the preset's default weight for each predictor."""
        base_wps = self._get_base_weighted_predictors(base_preset)
        return {wp.predictor.name: wp.weight for wp in base_wps}

    def _evaluate(self, engine: AutocompleteEngine) -> float:
        """Run the harness and return the chosen metric value."""
        result = self._harness.run(engine)
        return self._get_metric(result)

    def grid_search(
        self,
        base_preset: str,
        weight_grid: dict[str, list[float]],
    ) -> OptimisationResult:
        """Exhaustive search over all weight combinations. Returns the best OptimisationResult."""
        baseline_weights = self._get_baseline_weights(base_preset)
        baseline_engine = self._build_engine_with_weights(base_preset, baseline_weights)
        baseline_score = self._evaluate(baseline_engine)

        total = 1
        for v in weight_grid.values():
            total *= len(v)

        if self._verbose:
            print(
                f"Grid search: {total} combinations × {self._harness.n_queries} queries "
                f"(indexes cached, ~{total * self._harness.n_queries * 0.001:.1f}s estimated)"
            )
            print(f"Baseline {self._metric}={baseline_score:.4f}")

        predictor_names = list(weight_grid.keys())
        weight_lists = [weight_grid[name] for name in predictor_names]

        best_weights: dict[str, float] = {**baseline_weights}
        best_score = baseline_score
        run_history: list[tuple[dict[str, float], float]] = []
        n_evals = 1  # baseline

        for combo in itertools.product(*weight_lists):
            trial_weights = {**baseline_weights, **dict(zip(predictor_names, combo, strict=True))}
            engine = self._build_engine_with_weights(base_preset, trial_weights)
            score = self._evaluate(engine)
            run_history.append((trial_weights.copy(), score))
            n_evals += 1

            if score > best_score:
                best_score = score
                best_weights = trial_weights.copy()
                if self._verbose:
                    changed = {k: v for k, v in trial_weights.items() if k in weight_grid}
                    print(f"  New best: {self._metric}={score:.4f}  weights={changed}")

        if self._verbose:
            changed_best = {k: v for k, v in best_weights.items() if k in weight_grid}
            print(f"Done. Best: {self._metric}={best_score:.4f}  weights={changed_best}")

        return OptimisationResult(
            best_weights=best_weights,
            baseline_score=baseline_score,
            best_score=best_score,
            metric=self._metric,
            strategy="grid_search",
            history=run_history,
            n_evaluations=n_evals,
        )

    def coordinate_descent(
        self,
        base_preset: str,
        weight_grid: dict[str, list[float]],
        *,
        max_rounds: int = 5,
    ) -> OptimisationResult:
        """
        Coordinate descent over predictor weights.

        Tunes one predictor at a time, cycling until convergence or max_rounds.
            ``best_weights`` always contains the full weight dict.
        """
        baseline_weights = self._get_baseline_weights(base_preset)
        baseline_engine = self._build_engine_with_weights(base_preset, baseline_weights)
        baseline_score = self._evaluate(baseline_engine)

        # Start from the middle value of each grid - a reasonable prior for
        # the initial search point when the user hasn't specified one.
        current_weights: dict[str, float] = {
            **baseline_weights,
            **{name: vals[len(vals) // 2] for name, vals in weight_grid.items()},
        }

        best_score = baseline_score
        run_history: list[tuple[dict[str, float], float]] = []
        n_evals = 1

        if self._verbose:
            total_evals = max_rounds * sum(len(v) for v in weight_grid.values())
            print(
                f"Coordinate descent: {max_rounds} rounds, {len(weight_grid)} predictors, "
                f"~{total_evals} evaluations max  ({self._harness.n_queries} queries each)"
            )
            print(f"Baseline {self._metric}={baseline_score:.4f}")

        for round_num in range(max_rounds):
            improved = False

            for predictor_name, candidate_weights in weight_grid.items():
                round_best_weight = current_weights[predictor_name]
                round_best_score = best_score

                for weight in candidate_weights:
                    trial_weights = {**current_weights, predictor_name: weight}
                    engine = self._build_engine_with_weights(base_preset, trial_weights)
                    score = self._evaluate(engine)
                    run_history.append((trial_weights.copy(), score))
                    n_evals += 1

                    if score > round_best_score:
                        round_best_score = score
                        round_best_weight = weight

                if abs(round_best_weight - current_weights[predictor_name]) > 1e-9:
                    current_weights[predictor_name] = round_best_weight
                    best_score = round_best_score
                    improved = True
                    if self._verbose:
                        print(
                            f"  Round {round_num+1} [{predictor_name}]: "
                            f"{self._metric}={best_score:.4f}  "
                            f"weight={round_best_weight:.3f}"
                        )

            if not improved:
                if self._verbose:
                    print(f"  Round {round_num+1}: converged (no improvement)")
                break

        if self._verbose:
            changed = {k: v for k, v in current_weights.items() if k in weight_grid}
            print(f"Done. Best: {self._metric}={best_score:.4f}  tuned_weights={changed}")

        return OptimisationResult(
            best_weights=current_weights,
            baseline_score=baseline_score,
            best_score=best_score,
            metric=self._metric,
            strategy="coordinate_descent",
            history=run_history,
            n_evaluations=n_evals,
        )
