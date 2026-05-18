"""RankingExplanation dataclass with enforced invariant: final_score == base_score + history_boost."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RankingExplanation:
    """
    Explains how a suggestion's final score was produced.

    Invariant: final_score == base_score + history_boost (enforced at construction).
    """

    value: str

    # Aggregated scores
    base_score: float
    history_boost: float
    final_score: float

    # Primary source (e.g. top-level ranker)
    source: str

    # Detailed breakdowns - always populated with all configured
    # predictors/rankers; 0.0 means "ran but didn't contribute".
    base_components: Mapping[str, float] = field(default_factory=dict)
    history_components: Mapping[str, float] = field(default_factory=dict)

    # Relative contribution of each source as a fraction of final_score.
    # {source: fraction} where fractions may not sum to exactly 1.0 due to
    # rounding.  Sources with zero contribution are omitted.
    # Useful for weight-tuning: "history is contributing 75% of the final
    # score - consider reducing its weight."
    contribution_pct: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = self.base_score + self.history_boost
        if abs(self.final_score - expected) > 1e-9:
            raise ValueError(
                "Invalid RankingExplanation: "
                f"final_score ({self.final_score}) "
                f"!= base_score + history_boost ({expected})"
            )

    def __repr__(self) -> str:
        return (
            f"RankingExplanation("
            f"value={self.value!r}, "
            f"base={self.base_score:.4f}, "
            f"boost={self.history_boost:+.4f}, "
            f"final={self.final_score:.4f}"
            f")"
        )

    def to_dict(self) -> dict[str, float | str | dict[str, float]]:
        """JSON-serialisable dict."""
        return asdict(self)

    def merge(self, other: RankingExplanation) -> RankingExplanation:
        """Merge explanations from two rankers into one. Raises if values differ."""
        if self.value != other.value:
            raise ValueError("Cannot merge explanations for different values")

        # 'source' from the first explanation is preserved.
        # base_components and history_components from both are merged so the
        # result reflects every ranker's contribution.

        merged_base = self.base_score + other.base_score
        merged_boost = self.history_boost + other.history_boost
        merged_final = merged_base + merged_boost

        merged_base_components: dict[str, float] = {
            **self.base_components,
            **other.base_components,
        }
        merged_history_components: dict[str, float] = {
            **self.history_components,
            **other.history_components,
        }

        # Recompute contribution_pct from merged components.
        # Use the same threshold and rounding as engine.explain().
        all_components = {**merged_base_components, **merged_history_components}
        if abs(merged_final) > 1e-12:
            contribution_pct: dict[str, float] = {
                k: round(v / merged_final, 4)
                for k, v in all_components.items()
                if abs(v) > 1e-12
            }
        else:
            contribution_pct = {}

        return RankingExplanation(
            value=self.value,
            base_score=merged_base,
            history_boost=merged_boost,
            final_score=merged_final,
            source=self.source,
            base_components=merged_base_components,
            history_components=merged_history_components,
            contribution_pct=contribution_pct,
        )

    @staticmethod
    def from_predictor(
        *,
        value: str,
        score: float,
        source: str,
    ) -> RankingExplanation:
        """Create a base explanation from a single predictor's score."""
        return RankingExplanation(
            value=value,
            base_score=score,
            history_boost=0.0,
            final_score=score,
            source=source,
            base_components={source: score},
            history_components={},
        )

    def apply_history_boost(
        self,
        *,
        boost: float,
        source: str,
    ) -> RankingExplanation:
        """Return a new explanation with boost added to history_boost."""
        new_history_boost = self.history_boost + boost
        return RankingExplanation(
            value=self.value,
            base_score=self.base_score,
            history_boost=new_history_boost,
            final_score=self.base_score + new_history_boost,
            source=self.source,
            base_components=dict(self.base_components),
            history_components={
                **self.history_components,
                source: self.history_components.get(source, 0.0) + boost,
            },
        )

    def short_summary(self) -> str:
        return (
            f"{self.value}: "
            f"base={self.base_score:.2f}, "
            f"history={self.history_boost:.2f}, "
            f"final={self.final_score:.2f}"
        )