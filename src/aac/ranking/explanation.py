from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RankingExplanation:
    """
    Explains how a final ranking score was produced for a suggestion.

    Scoring lifecycle:
    - Predictors contribute base score components
    - Ranking layers apply adjustments (e.g. learning)
    - Final score is derived deterministically

    Invariants:
    - final_score == base_score + history_boost
    """

    value: str

    # Aggregated scores
    base_score: float
    history_boost: float
    final_score: float

    # Primary source (e.g. top-level ranker)
    source: str

    # Optional detailed breakdowns
    base_components: Mapping[str, float] = field(default_factory=dict)
    history_components: Mapping[str, float] = field(default_factory=dict)

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
        """
        JSON-serializable representation.
        """
        return asdict(self)

    def merge(self, other: RankingExplanation) -> RankingExplanation:
        """
        Merge another explanation into this one.

        Intended for combining contributions from multiple rankers.

        Note on arithmetic: ``final_score`` is computed as
        ``(self.base_score + other.base_score) + (self.history_boost + other.history_boost)``
        rather than summing all four terms independently.  This matches the
        invariant check in ``__post_init__`` (``base + boost``) and avoids
        floating-point rounding errors where a four-way sum evaluates
        differently from a two-way sum of the same values.
        """
        if self.value != other.value:
            raise ValueError("Cannot merge explanations for different values")

        # NOTE:
        # 'source' intentionally preserved from the first explanation.
        # Component maps capture multi-ranker contributions.

        merged_base = self.base_score + other.base_score
        merged_boost = self.history_boost + other.history_boost

        return RankingExplanation(
            value=self.value,
            base_score=merged_base,
            history_boost=merged_boost,
            final_score=merged_base + merged_boost,
            source=self.source,
            base_components={
                **self.base_components,
                **other.base_components,
            },
            history_components={
                **self.history_components,
                **other.history_components,
            },
        )

    @staticmethod
    def from_predictor(
        *,
        value: str,
        score: float,
        source: str,
    ) -> RankingExplanation:
        """
        Create an explanation from a single predictor contribution.

        Used by rankers that surface a predictor's raw score as the base.
        Populates ``base_components`` with the source and score so that
        ``explain_as_dicts()`` can show a per-predictor breakdown.
        """
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
        """
        Return a new explanation with a history boost applied.

        Used by rankers that layer a recency or learning adjustment on top
        of an existing base explanation. Accumulates the boost into
        ``history_components`` under the named source so that
        ``explain_as_dicts()`` can show a per-ranker boost breakdown.

        Note on arithmetic: ``new_history_boost`` is computed first, then
        ``final_score = base_score + new_history_boost``.  This matches
        the invariant check in ``__post_init__`` and avoids floating-point
        rounding errors where a three-term sum evaluates differently from
        the two-term sum the invariant check uses.
        """
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