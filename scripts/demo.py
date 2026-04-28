"""
End-to-end demonstration of adaptive-autocomplete.

Shows four things:
  1. Frequency-ranked suggestions from the full 48k vocabulary
  2. Per-suggestion score explanations
  3. Learning: a word selected 5 times rises from last to first
  4. Typo recovery: 'programing' -> 'programming'

The learning demo uses a small controlled vocabulary so the score movement
is visible after five selections. With the full 48k vocabulary, a single
selection already lifts a word above most frequency-only candidates -
log-normalised scores keep all predictors in the same (0, 1] scale -
but the visual jump is less dramatic in a ranked list of 48k words.
"""
from __future__ import annotations

from aac.domain.history import History
from aac.domain.types import WeightedPredictor
from aac.engine.engine import AutocompleteEngine
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.presets import create_engine
from aac.ranking.decay import DecayFunction, DecayRanker
from aac.ranking.score import ScoreRanker

_W = 72  # display width


def _header(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, _W - len(title) - 4)}")


def _section_1_frequency() -> None:
    _header("Frequency-ranked suggestions for 'he' (production preset, 48k vocab)")
    engine = create_engine("production")
    for s in engine.suggest("he"):
        print(f"  {s}")


def _section_2_explain() -> None:
    _header("Score breakdown for 'he' (recency preset, after selecting 'hello' twice)")
    # Use the recency preset with two history selections so the boost column
    # is non-zero for at least one row - otherwise score==base for every line
    # and the explain output demonstrates nothing about how the system works.
    history = History()
    engine = create_engine("recency", history=history)
    engine.record_selection("he", "hello")
    engine.record_selection("he", "hello")

    explanations = engine.explain("he")
    max_score = explanations[0].final_score if explanations else 1.0
    for exp in explanations:
        pct = exp.final_score / max_score * 100
        print(
            f"  {exp.value:<14s}"
            f"  score={exp.final_score:6.4f} ({pct:5.1f}%)"
            f"  base={exp.base_score:6.4f}"
            f"  boost={exp.history_boost:+.4f}"
        )


def _section_3_learning() -> None:
    _header("Learning: 'her' selected 5 times rises from last to first")

    # Small vocabulary so the frequency gap between words is narrow enough
    # that five selections visibly move 'her' past 'help'.
    # With DecayRanker(weight=100), five recent selections produce a boost
    # of ~500 - enough to clear the frequency gap between 'her' (lowest
    # frequency) and 'help' (highest frequency).
    vocab = {
        "help": 500,
        "hello": 400,
        "health": 300,
        "heard": 200,
        "hero": 100,
        "her": 50,
    }
    history = History()
    engine = AutocompleteEngine(
        predictors=[
            WeightedPredictor(FrequencyPredictor(vocab), 1.0),
            WeightedPredictor(HistoryPredictor(history), 1.0),
        ],
        ranker=[
            ScoreRanker(),
            # weight=100 so 5 selections produce ~500 boost, enough to
            # move 'her' (base score ≈0.74) past 'help' (base score ≈1.0).
            DecayRanker(history, DecayFunction(half_life_seconds=3600), weight=100.0),
        ],
        history=history,
    )

    before = engine.suggest("he")
    print(f"  before:              {before}")

    for _ in range(5):
        engine.record_selection("he", "her")

    after = engine.suggest("he")
    print(f"  after 5× 'her':      {after}")

    print()
    for exp in engine.explain("he"):
        boost_bar = "▓" * int(round(exp.history_boost) // 50)
        print(
            f"  {exp.value:<8s}"
            f"  base={exp.base_score:5.2f}"
            f"  boost={exp.history_boost:7.1f}"
            f"  final={exp.final_score:7.2f}"
            f"  {boost_bar}"
        )


def _section_4_typo() -> None:
    _header("Typo recovery: 'programing' → 'programming' (production preset)")
    engine = create_engine("production")
    results = engine.suggest("programing")
    print(f"  aac suggest programing  →  {results[:5]}")
    assert "programming" in results, "programming not found - typo recovery failed"


if __name__ == "__main__":
    print("adaptive-autocomplete demo")
    print("=" * _W)
    _section_1_frequency()
    _section_2_explain()
    _section_3_learning()
    _section_4_typo()
    print(f"\n{'─' * _W}")
    print("Done. Run 'make benchmark' for latency numbers.")
