"""
Shared scoring formula for distance-based predictors (SymSpell, Trigram, EditDistance).

    score = base_score / (1 + distance) * (1 + FREQ_WEIGHT * freq_score)

FREQ_WEIGHT=0.5 keeps distance dominant across buckets while giving frequency
meaningful separation within each bucket (1.5x spread). A maximally-frequent
distance-2 word ties but never beats a zero-frequency distance-1 word.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

# Frequency bonus weight applied within each edit-distance bucket.
# See module docstring for full derivation.
FREQ_WEIGHT: float = 0.5


def build_freq_scores(
    frequencies: Mapping[str, int] | None,
) -> dict[str, float]:
    """Pre-compute log-normalised frequency scores. Missing keys should be treated as 0.0."""
    if not frequencies:
        return {}
    max_freq = max(frequencies.values()) or 1
    return {
        word: math.log1p(freq) / math.log1p(max_freq)
        for word, freq in frequencies.items()
    }


def distance_score(
    base_score: float,
    distance: int,
    freq_score: float,
) -> float:
    """Combined distance + frequency score. freq_score=0.0 for unknown words."""
    return (base_score / (1 + distance)) * (1.0 + FREQ_WEIGHT * freq_score)


def edit_confidence(distance: int, max_distance: int) -> float:
    """1.0 for exact match, 0.0 at max_distance+1. Linear decay in between."""
    return max(0.0, 1.0 - (distance / (max_distance + 1)))
