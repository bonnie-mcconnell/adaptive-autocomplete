"""
Shared scoring constants and helpers for distance-based predictors.

All three approximate-match predictors (SymSpell, Trigram, EditDistance) use
the same scoring formula so their scores are directly comparable when combined
in a weighted predictor stack.

Formula
-------
    score = base_score / (1 + distance) * (1 + FREQ_WEIGHT * freq_score)

where:
    distance   = Levenshtein edit distance between query and candidate
    freq_score = log(1 + freq) / log(1 + max_freq)  ∈ (0, 1]
    FREQ_WEIGHT = 0.5

Design rationale for FREQ_WEIGHT = 0.5
---------------------------------------
Distance determines the ordering of buckets; frequency orders within buckets.
The weight is set so that:

- Within the same distance bucket: the highest-frequency word scores 1.5x
  the lowest-frequency word (50% spread - meaningful separation).
- Across bucket boundaries: a maximally-frequent distance-2 word ties but
  does not beat a zero-frequency distance-1 word:
      dist-1, freq=0:   (1/2) * (1 + 0.5 * 0) = 0.500
      dist-2, freq=max: (1/3) * (1 + 0.5 * 1) = 0.500  ← tied at the extreme
  In practice distance-1 words have non-zero frequency and always lead.

This keeps distance strictly dominant while giving frequency real power
within each bucket. The old additive formula capped frequency at 10% of
the distance score, making it nearly invisible in practice.
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
    """
    Pre-compute log-normalised frequency scores from a raw frequency mapping.

    Returns an empty dict if ``frequencies`` is None or empty - callers
    must treat a missing key as score 0.0.

    The log normalisation compresses the frequency range so that extremely
    common words (''the'': 537k) don't completely dominate rare ones (''tea'': 537).
    log(1 + 537000) / log(1 + 537) ≈ 2.9, which is then divided by
    log(1 + max_freq) to bring everything into [0, 1].
    """
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
    """
    Compute the combined distance + frequency score for a candidate.

    Parameters:
        base_score:  Predictor's base weight (typically 1.0).
        distance:    Levenshtein edit distance (0 = exact match).
        freq_score:  Pre-computed log-normalised frequency in [0, 1].
                     Use 0.0 for words not in the frequency table.
    """
    return (base_score / (1 + distance)) * (1.0 + FREQ_WEIGHT * freq_score)


def edit_confidence(distance: int, max_distance: int) -> float:
    """
    Confidence score for a candidate at given edit distance.

    Returns 1.0 for exact matches (distance=0), 0.0 for distance=max_distance+1.
    Used to populate PredictorExplanation.confidence.
    """
    return max(0.0, 1.0 - (distance / (max_distance + 1)))
