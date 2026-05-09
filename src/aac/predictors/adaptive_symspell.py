"""
Adaptive SymSpell predictor that tightens max_distance for short prefixes.

Problem
-------
SymSpell at max_distance=2 on a 2-3 char prefix like "he" matches almost
every 2-4 char word in a 48k-word vocabulary (edit distance 2 from "he"
covers "the", "hex", "hot", "a", "be", ...) - hundreds of candidates that
drown out the frequency signal and produce noisy suggestions.

This is not a bug in SymSpell. SymSpell is correct: at distance 2, "he"
genuinely has hundreds of neighbours. The problem is that max_distance=2
is too permissive for the first 1-3 keystrokes, where the user is still
narrowing their intent and typo recovery should be conservative.

Solution
--------
AdaptiveSymSpellPredictor builds two internal SymSpell indexes:

  - _inner_tight: max_distance=1, used when prefix length < short_prefix_len
  - _inner_full:  max_distance=max_distance, used for longer prefixes

At 1-3 chars, distance=1 is still generous (catches single-key typos) but
does not explode the candidate set. At 4+ chars, the full max_distance
kicks in for richer typo recovery.

The two indexes share no state - each is a fully independent SymSpellPredictor.
Memory overhead is roughly 2x a single SymSpell index: acceptable for server
deployment where SymSpell is chosen specifically because memory is not the
constraint.

The split threshold is configurable (short_prefix_len, default 4) so callers
can tune for their vocabulary and UX requirements.

This wrapper is used by the 'production' preset to give 'robust' preset
behaviour across all prefix lengths without the noise on short queries.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from aac.domain.types import (
    CompletionContext,
    Predictor,
    ScoredSuggestion,
    ensure_context,
)
from aac.predictors.symspell import SymSpellPredictor

_DEFAULT_MAX_DISTANCE = 2
_DEFAULT_SHORT_PREFIX_LEN = 4
_DEFAULT_SHORT_MAX_DISTANCE = 1


class AdaptiveSymSpellPredictor(Predictor):
    """
    SymSpell predictor that applies conservative edit distance on short prefixes.

    Delegates to two pre-built SymSpell indexes:
        - Tight index (max_distance=1): used for prefix length < short_prefix_len
        - Full index  (max_distance=N): used for prefix length >= short_prefix_len

    This eliminates the noise that SymSpell at max_distance=2 produces on
    1-3 char inputs, while preserving full typo recovery for longer prefixes.

    Parameters:
        vocabulary:          Iterable of words to index.
        max_distance:        Maximum edit distance for long prefixes. Default: 2.
        short_prefix_len:    Prefix length at which to switch from tight to full
                             distance. Default: 4. Prefixes shorter than this
                             value use short_max_distance.
        short_max_distance:  Edit distance for short prefixes. Default: 1.
        frequencies:         Optional word-frequency mapping for tiebreaking.
        base_score:          Base score passed to inner SymSpell instances.

    Example::

        predictor = AdaptiveSymSpellPredictor(
            vocabulary=word_list,
            max_distance=2,
            short_prefix_len=4,
        )
        predictor.predict("he")      # uses distance=1 (3 chars < 4)
        predictor.predict("hello")   # uses distance=2 (5 chars >= 4)
    """

    # The public name is "symspell" (not "adaptive_symspell") so that:
    # 1. explain() base_components shows "symspell" consistently, regardless
    #    of whether the engine uses SymSpellPredictor or AdaptiveSymSpellPredictor.
    # 2. EngineConfig serialises the predictor name as "symspell", matching
    #    the PredictorRegistry key used to reconstruct it.
    name = "symspell"

    def __init__(
        self,
        vocabulary: Iterable[str],
        *,
        max_distance: int = _DEFAULT_MAX_DISTANCE,
        short_prefix_len: int = _DEFAULT_SHORT_PREFIX_LEN,
        short_max_distance: int = _DEFAULT_SHORT_MAX_DISTANCE,
        frequencies: Mapping[str, int] | None = None,
        base_score: float = 1.0,
    ) -> None:
        self._short_prefix_len = short_prefix_len

        # Materialise once - both indexes need the same vocab list.
        vocab_list = list(vocabulary)

        self._inner_tight = SymSpellPredictor(
            vocab_list,
            max_distance=short_max_distance,
            frequencies=frequencies,
            base_score=base_score,
        )

        if max_distance == short_max_distance:
            # Avoid building an identical index twice.
            self._inner_full = self._inner_tight
        else:
            self._inner_full = SymSpellPredictor(
                vocab_list,
                max_distance=max_distance,
                frequencies=frequencies,
                base_score=base_score,
            )

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        # Delegate to the appropriate inner index based on prefix length.
        # SymSpellPredictor already excludes exact matches (word == prefix),
        # so no additional filtering is needed here.  Do NOT re-add the prefix
        # as a top-scoring exact match: completing a word to itself is noise,
        # not a suggestion.  The old version of this method explicitly inserted
        # the prefix at score=max(1.0, ...) which caused low-frequency words
        # like "programing" (freq=0 in corpus) to appear as the top suggestion
        # for their own query, burying high-frequency completions.
        inner = self._inner_tight if len(prefix) < self._short_prefix_len else self._inner_full
        return inner.predict(ctx)
