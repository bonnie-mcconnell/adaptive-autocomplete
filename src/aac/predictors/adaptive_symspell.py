"""AdaptiveSymSpellPredictor: tighter edit distance on short prefixes to reduce noise."""
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
    SymSpell with tighter edit distance (max_distance=1) on short prefixes,
    switching to full max_distance at short_prefix_len chars (default 4).

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
