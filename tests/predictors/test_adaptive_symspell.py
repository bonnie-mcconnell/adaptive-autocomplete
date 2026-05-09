"""
Tests for AdaptiveSymSpellPredictor.

The adaptive predictor dispatches to two internal SymSpell indexes based on
prefix length: a tight index (max_distance=1) for short prefixes and a full
index (max_distance=N) for longer ones.  This prevents hundreds of noise
candidates on 1-3 char queries while preserving full typo recovery for 4+ chars.
"""
from __future__ import annotations

from aac.domain.types import CompletionContext
from aac.predictors.adaptive_symspell import AdaptiveSymSpellPredictor

_VOCAB = [
    "hello", "help", "hero", "her", "here", "heap", "world", "word",
    "programming", "program", "the", "they", "them", "then", "there",
]


class TestAdaptiveSymSpellPredictor:
    def _predictor(self, **kwargs: object) -> AdaptiveSymSpellPredictor:
        return AdaptiveSymSpellPredictor(_VOCAB, **kwargs)  # type: ignore[arg-type]

    def test_exact_match_excluded(self) -> None:
        """SymSpell excludes word==prefix: completing to itself is noise."""
        results = [s.suggestion.value for s in self._predictor().predict(CompletionContext("hello"))]
        assert "hello" not in results, "Exact match must be excluded"
        # Distance-2 neighbours must be present
        assert "help" in results or "hero" in results, (
            "Distance-2 neighbours of 'hello' ('help', 'hero') must be returned"
        )

    def test_long_prefix_recovers_distance_2_typo(self) -> None:
        """4+ char prefix: full max_distance used."""
        results = [s.suggestion.value for s in self._predictor(max_distance=2, short_prefix_len=4).predict(CompletionContext("helo"))]
        assert "hello" in results

    def test_short_prefix_uses_tight_distance(self) -> None:
        """1-3 char prefix: max_distance clamped to short_max_distance."""
        adaptive = self._predictor(max_distance=2, short_prefix_len=4, short_max_distance=1)
        full = AdaptiveSymSpellPredictor(_VOCAB, max_distance=2, short_prefix_len=100)

        adaptive_results = {s.suggestion.value for s in adaptive.predict(CompletionContext("he"))}
        full_results = {s.suggestion.value for s in full.predict(CompletionContext("he"))}

        assert len(adaptive_results) <= len(full_results), (
            "adaptive should return no more candidates than full on short prefix"
        )

    def test_short_prefix_still_finds_distance_1_typo(self) -> None:
        """Single-char typo on short prefix must still be recovered."""
        p = self._predictor(max_distance=2, short_prefix_len=4, short_max_distance=1)
        # "hdr" is distance 1 from "her" (substitute d→e)
        results = [s.suggestion.value for s in p.predict(CompletionContext("hdr"))]
        assert "her" in results, f"expected 'her' (distance=1 from 'hdr') in {results}"

    def test_exact_match_excluded_on_short_prefix(self) -> None:
        """word==prefix is excluded even on short prefixes."""
        p = self._predictor(max_distance=2, short_prefix_len=4, short_max_distance=1)
        results = [s.suggestion.value for s in p.predict(CompletionContext("her"))]
        assert "her" not in results, "Exact match must be excluded even on short prefixes"
        assert "here" in results or "hero" in results, (
            "Distance-1 neighbours of 'her' must be returned by tight index"
        )

    def test_empty_prefix_returns_empty(self) -> None:
        assert self._predictor().predict(CompletionContext("")) == []

    def test_no_candidates_returns_empty(self) -> None:
        assert self._predictor(max_distance=1).predict(CompletionContext("zzzzz")) == []

    def test_name_is_symspell(self) -> None:
        """Name must be 'symspell' (not 'adaptive_symspell') for EngineConfig
        serialisation consistency."""
        assert self._predictor().name == "symspell"

    def test_same_distance_avoids_duplicate_index(self) -> None:
        """When max_distance == short_max_distance, only one index is built."""
        p = AdaptiveSymSpellPredictor(_VOCAB, max_distance=1, short_prefix_len=4, short_max_distance=1)
        assert p._inner_tight is p._inner_full

    def test_threshold_boundary_uses_full_index(self) -> None:
        """Queries at exactly short_prefix_len chars must use the full index."""
        p = self._predictor(max_distance=2, short_prefix_len=4, short_max_distance=1)
        # "hell" is 4 chars - exactly at threshold, uses full distance=2
        results = [s.suggestion.value for s in p.predict(CompletionContext("hell"))]
        assert "hello" in results

    def test_scores_are_positive(self) -> None:
        for s in self._predictor().predict(CompletionContext("hello")):
            assert s.score > 0, f"score must be positive, got {s.score}"
