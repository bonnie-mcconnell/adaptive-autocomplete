"""
Critical invariant: explain() must return suggestions in the same order
as suggest() for the same input.

The most important behavioural guarantee of the engine that
had no direct test. If explain() and suggest() diverge in ordering,
the explanation shown to a user describes a *different* ranking than
the one they actually see. That makes explain() useless for debugging.

The invariant is tested across:
  - All available presets (stateless, default, recency, production, robust)
  - Multiple prefix lengths (1–7 chars)
  - Prefixes with and without history
  - Prefixes that trigger typo recovery
"""
from __future__ import annotations

import pytest

from aac.domain.history import History
from aac.presets import available_presets, create_engine

_PREFIXES = [
    "t",          # 1 char - broad, many candidates
    "he",         # 2 chars - short prefix, AdaptiveSymSpell tight mode
    "hel",        # 3 chars - still tight mode
    "prog",       # 4 chars - switches to full SymSpell distance
    "programing", # typo - tests recovery path
    "recieve",    # transposition typo
    "wh",         # prefix for common words (what, when, where, ...)
    "python",     # specific word
]


@pytest.mark.parametrize("preset_name", available_presets())
@pytest.mark.parametrize("prefix", _PREFIXES)
def test_explain_order_matches_suggest_no_history(preset_name: str, prefix: str) -> None:
    """Without history, explain() and suggest() must return the same order."""
    engine = create_engine(preset_name)
    suggest_order = engine.suggest(prefix)
    explain_order = [e.value for e in engine.explain(prefix)]
    assert suggest_order == explain_order, (
        f"Ordering mismatch for preset={preset_name!r}, prefix={prefix!r}.\n"
        f"  suggest: {suggest_order[:5]}\n"
        f"  explain: {explain_order[:5]}"
    )


@pytest.mark.parametrize("preset_name", available_presets())
@pytest.mark.parametrize("prefix", _PREFIXES[:5])  # subset is enough with history
def test_explain_order_matches_suggest_with_history(preset_name: str, prefix: str) -> None:
    """With history recorded, explain() and suggest() must still agree on order."""
    history = History()
    engine = create_engine(preset_name, history=history)

    # Seed some history so learning/decay rankers have signal to work with
    engine.record_selection(prefix, prefix + "oo")  # likely non-word → low freq
    engine.record_selection(prefix, prefix + "oo")
    engine.record_selection(prefix, prefix + "oo")

    suggest_order = engine.suggest(prefix)
    explain_order = [e.value for e in engine.explain(prefix)]
    assert suggest_order == explain_order, (
        f"Ordering mismatch with history for preset={preset_name!r}, prefix={prefix!r}.\n"
        f"  suggest: {suggest_order[:5]}\n"
        f"  explain: {explain_order[:5]}"
    )


def test_explain_order_matches_suggest_all_presets_programing() -> None:
    """
    Regression: typo recovery prefix must have consistent ordering.
    'programing' (distance-1 misspelling) exercises the SymSpell path
    and the production preset's AdaptiveSymSpell dispatch.
    """
    for preset_name in available_presets():
        engine = create_engine(preset_name)
        suggest_order = engine.suggest("programing")
        explain_order = [e.value for e in engine.explain("programing")]
        assert suggest_order == explain_order, (
            f"Typo ordering mismatch for preset={preset_name!r}.\n"
            f"  suggest: {suggest_order[:5]}\n"
            f"  explain: {explain_order[:5]}"
        )


def test_explain_order_stable_after_multiple_selections() -> None:
    """
    After many selections, explain() must still agree with suggest().
    This exercises the decay ranker's cache behaviour under accumulated history.
    """
    from aac.presets import create_engine

    engine = create_engine("production")
    for _ in range(10):
        engine.record_selection("he", "hello")
    for _ in range(5):
        engine.record_selection("he", "help")

    suggest_order = engine.suggest("he")
    explain_order = [e.value for e in engine.explain("he")]
    assert suggest_order == explain_order, (
        f"Ordering mismatch after accumulated history.\n"
        f"  suggest: {suggest_order[:5]}\n"
        f"  explain: {explain_order[:5]}"
    )


def test_explain_final_scores_match_ranked_order() -> None:
    """
    explain() final_score values must be non-increasing (highest-ranked = highest score).
    Verifies the explanation scores are consistent with the sort order.
    """
    engine = create_engine("production")
    engine.record_selection("prog", "programming")
    engine.record_selection("prog", "programming")

    explanations = engine.explain("prog")

    final_scores = [e.final_score for e in explanations]
    assert final_scores == sorted(final_scores, reverse=True), (
        f"explain() final_scores are not non-increasing: {final_scores[:5]}"
    )


def test_limit_does_not_affect_order() -> None:
    """
    suggest(text, limit=k) must return the first k items of suggest(text).
    Verifies the limit path doesn't re-sort or otherwise change order.
    """
    engine = create_engine("production")
    full = engine.suggest("pro")
    for k in (1, 3, 5):
        limited = engine.suggest("pro", limit=k)
        assert limited == full[:k], (
            f"suggest with limit={k} does not match full[:k].\n"
            f"  limited: {limited}\n"
            f"  full[:k]: {full[:k]}"
        )
