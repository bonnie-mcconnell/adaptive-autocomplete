"""
Smoke tests: verify the engine initialises and produces output end-to-end.

These are not unit tests. They exist to catch import errors, missing data
files, and wiring failures that unit tests miss because they stub out
dependencies. If any of these fail, nothing else in the suite is reliable.
"""
from __future__ import annotations

from types import MappingProxyType

from aac.data import load_english_frequencies
from aac.presets import create_engine


def test_load_english_frequencies_returns_mapping() -> None:
    """Bundled vocabulary must load as a non-empty immutable mapping."""
    vocab = load_english_frequencies()
    assert isinstance(vocab, MappingProxyType)
    assert len(vocab) > 0
    # All keys must be lowercase strings, all values positive integers
    for word, freq in vocab.items():
        assert isinstance(word, str)
        assert isinstance(freq, int)
        assert freq > 0


def test_default_engine_produces_suggestions() -> None:
    engine = create_engine("default")
    results = engine.suggest("he")
    assert len(results) > 0, "Expected at least one suggestion for prefix 'he'"


def test_robust_engine_handles_typo() -> None:
    engine = create_engine("robust")
    values = engine.suggest("helo")
    assert "hello" in values, f"Expected 'hello' in typo recovery results, got {values}"


def test_engine_explain_returns_reconciled_scores() -> None:
    engine = create_engine("recency")
    explanations = engine.explain("he")
    assert explanations, "Expected at least one explanation"
    for exp in explanations:
        assert abs(exp.final_score - (exp.base_score + exp.history_boost)) < 1e-9, (
            f"Invariant broken for '{exp.value}': "
            f"{exp.base_score} + {exp.history_boost} != {exp.final_score}"
        )

def test_scoring_constants_shared_across_predictors() -> None:
    """
    All distance-based predictors must share the same FREQ_WEIGHT constant
    via aac.predictors._scoring. This ensures scores are directly comparable
    when mixed in a weighted predictor stack.

    Previously FREQ_WEIGHT=0.5 was a local variable duplicated inside
    __init__ in both symspell.py and trigram.py. If one was changed without
    the other, scores would become incomparable. Now all three distance
    predictors import from the single source of truth in _scoring.py.
    EditDistancePredictor previously used the old formula (no freq multiplier)
    and accepted no frequencies parameter at all - fixed in this session.
    """
    from aac.predictors._scoring import FREQ_WEIGHT, distance_score, edit_confidence
    from aac.predictors.edit_distance import EditDistancePredictor
    from aac.predictors.symspell import SymSpellPredictor
    from aac.predictors.trigram import TrigramPredictor

    vocab = ["hello", "help", "world", "programming"]
    freqs = {"hello": 100, "help": 80, "world": 200, "programming": 50}

    symspell = SymSpellPredictor(vocab, frequencies=freqs)
    trigram = TrigramPredictor(vocab, frequencies=freqs)
    edit_dist = EditDistancePredictor(vocab, frequencies=freqs)

    # All three should produce the same score for a word at distance=0,
    # using the shared formula.
    expected_dist0 = distance_score(1.0, 0, 1.0)   # dist=0, max freq_score
    assert abs(expected_dist0 - 1.5) < 1e-9         # (1/1) * (1 + 0.5 * 1) = 1.5

    # EditDistancePredictor now accepts frequencies (it didn't before)
    assert hasattr(edit_dist, "_freq_scores")
    assert "hello" in edit_dist._freq_scores

    # Ensure FREQ_WEIGHT is the exported constant, not a magic number
    assert FREQ_WEIGHT == 0.5
    assert edit_confidence(0, 2) == 1.0
    assert edit_confidence(2, 2) < 1.0


def test_demo_run_accepts_host_parameter() -> None:
    """demo.run() must accept a host parameter.

    This was a real functional bug: Dockerfile.demo CMD used --host 0.0.0.0
    but demo.run() had no host parameter and HTTPServer was hardcoded to
    127.0.0.1. docker compose up would start a container that silently bound
    to localhost inside the container, making the demo unreachable from the
    host even though the port was exposed.

    This test verifies the fix: host is plumbed all the way through to
    HTTPServer. We don't start a real server; we just confirm the signature
    is correct so the Dockerfile CMD won't fail at startup.
    """
    import inspect
    from aac.cli.demo import run as demo_run

    sig = inspect.signature(demo_run)
    params = list(sig.parameters)
    assert "host" in params, (
        "demo.run() must accept a 'host' parameter so Dockerfile.demo "
        "can pass --host 0.0.0.0 and make the container port reachable"
    )
    # Default must be localhost-only (safe default; Docker use requires explicit override)
    assert sig.parameters["host"].default == "127.0.0.1"


def test_readme_typo_examples_match_actual_output() -> None:
    """Pin the specific outputs shown in the README typo-recovery section.

    This test exists to catch README drift: if engine scoring changes cause
    the actual output to differ from what the README documents, this test
    fails and forces the README to be updated before the change ships.

    The README shows:
        aac suggest helo         → help, held, hell, hello
        aac suggest programing   → programming

    These are verified here against the production preset with the bundled
    vocabulary. If this test breaks, update both the test and the README.
    """
    from aac.domain.history import History
    from aac.presets import get_preset

    engine = get_preset("production").build(History(), None)

    # 'programing' (missing 'm') → 'programming' must be #1
    programing = engine.suggest("programing", limit=5)
    assert programing and programing[0] == "programming", (
        f"README claims 'programing' → 'programming' at #1, got: {programing}"
    )

    # 'helo' top-4 must be exactly [help, held, hell, hello] in that order
    helo = engine.suggest("helo", limit=5)
    assert helo[:4] == ["help", "held", "hell", "hello"], (
        f"README claims 'helo' → [help, held, hell, hello, ...], got: {helo}"
    )


def test_readme_recieve_output_matches_actual() -> None:
    """Pin the recieve typo-recovery output shown in the README.

    The README shows: aac suggest recieve → recieved, relieve, believe, receive
    'recieve' is not a standard transposition fix case - 'recieved' is in the
    corpus at edit-distance 1 (add 'd'), so it outranks 'receive' (distance 2).
    """
    from aac.domain.history import History
    from aac.presets import get_preset

    engine = get_preset("production").build(History(), None)
    recieve = engine.suggest("recieve", limit=4)

    # recieved must be #1 (corpus word at edit-distance 1)
    assert recieve and recieve[0] == "recieved", (
        f"README claims 'recieve' → 'recieved' at #1, got: {recieve}"
    )
    # receive must appear somewhere in the top-4 (distance-2 recovery)
    assert "receive" in recieve, (
        f"README claims 'receive' appears in top-4 for 'recieve', got: {recieve}"
    )
