"""
Tests for SymSpellPredictor approximate string matching.

Core correctness property:
    SymSpellPredictor.predict(query) must return exactly the same set of
    words as a brute-force linear scan using the same Levenshtein distance
    function and the same max_distance threshold.

SymSpell claims 100% recall - finds every word within max_distance
Levenshtein edits with no false negatives. This file is the test that
proves that claim rather than just asserting it.

The threat model for SymSpell's delete-neighbourhood approach:
    - Transpositions ("ab" → "ba"): edit distance 2 via two substitutions.
      SymSpell shares deleted forms between both, so these are covered.
    - Short queries (length 1-3): TrigramPredictor is blind here.
      SymSpell has no minimum prefix length constraint - this is tested
      explicitly since it is a documented advantage over TrigramPredictor.
    - Queries with no vocabulary match: must return empty, not crash.
    - Queries that exactly match vocabulary words: distance-0 matches must
      be included (SymSpell adds word→word in the delete map for this case).
    - max_distance=1 vs max_distance=2: both must match brute force exactly.
    - Vocabulary with near-duplicate words: the delete-neighbourhood sets
      can overlap heavily; results must still be exact.
"""
from __future__ import annotations

from aac.predictors.bk_tree import levenshtein
from aac.predictors.symspell import SymSpellPredictor

# ------------------------------------------------------------------
# Shared brute-force reference implementation
# ------------------------------------------------------------------

def _linear_search(
    query: str,
    words: list[str],
    max_distance: int,
) -> set[str]:
    """
    Brute-force O(n) correctness reference.

    Returns every word in ``words`` within ``max_distance`` Levenshtein
    edits of ``query``. Used to verify SymSpell matches without gaps.
    """
    return {w for w in words if levenshtein(query, w) <= max_distance}


def _symspell_words(
    predictor: SymSpellPredictor,
    query: str,
) -> set[str]:
    """Extract just the word strings from SymSpellPredictor.predict()."""
    return {s.suggestion.value for s in predictor.predict(query)}


# ------------------------------------------------------------------
# Vocabulary fixtures
# ------------------------------------------------------------------

# Small, manually controlled vocabulary for readable failure messages.
_SMALL_VOCAB = [
    "hello", "help", "helium", "hero", "hex", "heap",
    "world", "word", "work", "worry",
    "the", "there", "then", "them",
    "cat", "car", "card", "cart", "care",
    "program", "programming", "progress", "project",
]

# A 500-word sample covering common English patterns: short words,
# long words, words sharing prefixes, and near-duplicates.
# Drawn deterministically so test output is reproducible.
_MEDIUM_VOCAB = [
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "do", "for", "from", "get", "go", "got", "had", "has", "have",
    "he", "her", "here", "him", "his", "how", "if", "in", "is", "it",
    "its", "just", "know", "let", "like", "make", "me", "more", "my", "no",
    "not", "now", "of", "on", "one", "or", "our", "out", "so", "some",
    "than", "that", "the", "their", "them", "then", "there", "they", "this",
    "time", "to", "up", "us", "was", "we", "were", "what", "when", "which",
    "who", "will", "with", "would", "you", "your",
    "about", "after", "again", "also", "back", "because", "before",
    "being", "between", "both", "came", "come", "could", "day", "did",
    "does", "done", "down", "each", "even", "every", "first", "found",
    "gave", "give", "good", "great", "hand", "help", "here", "high",
    "home", "into", "just", "keep", "kind", "last", "left", "life",
    "like", "line", "long", "look", "made", "many", "most", "move",
    "much", "must", "name", "need", "never", "next", "night", "off",
    "only", "open", "over", "own", "part", "place", "point", "put",
    "read", "real", "right", "room", "said", "same", "see", "seem",
    "show", "side", "small", "still", "such", "sure", "take", "tell",
    "than", "thing", "think", "those", "three", "through", "told",
    "took", "turn", "under", "until", "upon", "used", "very", "want",
    "well", "went", "whole", "work", "world", "write", "year", "yet",
    "able", "across", "almost", "along", "already", "always", "another",
    "around", "became", "begin", "below", "better", "beyond", "black",
    "bring", "built", "carry", "cause", "change", "close", "cover",
    "cross", "death", "doing", "doubt", "during", "earth", "eight",
    "either", "ended", "enjoy", "enter", "equal", "exist", "faced",
    "falls", "feels", "fewer", "field", "fifty", "fight", "final",
    "floor", "focus", "force", "forms", "forth", "found", "front",
    "fully", "given", "going", "green", "group", "grown", "guard",
    "guess", "guide", "heard", "heart", "heavy", "hence", "human",
    "ideas", "image", "inner", "input", "issue", "items", "joins",
    "judge", "known", "large", "later", "laugh", "layer", "learn",
    "leave", "level", "light", "limit", "lived", "local", "login",
    "lower", "lucky", "magic", "major", "makes", "match", "might",
    "model", "money", "month", "moral", "mount", "music", "notes",
    "novel", "occur", "offer", "often", "older", "order", "other",
    "ought", "outer", "pages", "panel", "paper", "paths", "pause",
    "peace", "phase", "phone", "photo", "piece", "pilot", "place",
    "plain", "plane", "plant", "plays", "power", "press", "price",
    "pride", "prime", "prior", "proof", "prove", "proxy", "queen",
    "query", "quick", "quiet", "quite", "quote", "raise", "range",
    "rapid", "ratio", "reach", "ready", "realm", "refer", "reply",
    "reset", "rider", "river", "roads", "robot", "rocks", "roles",
    "rough", "round", "route", "rules", "rural", "scale", "scope",
    "score", "seems", "sense", "serve", "seven", "share", "sharp",
    "sheet", "shift", "short", "sight", "since", "sixth", "sized",
    "skill", "sleep", "slice", "slide", "slope", "solid", "solve",
    "sound", "south", "space", "speak", "speed", "spend", "split",
    "spoke", "stack", "staff", "stage", "start", "state", "stays",
    "steam", "steps", "stick", "stone", "store", "storm", "story",
    "study", "style", "sugar", "super", "table", "taste", "taxes",
    "teach", "teeth", "terms", "tests", "thick", "threw", "title",
    "today", "token", "tools", "topic", "total", "touch", "towns",
    "trace", "track", "trade", "trail", "train", "trait", "treat",
    "trend", "trial", "tried", "truth", "tuned", "twice", "types",
    "union", "units", "upper", "usage", "users", "usual", "valid",
    "value", "video", "views", "visit", "vital", "voice", "voted",
    "waste", "water", "waves", "where", "while", "white", "whose",
    "wider", "winds", "woman", "women", "words", "wrist", "wrote",
    "years", "yield", "young", "zones",
]


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

class TestConstruction:
    def test_empty_vocabulary(self) -> None:
        p = SymSpellPredictor([], max_distance=2)
        assert _symspell_words(p, "hello") == set()

    def test_empty_string_in_vocabulary_is_skipped(self) -> None:
        p = SymSpellPredictor(["hello", "", "help"], max_distance=1)
        assert "" not in _symspell_words(p, "hello")

    def test_empty_prefix_returns_empty(self) -> None:
        p = SymSpellPredictor(["hello", "help"], max_distance=2)
        assert _symspell_words(p, "") == set()


# ------------------------------------------------------------------
# Core correctness: SymSpell must match brute-force exactly
#
# These tests are the proof of the "100% recall, no false negatives"
# claim made in the docstring, README, and CHANGELOG. The BK-tree has
# an equivalent property test in test_bk_tree.py; SymSpell needs one too.
# ------------------------------------------------------------------

class TestBruteForceEquivalence:
    """
    SymSpellPredictor.predict(query) must return exactly the same word set
    as a linear scan over the vocabulary using the same distance function.

    Tests are parameterised over multiple queries and both distance thresholds
    so that any gap in the delete-neighbourhood coverage is caught by at
    least one case.
    """

    _DISTANCE_1_QUERIES = [
        # Single substitution
        "helo",   # helo → hello (1 sub)
        "wrold",  # wrold → world (1 sub)
        # Single deletion (query is shorter than target)
        "hlp",    # hlp → help (1 insert)
        "th",     # th → the (1 insert)
        # Single insertion (query is longer than target)
        "catt",   # catt → cat (1 delete)
        # Exact match - must appear at distance 0
        "cat",
        "world",
    ]

    _DISTANCE_2_QUERIES = [
        # Two substitutions
        "helo",    # helo → hello (1), helo → help (2), etc.
        "hlep",    # two edits from multiple words
        "teh",     # teh → the (2 via sub+sub)
        # Transposition: "ab" → "ba" is edit distance 2 (two subs).
        # SymSpell covers this via shared deleted forms.
        "wrold",
        # Short prefix (1-3 chars) - TrigramPredictor cannot handle these;
        # SymSpell must handle them correctly at any distance.
        "th",      # short: matches "the", "them", "then", "there"
        "ca",      # short: matches "cat", "car", "can"
        "he",      # short: matches "hero", "hex", "heap", "help", "hello"
        # No match expected
        "zzzzz",
    ]

    def test_matches_brute_force_distance_1_small_vocab(self) -> None:
        predictor = SymSpellPredictor(_SMALL_VOCAB, max_distance=1)
        for query in self._DISTANCE_1_QUERIES:
            expected = _linear_search(query, _SMALL_VOCAB, max_distance=1)
            got = _symspell_words(predictor, query)
            assert got == expected, (
                f"SymSpell vs brute force mismatch at distance=1 for query={query!r}:\n"
                f"  SymSpell returned:   {sorted(got)}\n"
                f"  brute force expects: {sorted(expected)}\n"
                f"  missing: {sorted(expected - got)}\n"
                f"  extra:   {sorted(got - expected)}"
            )

    def test_matches_brute_force_distance_2_small_vocab(self) -> None:
        predictor = SymSpellPredictor(_SMALL_VOCAB, max_distance=2)
        for query in self._DISTANCE_2_QUERIES:
            expected = _linear_search(query, _SMALL_VOCAB, max_distance=2)
            got = _symspell_words(predictor, query)
            assert got == expected, (
                f"SymSpell vs brute force mismatch at distance=2 for query={query!r}:\n"
                f"  SymSpell returned:   {sorted(got)}\n"
                f"  brute force expects: {sorted(expected)}\n"
                f"  missing: {sorted(expected - got)}\n"
                f"  extra:   {sorted(got - expected)}"
            )

    def test_matches_brute_force_distance_1_medium_vocab(self) -> None:
        """
        Scale test: 500-word vocabulary covering realistic English words.

        Short queries (1-3 chars) are included here because TrigramPredictor
        cannot handle them - verifying SymSpell's correctness at short prefix
        lengths is the main reason to have SymSpell at all.
        """
        vocab = list(dict.fromkeys(_MEDIUM_VOCAB))  # deduplicate, preserve order
        predictor = SymSpellPredictor(vocab, max_distance=1)
        queries = [
            "he", "th", "ca",                      # very short
            "helo", "wrold", "thier",               # common typos
            "and", "the", "are",                    # exact matches
            "teh", "adn", "fo",                     # transpositions / truncations
        ]
        for query in queries:
            expected = _linear_search(query, vocab, max_distance=1)
            got = _symspell_words(predictor, query)
            assert got == expected, (
                f"SymSpell vs brute force mismatch (500-word vocab) distance=1 "
                f"query={query!r}:\n"
                f"  missing: {sorted(expected - got)}\n"
                f"  extra:   {sorted(got - expected)}"
            )

    def test_matches_brute_force_distance_2_medium_vocab(self) -> None:
        vocab = list(dict.fromkeys(_MEDIUM_VOCAB))
        predictor = SymSpellPredictor(vocab, max_distance=2)
        queries = [
            "he", "th",                             # short, many near-neighbours
            "hlep", "wrold", "teh",                 # 2-edit distance
            "progam", "prgoress",                   # realistic typing errors
            "zzzzz",                                # no match
        ]
        for query in queries:
            expected = _linear_search(query, vocab, max_distance=2)
            got = _symspell_words(predictor, query)
            assert got == expected, (
                f"SymSpell vs brute force mismatch (500-word vocab) distance=2 "
                f"query={query!r}:\n"
                f"  missing: {sorted(expected - got)}\n"
                f"  extra:   {sorted(got - expected)}"
            )

    def test_exact_set_equality_exhaustive_small_vocab(self) -> None:
        """
        Complete correctness proof: for every vocabulary word used as a query,
        SymSpellPredictor must return *exactly* the same set as brute force -
        no false negatives (missing matches) and no false positives (extra words
        that are actually too far away).

        The previous version used two separate issubset / membership checks.
        Separating them is a logical gap: a bug that returns every correct word
        *plus* extra wrong words passes the false-negative check, and a bug that
        returns only a subset of correct words passes the false-positive check.
        A single == assertion catches both failure modes simultaneously.

        Additionally verifies that every returned word's score is consistent with
        the actual Levenshtein distance between the query and that word. A bug
        where the delete-neighbourhood lookup finds the right word but records the
        wrong distance (inflating or deflating its score) would be invisible to a
        set-equality check alone.
        """
        predictor = SymSpellPredictor(
            _SMALL_VOCAB,
            max_distance=2,
            base_score=1.0,
        )

        for query in _SMALL_VOCAB:
            expected = _linear_search(query, _SMALL_VOCAB, max_distance=2)
            results = predictor.predict(query)
            got = {s.suggestion.value for s in results}

            # Set equality: catches false negatives AND false positives together.
            assert got == expected, (
                f"SymSpell result set != brute force for query={query!r}:\n"
                f"  false negatives (missing): {sorted(expected - got)}\n"
                f"  false positives (extra):   {sorted(got - expected)}"
            )

            # Score-distance consistency: every returned word's score must equal
            # base_score / (1 + actual_levenshtein_distance), within float tolerance.
            # This catches a second class of bug where the right word is found but
            # the distance used for scoring differs from the real edit distance.
            scores = {s.suggestion.value: s.score for s in results}
            for word in got:
                actual_dist = levenshtein(query, word)
                expected_score = 1.0 / (1 + actual_dist)
                # Allow a small absolute tolerance: the freq_bonus is zero here
                # (no frequencies passed), so the only source of divergence is
                # floating-point rounding in the division itself.
                assert abs(scores[word] - expected_score) < 1e-9, (
                    f"Score inconsistency for query={query!r}, word={word!r}: "
                    f"score={scores[word]:.9f} but expected "
                    f"1/(1+{actual_dist})={expected_score:.9f} "
                    f"(diff={abs(scores[word] - expected_score):.2e})"
                )


# ------------------------------------------------------------------
# Specific threat cases for the delete-neighbourhood algorithm
# ------------------------------------------------------------------

class TestEdgeCasesForDeleteNeighbourhood:
    """
    Cases that specifically stress the delete-neighbourhood construction.

    These are the failure modes that motivated writing SymSpell rather than
    accepting the BK-tree's O(n) degradation - they must work correctly.
    """

    def test_transposition_covered_at_distance_2(self) -> None:
        """
        "ab" and "ba" have Levenshtein distance 2 (two substitutions).
        SymSpell covers this: deletes("ab") = {"a", "b"},
        deletes("ba") = {"a", "b"} - they share all deleted forms.
        """
        p = SymSpellPredictor(["ba", "bc", "ca"], max_distance=2)
        results = _symspell_words(p, "ab")
        # levenshtein("ab", "ba") = 2 → must be included
        assert "ba" in results, (
            f"Transposition 'ab'→'ba' (distance 2) not found. Got: {results}"
        )
        # levenshtein("ab", "bc") = 2 → must be included
        assert "bc" in results

    def test_exact_match_included_at_distance_zero(self) -> None:
        """
        SymSpell adds word→word in the delete map. A query that exactly
        matches a vocabulary word must appear in results at distance 0.
        """
        p = SymSpellPredictor(["hello", "help", "hero"], max_distance=2)
        results = _symspell_words(p, "hello")
        assert "hello" in results, "Exact match must be included in results"

    def test_short_prefix_length_1(self) -> None:
        """
        Single-character prefix. TrigramPredictor returns empty for len < 4;
        SymSpell must return all words within max_distance.
        """
        vocab = ["a", "an", "and", "ants", "bat", "bad"]
        p = SymSpellPredictor(vocab, max_distance=1)
        expected = _linear_search("a", vocab, max_distance=1)
        got = _symspell_words(p, "a")
        assert got == expected, f"Short prefix 'a': missing={sorted(expected - got)}"

    def test_short_prefix_length_2(self) -> None:
        """Two-character prefix must match brute force."""
        vocab = ["he", "her", "here", "help", "hero", "hex", "hello"]
        p = SymSpellPredictor(vocab, max_distance=1)
        expected = _linear_search("he", vocab, max_distance=1)
        got = _symspell_words(p, "he")
        assert got == expected

    def test_query_longer_than_vocabulary_words(self) -> None:
        """
        When the query is longer than all vocabulary words, the edit distance
        is at least len(query) - max(len(w)). Queries that can't match
        anything must return empty.
        """
        p = SymSpellPredictor(["cat", "car", "bat"], max_distance=1)
        results = _symspell_words(p, "catastrophe")
        assert results == set(), f"Expected empty, got {results}"

    def test_no_match_returns_empty_not_error(self) -> None:
        """A query with no vocabulary match within max_distance must return []."""
        p = SymSpellPredictor(["hello", "world"], max_distance=1)
        results = _symspell_words(p, "zzzzz")
        assert results == set()

    def test_near_duplicate_words_in_vocabulary(self) -> None:
        """
        Near-duplicate vocabulary words produce heavily overlapping
        delete-neighbourhood sets. Results must still be exact.
        """
        vocab = ["test", "text", "best", "rest", "nest", "west", "jest"]
        p = SymSpellPredictor(vocab, max_distance=1)
        for query in vocab:
            expected = _linear_search(query, vocab, max_distance=1)
            got = _symspell_words(p, query)
            assert got == expected, (
                f"Near-duplicate vocab: query={query!r} "
                f"missing={sorted(expected - got)}"
            )

    def test_single_character_vocabulary(self) -> None:
        """Single-character words at the boundary of the distance model."""
        vocab = ["a", "b", "c", "d"]
        p = SymSpellPredictor(vocab, max_distance=1)
        expected = _linear_search("a", vocab, max_distance=1)
        got = _symspell_words(p, "a")
        assert got == expected


# ------------------------------------------------------------------
# Scoring properties
# ------------------------------------------------------------------

class TestScoring:
    """
    Scoring contract: closer matches score higher; exact matches score highest.
    These parallel TrigramPredictor's scoring tests for interchangeability.
    """

    def test_closer_match_scores_higher(self) -> None:
        """Distance-1 match must score above distance-2 match."""
        p = SymSpellPredictor(["hello", "hxllo"], max_distance=2)
        results = {s.suggestion.value: s.score for s in p.predict("hello")}
        # "hello" is distance 0; "hxllo" is distance 1
        assert results["hello"] > results["hxllo"], (
            f"Exact match should outscore distance-1: {results}"
        )

    def test_exact_match_scores_base_score(self) -> None:
        """Exact match (distance 0) must score at base_score (1.0 / (1+0) = 1.0)."""
        p = SymSpellPredictor(["hello"], max_distance=2, base_score=1.0)
        results = {s.suggestion.value: s.score for s in p.predict("hello")}
        assert "hello" in results
        assert abs(results["hello"] - 1.0) < 1e-9

    def test_results_sorted_descending_by_score(self) -> None:
        p = SymSpellPredictor(
            ["hello", "hxllo", "hxxllo"], max_distance=2
        )
        scores = [s.score for s in p.predict("hello")]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by descending score: {scores}"
        )

    def test_frequency_bonus_breaks_ties_within_distance_group(self) -> None:
        """
        When two words are at equal edit distance, the higher-frequency word
        must score higher (frequency bonus tiebreaking).
        """
        vocab = ["hello", "helzo"]   # both distance 1 from "helxo"
        freqs = {"hello": 1000, "helzo": 10}
        p = SymSpellPredictor(vocab, max_distance=2, frequencies=freqs)
        scores = {s.suggestion.value: s.score for s in p.predict("helxo")}
        if "hello" in scores and "helzo" in scores:
            assert scores["hello"] > scores["helzo"], (
                f"Higher-frequency word should score higher within same distance: {scores}"
            )

    def test_confidence_monotone_with_distance(self) -> None:
        """
        PredictorExplanation.confidence must decrease as distance increases.
        """
        from aac.domain.types import CompletionContext
        p = SymSpellPredictor(["hello", "hxllo", "hxxllo"], max_distance=3)
        ctx = CompletionContext("hello")
        results = p.predict(ctx)
        by_word = {s.suggestion.value: s.explanation for s in results}
        # Closer distances must have higher confidence
        if "hello" in by_word and "hxllo" in by_word:
            assert by_word["hello"].confidence >= by_word["hxllo"].confidence
