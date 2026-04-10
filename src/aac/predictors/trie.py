from __future__ import annotations

from collections.abc import Iterable

from aac.domain.types import (
    CompletionContext,
    Predictor,
    PredictorExplanation,
    ScoredSuggestion,
    Suggestion,
    ensure_context,
)


class TrieNode:
    """
    A single node in the trie.

    Mutable by design - nodes accumulate children during construction.
    Not a frozen dataclass because the trie is built incrementally.
    """

    __slots__ = ("children", "is_terminal", "value")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_terminal: bool = False
        self.value: str | None = None


class Trie:
    def __init__(self, words: Iterable[str]) -> None:
        self._root = TrieNode()
        for word in words:
            self.insert(word)

    def insert(self, word: str) -> None:
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_terminal = True
        node.value = word

    def find_prefix(self, prefix: str, *, limit: int) -> list[str]:
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        results: list[str] = []
        self._collect(node, results, limit)
        return results

    def _collect(self, node: TrieNode, out: list[str], limit: int) -> None:
        if len(out) >= limit:
            return

        if node.is_terminal and node.value is not None:
            out.append(node.value)

        for key in sorted(node.children):
            self._collect(node.children[key], out, limit)
            if len(out) >= limit:
                return


class TriePrefixPredictor(Predictor):
    """
    Prefix predictor backed by a trie for O(prefix_length) lookup.

    Suitable when you have a word list but no frequency data. Scores
    all matches equally at 1.0 - combine with a history or frequency
    predictor for score differentiation. For use cases where frequency
    data is available, FrequencyPredictor builds its own prefix index
    and carries per-word scores through the pipeline.
    """

    name = "trie_prefix"

    def __init__(self, words: Iterable[str], *, max_results: int = 10) -> None:
        self._trie = Trie(words)
        self._max_results = max_results

    def predict(self, ctx: CompletionContext | str) -> list[ScoredSuggestion]:
        ctx = ensure_context(ctx)
        prefix = ctx.prefix()

        if not prefix:
            return []

        matches = self._trie.find_prefix(prefix, limit=self._max_results)
        results: list[ScoredSuggestion] = []

        for word in matches:
            if word == prefix:
                continue

            results.append(
                ScoredSuggestion(
                    suggestion=Suggestion(value=word),
                    score=1.0,
                    explanation=PredictorExplanation(
                        value=word,
                        score=1.0,
                        confidence=1.0,
                        source=self.name,
                    ),
                )
            )

        return results