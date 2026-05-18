"""
BK-tree for approximate string matching via Levenshtein distance.

Exploits the triangle inequality to prune subtrees during search.
O(log n) average; degrades toward O(n) when max_distance is large
relative to query length. See EditDistancePredictor for the predictor
wrapper. Ref: Burkhard & Keller (1973), CACM 16(4).
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator


class _Node:
    """One node: a word and children keyed by Levenshtein distance at insertion time."""

    __slots__ = ("word", "children")

    def __init__(self, word: str) -> None:
        self.word = word
        self.children: dict[int, _Node] = {}


class BKTree:
    """
    BK-tree for approximate string matching. Returns all words within
    max_distance Levenshtein edits of a query, including first-char differences.

    Usage:
        tree = BKTree(words)
        results = list(tree.search("helo", max_distance=2))
        # [("hello", 1), ("help", 1), ("hero", 1), ...]
    """

    def __init__(self, words: Iterable[str]) -> None:
        self._root: _Node | None = None
        self._size = 0

        for word in words:
            self._insert(word)

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._root is not None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _insert(self, word: str) -> None:
        if not word:
            return

        if self._root is None:
            self._root = _Node(word)
            self._size += 1
            return

        node = self._root
        while True:
            d = levenshtein(word, node.word)
            if d == 0:
                # Duplicate word - BK-trees don't store duplicates.
                return
            if d not in node.children:
                node.children[d] = _Node(word)
                self._size += 1
                return
            node = node.children[d]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_distance: int,
    ) -> Iterator[tuple[str, int]]:
        """
        Yield (word, distance) pairs within max_distance of query.

        Results are yielded in tree traversal order, not sorted by
        distance. Callers that need sorted results should sort the output.

        Args:
            query: The string to search for.
            max_distance: Maximum Levenshtein distance to include.

        Yields:
            Tuples of (word, distance) for each match.
        """
        if self._root is None:
            return

        # Iterative DFS using an explicit stack. A recursive approach
        # risks stack overflow on degenerate trees where every word is
        # inserted at the same edit distance from all ancestors.
        stack: list[_Node] = [self._root]

        while stack:
            node = stack.pop()
            d = levenshtein(query, node.word)

            if d <= max_distance:
                yield node.word, d

            # Triangle inequality pruning: only recurse into children
            # at keys k where |d - k| <= max_distance.
            lo = d - max_distance
            hi = d + max_distance

            for key, child in node.children.items():
                if lo <= key <= hi:
                    stack.append(child)


# ------------------------------------------------------------------
# Levenshtein distance
# ------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance using a space-optimised DP.

    Swaps arguments so the shorter string drives the outer loop,
    keeping memory at O(min(|a|, |b|)) rather than O(|a| x |b|).

    Cost model: insertion=1, deletion=1, substitution=1.
    """
    # Identical strings - O(1) check saves full DP on exact cache hits
    # and on BK-tree duplicate detection during index construction.
    if a == b:
        return 0

    if len(a) > len(b):
        a, b = b, a

    if not a:
        return len(b)

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,        # deletion
                    curr[j - 1] + 1,    # insertion
                    prev[j - 1] + cost  # substitution
                )
            )
        prev = curr

    return prev[-1]


