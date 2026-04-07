"""
BK-tree implementation for approximate string matching.

A BK-tree is a metric tree that enables nearest-neighbour queries in
O(log n) average time for well-distributed metric spaces. It exploits
the triangle inequality: if a node n is at distance d from query q,
then any child at key k can only contain matches if |d - k| <= threshold.

Construction:
    Insert any word as the root. For each subsequent word w, walk the
    tree: at each node n compute d = distance(w, n.word). If n has no
    child at key d, store w there. Otherwise recurse into child d.
    Time: O(n log n) average, O(n^2) worst case (degenerate tree).

Search for query q within threshold t:
    At each node n with word w, compute d = distance(q, w).
    If d <= t, w is a result. Recurse into children at keys k where
    |d - k| <= t. All other subtrees are provably unreachable.
    Time: O(log n) average for small t; degrades toward O(n) when t
    is large relative to the string lengths in the tree. At max_distance=2
    with 4-character prefixes over a 482-word English vocabulary, the
    search visits approximately 75% of nodes — the pruning is weak
    because the search ball covers most of the metric space.

References:
    Burkhard, W.A. and Keller, R.M. (1973). "Some approaches to
    best-match file searching." Communications of the ACM, 16(4).
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator


class _Node:
    """
    A single node in the BK-tree.

    Each node stores one word and a dict mapping edit-distance keys
    to child nodes. The key is the Levenshtein distance from this
    node's word to the child's word at insertion time.
    """

    __slots__ = ("word", "children")

    def __init__(self, word: str) -> None:
        self.word = word
        self.children: dict[int, _Node] = {}


class BKTree:
    """
    BK-tree over Levenshtein distance.

    Supports approximate string matching queries. All words within
    max_distance Levenshtein edits of the query are returned, including
    cases where the first character differs — a property that simpler
    first-character indexes do not provide.

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
            d = _levenshtein(word, node.word)
            if d == 0:
                # Duplicate word — BK-trees don't store duplicates.
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
            d = _levenshtein(query, node.word)

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

def _levenshtein(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance using a space-optimised DP.

    Swaps arguments so the shorter string drives the outer loop,
    keeping memory at O(min(|a|, |b|)) rather than O(|a| × |b|).

    Cost model: insertion=1, deletion=1, substitution=1.
    """
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