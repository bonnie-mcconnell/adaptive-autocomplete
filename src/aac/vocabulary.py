"""Utilities for building frequency vocabularies from word lists and text corpora."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


def vocabulary_from_wordlist(
    words: Iterable[str],
    *,
    default_frequency: int = 1,
    strip: bool = True,
    skip_empty: bool = True,
) -> dict[str, int]:
    """
    Build a {word: frequency} dict from a word list. All words get default_frequency (default 1).

    Raise default_frequency (e.g. 1000) to mix domain terms with the bundled English vocab.

    Example::

        vocab = vocabulary_from_wordlist(Path("commands.txt").read_text().splitlines())
    """
    if default_frequency < 1:
        raise ValueError(
            f"default_frequency must be >= 1, got {default_frequency!r}"
        )

    result: dict[str, int] = {}
    for word in words:
        if strip:
            word = word.strip()
        if skip_empty and not word:
            continue
        result[word] = default_frequency
    return result


def vocabulary_from_text(
    text: str,
    *,
    min_count: int = 1,
    min_length: int = 2,
    lowercase: bool = True,
    token_pattern: str = r"[a-zA-Z][a-zA-Z0-9_'-]*",
) -> dict[str, int]:
    """
    Build a {word: count} dict by tokenising a text corpus. Words below min_count are excluded.

    Example::

        vocab = vocabulary_from_text(corpus, min_count=3)
    """
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count!r}")
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length!r}")

    tokens = re.findall(token_pattern, text)
    if lowercase:
        tokens = [t.lower() for t in tokens]

    counts = Counter(t for t in tokens if len(t) >= min_length)
    return {word: count for word, count in counts.items() if count >= min_count}


def vocabulary_from_file(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    fmt: str = "wordlist",
    default_frequency: int = 1,
    strip: bool = True,
    skip_empty: bool = True,
    min_count: int = 1,
    min_length: int = 2,
    lowercase: bool = True,
    token_pattern: str = r"[a-zA-Z][a-zA-Z0-9_'-]*",
) -> dict[str, int]:
    """Load a vocabulary from a file. fmt="wordlist" (one word per line) or "text" (corpus)."""
    path = Path(path)
    content = path.read_text(encoding=encoding)

    if fmt == "wordlist":
        return vocabulary_from_wordlist(
            content.splitlines(),
            default_frequency=default_frequency,
            strip=strip,
            skip_empty=skip_empty,
        )
    elif fmt == "text":
        return vocabulary_from_text(
            content,
            min_count=min_count,
            min_length=min_length,
            lowercase=lowercase,
            token_pattern=token_pattern,
        )
    else:
        raise ValueError(
            f"Unknown fmt {fmt!r}. Use 'wordlist' (one word per line) "
            f"or 'text' (free-form text corpus)."
        )
