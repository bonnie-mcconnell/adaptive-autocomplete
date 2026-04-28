"""
Utilities for building frequency vocabularies from common input formats.

Most real-world use cases have a word list or text corpus, not a pre-built
frequency dict. These helpers bridge the gap between what users have and what
``create_engine()`` and ``FrequencyPredictor`` expect.

Quick start::

    from aac.presets import create_engine
    from aac.vocabulary import vocabulary_from_wordlist, vocabulary_from_text

    # From a plain word list (all words get equal weight of 1)
    vocab = vocabulary_from_wordlist(["git commit", "git push", "git pull"])
    engine = create_engine("production", vocabulary=vocab)

    # From a text corpus (words weighted by frequency)
    with open("corpus.txt") as f:
        vocab = vocabulary_from_text(f.read())
    engine = create_engine("default", vocabulary=vocab)

    # From a file, one word per line
    vocab = vocabulary_from_wordlist(Path("words.txt").read_text().splitlines())
    engine = create_engine("production", vocabulary=vocab)
"""
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
    Build a frequency vocabulary from a plain word list.

    All words receive the same ``default_frequency`` weight. Use this when
    you have a flat list of domain terms and no frequency information.

    Args:
        words:             Iterable of strings (list, generator, file lines, …).
        default_frequency: Frequency assigned to each word. Default: 1.
                           Raise this (e.g. ``default_frequency=100``) if you
                           plan to mix this vocabulary with the bundled English
                           frequencies and want your terms to appear prominently.
        strip:             Strip leading/trailing whitespace from each word.
                           Default: True.
        skip_empty:        Skip empty strings after stripping. Default: True.

    Returns:
        ``{word: default_frequency}`` dict suitable for ``create_engine()``
        or ``FrequencyPredictor``.

    Examples::

        # Product names
        vocab = vocabulary_from_wordlist(["iPhone 15", "MacBook Pro", "AirPods"])
        engine = create_engine("production", vocabulary=vocab)
        engine.suggest("iP")   # → ["iPhone 15"]  (matches prefix of full phrase)
        engine.suggest("Mac")  # → ["MacBook Pro"]

        # Commands from a text file, one per line
        vocab = vocabulary_from_wordlist(Path("commands.txt").read_text().splitlines())

        # Raise frequency so domain terms surface above common English words
        # when mixing with the bundled vocabulary
        vocab = vocabulary_from_wordlist(my_terms, default_frequency=1000)

    Note on multi-word phrases:
        ``CompletionContext`` extracts the **last word** of the input as the
        completion prefix. For multi-word vocabulary items like ``"git commit"``,
        matching is based on the prefix of the full phrase string - so typing
        ``"git"`` returns ``"git commit"`` (prefix ``"git"`` matches the start
        of the phrase), but typing ``"git c"`` matches items whose last word
        starts with ``"c"``, not items beginning with ``"git c"``. For
        progressive command completion (``"git"`` → ``"git c"`` → ``"git co"``),
        pass the full typed string directly to ``engine.suggest()`` without
        space splitting, or build your vocabulary using only the first token
        of each phrase.
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
    Build a frequency vocabulary by counting words in a text corpus.

    Tokenises ``text`` using ``token_pattern``, counts occurrences, and
    returns words with count >= ``min_count``. Use this when you have a
    representative corpus and want frequency-proportional ranking.

    Args:
        text:          The input text to tokenise and count.
        min_count:     Minimum occurrence count to include a word. Default: 1.
        min_length:    Minimum word length in characters. Default: 2.
        lowercase:     Lowercase all tokens before counting. Default: True.
        token_pattern: Regex pattern for tokenisation. Default matches
                       words including hyphens, apostrophes, and underscores.

    Returns:
        ``{word: count}`` dict suitable for ``create_engine()``
        or ``FrequencyPredictor``.

    Examples::

        # From a markdown README
        with open("README.md") as f:
            vocab = vocabulary_from_text(f.read())

        # From code - use identifier-friendly tokenisation
        vocab = vocabulary_from_text(source_code, token_pattern=r"[a-zA-Z_][a-zA-Z0-9_]*")

        # Only include words that appear at least 3 times
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
    format: str = "wordlist",  # noqa: A002
    **kwargs: object,
) -> dict[str, int]:
    """
    Load a vocabulary from a file.

    Convenience wrapper around ``vocabulary_from_wordlist`` and
    ``vocabulary_from_text`` for common file-loading patterns.

    Args:
        path:     Path to the vocabulary file.
        encoding: File encoding. Default: ``utf-8``.
        format:   Either ``"wordlist"`` (one word per line) or ``"text"``
                  (free-form text, tokenised and counted). Default: ``"wordlist"``.
        **kwargs: Passed to the underlying function.

    Returns:
        ``{word: frequency}`` dict suitable for ``create_engine()``.

    Examples::

        # One word/phrase per line
        vocab = vocabulary_from_file("commands.txt")
        engine = create_engine("production", vocabulary=vocab)

        # Free-form text corpus
        vocab = vocabulary_from_file("corpus.txt", format="text", min_count=3)
        engine = create_engine("default", vocabulary=vocab)
    """
    path = Path(path)
    content = path.read_text(encoding=encoding)

    if format == "wordlist":
        return vocabulary_from_wordlist(content.splitlines(), **kwargs)  # type: ignore[arg-type]
    elif format == "text":
        return vocabulary_from_text(content, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(
            f"Unknown format {format!r}. Use 'wordlist' (one word per line) "
            f"or 'text' (free-form text corpus)."
        )
