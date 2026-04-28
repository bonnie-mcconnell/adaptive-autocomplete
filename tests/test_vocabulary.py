"""
Tests for vocabulary utility functions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aac.vocabulary import (
    vocabulary_from_file,
    vocabulary_from_text,
    vocabulary_from_wordlist,
)


class TestVocabularyFromWordlist:
    def test_basic_wordlist(self) -> None:
        vocab = vocabulary_from_wordlist(["hello", "help", "hero"])
        assert vocab == {"hello": 1, "help": 1, "hero": 1}

    def test_default_frequency_is_one(self) -> None:
        vocab = vocabulary_from_wordlist(["hello"])
        assert vocab["hello"] == 1

    def test_custom_default_frequency(self) -> None:
        vocab = vocabulary_from_wordlist(["hello", "help"], default_frequency=500)
        assert vocab["hello"] == 500
        assert vocab["help"] == 500

    def test_strips_whitespace_by_default(self) -> None:
        vocab = vocabulary_from_wordlist(["  hello  ", "\thelo\n"])
        assert "hello" in vocab
        assert "helo" in vocab

    def test_skips_empty_strings_by_default(self) -> None:
        vocab = vocabulary_from_wordlist(["hello", "", "  ", "help"])
        assert "" not in vocab
        assert "  " not in vocab
        assert len(vocab) == 2

    def test_strip_false_preserves_whitespace(self) -> None:
        vocab = vocabulary_from_wordlist(["  hello  "], strip=False)
        assert "  hello  " in vocab

    def test_skip_empty_false_includes_empty(self) -> None:
        vocab = vocabulary_from_wordlist(["hello", ""], skip_empty=False, strip=False)
        assert "" in vocab

    def test_generator_input(self) -> None:
        vocab = vocabulary_from_wordlist(w for w in ["hello", "help"])
        assert len(vocab) == 2

    def test_rejects_zero_frequency(self) -> None:
        with pytest.raises(ValueError, match="default_frequency"):
            vocabulary_from_wordlist(["hello"], default_frequency=0)

    def test_multi_word_phrases(self) -> None:
        vocab = vocabulary_from_wordlist(["git commit", "git push"])
        assert "git commit" in vocab
        assert "git push" in vocab

    def test_works_with_create_engine(self) -> None:
        from aac.presets import create_engine
        vocab = vocabulary_from_wordlist(["hello", "help", "hero"])
        engine = create_engine("default", vocabulary=vocab)
        results = engine.suggest("he")
        assert set(results) == {"hello", "help", "hero"}


class TestVocabularyFromText:
    def test_basic_counting(self) -> None:
        vocab = vocabulary_from_text("hello hello world hello world")
        assert vocab["hello"] == 3
        assert vocab["world"] == 2

    def test_lowercased_by_default(self) -> None:
        vocab = vocabulary_from_text("Hello HELLO hello")
        assert vocab["hello"] == 3
        assert "Hello" not in vocab
        assert "HELLO" not in vocab

    def test_min_count_filters(self) -> None:
        vocab = vocabulary_from_text("hello hello world", min_count=2)
        assert "hello" in vocab
        assert "world" not in vocab

    def test_min_length_filters(self) -> None:
        vocab = vocabulary_from_text("a bb ccc dddd")
        assert "a" not in vocab  # len < 2
        assert "bb" in vocab

    def test_custom_token_pattern(self) -> None:
        vocab = vocabulary_from_text(
            "snake_case camelCase",
            token_pattern=r"[a-zA-Z_][a-zA-Z0-9_]*",
        )
        assert "snake_case" in vocab

    def test_rejects_zero_min_count(self) -> None:
        with pytest.raises(ValueError, match="min_count"):
            vocabulary_from_text("hello", min_count=0)

    def test_rejects_zero_min_length(self) -> None:
        with pytest.raises(ValueError, match="min_length"):
            vocabulary_from_text("hello", min_length=0)

    def test_empty_text_returns_empty(self) -> None:
        vocab = vocabulary_from_text("")
        assert vocab == {}


class TestVocabularyFromFile:
    def test_wordlist_format(self, tmp_path: Path) -> None:
        f = tmp_path / "words.txt"
        f.write_text("hello\nhelp\nhero\n")
        vocab = vocabulary_from_file(f)
        assert vocab == {"hello": 1, "help": 1, "hero": 1}

    def test_text_format(self, tmp_path: Path) -> None:
        f = tmp_path / "corpus.txt"
        f.write_text("hello hello world")
        vocab = vocabulary_from_file(f, format="text")
        assert vocab["hello"] == 2
        assert vocab["world"] == 1

    def test_rejects_unknown_format(self, tmp_path: Path) -> None:
        f = tmp_path / "words.txt"
        f.write_text("hello\n")
        with pytest.raises(ValueError, match="Unknown format"):
            vocabulary_from_file(f, format="csv")  # type: ignore[arg-type]

    def test_passes_kwargs_to_underlying(self, tmp_path: Path) -> None:
        f = tmp_path / "words.txt"
        f.write_text("hello\nhelp\nhero\n")
        vocab = vocabulary_from_file(f, default_frequency=999)
        assert vocab["hello"] == 999

    def test_custom_encoding(self, tmp_path: Path) -> None:
        f = tmp_path / "words.txt"
        f.write_bytes("héllo\n".encode())
        vocab = vocabulary_from_file(f, encoding="utf-8")
        assert "héllo" in vocab


class TestVocabularyPublicExport:
    def test_importable_from_top_level(self) -> None:
        from aac import (  # noqa: F401
            vocabulary_from_file,
            vocabulary_from_text,
            vocabulary_from_wordlist,
        )
        assert callable(vocabulary_from_wordlist)
        assert callable(vocabulary_from_text)
        assert callable(vocabulary_from_file)
