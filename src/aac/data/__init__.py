"""
Static data assets for the autocomplete engine.

Provides access to bundled word frequency data.

Data source:
    The bundled vocabulary is derived from the wordfreq library
    (https://github.com/rspeer/wordfreq), which is MIT licensed.
    wordfreq aggregates frequency data from Wikipedia, OpenSubtitles,
    and other corpora. See the wordfreq repository for full attribution.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

_DATA_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def load_english_frequencies() -> MappingProxyType[str, int]:
    """
    Load the bundled English word frequency table.

    Frequencies are derived from wordfreq (top 50k English words),
    scaled to integer counts by multiplying raw probabilities by 1e7.
    Higher value = more common. The vocabulary covers approximately
    48k words after filtering to alphabetic tokens of length >= 2.

    Returns:
        An immutable mapping of lowercase word -> frequency score.
        The same object is returned on every call (cached).

    Raises:
        ValueError: If the data file is malformed.
        FileNotFoundError: If the data file is missing.
    """
    path = _DATA_DIR / "english_frequencies.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")

    return MappingProxyType({str(k): int(v) for k, v in data.items()})