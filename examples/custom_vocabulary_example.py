"""
custom_vocabulary_example.py - domain-specific vocabulary

Demonstrates bringing a custom vocabulary rather than the bundled 48k
English corpus. Useful when:
  - Your application has a domain-specific lexicon (medical, legal, code)
  - You want completions for usernames, product names, or any non-English corpus
  - You want to restrict suggestions to a controlled set of valid terms

The key points this example demonstrates:
  1. Custom vocab is a dict[str, int] (word → frequency count)
  2. Relative frequencies matter, not absolute values
  3. EngineConfig.to_json() captures the engine structure; the vocab is
     supplied separately at build() time (it's too large to embed in JSON)
  4. Typo recovery works on custom vocabularies exactly as on the bundled one

Run:
    python examples/custom_vocabulary_example.py
"""
from __future__ import annotations

import json

from aac.engine.config import EngineConfig
from aac.presets import create_engine

# ---------------------------------------------------------------------------
# Simulated domain vocabulary: Python standard library modules
# Frequencies represent rough usage frequency from a hypothetical codebase.
# ---------------------------------------------------------------------------

STDLIB_VOCAB: dict[str, int] = {
    "os": 10000,
    "os.path": 8000,
    "sys": 9000,
    "json": 7500,
    "pathlib": 6000,
    "collections": 5500,
    "itertools": 4000,
    "functools": 3800,
    "datetime": 5000,
    "typing": 7000,
    "dataclasses": 4500,
    "logging": 6500,
    "unittest": 3000,
    "subprocess": 4200,
    "threading": 3500,
    "asyncio": 5800,
    "socket": 2800,
    "http": 2500,
    "urllib": 2200,
    "re": 6000,
    "math": 4000,
    "random": 3200,
    "hashlib": 2000,
    "abc": 3800,
    "io": 4100,
    "contextlib": 3600,
    "copy": 3000,
    "time": 5200,
    "struct": 1800,
    "pickle": 2400,
    "csv": 3300,
    "xml": 1600,
    "sqlite3": 2700,
    "argparse": 4600,
    "configparser": 2100,
    "shutil": 3400,
    "tempfile": 2900,
    "glob": 2600,
    "fnmatch": 1400,
    "pprint": 2300,
    "textwrap": 1900,
    "enum": 4800,
    "decimal": 2200,
    "fractions": 1000,
    "statistics": 2800,
    "heapq": 2000,
    "bisect": 1700,
    "array": 1500,
    "queue": 2400,
    "weakref": 1800,
    "gc": 1200,
    "traceback": 2600,
    "warnings": 3000,
    "inspect": 3200,
    "importlib": 2800,
    "pkgutil": 1400,
    "platform": 2200,
    "signal": 1800,
    "multiprocessing": 3600,
    "concurrent": 3400,
    "concurrent.futures": 3200,
    "zipfile": 2400,
    "tarfile": 1600,
    "gzip": 1800,
    "lzma": 1200,
    "base64": 2600,
    "hmac": 1400,
    "secrets": 2000,
    "uuid": 3800,
}


def main() -> None:
    print("=== Custom vocabulary: Python stdlib modules ===\n")

    # Build a production-quality engine with the custom vocabulary.
    # The production preset uses SymSpell + trigram for typo recovery -
    # exactly as useful for "os.pat" → "os.path" as for natural language.
    engine = create_engine("production", vocabulary=STDLIB_VOCAB)

    # Exact prefix completion
    print('suggest("os"):      ', engine.suggest("os", limit=5))
    print('suggest("co"):      ', engine.suggest("co", limit=5))
    print('suggest("con"):     ', engine.suggest("con", limit=5))
    print('suggest("async"):   ', engine.suggest("async", limit=5))

    # Typo recovery on domain terms
    print('\nTypo recovery:')
    print('suggest("jsn"):     ', engine.suggest("jsn", limit=3))    # json
    print('suggest("pthlib"):  ', engine.suggest("pthlib", limit=3)) # pathlib
    print('suggest("loggin"):  ', engine.suggest("loggin", limit=3)) # logging

    # Learning: simulate a developer who uses asyncio heavily
    print('\n=== After recording asyncio selections ===')
    for _ in range(8):
        engine.record_selection("as", "asyncio")
    for _ in range(3):
        engine.record_selection("as", "abc")

    print('suggest("as"):      ', engine.suggest("as", limit=5))
    # asyncio should float to the top despite lower frequency than some others

    # Explain why asyncio ranks where it does
    print('\nExplanation for "as":')
    for e in engine.explain("as")[:4]:
        pct = (e.history_boost / e.final_score * 100) if e.final_score > 0 else 0
        print(
            f"  {e.value:<30s}  base={e.base_score:.3f}  "
            f"boost=+{e.history_boost:.3f}  final={e.final_score:.3f}  "
            f"history={pct:.0f}%"
        )

    # Demonstrate EngineConfig round-trip with custom vocabulary.
    # The vocab is not embedded in the config (too large); it is supplied
    # separately when calling build().
    # We pass preset="production" so build() uses the fast preset path,
    # which guarantees identical predictor construction.
    print('\n=== Config serialisation ===')

    fresh_engine = create_engine("production", vocabulary=STDLIB_VOCAB)
    config = fresh_engine.to_config(preset="production")
    config_json = config.to_json()
    print(f'Config JSON ({len(config_json)} chars):')
    parsed = json.loads(config_json)
    print(f'  preset: {parsed["preset"]}')
    print(f'  predictors: {[p["name"] for p in parsed["predictors"]]}')
    print(f'  rankers: {[r["name"] for r in parsed["rankers"]]}')

    # Rebuild from config with the same vocab
    engine2 = EngineConfig.from_json(config_json).build(vocabulary=STDLIB_VOCAB)
    original_top = fresh_engine.suggest("co", limit=5)
    rebuilt_top = engine2.suggest("co", limit=5)
    match = original_top == rebuilt_top
    print(f'\nConfig round-trip match for "co": {match}')
    assert match, f"Round-trip mismatch: {original_top} != {rebuilt_top}"
    print("✓ Custom vocabulary config round-trip works correctly")


if __name__ == "__main__":
    main()
