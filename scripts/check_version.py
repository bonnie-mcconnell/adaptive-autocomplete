"""
Version consistency check: verifies that src/aac/__init__.py __version__
matches the version in pyproject.toml.

Run via:
    python scripts/check_version.py
    make version-check

Exit code 0 if versions match, 1 if they differ.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def _init_version() -> str:
    text = (ROOT / "src" / "aac" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise ValueError("Could not find __version__ in src/aac/__init__.py")
    return match.group(1)


def main() -> int:
    pyproject = _pyproject_version()
    init = _init_version()

    if pyproject == init:
        print(f"Version OK: {pyproject}")
        return 0
    else:
        print(
            f"Version mismatch!\n"
            f"  pyproject.toml:       {pyproject}\n"
            f"  src/aac/__init__.py:  {init}\n"
            f"\n"
            f"Update one to match the other, then commit both.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
