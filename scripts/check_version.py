"""
Version consistency check: verifies that src/aac/__init__.py, pyproject.toml,
and the most recent CHANGELOG.md entry all agree on the current version.

Run via:
    python scripts/check_version.py
    make version-check

Exit code 0 if all three sources agree, 1 if any differ.
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


def _changelog_version() -> str:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    # Match the first "## [x.y.z]" heading - that's the most recent release.
    match = re.search(r"^##\s+\[([^\]]+)\]", text, re.MULTILINE)
    if not match:
        raise ValueError(
            "Could not find a version heading in CHANGELOG.md. "
            "Expected a line matching '## [x.y.z]'."
        )
    return match.group(1)


def main() -> int:
    pyproject = _pyproject_version()
    init = _init_version()
    changelog = _changelog_version()

    sources = {
        "pyproject.toml": pyproject,
        "src/aac/__init__.py": init,
        "CHANGELOG.md": changelog,
    }

    versions = set(sources.values())
    if len(versions) == 1:
        print(f"Version OK: {pyproject}")
        return 0

    print("Version mismatch!", file=sys.stderr)
    for source, version in sources.items():
        marker = "  ✓" if version == max(versions) else "  ✗"
        print(f"{marker}  {source:<28s}  {version}", file=sys.stderr)
    print(
        "\nUpdate all three to agree, then commit together.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
