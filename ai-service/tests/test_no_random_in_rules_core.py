"""
Smoke tests to enforce RR‑CANON R190 for the Python rules core.

These tests ensure that no direct RNG usage (`import random`, `np.random`)
is introduced into the core rules engine modules. Randomness remains
confined to AI, training, and Zobrist helpers.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"


def _read_text(paths):
    chunks = []
    for path in paths:
        if path.is_dir():
            for dirpath, _, filenames in os.walk(path):
                for name in filenames:
                    if name.endswith(".py"):
                        file_path = Path(dirpath) / name
                        with open(file_path, encoding="utf-8") as f:
                            chunks.append(f.read())
        elif path.is_file():
            with open(path, encoding="utf-8") as f:
                chunks.append(f.read())
    return "\n".join(chunks)


def test_no_random_in_rules_core():
    """Rules core (game_engine + rules package) must not use RNG directly."""
    targets = [
        APP_ROOT / "game_engine.py",
        APP_ROOT / "rules",
    ]
    text = _read_text(targets)
    assert "import random" not in text
    assert "np.random" not in text
