"""Deprecated game engine archive.

This package contains the archived legacy game engine.
DO NOT import directly from this package in new code.

Use instead:
    from app.game_engine import GameEngine
"""

import warnings

warnings.warn(
    "Importing from archive.deprecated_game_engine is deprecated. "
    "Use 'from app.game_engine import GameEngine' instead. "
    "This archive will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)
