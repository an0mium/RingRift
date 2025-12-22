"""Centralized parity exclusion list for training data.

RR-PARITY-FIX-2025-12-21: This module defines databases that should be
excluded from training due to known parity failures or phase coercion issues.

Usage:
    from app.training.parity_exclusions import should_exclude_database

    for db_path in data_dir.glob("*.db"):
        if should_exclude_database(db_path):
            continue  # Skip non-canonical database
        # Process database...
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

# Databases with known parity failures (TS vs Python divergence)
# These databases contain games that cannot be replayed identically
# by both engines, indicating rules engine bugs or data corruption.
PARITY_FAILURE_PATTERNS: tuple[str, ...] = (
    # Line detection divergence: Python sees no lines, TS sees more lines
    # at move 71 after processing a line in hexagonal games
    "canonical_hexagonal_2p",
)

# Databases with phase coercion issues
# These databases contain games generated before phase coercion was forbidden
# per RR-CANON-R075 (no phase skipping). They have consecutive moves that
# skip intermediate phases (e.g., place_ring -> place_ring without movement).
PHASE_COERCION_PATTERNS: tuple[str, ...] = (
    # GPU selfplay games with consecutive place_ring moves
    "mcts_square8_2p_gh200f",
)

# Combined exclusion patterns
EXCLUDED_DB_PATTERNS: tuple[str, ...] = (
    *PARITY_FAILURE_PATTERNS,
    *PHASE_COERCION_PATTERNS,
)


def should_exclude_database(
    db_path: Path | str,
    exclusion_patterns: Sequence[str] | None = None,
) -> bool:
    """Check if a database should be excluded from training.

    Args:
        db_path: Path to the database file
        exclusion_patterns: Optional custom exclusion patterns.
            Defaults to EXCLUDED_DB_PATTERNS.

    Returns:
        True if the database should be excluded, False otherwise.
    """
    if exclusion_patterns is None:
        exclusion_patterns = EXCLUDED_DB_PATTERNS

    db_name = Path(db_path).stem
    return any(pattern in db_name for pattern in exclusion_patterns)


def get_exclusion_reason(db_path: Path | str) -> str | None:
    """Get the reason why a database is excluded.

    Args:
        db_path: Path to the database file

    Returns:
        Reason string if excluded, None otherwise.
    """
    db_name = Path(db_path).stem

    for pattern in PARITY_FAILURE_PATTERNS:
        if pattern in db_name:
            return f"parity_failure: {pattern}"

    for pattern in PHASE_COERCION_PATTERNS:
        if pattern in db_name:
            return f"phase_coercion: {pattern}"

    return None


__all__ = [
    "EXCLUDED_DB_PATTERNS",
    "PARITY_FAILURE_PATTERNS",
    "PHASE_COERCION_PATTERNS",
    "get_exclusion_reason",
    "should_exclude_database",
]
