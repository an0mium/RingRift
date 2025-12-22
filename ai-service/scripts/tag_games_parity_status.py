#!/usr/bin/env python
"""Tag games in databases with their parity validation status.

RR-PARITY-FIX-2025-12-21: This script adds a `parity_status` column to game
databases and runs parity validation on games to populate it.

Statuses:
- 'passed': Game replays identically in both Python and TypeScript engines
- 'failed': Game has divergence between engines
- 'error': Parity check failed due to error (timeout, missing data, etc.)
- 'pending': Not yet validated
- 'skipped': Skipped due to configuration (non-canonical config, etc.)

Usage:
    # Tag all games in a specific database
    python scripts/tag_games_parity_status.py data/games/canonical_square8.db

    # Tag games in all databases
    python scripts/tag_games_parity_status.py --all

    # Dry run (don't modify database)
    python scripts/tag_games_parity_status.py data/games/my.db --dry-run

    # Limit number of games to check
    python scripts/tag_games_parity_status.py data/games/my.db --limit 100

    # Only check untagged games
    python scripts/tag_games_parity_status.py data/games/my.db --pending-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.parity_validator import (
    ParityDivergence,
    ParityMode,
    validate_game_parity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TagResult:
    """Result of tagging a single game."""

    game_id: str
    status: str  # 'passed', 'failed', 'error', 'skipped'
    duration_ms: int
    error_message: str | None = None
    divergence_move: int | None = None


@dataclass
class BatchResult:
    """Result of tagging a batch of games."""

    db_path: str
    total_games: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_seconds: float


def add_parity_status_column(db_path: str) -> bool:
    """Add parity_status column to games table if it doesn't exist.

    Returns:
        True if column was added or already exists, False on error.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute("PRAGMA table_info(games)")
        columns = [row[1] for row in cursor.fetchall()]

        if "parity_status" not in columns:
            cursor.execute(
                "ALTER TABLE games ADD COLUMN parity_status TEXT DEFAULT 'pending'"
            )
            logger.info(f"Added parity_status column to {db_path}")

        if "parity_checked_at" not in columns:
            cursor.execute("ALTER TABLE games ADD COLUMN parity_checked_at TEXT")
            logger.info(f"Added parity_checked_at column to {db_path}")

        if "parity_divergence_move" not in columns:
            cursor.execute(
                "ALTER TABLE games ADD COLUMN parity_divergence_move INTEGER"
            )
            logger.info(f"Added parity_divergence_move column to {db_path}")

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to add columns to {db_path}: {e}")
        return False


def get_games_to_check(
    db_path: str,
    pending_only: bool = False,
    limit: int | None = None,
) -> list[str]:
    """Get list of game IDs to check.

    Args:
        db_path: Path to the database
        pending_only: Only return games with status 'pending' or NULL
        limit: Maximum number of games to return

    Returns:
        List of game IDs
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if pending_only:
        query = """
            SELECT game_id FROM games
            WHERE parity_status IS NULL OR parity_status = 'pending'
            ORDER BY created_at DESC
        """
    else:
        query = "SELECT game_id FROM games ORDER BY created_at DESC"

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    game_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    return game_ids


def update_game_parity_status(
    db_path: str,
    game_id: str,
    status: str,
    divergence_move: int | None = None,
) -> None:
    """Update the parity status for a game.

    Args:
        db_path: Path to the database
        game_id: Game ID to update
        status: New status ('passed', 'failed', 'error', 'skipped')
        divergence_move: Move number where divergence occurred (if failed)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE games SET
            parity_status = ?,
            parity_checked_at = datetime('now'),
            parity_divergence_move = ?
        WHERE game_id = ?
        """,
        (status, divergence_move, game_id),
    )

    conn.commit()
    conn.close()


def check_game_parity(db_path: str, game_id: str) -> TagResult:
    """Check parity for a single game.

    Args:
        db_path: Path to the database
        game_id: Game ID to check

    Returns:
        TagResult with status and details
    """
    start_time = time.time()

    try:
        # Run parity validation in 'warn' mode to get divergence details
        divergence = validate_game_parity(
            db_path=db_path,
            game_id=game_id,
            mode=ParityMode.WARN,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        if divergence is None:
            return TagResult(
                game_id=game_id,
                status="passed",
                duration_ms=duration_ms,
            )
        else:
            return TagResult(
                game_id=game_id,
                status="failed",
                duration_ms=duration_ms,
                divergence_move=divergence.divergence_move_index,
            )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return TagResult(
            game_id=game_id,
            status="error",
            duration_ms=duration_ms,
            error_message=str(e),
        )


def tag_database(
    db_path: str,
    pending_only: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> BatchResult:
    """Tag all games in a database with their parity status.

    Args:
        db_path: Path to the database
        pending_only: Only check games with pending status
        limit: Maximum number of games to check
        dry_run: If True, don't actually update the database

    Returns:
        BatchResult with statistics
    """
    start_time = time.time()

    # Add columns if needed
    if not dry_run:
        if not add_parity_status_column(db_path):
            return BatchResult(
                db_path=db_path,
                total_games=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                duration_seconds=0,
            )

    # Get games to check
    game_ids = get_games_to_check(db_path, pending_only, limit)
    total = len(game_ids)

    if total == 0:
        logger.info(f"No games to check in {db_path}")
        return BatchResult(
            db_path=db_path,
            total_games=0,
            passed=0,
            failed=0,
            errors=0,
            skipped=0,
            duration_seconds=0,
        )

    logger.info(f"Checking {total} games in {db_path}")

    passed = 0
    failed = 0
    errors = 0
    skipped = 0

    for i, game_id in enumerate(game_ids, 1):
        result = check_game_parity(db_path, game_id)

        if result.status == "passed":
            passed += 1
        elif result.status == "failed":
            failed += 1
            logger.warning(
                f"Game {game_id} failed parity at move {result.divergence_move}"
            )
        elif result.status == "error":
            errors += 1
            logger.warning(f"Game {game_id} parity error: {result.error_message}")
        else:
            skipped += 1

        # Update database
        if not dry_run:
            update_game_parity_status(
                db_path,
                game_id,
                result.status,
                result.divergence_move,
            )

        # Progress logging
        if i % 10 == 0 or i == total:
            pct = (i / total) * 100
            logger.info(
                f"Progress: {i}/{total} ({pct:.1f}%) - "
                f"passed={passed}, failed={failed}, errors={errors}"
            )

    duration = time.time() - start_time

    return BatchResult(
        db_path=db_path,
        total_games=total,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        duration_seconds=duration,
    )


def find_all_databases(data_dir: str = "data/games") -> list[str]:
    """Find all game databases in the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    db_paths = []
    for db_file in data_path.glob("*.db"):
        # Skip temporary and system databases
        if any(x in db_file.name for x in ["tmp_", "-shm", "-wal", "-journal"]):
            continue
        db_paths.append(str(db_file))

    return sorted(db_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Tag games with parity validation status"
    )
    parser.add_argument(
        "db_path",
        nargs="?",
        help="Path to database file (or use --all for all databases)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all databases in data/games",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Only check games with pending status",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of games to check per database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually update the database",
    )
    parser.add_argument(
        "--data-dir",
        default="data/games",
        help="Data directory for --all mode",
    )

    args = parser.parse_args()

    if not args.db_path and not args.all:
        parser.error("Either provide a database path or use --all")

    # Get list of databases
    if args.all:
        db_paths = find_all_databases(args.data_dir)
        if not db_paths:
            logger.error(f"No databases found in {args.data_dir}")
            return 1
        logger.info(f"Found {len(db_paths)} databases")
    else:
        if not os.path.exists(args.db_path):
            logger.error(f"Database not found: {args.db_path}")
            return 1
        db_paths = [args.db_path]

    # Process each database
    all_results = []
    for db_path in db_paths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {db_path}")
        logger.info(f"{'='*60}")

        result = tag_database(
            db_path=db_path,
            pending_only=args.pending_only,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_errors = sum(r.errors for r in all_results)
    total_games = sum(r.total_games for r in all_results)

    for result in all_results:
        if result.total_games > 0:
            pass_rate = (result.passed / result.total_games) * 100
            print(
                f"{Path(result.db_path).name}: "
                f"{result.passed}/{result.total_games} passed ({pass_rate:.1f}%)"
            )

    print("-" * 60)
    if total_games > 0:
        overall_pass_rate = (total_passed / total_games) * 100
        print(f"TOTAL: {total_passed}/{total_games} passed ({overall_pass_rate:.1f}%)")
        print(f"Failed: {total_failed}, Errors: {total_errors}")

    if args.dry_run:
        print("\n(DRY RUN - no changes made)")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
