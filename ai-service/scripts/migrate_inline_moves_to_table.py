#!/usr/bin/env python3
"""Migrate inline moves from games.moves column to normalized game_moves table.

This script converts games that have moves stored as JSON in the games.moves column
to the normalized game_moves table format. This is the recommended storage format
for training data.

Usage:
    # Dry run - show what would be migrated
    python scripts/migrate_inline_moves_to_table.py --db data/games/my_db.db --dry-run

    # Migrate all games
    python scripts/migrate_inline_moves_to_table.py --db data/games/my_db.db

    # Migrate with batch size and progress
    python scripts/migrate_inline_moves_to_table.py --db data/games/my_db.db --batch-size 1000

    # Migrate all databases matching a pattern
    python scripts/migrate_inline_moves_to_table.py --pattern "data/**/*.db"

December 2025: Created for inline moves migration.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.db.move_data_validator import MIN_MOVES_REQUIRED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_game_moves_table(conn: sqlite3.Connection) -> None:
    """Ensure game_moves table exists with correct schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            turn_number INTEGER,
            player INTEGER,
            phase TEXT,
            move_type TEXT,
            move_json TEXT,
            time_remaining_ms INTEGER,
            engine_eval REAL,
            engine_eval_type TEXT,
            engine_depth INTEGER,
            engine_nodes INTEGER,
            engine_pv TEXT,
            engine_time_ms INTEGER,
            move_probs TEXT,
            search_stats_json TEXT,
            UNIQUE(game_id, move_number)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_game_moves_game_id ON game_moves(game_id)
    """)


def count_games_needing_migration(conn: sqlite3.Connection) -> tuple[int, int]:
    """Count games that need migration.

    Returns:
        Tuple of (games_with_inline_only, games_already_migrated)
    """
    # Count games with inline moves but NOT in game_moves table
    cursor = conn.execute("""
        SELECT COUNT(*) FROM games g
        WHERE g.moves IS NOT NULL
          AND LENGTH(g.moves) > 100
          AND NOT EXISTS (
              SELECT 1 FROM game_moves m WHERE m.game_id = g.game_id LIMIT 1
          )
    """)
    inline_only = cursor.fetchone()[0]

    # Count games already in game_moves table
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT game_id) FROM game_moves
    """)
    already_migrated = cursor.fetchone()[0]

    return inline_only, already_migrated


def migrate_game(
    conn: sqlite3.Connection,
    game_id: str,
    moves_json: str,
) -> int:
    """Migrate a single game's moves to game_moves table.

    Args:
        conn: Database connection
        game_id: Game ID
        moves_json: JSON string of moves array

    Returns:
        Number of moves inserted
    """
    try:
        moves = json.loads(moves_json)
        if not isinstance(moves, list):
            return 0

        count = 0
        for i, move in enumerate(moves):
            if not isinstance(move, dict):
                continue

            move_number = move.get("moveNumber", i + 1)
            turn_number = move.get("turnNumber")
            player = move.get("player")
            phase = move.get("phase")
            move_type = move.get("moveType") or move.get("type")

            conn.execute(
                """
                INSERT OR IGNORE INTO game_moves
                (game_id, move_number, turn_number, player, phase, move_type, move_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    move_number,
                    turn_number,
                    player,
                    phase,
                    move_type,
                    json.dumps(move),
                ),
            )
            count += 1

        return count

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse moves for game {game_id}: {e}")
        return 0


def migrate_database(
    db_path: Path,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> dict:
    """Migrate all inline moves in a database to game_moves table.

    Args:
        db_path: Path to database
        batch_size: Number of games to process per batch
        dry_run: If True, don't actually modify database

    Returns:
        Dict with migration statistics
    """
    stats = {
        "db_path": str(db_path),
        "games_migrated": 0,
        "moves_inserted": 0,
        "games_skipped": 0,
        "errors": 0,
        "duration_seconds": 0,
    }

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        stats["errors"] = 1
        return stats

    start_time = time.time()

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Check if games table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
        )
        if not cursor.fetchone():
            logger.warning(f"No games table in {db_path}")
            conn.close()
            return stats

        # Check if moves column exists
        cursor = conn.execute("PRAGMA table_info(games)")
        columns = {row[1] for row in cursor.fetchall()}
        if "moves" not in columns:
            logger.info(f"No moves column in {db_path} - nothing to migrate")
            conn.close()
            return stats

        # Count what needs migration
        inline_only, already_migrated = count_games_needing_migration(conn)
        logger.info(
            f"{db_path.name}: {inline_only} games to migrate, "
            f"{already_migrated} already in game_moves"
        )

        if inline_only == 0:
            logger.info("Nothing to migrate")
            conn.close()
            stats["duration_seconds"] = time.time() - start_time
            return stats

        if dry_run:
            logger.info(f"[DRY RUN] Would migrate {inline_only} games")
            conn.close()
            stats["games_migrated"] = inline_only
            stats["duration_seconds"] = time.time() - start_time
            return stats

        # Ensure game_moves table exists
        ensure_game_moves_table(conn)

        # Migrate in batches
        offset = 0
        total_migrated = 0
        total_moves = 0

        while True:
            cursor = conn.execute(
                """
                SELECT g.game_id, g.moves FROM games g
                WHERE g.moves IS NOT NULL
                  AND LENGTH(g.moves) > 100
                  AND NOT EXISTS (
                      SELECT 1 FROM game_moves m WHERE m.game_id = g.game_id LIMIT 1
                  )
                LIMIT ?
                """,
                (batch_size,),
            )

            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                game_id = row["game_id"]
                moves_json = row["moves"]

                moves_count = migrate_game(conn, game_id, moves_json)
                if moves_count >= MIN_MOVES_REQUIRED:
                    total_migrated += 1
                    total_moves += moves_count
                else:
                    stats["games_skipped"] += 1

            conn.commit()
            offset += len(rows)
            logger.info(f"  Migrated {total_migrated} games, {total_moves} moves...")

        stats["games_migrated"] = total_migrated
        stats["moves_inserted"] = total_moves
        stats["duration_seconds"] = time.time() - start_time

        logger.info(
            f"Migration complete: {total_migrated} games, {total_moves} moves "
            f"in {stats['duration_seconds']:.1f}s"
        )

        conn.close()

    except Exception as e:
        logger.error(f"Migration error for {db_path}: {e}")
        stats["errors"] = 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate inline moves to normalized game_moves table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to a single database to migrate",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Glob pattern to find databases (e.g., 'data/**/*.db')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of games to process per batch (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without modifying database",
    )

    args = parser.parse_args()

    if not args.db and not args.pattern:
        parser.error("Either --db or --pattern is required")

    db_paths = []
    if args.db:
        db_paths.append(Path(args.db))
    if args.pattern:
        db_paths.extend(Path(p) for p in glob.glob(args.pattern, recursive=True))

    if not db_paths:
        logger.error("No databases found")
        return 1

    logger.info(f"Found {len(db_paths)} database(s) to process")

    total_stats = {
        "databases_processed": 0,
        "games_migrated": 0,
        "moves_inserted": 0,
        "errors": 0,
    }

    for db_path in db_paths:
        logger.info(f"\nProcessing: {db_path}")
        stats = migrate_database(db_path, args.batch_size, args.dry_run)

        total_stats["databases_processed"] += 1
        total_stats["games_migrated"] += stats["games_migrated"]
        total_stats["moves_inserted"] += stats["moves_inserted"]
        total_stats["errors"] += stats["errors"]

    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Databases processed: {total_stats['databases_processed']}")
    logger.info(f"Games migrated: {total_stats['games_migrated']}")
    logger.info(f"Moves inserted: {total_stats['moves_inserted']}")
    logger.info(f"Errors: {total_stats['errors']}")

    return 0 if total_stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
