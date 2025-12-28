#!/usr/bin/env python3
"""Clean up games without move data from databases.

This script identifies and removes games that have metadata (in the games table)
but no corresponding move data (in the game_moves table). Such games are useless
for training and pollute the database.

Usage:
    # Dry run - show what would be deleted
    python scripts/cleanup_games_without_moves.py --dry-run

    # Clean all canonical databases
    python scripts/cleanup_games_without_moves.py --execute

    # Clean specific database
    python scripts/cleanup_games_without_moves.py --db data/games/canonical_hex8.db --execute

    # List all databases with orphan games
    python scripts/cleanup_games_without_moves.py --list-orphans

December 2025: Created for move data integrity enforcement.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("cleanup")


@dataclass
class OrphanGameInfo:
    """Information about an orphan game (game without move data)."""
    game_id: str
    board_type: str
    num_players: int
    total_moves: int  # Claimed moves in games table
    actual_moves: int  # Actual moves in game_moves table
    created_at: str
    game_status: str


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    db_path: str = ""
    orphan_games: list[OrphanGameInfo] = field(default_factory=list)
    games_deleted: int = 0
    related_records_deleted: dict[str, int] = field(default_factory=dict)
    dry_run: bool = True
    error: str | None = None


def find_canonical_databases(base_dir: Path) -> list[Path]:
    """Find all canonical game databases."""
    databases = []
    for pattern in ["canonical_*.db", "games/*.db", "**/canonical_*.db"]:
        databases.extend(base_dir.glob(pattern))
    # Filter out non-game databases
    databases = [
        db for db in databases
        if "jsonl" not in db.name and "sync" not in db.name and "elo" not in db.name
    ]
    return list(set(databases))


def find_orphan_games(db_path: Path) -> list[OrphanGameInfo]:
    """Find games that have no corresponding move data.

    Returns:
        List of OrphanGameInfo for games without move data
    """
    orphans = []

    try:
        conn = sqlite3.connect(str(db_path))

        # Check if game_moves table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        has_moves_table = cursor.fetchone() is not None

        if not has_moves_table:
            # All games are orphans if no moves table
            cursor = conn.execute("""
                SELECT game_id, board_type, num_players, total_moves,
                       created_at, game_status
                FROM games
            """)
            for row in cursor:
                orphans.append(OrphanGameInfo(
                    game_id=row[0],
                    board_type=row[1],
                    num_players=row[2],
                    total_moves=row[3],
                    actual_moves=0,
                    created_at=row[4],
                    game_status=row[5],
                ))
        else:
            # Find games with no moves in game_moves table
            cursor = conn.execute("""
                SELECT g.game_id, g.board_type, g.num_players, g.total_moves,
                       g.created_at, g.game_status,
                       COALESCE(COUNT(m.game_id), 0) as actual_moves
                FROM games g
                LEFT JOIN game_moves m ON g.game_id = m.game_id
                GROUP BY g.game_id
                HAVING actual_moves = 0
            """)
            for row in cursor:
                orphans.append(OrphanGameInfo(
                    game_id=row[0],
                    board_type=row[1],
                    num_players=row[2],
                    total_moves=row[3],
                    actual_moves=row[6],
                    created_at=row[4],
                    game_status=row[5],
                ))

        conn.close()
        return orphans

    except sqlite3.Error as e:
        logger.error(f"Error finding orphan games in {db_path}: {e}")
        return []


def cleanup_orphan_games(
    db_path: Path,
    dry_run: bool = True,
) -> CleanupResult:
    """Remove orphan games and their related records.

    Args:
        db_path: Path to database
        dry_run: If True, don't actually delete anything

    Returns:
        CleanupResult with details of what was/would be deleted
    """
    result = CleanupResult(db_path=str(db_path), dry_run=dry_run)

    # Find orphan games first
    result.orphan_games = find_orphan_games(db_path)

    if not result.orphan_games:
        return result

    orphan_ids = [g.game_id for g in result.orphan_games]

    # Related tables to clean up
    RELATED_TABLES = [
        "game_moves",
        "game_initial_state",
        "game_state_snapshots",
        "game_players",
        "game_choices",
        "game_history_entries",
        "game_nnue_features",
    ]

    try:
        conn = sqlite3.connect(str(db_path))

        # Count and delete related records
        for table_name in RELATED_TABLES:
            try:
                cursor = conn.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                if not cursor.fetchone():
                    continue

                # Count records that would be deleted
                placeholders = ",".join(["?" for _ in orphan_ids])
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE game_id IN ({placeholders})",
                    orphan_ids
                )
                count = cursor.fetchone()[0]
                result.related_records_deleted[table_name] = count

                if not dry_run and count > 0:
                    conn.execute(
                        f"DELETE FROM {table_name} WHERE game_id IN ({placeholders})",
                        orphan_ids
                    )

            except sqlite3.Error as e:
                logger.debug(f"Error accessing {table_name}: {e}")

        # Delete from games table
        placeholders = ",".join(["?" for _ in orphan_ids])

        if not dry_run:
            conn.execute(
                f"DELETE FROM games WHERE game_id IN ({placeholders})",
                orphan_ids
            )
            result.games_deleted = len(orphan_ids)
            conn.commit()
            logger.info(f"Deleted {result.games_deleted} orphan games from {db_path.name}")
        else:
            result.games_deleted = len(orphan_ids)
            logger.info(f"[DRY-RUN] Would delete {result.games_deleted} orphan games from {db_path.name}")

        conn.close()

    except sqlite3.Error as e:
        result.error = str(e)
        logger.error(f"Error cleaning up {db_path}: {e}")

    return result


def list_orphans_by_config(db_path: Path) -> dict[str, int]:
    """Get orphan game counts by config.

    Returns:
        Dict mapping config key (e.g., 'hex8_2p') to orphan count
    """
    orphan_counts: dict[str, int] = {}

    try:
        conn = sqlite3.connect(str(db_path))

        # Check if game_moves table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        has_moves_table = cursor.fetchone() is not None

        if not has_moves_table:
            # All games are orphans
            cursor = conn.execute("""
                SELECT board_type, num_players, COUNT(*) as count
                FROM games
                GROUP BY board_type, num_players
            """)
        else:
            # Count games without moves
            cursor = conn.execute("""
                SELECT g.board_type, g.num_players, COUNT(g.game_id) as count
                FROM games g
                LEFT JOIN game_moves m ON g.game_id = m.game_id
                WHERE m.game_id IS NULL
                GROUP BY g.board_type, g.num_players
            """)

        for row in cursor:
            config_key = f"{row[0]}_{row[1]}p"
            orphan_counts[config_key] = row[2]

        conn.close()
        return orphan_counts

    except sqlite3.Error as e:
        logger.error(f"Error counting orphans in {db_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Clean up games without move data from databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db", type=str,
        help="Specific database to clean (default: all canonical databases)"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).parent.parent / "data" / "games"),
        help="Directory containing game databases"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without deleting"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually delete orphan games (required to make changes)"
    )
    parser.add_argument(
        "--list-orphans", action="store_true",
        help="List all databases and their orphan game counts"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine which databases to process
    if args.db:
        databases = [Path(args.db)]
        if not databases[0].exists():
            logger.error(f"Database not found: {args.db}")
            sys.exit(1)
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            sys.exit(1)
        databases = find_canonical_databases(data_dir)

    if not databases:
        logger.warning("No databases found to process")
        sys.exit(0)

    # List orphans mode
    if args.list_orphans:
        print("\n" + "=" * 70)
        print("ORPHAN GAMES BY DATABASE AND CONFIG")
        print("=" * 70)

        total_orphans = 0
        for db in sorted(databases):
            orphan_counts = list_orphans_by_config(db)
            if orphan_counts:
                db_total = sum(orphan_counts.values())
                total_orphans += db_total
                print(f"\n{db.name}: {db_total} total orphans")
                for config, count in sorted(orphan_counts.items()):
                    print(f"  {config}: {count}")

        print(f"\n{'=' * 70}")
        print(f"TOTAL ORPHAN GAMES: {total_orphans}")
        return

    # Validate execution mode
    if not args.dry_run and not args.execute:
        logger.error("Must specify --dry-run or --execute")
        parser.print_help()
        sys.exit(1)

    dry_run = not args.execute

    if dry_run:
        print("\nDRY RUN - no changes will be made")
    else:
        print("\nEXECUTE MODE - orphan games will be deleted")

    # Process databases
    results = []
    total_deleted = 0

    for db in databases:
        result = cleanup_orphan_games(db, dry_run=dry_run)
        results.append(result)
        total_deleted += result.games_deleted

        if result.orphan_games and args.verbose:
            print(f"\n  Orphan games in {db.name}:")
            for orphan in result.orphan_games[:10]:  # Show first 10
                print(f"    {orphan.game_id}: {orphan.board_type}_{orphan.num_players}p "
                      f"(claimed: {orphan.total_moves}, actual: {orphan.actual_moves})")
            if len(result.orphan_games) > 10:
                print(f"    ... and {len(result.orphan_games) - 10} more")

    # Summary
    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)

    for r in results:
        if r.games_deleted > 0:
            db_name = Path(r.db_path).name
            action = "Would delete" if dry_run else "Deleted"
            print(f"  {db_name}: {action} {r.games_deleted} orphan games")
            if r.related_records_deleted:
                for table, count in r.related_records_deleted.items():
                    if count > 0:
                        print(f"    {table}: {count} records")

    print(f"\n{'Total games' if dry_run else 'Games'} {'to delete' if dry_run else 'deleted'}: {total_deleted}")

    if dry_run:
        print("\n(DRY RUN - use --execute to actually delete)")


if __name__ == "__main__":
    main()
