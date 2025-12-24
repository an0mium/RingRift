#!/usr/bin/env python3
"""
Quarantine and optionally delete old/buggy game data.

The data quality analysis showed that games before Dec 22, 2025 had bugs
causing severe balance issues (e.g., 88.5% P1 win rate in square8_3p).

This script:
1. Moves old games to a quarantine table (games_quarantined)
2. Optionally deletes them from the main games table
3. Reports statistics before/after

Usage:
    # Dry run (report only)
    python scripts/quarantine_old_data.py --db data/games/selfplay.db --dry-run

    # Quarantine games before Dec 22
    python scripts/quarantine_old_data.py --db data/games/selfplay.db --cutoff 2025-12-22

    # Delete after quarantining
    python scripts/quarantine_old_data.py --db data/games/selfplay.db --cutoff 2025-12-22 --delete
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def get_db_stats(conn: sqlite3.Connection) -> dict:
    """Get database statistics."""
    cursor = conn.cursor()

    # Total games
    cursor.execute("SELECT COUNT(*) FROM games")
    total = cursor.fetchone()[0]

    # By date range
    cursor.execute("""
        SELECT
            MIN(created_at) as oldest,
            MAX(created_at) as newest
        FROM games
    """)
    row = cursor.fetchone()
    oldest, newest = row[0], row[1]

    # By config
    cursor.execute("""
        SELECT board_type, num_players, COUNT(*)
        FROM games
        GROUP BY board_type, num_players
        ORDER BY COUNT(*) DESC
    """)
    by_config = cursor.fetchall()

    return {
        "total": total,
        "oldest": oldest,
        "newest": newest,
        "by_config": by_config
    }


def quarantine_old_games(
    db_path: str,
    cutoff_date: str,
    dry_run: bool = True,
    delete_after: bool = False
) -> dict:
    """Quarantine games older than cutoff date."""

    if not os.path.exists(db_path):
        return {"error": f"Database not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get before stats
    before_stats = get_db_stats(conn)

    # Count games to quarantine
    cursor.execute(
        "SELECT COUNT(*) FROM games WHERE created_at < ?",
        (cutoff_date,)
    )
    to_quarantine = cursor.fetchone()[0]

    # Get breakdown by config
    cursor.execute("""
        SELECT board_type, num_players, COUNT(*)
        FROM games
        WHERE created_at < ?
        GROUP BY board_type, num_players
        ORDER BY COUNT(*) DESC
    """, (cutoff_date,))
    quarantine_by_config = cursor.fetchall()

    result = {
        "db_path": db_path,
        "cutoff_date": cutoff_date,
        "before_total": before_stats["total"],
        "to_quarantine": to_quarantine,
        "quarantine_by_config": quarantine_by_config,
        "dry_run": dry_run
    }

    if dry_run:
        conn.close()
        return result

    # Create quarantine table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games_quarantined (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            created_at TEXT,
            winner INTEGER,
            termination_reason TEXT,
            total_moves INTEGER,
            source TEXT,
            quarantined_at TEXT,
            quarantine_reason TEXT
        )
    """)

    # Move games to quarantine
    cursor.execute("""
        INSERT OR IGNORE INTO games_quarantined
        (game_id, board_type, num_players, created_at, winner,
         termination_reason, total_moves, source, quarantined_at, quarantine_reason)
        SELECT
            game_id, board_type, num_players, created_at, winner,
            termination_reason, total_moves, source,
            datetime('now'), 'pre_dec22_bug_data'
        FROM games
        WHERE created_at < ?
    """, (cutoff_date,))

    quarantined = cursor.rowcount
    result["quarantined"] = quarantined

    if delete_after:
        # Delete from main table
        cursor.execute(
            "DELETE FROM games WHERE created_at < ?",
            (cutoff_date,)
        )
        deleted = cursor.rowcount
        result["deleted"] = deleted

        # Also delete from moves table if exists
        try:
            cursor.execute("""
                DELETE FROM moves WHERE game_id IN (
                    SELECT game_id FROM games_quarantined
                )
            """)
            moves_deleted = cursor.rowcount
            result["moves_deleted"] = moves_deleted
        except sqlite3.OperationalError:
            pass  # moves table might not exist

    conn.commit()

    if delete_after:
        # Vacuum to reclaim space (must be outside transaction)
        conn.execute("VACUUM")

    # Get after stats
    after_stats = get_db_stats(conn)
    result["after_total"] = after_stats["total"]

    conn.close()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Quarantine old/buggy game data"
    )
    parser.add_argument(
        "--db", type=str, required=True,
        help="Path to database file"
    )
    parser.add_argument(
        "--cutoff", type=str, default="2025-12-22",
        help="Cutoff date (games before this are quarantined). Default: 2025-12-22"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report what would be done, don't modify"
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete games after quarantining (default: keep in quarantine table)"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"DATA QUARANTINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Database: {args.db}")
    print(f"Cutoff:   {args.cutoff}")
    print(f"Mode:     {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print(f"Delete:   {'YES' if args.delete else 'NO (quarantine only)'}")
    print(f"{'='*60}\n")

    result = quarantine_old_games(
        args.db,
        args.cutoff,
        dry_run=args.dry_run,
        delete_after=args.delete
    )

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return 1

    print(f"Before: {result['before_total']} games")
    print(f"To quarantine: {result['to_quarantine']} games")
    print(f"\nBreakdown by config:")
    for board, players, count in result.get("quarantine_by_config", []):
        print(f"  {board}_{players}p: {count} games")

    if not args.dry_run:
        print(f"\nQuarantined: {result.get('quarantined', 0)} games")
        if args.delete:
            print(f"Deleted: {result.get('deleted', 0)} games")
            print(f"Moves deleted: {result.get('moves_deleted', 0)}")
        print(f"After: {result.get('after_total', 'N/A')} games")
    else:
        print(f"\n[DRY RUN] No changes made. Run without --dry-run to execute.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
