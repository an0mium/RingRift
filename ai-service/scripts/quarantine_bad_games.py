#!/usr/bin/env python3
"""Quarantine problematic games from selfplay databases.

This script identifies and quarantines games with:
- NULL or empty termination_reason
- No winner (winner IS NULL or winner = 0)
- High move counts indicating timeout (>= move limit for board type)
- env_done_flag termination

Games are marked with excluded_from_training=1 so they are excluded from
training data queries but preserved for analysis.

Usage:
    # Dry run - report what would be quarantined
    python scripts/quarantine_bad_games.py --dry-run

    # Apply quarantine to all databases in data/games
    python scripts/quarantine_bad_games.py --apply

    # Quarantine specific database
    python scripts/quarantine_bad_games.py --db data/games/selfplay.db --apply
"""

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Move limits by board type
MOVE_LIMITS = {
    "square8": 200,
    "square19": 400,
    "hexagonal": 300,
}


def get_problematic_games(conn: sqlite3.Connection) -> dict:
    """Identify all problematic games in a database.

    Returns dict with game_ids grouped by issue type.
    """
    cursor = conn.cursor()

    # Get column info
    cursor.execute("PRAGMA table_info(games)")
    cols = {row[1] for row in cursor.fetchall()}

    issues = defaultdict(set)

    # NULL or empty termination_reason
    if "termination_reason" in cols:
        cursor.execute("""
            SELECT game_id FROM games
            WHERE termination_reason IS NULL
               OR termination_reason = ''
               OR termination_reason LIKE '%env_done%'
        """)
        for row in cursor.fetchall():
            issues["null_or_bad_termination"].add(row[0])

    # No winner
    if "winner" in cols:
        cursor.execute("""
            SELECT game_id FROM games
            WHERE winner IS NULL OR winner = 0
        """)
        for row in cursor.fetchall():
            issues["no_winner"].add(row[0])

    # High move counts (timeout games)
    # Only exclude if the game didn't complete normally - games that ended by
    # elimination/victory are valid even if they have many moves
    moves_col = "total_moves" if "total_moves" in cols else ("num_moves" if "num_moves" in cols else None)
    has_termination = "termination_reason" in cols
    if moves_col and "board_type" in cols:
        for board_type, limit in MOVE_LIMITS.items():
            if has_termination:
                # Only exclude high-move games that didn't complete normally
                cursor.execute(f"""
                    SELECT game_id FROM games
                    WHERE board_type = ? AND {moves_col} >= ?
                    AND (termination_reason IS NULL
                         OR termination_reason = ''
                         OR termination_reason LIKE '%timeout%'
                         OR termination_reason LIKE '%env_done%'
                         OR termination_reason NOT LIKE '%completed%')
                """, (board_type, limit))
            else:
                cursor.execute(f"""
                    SELECT game_id FROM games
                    WHERE board_type = ? AND {moves_col} >= ?
                """, (board_type, limit))
            for row in cursor.fetchall():
                issues["timeout_high_moves"].add(row[0])
    elif moves_col:
        # If no board_type, use conservative limit
        if has_termination:
            cursor.execute(f"""
                SELECT game_id FROM games
                WHERE {moves_col} >= 400
                AND (termination_reason IS NULL
                     OR termination_reason = ''
                     OR termination_reason LIKE '%timeout%'
                     OR termination_reason LIKE '%env_done%'
                     OR termination_reason NOT LIKE '%completed%')
            """)
        else:
            cursor.execute(f"""
                SELECT game_id FROM games WHERE {moves_col} >= 400
            """)
        for row in cursor.fetchall():
            issues["timeout_high_moves"].add(row[0])

    return issues


def quarantine_games(db_path: Path, dry_run: bool = True) -> dict:
    """Quarantine problematic games in a database.

    Returns statistics about quarantined games.
    """
    stats = {
        "db": str(db_path),
        "total_games": 0,
        "quarantined": 0,
        "by_issue": {},
        "errors": [],
    }

    if not db_path.exists():
        stats["errors"].append(f"Database not found: {db_path}")
        return stats

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()

        # Check if games table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if not cursor.fetchone():
            return stats

        # Get total games
        cursor.execute("SELECT COUNT(*) FROM games")
        stats["total_games"] = cursor.fetchone()[0]

        if stats["total_games"] == 0:
            conn.close()
            return stats

        # Find problematic games
        issues = get_problematic_games(conn)

        # Collect all unique game_ids to quarantine
        all_bad_ids = set()
        for issue_type, game_ids in issues.items():
            stats["by_issue"][issue_type] = len(game_ids)
            all_bad_ids.update(game_ids)

        stats["quarantined"] = len(all_bad_ids)

        if dry_run:
            conn.close()
            return stats

        if not all_bad_ids:
            conn.close()
            return stats

        # Add excluded_from_training column if it doesn't exist
        cursor.execute("PRAGMA table_info(games)")
        cols = {row[1] for row in cursor.fetchall()}
        if "excluded_from_training" not in cols:
            cursor.execute("ALTER TABLE games ADD COLUMN excluded_from_training INTEGER DEFAULT 0")

        # Mark games as excluded
        game_id_list = list(all_bad_ids)
        for i in range(0, len(game_id_list), 500):
            batch = game_id_list[i:i+500]
            placeholders = ",".join(["?"] * len(batch))
            cursor.execute(f"""
                UPDATE games
                SET excluded_from_training = 1
                WHERE game_id IN ({placeholders})
            """, batch)

        conn.commit()
        conn.close()

    except Exception as e:
        stats["errors"].append(str(e))

    return stats


def unquarantine_valid_games(db_path: Path, dry_run: bool = True) -> dict:
    """Un-quarantine games that were incorrectly excluded.

    This finds games marked excluded_from_training=1 that actually completed
    normally (have a winner and valid termination_reason) and clears the flag.
    """
    stats = {
        "db": str(db_path),
        "unquarantined": 0,
        "errors": [],
    }

    if not db_path.exists():
        stats["errors"].append(f"Database not found: {db_path}")
        return stats

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()

        # Check columns
        cursor.execute("PRAGMA table_info(games)")
        cols = {row[1] for row in cursor.fetchall()}

        if "excluded_from_training" not in cols:
            conn.close()
            return stats

        # Find games that are excluded but completed normally
        # These should be un-quarantined
        cursor.execute("""
            SELECT game_id FROM games
            WHERE excluded_from_training = 1
            AND winner IS NOT NULL AND winner > 0
            AND termination_reason LIKE '%completed%'
        """)
        valid_game_ids = [row[0] for row in cursor.fetchall()]

        stats["unquarantined"] = len(valid_game_ids)

        if dry_run or not valid_game_ids:
            conn.close()
            return stats

        # Clear the exclusion flag
        for i in range(0, len(valid_game_ids), 500):
            batch = valid_game_ids[i:i+500]
            placeholders = ",".join(["?"] * len(batch))
            cursor.execute(f"""
                UPDATE games
                SET excluded_from_training = 0
                WHERE game_id IN ({placeholders})
            """, batch)

        conn.commit()
        conn.close()

    except Exception as e:
        stats["errors"].append(str(e))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Quarantine problematic games")
    parser.add_argument("--db", type=str, help="Specific database to process")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory to scan")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Report only, don't modify")
    parser.add_argument("--apply", action="store_true", help="Actually apply quarantine")
    parser.add_argument("--unquarantine", action="store_true", help="Un-quarantine valid games that were incorrectly excluded")
    args = parser.parse_args()

    dry_run = not args.apply

    if args.db:
        db_paths = [Path(args.db)]
    else:
        data_dir = Path(args.data_dir)
        db_paths = list(data_dir.glob("**/*.db"))

    # Handle unquarantine mode
    if args.unquarantine:
        print(f"{'[DRY RUN] ' if dry_run else ''}Un-quarantine Valid Games")
        print(f"Time: {datetime.now().isoformat()}")
        print("=" * 70)
        print(f"Scanning {len(db_paths)} databases...\n")

        total_unquarantined = 0
        for db_path in sorted(db_paths):
            if any(x in str(db_path) for x in ["manifest", "elo", "tournament"]):
                continue
            stats = unquarantine_valid_games(db_path, dry_run=dry_run)
            if stats["unquarantined"] > 0:
                print(f"  {db_path.name}: {stats['unquarantined']} games to restore")
                total_unquarantined += stats["unquarantined"]

        print(f"\nTotal games to un-quarantine: {total_unquarantined:,}")
        if dry_run:
            print("\n[DRY RUN] No changes made. Use --apply --unquarantine to restore games.")
        else:
            print(f"\n[APPLIED] Restored {total_unquarantined:,} games (set excluded_from_training=0)")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Quarantine Bad Games")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 70)

    print(f"Scanning {len(db_paths)} databases...\n")

    total_stats = {
        "total_games": 0,
        "quarantined": 0,
        "by_issue": defaultdict(int),
    }

    problem_dbs = []

    for db_path in sorted(db_paths):
        # Skip known non-game databases
        if any(x in str(db_path) for x in ["manifest", "elo", "tournament"]):
            continue

        stats = quarantine_games(db_path, dry_run=dry_run)

        if stats["quarantined"] > 0:
            problem_dbs.append(stats)
            total_stats["total_games"] += stats["total_games"]
            total_stats["quarantined"] += stats["quarantined"]
            for issue, count in stats["by_issue"].items():
                total_stats["by_issue"][issue] += count

    # Print results
    print(f"Found {len(problem_dbs)} databases with problematic games:\n")

    for stats in sorted(problem_dbs, key=lambda x: -x["quarantined"])[:30]:
        db_name = Path(stats["db"]).name
        pct = 100 * stats["quarantined"] / max(1, stats["total_games"])
        issues_str = ", ".join(f"{k}={v}" for k, v in stats["by_issue"].items())
        print(f"  {db_name}: {stats['quarantined']}/{stats['total_games']} ({pct:.1f}%) - {issues_str}")

    print("\n" + "=" * 70)
    print("TOTALS:")
    print(f"  Total games scanned: {total_stats['total_games']:,}")
    print(f"  Games to quarantine: {total_stats['quarantined']:,}")
    if total_stats["total_games"] > 0:
        print(f"  Percentage: {100 * total_stats['quarantined'] / total_stats['total_games']:.2f}%")
    print("\n  By issue type:")
    for issue, count in sorted(total_stats["by_issue"].items(), key=lambda x: -x[1]):
        print(f"    {issue}: {count:,}")

    if dry_run:
        print("\n[DRY RUN] No changes made. Use --apply to quarantine games.")
    else:
        print(f"\n[APPLIED] Marked {total_stats['quarantined']:,} games as excluded_from_training=1")


if __name__ == "__main__":
    main()
