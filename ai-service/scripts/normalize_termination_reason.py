#!/usr/bin/env python
"""Normalize termination_reason values to canonical format.

Per GAME_RECORD_SPEC.md, canonical termination reasons are:
- "ring_elimination": Winner reached ring elimination threshold
- "territory": Winner reached territory threshold
- "timeout": Game hit max_moves limit
- "stalemate": Last-player-standing or trapped stalemate
- "resignation": Player resigned (not in selfplay)

This script normalizes legacy format values like:
- "status:completed:lps" -> "stalemate"
- "status:completed:elimination" -> "ring_elimination"
- "status:completed:territory" -> "territory"

Usage:
    python scripts/normalize_termination_reason.py [--dry-run] [db_path...]
    python scripts/normalize_termination_reason.py --all  # Process all DBs in data/
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys


# Mapping from legacy to canonical values
NORMALIZATION_MAP = {
    "status:completed:lps": "lps",
    "status:completed:elimination": "ring_elimination",
    "status:completed:territory": "territory",
    "status:completed:timeout": "timeout",
    "status:completed:resignation": "resignation",
    # Also handle partial matches
    "elimination": "ring_elimination",
}

# Canonical values that don't need normalization
CANONICAL_VALUES = {
    "ring_elimination",
    "territory",
    "timeout",
    "stalemate",
    "lps",
    "resignation",
}


def normalize_value(value: str | None) -> str | None:
    """Normalize a termination_reason value to canonical format."""
    if value is None:
        return None
    if value in CANONICAL_VALUES:
        return value
    if value in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[value]
    # Unknown value, leave as-is
    return value


def normalize_db(db_path: str, dry_run: bool = False) -> dict:
    """Normalize termination_reason values in a database.

    Returns stats about changes made.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        return {"error": "not_found"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if games table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
    )
    if not cursor.fetchone():
        conn.close()
        return {"error": "no_games_table"}

    # Get current distribution of termination_reason values
    cursor.execute(
        "SELECT termination_reason, COUNT(*) FROM games GROUP BY termination_reason"
    )
    before_counts = dict(cursor.fetchall())

    stats = {
        "db": db_path,
        "before": before_counts,
        "updates": {},
        "already_canonical": 0,
        "updated": 0,
        "unknown": 0,
    }

    # Count what needs updating
    for old_value, count in before_counts.items():
        new_value = normalize_value(old_value)
        if old_value == new_value:
            if old_value in CANONICAL_VALUES:
                stats["already_canonical"] += count
            else:
                stats["unknown"] += count
        else:
            stats["updates"][old_value] = {"new": new_value, "count": count}
            stats["updated"] += count

    if dry_run:
        conn.close()
        return stats

    # Apply updates
    for old_value, info in stats["updates"].items():
        cursor.execute(
            "UPDATE games SET termination_reason = ? WHERE termination_reason = ?",
            (info["new"], old_value),
        )

    conn.commit()

    # Verify
    cursor.execute(
        "SELECT termination_reason, COUNT(*) FROM games GROUP BY termination_reason"
    )
    stats["after"] = dict(cursor.fetchall())

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Normalize termination_reason values to canonical format"
    )
    parser.add_argument(
        "db_paths",
        nargs="*",
        help="Database file paths to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all *.db files in data/ directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    args = parser.parse_args()

    db_paths = args.db_paths
    if args.all:
        # Find all DB files in data/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), "data")
        db_paths = glob.glob(os.path.join(data_dir, "**/*.db"), recursive=True)
        print(f"Found {len(db_paths)} database files in {data_dir}")

    if not db_paths:
        parser.print_help()
        sys.exit(1)

    total_updated = 0
    total_already_canonical = 0
    total_unknown = 0

    for db_path in db_paths:
        stats = normalize_db(db_path, dry_run=args.dry_run)

        if "error" in stats:
            print(f"  SKIP {db_path}: {stats['error']}")
            continue

        if stats["updates"]:
            prefix = "[DRY-RUN] " if args.dry_run else ""
            print(f"{prefix}Processed {db_path}:")
            for old, info in stats["updates"].items():
                print(f"    {old} -> {info['new']} ({info['count']} rows)")
            total_updated += stats["updated"]
        else:
            print(f"  OK {db_path}: all {stats['already_canonical']} already canonical")

        total_already_canonical += stats["already_canonical"]
        total_unknown += stats["unknown"]

    print()
    print(f"Summary:")
    print(f"  Already canonical: {total_already_canonical}")
    print(f"  Updated: {total_updated}")
    if total_unknown:
        print(f"  Unknown values: {total_unknown}")


if __name__ == "__main__":
    main()
