#!/usr/bin/env python3
"""Clean up invalid victory types from game databases and JSONL files.

This script:
1. Scans databases for games with invalid/unknown victory types
2. Deletes games with unrecognized victory types (other, unknown, None)
3. Normalizes remaining victory types to canonical forms
4. Handles JSONL files similarly

Victory Types (canonical):
- ring_elimination: Winner reached ring elimination threshold
- territory: Winner reached territory threshold
- timeout: Game hit max_moves limit
- lps: Last-player-standing (opponent eliminated)
- stalemate: Bare-board stalemate resolved by tiebreaker

Usage:
    # Scan databases for invalid games (dry run)
    python scripts/cleanup_victory_types.py --scan

    # Delete invalid games and normalize remaining (dry run)
    python scripts/cleanup_victory_types.py --clean --dry-run

    # Actually apply changes
    python scripts/cleanup_victory_types.py --clean --apply

    # Process JSONL files
    python scripts/cleanup_victory_types.py --clean-jsonl --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.victory_type import (
    VICTORY_TYPES,
    is_valid_victory_type,
    normalize_victory_type,
)


@dataclass
class CleanupStats:
    """Statistics from a cleanup operation."""

    database: str
    total_games: int
    invalid_games: int
    normalized_games: int
    deleted_games: int = 0
    updated_games: int = 0

    def __str__(self) -> str:
        return (
            f"{self.database}: {self.total_games} total, "
            f"{self.invalid_games} invalid, {self.normalized_games} normalized"
        )


def scan_database(db_path: str) -> CleanupStats:
    """Scan a database and report victory type statistics."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get all unique termination reasons
    cursor = conn.execute(
        "SELECT termination_reason, COUNT(*) as count FROM games GROUP BY termination_reason"
    )
    reason_counts: dict[str | None, int] = {}
    for row in cursor:
        reason_counts[row["termination_reason"]] = row["count"]

    total = sum(reason_counts.values())
    invalid = 0
    needs_normalization = 0

    print(f"\n{db_path}:")
    print(f"  Total games: {total}")
    print("  Victory types:")

    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        normalized = normalize_victory_type(reason)
        if normalized is None:
            status = "INVALID (will delete)"
            invalid += count
        elif reason != normalized:
            status = f"-> {normalized} (will normalize)"
            needs_normalization += count
        else:
            status = "OK"
        print(f"    {reason!r}: {count} {status}")

    conn.close()

    return CleanupStats(
        database=db_path,
        total_games=total,
        invalid_games=invalid,
        normalized_games=needs_normalization,
    )


def clean_database(db_path: str, dry_run: bool = True) -> CleanupStats:
    """Clean invalid victory types from a database.

    Args:
        db_path: Path to the SQLite database
        dry_run: If True, only report what would be done
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get current counts
    cursor = conn.execute(
        "SELECT termination_reason, COUNT(*) as count FROM games GROUP BY termination_reason"
    )
    reason_counts: dict[str | None, int] = {}
    for row in cursor:
        reason_counts[row["termination_reason"]] = row["count"]

    total = sum(reason_counts.values())
    deleted = 0
    updated = 0
    invalid = 0
    needs_normalization = 0

    # Calculate what needs to change
    for reason, count in reason_counts.items():
        normalized = normalize_victory_type(reason)
        if normalized is None:
            invalid += count
        elif reason != normalized:
            needs_normalization += count

    if dry_run:
        print(f"\n{db_path} (DRY RUN):")
        print(f"  Would delete {invalid} invalid games")
        print(f"  Would normalize {needs_normalization} games")
    else:
        print(f"\n{db_path}:")

        # Delete invalid games
        for reason in list(reason_counts.keys()):
            if normalize_victory_type(reason) is None:
                # Build query based on whether reason is None or a string
                if reason is None:
                    cursor = conn.execute(
                        "SELECT game_id FROM games WHERE termination_reason IS NULL"
                    )
                else:
                    cursor = conn.execute(
                        "SELECT game_id FROM games WHERE termination_reason = ?",
                        (reason,),
                    )
                game_ids = [row["game_id"] for row in cursor]

                if game_ids:
                    placeholders = ",".join("?" * len(game_ids))

                    # Delete from moves table (may not exist in all schemas)
                    try:
                        conn.execute(
                            f"DELETE FROM moves WHERE game_id IN ({placeholders})", game_ids
                        )
                    except sqlite3.OperationalError:
                        pass  # Table doesn't exist

                    # Delete from game_players table (may not exist in all schemas)
                    try:
                        conn.execute(
                            f"DELETE FROM game_players WHERE game_id IN ({placeholders})",
                            game_ids,
                        )
                    except sqlite3.OperationalError:
                        pass  # Table doesn't exist

                    # Delete from games table
                    conn.execute(
                        f"DELETE FROM games WHERE game_id IN ({placeholders})", game_ids
                    )

                    deleted += len(game_ids)
                    print(f"  Deleted {len(game_ids)} games with termination_reason={reason!r}")

        # Normalize remaining games
        for reason in list(reason_counts.keys()):
            normalized = normalize_victory_type(reason)
            if normalized and reason != normalized:
                cursor = conn.execute(
                    "UPDATE games SET termination_reason = ? WHERE termination_reason = ?",
                    (normalized, reason),
                )
                count = cursor.rowcount
                updated += count
                print(f"  Normalized {count} games: {reason!r} -> {normalized!r}")

        conn.commit()
        print(f"  Total: deleted {deleted}, updated {updated}")

    conn.close()

    return CleanupStats(
        database=db_path,
        total_games=total,
        invalid_games=invalid,
        normalized_games=needs_normalization,
        deleted_games=deleted,
        updated_games=updated,
    )


def find_databases(search_paths: list[str]) -> list[str]:
    """Find all .db files in the given paths."""
    databases = []
    for search_path in search_paths:
        path = Path(search_path)
        if path.is_file() and path.suffix == ".db":
            databases.append(str(path))
        elif path.is_dir():
            for db_file in path.rglob("*.db"):
                databases.append(str(db_file))
    return sorted(databases)


def clean_jsonl_file(jsonl_path: str, dry_run: bool = True) -> tuple[int, int, int]:
    """Clean invalid victory types from a JSONL file.

    Returns:
        Tuple of (total_lines, invalid_lines, normalized_lines)
    """
    path = Path(jsonl_path)
    if not path.exists():
        print(f"File not found: {jsonl_path}")
        return (0, 0, 0)

    lines = path.read_text().strip().split("\n")
    total = len(lines)
    invalid = 0
    normalized = 0
    clean_lines = []

    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Check various fields that might contain victory type
        for field in ["termination_reason", "victory_type", "result_type"]:
            if field in record:
                raw = record[field]
                norm = normalize_victory_type(raw)
                if norm is None:
                    invalid += 1
                    record = None  # Mark for deletion
                    break
                elif raw != norm:
                    normalized += 1
                    record[field] = norm

        if record is not None:
            clean_lines.append(json.dumps(record))

    if dry_run:
        print(f"\n{jsonl_path} (DRY RUN):")
        print(f"  Would delete {invalid} invalid records")
        print(f"  Would normalize {normalized} records")
    else:
        print(f"\n{jsonl_path}:")
        # Write back cleaned file
        path.write_text("\n".join(clean_lines) + "\n")
        print(f"  Deleted {invalid}, normalized {normalized}")

    return (total, invalid, normalized)


def find_jsonl_files(search_paths: list[str]) -> list[str]:
    """Find all .jsonl files in the given paths."""
    files = []
    for search_path in search_paths:
        path = Path(search_path)
        if path.is_file() and path.suffix == ".jsonl":
            files.append(str(path))
        elif path.is_dir():
            for jsonl_file in path.rglob("*.jsonl"):
                files.append(str(jsonl_file))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Clean up invalid victory types")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan databases and report statistics",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean databases (delete invalid, normalize remaining)",
    )
    parser.add_argument(
        "--clean-jsonl",
        action="store_true",
        help="Clean JSONL files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (opposite of --dry-run)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "data/games",
            "data/training",
        ],
        help="Paths to search for databases or JSONL files",
    )

    args = parser.parse_args()

    # Default to dry run unless --apply is specified
    dry_run = not args.apply

    if args.scan:
        databases = find_databases(args.paths)
        print(f"Found {len(databases)} databases")

        all_stats = []
        for db in databases:
            try:
                stats = scan_database(db)
                all_stats.append(stats)
            except Exception as e:
                print(f"Error scanning {db}: {e}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        total_games = sum(s.total_games for s in all_stats)
        total_invalid = sum(s.invalid_games for s in all_stats)
        total_normalize = sum(s.normalized_games for s in all_stats)
        print(f"Total games: {total_games}")
        print(f"Invalid games (will delete): {total_invalid}")
        print(f"Games needing normalization: {total_normalize}")

    elif args.clean:
        databases = find_databases(args.paths)
        print(f"Found {len(databases)} databases")
        if dry_run:
            print("DRY RUN - no changes will be made")
        else:
            print("APPLYING CHANGES")

        all_stats = []
        for db in databases:
            try:
                stats = clean_database(db, dry_run=dry_run)
                all_stats.append(stats)
            except Exception as e:
                print(f"Error cleaning {db}: {e}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if dry_run:
            print("DRY RUN - no changes were made")
            total_would_delete = sum(s.invalid_games for s in all_stats)
            total_would_normalize = sum(s.normalized_games for s in all_stats)
            print(f"Would delete: {total_would_delete} games")
            print(f"Would normalize: {total_would_normalize} games")
        else:
            total_deleted = sum(s.deleted_games for s in all_stats)
            total_updated = sum(s.updated_games for s in all_stats)
            print(f"Deleted: {total_deleted} games")
            print(f"Normalized: {total_updated} games")

    elif args.clean_jsonl:
        jsonl_files = find_jsonl_files(args.paths)
        print(f"Found {len(jsonl_files)} JSONL files")
        if dry_run:
            print("DRY RUN - no changes will be made")

        total_invalid = 0
        total_normalized = 0
        for jsonl_file in jsonl_files:
            try:
                _, invalid, normalized = clean_jsonl_file(jsonl_file, dry_run=dry_run)
                total_invalid += invalid
                total_normalized += normalized
            except Exception as e:
                print(f"Error cleaning {jsonl_file}: {e}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if dry_run:
            print(f"Would delete: {total_invalid} records")
            print(f"Would normalize: {total_normalized} records")
        else:
            print(f"Deleted: {total_invalid} records")
            print(f"Normalized: {total_normalized} records")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
