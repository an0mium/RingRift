#!/usr/bin/env python3
"""
Audit databases for parity compliance.
Identifies training-grade data vs buggy data that should be deleted.
"""

import os
import sys
import sqlite3
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")


def test_database_parity(db_path: str, sample_size: int = 5) -> dict:
    """Test parity compliance for a database."""
    result = {
        "path": db_path,
        "size_mb": 0,
        "game_count": 0,
        "sample_tested": 0,
        "passed": 0,
        "failed": 0,
        "errors": [],
        "compliant": False,
        "error_msg": None,
    }

    try:
        db_stat = os.stat(db_path)
        result["size_mb"] = db_stat.st_size / (1024 * 1024)
        result["mtime"] = datetime.fromtimestamp(db_stat.st_mtime).isoformat()

        # Get game count
        conn = sqlite3.connect(db_path)
        game_count = conn.execute(
            "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL"
        ).fetchone()[0]
        result["game_count"] = game_count

        if game_count == 0:
            result["compliant"] = True
            result["error_msg"] = "empty"
            conn.close()
            return result

        # Get sample games
        games = conn.execute(
            "SELECT game_id, total_moves FROM games WHERE winner IS NOT NULL AND total_moves > 5 ORDER BY RANDOM() LIMIT ?",
            (sample_size,),
        ).fetchall()
        conn.close()

        # Import replay after connection is closed to avoid issues
        from app.db.game_replay import GameReplayDB

        replay = GameReplayDB(db_path)

        result["sample_tested"] = len(games)

        for game_id, total_moves in games:
            try:
                # Try to replay to near end of game
                state = replay.get_state_at_move(game_id, min(total_moves - 1, 50))
                if state is not None:
                    result["passed"] += 1
                else:
                    result["failed"] += 1
                    result["errors"].append(f"{game_id}: null_state")
            except Exception as e:
                result["failed"] += 1
                error_str = str(e)[:100]
                result["errors"].append(f"{game_id}: {error_str}")

        # Consider compliant if >80% pass
        pass_rate = result["passed"] / max(result["sample_tested"], 1)
        result["compliant"] = pass_rate >= 0.8

    except Exception as e:
        result["error_msg"] = str(e)[:200]
        result["compliant"] = False

    return result


def scan_directory(directory: str, sample_size: int = 3) -> list:
    """Scan a directory for databases and test each one."""
    results = []

    db_files = list(Path(directory).rglob("*.db"))
    # Filter out journal/wal files
    db_files = [
        p for p in db_files if not any(x in str(p) for x in ["-shm", "-wal", "-journal"])
    ]

    print(f"Found {len(db_files)} databases to test")

    for i, db_path in enumerate(db_files):
        print(f"[{i+1}/{len(db_files)}] Testing: {db_path}")
        result = test_database_parity(str(db_path), sample_size)
        results.append(result)

        status = "PASS" if result["compliant"] else "FAIL"
        print(
            f"  {status}: {result['game_count']} games, {result['passed']}/{result['sample_tested']} passed"
        )
        if result["errors"]:
            for err in result["errors"][:2]:
                print(f"    Error: {err[:80]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Audit databases for parity compliance")
    parser.add_argument("--path", required=True, help="Directory to scan")
    parser.add_argument("--sample-size", type=int, default=3, help="Games to test per DB")
    parser.add_argument("--output", help="JSON output file")
    parser.add_argument(
        "--delete-failed", action="store_true", help="Delete non-compliant databases"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    args = parser.parse_args()

    print(f"Auditing: {args.path}")
    print(f"Sample size: {args.sample_size}")
    print("=" * 60)

    results = scan_directory(args.path, args.sample_size)

    # Summary
    compliant = [r for r in results if r["compliant"]]
    non_compliant = [r for r in results if not r["compliant"]]

    total_compliant_size = sum(r["size_mb"] for r in compliant)
    total_non_compliant_size = sum(r["size_mb"] for r in non_compliant)
    total_compliant_games = sum(r["game_count"] for r in compliant)
    total_non_compliant_games = sum(r["game_count"] for r in non_compliant)

    print("=" * 60)
    print("SUMMARY:")
    print(
        f"  Compliant: {len(compliant)} databases, {total_compliant_games:,} games, {total_compliant_size:.1f}MB"
    )
    print(
        f"  Non-compliant: {len(non_compliant)} databases, {total_non_compliant_games:,} games, {total_non_compliant_size:.1f}MB"
    )
    print(f"  Space to reclaim: {total_non_compliant_size:.1f}MB")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "summary": {
                        "compliant_count": len(compliant),
                        "non_compliant_count": len(non_compliant),
                        "compliant_size_mb": total_compliant_size,
                        "non_compliant_size_mb": total_non_compliant_size,
                        "compliant_games": total_compliant_games,
                        "non_compliant_games": total_non_compliant_games,
                    },
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nResults written to: {args.output}")

    if args.delete_failed or args.dry_run:
        prefix = "DRY RUN - " if args.dry_run else ""
        print(f"\n{prefix}Deleting non-compliant databases:")
        deleted_size = 0
        deleted_count = 0
        for r in non_compliant:
            if r["game_count"] > 0:  # Don't delete empty DBs
                print(f"  {r['path']} ({r['size_mb']:.1f}MB, {r['game_count']} games)")
                if not args.dry_run:
                    try:
                        os.remove(r["path"])
                        # Also remove associated files
                        for ext in ["-shm", "-wal", "-journal"]:
                            assoc = r["path"] + ext
                            if os.path.exists(assoc):
                                os.remove(assoc)
                        print("    DELETED")
                        deleted_size += r["size_mb"]
                        deleted_count += 1
                    except Exception as e:
                        print(f"    ERROR: {e}")
        if not args.dry_run:
            print(f"\nDeleted {deleted_count} databases, freed {deleted_size:.1f}MB")


if __name__ == "__main__":
    main()
