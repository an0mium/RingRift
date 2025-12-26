#!/usr/bin/env python3
"""Elo Progress Monitor - Track progress towards 2000+ Elo across all configs.

December 2025 - Phase 2 quick win for progress tracking.

Usage:
    python scripts/elo_progress_monitor.py
    python scripts/elo_progress_monitor.py --watch  # Continuous monitoring
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

# Target Elo for all configs
TARGET_ELO = 2000.0

# All 12 canonical configurations
CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


def get_elo_db_path() -> Path:
    """Get path to unified Elo database."""
    candidates = [
        Path("data/unified_elo.db"),
        Path("/Users/armand/Development/RingRift/ai-service/data/unified_elo.db"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find unified_elo.db")


def get_config_stats(conn: sqlite3.Connection, board_type: str, num_players: int) -> dict:
    """Get Elo stats for a specific configuration."""
    cursor = conn.execute("""
        SELECT
            participant_id,
            rating,
            games_played,
            wins,
            losses,
            peak_rating
        FROM elo_ratings
        WHERE board_type = ? AND num_players = ?
        ORDER BY rating DESC
        LIMIT 1
    """, (board_type, num_players))

    row = cursor.fetchone()
    if row:
        return {
            "participant_id": row[0],
            "rating": row[1],
            "games": row[2],
            "wins": row[3],
            "losses": row[4],
            "peak": row[5],
            "gap": TARGET_ELO - row[1],
        }
    return {
        "participant_id": "none",
        "rating": 0,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "peak": 0,
        "gap": TARGET_ELO,
    }


def print_progress_report(conn: sqlite3.Connection) -> None:
    """Print Elo progress report for all configs."""
    print(f"\n{'='*80}")
    print(f"ELO PROGRESS MONITOR - Target: {TARGET_ELO}")
    print(f"{'='*80}")
    print(f"{'Config':<15} {'Best Elo':>10} {'Gap':>8} {'Games':>7} {'W/L':>10} {'Status':<10}")
    print(f"{'-'*80}")

    at_target = 0
    close_to_target = 0

    for board_type, num_players in CONFIGS:
        config_key = f"{board_type}_{num_players}p"
        stats = get_config_stats(conn, board_type, num_players)

        # Determine status
        if stats["rating"] >= TARGET_ELO:
            status = "✅ DONE"
            at_target += 1
        elif stats["gap"] <= 50:
            status = "🔥 CLOSE"
            close_to_target += 1
        elif stats["gap"] <= 200:
            status = "📈 MAKING"
        elif stats["games"] > 0:
            status = "🔄 TRAINING"
        else:
            status = "⏳ NO DATA"

        wl = f"{stats['wins']}/{stats['losses']}" if stats["games"] > 0 else "-"

        print(f"{config_key:<15} {stats['rating']:>10.1f} {stats['gap']:>+8.1f} {stats['games']:>7} {wl:>10} {status:<10}")

    print(f"{'-'*80}")
    print(f"Summary: {at_target}/12 at target, {close_to_target} close (<50 Elo gap)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor Elo progress towards 2000+")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval (seconds)")
    args = parser.parse_args()

    try:
        db_path = get_elo_db_path()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))

    try:
        if args.watch:
            print(f"Watching Elo progress (refresh every {args.interval}s, Ctrl+C to stop)...")
            while True:
                print_progress_report(conn)
                time.sleep(args.interval)
        else:
            print_progress_report(conn)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
