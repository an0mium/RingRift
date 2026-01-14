#!/usr/bin/env python3
"""Elo velocity dashboard - shows per-config Elo/hour and progress to target.

Displays:
- Current Elo ratings for canonical models across all 12 configs
- Elo velocity (Elo gained per hour) over configurable lookback period
- Gap to target Elo and estimated time to reach target
- Color-coded status indicators

Usage:
    cd ai-service

    # One-time report
    python scripts/elo_velocity_report.py

    # Watch mode (updates every 60 seconds)
    python scripts/elo_velocity_report.py --watch

    # Custom lookback period
    python scripts/elo_velocity_report.py --hours 48

    # Custom target Elo
    python scripts/elo_velocity_report.py --target 2000
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_TARGET_ELO = 1600
DB_PATH = Path(__file__).parent.parent / "data" / "unified_elo.db"


def get_elo_velocity(db_path: Path, hours: int = 24, target_elo: float = DEFAULT_TARGET_ELO) -> dict:
    """Calculate Elo velocity (Elo gained per hour) for each config.

    Returns dict of config_key -> {rating, velocity, games, gap, hours_to_target}
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    now = time.time()
    cutoff = now - (hours * 3600)  # Unix timestamp for lookback

    # Get current ratings for canonical models
    cursor.execute("""
        SELECT
            r.participant_id,
            r.board_type,
            r.num_players,
            r.rating,
            r.games_played
        FROM elo_ratings r
        WHERE r.participant_id LIKE 'canonical_%'
        ORDER BY r.board_type, r.num_players
    """)

    results = {}
    for participant_id, board_type, num_players, rating, games in cursor.fetchall():
        config = f"{board_type}_{num_players}p"

        # Find historical rating from lookback period
        cursor.execute("""
            SELECT rating
            FROM rating_history
            WHERE participant_id = ?
              AND board_type = ?
              AND num_players = ?
              AND timestamp < ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (participant_id, board_type, num_players, cutoff))

        old_row = cursor.fetchone()
        if old_row:
            old_rating = old_row[0]
            velocity = (rating - old_rating) / hours
        else:
            # No historical data, check if there's ANY history
            cursor.execute("""
                SELECT MIN(timestamp), MIN(rating)
                FROM rating_history
                WHERE participant_id = ?
                  AND board_type = ?
                  AND num_players = ?
            """, (participant_id, board_type, num_players))
            first_row = cursor.fetchone()
            if first_row and first_row[0]:
                first_timestamp, first_rating = first_row
                elapsed_hours = (now - first_timestamp) / 3600
                if elapsed_hours > 0:
                    velocity = (rating - first_rating) / elapsed_hours
                else:
                    velocity = 0.0
            else:
                velocity = 0.0

        gap = target_elo - rating
        if velocity > 0 and gap > 0:
            hours_to_target = gap / velocity
        elif gap <= 0:
            hours_to_target = 0  # Already at target
        else:
            hours_to_target = float('inf')

        # Use the latest entry if multiple canonical models exist for same config
        if config not in results or rating > results[config]['rating']:
            results[config] = {
                'participant': participant_id,
                'rating': rating,
                'velocity': velocity,
                'games': games,
                'gap': gap,
                'hours_to_target': hours_to_target,
            }

    conn.close()
    return results


def print_report(results: dict, target_elo: float, hours: int):
    """Print formatted velocity report."""
    print(f"\n{'Config':<15} {'Rating':>8} {'Velocity':>12} {'Gap':>8} {'ETA':>12} {'Games':>8} {'Status'}")
    print("-" * 80)

    # Sort by board type then player count
    def sort_key(item):
        config = item[0]
        parts = config.rsplit('_', 1)
        board = parts[0]
        players = int(parts[1].replace('p', ''))
        # Sort order: hex8, hexagonal, square8, square19
        board_order = {'hex8': 0, 'hexagonal': 1, 'square8': 2, 'square19': 3}
        return (board_order.get(board, 99), players)

    sorted_items = sorted(results.items(), key=sort_key)

    for config, data in sorted_items:
        rating = data['rating']
        velocity = data['velocity']
        gap = data['gap']
        hours_to_target = data['hours_to_target']
        games = data['games']

        # Format ETA
        if hours_to_target == 0:
            eta = "DONE"
        elif hours_to_target < 24:
            eta = f"{hours_to_target:.1f}h"
        elif hours_to_target < 168:  # 1 week
            eta = f"{hours_to_target / 24:.1f}d"
        elif hours_to_target < 1000:
            eta = f"{hours_to_target / 168:.1f}w"
        else:
            eta = "---"

        # Status indicator
        if rating >= target_elo:
            status = "TARGET"
        elif velocity > 1.0:
            status = "FAST"
        elif velocity > 0.1:
            status = "OK"
        elif velocity > 0:
            status = "SLOW"
        else:
            status = "STALLED"

        # Color-code velocity
        vel_str = f"{velocity:+.2f}/h"

        print(f"{config:<15} {rating:>8.1f} {vel_str:>12} {gap:>+8.0f} {eta:>12} {games:>8} {status}")

    # Summary stats
    print("-" * 80)
    at_target = sum(1 for d in results.values() if d['rating'] >= target_elo)
    stalled = sum(1 for d in results.values() if d['velocity'] <= 0)
    avg_velocity = sum(d['velocity'] for d in results.values()) / len(results) if results else 0

    print(f"Summary: {at_target}/{len(results)} at target | "
          f"{stalled} stalled | "
          f"Avg velocity: {avg_velocity:+.2f}/h")


def main():
    parser = argparse.ArgumentParser(
        description="Elo velocity dashboard - track training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/elo_velocity_report.py              # One-time report
    python scripts/elo_velocity_report.py --watch      # Live updates every 60s
    python scripts/elo_velocity_report.py --hours 48   # 48-hour lookback
    python scripts/elo_velocity_report.py --target 2000  # Custom target Elo
        """
    )
    parser.add_argument("--watch", action="store_true", help="Watch mode (continuous updates)")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds (default: 60)")
    parser.add_argument("--hours", type=int, default=24, help="Lookback period in hours (default: 24)")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET_ELO, help=f"Target Elo (default: {DEFAULT_TARGET_ELO})")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)

    while True:
        # Clear screen in watch mode
        if args.watch:
            print("\033[2J\033[H", end="")  # ANSI clear screen

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"=== Elo Velocity Report ({timestamp}) ===")
        print(f"Target: {args.target:.0f} Elo | Lookback: {args.hours}h")

        try:
            results = get_elo_velocity(DB_PATH, hours=args.hours, target_elo=args.target)
            if results:
                print_report(results, args.target, args.hours)
            else:
                print("\nNo canonical models found in database.")
                print("Run gauntlet evaluations to populate Elo ratings.")
        except sqlite3.Error as e:
            print(f"\nDatabase error: {e}")

        if not args.watch:
            break

        print(f"\nRefreshing in {args.interval}s... (Ctrl+C to exit)")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
