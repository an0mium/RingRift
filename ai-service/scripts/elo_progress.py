#!/usr/bin/env python3
"""Elo Progress CLI - View training improvement over time.

Usage:
    # Take a snapshot now
    python scripts/elo_progress.py --snapshot

    # View progress report (default: 7 days)
    python scripts/elo_progress.py --report

    # View progress for specific config
    python scripts/elo_progress.py --report --config hex8_2p

    # View last 30 days
    python scripts/elo_progress.py --report --days 30

    # Export as CSV
    python scripts/elo_progress.py --report --csv > progress.csv

December 31, 2025: Created for training loop effectiveness monitoring.
"""

import argparse
import asyncio
import csv
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(
        description="Elo Progress - Track training improvement over time"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Take a snapshot of current best model Elo for all configs",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print progress report",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=7.0,
        help="Number of days to include in report (default: 7)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Specific config to report on (e.g., hex8_2p)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output report as CSV",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show full history for a config (requires --config)",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Show training run statistics with before/after Elo deltas",
    )

    args = parser.parse_args()

    # Default to report if no action specified
    if not args.snapshot and not args.report and not args.history and not args.training:
        args.report = True

    if args.snapshot:
        run_snapshot()

    if args.report:
        if args.csv:
            print_report_csv(args.days, args.config)
        else:
            print_report(args.days, args.config)

    if args.history:
        if not args.config:
            print("Error: --history requires --config", file=sys.stderr)
            sys.exit(1)
        print_history(args.config, args.days)

    if args.training:
        print_training_stats(args.config, args.days)


def run_snapshot():
    """Take a snapshot of all configs."""
    from app.coordination.elo_progress_tracker import snapshot_all_configs

    print("Taking Elo snapshots for all configs...")
    results = asyncio.run(snapshot_all_configs())

    print("\nResults:")
    for config_key, snapshot in sorted(results.items()):
        if snapshot:
            print(
                f"  {config_key}: {snapshot.best_model_id[:40]}... "
                f"@ {snapshot.best_elo:.1f} Elo ({snapshot.games_played} games)"
            )
        else:
            print(f"  {config_key}: No data")

    print(f"\nTotal: {sum(1 for v in results.values() if v)}/{len(results)} configs recorded")


def print_report(days: float, config_key: str | None = None):
    """Print progress report."""
    from app.coordination.elo_progress_tracker import (
        get_elo_progress_tracker,
        print_progress_summary,
    )

    if config_key:
        tracker = get_elo_progress_tracker()
        report = tracker.get_progress_report(config_key, days=days)

        print(f"\nProgress Report for {config_key} (last {days:.1f} days)")
        print("=" * 50)

        if report.num_snapshots == 0:
            print("No data available")
            return

        print(f"Snapshots: {report.num_snapshots}")
        print(f"Start Elo: {report.start_elo:.1f}" if report.start_elo else "Start Elo: N/A")
        print(f"End Elo: {report.end_elo:.1f}" if report.end_elo else "End Elo: N/A")
        print(f"Delta: {report.elo_delta:+.1f}" if report.elo_delta else "Delta: N/A")

        if report.improvement_rate_per_day:
            print(f"Rate: {report.improvement_rate_per_day:+.2f} Elo/day")

        trend = "Improving" if report.is_improving else "Declining" if report.elo_delta and report.elo_delta < 0 else "Stable"
        print(f"Trend: {trend}")

        if report.start_time and report.end_time:
            print(f"Period: {report.start_time.strftime('%Y-%m-%d %H:%M')} to {report.end_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        print_progress_summary(days=days)


def print_report_csv(days: float, config_key: str | None = None):
    """Print progress report as CSV."""
    from app.coordination.elo_progress_tracker import get_elo_progress_tracker

    tracker = get_elo_progress_tracker()
    writer = csv.writer(sys.stdout)

    # Header
    writer.writerow([
        "config_key", "start_elo", "end_elo", "elo_delta",
        "rate_per_day", "num_snapshots", "is_improving"
    ])

    if config_key:
        configs = [config_key]
    else:
        from app.coordination.elo_progress_tracker import ALL_CONFIGS
        configs = [f"{bt}_{np}p" for bt, np in ALL_CONFIGS]

    for cfg in configs:
        report = tracker.get_progress_report(cfg, days=days)
        writer.writerow([
            cfg,
            f"{report.start_elo:.1f}" if report.start_elo else "",
            f"{report.end_elo:.1f}" if report.end_elo else "",
            f"{report.elo_delta:.1f}" if report.elo_delta else "",
            f"{report.improvement_rate_per_day:.2f}" if report.improvement_rate_per_day else "",
            report.num_snapshots,
            report.is_improving,
        ])


def print_history(config_key: str, days: float):
    """Print full history for a config."""
    from app.coordination.elo_progress_tracker import get_elo_progress_tracker
    import time

    tracker = get_elo_progress_tracker()
    since = time.time() - (days * 86400)
    snapshots = tracker.get_snapshots(config_key, since_timestamp=since, limit=1000)

    print(f"\nElo History for {config_key} (last {days:.1f} days)")
    print("=" * 70)
    print(f"{'Timestamp':<20} {'Elo':>8} {'Games':>8} {'Model ID':<30}")
    print("-" * 70)

    for snapshot in snapshots:
        ts = datetime.fromtimestamp(snapshot.timestamp, tz=timezone.utc)
        print(
            f"{ts.strftime('%Y-%m-%d %H:%M'):<20} "
            f"{snapshot.best_elo:>8.1f} "
            f"{snapshot.games_played:>8} "
            f"{snapshot.best_model_id[:30]:<30}"
        )

    print("=" * 70)
    print(f"Total snapshots: {len(snapshots)}")


def print_training_stats(config_key: str | None = None, days: float = 7.0):
    """Print training run statistics with before/after Elo deltas.

    Jan 6, 2026: Added for P4 - training progress visibility.
    Shows before_elo and final_elo from training_history to demonstrate
    that training runs are producing model improvements.
    """
    import sqlite3
    import time
    from pathlib import Path

    # Use the training coordinator's database
    from app.utils.paths import DATA_DIR
    db_path = DATA_DIR / "training_coordination.db"

    if not db_path.exists():
        print(f"Training database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check if training_history table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='training_history'"
    )
    if not cursor.fetchone():
        print(f"\ntraining_history table not found in {db_path}")
        print("This table is created when training runs complete.")
        print("Run a training job first to populate this data.")
        conn.close()
        return

    since = time.time() - (days * 86400)

    # Build query
    query = """
        SELECT
            job_id,
            board_type,
            num_players,
            node_name,
            started_at,
            completed_at,
            status,
            final_val_loss,
            final_elo,
            before_elo,
            epochs_completed
        FROM training_history
        WHERE completed_at > ?
    """
    params = [since]

    if config_key:
        # Parse config_key (e.g., "hex8_2p" -> "hex8", 2)
        parts = config_key.rsplit("_", 1)
        if len(parts) == 2 and parts[1].endswith("p"):
            board_type = parts[0]
            num_players = int(parts[1][:-1])
            query += " AND board_type = ? AND num_players = ?"
            params.extend([board_type, num_players])

    query += " ORDER BY completed_at DESC"

    cursor = conn.execute(query, params)
    rows = list(cursor.fetchall())
    conn.close()

    if not rows:
        print(f"\nNo training runs found in the last {days:.1f} days")
        if config_key:
            print(f"  (filtered by config: {config_key})")
        return

    print(f"\nTraining Run Statistics (last {days:.1f} days)")
    if config_key:
        print(f"  Filtered by config: {config_key}")
    print("=" * 90)
    print(f"{'Config':<14} {'Status':<10} {'Before':<8} {'After':<8} {'Delta':<8} {'Epochs':<8} {'Node':<20}")
    print("-" * 90)

    total_delta = 0.0
    completed_count = 0
    improved_count = 0

    for row in rows:
        config = f"{row['board_type']}_{row['num_players']}p"
        status = row["status"] or "unknown"
        before = row["before_elo"] or 0.0
        after = row["final_elo"] or 0.0
        delta = after - before if before > 0 and after > 0 else None

        epochs = row["epochs_completed"] or 0
        node = (row["node_name"] or "unknown")[:20]

        # Format values
        before_str = f"{before:.0f}" if before > 0 else "-"
        after_str = f"{after:.0f}" if after > 0 else "-"
        delta_str = f"{delta:+.0f}" if delta is not None else "-"

        print(f"{config:<14} {status:<10} {before_str:<8} {after_str:<8} {delta_str:<8} {epochs:<8} {node:<20}")

        if status == "completed":
            completed_count += 1
            if delta is not None:
                total_delta += delta
                if delta > 0:
                    improved_count += 1

    print("=" * 90)
    print(f"Total training runs: {len(rows)}")
    print(f"Completed: {completed_count}")

    if completed_count > 0:
        avg_delta = total_delta / completed_count if completed_count > 0 else 0
        print(f"Average Elo delta: {avg_delta:+.1f}")
        print(f"Improved: {improved_count}/{completed_count} ({100*improved_count/completed_count:.0f}%)")

        if avg_delta > 0:
            print("\n✓ Evidence of improvement: Training runs are producing better models!")
        elif avg_delta == 0:
            print("\n⚠ No net improvement: Training may be stalled")
        else:
            print("\n⚠ Regression detected: Models getting worse - investigate training data quality")


if __name__ == "__main__":
    main()
