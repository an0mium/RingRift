#!/usr/bin/env python3
"""CLI for Elo Rating Reconciliation.

Provides command-line tools for managing Elo rating consistency across
distributed P2P nodes. Complements the unified_ai_loop.py by providing
manual reconciliation capabilities.

Usage:
    # Check for Elo drift
    python scripts/elo_reconciliation_cli.py check-drift

    # Sync from a specific remote host
    python scripts/elo_reconciliation_cli.py sync --host 192.168.1.100

    # Run full reconciliation across all configured hosts
    python scripts/elo_reconciliation_cli.py reconcile-all

    # Show detailed drift report with JSON output
    python scripts/elo_reconciliation_cli.py check-drift --json --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.training.elo_reconciliation import (
    EloDrift,
    EloReconciler,
    ReconciliationReport,
    SyncResult,
    check_elo_drift,
    sync_elo_from_remote,
)


def format_drift_report(drift: EloDrift, verbose: bool = False) -> str:
    """Format drift report for console output."""
    lines = [
        "=== Elo Drift Report ===",
        f"Source: {drift.source}",
        f"Target: {drift.target}",
        f"Checked at: {drift.checked_at}",
        "",
        f"Participants in source: {drift.participants_in_source}",
        f"Participants in target: {drift.participants_in_target}",
        f"Participants in both: {drift.participants_in_both}",
        "",
        f"Max rating diff: {drift.max_rating_diff:.1f}",
        f"Avg rating diff: {drift.avg_rating_diff:.1f}",
        f"Significant drift: {'YES' if drift.is_significant else 'No'}",
    ]

    if drift.is_significant:
        lines.append("")
        lines.append("WARNING: Significant drift detected!")
        lines.append("Consider running a full reconciliation.")

    if verbose and drift.rating_diffs:
        lines.append("")
        lines.append("=== Rating Differences ===")
        sorted_diffs = sorted(
            drift.rating_diffs.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for participant, diff in sorted_diffs[:20]:  # Top 20
            sign = "+" if diff > 0 else ""
            lines.append(f"  {participant}: {sign}{diff:.1f}")
        if len(drift.rating_diffs) > 20:
            lines.append(f"  ... and {len(drift.rating_diffs) - 20} more")

    return "\n".join(lines)


def format_sync_result(result: SyncResult) -> str:
    """Format sync result for console output."""
    lines = [
        f"=== Sync Result from {result.remote_host} ===",
        f"Synced at: {result.synced_at}",
        f"Matches added: {result.matches_added}",
        f"Matches skipped (duplicates): {result.matches_skipped}",
        f"Conflicts detected: {result.matches_conflict}",
        f"Participants added: {result.participants_added}",
    ]

    if result.error:
        lines.append("")
        lines.append(f"ERROR: {result.error}")

    return "\n".join(lines)


def cmd_check_drift(args: argparse.Namespace) -> int:
    """Check for Elo drift."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
    )

    remote_db = Path(args.remote_db) if args.remote_db else None

    drift = reconciler.check_drift(
        remote_db_path=remote_db,
        board_type=args.board_type,
        num_players=args.num_players,
    )

    if args.json:
        print(json.dumps(drift.to_dict(), indent=2))
    else:
        print(format_drift_report(drift, verbose=args.verbose))

    # Return non-zero if significant drift
    return 1 if drift.is_significant else 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Sync from a remote host."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
        ssh_timeout=args.timeout,
    )

    result = reconciler.sync_from_remote(
        remote_host=args.host,
        remote_db_path=args.remote_path,
        ssh_user=args.user,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_sync_result(result))

    return 0 if result.error is None else 1


def cmd_reconcile_all(args: argparse.Namespace) -> int:
    """Run full reconciliation across all hosts."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
        ssh_timeout=args.timeout,
    )

    hosts = args.hosts.split(",") if args.hosts else None

    report = reconciler.reconcile_all(hosts=hosts)

    if args.json:
        output = {
            "started_at": report.started_at,
            "completed_at": report.completed_at,
            "nodes_synced": report.nodes_synced,
            "nodes_failed": report.nodes_failed,
            "total_matches_added": report.total_matches_added,
            "total_matches_skipped": report.total_matches_skipped,
            "total_conflicts": report.total_conflicts,
            "drift_detected": report.drift_detected,
            "max_drift": report.max_drift,
            "sync_results": [r.to_dict() for r in report.sync_results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(report.summary())

        if args.verbose:
            print("\n=== Per-Host Results ===")
            for result in report.sync_results:
                print()
                print(format_sync_result(result))

    return 1 if report.nodes_failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show current status and configuration."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
    )

    status = {
        "local_db_path": str(reconciler.local_db_path),
        "local_db_exists": reconciler.local_db_path.exists(),
        "remote_hosts_config": str(reconciler.remote_hosts_config),
        "config_exists": reconciler.remote_hosts_config.exists(),
        "ssh_timeout": reconciler.ssh_timeout,
    }

    # Check local DB stats
    if reconciler.local_db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(reconciler.local_db_path), timeout=10)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM participants")
            status["total_participants"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM match_history")
            status["total_matches"] = cur.fetchone()[0]
        except sqlite3.OperationalError:
            status["total_participants"] = "N/A (table not found)"
            status["total_matches"] = "N/A (table not found)"
        finally:
            conn.close()

    # Load configured hosts
    hosts = reconciler._load_p2p_hosts()
    status["configured_hosts"] = hosts
    status["num_configured_hosts"] = len(hosts)

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print("=== Elo Reconciliation Status ===")
        print(f"Local DB: {status['local_db_path']}")
        print(f"  Exists: {status['local_db_exists']}")
        if status['local_db_exists']:
            print(f"  Participants: {status.get('total_participants', 'N/A')}")
            print(f"  Matches: {status.get('total_matches', 'N/A')}")
        print()
        print(f"Config: {status['remote_hosts_config']}")
        print(f"  Exists: {status['config_exists']}")
        print(f"  Configured hosts: {status['num_configured_hosts']}")
        if hosts:
            for host in hosts:
                print(f"    - {host}")
        print()
        print(f"SSH timeout: {status['ssh_timeout']}s")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Elo Rating Reconciliation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--local-db",
        type=str,
        help="Path to local Elo database (default: data/unified_elo.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check-drift command
    drift_parser = subparsers.add_parser(
        "check-drift",
        help="Check for Elo drift between local and remote databases",
    )
    drift_parser.add_argument(
        "--remote-db",
        type=str,
        help="Path to remote database (local copy) to compare against",
    )
    drift_parser.add_argument(
        "--board-type",
        type=str,
        help="Filter to specific board type",
    )
    drift_parser.add_argument(
        "--num-players",
        type=int,
        help="Filter to specific number of players",
    )

    # sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync match history from a remote host",
    )
    sync_parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="Remote host IP or hostname",
    )
    sync_parser.add_argument(
        "--remote-path",
        type=str,
        default="~/ringrift/ai-service/data/unified_elo.db",
        help="Path to Elo database on remote host",
    )
    sync_parser.add_argument(
        "--user",
        type=str,
        default="ubuntu",
        help="SSH username (default: ubuntu)",
    )
    sync_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="SSH timeout in seconds (default: 30)",
    )

    # reconcile-all command
    reconcile_parser = subparsers.add_parser(
        "reconcile-all",
        help="Run full reconciliation across all configured hosts",
    )
    reconcile_parser.add_argument(
        "--hosts",
        type=str,
        help="Comma-separated list of hosts to sync (overrides config)",
    )
    reconcile_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="SSH timeout in seconds (default: 30)",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show current status and configuration",
    )

    # Add common args to all subparsers for convenience
    for subparser in [drift_parser, sync_parser, reconcile_parser, status_parser]:
        subparser.add_argument(
            "--json",
            action="store_true",
            help="Output in JSON format",
            dest="subcommand_json",
        )
        subparser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Verbose output",
            dest="subcommand_verbose",
        )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Merge global and subcommand flags
    args.json = getattr(args, "json", False) or getattr(args, "subcommand_json", False)
    args.verbose = getattr(args, "verbose", False) or getattr(args, "subcommand_verbose", False)

    if args.command == "check-drift":
        return cmd_check_drift(args)
    elif args.command == "sync":
        return cmd_sync(args)
    elif args.command == "reconcile-all":
        return cmd_reconcile_all(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
