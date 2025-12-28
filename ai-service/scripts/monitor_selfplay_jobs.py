#!/usr/bin/env python3
"""Monitor selfplay jobs across cluster and trigger sync on completion.

Usage:
    # One-shot status check
    python scripts/monitor_selfplay_jobs.py

    # Watch mode (check every 5 minutes)
    python scripts/monitor_selfplay_jobs.py --watch

    # Watch with custom interval
    python scripts/monitor_selfplay_jobs.py --watch --interval 120

December 28, 2025 - Job monitoring for selfplay automation.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Nodes to monitor
MONITORED_NODES = [
    "nebius-backbone-1",
    "nebius-h100-3",
    "vultr-a100-20gb",
]

# Screen session patterns that indicate selfplay
SELFPLAY_PATTERNS = [
    "selfplay",
    "square8",
    "square19",
    "hex8",
    "hexagonal",
]


@dataclass
class JobStatus:
    """Status of a selfplay job."""

    node: str
    session_name: str
    started_at: str
    is_running: bool
    config: str = ""


def run_ssh_command(host: str, command: str, timeout: int = 30) -> tuple[bool, str]:
    """Run SSH command and return success status and output."""
    try:
        result = subprocess.run(
            ["ssh", host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def get_running_jobs(node: str) -> list[JobStatus]:
    """Get list of running selfplay jobs on a node."""
    jobs = []

    # Get screen sessions
    success, output = run_ssh_command(node, "screen -ls 2>/dev/null")
    if not success:
        logger.warning(f"Could not connect to {node}")
        return jobs

    for line in output.split("\n"):
        line = line.strip()
        if not line or "Socket" in line or "screen" in line.lower():
            continue

        # Parse screen session line: "12345.session_name (date) (status)"
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        session_info = parts[0].strip()
        if "." not in session_info:
            continue

        pid_session = session_info.split(".")
        if len(pid_session) < 2:
            continue

        session_name = pid_session[1]

        # Check if this is a selfplay session
        is_selfplay = any(p in session_name.lower() for p in SELFPLAY_PATTERNS)
        if not is_selfplay:
            continue

        # Extract date if available
        started_at = ""
        if len(parts) >= 2:
            date_part = parts[1].strip().strip("()")
            started_at = date_part

        # Determine config from session name
        config = ""
        for pattern in ["hex8", "square8", "square19", "hexagonal"]:
            if pattern in session_name.lower():
                for players in ["2p", "3p", "4p", "_2", "_3", "_4"]:
                    if players in session_name:
                        p = players.replace("_", "").replace("p", "")
                        config = f"{pattern}_{p}p"
                        break
                if not config:
                    config = pattern
                break

        jobs.append(
            JobStatus(
                node=node,
                session_name=session_name,
                started_at=started_at,
                is_running=True,
                config=config,
            )
        )

    return jobs


def check_job_progress(node: str, session_name: str) -> dict:
    """Check progress of a specific job by reading its log."""
    log_patterns = [
        f"logs/{session_name}.log",
        f"logs/{session_name.replace('selfplay_', '')}.log",
    ]

    for log_path in log_patterns:
        success, output = run_ssh_command(
            node,
            f"tail -5 ~/ringrift/ai-service/{log_path} 2>/dev/null",
        )
        if success and output:
            # Try to extract progress info
            lines = output.split("\n")
            for line in reversed(lines):
                if "game" in line.lower() and ("/" in line or "completed" in line.lower()):
                    return {"log": log_path, "last_line": line}
            return {"log": log_path, "last_line": lines[-1] if lines else ""}

    return {"log": "not found", "last_line": ""}


def trigger_data_sync(source_node: str, config: str) -> bool:
    """Trigger data sync from a node after job completion."""
    logger.info(f"Triggering sync from {source_node} for {config}")

    # Use rsync to pull data
    local_dir = Path("data/games")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Find the selfplay database
    success, output = run_ssh_command(
        source_node,
        f"find ~/ringrift/ai-service/data/selfplay -name '*.db' -newer ~/ringrift/ai-service/data/selfplay -mmin -60 2>/dev/null | head -5",
    )

    if success and output:
        for db_path in output.split("\n"):
            if db_path.strip():
                logger.info(f"Found new database: {db_path}")
                # Could trigger rsync here
                # For now just log it
                return True

    return False


def print_status(all_jobs: list[JobStatus]) -> None:
    """Print formatted status of all jobs."""
    print("\n" + "=" * 70)
    print(f"SELFPLAY JOB STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not all_jobs:
        print("No selfplay jobs running.")
        return

    # Group by node
    by_node: dict[str, list[JobStatus]] = {}
    for job in all_jobs:
        by_node.setdefault(job.node, []).append(job)

    for node, jobs in sorted(by_node.items()):
        print(f"\n{node}:")
        for job in jobs:
            status = "RUNNING" if job.is_running else "STOPPED"
            config = f"[{job.config}]" if job.config else ""
            print(f"  {job.session_name:30s} {status:8s} {config}")
            if job.started_at:
                print(f"    Started: {job.started_at}")

    print("\n" + "-" * 70)
    print(f"Total: {len(all_jobs)} jobs across {len(by_node)} nodes")
    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor selfplay jobs across cluster")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor (default: one-shot)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds for watch mode (default: 300)",
    )
    parser.add_argument(
        "--sync-on-complete",
        action="store_true",
        help="Trigger data sync when jobs complete",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="Comma-separated list of nodes to monitor",
    )
    args = parser.parse_args()

    nodes = args.nodes.split(",") if args.nodes else MONITORED_NODES

    previous_jobs: set[str] = set()

    while True:
        all_jobs = []
        for node in nodes:
            jobs = get_running_jobs(node)
            all_jobs.extend(jobs)

        print_status(all_jobs)

        # Check for completed jobs
        current_jobs = {f"{j.node}:{j.session_name}" for j in all_jobs}
        completed = previous_jobs - current_jobs

        if completed and args.sync_on_complete:
            for job_id in completed:
                node, session = job_id.split(":", 1)
                logger.info(f"Job completed: {session} on {node}")
                # Could trigger sync here

        previous_jobs = current_jobs

        if not args.watch:
            break

        logger.info(f"Next check in {args.interval} seconds...")
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
