#!/usr/bin/env python3
"""Periodic cluster health monitoring with structured logging.

This script monitors cluster health and records observations to JSONL
for later analysis. Use for 60-minute observation periods.

Usage:
    # Run 60-minute observation (default)
    python scripts/cluster_health_monitor.py

    # Custom duration and interval
    python scripts/cluster_health_monitor.py --duration 60 --interval 10

    # Analyze previous observation
    python scripts/cluster_health_monitor.py --analyze

    # Custom log file
    python scripts/cluster_health_monitor.py --log-file my_observations.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/cluster_monitor.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_DURATION_MINUTES = 60
DEFAULT_INTERVAL_MINUTES = 10
DEFAULT_LOG_FILE = Path("cluster_health_log.jsonl")
DISK_WARNING_THRESHOLD = 80  # percent
DISK_CRITICAL_THRESHOLD = 90  # percent
DISK_FREE_CRITICAL_GB = 10  # GB
MEMORY_WARNING_THRESHOLD = 85  # percent
HEARTBEAT_STALE_THRESHOLD = 120  # seconds

# SSH config for cluster access
SSH_OPTS = "-o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no"

# Known cluster entry points (fallback when localhost unavailable)
# Use Tailscale IPs for reliable connectivity
CLUSTER_ENTRY_POINTS = [
    ("100.126.21.102", 22, "root"),   # hetzner-cpu3 (voter)
    ("100.94.174.19", 22, "root"),    # hetzner-cpu1 (voter)
    ("100.67.131.72", 22, "root"),    # hetzner-cpu2 (voter)
    ("100.94.201.92", 22, "root"),    # vultr-a100-20gb (voter)
]


@dataclass
class ClusterStatus:
    """Snapshot of cluster health."""

    timestamp: str  # ISO format string for JSON serialization
    epoch: float  # Unix timestamp
    check_number: int
    leader_id: str
    leader_role: str
    quorum_ok: bool
    voters_alive: int
    total_nodes: int
    alive_peers: list[str] = field(default_factory=list)
    total_selfplay: int = 0
    total_training: int = 0
    disk_critical: list[str] = field(default_factory=list)
    disk_warning: list[str] = field(default_factory=list)
    memory_warning: list[str] = field(default_factory=list)
    stale_nodes: list[str] = field(default_factory=list)
    unhealthy_nodes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def run_ssh_command(host: str, port: int, user: str, cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Run SSH command and return (success, output)."""
    ssh_cmd = f"ssh {SSH_OPTS} -p {port} {user}@{host} '{cmd}'"
    try:
        result = subprocess.run(
            ssh_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        # Filter out Vast.ai welcome message
        lines = [l for l in output.split("\n") if not l.startswith("Welcome to vast") and not l.startswith("Have fun")]
        return result.returncode == 0, "\n".join(lines)
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def get_p2p_status() -> dict[str, Any] | None:
    """Fetch P2P status from any reachable cluster node.

    First tries localhost (for when running on a cluster node),
    then falls back to SSH to entry points.
    """
    import urllib.request
    import urllib.error

    # First try localhost - we might be running on a cluster node
    try:
        with urllib.request.urlopen("http://localhost:8770/status", timeout=10) as response:
            data = response.read().decode("utf-8")
            if data.strip().startswith("{"):
                return json.loads(data.strip())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        pass  # Fall through to SSH approach

    # Fall back to SSH to entry points
    for host, port, user in CLUSTER_ENTRY_POINTS:
        success, output = run_ssh_command(
            host, port, user,
            "curl -s http://localhost:8770/status",
            timeout=15,
        )
        if success and output.strip().startswith("{"):
            try:
                return json.loads(output.strip())
            except json.JSONDecodeError:
                continue
    return None


def analyze_cluster_health(
    status: dict[str, Any], check_number: int = 0
) -> ClusterStatus:
    """Analyze P2P status and identify problems."""
    now = datetime.utcnow()
    errors = []

    peers = status.get("peers", {})
    alive_peers = status.get("alive_peers", [])

    # If alive_peers is not in status, try to derive from peers dict
    if not alive_peers and peers:
        alive_peers = [name for name, p in peers.items() if p.get("status") == "alive"]

    disk_critical = []
    disk_warning = []
    memory_warning = []
    stale_nodes = []
    unhealthy_nodes = []

    total_selfplay = 0
    total_training = 0

    for name, peer in peers.items():
        # Disk checks
        disk_free = peer.get("disk_free_gb", 999)
        disk_pct = peer.get("disk_percent", 0)
        if disk_free < DISK_FREE_CRITICAL_GB or disk_pct > DISK_CRITICAL_THRESHOLD:
            disk_critical.append(f"{name}: {disk_free:.1f}GB free ({disk_pct:.0f}%)")
        elif disk_pct > DISK_WARNING_THRESHOLD:
            disk_warning.append(f"{name}: {disk_free:.1f}GB free ({disk_pct:.0f}%)")

        # Memory checks
        mem_pct = peer.get("memory_percent", 0)
        if mem_pct > MEMORY_WARNING_THRESHOLD:
            memory_warning.append(f"{name}: {mem_pct:.0f}%")

        # Heartbeat checks
        stale_seconds = peer.get("seconds_since_heartbeat", 0)
        if stale_seconds > HEARTBEAT_STALE_THRESHOLD:
            stale_nodes.append(f"{name}: {stale_seconds:.0f}s")

        # Node health
        failures = peer.get("consecutive_failures", 0)
        errors_count = peer.get("errors_last_hour", 0)
        if failures >= 5 or errors_count > 10:
            unhealthy_nodes.append(f"{name}: failures={failures}, errors={errors_count}")

        # Job counts
        total_selfplay += peer.get("selfplay_jobs", 0)
        total_training += peer.get("training_jobs", 0)

    return ClusterStatus(
        timestamp=now.isoformat(),
        epoch=time.time(),
        check_number=check_number,
        leader_id=status.get("leader_id", "NONE"),
        leader_role=status.get("role", "unknown"),
        quorum_ok=status.get("voter_quorum_ok", False),
        voters_alive=status.get("voters_alive", 0),
        total_nodes=len(peers) if peers else len(alive_peers),
        alive_peers=alive_peers,
        total_selfplay=total_selfplay,
        total_training=total_training,
        disk_critical=disk_critical,
        disk_warning=disk_warning,
        memory_warning=memory_warning,
        stale_nodes=stale_nodes,
        unhealthy_nodes=unhealthy_nodes,
        errors=errors,
    )


def cleanup_disk_on_node(host: str, port: int, user: str, node_name: str) -> bool:
    """Attempt to clean up disk space on a node."""
    logger.info(f"Attempting disk cleanup on {node_name}")

    cleanup_commands = [
        # Clean old logs
        "find ~/ringrift/ai-service/logs -name '*.log' -mtime +1 -delete 2>/dev/null || true",
        # Clean old checkpoints (keep last 3)
        "cd ~/ringrift/ai-service/models && ls -t *.pth 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true",
        # Clean __pycache__
        "find ~/ringrift -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
        # Clean pip cache
        "pip cache purge 2>/dev/null || true",
        # Report remaining space
        "df -h ~ | tail -1",
    ]

    for cmd in cleanup_commands:
        success, output = run_ssh_command(host, port, user, cmd, timeout=60)
        if output.strip():
            logger.info(f"  {cmd[:50]}... -> {output.strip()[:100]}")

    return True


def restart_stale_p2p(host: str, port: int, user: str, node_name: str) -> bool:
    """Attempt to restart P2P on a stale node."""
    logger.warning(f"Node {node_name} appears stale, checking P2P status")

    # First check if P2P is running
    success, output = run_ssh_command(
        host, port, user,
        "pgrep -f p2p_orchestrator | head -1",
        timeout=10,
    )

    if not output.strip():
        logger.warning(f"P2P not running on {node_name}")
        # Don't auto-restart P2P - too risky
        return False

    return True


def take_corrective_actions(status: ClusterStatus) -> list[str]:
    """Take corrective actions based on cluster status."""
    actions_taken = []

    # Handle critical disk issues
    for disk_issue in status.disk_critical:
        node_name = disk_issue.split(":")[0]
        logger.warning(f"CRITICAL disk on {node_name}")

        # Find SSH details for this node
        # For now, just log - we'd need a node->SSH mapping
        actions_taken.append(f"Logged critical disk warning for {node_name}")

    # Handle quorum issues
    if not status.quorum_ok:
        logger.error("QUORUM FAILED - cluster coordination impaired!")
        actions_taken.append("Logged quorum failure alert")

    # Handle no leader
    if status.leader_id == "NONE" or not status.leader_id:
        logger.error("NO LEADER - job dispatch disabled!")
        actions_taken.append("Logged no-leader alert")

    return actions_taken


def log_status_summary(status: ClusterStatus) -> None:
    """Log a summary of cluster status."""
    logger.info("=" * 60)
    logger.info(f"CHECK #{status.check_number} at {status.timestamp}")
    logger.info("=" * 60)
    logger.info(f"Leader: {status.leader_id} (role: {status.leader_role})")
    logger.info(f"Quorum: {'OK' if status.quorum_ok else 'FAILED'} ({status.voters_alive}/5 voters)")

    # Handle alive_peers being either a list or an int
    if isinstance(status.alive_peers, list):
        alive_count = len(status.alive_peers)
        alive_display = f"{status.alive_peers[:10]}{'...' if len(status.alive_peers) > 10 else ''}"
    else:
        alive_count = status.alive_peers if isinstance(status.alive_peers, int) else 0
        alive_display = str(status.alive_peers)

    logger.info(f"Nodes: {status.total_nodes} total, {alive_count} alive")
    logger.info(f"Alive peers: {alive_display}")
    logger.info(f"Jobs: {status.total_selfplay} selfplay, {status.total_training} training")

    if status.disk_critical:
        logger.warning(f"DISK CRITICAL: {status.disk_critical}")
    if status.disk_warning:
        logger.info(f"Disk warnings: {status.disk_warning}")
    if status.memory_warning:
        logger.info(f"Memory warnings: {status.memory_warning}")
    if status.stale_nodes:
        logger.warning(f"Stale nodes: {status.stale_nodes}")
    if status.unhealthy_nodes:
        logger.warning(f"Unhealthy nodes: {status.unhealthy_nodes}")
    if status.errors:
        logger.error(f"Errors: {status.errors}")

    # Overall health assessment
    if status.quorum_ok and status.leader_id and status.leader_id != "NONE" and not status.disk_critical:
        logger.info("Cluster is HEALTHY")
    else:
        issues = []
        if not status.quorum_ok:
            issues.append("quorum failed")
        if not status.leader_id or status.leader_id == "NONE":
            issues.append("no leader")
        if status.disk_critical:
            issues.append("disk critical")
        logger.warning(f"Cluster has issues: {', '.join(issues)}")


def write_status_to_log(status: ClusterStatus, log_file: Path) -> None:
    """Append status to JSONL log file."""
    with open(log_file, "a") as f:
        f.write(json.dumps(status.to_dict()) + "\n")


def monitor_loop(
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    log_file: Path = DEFAULT_LOG_FILE,
) -> None:
    """Main monitoring loop with duration and JSONL logging.

    Args:
        duration_minutes: Total duration to monitor (default 60)
        interval_minutes: Check interval in minutes (default 10)
        log_file: Path to JSONL log file
    """
    interval_seconds = interval_minutes * 60
    end_time = time.time() + (duration_minutes * 60) if duration_minutes > 0 else float("inf")

    logger.info(f"Starting cluster health monitor")
    logger.info(f"  Duration: {duration_minutes} minutes (or infinite if 0)")
    logger.info(f"  Interval: {interval_minutes} minutes")
    logger.info(f"  Log file: {log_file}")

    cycle = 0
    while time.time() < end_time:
        cycle += 1
        logger.info(f"\n--- Monitoring cycle {cycle} ---")

        try:
            # Fetch status
            p2p_status = get_p2p_status()

            if p2p_status is None:
                logger.error("Could not reach any cluster entry point!")
                # Record error in log
                error_status = ClusterStatus(
                    timestamp=datetime.utcnow().isoformat(),
                    epoch=time.time(),
                    check_number=cycle,
                    leader_id="UNKNOWN",
                    leader_role="unknown",
                    quorum_ok=False,
                    voters_alive=0,
                    total_nodes=0,
                    errors=["Could not reach any cluster entry point"],
                )
                write_status_to_log(error_status, log_file)
                time.sleep(60)  # Short retry
                continue

            # Analyze health
            status = analyze_cluster_health(p2p_status, check_number=cycle)

            # Log summary to console
            log_status_summary(status)

            # Write to JSONL log
            write_status_to_log(status, log_file)

            # Take corrective actions if needed
            if status.disk_critical or not status.quorum_ok or not status.leader_id:
                actions = take_corrective_actions(status)
                if actions:
                    logger.info(f"Actions taken: {actions}")

        except Exception as e:
            logger.exception(f"Error in monitoring cycle: {e}")

        # Check if we should continue
        remaining = end_time - time.time()
        if remaining <= 0:
            break

        # Sleep until next cycle (but don't overshoot end time)
        sleep_time = min(interval_seconds, remaining)
        logger.info(f"Sleeping {sleep_time/60:.1f} minutes until next check...")
        time.sleep(sleep_time)

    logger.info(f"\nMonitoring complete. {cycle} checks recorded to {log_file}")


def analyze_observation_log(log_file: Path) -> None:
    """Analyze a JSONL observation log and print summary."""
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    entries = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        print("No entries found in log file")
        return

    print("=" * 70)
    print(f"CLUSTER HEALTH OBSERVATION ANALYSIS")
    print(f"Log file: {log_file}")
    print(f"Entries: {len(entries)}")
    print("=" * 70)

    # Time range
    first = entries[0]
    last = entries[-1]
    print(f"\nTime range: {first.get('timestamp', 'N/A')} to {last.get('timestamp', 'N/A')}")
    if first.get("epoch") and last.get("epoch"):
        duration_min = (last["epoch"] - first["epoch"]) / 60
        print(f"Duration: {duration_min:.1f} minutes")

    # Node count statistics
    node_counts = [e.get("total_nodes", 0) for e in entries]
    # Handle alive_peers being either a list or an int
    def get_alive_count(entry):
        peers = entry.get("alive_peers", 0)
        return len(peers) if isinstance(peers, list) else (peers if isinstance(peers, int) else 0)
    alive_counts = [get_alive_count(e) for e in entries]
    print(f"\nNode counts:")
    print(f"  Total range: {min(node_counts)} - {max(node_counts)}")
    print(f"  Alive range: {min(alive_counts)} - {max(alive_counts)}")
    print(f"  Alive avg: {sum(alive_counts)/len(alive_counts):.1f}")

    # Leader stability
    leaders = [e.get("leader_id", "NONE") for e in entries]
    unique_leaders = set(leaders)
    leader_changes = sum(1 for i in range(1, len(leaders)) if leaders[i] != leaders[i-1])
    print(f"\nLeader stability:")
    print(f"  Unique leaders: {unique_leaders}")
    print(f"  Leader changes: {leader_changes}")

    # Quorum health
    quorum_ok_count = sum(1 for e in entries if e.get("quorum_ok", False))
    print(f"\nQuorum health:")
    print(f"  OK: {quorum_ok_count}/{len(entries)} ({100*quorum_ok_count/len(entries):.1f}%)")

    # Issues summary
    from collections import Counter
    issues = Counter()
    for e in entries:
        if not e.get("quorum_ok"):
            issues["quorum_failed"] += 1
        if e.get("leader_id") in (None, "NONE", "UNKNOWN"):
            issues["no_leader"] += 1
        if e.get("disk_critical"):
            issues["disk_critical"] += 1
        if e.get("stale_nodes"):
            issues["stale_nodes"] += 1
        if e.get("unhealthy_nodes"):
            issues["unhealthy_nodes"] += 1
        if e.get("errors"):
            issues["errors"] += 1
        peers = e.get("alive_peers", 0)
        alive = len(peers) if isinstance(peers, list) else (peers if isinstance(peers, int) else 0)
        if alive < 20:
            issues["below_20_nodes"] += 1
        if alive < 15:
            issues["below_15_nodes"] += 1
        if alive < 10:
            issues["below_10_nodes"] += 1

    print(f"\nIssues detected:")
    for issue, count in sorted(issues.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count} occurrences ({100*count/len(entries):.1f}%)")

    # Job activity
    selfplay_counts = [e.get("total_selfplay", 0) for e in entries]
    training_counts = [e.get("total_training", 0) for e in entries]
    print(f"\nJob activity:")
    print(f"  Selfplay range: {min(selfplay_counts)} - {max(selfplay_counts)}")
    print(f"  Training range: {min(training_counts)} - {max(training_counts)}")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)
    target_nodes = 20
    below_target = sum(1 for c in alive_counts if c < target_nodes)
    print(f"  20+ nodes: {'PASS' if below_target == 0 else 'FAIL'} ({len(entries) - below_target}/{len(entries)} checks passed)")
    print(f"  Quorum stable: {'PASS' if quorum_ok_count == len(entries) else 'FAIL'}")
    print(f"  Leader stable: {'PASS' if leader_changes <= 2 else 'FAIL'} ({leader_changes} changes)")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster health monitor with structured logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 60-minute observation with 10-minute intervals
  python scripts/cluster_health_monitor.py

  # Run 30-minute observation with 5-minute intervals
  python scripts/cluster_health_monitor.py --duration 30 --interval 5

  # Run indefinitely (until Ctrl+C)
  python scripts/cluster_health_monitor.py --duration 0

  # Analyze previous observation
  python scripts/cluster_health_monitor.py --analyze

  # Analyze specific log file
  python scripts/cluster_health_monitor.py --analyze --log-file my_log.jsonl
""",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_MINUTES,
        help=f"Monitoring duration in minutes (0=infinite, default={DEFAULT_DURATION_MINUTES})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help=f"Check interval in minutes (default={DEFAULT_INTERVAL_MINUTES})",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Path to JSONL log file (default={DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing log file instead of monitoring",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.analyze:
        analyze_observation_log(args.log_file)
    else:
        try:
            monitor_loop(
                duration_minutes=args.duration,
                interval_minutes=args.interval,
                log_file=args.log_file,
            )
        except KeyboardInterrupt:
            logger.info("\nMonitor stopped by user")
            sys.exit(0)
