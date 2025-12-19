#!/usr/bin/env python
"""Long-running cluster health monitor.

.. deprecated::
    This module is deprecated in favor of unified_cluster_monitor.py.
    Use instead:

        python scripts/unified_cluster_monitor.py --continuous --interval 120
        python scripts/unified_cluster_monitor.py --deep-checks  # For SSH/GPU checks

    Or via CLI:

        python scripts/cli.py cluster --continuous

    This script will be removed in a future release.

Monitors cluster nodes every 2 minutes for 10 hours, checking:
- Node connectivity (SSH)
- GPU utilization
- Training loop status
- Self-play progress
- Disk space
- Memory usage

Usage:
    python scripts/monitor_cluster_health.py --duration 600  # 10 hours in minutes
    python scripts/monitor_cluster_health.py --interval 120  # Check every 2 min
"""
import warnings
warnings.warn(
    "monitor_cluster_health.py is deprecated. Use unified_cluster_monitor.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from scripts.lib.ssh import run_ssh_command

# Ensure app imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import yaml
except ImportError:
    yaml = None

# Configuration
HOSTS_CONFIG = Path(ROOT) / "config" / "distributed_hosts.yaml"
LOG_DIR = Path(ROOT) / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Cluster nodes to monitor
PRIORITY_NODES = [
    "lambda-gh200-a", "lambda-gh200-c", "lambda-gh200-d",
    "lambda-gh200-e", "lambda-gh200-f",
    "lambda-h100", "lambda-2xh100", "lambda-a10"
]


@dataclass
class NodeStatus:
    """Status of a single cluster node."""
    name: str
    reachable: bool = False
    gpu_util: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    cpu_util: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_used_pct: float = 0.0
    selfplay_running: bool = False
    training_running: bool = False
    last_check: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class ClusterHealth:
    """Overall cluster health status."""
    nodes: Dict[str, NodeStatus] = field(default_factory=dict)
    total_nodes: int = 0
    online_nodes: int = 0
    avg_gpu_util: float = 0.0
    total_games_generated: int = 0
    last_check: Optional[datetime] = None
    alerts: List[str] = field(default_factory=list)


def load_hosts_config() -> Dict:
    """Load cluster hosts from config file."""
    if not yaml or not HOSTS_CONFIG.exists():
        return {}
    with open(HOSTS_CONFIG) as f:
        return yaml.safe_load(f).get('hosts', {})




def check_node_status(node_name: str, config: Dict) -> NodeStatus:
    """Check status of a single cluster node."""
    status = NodeStatus(name=node_name)

    node_config = config.get(node_name, {})
    ssh_host = node_config.get('ssh_host') or node_config.get('tailscale_ip')
    ssh_user = node_config.get('ssh_user', 'ubuntu')
    ringrift_path = node_config.get('ringrift_path', '~/ringrift/ai-service')

    if not ssh_host:
        status.error = "No SSH host configured"
        return status

    # Check connectivity
    success, output = run_ssh_command(ssh_host, "echo 'alive'", user=ssh_user)
    if not success:
        status.error = f"Unreachable: {output}"
        return status

    status.reachable = True
    status.last_check = datetime.now()

    # Check GPU status
    success, output = run_ssh_command(
        ssh_host,
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'N/A'",
        user=ssh_user,
    )
    if success and output and output != 'N/A':
        try:
            parts = output.split(',')
            if len(parts) >= 3:
                status.gpu_util = float(parts[0].strip())
                status.gpu_memory_used = float(parts[1].strip()) / 1024  # Convert to GB
                status.gpu_memory_total = float(parts[2].strip()) / 1024
        except (ValueError, IndexError):
            pass

    # Check memory
    success, output = run_ssh_command(
        ssh_host,
        "free -g | awk '/^Mem:/ {print $2,$3}'",
        user=ssh_user,
    )
    if success and output:
        try:
            parts = output.split()
            if len(parts) >= 2:
                status.memory_total_gb = float(parts[0])
                status.memory_used_gb = float(parts[1])
        except (ValueError, IndexError):
            pass

    # Check disk usage
    success, output = run_ssh_command(
        ssh_host,
        f"df -h {ringrift_path} 2>/dev/null | tail -1 | awk '{{print $5}}' | tr -d '%'",
        user=ssh_user,
    )
    if success and output:
        try:
            status.disk_used_pct = float(output)
        except ValueError:
            pass

    # Check for running selfplay
    success, output = run_ssh_command(
        ssh_host,
        "pgrep -f 'generate_gpu_training_data|benchmark_gpu_selfplay' | wc -l",
        user=ssh_user,
    )
    if success:
        try:
            status.selfplay_running = int(output.strip()) > 0
        except ValueError:
            pass

    # Check for running training
    success, output = run_ssh_command(
        ssh_host,
        "pgrep -f 'train_nnue|unified_ai_loop' | wc -l",
        user=ssh_user,
    )
    if success:
        try:
            status.training_running = int(output.strip()) > 0
        except ValueError:
            pass

    return status


def check_cluster_health(nodes: List[str]) -> ClusterHealth:
    """Check health of all cluster nodes."""
    health = ClusterHealth()
    health.last_check = datetime.now()

    config = load_hosts_config()

    total_gpu_util = 0.0
    online_count = 0

    for node_name in nodes:
        status = check_node_status(node_name, config)
        health.nodes[node_name] = status
        health.total_nodes += 1

        if status.reachable:
            online_count += 1
            total_gpu_util += status.gpu_util

            # Check for alerts
            if status.gpu_util < 10 and status.selfplay_running:
                health.alerts.append(f"{node_name}: Low GPU util ({status.gpu_util:.0f}%) with selfplay running")
            if status.disk_used_pct > 90:
                health.alerts.append(f"{node_name}: High disk usage ({status.disk_used_pct:.0f}%)")
            if status.memory_total_gb > 0 and status.memory_used_gb / status.memory_total_gb > 0.95:
                health.alerts.append(f"{node_name}: High memory usage")
        else:
            health.alerts.append(f"{node_name}: OFFLINE - {status.error}")

    health.online_nodes = online_count
    health.avg_gpu_util = total_gpu_util / online_count if online_count > 0 else 0

    return health


def format_status_report(health: ClusterHealth) -> str:
    """Format a status report for logging/display."""
    lines = [
        "",
        "=" * 70,
        f"CLUSTER HEALTH CHECK - {health.last_check.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        f"Nodes Online: {health.online_nodes}/{health.total_nodes}",
        f"Average GPU Utilization: {health.avg_gpu_util:.1f}%",
        "",
        f"{'Node':<20} {'Status':<10} {'GPU%':<8} {'GPU Mem':<12} {'Selfplay':<10} {'Training':<10}",
        "-" * 70,
    ]

    for name, status in sorted(health.nodes.items()):
        if status.reachable:
            gpu_mem = f"{status.gpu_memory_used:.1f}/{status.gpu_memory_total:.0f}GB"
            selfplay = "Running" if status.selfplay_running else "-"
            training = "Running" if status.training_running else "-"
            lines.append(
                f"{name:<20} {'Online':<10} {status.gpu_util:<8.1f} {gpu_mem:<12} {selfplay:<10} {training:<10}"
            )
        else:
            lines.append(f"{name:<20} {'OFFLINE':<10} {'-':<8} {'-':<12} {'-':<10} {'-':<10}")

    if health.alerts:
        lines.extend([
            "",
            "ALERTS:",
            "-" * 70,
        ])
        for alert in health.alerts:
            lines.append(f"  ! {alert}")

    lines.append("=" * 70)
    return "\n".join(lines)


def run_monitor(duration_minutes: int, interval_seconds: int, nodes: List[str]):
    """Run the monitoring loop."""
    log_file = LOG_DIR / f"cluster_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"\nCluster Health Monitor Started")
    print(f"Duration: {duration_minutes} minutes ({duration_minutes/60:.1f} hours)")
    print(f"Interval: {interval_seconds} seconds")
    print(f"Monitoring {len(nodes)} nodes")
    print(f"Log file: {log_file}")
    print("-" * 50)

    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    check_count = 0

    with open(log_file, 'w') as log:
        log.write(f"Cluster Monitor Started: {datetime.now()}\n")
        log.write(f"Duration: {duration_minutes} minutes\n")
        log.write(f"Interval: {interval_seconds} seconds\n")
        log.write(f"Nodes: {', '.join(nodes)}\n\n")

        while datetime.now() < end_time:
            check_count += 1

            print(f"\n[Check #{check_count}] {datetime.now().strftime('%H:%M:%S')} - Checking cluster health...")

            health = check_cluster_health(nodes)
            report = format_status_report(health)

            print(report)
            log.write(report + "\n\n")
            log.flush()

            # Summary line
            status_line = (
                f"[{datetime.now().strftime('%H:%M')}] "
                f"Online: {health.online_nodes}/{health.total_nodes}, "
                f"GPU: {health.avg_gpu_util:.0f}%, "
                f"Alerts: {len(health.alerts)}"
            )
            print(f"\nStatus: {status_line}")

            remaining = (end_time - datetime.now()).total_seconds()
            if remaining <= 0:
                break

            print(f"Next check in {interval_seconds}s (remaining: {remaining/60:.0f} min)")
            time.sleep(min(interval_seconds, remaining))

    print(f"\n{'='*50}")
    print("Monitoring Complete!")
    print(f"Log saved to: {log_file}")
    print(f"Total checks: {check_count}")


def main():
    parser = argparse.ArgumentParser(description="Cluster Health Monitor")
    parser.add_argument("--duration", type=int, default=600,
                        help="Monitoring duration in minutes (default: 600 = 10 hours)")
    parser.add_argument("--interval", type=int, default=120,
                        help="Check interval in seconds (default: 120 = 2 min)")
    parser.add_argument("--nodes", type=str, default=None,
                        help="Comma-separated node list (default: priority nodes)")
    args = parser.parse_args()

    nodes = args.nodes.split(',') if args.nodes else PRIORITY_NODES

    try:
        run_monitor(args.duration, args.interval, nodes)
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
