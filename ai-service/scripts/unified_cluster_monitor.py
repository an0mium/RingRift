#!/usr/bin/env python3
"""Unified Cluster Monitor - Single entry point for all cluster monitoring.

Consolidates cluster_monitor.py, cluster_health_monitor.py, monitor_cluster_health.py,
and cluster_health_check.py into a unified monitoring system.

Features:
- HTTP health endpoint checks (primary) with Tailscale fallback
- SSH-based deep checks for GPU/process status
- Leader API integration for training job status
- Unified alerting (console, webhook, metrics)
- Both one-shot and continuous modes
- Centralized config from distributed_hosts.yaml

Usage:
    # Quick health check (one-shot)
    python scripts/unified_cluster_monitor.py --quick

    # Full health check with SSH checks
    python scripts/unified_cluster_monitor.py --deep

    # Continuous monitoring (default 60s interval)
    python scripts/unified_cluster_monitor.py --continuous

    # With webhook alerts
    python scripts/unified_cluster_monitor.py --continuous --webhook https://discord.com/api/...

    # JSON output
    python scripts/unified_cluster_monitor.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Ensure app imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:
    yaml = None

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from app.utils.paths import CONFIG_DIR
from app.monitoring.thresholds import get_threshold

# Configuration paths
CONFIG_PATH = CONFIG_DIR / "distributed_hosts.yaml"
LOG_DIR = ROOT / "logs"
HEALTH_PORT = 8770


@dataclass
class NodeHealth:
    """Health status for a single node."""
    name: str
    status: str = "unknown"  # healthy, unhealthy, unreachable, error
    via_tailscale: bool = False

    # HTTP health data
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    selfplay_active: bool = False
    games_played: int = 0
    version: str = "unknown"

    # SSH deep check data (optional)
    gpu_util: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    training_running: bool = False

    # Error info
    error: Optional[str] = None
    last_check: Optional[datetime] = None


@dataclass
class LeaderHealth:
    """Health status from cluster leader."""
    is_leader: bool = False
    node_id: str = "unknown"
    selfplay_jobs: int = 0
    selfplay_rate: float = 0.0
    training_nnue_running: int = 0
    training_cmaes_running: int = 0
    training_failed: int = 0
    auto_nnue_enabled: bool = False
    auto_cmaes_enabled: bool = False
    error: Optional[str] = None


@dataclass
class ClusterHealth:
    """Overall cluster health status."""
    nodes: Dict[str, NodeHealth] = field(default_factory=dict)
    leader: Optional[LeaderHealth] = None

    # Summary metrics
    total_nodes: int = 0
    healthy_nodes: int = 0
    avg_gpu_util: float = 0.0
    total_games: int = 0

    # Alerts
    alerts: List[str] = field(default_factory=list)
    critical_alerts: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)


class ClusterConfig:
    """Cluster configuration from distributed_hosts.yaml."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.leader_url: Optional[str] = None
        self._load()

    def _load(self):
        """Load configuration from YAML file."""
        if not yaml or not CONFIG_PATH.exists():
            print(f"Warning: Config not found: {CONFIG_PATH}", file=sys.stderr)
            return

        try:
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)

            for name, cfg in config.get('hosts', {}).items():
                # Skip disabled/stopped hosts
                if cfg.get('status') in ('stopped', 'disabled', 'setup'):
                    continue

                node = {
                    'name': name,
                    'ssh_host': cfg.get('ssh_host'),
                    'ssh_user': cfg.get('ssh_user', 'ubuntu'),
                    'ssh_port': cfg.get('ssh_port', 22),
                    'tailscale_ip': cfg.get('tailscale_ip'),
                    'p2p_port': cfg.get('p2p_port', HEALTH_PORT),
                    'ringrift_path': cfg.get('ringrift_path', '~/ringrift/ai-service'),
                    'is_leader': cfg.get('role') == 'leader',
                }

                # Build URLs
                if node['ssh_host']:
                    node['primary_url'] = f"http://{node['ssh_host']}:{node['p2p_port']}/health"
                if node['tailscale_ip']:
                    node['tailscale_url'] = f"http://{node['tailscale_ip']}:{node['p2p_port']}/health"

                self.nodes[name] = node

                # Track leader
                if node['is_leader'] and node.get('primary_url'):
                    base_url = f"http://{node['ssh_host']}:{node['p2p_port']}"
                    self.leader_url = base_url

        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)

    def get_node_names(self) -> List[str]:
        """Get list of active node names."""
        return list(self.nodes.keys())


class UnifiedClusterMonitor:
    """Unified cluster monitoring system."""

    def __init__(
        self,
        config: ClusterConfig,
        webhook_url: Optional[str] = None,
        check_interval: int = 60,
        deep_checks: bool = False,
    ):
        self.config = config
        self.webhook_url = webhook_url
        self.check_interval = check_interval
        self.deep_checks = deep_checks

        # Alert cooldown tracking
        self._last_alerts: Dict[str, float] = {}
        self._alert_cooldown = 300  # 5 minutes

        # Thresholds
        self.disk_warning = get_threshold("disk", "warning", default=60)
        self.disk_critical = get_threshold("disk", "critical", default=70)
        self.memory_warning = get_threshold("memory", "warning", default=70)
        self.memory_critical = get_threshold("memory", "critical", default=80)

    # --- HTTP Health Checks ---

    def _http_get_json(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """Fetch JSON from URL."""
        try:
            req = Request(url, headers={"User-Agent": "UnifiedClusterMonitor/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            return {"error": f"HTTP {e.code}"}
        except URLError as e:
            return {"error": str(e.reason)}
        except Exception as e:
            return {"error": str(e)}

    def check_node_http(self, node_name: str) -> NodeHealth:
        """Check node health via HTTP endpoints."""
        health = NodeHealth(name=node_name, last_check=datetime.now())
        node_cfg = self.config.nodes.get(node_name, {})

        # Try primary URL first
        primary_url = node_cfg.get('primary_url')
        if primary_url:
            data = self._http_get_json(primary_url)
            if "error" not in data:
                self._populate_health_from_http(health, data)
                return health

        # Fallback to Tailscale
        tailscale_url = node_cfg.get('tailscale_url')
        if tailscale_url:
            data = self._http_get_json(tailscale_url)
            if "error" not in data:
                self._populate_health_from_http(health, data)
                health.via_tailscale = True
                return health

        # Both failed
        health.status = "unreachable"
        health.error = data.get("error", "No URLs configured")
        return health

    def _populate_health_from_http(self, health: NodeHealth, data: Dict[str, Any]):
        """Populate NodeHealth from HTTP response data."""
        health.status = "healthy" if data.get("healthy", True) else "unhealthy"
        health.cpu_percent = data.get("cpu_percent", 0)
        health.memory_percent = data.get("memory_percent", 0)
        health.disk_percent = data.get("disk_percent", 0)
        health.selfplay_active = data.get("selfplay_active", False)
        health.games_played = data.get("games_played", 0)
        health.version = data.get("version", "unknown")

    # --- SSH Deep Checks ---

    def _run_ssh(self, host: str, user: str, port: int, cmd: str, timeout: int = 15) -> tuple:
        """Run command via SSH."""
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                 "-o", "BatchMode=yes", "-p", str(port), f"{user}@{host}", cmd],
                capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "SSH timeout"
        except Exception as e:
            return False, str(e)

    def check_node_deep(self, node_name: str, health: NodeHealth) -> NodeHealth:
        """Perform deep checks via SSH (GPU, processes)."""
        node_cfg = self.config.nodes.get(node_name, {})
        ssh_host = node_cfg.get('ssh_host') or node_cfg.get('tailscale_ip')
        ssh_user = node_cfg.get('ssh_user', 'ubuntu')
        ssh_port = node_cfg.get('ssh_port', 22)

        if not ssh_host:
            return health

        # GPU check
        ok, output = self._run_ssh(
            ssh_host, ssh_user, ssh_port,
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null || echo 'N/A'"
        )
        if ok and output and output != 'N/A':
            try:
                parts = output.split(',')
                if len(parts) >= 3:
                    health.gpu_util = float(parts[0].strip())
                    health.gpu_memory_used_gb = float(parts[1].strip()) / 1024
                    health.gpu_memory_total_gb = float(parts[2].strip()) / 1024
            except (ValueError, IndexError):
                pass

        # Training process check
        ok, output = self._run_ssh(
            ssh_host, ssh_user, ssh_port,
            "pgrep -f 'train_nnue|unified_ai_loop' | wc -l"
        )
        if ok:
            try:
                health.training_running = int(output.strip()) > 0
            except ValueError:
                pass

        return health

    # --- Leader API ---

    def check_leader(self) -> Optional[LeaderHealth]:
        """Check leader for training/selfplay status."""
        if not self.config.leader_url:
            return None

        leader = LeaderHealth()

        # Get status
        data = self._http_get_json(f"{self.config.leader_url}/status")
        if "error" not in data:
            leader.is_leader = data.get("role") == "leader"
            leader.node_id = data.get("node_id", "unknown")

        # Get health
        data = self._http_get_json(f"{self.config.leader_url}/health")
        if "error" not in data:
            leader.selfplay_jobs = data.get("selfplay_jobs", 0)
            cluster_util = data.get("cluster_utilization", {})
            leader.selfplay_rate = cluster_util.get("selfplay_rate", 0)
        else:
            leader.error = data.get("error")

        # Get training status
        data = self._http_get_json(f"{self.config.leader_url}/training/status")
        if "error" not in data:
            jobs = data.get("jobs", [])
            leader.training_nnue_running = len([j for j in jobs if j.get("status") == "running" and j.get("job_type") == "nnue"])
            leader.training_cmaes_running = len([j for j in jobs if j.get("status") == "running" and j.get("job_type") == "cmaes"])
            leader.training_failed = len([j for j in jobs if j.get("status") == "failed"])
            thresholds = data.get("thresholds", {})
            leader.auto_nnue_enabled = thresholds.get("auto_nnue_enabled", False)
            leader.auto_cmaes_enabled = thresholds.get("auto_cmaes_enabled", False)

        return leader

    # --- Alert Generation ---

    def _should_alert(self, key: str) -> bool:
        """Check if alert should fire (respects cooldown)."""
        last = self._last_alerts.get(key, 0)
        if time.time() - last > self._alert_cooldown:
            self._last_alerts[key] = time.time()
            return True
        return False

    def generate_alerts(self, cluster: ClusterHealth) -> None:
        """Generate alerts from cluster health status."""
        # Node alerts
        for name, health in cluster.nodes.items():
            if health.status == "unreachable":
                if self._should_alert(f"unreachable_{name}"):
                    cluster.critical_alerts.append(f"{name}: UNREACHABLE - {health.error}")

            elif health.status == "unhealthy":
                if self._should_alert(f"unhealthy_{name}"):
                    cluster.alerts.append(f"{name}: Reports unhealthy")

            elif health.status == "healthy":
                # Disk alerts
                if health.disk_percent >= self.disk_critical:
                    if self._should_alert(f"disk_crit_{name}"):
                        cluster.critical_alerts.append(f"{name}: CRITICAL disk {health.disk_percent:.1f}%")
                elif health.disk_percent >= self.disk_warning:
                    if self._should_alert(f"disk_warn_{name}"):
                        cluster.alerts.append(f"{name}: High disk {health.disk_percent:.1f}%")

                # Memory alerts
                if health.memory_percent >= self.memory_critical:
                    if self._should_alert(f"mem_crit_{name}"):
                        cluster.critical_alerts.append(f"{name}: CRITICAL memory {health.memory_percent:.1f}%")
                elif health.memory_percent >= self.memory_warning:
                    if self._should_alert(f"mem_warn_{name}"):
                        cluster.alerts.append(f"{name}: High memory {health.memory_percent:.1f}%")

                # GPU idle alert (if deep checks enabled)
                if self.deep_checks and health.gpu_util < 10 and health.selfplay_active:
                    if self._should_alert(f"gpu_idle_{name}"):
                        cluster.alerts.append(f"{name}: Low GPU ({health.gpu_util:.0f}%) with selfplay active")

        # Leader alerts
        if cluster.leader:
            if cluster.leader.error:
                if self._should_alert("leader_error"):
                    cluster.critical_alerts.append(f"Leader unreachable: {cluster.leader.error}")

            if cluster.leader.training_failed > 0:
                if self._should_alert("training_failed"):
                    cluster.alerts.append(f"Training: {cluster.leader.training_failed} failed jobs")

        # Cluster-wide alerts
        if cluster.healthy_nodes == 0:
            if self._should_alert("cluster_down"):
                cluster.critical_alerts.append("CLUSTER DOWN: No healthy nodes!")

    def send_webhook_alert(self, message: str, level: str = "warning"):
        """Send alert to webhook."""
        if not self.webhook_url:
            return

        color = {"critical": 0xFF0000, "warning": 0xFFA500, "info": 0x00FF00}.get(level, 0x808080)
        payload = {
            "embeds": [{
                "title": f"RingRift Cluster ({level.upper()})",
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
            }]
        }

        try:
            import urllib.request
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"Webhook error: {e}", file=sys.stderr)

    # --- Main Check Logic ---

    def check_cluster(self) -> ClusterHealth:
        """Perform a complete cluster health check."""
        cluster = ClusterHealth(timestamp=datetime.now())

        # Check all nodes
        for node_name in self.config.get_node_names():
            health = self.check_node_http(node_name)

            if self.deep_checks and health.status in ("healthy", "unhealthy"):
                health = self.check_node_deep(node_name, health)

            cluster.nodes[node_name] = health

        # Check leader
        cluster.leader = self.check_leader()

        # Calculate summary metrics
        cluster.total_nodes = len(cluster.nodes)
        healthy_nodes = [h for h in cluster.nodes.values() if h.status == "healthy"]
        cluster.healthy_nodes = len(healthy_nodes)

        if healthy_nodes:
            cluster.avg_gpu_util = sum(h.gpu_util for h in healthy_nodes) / len(healthy_nodes)
            cluster.total_games = sum(h.games_played for h in healthy_nodes)

        # Generate alerts
        self.generate_alerts(cluster)

        # Send webhook for critical alerts
        if cluster.critical_alerts and self.webhook_url:
            self.send_webhook_alert("\n".join(cluster.critical_alerts), "critical")

        return cluster

    # --- Output Formatting ---

    def format_text(self, cluster: ClusterHealth) -> str:
        """Format cluster health as text."""
        lines = [
            "",
            "=" * 70,
            f"  CLUSTER HEALTH - {cluster.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
        ]

        # Summary
        lines.append(f"  Nodes: {cluster.healthy_nodes}/{cluster.total_nodes} healthy")
        if cluster.leader:
            lines.append(f"  Leader: {cluster.leader.node_id} | Selfplay: {cluster.leader.selfplay_jobs} jobs @ {cluster.leader.selfplay_rate:.0f}/hr")
            lines.append(f"  Training: {cluster.leader.training_nnue_running} NNUE + {cluster.leader.training_cmaes_running} CMA-ES | Auto: NNUE={'ON' if cluster.leader.auto_nnue_enabled else 'OFF'} CMA-ES={'ON' if cluster.leader.auto_cmaes_enabled else 'OFF'}")

        lines.append("")

        # Node table header
        if self.deep_checks:
            lines.append(f"  {'Node':<20} {'Status':<10} {'GPU%':<6} {'Selfplay':<10} {'Disk%':<8} {'Mem%':<8}")
        else:
            lines.append(f"  {'Node':<20} {'Status':<10} {'Selfplay':<10} {'Disk%':<8} {'Mem%':<8} {'Games':<10}")
        lines.append("  " + "-" * 66)

        # Node rows
        for name, h in sorted(cluster.nodes.items()):
            if h.status == "healthy":
                selfplay = "Active" if h.selfplay_active else "-"
                via = " (TS)" if h.via_tailscale else ""
                if self.deep_checks:
                    lines.append(f"  {name:<20} {'OK' + via:<10} {h.gpu_util:<6.0f} {selfplay:<10} {h.disk_percent:<8.1f} {h.memory_percent:<8.1f}")
                else:
                    lines.append(f"  {name:<20} {'OK' + via:<10} {selfplay:<10} {h.disk_percent:<8.1f} {h.memory_percent:<8.1f} {h.games_played:<10,}")
            else:
                error = (h.error or "")[:20]
                lines.append(f"  {name:<20} {h.status.upper():<10} {error}")

        # Alerts
        if cluster.critical_alerts or cluster.alerts:
            lines.extend(["", "  ALERTS:"])
            for alert in cluster.critical_alerts:
                lines.append(f"    [CRITICAL] {alert}")
            for alert in cluster.alerts:
                lines.append(f"    [WARN] {alert}")
        else:
            lines.append("")
            lines.append("  OK: No alerts - cluster healthy")

        lines.append("=" * 70)
        return "\n".join(lines)

    def format_json(self, cluster: ClusterHealth) -> str:
        """Format cluster health as JSON."""
        return json.dumps({
            "timestamp": cluster.timestamp.isoformat(),
            "summary": {
                "total_nodes": cluster.total_nodes,
                "healthy_nodes": cluster.healthy_nodes,
                "avg_gpu_util": cluster.avg_gpu_util,
                "total_games": cluster.total_games,
            },
            "leader": {
                "node_id": cluster.leader.node_id if cluster.leader else None,
                "selfplay_jobs": cluster.leader.selfplay_jobs if cluster.leader else 0,
                "selfplay_rate": cluster.leader.selfplay_rate if cluster.leader else 0,
                "training_running": (cluster.leader.training_nnue_running + cluster.leader.training_cmaes_running) if cluster.leader else 0,
            } if cluster.leader else None,
            "nodes": {
                name: {
                    "status": h.status,
                    "via_tailscale": h.via_tailscale,
                    "cpu_percent": h.cpu_percent,
                    "memory_percent": h.memory_percent,
                    "disk_percent": h.disk_percent,
                    "selfplay_active": h.selfplay_active,
                    "games_played": h.games_played,
                    "gpu_util": h.gpu_util,
                    "error": h.error,
                }
                for name, h in cluster.nodes.items()
            },
            "alerts": cluster.alerts,
            "critical_alerts": cluster.critical_alerts,
        }, indent=2)

    # --- Run Modes ---

    def run_once(self, output_json: bool = False) -> ClusterHealth:
        """Run a single health check."""
        cluster = self.check_cluster()

        if output_json:
            print(self.format_json(cluster))
        else:
            print(self.format_text(cluster))

        return cluster

    def run_continuous(self, output_json: bool = False):
        """Run continuous monitoring."""
        print(f"Starting continuous monitoring (interval: {self.check_interval}s)")
        print(f"Monitoring {len(self.config.nodes)} nodes")
        print(f"Deep checks: {'enabled' if self.deep_checks else 'disabled'}")
        print(f"Webhook: {'configured' if self.webhook_url else 'not configured'}")
        print("-" * 50)

        check_count = 0
        while True:
            try:
                check_count += 1
                cluster = self.check_cluster()

                if output_json:
                    print(self.format_json(cluster))
                else:
                    print(self.format_text(cluster))
                    print(f"\n[Check #{check_count}] Next check in {self.check_interval}s...")

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                print("\nStopping monitor...")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Unified Cluster Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick health check
    python scripts/unified_cluster_monitor.py --quick

    # Full check with SSH/GPU data
    python scripts/unified_cluster_monitor.py --deep

    # Continuous monitoring with 30s interval
    python scripts/unified_cluster_monitor.py --continuous --interval 30

    # JSON output for scripting
    python scripts/unified_cluster_monitor.py --json
        """
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", "-q", action="store_true", help="Quick HTTP-only check (default)")
    mode.add_argument("--deep", "-d", action="store_true", help="Deep check including SSH/GPU status")
    mode.add_argument("--continuous", "-c", action="store_true", help="Continuous monitoring mode")

    # Options
    parser.add_argument("--interval", "-i", type=int, default=60, help="Check interval in seconds (default: 60)")
    parser.add_argument("--webhook", "-w", help="Webhook URL for alerts (Discord/Slack)")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Load config
    config = ClusterConfig()
    if not config.nodes:
        print("Error: No cluster nodes found in config", file=sys.stderr)
        sys.exit(1)

    # Create monitor
    monitor = UnifiedClusterMonitor(
        config=config,
        webhook_url=args.webhook,
        check_interval=args.interval,
        deep_checks=args.deep,
    )

    # Run
    if args.continuous:
        monitor.run_continuous(output_json=args.json)
    else:
        cluster = monitor.run_once(output_json=args.json)

        # Exit code based on health
        if cluster.critical_alerts:
            sys.exit(2)
        elif cluster.alerts:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
