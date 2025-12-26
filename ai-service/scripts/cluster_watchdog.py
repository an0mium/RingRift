#!/usr/bin/env python3
"""
Cluster Watchdog - Monitors and auto-restarts critical processes for sustained operation.

Usage:
    python scripts/cluster_watchdog.py              # Run watchdog (checks every 5 minutes)
    python scripts/cluster_watchdog.py --status     # Show current cluster status
    python scripts/cluster_watchdog.py --once       # Run once and exit
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import aiohttp
except ImportError:
    aiohttp = None


class ClusterWatchdog:
    """Monitors cluster health and auto-recovers critical processes."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.ai_service = project_root / "ai-service"
        self.logs_dir = self.ai_service / "logs"
        self.venv_python = self.ai_service / ".venv" / "bin" / "python"

        # Check intervals
        self.check_interval = 300  # 5 minutes
        self.p2p_timeout = 30

        # Minimum thresholds for healthy cluster
        self.min_alive_peers = 5
        self.min_selfplay_jobs = 10

    def log(self, msg: str, level: str = "INFO"):
        """Log with timestamp."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{level}] {msg}")

    def run_command(self, cmd: list, timeout: int = 30) -> tuple[bool, str]:
        """Run command and return (success, output)."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def get_process_status(self, pattern: str) -> list[dict]:
        """Get running processes matching pattern."""
        # Use ps instead of pgrep for more reliable matching
        ok, output = self.run_command(
            ["ps", "aux"], timeout=5
        )
        if not ok:
            return []

        processes = []
        for line in output.strip().split("\n"):
            if pattern in line and "python" in line.lower() and "grep" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        processes.append({
                            "pid": int(parts[1]),
                            "cmd": " ".join(parts[10:])
                        })
                    except (ValueError, IndexError):
                        pass
        return processes

    def get_p2p_status(self) -> Optional[dict]:
        """Get P2P cluster status."""
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", str(self.p2p_timeout),
                 "http://localhost:8770/status"],
                capture_output=True, text=True, timeout=self.p2p_timeout + 5
            )
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return None

    def count_selfplay_via_ssh(self, host: str, port: int = 22, user: str = "ubuntu") -> int:
        """Count selfplay processes on remote host via SSH."""
        try:
            cmd = [
                "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                "-p", str(port), f"{user}@{host}",
                "pgrep -c -f selfplay.py 2>/dev/null || echo 0"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, ValueError):
            pass
        return 0

    def start_p2p(self) -> bool:
        """Start P2P orchestrator."""
        self.log("Starting P2P orchestrator...")

        # Get Tailscale IP for advertising
        ok, output = self.run_command(["tailscale", "ip", "-4"], timeout=5)
        advertise_ip = output.strip() if ok else "100.111.184.80"

        bootstrap_peers = "89.169.112.47:8770,89.169.111.139:8770,89.169.108.182:8770"

        env = os.environ.copy()
        env["RINGRIFT_ADVERTISE_HOST"] = advertise_ip

        try:
            subprocess.Popen(
                [str(self.venv_python), "scripts/p2p_orchestrator.py",
                 "--node-id", "mac-studio", "--peers", bootstrap_peers],
                cwd=self.ai_service,
                env=env,
                stdout=open(self.logs_dir / "p2p.log", "a"),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
            time.sleep(5)
            return True
        except Exception as e:
            self.log(f"Failed to start P2P: {e}", "ERROR")
            return False

    def start_master_loop(self) -> bool:
        """Start master loop."""
        self.log("Starting master_loop...")

        try:
            subprocess.Popen(
                [str(self.venv_python), "scripts/master_loop.py", "--skip-daemons"],
                cwd=self.ai_service,
                stdout=open(self.logs_dir / "master_loop.log", "a"),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
            time.sleep(5)
            return True
        except Exception as e:
            self.log(f"Failed to start master_loop: {e}", "ERROR")
            return False

    def check_and_recover(self) -> dict:
        """Check cluster health and recover if needed."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "p2p_running": False,
            "master_loop_running": False,
            "p2p_alive_peers": 0,
            "p2p_leader": None,
            "selfplay_jobs": 0,
            "actions_taken": []
        }

        # Check P2P process
        p2p_procs = self.get_process_status("p2p_orchestrator")
        status["p2p_running"] = len(p2p_procs) > 0

        # Check master_loop process
        ml_procs = self.get_process_status("master_loop")
        status["master_loop_running"] = len(ml_procs) > 0

        # Check P2P cluster status
        p2p_status = self.get_p2p_status()
        if p2p_status:
            status["p2p_alive_peers"] = p2p_status.get("alive_peers", 0)
            status["p2p_leader"] = p2p_status.get("leader_id")

            # Count selfplay from P2P
            peers = p2p_status.get("peers", {})
            now = time.time()
            for _, info in peers.items():
                last_hb = info.get("last_heartbeat", 0)
                if now - last_hb < 120:
                    status["selfplay_jobs"] += info.get("selfplay_jobs", 0)

        # Recovery actions
        if not status["p2p_running"]:
            self.log("P2P not running - starting...", "WARN")
            if self.start_p2p():
                status["actions_taken"].append("started_p2p")

        if not status["master_loop_running"]:
            self.log("Master loop not running - starting...", "WARN")
            if self.start_master_loop():
                status["actions_taken"].append("started_master_loop")

        if status["p2p_running"] and status["p2p_alive_peers"] < self.min_alive_peers:
            self.log(f"Low peer count ({status['p2p_alive_peers']}), may need P2P restart", "WARN")

        return status

    def print_status(self, status: dict):
        """Print status in readable format."""
        print("\n" + "=" * 60)
        print(f"Cluster Status - {status['timestamp']}")
        print("=" * 60)

        p2p_icon = "✓" if status["p2p_running"] else "✗"
        ml_icon = "✓" if status["master_loop_running"] else "✗"

        print(f"  P2P Orchestrator:  [{p2p_icon}] {'Running' if status['p2p_running'] else 'Stopped'}")
        print(f"  Master Loop:       [{ml_icon}] {'Running' if status['master_loop_running'] else 'Stopped'}")
        print(f"  P2P Alive Peers:   {status['p2p_alive_peers']}")
        print(f"  P2P Leader:        {status['p2p_leader'] or 'None'}")
        print(f"  Selfplay Jobs:     {status['selfplay_jobs']} (P2P reported)")

        if status["actions_taken"]:
            print(f"\n  Actions Taken:     {', '.join(status['actions_taken'])}")

        print("=" * 60 + "\n")

    def run_once(self) -> dict:
        """Run a single check cycle."""
        status = self.check_and_recover()
        self.print_status(status)
        return status

    def run_forever(self):
        """Run watchdog loop indefinitely."""
        self.log("Starting cluster watchdog (Ctrl+C to stop)")
        self.log(f"Check interval: {self.check_interval}s")

        try:
            while True:
                status = self.check_and_recover()
                self.print_status(status)

                if status["actions_taken"]:
                    # Wait a bit after taking actions
                    time.sleep(30)
                else:
                    time.sleep(self.check_interval)

        except KeyboardInterrupt:
            self.log("Watchdog stopped by user")


def main():
    parser = argparse.ArgumentParser(description="Cluster watchdog for sustained operation")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    ai_service = script_path.parent.parent
    project_root = ai_service.parent

    watchdog = ClusterWatchdog(project_root)
    watchdog.check_interval = args.interval

    if args.status or args.once:
        watchdog.run_once()
    else:
        watchdog.run_forever()


if __name__ == "__main__":
    main()
