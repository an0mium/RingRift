#!/usr/bin/env python3
"""Node resilience daemon - keeps nodes utilized even when disconnected from P2P.

This script runs as a background daemon on each node and ensures:
1. P2P orchestrator is running and connected
2. If P2P is unavailable, runs local selfplay as fallback
3. Periodically attempts to reconnect to P2P network
4. Auto-registers with coordinator when IP changes

Usage:
    # Run as daemon
    python scripts/node_resilience.py --node-id vast-5090-quad --coordinator http://192.222.53.22:8770

    # Run once (for cron)
    python scripts/node_resilience.py --node-id vast-5090-quad --coordinator http://192.222.53.22:8770 --once
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/node_resilience.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    """Configuration for this node."""
    node_id: str
    coordinator_url: str
    ai_service_dir: str
    num_gpus: int
    selfplay_script: str = "scripts/run_gpu_selfplay.py"
    p2p_port: int = 8765
    check_interval: int = 60  # seconds
    reconnect_interval: int = 300  # seconds
    max_local_selfplay_procs: int = 4
    disk_threshold: int = 80  # percent - trigger cleanup above this


class NodeResilience:
    """Keeps a node utilized and connected to the cluster."""

    def __init__(self, config: NodeConfig):
        self.config = config
        self.local_selfplay_pids: List[int] = []
        self.last_p2p_check = 0
        self.last_registration = 0
        self.p2p_connected = False
        self.running = True

    def get_public_ip(self) -> Optional[str]:
        """Get this machine's public IP address."""
        services = [
            "https://api.ipify.org",
            "https://icanhazip.com",
            "https://ifconfig.me/ip",
        ]
        for url in services:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    ip = response.read().decode().strip()
                    if ip:
                        return ip
            except Exception:
                continue
        return None

    def check_p2p_health(self) -> bool:
        """Check if P2P orchestrator is running and connected."""
        try:
            url = f"http://localhost:{self.config.p2p_port}/health"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                return data.get("status") == "ok"
        except Exception:
            return False

    def check_coordinator_reachable(self) -> bool:
        """Check if the coordinator is reachable."""
        try:
            url = f"{self.config.coordinator_url}/health"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get("status") == "ok"
        except Exception:
            return False

    def register_with_coordinator(self) -> bool:
        """Register this node with the coordinator."""
        ip = self.get_public_ip()
        if not ip:
            logger.warning("Failed to get public IP for registration")
            return False

        try:
            url = f"{self.config.coordinator_url}/register"
            payload = {
                "node_id": self.config.node_id,
                "host": ip,
                "port": self._get_ssh_port(),
            }
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode())
                if result.get("success"):
                    logger.info(f"Registered {self.config.node_id} at {ip}")
                    return True
        except Exception as e:
            logger.warning(f"Registration failed: {e}")
        return False

    def _get_ssh_port(self) -> int:
        """Get SSH port from environment or default."""
        return int(os.environ.get("SSH_PORT", 22))

    def start_p2p_orchestrator(self) -> bool:
        """Start the P2P orchestrator if not running."""
        if self.check_p2p_health():
            return True

        logger.info("Starting P2P orchestrator...")
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self.config.ai_service_dir

            proc = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(self.config.ai_service_dir, "scripts/p2p_orchestrator.py"),
                    "--port", str(self.config.p2p_port),
                ],
                cwd=self.config.ai_service_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(3)
            if proc.poll() is None and self.check_p2p_health():
                logger.info(f"P2P orchestrator started (PID {proc.pid})")
                return True
        except Exception as e:
            logger.error(f"Failed to start P2P orchestrator: {e}")
        return False

    def start_local_selfplay(self) -> None:
        """Start local selfplay processes as fallback when P2P is unavailable."""
        # Clean up dead processes
        self.local_selfplay_pids = [
            pid for pid in self.local_selfplay_pids
            if self._process_running(pid)
        ]

        num_to_start = min(
            self.config.num_gpus,
            self.config.max_local_selfplay_procs
        ) - len(self.local_selfplay_pids)

        if num_to_start <= 0:
            return

        logger.info(f"Starting {num_to_start} local selfplay processes (P2P fallback)")

        for i in range(num_to_start):
            gpu_id = (len(self.local_selfplay_pids) + i) % self.config.num_gpus
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = self.config.ai_service_dir
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

                proc = subprocess.Popen(
                    [
                        sys.executable,
                        os.path.join(self.config.ai_service_dir, self.config.selfplay_script),
                        "--board-type", "square8",
                        "--num-players", "2",
                        "--games", "1000",
                        "--output-dir", os.path.join(
                            self.config.ai_service_dir,
                            f"data/selfplay/local_fallback_{self.config.node_id}"
                        ),
                    ],
                    cwd=self.config.ai_service_dir,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.local_selfplay_pids.append(proc.pid)
                logger.info(f"Started local selfplay on GPU {gpu_id} (PID {proc.pid})")
            except Exception as e:
                logger.error(f"Failed to start selfplay on GPU {gpu_id}: {e}")

    def stop_local_selfplay(self) -> None:
        """Stop all local selfplay processes (when P2P reconnects)."""
        for pid in self.local_selfplay_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Stopped local selfplay (PID {pid})")
            except ProcessLookupError:
                pass
        self.local_selfplay_pids = []

    def check_and_cleanup_disk(self) -> bool:
        """Check disk usage and run cleanup if needed."""
        try:
            stat = os.statvfs("/")
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used_percent = ((total - free) / total) * 100 if total > 0 else 0

            if used_percent > self.config.disk_threshold:
                logger.warning(f"Disk usage {used_percent:.1f}% exceeds threshold {self.config.disk_threshold}%")

                # Try to run disk_monitor.py if available
                disk_monitor = os.path.join(self.config.ai_service_dir, "scripts/disk_monitor.py")
                if os.path.exists(disk_monitor):
                    logger.info("Running disk cleanup...")
                    result = subprocess.run(
                        [sys.executable, disk_monitor, "--aggressive", "--threshold", str(self.config.disk_threshold)],
                        cwd=self.config.ai_service_dir,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode == 0:
                        logger.info("Disk cleanup completed successfully")
                        return True
                    else:
                        logger.warning(f"Disk cleanup failed: {result.stderr}")
                else:
                    logger.warning("disk_monitor.py not found, skipping cleanup")

                return False
            return True
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return False

    def _process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    def detect_num_gpus(self) -> int:
        """Detect number of available GPUs."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return len(result.stdout.strip().split("\n"))
        except Exception:
            pass
        return 0

    def run_once(self) -> None:
        """Run a single check cycle."""
        now = time.time()

        # Check disk and cleanup if needed (critical for Vast instances)
        self.check_and_cleanup_disk()

        # Check P2P health
        p2p_healthy = self.check_p2p_health()
        coordinator_reachable = self.check_coordinator_reachable()

        if p2p_healthy and coordinator_reachable:
            if not self.p2p_connected:
                logger.info("P2P connection restored - stopping local fallback")
                self.stop_local_selfplay()
            self.p2p_connected = True
        else:
            if self.p2p_connected:
                logger.warning("P2P connection lost - starting local fallback")
            self.p2p_connected = False

            # Try to start P2P orchestrator
            if not p2p_healthy:
                self.start_p2p_orchestrator()

            # Start local selfplay as fallback
            if self.config.num_gpus > 0:
                self.start_local_selfplay()

        # Periodic registration
        if now - self.last_registration > self.config.reconnect_interval:
            if coordinator_reachable:
                self.register_with_coordinator()
            self.last_registration = now

    def run_daemon(self) -> None:
        """Run as a continuous daemon."""
        logger.info(f"Node resilience daemon started for {self.config.node_id}")
        logger.info(f"Coordinator: {self.config.coordinator_url}")
        logger.info(f"GPUs detected: {self.config.num_gpus}")

        def handle_signal(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False
            self.stop_local_selfplay()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            time.sleep(self.config.check_interval)


def main():
    parser = argparse.ArgumentParser(description="Node resilience daemon")
    parser.add_argument("--node-id", required=True, help="Node identifier")
    parser.add_argument("--coordinator", required=True, help="Coordinator URL")
    parser.add_argument("--ai-service-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="AI service directory")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (auto-detect if 0)")
    parser.add_argument("--p2p-port", type=int, default=8765, help="P2P orchestrator port")
    parser.add_argument("--check-interval", type=int, default=60, help="Health check interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run once and exit (for cron)")

    args = parser.parse_args()

    config = NodeConfig(
        node_id=args.node_id,
        coordinator_url=args.coordinator,
        ai_service_dir=args.ai_service_dir,
        num_gpus=args.num_gpus,
        p2p_port=args.p2p_port,
        check_interval=args.check_interval,
    )

    resilience = NodeResilience(config)

    # Auto-detect GPUs if not specified
    if config.num_gpus == 0:
        config.num_gpus = resilience.detect_num_gpus()
        logger.info(f"Auto-detected {config.num_gpus} GPUs")

    if args.once:
        resilience.run_once()
    else:
        resilience.run_daemon()


if __name__ == "__main__":
    main()
