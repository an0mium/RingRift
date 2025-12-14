#!/usr/bin/env python3
"""Cluster health check script for RingRift training infrastructure.

Checks:
1. Node connectivity via SSH
2. GPU availability and utilization
3. P2P orchestrator status
4. AI service health endpoint
5. Disk space

Usage:
    python scripts/cluster_health_check.py
    python scripts/cluster_health_check.py --verbose --json
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Cluster nodes configuration
CLUSTER_NODES = [
    {"host": "192.222.53.22", "name": "lambda-gh200-1", "gpu": "GH200"},
    {"host": "192.222.53.23", "name": "lambda-gh200-2", "gpu": "GH200"},
    {"host": "192.222.53.24", "name": "lambda-gh200-3", "gpu": "GH200"},
]

AI_SERVICE_PORT = 8765
P2P_ORCHESTRATOR_PROCESS = "p2p_orchestrator"


@dataclass
class NodeHealth:
    """Health status of a single node."""
    host: str
    name: str
    gpu_type: str
    ssh_reachable: bool
    gpu_available: bool
    gpu_utilization: Optional[float]
    gpu_memory_used_gb: Optional[float]
    gpu_memory_total_gb: Optional[float]
    p2p_running: bool
    ai_service_healthy: bool
    disk_free_gb: Optional[float]
    error: Optional[str]

    @property
    def is_healthy(self) -> bool:
        """Check if node is fully healthy."""
        return (
            self.ssh_reachable and
            self.gpu_available and
            self.p2p_running and
            self.ai_service_healthy and
            (self.disk_free_gb is None or self.disk_free_gb > 10)
        )


def check_ssh(host: str, timeout: int = 5) -> bool:
    """Check if SSH to host is available."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, "echo", "ok"],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False


def check_gpu(host: str) -> Dict:
    """Check GPU status via nvidia-smi."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", host,
             "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"available": False}

        # Parse output: "45, 12000, 48000"
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            return {
                "available": True,
                "utilization": float(parts[0]) / 100.0,
                "memory_used_gb": float(parts[1]) / 1024,
                "memory_total_gb": float(parts[2]) / 1024,
            }
        return {"available": True, "utilization": None}
    except Exception:
        return {"available": False}


def check_p2p_orchestrator(host: str) -> bool:
    """Check if P2P orchestrator is running."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", host,
             f"pgrep -f {P2P_ORCHESTRATOR_PROCESS}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


def check_ai_service(host: str, port: int = AI_SERVICE_PORT) -> bool:
    """Check AI service health endpoint."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", host,
             f"curl -s --connect-timeout 2 http://localhost:{port}/health"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout.lower()
    except Exception:
        return False


def check_disk_space(host: str) -> Optional[float]:
    """Check available disk space in GB."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", host,
             "df -BG / | tail -1 | awk '{print $4}' | tr -d 'G'"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
        return None
    except Exception:
        return None


def check_node(node: Dict, verbose: bool = False) -> NodeHealth:
    """Check health of a single node."""
    host = node["host"]
    name = node["name"]
    gpu_type = node["gpu"]
    error = None

    if verbose:
        print(f"  Checking {name} ({host})...", end="", flush=True)

    # Check SSH first
    ssh_ok = check_ssh(host)
    if not ssh_ok:
        if verbose:
            print(" SSH failed")
        return NodeHealth(
            host=host, name=name, gpu_type=gpu_type,
            ssh_reachable=False, gpu_available=False,
            gpu_utilization=None, gpu_memory_used_gb=None, gpu_memory_total_gb=None,
            p2p_running=False, ai_service_healthy=False,
            disk_free_gb=None, error="SSH unreachable",
        )

    # Check all services
    gpu = check_gpu(host)
    p2p = check_p2p_orchestrator(host)
    ai_health = check_ai_service(host)
    disk = check_disk_space(host)

    health = NodeHealth(
        host=host, name=name, gpu_type=gpu_type,
        ssh_reachable=True,
        gpu_available=gpu.get("available", False),
        gpu_utilization=gpu.get("utilization"),
        gpu_memory_used_gb=gpu.get("memory_used_gb"),
        gpu_memory_total_gb=gpu.get("memory_total_gb"),
        p2p_running=p2p,
        ai_service_healthy=ai_health,
        disk_free_gb=disk,
        error=None,
    )

    if verbose:
        status = "OK" if health.is_healthy else "DEGRADED"
        print(f" {status}")

    return health


def check_cluster(nodes: List[Dict] = CLUSTER_NODES, verbose: bool = False) -> List[NodeHealth]:
    """Check health of all cluster nodes."""
    results = []
    for node in nodes:
        health = check_node(node, verbose=verbose)
        results.append(health)
    return results


def print_summary(results: List[NodeHealth]) -> None:
    """Print human-readable summary."""
    healthy = sum(1 for r in results if r.is_healthy)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"CLUSTER HEALTH: {healthy}/{total} nodes healthy")
    print(f"{'='*60}\n")

    for r in results:
        status = "OK" if r.is_healthy else "ISSUE"
        gpu_util = f"{r.gpu_utilization*100:.0f}%" if r.gpu_utilization is not None else "N/A"
        gpu_mem = f"{r.gpu_memory_used_gb:.1f}/{r.gpu_memory_total_gb:.1f}GB" if r.gpu_memory_used_gb else "N/A"

        print(f"[{status:5}] {r.name:20} ({r.gpu_type})")
        print(f"        SSH: {'OK' if r.ssh_reachable else 'FAIL'}")
        print(f"        GPU: {'OK' if r.gpu_available else 'FAIL'} (util: {gpu_util}, mem: {gpu_mem})")
        print(f"        P2P: {'OK' if r.p2p_running else 'FAIL'}")
        print(f"        AI:  {'OK' if r.ai_service_healthy else 'FAIL'}")
        if r.disk_free_gb is not None:
            print(f"        Disk: {r.disk_free_gb:.0f}GB free")
        if r.error:
            print(f"        Error: {r.error}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Cluster health check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--fail-on-issue", action="store_true",
                        help="Exit with non-zero code if any node has issues")
    args = parser.parse_args()

    if args.verbose:
        print("Checking cluster health...\n")

    results = check_cluster(CLUSTER_NODES, verbose=args.verbose)

    if args.json:
        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "healthy_nodes": sum(1 for r in results if r.is_healthy),
            "total_nodes": len(results),
            "nodes": [asdict(r) for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print_summary(results)

    if args.fail_on_issue:
        unhealthy = sum(1 for r in results if not r.is_healthy)
        return unhealthy

    return 0


if __name__ == "__main__":
    sys.exit(main())
