#!/usr/bin/env python3
"""Simple cluster health monitoring.

Queries P2P health endpoints across all nodes and reports status.

Usage:
    python scripts/cluster_health_check.py
    python scripts/cluster_health_check.py --json
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

def _load_nodes_from_config() -> Dict[str, str]:
    """Load cluster nodes from config/distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"

    if not config_path.exists():
        print(f"[ClusterHealth] Warning: {config_path} not found")
        print("[ClusterHealth] Copy distributed_hosts.yaml.example to distributed_hosts.yaml")
        return {}

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        nodes = {}
        for name, info in config.get("hosts", {}).items():
            host = info.get("tailscale_ip") or info.get("ssh_host")
            port = info.get("p2p_port", 8770)
            if host:
                nodes[name] = f"{host}:{port}"

        return nodes
    except Exception as e:
        print(f"[ClusterHealth] Error loading config: {e}")
        return {}

NODES = _load_nodes_from_config()


def check_node(name: str, addr: str) -> Dict:
    """Check health of a single node."""
    url = f"http://{addr}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ClusterHealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return {
                "name": name,
                "status": "healthy",
                "selfplay_active": data.get("selfplay_active", False),
                "games_played": data.get("games_played", 0),
                "memory_available_gb": data.get("memory_available_gb"),
                "version": data.get("version", "unknown"),
            }
    except urllib.error.URLError as e:
        return {"name": name, "status": "unreachable", "error": str(e)}
    except Exception as e:
        return {"name": name, "status": "error", "error": str(e)}


def check_cluster() -> List[Dict]:
    """Check all nodes in the cluster."""
    results = []
    for name, addr in NODES.items():
        results.append(check_node(name, addr))
    return results


def main():
    parser = argparse.ArgumentParser(description="Cluster health check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = check_cluster()

    if args.json:
        print(json.dumps({"timestamp": datetime.now().isoformat(), "nodes": results}, indent=2))
        return

    # Text output
    print(f"\n{'='*60}")
    print(f"  CLUSTER HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    healthy = 0
    total = len(results)

    for node in results:
        status = node["status"]
        name = node["name"]

        if status == "healthy":
            healthy += 1
            selfplay = "✓" if node.get("selfplay_active") else "✗"
            games = node.get("games_played", 0)
            mem = node.get("memory_available_gb")
            mem_str = f"{mem:.0f}GB" if mem else "N/A"
            print(f"  ✅ {name:20} | selfplay={selfplay} | games={games:,} | mem={mem_str}")
        else:
            error = node.get("error", "unknown")[:40]
            print(f"  ❌ {name:20} | {status} ({error})")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {healthy}/{total} nodes healthy ({healthy/total*100:.0f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
