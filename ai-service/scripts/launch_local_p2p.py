#!/usr/bin/env python3
"""Launch P2P orchestrator with dynamically generated peers from config.

This script replaces hardcoded --peers in the launchd plist with a dynamic
lookup from distributed_hosts.yaml. This ensures the peer list stays in sync
with the cluster config without requiring manual plist updates.

Usage (from launchd plist):
    /path/to/.venv/bin/python /path/to/scripts/launch_local_p2p.py \
        --node-id local-mac --port 8770 --advertise-host 100.69.164.58

All unrecognized arguments are forwarded to p2p_orchestrator.py.

Feb 2026: Created to eliminate hardcoded peer lists in launchd plists.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure ai-service is on PYTHONPATH
AI_SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_SERVICE_DIR))


def get_all_peer_urls(exclude_node: str = "", port: int = 8770) -> list[str]:
    """Build peer URL list from distributed_hosts.yaml.

    Includes all active nodes with Tailscale IPs (not just voters),
    giving the orchestrator maximum bootstrap connectivity.

    Args:
        exclude_node: Node ID to exclude (self).
        port: P2P port.

    Returns:
        List of peer URLs like ["http://100.x.x.x:8770", ...].
    """
    try:
        from app.config.cluster_config import get_cluster_nodes
    except ImportError:
        # Fallback: load YAML directly
        return _get_peers_from_yaml(exclude_node, port)

    peers: list[str] = []
    seen_ips: set[str] = set()

    try:
        nodes = get_cluster_nodes()
    except Exception as e:
        print(f"[launch_local_p2p] Warning: could not load cluster config: {e}", file=sys.stderr)
        return _get_peers_from_yaml(exclude_node, port)

    for name, node in nodes.items():
        if name == exclude_node:
            continue
        # Skip inactive/retired nodes
        if hasattr(node, "status") and node.status in ("retired", "offline", "terminated", "archived"):
            continue
        ip = getattr(node, "tailscale_ip", None)
        if ip and str(ip) not in seen_ips:
            peers.append(f"http://{ip}:{port}")
            seen_ips.add(str(ip))

    return peers


def _get_peers_from_yaml(exclude_node: str, port: int) -> list[str]:
    """Fallback: load peers directly from YAML."""
    import yaml

    config_path = AI_SERVICE_DIR / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        return []

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    peers: list[str] = []
    seen_ips: set[str] = set()
    hosts = data.get("hosts", {})

    for name, info in hosts.items():
        if name == exclude_node:
            continue
        status = info.get("status", "")
        if status in ("retired", "offline", "terminated", "archived"):
            continue
        ip = info.get("tailscale_ip")
        if ip and ip not in seen_ips:
            peers.append(f"http://{ip}:{port}")
            seen_ips.add(ip)

    return peers


def main() -> None:
    """Generate peers and exec into p2p_orchestrator.py."""
    # Parse only what we need; pass the rest through
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--node-id", default="")
    parser.add_argument("--port", type=int, default=8770)
    known, remaining = parser.parse_known_args()

    node_id = known.node_id
    port = known.port

    # Generate peer list dynamically
    peers = get_all_peer_urls(exclude_node=node_id, port=port)
    peer_str = ",".join(peers)

    print(f"[launch_local_p2p] Generated {len(peers)} peers from config", file=sys.stderr)

    # Build the orchestrator command
    orchestrator_script = str(AI_SERVICE_DIR / "scripts" / "p2p_orchestrator.py")
    args = [
        sys.executable,
        orchestrator_script,
        "--node-id", node_id,
        "--port", str(port),
    ]
    if peers:
        args.extend(["--peers", peer_str])
    args.extend(remaining)

    # Replace this process with the orchestrator
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
