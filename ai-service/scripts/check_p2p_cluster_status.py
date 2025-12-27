#!/usr/bin/env python3
"""
Quick P2P cluster health check across all nodes.
"""
import asyncio
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def check_node_p2p(node_id, ssh_host, ssh_port, ssh_user, ssh_key):
    """Check if P2P is running on a node."""
    ssh_cmd = [
        "ssh",
        "-i",
        os.path.expanduser(ssh_key),
        "-p",
        str(ssh_port),
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=no",
        f"{ssh_user}@{ssh_host}",
        "curl -s --connect-timeout 3 http://localhost:8770/health 2>/dev/null",
    ]

    try:
        result = await asyncio.create_subprocess_exec(
            *ssh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10)

        if result.returncode == 0 and stdout:
            import json

            try:
                data = json.loads(stdout.decode())
                healthy = data.get("healthy", False)
                role = data.get("role", "unknown")
                leader = data.get("leader_id", "unknown")
                peers = data.get("active_peers", 0)
                return {
                    "status": "HEALTHY" if healthy else "UNHEALTHY",
                    "role": role,
                    "leader": leader,
                    "peers": peers,
                }
            except json.JSONDecodeError:
                return {"status": "ERROR", "error": "Invalid JSON"}
        else:
            return {"status": "DOWN", "error": "No response"}
    except asyncio.TimeoutError:
        return {"status": "TIMEOUT", "error": "Timeout"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


async def main():
    # Load cluster config
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    with open(config_path) as f:
        cluster_config = yaml.safe_load(f)

    # Get all nodes with P2P enabled
    nodes = []
    for node_id, node_data in cluster_config.get("hosts", {}).items():
        if node_data.get("p2p_enabled", True) and node_data.get("status") == "ready":
            if node_data.get("role") != "proxy":
                nodes.append(
                    {
                        "node_id": node_id,
                        "ssh_host": node_data.get("ssh_host"),
                        "ssh_port": node_data.get("ssh_port", 22),
                        "ssh_user": node_data.get("ssh_user", "root"),
                        "ssh_key": node_data.get("ssh_key", "~/.ssh/id_cluster"),
                    }
                )

    print(f"Checking P2P status on {len(nodes)} nodes...\n")

    # Check all nodes in parallel
    tasks = [
        check_node_p2p(
            n["node_id"], n["ssh_host"], n["ssh_port"], n["ssh_user"], n["ssh_key"]
        )
        for n in nodes
    ]
    results = await asyncio.gather(*tasks)

    # Print results
    print("=" * 100)
    print(f"{'Node ID':<30s} {'Status':<12s} {'Role':<10s} {'Leader':<20s} {'Peers':<6s}")
    print("=" * 100)

    status_counts = {"HEALTHY": 0, "UNHEALTHY": 0, "DOWN": 0, "TIMEOUT": 0, "ERROR": 0}

    for node, result in zip(nodes, results):
        status = result.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

        role = result.get("role", "-")
        leader = result.get("leader", "-")
        peers = result.get("peers", "-")
        error = result.get("error", "")

        if status == "HEALTHY":
            icon = "✓"
        elif status == "UNHEALTHY":
            icon = "~"
        else:
            icon = "✗"

        if error:
            info = f"Error: {error}"
        else:
            info = f"{str(role):<10s} {str(leader):<20s} {str(peers):<6s}"

        print(f"{icon} {node['node_id']:<28s} {status:<12s} {info}")

    print("=" * 100)
    print(f"\nSummary:")
    print(f"  Healthy:   {status_counts['HEALTHY']}")
    print(f"  Unhealthy: {status_counts['UNHEALTHY']}")
    print(f"  Down:      {status_counts['DOWN']}")
    print(f"  Timeout:   {status_counts['TIMEOUT']}")
    print(f"  Error:     {status_counts['ERROR']}")
    print(f"  Total:     {len(nodes)}")


if __name__ == "__main__":
    asyncio.run(main())
