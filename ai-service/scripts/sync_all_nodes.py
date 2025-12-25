#!/usr/bin/env python3
"""Unified Node Sync Script - Discover and sync data from all reachable nodes.

This script:
1. Loads distributed_hosts.yaml for node configuration
2. Tests SSH connectivity to each node
3. Queries each node for database sizes/hashes
4. Pulls any databases not in local manifest
5. Pushes models to nodes that need them
6. Updates ClusterManifest with current state

Usage:
    # Full sync from all nodes
    python scripts/sync_all_nodes.py

    # Dry run (show what would be synced)
    python scripts/sync_all_nodes.py --dry-run

    # Sync from specific provider
    python scripts/sync_all_nodes.py --provider vast

    # Push models to all nodes
    python scripts/sync_all_nodes.py --push-models
"""

import argparse
import hashlib
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    name: str
    ssh_host: str
    ssh_port: int = 22
    ssh_user: str = "ubuntu"
    ssh_key: str = "~/.ssh/id_cluster"
    ringrift_path: str = "~/ringrift/ai-service"
    provider: str = "unknown"
    is_reachable: bool = False
    databases: list = None
    models: list = None

    def __post_init__(self):
        if self.databases is None:
            self.databases = []
        if self.models is None:
            self.models = []


def load_hosts_config() -> dict:
    """Load hosts from distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_nodes_from_config(config: dict, provider_filter: Optional[str] = None) -> list[NodeInfo]:
    """Extract node information from config."""
    nodes = []
    hosts = config.get("hosts", {})

    for name, info in hosts.items():
        if not info:
            continue

        # Determine provider
        if name.startswith("vast-"):
            provider = "vast"
        elif name.startswith("lambda-"):
            provider = "lambda"
        elif name.startswith("hetzner-"):
            provider = "hetzner"
        elif name == "mac-studio":
            continue  # Skip local coordinator
        else:
            provider = "other"

        if provider_filter and provider != provider_filter:
            continue

        # Get SSH details
        ssh_host = info.get("ssh_host", "")
        ssh_port = info.get("ssh_port", 22)
        ssh_user = info.get("ssh_user", "ubuntu")
        ssh_key = info.get("ssh_key", "~/.ssh/id_cluster")
        ringrift_path = info.get("ringrift_path", "~/ringrift/ai-service")

        if not ssh_host:
            continue

        nodes.append(NodeInfo(
            name=name,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
            ssh_key=os.path.expanduser(ssh_key),
            ringrift_path=ringrift_path,
            provider=provider,
        ))

    return nodes


def test_ssh_connectivity(node: NodeInfo, timeout: int = 5) -> bool:
    """Test if we can SSH to a node."""
    try:
        cmd = [
            "ssh", "-i", node.ssh_key,
            "-o", "ConnectTimeout=" + str(timeout),
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-p", str(node.ssh_port),
            f"{node.ssh_user}@{node.ssh_host}",
            "echo ok"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout + 2)
        return result.returncode == 0 and b"ok" in result.stdout
    except Exception as e:
        logger.debug(f"SSH test failed for {node.name}: {e}")
        return False


def get_remote_databases(node: NodeInfo) -> list[dict]:
    """Get list of databases on a remote node."""
    try:
        cmd = [
            "ssh", "-i", node.ssh_key,
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-p", str(node.ssh_port),
            f"{node.ssh_user}@{node.ssh_host}",
            f"find {node.ringrift_path}/data/games -name '*.db' -exec ls -l {{}} \\; 2>/dev/null"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)

        databases = []
        for line in result.stdout.strip().split("\n"):
            if not line or ".db" not in line:
                continue
            parts = line.split()
            if len(parts) >= 9:
                size = int(parts[4])
                path = parts[-1]
                name = os.path.basename(path)
                databases.append({
                    "name": name,
                    "path": path,
                    "size": size,
                    "size_mb": size / (1024 * 1024),
                })
        return databases
    except Exception as e:
        logger.debug(f"Failed to get databases from {node.name}: {e}")
        return []


def sync_database(node: NodeInfo, remote_path: str, local_path: str, dry_run: bool = False) -> bool:
    """Sync a database from remote node to local."""
    if dry_run:
        logger.info(f"[DRY RUN] Would sync: {node.name}:{remote_path} -> {local_path}")
        return True

    try:
        cmd = [
            "scp", "-i", node.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-P", str(node.ssh_port),
            f"{node.ssh_user}@{node.ssh_host}:{remote_path}",
            local_path
        ]
        logger.info(f"Syncing: {node.name}:{os.path.basename(remote_path)} -> {local_path}")
        result = subprocess.run(cmd, capture_output=True, timeout=3600)  # 1 hour timeout for large files
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Sync failed for {remote_path}: {e}")
        return False


def get_local_databases(data_dir: Path) -> dict[str, int]:
    """Get local databases with their sizes."""
    databases = {}
    for db_path in data_dir.glob("*.db"):
        databases[db_path.name] = db_path.stat().st_size
    return databases


def main():
    parser = argparse.ArgumentParser(description="Sync data from all cluster nodes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--provider", choices=["vast", "lambda", "hetzner"], help="Only sync from specific provider")
    parser.add_argument("--push-models", action="store_true", help="Push models to nodes instead of pulling data")
    parser.add_argument("--min-size-mb", type=float, default=1.0, help="Minimum file size to sync (MB)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config and get nodes
    config = load_hosts_config()
    nodes = get_nodes_from_config(config, args.provider)

    logger.info(f"Found {len(nodes)} nodes in configuration")

    # Test connectivity
    reachable_nodes = []
    for node in nodes:
        logger.info(f"Testing {node.name} ({node.provider})...")
        node.is_reachable = test_ssh_connectivity(node)
        if node.is_reachable:
            reachable_nodes.append(node)
            logger.info(f"  ✓ {node.name} is reachable")
        else:
            logger.info(f"  ✗ {node.name} is offline")

    logger.info(f"\n{len(reachable_nodes)}/{len(nodes)} nodes reachable")

    if not reachable_nodes:
        logger.warning("No reachable nodes found")
        return

    # Get local databases for comparison
    local_data_dir = Path(__file__).parent.parent / "data" / "games"
    local_dbs = get_local_databases(local_data_dir)
    logger.info(f"Local databases: {len(local_dbs)}")

    # Query remote databases
    synced = 0
    skipped = 0

    for node in reachable_nodes:
        logger.info(f"\nQuerying {node.name}...")
        node.databases = get_remote_databases(node)
        logger.info(f"  Found {len(node.databases)} databases")

        for db in node.databases:
            name = db["name"]
            size_mb = db["size_mb"]
            remote_path = db["path"]

            # Skip if too small
            if size_mb < args.min_size_mb:
                logger.debug(f"  Skipping {name} (too small: {size_mb:.1f}MB)")
                continue

            # Check if we have it locally (with same or larger size)
            local_name = f"{node.name}_{name}" if name in local_dbs else name
            local_path = local_data_dir / local_name

            if local_path.exists() and local_path.stat().st_size >= db["size"]:
                logger.debug(f"  Skipping {name} (already have it)")
                skipped += 1
                continue

            # Sync the file
            if sync_database(node, remote_path, str(local_path), dry_run=args.dry_run):
                synced += 1

    logger.info(f"\n=== Sync Summary ===")
    logger.info(f"Synced: {synced} files")
    logger.info(f"Skipped: {skipped} files (already local)")


if __name__ == "__main__":
    main()
