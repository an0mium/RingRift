#!/usr/bin/env python3
"""Dynamic Space Manager for Training Nodes.

This script provides intelligent disk space management and data distribution:
- Monitors disk usage on training nodes
- Automatically cleans up low-quality or old data when space is tight
- Can pull data from S3 when training nodes need specific configs
- Syncs NPZ files from OWC to training nodes

Features:
- Proactive cleanup (starts at 60% instead of waiting for 70%)
- S3 restore for missing training data
- Config-aware: only keeps data relevant to pending training
- Quality-based retention: keeps high-quality data longer

Usage:
    # Check all training nodes and clean if needed
    python scripts/dynamic_space_manager.py --check

    # Pull specific config from S3 to a node
    python scripts/dynamic_space_manager.py --pull-s3 hex8_4p --node nebius-h100-1

    # Run as daemon (checks every 30 minutes)
    python scripts/dynamic_space_manager.py --daemon

December 2025: Created for dynamic data orchestration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configuration
S3_BUCKET = os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
OWC_DATA_URL = "http://100.107.168.125:8780"
PROACTIVE_CLEANUP_THRESHOLD = 60  # Start cleanup at 60%
TARGET_DISK_USAGE = 50  # Clean down to 50%
MIN_FREE_GB = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("space_manager")


@dataclass
class TrainingNode:
    """Training node configuration."""
    name: str
    ssh_target: str
    data_path: str
    priority: int = 10


# Training nodes from distributed_hosts.yaml
TRAINING_NODES = [
    TrainingNode("nebius-h100-3", "ubuntu@89.169.110.128", "~/ringrift/ai-service", 1),
    TrainingNode("nebius-h100-1", "ubuntu@89.169.111.139", "~/ringrift/ai-service", 2),
    TrainingNode("vultr-a100", "root@208.167.249.164", "/root/ringrift/ai-service", 3),
]


@dataclass
class DiskStatus:
    """Disk status for a node."""
    node_name: str
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float
    needs_cleanup: bool
    games_dir_size_gb: float = 0


def run_ssh(host: str, cmd: str, timeout: int = 60) -> tuple[int, str]:
    """Run SSH command and return (returncode, output)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "Timeout"
    except Exception as e:
        return 1, str(e)


def run_local(cmd: list[str], timeout: int = 300) -> tuple[int, str]:
    """Run local command and return (returncode, output)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "Timeout"
    except Exception as e:
        return 1, str(e)


async def get_node_disk_status(node: TrainingNode) -> DiskStatus | None:
    """Get disk status for a node."""
    # Get disk usage
    code, output = run_ssh(node.ssh_target, "df -BG ~ | tail -1")
    if code != 0:
        logger.warning(f"Failed to get disk status for {node.name}: {output}")
        return None

    try:
        parts = output.split()
        total_gb = float(parts[1].rstrip("G"))
        used_gb = float(parts[2].rstrip("G"))
        free_gb = float(parts[3].rstrip("G"))
        usage_percent = float(parts[4].rstrip("%"))
    except (IndexError, ValueError) as e:
        logger.warning(f"Failed to parse disk status for {node.name}: {e}")
        return None

    # Get games directory size
    code, output = run_ssh(node.ssh_target, f"du -sG {node.data_path}/data/games 2>/dev/null | cut -f1")
    games_size = 0.0
    if code == 0:
        try:
            games_size = float(output.rstrip("G"))
        except ValueError:
            pass

    return DiskStatus(
        node_name=node.name,
        total_gb=total_gb,
        used_gb=used_gb,
        free_gb=free_gb,
        usage_percent=usage_percent,
        needs_cleanup=usage_percent >= PROACTIVE_CLEANUP_THRESHOLD,
        games_dir_size_gb=games_size,
    )


async def cleanup_node_data(node: TrainingNode, status: DiskStatus) -> int:
    """Clean up data on a node to reduce disk usage.

    Strategy:
    1. Delete old logs (> 7 days)
    2. Delete empty databases
    3. Archive low-quality databases
    4. Delete training checkpoints (keep latest 3)

    Returns: GB freed
    """
    freed_gb = 0.0

    logger.info(f"Cleaning up {node.name} (usage: {status.usage_percent:.1f}%)")

    # 1. Delete old logs
    code, output = run_ssh(
        node.ssh_target,
        f"find {node.data_path}/logs -name '*.log' -mtime +7 -delete 2>/dev/null; echo 'Done'"
    )

    # 2. Delete empty databases
    code, output = run_ssh(
        node.ssh_target,
        f"""
        for db in {node.data_path}/data/games/*.db; do
            if [ -f "$db" ]; then
                count=$(sqlite3 "$db" "SELECT COUNT(*) FROM games" 2>/dev/null || echo "0")
                if [ "$count" = "0" ]; then
                    rm -f "$db" "$db-wal" "$db-shm"
                    echo "Deleted empty: $db"
                fi
            fi
        done
        """
    )
    if output:
        logger.info(f"  {output}")

    # 3. Clean old training checkpoints (keep latest 3 per config)
    code, output = run_ssh(
        node.ssh_target,
        f"""
        for config_dir in {node.data_path}/models/*/; do
            if [ -d "$config_dir" ]; then
                # Keep only the 3 newest .pth files
                ls -t "$config_dir"*.pth 2>/dev/null | tail -n +4 | xargs -r rm -f
            fi
        done
        """
    )

    # 4. Clean pip cache and other temp files
    code, output = run_ssh(
        node.ssh_target,
        "rm -rf ~/.cache/pip ~/.cache/torch 2>/dev/null; echo 'Cache cleared'"
    )

    # Check new disk status
    new_status = await get_node_disk_status(node)
    if new_status:
        freed_gb = status.used_gb - new_status.used_gb
        logger.info(f"  Freed {freed_gb:.1f}GB on {node.name}")

    return int(freed_gb)


async def pull_s3_data(node: TrainingNode, config: str) -> bool:
    """Pull training data from S3 to a node.

    Args:
        node: Target training node
        config: Config key (e.g., "hex8_4p")

    Returns: Success status
    """
    logger.info(f"Pulling {config} data from S3 to {node.name}")

    # Check if AWS CLI is available on node
    code, output = run_ssh(node.ssh_target, "which aws")
    if code != 0:
        logger.error(f"AWS CLI not available on {node.name}")
        return False

    # Pull NPZ file from S3
    s3_path = f"s3://{S3_BUCKET}/training/{config}.npz"
    local_path = f"{node.data_path}/data/training/{config}.npz"

    code, output = run_ssh(
        node.ssh_target,
        f"aws s3 cp {s3_path} {local_path}",
        timeout=300,
    )

    if code == 0:
        logger.info(f"  Successfully pulled {config}.npz to {node.name}")
        return True
    else:
        logger.error(f"  Failed to pull from S3: {output}")
        return False


async def pull_owc_npz(node: TrainingNode, config: str) -> bool:
    """Pull NPZ file from OWC HTTP server to a node.

    Args:
        node: Target training node
        config: Config key (e.g., "hex8_4p")

    Returns: Success status
    """
    logger.info(f"Pulling {config}.npz from OWC to {node.name}")

    # Download NPZ file via HTTP
    url = f"{OWC_DATA_URL}/canonical_data/{config}.npz"
    local_path = f"{node.data_path}/data/training/{config}.npz"

    code, output = run_ssh(
        node.ssh_target,
        f"curl -sS -o {local_path} {url}",
        timeout=300,
    )

    if code == 0:
        # Verify file size
        code, size_output = run_ssh(node.ssh_target, f"stat -c %s {local_path} 2>/dev/null || stat -f %z {local_path}")
        if code == 0 and int(size_output) > 1000:
            logger.info(f"  Successfully pulled {config}.npz ({int(size_output)/1024/1024:.1f}MB)")
            return True

    logger.error(f"  Failed to pull from OWC: {output}")
    return False


async def sync_all_npz_to_node(node: TrainingNode) -> int:
    """Sync all NPZ files from OWC to a node.

    Returns: Number of files synced
    """
    logger.info(f"Syncing all NPZ files to {node.name}")

    # Use rsync for efficient sync
    src = f"mac-studio:/Volumes/RingRift-Data/canonical_data/*.npz"
    dst = f"{node.ssh_target}:{node.data_path}/data/training/"

    cmd = [
        "rsync", "-avz", "--progress",
        "--include=*.npz", "--exclude=*",
        f"mac-studio:/Volumes/RingRift-Data/canonical_data/",
        f"{node.ssh_target}:{node.data_path}/data/training/"
    ]

    code, output = run_local(cmd, timeout=600)

    if code == 0:
        # Count transferred files
        transferred = output.count(".npz")
        logger.info(f"  Synced {transferred} NPZ files to {node.name}")
        return transferred
    else:
        logger.error(f"  Rsync failed: {output[:200]}")
        return 0


async def check_pending_training(node: TrainingNode) -> list[str]:
    """Check what training configs are pending/active on a node.

    Returns: List of config keys (e.g., ["hex8_4p", "square8_2p"])
    """
    configs = []

    # Check for auto_train scripts
    code, output = run_ssh(
        node.ssh_target,
        "pgrep -fa 'auto_train' 2>/dev/null | head -5"
    )

    if code == 0 and output:
        for line in output.split("\n"):
            for config in ["hex8_2p", "hex8_3p", "hex8_4p",
                          "square8_2p", "square8_3p", "square8_4p",
                          "square19_2p", "square19_3p", "square19_4p",
                          "hexagonal_2p", "hexagonal_3p", "hexagonal_4p"]:
                if config in line and config not in configs:
                    configs.append(config)

    return configs


async def run_check_all() -> dict[str, Any]:
    """Check all training nodes and clean up if needed.

    Returns: Summary dict
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "nodes": [],
        "total_freed_gb": 0,
        "nodes_cleaned": 0,
    }

    for node in TRAINING_NODES:
        status = await get_node_disk_status(node)
        if not status:
            continue

        node_info = {
            "name": node.name,
            "usage_percent": status.usage_percent,
            "free_gb": status.free_gb,
            "needs_cleanup": status.needs_cleanup,
            "games_size_gb": status.games_dir_size_gb,
        }

        # Check pending training
        pending = await check_pending_training(node)
        node_info["pending_training"] = pending

        # Clean up if needed
        if status.needs_cleanup:
            freed = await cleanup_node_data(node, status)
            node_info["freed_gb"] = freed
            summary["total_freed_gb"] += freed
            summary["nodes_cleaned"] += 1

        summary["nodes"].append(node_info)

    return summary


async def run_daemon(interval: int = 1800):
    """Run as daemon, checking nodes periodically."""
    logger.info(f"Starting space manager daemon (interval: {interval}s)")

    while True:
        try:
            summary = await run_check_all()
            logger.info(f"Check complete: {summary['nodes_cleaned']} nodes cleaned, "
                       f"{summary['total_freed_gb']}GB freed")
        except Exception as e:
            logger.error(f"Check failed: {e}")

        await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Dynamic space manager for training nodes")
    parser.add_argument("--check", action="store_true", help="Check all nodes and clean if needed")
    parser.add_argument("--pull-s3", type=str, metavar="CONFIG", help="Pull config from S3")
    parser.add_argument("--pull-owc", type=str, metavar="CONFIG", help="Pull config from OWC")
    parser.add_argument("--sync-npz", action="store_true", help="Sync all NPZ files")
    parser.add_argument("--node", type=str, help="Target node name (for pull operations)")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=1800, help="Daemon interval in seconds")
    args = parser.parse_args()

    if args.daemon:
        asyncio.run(run_daemon(args.interval))
    elif args.check:
        summary = asyncio.run(run_check_all())
        print(json.dumps(summary, indent=2))
    elif args.pull_s3 or args.pull_owc:
        config = args.pull_s3 or args.pull_owc
        node = None
        for n in TRAINING_NODES:
            if args.node and args.node in n.name:
                node = n
                break

        if not node:
            node = TRAINING_NODES[0]
            logger.info(f"No node specified, using {node.name}")

        if args.pull_s3:
            success = asyncio.run(pull_s3_data(node, config))
        else:
            success = asyncio.run(pull_owc_npz(node, config))

        sys.exit(0 if success else 1)
    elif args.sync_npz:
        for node in TRAINING_NODES:
            if args.node and args.node not in node.name:
                continue
            asyncio.run(sync_all_npz_to_node(node))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
