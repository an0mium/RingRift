#!/usr/bin/env python3
"""Sync training data (NPZ files) to training nodes.

This script distributes NPZ training files from the OWC drive to cluster
training nodes based on their storage capacity and training needs.

Usage:
    # Sync to a specific node
    python scripts/sync_training_data.py --node nebius-h100-3 --config square8_2p

    # Sync all critical configs to all training nodes
    python scripts/sync_training_data.py --all-critical

    # Check storage on training nodes
    python scripts/sync_training_data.py --check-storage

December 2025: Created for distributing Lambda backup data to training nodes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Data sources
OWC_NPZ_PATH = "/Volumes/RingRift-Data/consolidated_training"
OWC_CANONICAL_PATH = "/Volumes/RingRift-Data/canonical_data"
MAC_STUDIO_HOST = "mac-studio"

# Training nodes with storage capacity
TRAINING_NODES = {
    "nebius-h100-3": {
        "host": "nebius-h100-3",
        "storage_gb": 452,
        "gpu": "H100 80GB",
        "data_path": "~/ringrift/ai-service/data/training",
        "priority": 1,  # Highest priority
    },
    "nebius-h100-1": {
        "host": "nebius-h100-1",
        "storage_gb": 58,
        "gpu": "H100 80GB",
        "data_path": "~/ringrift/ai-service/data/training",
        "priority": 2,
    },
    "nebius-backbone-1": {
        "host": "nebius-backbone-1",
        "storage_gb": 61,
        "gpu": "L40S 48GB",
        "data_path": "~/ringrift/ai-service/data/training",
        "priority": 3,
    },
}

# Critical gap configurations (prioritize these)
CRITICAL_CONFIGS = [
    "square8_4p",      # 0 games in canonical, 130K in Lambda
    "hexagonal_2p",    # 21 games in canonical, 118K in Lambda
    "hexagonal_3p",    # 0 games in canonical, 60K in Lambda
    "hexagonal_4p",    # 8 games in canonical, 65K in Lambda
    "square19_2p",     # 77 games in canonical, 86K in Lambda
    "square19_3p",     # 150 games in canonical, 61K in Lambda
    "square19_4p",     # 0 games in canonical, 33K in Lambda
    "square8_3p",      # 494 games in canonical, 81K in Lambda
]

# All configs
ALL_CONFIGS = [
    "square8_2p", "square8_3p", "square8_4p",
    "hex8_2p", "hex8_3p", "hex8_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    "square19_2p", "square19_3p", "square19_4p",
]


@dataclass
class SyncResult:
    """Result of a sync operation."""
    config: str
    node: str
    success: bool
    size_mb: float
    duration_seconds: float
    error: str = ""


def run_ssh_command(host: str, command: str, timeout: int = 30) -> tuple[bool, str]:
    """Run SSH command and return (success, output)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def check_node_storage(node_name: str) -> dict[str, Any]:
    """Check available storage on a training node."""
    node = TRAINING_NODES.get(node_name)
    if not node:
        return {"error": f"Unknown node: {node_name}"}

    success, output = run_ssh_command(
        node["host"],
        "df -h / 2>/dev/null | tail -1"
    )

    if success:
        parts = output.split()
        if len(parts) >= 4:
            try:
                avail = parts[3]
                # Parse available space
                if avail.endswith("G"):
                    avail_gb = float(avail[:-1])
                elif avail.endswith("T"):
                    avail_gb = float(avail[:-1]) * 1024
                elif avail.endswith("M"):
                    avail_gb = float(avail[:-1]) / 1024
                else:
                    avail_gb = 0

                return {
                    "node": node_name,
                    "host": node["host"],
                    "gpu": node["gpu"],
                    "available_gb": avail_gb,
                    "configured_gb": node["storage_gb"],
                    "status": "ready" if avail_gb > 10 else "low_space",
                }
            except ValueError:
                pass

    return {"node": node_name, "error": output or "Failed to get storage info"}


def get_npz_file_info(config: str) -> dict[str, Any]:
    """Get info about NPZ file for a config."""
    # Check consolidated training directory first
    npz_path = f"{OWC_NPZ_PATH}/lambda_{config}.npz"

    success, output = run_ssh_command(
        MAC_STUDIO_HOST,
        f"ls -lh {npz_path} 2>/dev/null"
    )

    if success and output:
        parts = output.split()
        if len(parts) >= 5:
            return {
                "config": config,
                "path": npz_path,
                "size": parts[4],
                "exists": True,
            }

    # Fall back to canonical data
    npz_path = f"{OWC_CANONICAL_PATH}/{config.split('_')[0]}/{config}.npz"
    success, output = run_ssh_command(
        MAC_STUDIO_HOST,
        f"ls -lh {npz_path} 2>/dev/null"
    )

    if success and output:
        parts = output.split()
        if len(parts) >= 5:
            return {
                "config": config,
                "path": npz_path,
                "size": parts[4],
                "exists": True,
            }

    return {"config": config, "exists": False}


def sync_npz_to_node(config: str, node_name: str) -> SyncResult:
    """Sync NPZ file for a config to a training node."""
    import time

    node = TRAINING_NODES.get(node_name)
    if not node:
        return SyncResult(config, node_name, False, 0, 0, f"Unknown node: {node_name}")

    npz_info = get_npz_file_info(config)
    if not npz_info.get("exists"):
        return SyncResult(config, node_name, False, 0, 0, f"NPZ not found for {config}")

    npz_path = npz_info["path"]
    dest_path = f"{node['host']}:{node['data_path']}/"

    logger.info(f"Syncing {config} to {node_name}...")
    start_time = time.time()

    try:
        # Create destination directory
        run_ssh_command(node["host"], f"mkdir -p {node['data_path']}")

        # Use rsync to sync the file
        result = subprocess.run(
            [
                "rsync", "-avz", "--progress",
                f"{MAC_STUDIO_HOST}:{npz_path}",
                dest_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            # Parse size from rsync output
            size_mb = 0
            for line in result.stdout.split("\n"):
                if "sent" in line and "bytes" in line:
                    try:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "bytes" and i > 0:
                                size_mb = int(parts[i-1].replace(",", "")) / (1024 * 1024)
                                break
                    except (ValueError, IndexError):
                        pass

            logger.info(f"Synced {config} to {node_name} ({size_mb:.1f}MB in {duration:.1f}s)")
            return SyncResult(config, node_name, True, size_mb, duration)
        else:
            return SyncResult(config, node_name, False, 0, duration, result.stderr[:200])

    except subprocess.TimeoutExpired:
        return SyncResult(config, node_name, False, 0, 600, "Sync timed out")
    except Exception as e:
        return SyncResult(config, node_name, False, 0, 0, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync training data to cluster nodes")
    parser.add_argument("--node", type=str, help="Target node name")
    parser.add_argument("--config", type=str, help="Config to sync (e.g., square8_2p)")
    parser.add_argument("--all-critical", action="store_true", help="Sync all critical configs")
    parser.add_argument("--all", action="store_true", help="Sync all configs")
    parser.add_argument("--check-storage", action="store_true", help="Check storage on all nodes")
    parser.add_argument("--list-npz", action="store_true", help="List available NPZ files")
    args = parser.parse_args()

    if args.check_storage:
        logger.info("Checking storage on training nodes...")
        for node_name in TRAINING_NODES:
            info = check_node_storage(node_name)
            if "error" in info:
                logger.warning(f"  {node_name}: {info['error']}")
            else:
                status = "✓" if info["status"] == "ready" else "⚠"
                logger.info(
                    f"  {status} {node_name} ({info['gpu']}): "
                    f"{info['available_gb']:.1f}GB available"
                )
        return

    if args.list_npz:
        logger.info("Checking available NPZ files...")
        for config in ALL_CONFIGS:
            info = get_npz_file_info(config)
            if info.get("exists"):
                logger.info(f"  ✓ {config}: {info['size']} at {info['path']}")
            else:
                logger.warning(f"  ✗ {config}: Not found")
        return

    # Determine what to sync
    if args.all_critical:
        configs = CRITICAL_CONFIGS
    elif args.all:
        configs = ALL_CONFIGS
    elif args.config:
        configs = [args.config]
    else:
        parser.print_help()
        return

    # Determine target nodes
    if args.node:
        nodes = [args.node]
    else:
        # Sort by priority (lowest first = highest priority)
        nodes = sorted(TRAINING_NODES.keys(), key=lambda n: TRAINING_NODES[n]["priority"])

    # Sync each config to each node
    results: list[SyncResult] = []
    for config in configs:
        for node in nodes:
            result = sync_npz_to_node(config, node)
            results.append(result)

            if result.success:
                break  # Successfully synced to one node, move to next config

    # Summary
    logger.info("\n=== Sync Summary ===")
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if successful:
        logger.info(f"Successful: {len(successful)}")
        for r in successful:
            logger.info(f"  ✓ {r.config} → {r.node} ({r.size_mb:.1f}MB in {r.duration_seconds:.1f}s)")

    if failed:
        logger.warning(f"Failed: {len(failed)}")
        for r in failed:
            logger.warning(f"  ✗ {r.config} → {r.node}: {r.error}")


if __name__ == "__main__":
    main()
