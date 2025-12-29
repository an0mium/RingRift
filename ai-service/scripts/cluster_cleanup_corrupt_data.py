#!/usr/bin/env python3
"""Cluster-wide cleanup of corrupt game data.

This script runs the cleanup_corrupt_games.py script across all storage locations
including local databases, cluster nodes, OWC drive, and S3 backups.

Usage:
    # Dry run (preview)
    python scripts/cluster_cleanup_corrupt_data.py --dry-run

    # Execute on all nodes
    python scripts/cluster_cleanup_corrupt_data.py --execute

    # Specific node only
    python scripts/cluster_cleanup_corrupt_data.py --node nebius-backbone-1 --execute

    # Local only (no SSH)
    python scripts/cluster_cleanup_corrupt_data.py --local-only --execute

    # Verify after cleanup
    python scripts/cluster_cleanup_corrupt_data.py --verify

December 28, 2025: Created for Phase 14 null position corruption cleanup.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("cluster_cleanup_corrupt_data")


@dataclass
class CleanupResult:
    """Result from cleaning a storage location."""

    location: str
    node: str
    corrupt_games: int = 0
    deleted_games: int = 0
    valid_games: int = 0
    error: str | None = None
    databases_scanned: int = 0


@dataclass
class ClusterCleanupStats:
    """Aggregate stats for cluster-wide cleanup."""

    total_databases: int = 0
    total_corrupt_games: int = 0
    total_deleted_games: int = 0
    total_valid_games: int = 0
    nodes_cleaned: int = 0
    nodes_failed: int = 0
    results: list[CleanupResult] = field(default_factory=list)


# Storage locations configuration
LOCAL_STORAGE = [
    {"path": "data/games", "description": "Local coordinator games"},
]

OWC_STORAGE = [
    {"path": "/Volumes/RingRift-Data/cluster_games", "description": "OWC cluster backups"},
    {"path": "/Volumes/RingRift-Data/canonical_data", "description": "OWC canonical databases"},
    {"path": "/Volumes/RingRift-Data/selfplay_repository", "description": "OWC selfplay repository"},
]


def get_cluster_nodes() -> list[dict[str, Any]]:
    """Get list of cluster nodes from configuration."""
    try:
        from app.config.cluster_config import get_cluster_nodes as _get_nodes

        nodes = _get_nodes()
        return [
            {
                "name": name,
                "ssh_host": node.ssh_host,
                "ssh_port": node.ssh_port,
                "ssh_user": node.ssh_user,
                "ssh_key": node.ssh_key,
                "tailscale_ip": node.tailscale_ip,
                "data_path": "ringrift/ai-service/data/games",
                "status": node.status,
            }
            for name, node in nodes.items()
            if node.status in ("ready", "active") and node.role not in ("coordinator",)
        ]
    except Exception as e:
        logger.warning(f"Could not load cluster config: {e}")
        return []


def run_cleanup_local(
    db_path: str,
    confirm: bool = False,
) -> CleanupResult:
    """Run cleanup on a local database."""
    result = CleanupResult(location=db_path, node="local")

    cleanup_script = ROOT / "scripts" / "cleanup_corrupt_games.py"
    if not cleanup_script.exists():
        result.error = "cleanup_corrupt_games.py not found"
        return result

    try:
        # First analyze
        analyze_cmd = [
            sys.executable,
            str(cleanup_script),
            "analyze",
            "--db",
            db_path,
        ]

        proc = subprocess.run(analyze_cmd, capture_output=True, text=True, timeout=300)

        # Parse analyze output
        for line in proc.stdout.split("\n"):
            if "Corrupt games:" in line:
                result.corrupt_games = int(line.split(":")[1].strip())
            elif "Valid games:" in line:
                result.valid_games = int(line.split(":")[1].strip())

        result.databases_scanned = 1

        if confirm and result.corrupt_games > 0:
            # Execute cleanup
            delete_cmd = [
                sys.executable,
                str(cleanup_script),
                "delete",
                "--db",
                db_path,
                "--confirm",
            ]

            proc = subprocess.run(delete_cmd, capture_output=True, text=True, timeout=600)

            # Parse delete output
            for line in proc.stdout.split("\n"):
                if "Deleted" in line and "corrupt games" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Deleted":
                            result.deleted_games = int(parts[i + 1])
                            break

    except subprocess.TimeoutExpired:
        result.error = "Cleanup timed out"
    except Exception as e:
        result.error = str(e)

    return result


def run_cleanup_remote(
    node: dict[str, Any],
    confirm: bool = False,
) -> CleanupResult:
    """Run cleanup on a remote cluster node."""
    node_name = node["name"]
    result = CleanupResult(location=node.get("data_path", "data/games"), node=node_name)

    # Build SSH command
    ssh_host = node.get("tailscale_ip") or node.get("ssh_host")
    ssh_port = node.get("ssh_port", 22)
    ssh_user = node.get("ssh_user", "root")
    ssh_key = node.get("ssh_key", "~/.ssh/id_cluster")

    if not ssh_host:
        result.error = "No SSH host available"
        return result

    # Build remote command
    data_path = node.get("data_path", "ringrift/ai-service/data/games")
    remote_script = f"~/ringrift/ai-service/scripts/cleanup_corrupt_games.py"

    # Analyze command
    analyze_cmd = f"cd ~/ringrift/ai-service && PYTHONPATH=. python {remote_script} analyze --db {data_path}"

    ssh_cmd = [
        "ssh",
        "-i",
        os.path.expanduser(ssh_key),
        "-p",
        str(ssh_port),
        "-o",
        "ConnectTimeout=30",
        "-o",
        "StrictHostKeyChecking=no",
        f"{ssh_user}@{ssh_host}",
        analyze_cmd,
    ]

    try:
        proc = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)

        if proc.returncode != 0:
            result.error = proc.stderr[:200] if proc.stderr else "SSH command failed"
            return result

        # Parse output
        for line in proc.stdout.split("\n"):
            if "Corrupt games:" in line:
                result.corrupt_games = int(line.split(":")[1].strip())
            elif "Valid games:" in line:
                result.valid_games = int(line.split(":")[1].strip())
            elif "Total games scanned:" in line:
                result.databases_scanned = 1

        if confirm and result.corrupt_games > 0:
            # Execute cleanup
            delete_cmd = f"cd ~/ringrift/ai-service && PYTHONPATH=. python {remote_script} delete --db {data_path} --confirm"

            ssh_cmd[-1] = delete_cmd
            proc = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=600)

            for line in proc.stdout.split("\n"):
                if "Deleted" in line and "corrupt games" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Deleted":
                            result.deleted_games = int(parts[i + 1])
                            break

    except subprocess.TimeoutExpired:
        result.error = "SSH timed out"
    except Exception as e:
        result.error = str(e)

    return result


def find_databases(directory: Path) -> list[Path]:
    """Find all .db files in a directory tree."""
    if not directory.exists():
        return []

    databases = []
    for db_file in directory.rglob("*.db"):
        # Skip WAL and SHM files
        if db_file.suffix in (".db-wal", ".db-shm"):
            continue
        databases.append(db_file)

    return databases


def cleanup_local_storage(
    storage_paths: list[dict[str, str]],
    confirm: bool = False,
) -> list[CleanupResult]:
    """Clean up all local storage locations."""
    results = []

    for storage in storage_paths:
        path = Path(storage["path"])
        description = storage.get("description", str(path))

        logger.info(f"Scanning {description}...")

        if not path.exists():
            logger.warning(f"  Path does not exist: {path}")
            continue

        databases = find_databases(path)

        if not databases:
            logger.info(f"  No databases found in {path}")
            continue

        logger.info(f"  Found {len(databases)} databases")

        for db_path in databases:
            result = run_cleanup_local(str(db_path), confirm=confirm)
            results.append(result)

            if result.error:
                logger.warning(f"  {db_path.name}: Error - {result.error}")
            elif result.corrupt_games > 0:
                logger.info(
                    f"  {db_path.name}: {result.corrupt_games} corrupt, "
                    f"{result.deleted_games} deleted, {result.valid_games} valid"
                )
            else:
                logger.debug(f"  {db_path.name}: {result.valid_games} valid games (no corruption)")

    return results


async def cleanup_cluster_async(
    nodes: list[dict[str, Any]],
    confirm: bool = False,
    max_concurrent: int = 5,
) -> list[CleanupResult]:
    """Clean up cluster nodes concurrently."""
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def cleanup_node(node: dict[str, Any]) -> CleanupResult:
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, run_cleanup_remote, node, confirm)

    tasks = [cleanup_node(node) for node in nodes]
    results = await asyncio.gather(*tasks)

    return list(results)


def cleanup_cluster(
    nodes: list[dict[str, Any]],
    confirm: bool = False,
) -> list[CleanupResult]:
    """Clean up cluster nodes."""
    return asyncio.run(cleanup_cluster_async(nodes, confirm))


def verify_cleanup(storage_paths: list[dict[str, str]]) -> bool:
    """Verify no corrupt games remain after cleanup."""
    all_clean = True

    for storage in storage_paths:
        path = Path(storage["path"])
        if not path.exists():
            continue

        databases = find_databases(path)

        for db_path in databases:
            result = run_cleanup_local(str(db_path), confirm=False)

            if result.corrupt_games > 0:
                logger.error(f"  {db_path}: Still has {result.corrupt_games} corrupt games!")
                all_clean = False
            else:
                logger.info(f"  {db_path}: Clean ({result.valid_games} valid games)")

    return all_clean


def main():
    parser = argparse.ArgumentParser(
        description="Cluster-wide cleanup of corrupt game data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview cleanup without deleting (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete corrupt games",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only clean local storage (no SSH to cluster)",
    )
    parser.add_argument(
        "--owc-only",
        action="store_true",
        help="Only clean OWC drive storage",
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Only clean cluster nodes (via SSH)",
    )
    parser.add_argument(
        "--node",
        type=str,
        help="Clean specific node only",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify no corrupt games remain",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    confirm = args.execute

    if confirm:
        logger.warning("EXECUTE MODE: Will delete corrupt games")
    else:
        logger.info("DRY RUN: Preview mode (use --execute to delete)")

    stats = ClusterCleanupStats()

    # Determine which locations to clean
    do_local = not args.owc_only and not args.cluster_only
    do_owc = not args.local_only and not args.cluster_only
    do_cluster = not args.local_only and not args.owc_only

    # Local coordinator storage
    if do_local:
        logger.info("\n=== Local Coordinator Storage ===")
        results = cleanup_local_storage(LOCAL_STORAGE, confirm=confirm)
        stats.results.extend(results)

    # OWC drive storage
    if do_owc:
        logger.info("\n=== OWC Drive Storage ===")
        results = cleanup_local_storage(OWC_STORAGE, confirm=confirm)
        stats.results.extend(results)

    # Cluster nodes
    if do_cluster:
        logger.info("\n=== Cluster Nodes ===")
        nodes = get_cluster_nodes()

        if args.node:
            nodes = [n for n in nodes if n["name"] == args.node]
            if not nodes:
                logger.error(f"Node not found: {args.node}")
                return 1

        if not nodes:
            logger.warning("No cluster nodes available")
        else:
            logger.info(f"Cleaning {len(nodes)} cluster nodes...")
            results = cleanup_cluster(nodes, confirm=confirm)
            stats.results.extend(results)

    # Aggregate stats
    for result in stats.results:
        stats.total_corrupt_games += result.corrupt_games
        stats.total_deleted_games += result.deleted_games
        stats.total_valid_games += result.valid_games
        stats.total_databases += result.databases_scanned

        if result.error:
            stats.nodes_failed += 1
        else:
            stats.nodes_cleaned += 1

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Databases scanned: {stats.total_databases}")
    print(f"Total corrupt games: {stats.total_corrupt_games}")
    print(f"Total deleted games: {stats.total_deleted_games}")
    print(f"Total valid games: {stats.total_valid_games}")
    print(f"Locations cleaned: {stats.nodes_cleaned}")
    print(f"Locations failed: {stats.nodes_failed}")

    if not confirm and stats.total_corrupt_games > 0:
        print("\n" + "-" * 60)
        print("To delete corrupt games, run with --execute")

    # Verify if requested
    if args.verify:
        logger.info("\n=== Verification ===")
        all_clean = verify_cleanup(LOCAL_STORAGE + OWC_STORAGE)
        if all_clean:
            logger.info("All storage locations are clean!")
            return 0
        else:
            logger.error("Some locations still have corrupt games")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
