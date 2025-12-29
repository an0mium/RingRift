#!/usr/bin/env python3
"""Sync selfplay data from cluster nodes to coordinator.

Usage:
    # Sync from all configured nodes
    python scripts/sync_selfplay_data.py

    # Sync from specific node
    python scripts/sync_selfplay_data.py --node nebius-h100-3

    # Sync only specific config
    python scripts/sync_selfplay_data.py --config hex8_2p

    # Dry run (show what would be synced)
    python scripts/sync_selfplay_data.py --dry-run

December 28, 2025 - Post-selfplay data synchronization.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Cluster nodes with selfplay data
SELFPLAY_NODES = [
    "nebius-backbone-1",
    "nebius-h100-3",
    "vultr-a100-20gb",
]

# Remote paths to check for selfplay data
REMOTE_PATHS = [
    "~/ringrift/ai-service/data/selfplay",
    "~/ringrift/ai-service/data/games",
]

# Local destination
LOCAL_GAMES_DIR = Path("data/games")
LOCAL_SELFPLAY_DIR = Path("data/selfplay")


@dataclass
class SyncResult:
    """Result of a sync operation."""

    node: str
    files_synced: int
    bytes_synced: int
    errors: list[str]
    duration_seconds: float


def run_command(cmd: list[str], timeout: int = 300) -> tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def find_databases(node: str, config: str | None = None) -> list[str]:
    """Find database files on a remote node."""
    databases = []

    for remote_path in REMOTE_PATHS:
        # Find all .db files
        cmd = ["ssh", node, f"find {remote_path} -name '*.db' -type f 2>/dev/null"]
        success, stdout, stderr = run_command(cmd, timeout=60)

        if success and stdout:
            for line in stdout.strip().split("\n"):
                if line.strip():
                    # Filter by config if specified
                    if config:
                        if config.replace("_", "") in line or config in line:
                            databases.append(line.strip())
                    else:
                        databases.append(line.strip())

    return databases


def sync_database(node: str, remote_path: str, dry_run: bool = False) -> SyncResult:
    """Sync a single database from remote node."""
    start_time = datetime.now()
    errors = []

    # Determine local destination based on path
    filename = Path(remote_path).name
    if "selfplay" in remote_path:
        local_dir = LOCAL_SELFPLAY_DIR / node
    else:
        local_dir = LOCAL_GAMES_DIR

    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename

    if dry_run:
        logger.info(f"[DRY RUN] Would sync: {node}:{remote_path} -> {local_path}")
        return SyncResult(
            node=node,
            files_synced=0,
            bytes_synced=0,
            errors=[],
            duration_seconds=0,
        )

    # Use rsync for efficient transfer
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        f"{node}:{remote_path}",
        str(local_path),
    ]

    logger.info(f"Syncing: {node}:{remote_path} -> {local_path}")
    success, stdout, stderr = run_command(cmd, timeout=600)

    if not success:
        errors.append(f"rsync failed: {stderr}")
        logger.error(f"Failed to sync {remote_path}: {stderr}")

    duration = (datetime.now() - start_time).total_seconds()

    # Get file size
    bytes_synced = 0
    if local_path.exists():
        bytes_synced = local_path.stat().st_size

    return SyncResult(
        node=node,
        files_synced=1 if success else 0,
        bytes_synced=bytes_synced,
        errors=errors,
        duration_seconds=duration,
    )


def consolidate_databases(config: str | None = None) -> None:
    """Consolidate synced databases into canonical databases."""
    logger.info("Consolidating databases...")

    # Find all synced databases
    synced_dbs = list(LOCAL_SELFPLAY_DIR.glob("**/*.db"))

    for db_path in synced_dbs:
        # Determine config from path/filename
        db_config = None
        for cfg in ["hex8_2p", "hex8_3p", "hex8_4p", "square8_2p", "square8_3p", "square8_4p",
                    "square19_2p", "square19_3p", "square19_4p", "hexagonal_2p", "hexagonal_3p", "hexagonal_4p"]:
            if cfg.replace("_", "") in str(db_path) or cfg in str(db_path):
                db_config = cfg
                break

        if config and db_config != config:
            continue

        if db_config:
            canonical_db = LOCAL_GAMES_DIR / f"canonical_{db_config}.db"
            logger.info(f"Would consolidate {db_path} into {canonical_db}")
            # Actual consolidation would use SQLite ATTACH and INSERT


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync selfplay data from cluster")
    parser.add_argument("--node", type=str, help="Specific node to sync from")
    parser.add_argument("--config", type=str, help="Specific config to sync (e.g., hex8_2p)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--consolidate", action="store_true", help="Consolidate after sync")
    args = parser.parse_args()

    nodes = [args.node] if args.node else SELFPLAY_NODES

    total_files = 0
    total_bytes = 0
    all_errors = []

    for node in nodes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Checking {node}...")
        logger.info(f"{'='*60}")

        databases = find_databases(node, args.config)
        logger.info(f"Found {len(databases)} database(s)")

        for db_path in databases:
            result = sync_database(node, db_path, args.dry_run)
            total_files += result.files_synced
            total_bytes += result.bytes_synced
            all_errors.extend(result.errors)

    # Summary
    print(f"\n{'='*60}")
    print("SYNC SUMMARY")
    print(f"{'='*60}")
    print(f"Files synced: {total_files}")
    print(f"Bytes synced: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Errors: {len(all_errors)}")

    if all_errors:
        print("\nErrors:")
        for err in all_errors:
            print(f"  - {err}")

    if args.consolidate and not args.dry_run:
        consolidate_databases(args.config)

    return 0 if not all_errors else 1


if __name__ == "__main__":
    sys.exit(main())
