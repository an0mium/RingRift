#!/usr/bin/env python3
"""Import Orphaned Databases - Register stranded databases into the ClusterManifest.

This script finds databases that exist on disk but are not registered in the
ClusterManifest, and imports them to enable proper replication.

Usage:
    # Scan local node and import orphaned databases
    python scripts/import_orphaned_databases.py

    # Dry-run mode (report only)
    python scripts/import_orphaned_databases.py --dry-run

    # Import from specific directory
    python scripts/import_orphaned_databases.py --data-dir /path/to/data/games

    # Trigger sync after import
    python scripts/import_orphaned_databases.py --trigger-sync

    # Import from remote node via SSH
    python scripts/import_orphaned_databases.py --remote lambda-gh200-b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class OrphanedDatabase:
    """Information about an orphaned database."""
    path: str
    size_bytes: int
    game_count: int
    board_type: str | None
    num_players: int | None
    game_ids: list[str]  # Sample of game IDs for registration


@dataclass
class ImportResult:
    """Result of importing orphaned databases."""
    databases_found: int = 0
    databases_imported: int = 0
    games_registered: int = 0
    errors: list[str] = None
    dry_run: bool = False

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def load_hosts_config() -> dict[str, dict]:
    """Load hosts configuration."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("hosts", {})


def find_orphaned_databases(
    data_dir: Path,
    node_id: str,
) -> list[OrphanedDatabase]:
    """Find databases not registered in the manifest.

    Args:
        data_dir: Directory to scan for databases
        node_id: Current node ID

    Returns:
        List of OrphanedDatabase objects
    """
    from app.distributed.cluster_manifest import get_cluster_manifest

    manifest = get_cluster_manifest()
    orphaned = []

    # Search for all .db files
    search_dirs = [data_dir]
    if data_dir.name == "games":
        # Also check selfplay and staging directories
        parent = data_dir.parent
        for subdir in ["selfplay", "staging"]:
            if (parent / subdir).exists():
                search_dirs.append(parent / subdir)

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for db_path in search_dir.rglob("*.db"):
            if not db_path.is_file() or db_path.stat().st_size == 0:
                continue

            # Skip manifest database
            if "manifest" in db_path.name.lower():
                continue

            # Check if this database's games are in the manifest
            try:
                is_orphaned, db_info = _check_database_orphaned(db_path, node_id, manifest)
                if is_orphaned and db_info:
                    orphaned.append(db_info)
            except Exception as e:
                logger.warning(f"Error checking {db_path}: {e}")

    return orphaned


def _check_database_orphaned(
    db_path: Path,
    node_id: str,
    manifest: Any,
) -> tuple[bool, OrphanedDatabase | None]:
    """Check if a database is orphaned (not in manifest).

    Returns:
        Tuple of (is_orphaned, OrphanedDatabase or None)
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        cursor = conn.cursor()

        # Get game count
        cursor.execute("SELECT COUNT(*) FROM games WHERE winner IS NOT NULL")
        game_count = cursor.fetchone()[0]

        if game_count == 0:
            conn.close()
            return False, None

        # Get board type and players
        cursor.execute("SELECT board_type, num_players FROM games LIMIT 1")
        row = cursor.fetchone()
        board_type = row[0] if row else None
        num_players = row[1] if row else None

        # Get sample of game IDs
        cursor.execute("SELECT game_id FROM games WHERE winner IS NOT NULL LIMIT 100")
        game_ids = [row[0] for row in cursor.fetchall()]

        conn.close()

        # Check manifest for these games
        registered_count = 0
        for game_id in game_ids[:10]:  # Check first 10 games
            locations = manifest.find_game(game_id)
            if any(loc.node_id == node_id for loc in locations):
                registered_count += 1

        # If less than 50% of sampled games are registered, consider orphaned
        if registered_count < len(game_ids[:10]) * 0.5:
            return True, OrphanedDatabase(
                path=str(db_path),
                size_bytes=db_path.stat().st_size,
                game_count=game_count,
                board_type=board_type,
                num_players=num_players,
                game_ids=game_ids,
            )

        return False, None

    except sqlite3.Error as e:
        logger.debug(f"SQLite error reading {db_path}: {e}")
        return False, None


def import_orphaned_database(
    db: OrphanedDatabase,
    node_id: str,
    dry_run: bool = False,
) -> tuple[int, str | None]:
    """Import a single orphaned database into the manifest.

    Args:
        db: OrphanedDatabase to import
        node_id: Node ID to register as source
        dry_run: If True, don't actually register

    Returns:
        Tuple of (games_registered, error_message or None)
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would import {db.path} ({db.game_count:,} games)")
        return db.game_count, None

    try:
        from app.distributed.cluster_manifest import get_cluster_manifest

        manifest = get_cluster_manifest()

        # Read all game IDs from the database
        conn = sqlite3.connect(f"file:{db.path}?mode=ro", uri=True, timeout=30.0)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT game_id, board_type, num_players FROM games WHERE winner IS NOT NULL"
        )
        games = cursor.fetchall()
        conn.close()

        # Register games in batches
        batch = []
        for game_id, board_type, num_players in games:
            batch.append((game_id, node_id, db.path))

            if len(batch) >= 1000:
                manifest.register_games_batch(
                    batch,
                    board_type=board_type or db.board_type,
                    num_players=num_players or db.num_players,
                )
                batch = []

        # Register remaining
        if batch:
            manifest.register_games_batch(
                batch,
                board_type=db.board_type,
                num_players=db.num_players,
            )

        logger.info(f"Imported {len(games):,} games from {Path(db.path).name}")
        return len(games), None

    except Exception as e:
        error_msg = f"Failed to import {db.path}: {e}"
        logger.error(error_msg)
        return 0, error_msg


def import_from_remote(
    host_name: str,
    dry_run: bool = False,
) -> ImportResult:
    """Import orphaned databases from a remote host.

    Args:
        host_name: Host name from distributed_hosts.yaml
        dry_run: If True, report only

    Returns:
        ImportResult
    """
    hosts_config = load_hosts_config()
    host_config = hosts_config.get(host_name)

    if not host_config:
        logger.error(f"Unknown host: {host_name}")
        return ImportResult(errors=[f"Unknown host: {host_name}"])

    ssh_host = host_config.get("tailscale_ip") or host_config.get("ssh_host")
    ssh_user = host_config.get("ssh_user", "ubuntu")
    ssh_key = os.path.expanduser(host_config.get("ssh_key", "~/.ssh/id_cluster"))
    ssh_port = host_config.get("ssh_port", 22)
    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")

    # Build remote script
    dry_run_flag = "--dry-run" if dry_run else ""
    remote_cmd = f"""
cd {ringrift_path} && \
PYTHONPATH=. python3 scripts/import_orphaned_databases.py {dry_run_flag} --json
"""

    cmd = [
        "ssh",
        "-i", ssh_key,
        "-p", str(ssh_port),
        "-o", "ConnectTimeout=30",
        "-o", "StrictHostKeyChecking=no",
        f"{ssh_user}@{ssh_host}",
        remote_cmd,
    ]

    logger.info(f"Running import on remote host {host_name}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"Remote import failed: {result.stderr}")
            return ImportResult(errors=[result.stderr])

        # Parse JSON output
        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                data = json.loads(line)
                return ImportResult(
                    databases_found=data.get("databases_found", 0),
                    databases_imported=data.get("databases_imported", 0),
                    games_registered=data.get("games_registered", 0),
                    errors=data.get("errors", []),
                    dry_run=dry_run,
                )

        logger.warning("No JSON output from remote")
        return ImportResult()

    except subprocess.TimeoutExpired:
        logger.error("Remote import timed out")
        return ImportResult(errors=["Timeout"])
    except Exception as e:
        logger.error(f"Remote import error: {e}")
        return ImportResult(errors=[str(e)])


def trigger_sync() -> bool:
    """Trigger data sync after import.

    Returns:
        True if sync was triggered successfully
    """
    try:
        import asyncio
        from app.coordination.auto_sync_daemon import get_auto_sync_daemon

        async def _trigger():
            daemon = get_auto_sync_daemon()
            if daemon.is_running():
                await daemon.trigger_sync()
                return True
            else:
                # Start daemon if not running
                await daemon.start()
                await daemon.trigger_sync()
                return True

        return asyncio.run(_trigger())

    except ImportError:
        logger.warning("AutoSyncDaemon not available")
        return False
    except Exception as e:
        logger.error(f"Failed to trigger sync: {e}")
        return False


def run_local_import(
    data_dir: Path | None = None,
    dry_run: bool = False,
    trigger_sync_after: bool = False,
) -> ImportResult:
    """Run import on local node.

    Args:
        data_dir: Directory to scan
        dry_run: If True, report only
        trigger_sync_after: If True, trigger sync after import

    Returns:
        ImportResult
    """
    result = ImportResult(dry_run=dry_run)

    # Determine data directory
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data" / "games"

    node_id = socket.gethostname()

    logger.info(f"Scanning for orphaned databases in {data_dir}")

    # Find orphaned databases
    orphaned = find_orphaned_databases(data_dir, node_id)
    result.databases_found = len(orphaned)

    if not orphaned:
        logger.info("No orphaned databases found")
        return result

    logger.info(f"Found {len(orphaned)} orphaned databases")

    # Import each database
    for db in orphaned:
        games_registered, error = import_orphaned_database(db, node_id, dry_run)

        if error:
            result.errors.append(error)
        else:
            result.databases_imported += 1
            result.games_registered += games_registered

    # Summary
    logger.info(
        f"Import complete: {result.databases_imported}/{result.databases_found} databases, "
        f"{result.games_registered:,} games registered"
    )

    # Trigger sync if requested
    if trigger_sync_after and not dry_run and result.databases_imported > 0:
        logger.info("Triggering data sync...")
        if trigger_sync():
            logger.info("Sync triggered successfully")
        else:
            result.errors.append("Failed to trigger sync")

    return result


def main():
    parser = argparse.ArgumentParser(description="Import orphaned databases into ClusterManifest")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory to scan for databases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only, don't import",
    )
    parser.add_argument(
        "--trigger-sync",
        action="store_true",
        help="Trigger data sync after import",
    )
    parser.add_argument(
        "--remote",
        type=str,
        help="Run on remote host (from distributed_hosts.yaml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Run import
    if args.remote:
        result = import_from_remote(args.remote, args.dry_run)
    else:
        data_dir = Path(args.data_dir) if args.data_dir else None
        result = run_local_import(data_dir, args.dry_run, args.trigger_sync)

    # Output results
    if args.json:
        output = {
            "databases_found": result.databases_found,
            "databases_imported": result.databases_imported,
            "games_registered": result.games_registered,
            "errors": result.errors,
            "dry_run": result.dry_run,
        }
        print(json.dumps(output))
    else:
        print("\n" + "=" * 50)
        print("IMPORT RESULTS")
        print("=" * 50)
        print(f"Mode: {'DRY RUN' if result.dry_run else 'LIVE'}")
        print(f"Databases found:    {result.databases_found}")
        print(f"Databases imported: {result.databases_imported}")
        print(f"Games registered:   {result.games_registered:,}")
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for error in result.errors[:5]:
                print(f"  - {error}")

    # Exit code
    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
