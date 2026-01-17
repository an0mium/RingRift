#!/usr/bin/env python3
"""Pull Elo databases from cluster coordinator to production web server.

This script is designed to run on the production web server (ringrift.ai)
to sync Elo data from the training cluster coordinator (mac-studio).

The production server is NOT part of the P2P cluster, so it needs to
pull data via HTTP from the coordinator's P2P endpoints.

Usage:
    python scripts/production_elo_sync.py

    # With custom coordinator URL
    RINGRIFT_COORDINATOR_URL=http://100.107.168.125:8770 python scripts/production_elo_sync.py

Cron (every 5 minutes):
    */5 * * * * cd /home/ubuntu/ringrift/ai-service && /usr/bin/python3 scripts/production_elo_sync.py >> /home/ubuntu/ringrift/ai-service/logs/elo-sync.log 2>&1

Environment Variables:
    RINGRIFT_COORDINATOR_URL: Coordinator P2P URL (default: http://100.107.168.125:8770)
    RINGRIFT_ELO_SYNC_TIMEOUT: Request timeout in seconds (default: 30)

Databases Synced:
    - unified_elo.db: Master Elo ratings (~1-2 MB)
    - elo_progress.db: Progress snapshots over time (~40 KB)
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Default coordinator URL (mac-studio Tailscale IP)
DEFAULT_COORDINATOR_URL = "http://100.107.168.125:8770"

# Sync timeout in seconds
DEFAULT_TIMEOUT = 30


def get_coordinator_url() -> str:
    """Get the coordinator URL from environment or default."""
    return os.environ.get("RINGRIFT_COORDINATOR_URL", DEFAULT_COORDINATOR_URL)


def get_timeout() -> int:
    """Get the sync timeout from environment or default."""
    try:
        return int(os.environ.get("RINGRIFT_ELO_SYNC_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def atomic_write(path: Path, content: bytes) -> None:
    """Atomically write content to a file using temp file + rename pattern.

    This prevents corruption if the write is interrupted.
    Sets file permissions to 644 for readability by web server.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".db"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Set readable permissions before rename (tempfile creates 600 by default)
    tmp_path.chmod(0o644)
    tmp_path.rename(path)


async def pull_database(
    session,
    coordinator_url: str,
    endpoint: str,
    local_path: Path,
    db_name: str,
) -> bool:
    """Pull a single database from the coordinator.

    Args:
        session: aiohttp ClientSession
        coordinator_url: Base URL of the coordinator
        endpoint: API endpoint for the database
        local_path: Local path to save the database
        db_name: Human-readable database name for logging

    Returns:
        True if sync succeeded, False otherwise
    """
    url = f"{coordinator_url}{endpoint}"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                content = await resp.read()
                if content:
                    atomic_write(local_path, content)
                    print(f"[{datetime.now().isoformat()}] Synced {db_name}: {len(content):,} bytes")
                    return True
                else:
                    print(f"[{datetime.now().isoformat()}] WARNING: {db_name} returned empty content")
                    return False
            elif resp.status == 404:
                print(f"[{datetime.now().isoformat()}] INFO: {db_name} not found on coordinator (404)")
                return False
            else:
                print(f"[{datetime.now().isoformat()}] ERROR: Failed to sync {db_name}: HTTP {resp.status}")
                return False
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ERROR: Failed to sync {db_name}: {e}")
        return False


async def pull_elo_databases() -> tuple[bool, bool]:
    """Pull both Elo databases from the cluster coordinator.

    Returns:
        Tuple of (unified_elo_success, elo_progress_success)
    """
    try:
        import aiohttp
    except ImportError:
        print(f"[{datetime.now().isoformat()}] ERROR: aiohttp not installed. Run: pip install aiohttp")
        return False, False

    coordinator_url = get_coordinator_url()
    timeout = get_timeout()
    local_data_dir = Path("data")

    print(f"[{datetime.now().isoformat()}] Starting Elo sync from {coordinator_url}")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        # Pull unified_elo.db
        unified_success = await pull_database(
            session,
            coordinator_url,
            "/elo/sync/db",
            local_data_dir / "unified_elo.db",
            "unified_elo.db",
        )

        # Pull elo_progress.db
        progress_success = await pull_database(
            session,
            coordinator_url,
            "/elo/progress/db",
            local_data_dir / "elo_progress.db",
            "elo_progress.db",
        )

    # Summary
    if unified_success and progress_success:
        print(f"[{datetime.now().isoformat()}] Sync completed successfully")
    elif unified_success or progress_success:
        print(f"[{datetime.now().isoformat()}] Sync partially completed")
    else:
        print(f"[{datetime.now().isoformat()}] Sync failed")

    return unified_success, progress_success


def main() -> int:
    """Main entry point."""
    unified_success, progress_success = asyncio.run(pull_elo_databases())

    # Return exit code based on success
    if unified_success:
        return 0  # Success (unified_elo.db is critical)
    else:
        return 1  # Failure


if __name__ == "__main__":
    sys.exit(main())
