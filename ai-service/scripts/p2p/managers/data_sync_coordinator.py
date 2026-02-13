"""DataSyncCoordinator: External storage scanning and sync coordination.

January 2026: Extracted from p2p_orchestrator.py for better modularity.
Handles OWC drive and S3 bucket metadata collection for unified cluster
data visibility.

This coordinator complements SyncPlanner by providing:
- External storage metadata collection (OWC, S3)
- Config extraction from database paths
- Storage availability checking
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from scripts.p2p.db_helpers import p2p_db_connection

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DataSyncCoordinatorConfig:
    """Configuration for DataSyncCoordinator.

    Attributes:
        owc_paths: Possible OWC mount paths to scan
        s3_bucket: S3 bucket name for game data
        s3_enabled: Whether S3 scanning is enabled
        s3_cache_ttl: Cache TTL for S3 metadata in seconds
        mac_studio_host: Hostname/IP for mac-studio (OWC host)
    """

    owc_paths: list[str] = field(
        default_factory=lambda: ["/Volumes/RingRift-Data", "/Volumes/OWC"]
    )
    s3_bucket: str = "ringrift-models-20251214"
    s3_enabled: bool = True
    s3_cache_ttl: float = 3600.0  # 1 hour
    mac_studio_host: str = "mac-studio"


@dataclass
class ExternalStorageMetadata:
    """Metadata from external storage sources (OWC, S3).

    Jan 2026: Unified structure for external storage visibility.
    """

    # OWC metadata
    owc_available: bool = False
    owc_games_by_config: dict[str, int] = field(default_factory=dict)
    owc_total_games: int = 0
    owc_total_size_bytes: int = 0
    owc_last_scan: float = 0.0
    owc_scan_error: str | None = None

    # S3 metadata
    s3_available: bool = False
    s3_games_by_config: dict[str, int] = field(default_factory=dict)
    s3_total_games: int = 0
    s3_total_size_bytes: int = 0
    s3_bucket: str = ""
    s3_last_scan: float = 0.0
    s3_scan_error: str | None = None

    # Collection timestamp
    collected_at: float = field(default_factory=time.time)


@dataclass
class DataSyncCoordinatorStats:
    """Statistics for DataSyncCoordinator operations."""

    owc_scans: int = 0
    owc_scan_errors: int = 0
    s3_scans: int = 0
    s3_scan_errors: int = 0
    last_owc_scan: float = 0.0
    last_s3_scan: float = 0.0


# ============================================================================
# Singleton management
# ============================================================================

_instance: DataSyncCoordinator | None = None


def get_data_sync_coordinator() -> DataSyncCoordinator | None:
    """Get the singleton DataSyncCoordinator instance."""
    return _instance


def set_data_sync_coordinator(coordinator: DataSyncCoordinator) -> None:
    """Set the singleton DataSyncCoordinator instance."""
    global _instance
    _instance = coordinator


def reset_data_sync_coordinator() -> None:
    """Reset the singleton DataSyncCoordinator instance (for testing)."""
    global _instance
    _instance = None


def create_data_sync_coordinator(
    config: DataSyncCoordinatorConfig | None = None,
) -> DataSyncCoordinator:
    """Create and register a DataSyncCoordinator instance.

    Args:
        config: Optional configuration

    Returns:
        The created DataSyncCoordinator instance
    """
    coordinator = DataSyncCoordinator(config=config)
    set_data_sync_coordinator(coordinator)
    return coordinator


# ============================================================================
# DataSyncCoordinator
# ============================================================================


class DataSyncCoordinator:
    """Coordinator for external storage scanning and sync operations.

    This class handles:
    - OWC drive metadata collection (local and remote via SSH)
    - S3 bucket metadata collection
    - Config key extraction from database paths
    - Caching of S3 metadata to avoid excessive API calls
    """

    def __init__(
        self,
        config: DataSyncCoordinatorConfig | None = None,
    ) -> None:
        """Initialize DataSyncCoordinator.

        Args:
            config: Optional configuration
        """
        self.config = config or DataSyncCoordinatorConfig()
        self.stats = DataSyncCoordinatorStats()

        # S3 metadata cache
        self._s3_cache: dict | None = None
        self._s3_cache_time: float = 0.0

    # ========================================================================
    # External Storage Metadata Collection
    # ========================================================================

    async def collect_external_storage_metadata(self) -> ExternalStorageMetadata:
        """Collect metadata from external storage sources (OWC drive, S3 bucket).

        Jan 2026: Added for unified cluster data visibility.

        Returns:
            ExternalStorageMetadata with OWC and S3 metadata.
        """
        external = ExternalStorageMetadata(collected_at=time.time())

        # Collect OWC drive metadata (if accessible)
        try:
            owc_metadata = await self.scan_owc_metadata()
            if owc_metadata:
                external.owc_available = True
                external.owc_games_by_config = owc_metadata.get("games_by_config", {})
                external.owc_total_games = owc_metadata.get("total_games", 0)
                external.owc_total_size_bytes = owc_metadata.get("total_size_bytes", 0)
                external.owc_last_scan = time.time()
            else:
                external.owc_scan_error = "OWC scan returned no data"
        except Exception as e:
            external.owc_scan_error = str(e)
            self.stats.owc_scan_errors += 1
            logger.warning(f"[ExternalStorage] OWC scan failed: {e}")

        # Collect S3 bucket metadata (if configured)
        try:
            s3_metadata = await self.scan_s3_metadata()
            if s3_metadata:
                external.s3_available = True
                external.s3_games_by_config = s3_metadata.get("games_by_config", {})
                external.s3_total_games = s3_metadata.get("total_games", 0)
                external.s3_total_size_bytes = s3_metadata.get("total_size_bytes", 0)
                external.s3_bucket = s3_metadata.get("bucket", "")
                external.s3_last_scan = time.time()
            else:
                external.s3_scan_error = "S3 scan returned no data (check boto3 and credentials)"
        except Exception as e:
            external.s3_scan_error = str(e)
            self.stats.s3_scan_errors += 1
            logger.warning(f"[ExternalStorage] S3 scan failed: {e}")

        return external

    # ========================================================================
    # OWC Drive Scanning
    # ========================================================================

    async def scan_owc_metadata(self) -> dict | None:
        """Scan OWC external drive for game data metadata.

        Jan 2026: OWC drive is mounted on mac-studio at /Volumes/RingRift-Data.

        Returns:
            Dict with games_by_config, total_games, total_size_bytes, or None if unavailable.
        """
        self.stats.owc_scans += 1
        self.stats.last_owc_scan = time.time()

        # Load OWC paths from config if available, otherwise use defaults
        owc_paths = list(self.config.owc_paths)

        try:
            from app.config.cluster_config import load_cluster_config

            config = load_cluster_config()
            sync_cfg = config.get_raw_section("sync_routing")
            for storage in sync_cfg.get("allowed_external_storage", []):
                if storage.get("host") == "mac-studio":
                    path = storage.get("path")
                    if path and path not in owc_paths:
                        owc_paths.insert(0, path)
        except Exception:
            pass

        # Check if running on mac-studio (OWC is local)
        hostname = socket.gethostname().lower()
        is_mac_studio = "mac-studio" in hostname or hostname == "mac-studio"

        if is_mac_studio:
            # Direct local access
            for owc_path in owc_paths:
                if os.path.exists(owc_path):
                    return await asyncio.to_thread(self._scan_owc_local, owc_path)
            return None

        # Remote access via SSH to mac-studio (get IP from config)
        mac_studio_host = self.config.mac_studio_host
        try:
            from app.config.cluster_config import load_cluster_config

            config = load_cluster_config()
            mac_studio_cfg = config.hosts_raw.get("mac-studio", {})
            # Prefer Tailscale IP for reliability
            mac_studio_host = (
                mac_studio_cfg.get("tailscale_ip")
                or mac_studio_cfg.get("ssh_host")
                or self.config.mac_studio_host
            )
        except Exception:
            pass  # Fall back to config default

        try:
            return await self._scan_owc_remote(mac_studio_host, owc_paths[0])
        except Exception as e:
            logger.warning(f"[OWC] Remote scan failed to {mac_studio_host}: {e}")
            return None

    def _scan_owc_local(self, base_path: str) -> dict:
        """Scan OWC drive locally for game databases.

        Returns dict with games_by_config, total_games, total_size_bytes.
        """
        games_by_config: dict[str, int] = {}
        total_games = 0
        total_size_bytes = 0

        # Look for game databases in standard locations
        data_paths = [
            Path(base_path) / "data" / "games",
            Path(base_path) / "games",
            Path(base_path) / "selfplay",
        ]

        for data_path in data_paths:
            if not data_path.exists():
                continue

            for db_file in data_path.glob("**/*.db"):
                try:
                    total_size_bytes += db_file.stat().st_size

                    # Extract config from filename or query DB
                    config_key = self.extract_config_from_path(db_file)
                    if not config_key:
                        continue

                    # Quick game count query
                    with p2p_db_connection(db_file, timeout=5.0) as conn:
                        try:
                            cursor = conn.execute(
                                "SELECT COUNT(*) FROM games WHERE status = 'completed'"
                            )
                            count = cursor.fetchone()[0]
                            games_by_config[config_key] = (
                                games_by_config.get(config_key, 0) + count
                            )
                            total_games += count
                        except sqlite3.OperationalError:
                            # Table doesn't exist or different schema
                            pass
                except Exception:
                    continue

        return {
            "games_by_config": games_by_config,
            "total_games": total_games,
            "total_size_bytes": total_size_bytes,
        }

    async def _scan_owc_remote(self, host: str, owc_path: str) -> dict | None:
        """Scan OWC drive via SSH to mac-studio.

        Returns dict with games_by_config, total_games, total_size_bytes.
        """
        # Use a simple SSH command to get file listing and sizes
        ssh_cmd = f"""
        cd {owc_path} 2>/dev/null && find . -name "*.db" -type f -exec stat -f '%z %N' {{}} \\; 2>/dev/null | head -100
        """

        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                host,
                "bash",
                "-c",
                ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)

            if proc.returncode != 0:
                return None

            # Parse output: "size path"
            games_by_config: dict[str, int] = {}
            total_size_bytes = 0
            total_games = 0

            for line in stdout.decode().strip().split("\n"):
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                size_str, path = parts
                try:
                    total_size_bytes += int(size_str)
                except ValueError:
                    continue

                # Extract config from path
                config_key = self.extract_config_from_path(Path(path))
                if config_key:
                    # Estimate 100 games per DB file as placeholder
                    # (accurate count would require opening each DB)
                    games_by_config[config_key] = games_by_config.get(config_key, 0) + 100
                    total_games += 100

            return {
                "games_by_config": games_by_config,
                "total_games": total_games,
                "total_size_bytes": total_size_bytes,
            }
        except (asyncio.TimeoutError, OSError):
            return None

    # ========================================================================
    # S3 Bucket Scanning
    # ========================================================================

    def get_s3_bucket_from_config(self) -> str | None:
        """Get S3 bucket name from config or environment.

        Priority:
        1. RINGRIFT_S3_BUCKET environment variable (backward compat)
        2. sync_routing.s3.bucket from distributed_hosts.yaml
        3. Default from config

        Returns:
            S3 bucket name or None if S3 is disabled.
        """
        # Priority 1: Environment variable
        s3_bucket = os.environ.get("RINGRIFT_S3_BUCKET")
        if s3_bucket:
            return s3_bucket

        # Priority 2: Load from YAML config
        try:
            from app.config.cluster_config import load_cluster_config

            config = load_cluster_config()
            s3_cfg = config.get_raw_section("sync_routing").get("s3", {})
            if not s3_cfg.get("enabled", True):
                logger.info("[S3] S3 disabled in config")
                return None
            bucket = s3_cfg.get("bucket")
            if bucket:
                return bucket
        except Exception as e:
            logger.debug(f"[S3] Could not load config from YAML: {e}")

        # Priority 3: Default bucket from config
        if self.config.s3_enabled:
            return self.config.s3_bucket
        return None

    async def scan_s3_metadata(self) -> dict | None:
        """Scan S3 bucket for game data metadata.

        Jan 2026: Uses boto3 if available, with caching to avoid repeated API calls.

        Returns:
            Dict with games_by_config, total_games, total_size_bytes, bucket, or None.
        """
        self.stats.s3_scans += 1
        self.stats.last_s3_scan = time.time()

        # Get S3 bucket from config (env var or YAML)
        s3_bucket = self.get_s3_bucket_from_config()
        if not s3_bucket:
            logger.info("[S3] S3 scanning disabled - no bucket configured")
            return None

        # Check for cached result
        if self._s3_cache and (time.time() - self._s3_cache_time) < self.config.s3_cache_ttl:
            return self._s3_cache

        try:
            import boto3
        except ImportError:
            logger.warning("[S3] boto3 not installed. Install with: pip install boto3")
            return None

        try:
            s3 = boto3.client("s3")

            games_by_config: dict[str, int] = {}
            total_games = 0
            total_size_bytes = 0

            # List objects in bucket (limit to games/ prefix)
            paginator = s3.get_paginator("list_objects_v2")
            for prefix in ["data/games/", "games/", "selfplay/"]:
                try:
                    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
                        for obj in page.get("Contents", []):
                            key = obj["Key"]
                            size = obj["Size"]

                            if not key.endswith(".db"):
                                continue

                            total_size_bytes += size

                            # Extract config from key
                            config_key = self.extract_config_from_path(Path(key))
                            if config_key:
                                # Estimate based on file size (rough: 1 game = 10KB)
                                est_games = max(1, size // 10000)
                                games_by_config[config_key] = (
                                    games_by_config.get(config_key, 0) + est_games
                                )
                                total_games += est_games
                except Exception:
                    continue

            result = {
                "games_by_config": games_by_config,
                "total_games": total_games,
                "total_size_bytes": total_size_bytes,
                "bucket": s3_bucket,
            }

            # Cache result
            self._s3_cache = result
            self._s3_cache_time = time.time()

            return result

        except Exception as e:
            logger.warning(f"[S3] Scan failed: {e}")
            return None

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def extract_config_from_path(self, db_path: Path) -> str | None:
        """Extract board_type and num_players from database path.

        Expected patterns:
        - canonical_hex8_2p.db
        - hex8_2p_selfplay.db
        - games_square8_4p.db

        Args:
            db_path: Path to database file

        Returns:
            Config key like "hex8_2p" or None if pattern doesn't match
        """
        filename = db_path.stem.lower()

        # Try common patterns
        patterns = [
            r"(hex8|hexagonal|square8|square19)_(\d)p",
            r"canonical_(hex8|hexagonal|square8|square19)_(\d)p",
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                board_type = match.group(1)
                num_players = match.group(2)
                return f"{board_type}_{num_players}p"

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get current coordinator statistics.

        Returns:
            Dictionary with scan counts and timestamps
        """
        return {
            "owc_scans": self.stats.owc_scans,
            "owc_scan_errors": self.stats.owc_scan_errors,
            "s3_scans": self.stats.s3_scans,
            "s3_scan_errors": self.stats.s3_scan_errors,
            "last_owc_scan": self.stats.last_owc_scan,
            "last_s3_scan": self.stats.last_s3_scan,
            "s3_cache_valid": (
                self._s3_cache is not None
                and (time.time() - self._s3_cache_time) < self.config.s3_cache_ttl
            ),
        }
