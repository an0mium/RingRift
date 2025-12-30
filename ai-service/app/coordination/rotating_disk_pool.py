"""Rotating Disk Pool Manager.

Manages a rotating disk pool (10% of free space by default) for data sync
facilitation. This enables coordinator nodes to temporarily accept synced
data without filling up disk space.

Dec 30, 2025: Phase 2.2 of Distributed Data Pipeline Architecture.

Design:
- 10% of free disk space is allocated as a "rotating pool"
- Data in the pool is cleaned up after configurable age (default 24h)
- Priority cleanup when disk pressure is detected
- Pool files are marked for rotation via metadata

Usage:
    from app.coordination.rotating_disk_pool import (
        get_rotating_pool_manager,
        RotatingPoolConfig,
    )

    manager = get_rotating_pool_manager()

    # Check if pool can accept new data
    if manager.can_accept_data(1024 * 1024 * 100):  # 100MB
        manager.mark_data_for_rotation(Path("data/games/synced/new_file.db"))

    # Cleanup old data
    cleaned = await manager.cleanup_oldest(target_free_bytes=1024**3)  # 1GB
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Default paths relative to ai-service root
DEFAULT_POOL_SUBDIRS = [
    "data/games/synced",
    "data/games/p2p_cache",
    "data/training/synced",
]

# Metadata database for pool tracking
POOL_METADATA_DB = "data/rotating_pool_metadata.db"


@dataclass
class RotatingPoolConfig:
    """Configuration for rotating disk pool.

    Attributes:
        quota_percent: Percentage of free space to allocate (default 10%)
        min_quota_gb: Minimum quota in GB (default 5GB)
        max_quota_gb: Maximum quota in GB (default 100GB)
        cleanup_age_hours: Remove data older than this (default 24h)
        priority_cleanup_on_pressure: Enable aggressive cleanup on disk pressure
        pool_subdirs: Subdirectories managed as part of the pool
    """

    quota_percent: float = 10.0
    min_quota_gb: float = 5.0
    max_quota_gb: float = 100.0
    cleanup_age_hours: int = 24
    priority_cleanup_on_pressure: bool = True
    pool_subdirs: list[str] = field(default_factory=lambda: DEFAULT_POOL_SUBDIRS.copy())

    @classmethod
    def from_yaml_config(cls, config: dict[str, Any]) -> RotatingPoolConfig:
        """Create config from YAML dictionary.

        Args:
            config: Dictionary from distributed_hosts.yaml disk_management.rotating_pool

        Returns:
            RotatingPoolConfig instance
        """
        return cls(
            quota_percent=config.get("quota_percent", 10.0),
            min_quota_gb=config.get("min_quota_gb", 5.0),
            max_quota_gb=config.get("max_quota_gb", 100.0),
            cleanup_age_hours=config.get("cleanup_age_hours", 24),
            priority_cleanup_on_pressure=config.get("priority_cleanup_on_pressure", True),
            pool_subdirs=config.get("pool_subdirs", DEFAULT_POOL_SUBDIRS.copy()),
        )


@dataclass
class PoolStats:
    """Statistics about the rotating pool."""

    quota_bytes: int
    used_bytes: int
    free_bytes: int
    file_count: int
    oldest_file_age_hours: float | None
    disk_free_bytes: int
    disk_total_bytes: int


class RotatingDiskPoolManager:
    """Manages rotating disk pool for data sync facilitation.

    The pool is a portion of free disk space (10% by default) that can be
    used for temporary data storage during sync operations. Files in the
    pool are automatically cleaned up after a configurable age.

    Thread-safe: Uses SQLite for metadata tracking.
    """

    _instance: RotatingDiskPoolManager | None = None

    def __init__(
        self,
        root_path: Path | str | None = None,
        config: RotatingPoolConfig | None = None,
    ):
        """Initialize the pool manager.

        Args:
            root_path: Root path to ai-service directory
            config: Pool configuration (loads from YAML if None)
        """
        if root_path is None:
            root_path = Path(__file__).parent.parent.parent
        self.root_path = Path(root_path)

        # Load config from YAML if not provided
        if config is None:
            config = self._load_config_from_yaml()
        self.config = config

        # Initialize metadata database
        self._db_path = self.root_path / POOL_METADATA_DB
        self._init_database()

    @classmethod
    def get_instance(cls) -> RotatingDiskPoolManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _load_config_from_yaml(self) -> RotatingPoolConfig:
        """Load configuration from distributed_hosts.yaml."""
        try:
            import yaml

            config_path = self.root_path / "config" / "distributed_hosts.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    full_config = yaml.safe_load(f) or {}
                disk_config = full_config.get("disk_management", {})
                pool_config = disk_config.get("rotating_pool", {})
                if pool_config.get("enabled", True):
                    return RotatingPoolConfig.from_yaml_config(pool_config)
        except Exception as e:
            logger.warning(f"[RotatingPool] Failed to load YAML config: {e}")

        return RotatingPoolConfig()

    def _init_database(self) -> None:
        """Initialize the metadata database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pool_files (
                    path TEXT PRIMARY KEY,
                    size_bytes INTEGER NOT NULL,
                    added_at REAL NOT NULL,
                    source TEXT,
                    config_key TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pool_files_added
                ON pool_files(added_at)
            """)
            conn.commit()

    def get_pool_quota_bytes(self) -> int:
        """Calculate current pool quota based on free disk space.

        Returns:
            Quota in bytes (10% of free space, clamped to min/max)
        """
        try:
            disk_usage = shutil.disk_usage(self.root_path)
            free_bytes = disk_usage.free

            # Calculate quota as percentage of free space
            quota = int(free_bytes * (self.config.quota_percent / 100.0))

            # Clamp to min/max
            min_bytes = int(self.config.min_quota_gb * 1024**3)
            max_bytes = int(self.config.max_quota_gb * 1024**3)

            return max(min_bytes, min(quota, max_bytes))
        except OSError as e:
            logger.warning(f"[RotatingPool] Failed to get disk usage: {e}")
            return int(self.config.min_quota_gb * 1024**3)

    def get_pool_usage_bytes(self) -> int:
        """Get current usage of rotating pool.

        Returns:
            Total bytes used by tracked pool files
        """
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM pool_files")
            return cursor.fetchone()[0]

    def can_accept_data(self, size_bytes: int) -> bool:
        """Check if pool has room for new data.

        Args:
            size_bytes: Size of data to accept

        Returns:
            True if pool has room, False otherwise
        """
        quota = self.get_pool_quota_bytes()
        usage = self.get_pool_usage_bytes()
        return (usage + size_bytes) <= quota

    def get_pool_stats(self) -> PoolStats:
        """Get comprehensive pool statistics.

        Returns:
            PoolStats with quota, usage, and file info
        """
        try:
            disk_usage = shutil.disk_usage(self.root_path)
        except OSError:
            disk_usage = type("obj", (object,), {"free": 0, "total": 0})()

        quota = self.get_pool_quota_bytes()
        usage = self.get_pool_usage_bytes()

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pool_files")
            file_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT MIN(added_at) FROM pool_files")
            oldest_time = cursor.fetchone()[0]

        oldest_age = None
        if oldest_time is not None:
            oldest_age = (time.time() - oldest_time) / 3600.0

        return PoolStats(
            quota_bytes=quota,
            used_bytes=usage,
            free_bytes=max(0, quota - usage),
            file_count=file_count,
            oldest_file_age_hours=oldest_age,
            disk_free_bytes=disk_usage.free,
            disk_total_bytes=disk_usage.total,
        )

    def mark_data_for_rotation(
        self,
        path: Path,
        source: str | None = None,
        config_key: str | None = None,
    ) -> bool:
        """Mark a data file as part of rotating pool.

        Args:
            path: Path to the data file
            source: Source of the data (e.g., node ID)
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            True if marked successfully, False otherwise
        """
        try:
            if not path.exists():
                logger.warning(f"[RotatingPool] File not found: {path}")
                return False

            size_bytes = path.stat().st_size
            path_str = str(path.absolute())

            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pool_files
                    (path, size_bytes, added_at, source, config_key)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (path_str, size_bytes, time.time(), source, config_key),
                )
                conn.commit()

            logger.debug(f"[RotatingPool] Marked for rotation: {path} ({size_bytes} bytes)")
            return True
        except Exception as e:
            logger.warning(f"[RotatingPool] Failed to mark file: {e}")
            return False

    async def cleanup_oldest(self, target_free_bytes: int = 0) -> int:
        """Remove oldest data until target free space achieved.

        Args:
            target_free_bytes: Target free space in pool (0 = cleanup expired only)

        Returns:
            Number of bytes freed
        """
        import asyncio

        return await asyncio.to_thread(self._cleanup_oldest_sync, target_free_bytes)

    def _cleanup_oldest_sync(self, target_free_bytes: int = 0) -> int:
        """Synchronous implementation of cleanup_oldest."""
        bytes_freed = 0
        files_removed = 0
        now = time.time()
        max_age_seconds = self.config.cleanup_age_hours * 3600

        with sqlite3.connect(self._db_path) as conn:
            # First pass: remove expired files
            cursor = conn.execute(
                """
                SELECT path, size_bytes FROM pool_files
                WHERE (? - added_at) > ?
                ORDER BY added_at ASC
                """,
                (now, max_age_seconds),
            )

            for row in cursor.fetchall():
                path_str, size = row
                if self._remove_file(Path(path_str)):
                    conn.execute("DELETE FROM pool_files WHERE path = ?", (path_str,))
                    bytes_freed += size
                    files_removed += 1

            # Second pass: if target not met, remove more (oldest first)
            if target_free_bytes > 0:
                current_usage = self.get_pool_usage_bytes()
                target_usage = max(0, self.get_pool_quota_bytes() - target_free_bytes)

                while current_usage > target_usage:
                    cursor = conn.execute(
                        """
                        SELECT path, size_bytes FROM pool_files
                        ORDER BY added_at ASC
                        LIMIT 1
                        """
                    )
                    row = cursor.fetchone()
                    if row is None:
                        break

                    path_str, size = row
                    if self._remove_file(Path(path_str)):
                        conn.execute("DELETE FROM pool_files WHERE path = ?", (path_str,))
                        bytes_freed += size
                        files_removed += 1
                        current_usage -= size
                    else:
                        # Remove from tracking if file doesn't exist
                        conn.execute("DELETE FROM pool_files WHERE path = ?", (path_str,))
                        current_usage -= size

            conn.commit()

        if bytes_freed > 0:
            logger.info(
                f"[RotatingPool] Cleaned up {files_removed} files, "
                f"freed {bytes_freed / 1024**2:.1f} MB"
            )

        return bytes_freed

    def _remove_file(self, path: Path) -> bool:
        """Remove a file from disk.

        Args:
            path: Path to remove

        Returns:
            True if removed successfully, False otherwise
        """
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            return True
        except OSError as e:
            logger.warning(f"[RotatingPool] Failed to remove {path}: {e}")
            return False

    def cleanup_missing_entries(self) -> int:
        """Remove database entries for files that no longer exist.

        Returns:
            Number of entries removed
        """
        removed = 0
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT path FROM pool_files")
            for (path_str,) in cursor.fetchall():
                if not Path(path_str).exists():
                    conn.execute("DELETE FROM pool_files WHERE path = ?", (path_str,))
                    removed += 1
            conn.commit()

        if removed > 0:
            logger.info(f"[RotatingPool] Removed {removed} stale database entries")
        return removed

    def is_in_pool_subdir(self, path: Path) -> bool:
        """Check if a path is within one of the pool subdirectories.

        Args:
            path: Path to check

        Returns:
            True if path is in a pool subdir, False otherwise
        """
        try:
            path_str = str(path.absolute())
            for subdir in self.config.pool_subdirs:
                pool_dir = str((self.root_path / subdir).absolute())
                if path_str.startswith(pool_dir):
                    return True
            return False
        except (ValueError, OSError):
            return False


# ============================================================================
# Singleton accessor
# ============================================================================


def get_rotating_pool_manager() -> RotatingDiskPoolManager:
    """Get the singleton RotatingDiskPoolManager instance.

    Returns:
        The singleton instance
    """
    return RotatingDiskPoolManager.get_instance()
