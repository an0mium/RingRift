"""Disaster Recovery Manager - Enable full cluster restoration from backups.

January 2026: Created as part of unified data synchronization plan.
Provides APIs for restoring data from S3 or OWC in disaster scenarios.

Key capabilities:
1. Restore all game databases from S3 bucket
2. Restore all game databases from OWC external drive
3. Verify backup completeness across both destinations
4. Support selective restoration by config key
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.distributed.cluster_manifest import (
    ClusterManifest,
    DataSource,
    get_cluster_manifest,
)

logger = logging.getLogger(__name__)


class RestoreStatus(str, Enum):
    """Status of a restore operation."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some files restored, some failed
    FAILED = "failed"
    NO_DATA = "no_data"  # No backup data found


@dataclass
class RestoredFile:
    """Information about a restored file."""

    config_key: str
    source_path: str  # S3 key or OWC path
    local_path: Path
    game_count: int
    file_size_mb: float
    restore_time_seconds: float
    success: bool
    error: str | None = None


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    status: RestoreStatus
    source: DataSource
    target_dir: Path
    files_restored: list[RestoredFile] = field(default_factory=list)
    files_failed: list[RestoredFile] = field(default_factory=list)
    total_games_restored: int = 0
    total_size_mb: float = 0.0
    total_time_seconds: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status in (RestoreStatus.SUCCESS, RestoreStatus.PARTIAL)


@dataclass
class BackupVerificationResult:
    """Result of backup completeness verification."""

    timestamp: float
    s3_verified: bool
    owc_verified: bool
    s3_configs: list[str]  # Configs with S3 backup
    owc_configs: list[str]  # Configs with OWC backup
    s3_only_configs: list[str]  # Only in S3, not OWC
    owc_only_configs: list[str]  # Only in OWC, not S3
    both_configs: list[str]  # Backed up to both
    missing_configs: list[str]  # Not backed up anywhere
    s3_total_games: int
    owc_total_games: int
    recommendation: str

    @property
    def fully_backed_up(self) -> bool:
        """True if all configs have at least one backup."""
        return len(self.missing_configs) == 0

    @property
    def redundant(self) -> bool:
        """True if all configs are backed up to both S3 and OWC."""
        return len(self.s3_only_configs) == 0 and len(self.owc_only_configs) == 0


@dataclass
class RecoveryConfig:
    """Configuration for DisasterRecoveryManager."""

    # S3 settings
    s3_bucket: str = "ringrift-models-20251214"
    s3_region: str = "us-east-1"
    s3_prefix: str = "databases/"

    # OWC settings
    owc_host: str = "mac-studio"
    owc_base_path: str = "/Volumes/RingRift-Data"
    owc_db_subpath: str = "databases"

    # Timeouts
    download_timeout: float = 600.0  # 10 minutes per file
    verify_timeout: float = 300.0  # 5 minutes for verification

    # Defaults
    target_dir: Path = field(default_factory=lambda: Path("data/games/restored"))


class DisasterRecoveryManager:
    """Manage disaster recovery scenarios.

    Provides APIs for:
    1. Full cluster restoration from S3
    2. Full cluster restoration from OWC
    3. Selective restoration by config
    4. Backup completeness verification
    """

    def __init__(self, config: RecoveryConfig | None = None):
        self.config = config or RecoveryConfig()
        self._manifest = get_cluster_manifest()

    async def restore_from_s3(
        self,
        target_dir: Path | None = None,
        config_keys: list[str] | None = None,
    ) -> RestoreResult:
        """Restore data from S3 bucket.

        Downloads all game databases from S3 to the target directory.
        If config_keys is specified, only restores those configs.

        Args:
            target_dir: Directory to restore to (default: config value)
            config_keys: Specific configs to restore (None = all)

        Returns:
            RestoreResult with restoration status
        """
        target = target_dir or self.config.target_dir
        target.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        files_restored: list[RestoredFile] = []
        files_failed: list[RestoredFile] = []

        # Get all S3 locations from manifest
        s3_locations = await self._get_s3_locations(config_keys)

        if not s3_locations:
            return RestoreResult(
                status=RestoreStatus.NO_DATA,
                source=DataSource.S3,
                target_dir=target,
                error="No S3 backup data found",
            )

        for loc in s3_locations:
            config_key = loc.get("config_key", "")
            s3_key = loc.get("s3_key", "")
            game_count = loc.get("game_count", 0)
            s3_bucket = loc.get("s3_bucket", self.config.s3_bucket)

            if not s3_key:
                continue

            file_start = time.time()
            local_path = target / f"{config_key}_s3_restored.db"

            try:
                success = await self._download_from_s3(
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                    local_path=local_path,
                )
                file_time = time.time() - file_start

                if success and local_path.exists():
                    size_mb = local_path.stat().st_size / (1024 * 1024)
                    files_restored.append(
                        RestoredFile(
                            config_key=config_key,
                            source_path=f"s3://{s3_bucket}/{s3_key}",
                            local_path=local_path,
                            game_count=game_count,
                            file_size_mb=size_mb,
                            restore_time_seconds=file_time,
                            success=True,
                        )
                    )
                else:
                    files_failed.append(
                        RestoredFile(
                            config_key=config_key,
                            source_path=f"s3://{s3_bucket}/{s3_key}",
                            local_path=local_path,
                            game_count=game_count,
                            file_size_mb=0,
                            restore_time_seconds=file_time,
                            success=False,
                            error="Download failed",
                        )
                    )
            except Exception as e:
                file_time = time.time() - file_start
                files_failed.append(
                    RestoredFile(
                        config_key=config_key,
                        source_path=f"s3://{s3_bucket}/{s3_key}",
                        local_path=local_path,
                        game_count=game_count,
                        file_size_mb=0,
                        restore_time_seconds=file_time,
                        success=False,
                        error=str(e),
                    )
                )

        total_time = time.time() - start_time
        total_games = sum(f.game_count for f in files_restored)
        total_size = sum(f.file_size_mb for f in files_restored)

        if len(files_restored) == 0:
            status = RestoreStatus.FAILED
        elif len(files_failed) == 0:
            status = RestoreStatus.SUCCESS
        else:
            status = RestoreStatus.PARTIAL

        return RestoreResult(
            status=status,
            source=DataSource.S3,
            target_dir=target,
            files_restored=files_restored,
            files_failed=files_failed,
            total_games_restored=total_games,
            total_size_mb=total_size,
            total_time_seconds=total_time,
        )

    async def restore_from_owc(
        self,
        target_dir: Path | None = None,
        config_keys: list[str] | None = None,
        owc_host: str | None = None,
    ) -> RestoreResult:
        """Restore data from OWC external drive.

        Downloads all game databases from OWC to the target directory.
        If config_keys is specified, only restores those configs.

        Args:
            target_dir: Directory to restore to (default: config value)
            config_keys: Specific configs to restore (None = all)
            owc_host: OWC host override (default: config value)

        Returns:
            RestoreResult with restoration status
        """
        target = target_dir or self.config.target_dir
        target.mkdir(parents=True, exist_ok=True)
        host = owc_host or self.config.owc_host

        start_time = time.time()
        files_restored: list[RestoredFile] = []
        files_failed: list[RestoredFile] = []

        # Get all OWC locations from manifest
        owc_locations = await self._get_owc_locations(config_keys)

        if not owc_locations:
            return RestoreResult(
                status=RestoreStatus.NO_DATA,
                source=DataSource.OWC,
                target_dir=target,
                error="No OWC backup data found",
            )

        for loc in owc_locations:
            config_key = loc.get("config_key", "")
            owc_path = loc.get("owc_path", "")
            game_count = loc.get("game_count", 0)
            loc_host = loc.get("owc_host", host)

            if not owc_path:
                continue

            file_start = time.time()
            local_path = target / f"{config_key}_owc_restored.db"

            try:
                success = await self._download_from_owc(
                    owc_host=loc_host,
                    owc_path=owc_path,
                    local_path=local_path,
                )
                file_time = time.time() - file_start

                if success and local_path.exists():
                    size_mb = local_path.stat().st_size / (1024 * 1024)
                    files_restored.append(
                        RestoredFile(
                            config_key=config_key,
                            source_path=f"{loc_host}:{owc_path}",
                            local_path=local_path,
                            game_count=game_count,
                            file_size_mb=size_mb,
                            restore_time_seconds=file_time,
                            success=True,
                        )
                    )
                else:
                    files_failed.append(
                        RestoredFile(
                            config_key=config_key,
                            source_path=f"{loc_host}:{owc_path}",
                            local_path=local_path,
                            game_count=game_count,
                            file_size_mb=0,
                            restore_time_seconds=file_time,
                            success=False,
                            error="Download failed",
                        )
                    )
            except Exception as e:
                file_time = time.time() - file_start
                files_failed.append(
                    RestoredFile(
                        config_key=config_key,
                        source_path=f"{loc_host}:{owc_path}",
                        local_path=local_path,
                        game_count=game_count,
                        file_size_mb=0,
                        restore_time_seconds=file_time,
                        success=False,
                        error=str(e),
                    )
                )

        total_time = time.time() - start_time
        total_games = sum(f.game_count for f in files_restored)
        total_size = sum(f.file_size_mb for f in files_restored)

        if len(files_restored) == 0:
            status = RestoreStatus.FAILED
        elif len(files_failed) == 0:
            status = RestoreStatus.SUCCESS
        else:
            status = RestoreStatus.PARTIAL

        return RestoreResult(
            status=status,
            source=DataSource.OWC,
            target_dir=target,
            files_restored=files_restored,
            files_failed=files_failed,
            total_games_restored=total_games,
            total_size_mb=total_size,
            total_time_seconds=total_time,
        )

    async def verify_backup_completeness(self) -> BackupVerificationResult:
        """Verify both S3 and OWC have complete backups.

        Checks that all known configs have backups in both S3 and OWC.

        Returns:
            BackupVerificationResult with verification details
        """
        # Get all S3 and OWC locations
        s3_locations = await self._get_s3_locations()
        owc_locations = await self._get_owc_locations()

        s3_configs = {loc.get("config_key", "") for loc in s3_locations if loc.get("config_key")}
        owc_configs = {loc.get("config_key", "") for loc in owc_locations if loc.get("config_key")}

        all_configs = s3_configs | owc_configs

        # Categorize configs
        both = s3_configs & owc_configs
        s3_only = s3_configs - owc_configs
        owc_only = owc_configs - s3_configs

        # Get known configs that aren't backed up anywhere
        # Check if manifest has method to get all known configs
        known_configs = set()
        if hasattr(self._manifest, "get_all_configs"):
            known_configs = set(self._manifest.get_all_configs())

        missing = known_configs - all_configs if known_configs else set()

        # Calculate totals
        s3_total_games = sum(
            loc.get("game_count", 0) for loc in s3_locations
        )
        owc_total_games = sum(
            loc.get("game_count", 0) for loc in owc_locations
        )

        # Generate recommendation
        if not missing and not s3_only and not owc_only:
            recommendation = "All configs are fully backed up to both S3 and OWC. No action needed."
        elif missing:
            recommendation = f"CRITICAL: {len(missing)} configs have no backup. Run backup immediately."
        elif s3_only or owc_only:
            recommendation = (
                f"WARNING: {len(s3_only)} configs only in S3, {len(owc_only)} only in OWC. "
                "Run backup to ensure redundancy."
            )
        else:
            recommendation = "Unknown state. Manual verification recommended."

        return BackupVerificationResult(
            timestamp=time.time(),
            s3_verified=len(s3_configs) > 0,
            owc_verified=len(owc_configs) > 0,
            s3_configs=sorted(s3_configs),
            owc_configs=sorted(owc_configs),
            s3_only_configs=sorted(s3_only),
            owc_only_configs=sorted(owc_only),
            both_configs=sorted(both),
            missing_configs=sorted(missing),
            s3_total_games=s3_total_games,
            owc_total_games=owc_total_games,
            recommendation=recommendation,
        )

    async def _get_s3_locations(
        self, config_keys: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Get S3 backup locations from manifest."""
        locations = []

        # If specific configs requested
        if config_keys:
            for config_key in config_keys:
                sources = self._manifest.find_across_all_sources(config_key)
                locations.extend(sources.get(DataSource.S3, []))
        else:
            # Get all external storage locations
            external = self._manifest.find_external_storage_for_config("")
            locations = [
                loc for loc in external
                if loc.get("source") == DataSource.S3.value
            ]

        return locations

    async def _get_owc_locations(
        self, config_keys: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Get OWC backup locations from manifest."""
        locations = []

        # If specific configs requested
        if config_keys:
            for config_key in config_keys:
                sources = self._manifest.find_across_all_sources(config_key)
                locations.extend(sources.get(DataSource.OWC, []))
        else:
            # Get all external storage locations
            external = self._manifest.find_external_storage_for_config("")
            locations = [
                loc for loc in external
                if loc.get("source") == DataSource.OWC.value
            ]

        return locations

    async def _download_from_s3(
        self,
        s3_bucket: str,
        s3_key: str,
        local_path: Path,
    ) -> bool:
        """Download a file from S3."""
        s3_uri = f"s3://{s3_bucket}/{s3_key}"

        def _do_download() -> bool:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_uri, str(local_path)],
                capture_output=True,
                timeout=int(self.config.download_timeout),
            )
            return result.returncode == 0

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_do_download),
                timeout=self.config.download_timeout,
            )
        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            logger.warning(f"[DisasterRecovery] S3 download timeout: {s3_uri}")
            return False

    async def _download_from_owc(
        self,
        owc_host: str,
        owc_path: str,
        local_path: Path,
    ) -> bool:
        """Download a file from OWC drive via rsync."""

        def _do_download() -> bool:
            result = subprocess.run(
                ["rsync", "-az", f"{owc_host}:{owc_path}", str(local_path)],
                capture_output=True,
                timeout=int(self.config.download_timeout),
            )
            return result.returncode == 0

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_do_download),
                timeout=self.config.download_timeout,
            )
        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            logger.warning(f"[DisasterRecovery] OWC download timeout: {owc_host}:{owc_path}")
            return False


# Singleton instance
_recovery_manager: DisasterRecoveryManager | None = None


def get_disaster_recovery_manager() -> DisasterRecoveryManager:
    """Get the singleton DisasterRecoveryManager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = DisasterRecoveryManager()
    return _recovery_manager


def reset_disaster_recovery_manager() -> None:
    """Reset the singleton instance (for testing)."""
    global _recovery_manager
    _recovery_manager = None
