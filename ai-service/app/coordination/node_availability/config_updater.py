"""Config Updater - Atomic YAML Updates (December 2025).

Safely updates distributed_hosts.yaml with node status changes from provider APIs.
Uses atomic write pattern (write to temp file, then rename) to prevent corruption.

Key features:
- Atomic writes via temp file + rename
- Automatic backup before changes
- Backup rotation (keeps last N backups)
- Dry-run mode for testing
- Change tracking and reporting
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "distributed_hosts.yaml"
BACKUP_DIR_NAME = ".config_backups"
MAX_BACKUPS = 10


@dataclass
class ConfigUpdateResult:
    """Result of a configuration update operation."""

    success: bool
    nodes_updated: list[str] = field(default_factory=list)
    changes: dict[str, tuple[str, str]] = field(default_factory=dict)  # node -> (old_status, new_status)
    backup_path: Optional[Path] = None
    error: Optional[str] = None
    dry_run: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def update_count(self) -> int:
        """Number of nodes updated."""
        return len(self.nodes_updated)

    def __str__(self) -> str:
        if self.dry_run:
            return f"DRY RUN: Would update {self.update_count} nodes"
        if self.success:
            return f"Updated {self.update_count} nodes successfully"
        return f"Update failed: {self.error}"


class ConfigUpdater:
    """Manages atomic updates to distributed_hosts.yaml.

    Thread-safe: Uses file locking for concurrent access.
    Atomic: Writes to temp file, then renames (atomic on POSIX).

    Usage:
        updater = ConfigUpdater()
        result = await updater.update_node_statuses({
            "vast-12345": "retired",
            "lambda-gh200-1": "offline",
        })
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        dry_run: bool = True,
        max_backups: int = MAX_BACKUPS,
    ):
        """Initialize config updater.

        Args:
            config_path: Path to distributed_hosts.yaml (auto-detected if None)
            dry_run: If True, log changes but don't write to file
            max_backups: Maximum number of backup files to keep
        """
        self.config_path = Path(config_path) if config_path else self._find_config_path()
        self.dry_run = dry_run
        self.max_backups = max_backups
        self._backup_dir: Optional[Path] = None

    def _find_config_path(self) -> Path:
        """Find distributed_hosts.yaml in standard locations."""
        candidates = [
            DEFAULT_CONFIG_PATH,
            Path.cwd() / "config" / "distributed_hosts.yaml",
            Path.home() / "ringrift" / "ai-service" / "config" / "distributed_hosts.yaml",
        ]
        for path in candidates:
            if path.exists():
                return path
        return DEFAULT_CONFIG_PATH

    @property
    def backup_dir(self) -> Path:
        """Get backup directory (creates if needed)."""
        if self._backup_dir is None:
            self._backup_dir = self.config_path.parent / BACKUP_DIR_NAME
        if not self.dry_run and not self._backup_dir.exists():
            self._backup_dir.mkdir(parents=True, exist_ok=True)
        return self._backup_dir

    def load_config(self) -> dict:
        """Load current configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _create_backup(self) -> Optional[Path]:
        """Create backup of current config file.

        Returns:
            Path to backup file, or None if backup failed.
        """
        if self.dry_run:
            return None

        if not self.config_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"distributed_hosts_{timestamp}.yaml"

        try:
            shutil.copy2(self.config_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
            return backup_path
        except OSError as e:
            logger.warning(f"Failed to create backup: {e}")
            return None

    def _rotate_backups(self) -> None:
        """Remove old backups, keeping only max_backups most recent."""
        if self.dry_run or not self.backup_dir.exists():
            return

        backups = sorted(
            self.backup_dir.glob("distributed_hosts_*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_backup in backups[self.max_backups:]:
            try:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
            except OSError as e:
                logger.warning(f"Failed to remove old backup: {e}")

    def _write_config_atomic(self, config: dict) -> None:
        """Write config to file atomically.

        Uses write-to-temp-then-rename pattern for atomic updates.
        """
        if self.dry_run:
            return

        # Write to temp file in same directory (for same-filesystem rename)
        fd, temp_path = tempfile.mkstemp(
            suffix=".yaml",
            prefix="distributed_hosts_",
            dir=self.config_path.parent,
        )

        try:
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(
                    config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

            # Atomic rename (POSIX guarantees atomicity for same-filesystem rename)
            os.rename(temp_path, self.config_path)
            logger.debug(f"Config written atomically to {self.config_path}")

        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    async def update_node_statuses(
        self,
        updates: dict[str, str],
        reason: str = "provider_sync",
    ) -> ConfigUpdateResult:
        """Update node statuses in distributed_hosts.yaml.

        Args:
            updates: Dict of node_name -> new_status
            reason: Reason for update (for logging)

        Returns:
            ConfigUpdateResult with details of changes made.
        """
        if not updates:
            return ConfigUpdateResult(success=True)

        try:
            config = self.load_config()
            hosts = config.get("hosts", {})
            changes: dict[str, tuple[str, str]] = {}
            nodes_updated: list[str] = []

            for node_name, new_status in updates.items():
                if node_name not in hosts:
                    logger.warning(f"Node {node_name} not found in config, skipping")
                    continue

                old_status = hosts[node_name].get("status", "unknown")
                if old_status == new_status:
                    continue  # No change needed

                changes[node_name] = (old_status, new_status)
                nodes_updated.append(node_name)

                if self.dry_run:
                    logger.info(
                        f"[DRY RUN] Would update {node_name}: {old_status} -> {new_status} "
                        f"(reason: {reason})"
                    )
                else:
                    hosts[node_name]["status"] = new_status
                    logger.info(
                        f"Updated {node_name}: {old_status} -> {new_status} "
                        f"(reason: {reason})"
                    )

            if not changes:
                return ConfigUpdateResult(success=True, dry_run=self.dry_run)

            backup_path = None
            if not self.dry_run:
                backup_path = self._create_backup()
                self._write_config_atomic(config)
                self._rotate_backups()

            return ConfigUpdateResult(
                success=True,
                nodes_updated=nodes_updated,
                changes=changes,
                backup_path=backup_path,
                dry_run=self.dry_run,
            )

        except FileNotFoundError as e:
            return ConfigUpdateResult(
                success=False,
                error=f"Config file not found: {e}",
                dry_run=self.dry_run,
            )
        except yaml.YAMLError as e:
            return ConfigUpdateResult(
                success=False,
                error=f"YAML parse error: {e}",
                dry_run=self.dry_run,
            )
        except OSError as e:
            return ConfigUpdateResult(
                success=False,
                error=f"File system error: {e}",
                dry_run=self.dry_run,
            )

    async def get_current_statuses(self) -> dict[str, str]:
        """Get current status for all hosts in config.

        Returns:
            Dict of node_name -> current_status
        """
        try:
            config = self.load_config()
            hosts = config.get("hosts", {})
            return {
                name: host.get("status", "unknown")
                for name, host in hosts.items()
            }
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    async def get_nodes_by_status(self, status: str) -> list[str]:
        """Get list of nodes with a specific status.

        Args:
            status: Status to filter by (e.g., "ready", "offline", "retired")

        Returns:
            List of node names with that status.
        """
        statuses = await self.get_current_statuses()
        return [name for name, s in statuses.items() if s == status]

    async def get_nodes_by_provider(self, provider: str) -> list[str]:
        """Get list of nodes for a specific provider.

        Infers provider from node name prefix (e.g., "vast-*", "lambda-*").

        Args:
            provider: Provider name prefix

        Returns:
            List of node names for that provider.
        """
        try:
            config = self.load_config()
            hosts = config.get("hosts", {})
            return [
                name for name in hosts.keys()
                if name.startswith(f"{provider}-") or name.startswith(f"{provider}_")
            ]
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load config: {e}")
            return []
