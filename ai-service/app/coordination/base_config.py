"""Base configuration class for coordination daemons.

Sprint 17.2 (January 4, 2026): Consolidates common configuration patterns
across daemon config dataclasses. Provides type-safe environment variable
loading and common fields shared by most daemons.

Usage:
    from app.coordination.base_config import BaseCoordinationConfig

    @dataclass
    class MyDaemonConfig(BaseCoordinationConfig):
        # Override env prefix for this daemon
        _env_prefix: ClassVar[str] = "RINGRIFT_MY_DAEMON"

        # Add daemon-specific fields
        max_retries: int = 3
        timeout_seconds: float = 30.0

        @classmethod
        def from_env(cls) -> "MyDaemonConfig":
            return cls(
                enabled=cls._get_env_bool("ENABLED", True),
                check_interval_seconds=cls._get_env_float("CHECK_INTERVAL", 60.0),
                max_retries=cls._get_env_int("MAX_RETRIES", 3),
                timeout_seconds=cls._get_env_float("TIMEOUT", 30.0),
            )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

T = TypeVar("T", bound="BaseCoordinationConfig")


@dataclass
class BaseCoordinationConfig:
    """Base configuration for coordination daemons.

    Provides:
    - Common fields used by most daemons (enabled, check_interval_seconds)
    - Type-safe environment variable getters
    - Consistent patterns for config loading

    Subclasses should:
    1. Override `_env_prefix` for their specific env var namespace
    2. Add daemon-specific fields as dataclass fields
    3. Implement `from_env()` classmethod using the helper methods

    Example:
        @dataclass
        class S3BackupConfig(BaseCoordinationConfig):
            _env_prefix: ClassVar[str] = "RINGRIFT_S3_BACKUP"

            bucket_name: str = "ringrift-backups"
            retention_days: int = 30

            @classmethod
            def from_env(cls) -> "S3BackupConfig":
                return cls(
                    enabled=cls._get_env_bool("ENABLED", True),
                    check_interval_seconds=cls._get_env_float("INTERVAL", 3600.0),
                    bucket_name=cls._get_env_str("BUCKET", "ringrift-backups"),
                    retention_days=cls._get_env_int("RETENTION_DAYS", 30),
                )
    """

    # Default environment variable prefix (override in subclasses)
    _env_prefix: ClassVar[str] = "RINGRIFT"

    # Whether the daemon is enabled
    enabled: bool = True

    # Main cycle interval in seconds
    check_interval_seconds: float = 60.0

    # Startup delay in seconds (staggered startup to avoid thundering herd)
    startup_delay_seconds: float = 0.0

    # Optional human-readable description
    description: str = ""

    # -------------------------------------------------------------------------
    # Environment Variable Helpers
    # -------------------------------------------------------------------------

    @classmethod
    def _make_env_key(cls, suffix: str) -> str:
        """Create full environment variable name from suffix.

        Args:
            suffix: The variable suffix (e.g., "ENABLED", "INTERVAL")

        Returns:
            Full env var name (e.g., "RINGRIFT_MY_DAEMON_ENABLED")
        """
        return f"{cls._env_prefix}_{suffix}"

    @classmethod
    def _get_env_bool(cls, suffix: str, default: bool) -> bool:
        """Get boolean from environment variable.

        Recognizes: "true", "1", "yes", "on" as True (case-insensitive)
        All other values (including "false", "0", "no", "off") → False

        Args:
            suffix: Env var suffix (full name = _env_prefix + "_" + suffix)
            default: Default value if env var not set

        Returns:
            Boolean value
        """
        key = cls._make_env_key(suffix)
        value = os.environ.get(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    @classmethod
    def _get_env_int(cls, suffix: str, default: int) -> int:
        """Get integer from environment variable.

        Args:
            suffix: Env var suffix
            default: Default value if env var not set or invalid

        Returns:
            Integer value
        """
        key = cls._make_env_key(suffix)
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def _get_env_float(cls, suffix: str, default: float) -> float:
        """Get float from environment variable.

        Args:
            suffix: Env var suffix
            default: Default value if env var not set or invalid

        Returns:
            Float value
        """
        key = cls._make_env_key(suffix)
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def _get_env_str(cls, suffix: str, default: str) -> str:
        """Get string from environment variable.

        Args:
            suffix: Env var suffix
            default: Default value if env var not set

        Returns:
            String value (stripped of whitespace)
        """
        key = cls._make_env_key(suffix)
        value = os.environ.get(key)
        if value is None:
            return default
        return value.strip()

    @classmethod
    def _get_env_list(
        cls,
        suffix: str,
        default: list[str] | None = None,
        separator: str = ",",
    ) -> list[str]:
        """Get list of strings from environment variable.

        Args:
            suffix: Env var suffix
            default: Default value if env var not set
            separator: Separator character (default: comma)

        Returns:
            List of strings (each item stripped of whitespace)
        """
        key = cls._make_env_key(suffix)
        value = os.environ.get(key)
        if value is None:
            return default if default is not None else []
        return [item.strip() for item in value.split(separator) if item.strip()]

    # -------------------------------------------------------------------------
    # Factory Method (override in subclasses)
    # -------------------------------------------------------------------------

    @classmethod
    def from_env(cls: type[T]) -> T:
        """Create config from environment variables.

        Subclasses should override this to load their specific fields.
        Default implementation loads only base fields.

        Returns:
            Config instance
        """
        return cls(
            enabled=cls._get_env_bool("ENABLED", True),
            check_interval_seconds=cls._get_env_float("CHECK_INTERVAL", 60.0),
            startup_delay_seconds=cls._get_env_float("STARTUP_DELAY", 0.0),
        )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Check if daemon is enabled.

        Convenience method that can be extended in subclasses
        to add additional enable conditions.
        """
        return self.enabled

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        from dataclasses import asdict, fields

        result = {}
        for f in fields(self):
            if not f.name.startswith("_"):
                result[f.name] = getattr(self, f.name)
        return result


# =============================================================================
# Pre-built Config Templates
# =============================================================================


@dataclass
class SyncDaemonConfig(BaseCoordinationConfig):
    """Base config for sync-related daemons."""

    _env_prefix: ClassVar[str] = "RINGRIFT_SYNC"

    # Sync-specific fields
    sync_timeout_seconds: float = 300.0
    max_concurrent_syncs: int = 1  # Feb 2026: 3 → 1 to prevent OOM
    retry_count: int = 3
    retry_delay_seconds: float = 5.0


@dataclass
class MonitorDaemonConfig(BaseCoordinationConfig):
    """Base config for monitoring daemons."""

    _env_prefix: ClassVar[str] = "RINGRIFT_MONITOR"

    # Monitor-specific fields
    health_check_interval_seconds: float = 30.0
    alert_threshold_count: int = 3
    alert_cooldown_seconds: float = 300.0


@dataclass
class RecoveryDaemonConfig(BaseCoordinationConfig):
    """Base config for recovery daemons."""

    _env_prefix: ClassVar[str] = "RINGRIFT_RECOVERY"

    # Recovery-specific fields
    grace_period_seconds: float = 30.0
    max_recovery_attempts: int = 3
    recovery_cooldown_seconds: float = 60.0
    escalation_enabled: bool = True
