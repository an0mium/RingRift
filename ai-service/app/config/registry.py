"""Configuration Registry - Unified access to all configuration values.

This module provides a single source of truth for runtime configuration
by consolidating scattered constants into a typed, validated registry.

Instead of importing from multiple sources:
    from app.config.thresholds import ELO_DROP_ROLLBACK
    from app.config.env import env
    from app.config.coordination_defaults import SyncDefaults

Use the unified registry:
    from app.config.registry import config_registry, ConfigCategory

    # Get a specific value
    elo_drop = config_registry.get(ConfigCategory.THRESHOLDS, "ELO_DROP_ROLLBACK")

    # Get all thresholds as a dict
    thresholds = config_registry.get_category(ConfigCategory.THRESHOLDS)

    # Get a typed value with validation
    value = config_registry.get_typed("ELO_DROP_ROLLBACK", int, default=50)

The registry consolidates values from:
- app.config.thresholds (training, promotion, rollback thresholds)
- app.config.env (environment variables)
- app.config.coordination_defaults (daemon/sync timing intervals)
- app.config.constants (general constants)

December 2025: Created for Phase 3 configuration consolidation.
See plan at ~/.claude/plans/wiggly-orbiting-raccoon.md for context.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, overload

logger = logging.getLogger(__name__)


class ConfigCategory(str, Enum):
    """Categories of configuration values."""

    # Threshold values (training, promotion, rollback)
    THRESHOLDS = "thresholds"

    # Timing intervals (sync, health check, cooldowns)
    INTERVALS = "intervals"

    # Resource limits (max concurrent, queue sizes)
    LIMITS = "limits"

    # Quality thresholds (data quality, model quality)
    QUALITY = "quality"

    # Elo targets and ratings
    ELO = "elo"

    # Daemon configuration
    DAEMONS = "daemons"

    # Cluster configuration
    CLUSTER = "cluster"


T = TypeVar("T")


@dataclass
class ConfigValue:
    """Metadata about a configuration value."""

    key: str
    value: Any
    category: ConfigCategory
    source: str  # Module/file where value is defined
    description: str = ""
    deprecated: bool = False
    deprecated_replacement: str | None = None


class ConfigRegistry:
    """Single source of truth for all runtime configuration.

    Provides unified access to configuration values from multiple sources
    with type validation and deprecation tracking.

    Usage:
        registry = get_config_registry()

        # Get a single value
        value = registry.get(ConfigCategory.THRESHOLDS, "TRAINING_TRIGGER_GAMES")

        # Get with type checking
        value = registry.get_typed("ELO_DROP_ROLLBACK", int, default=50)

        # Get all values in a category
        thresholds = registry.get_category(ConfigCategory.THRESHOLDS)

        # List all keys in a category
        keys = registry.list_keys(ConfigCategory.THRESHOLDS)
    """

    def __init__(self) -> None:
        self._values: dict[str, ConfigValue] = {}
        self._categories: dict[ConfigCategory, set[str]] = {
            cat: set() for cat in ConfigCategory
        }
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Load configuration values lazily on first access."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return
            self._load_all_configs()
            self._initialized = True

    def _load_all_configs(self) -> None:
        """Load configuration from all sources."""
        self._load_thresholds()
        self._load_intervals()
        self._load_limits()
        self._load_quality()
        self._load_elo()
        self._load_daemons()

        logger.debug(
            f"[ConfigRegistry] Loaded {len(self._values)} configuration values "
            f"across {len(ConfigCategory)} categories"
        )

    def _register(
        self,
        key: str,
        value: Any,
        category: ConfigCategory,
        source: str,
        description: str = "",
        deprecated: bool = False,
        deprecated_replacement: str | None = None,
    ) -> None:
        """Register a configuration value."""
        self._values[key] = ConfigValue(
            key=key,
            value=value,
            category=category,
            source=source,
            description=description,
            deprecated=deprecated,
            deprecated_replacement=deprecated_replacement,
        )
        self._categories[category].add(key)

    def _load_thresholds(self) -> None:
        """Load threshold values from thresholds.py."""
        try:
            from app.config import thresholds as t

            # Training thresholds
            self._register("TRAINING_TRIGGER_GAMES", t.TRAINING_TRIGGER_GAMES,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Games needed to trigger training per config")
            self._register("TRAINING_MIN_INTERVAL_SECONDS", t.TRAINING_MIN_INTERVAL_SECONDS,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Minimum interval between training runs")
            self._register("TRAINING_STALENESS_HOURS", t.TRAINING_STALENESS_HOURS,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Hours before config is considered stale")
            self._register("TRAINING_BOOTSTRAP_GAMES", t.TRAINING_BOOTSTRAP_GAMES,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Bootstrap threshold for new configs")
            self._register("TRAINING_MAX_CONCURRENT", t.TRAINING_MAX_CONCURRENT,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Maximum concurrent training jobs")

            # Rollback thresholds
            self._register("ELO_DROP_ROLLBACK", t.ELO_DROP_ROLLBACK,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Elo drop that triggers rollback consideration")
            self._register("WIN_RATE_DROP_ROLLBACK", t.WIN_RATE_DROP_ROLLBACK,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Win rate drop percentage that triggers rollback")
            self._register("ERROR_RATE_ROLLBACK", t.ERROR_RATE_ROLLBACK,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Error rate threshold for rollback")
            self._register("MIN_GAMES_REGRESSION", t.MIN_GAMES_REGRESSION,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Minimum games for reliable regression detection")
            self._register("CONSECUTIVE_REGRESSIONS_FORCE", t.CONSECUTIVE_REGRESSIONS_FORCE,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Consecutive regressions before forced rollback")

            # Promotion thresholds
            self._register("ELO_IMPROVEMENT_PROMOTE", t.ELO_IMPROVEMENT_PROMOTE,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Elo improvement required for promotion")
            self._register("MIN_GAMES_PROMOTE", t.MIN_GAMES_PROMOTE,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Minimum games before eligible for promotion")
            self._register("MIN_WIN_RATE_PROMOTE", t.MIN_WIN_RATE_PROMOTE,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Minimum win rate for promotion consideration")
            self._register("WIN_RATE_BEAT_BEST", t.WIN_RATE_BEAT_BEST,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Win rate required to beat current best")

            # Production thresholds
            self._register("PRODUCTION_ELO_THRESHOLD", t.PRODUCTION_ELO_THRESHOLD,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Minimum ELO for production promotion")
            self._register("PRODUCTION_MIN_GAMES", t.PRODUCTION_MIN_GAMES,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Minimum games before production promotion")
            self._register("PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC",
                          t.PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Minimum win rate vs heuristic for production")
            self._register("PRODUCTION_MIN_WIN_RATE_VS_RANDOM",
                          t.PRODUCTION_MIN_WIN_RATE_VS_RANDOM,
                          ConfigCategory.THRESHOLDS, "thresholds",
                          "Minimum win rate vs random for production")

        except ImportError as e:
            logger.warning(f"[ConfigRegistry] Failed to load thresholds: {e}")

    def _load_intervals(self) -> None:
        """Load timing interval values from coordination_defaults.py."""
        try:
            from app.config.coordination_defaults import (
                HealthDefaults,
                HeartbeatDefaults,
                RetryDefaults,
                SyncDefaults,
            )

            # Sync intervals
            self._register("DATA_SYNC_INTERVAL", SyncDefaults.DATA_SYNC_INTERVAL,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Default data sync interval (seconds)")
            self._register("MODEL_SYNC_INTERVAL", SyncDefaults.MODEL_SYNC_INTERVAL,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Model sync interval (seconds)")
            self._register("ELO_SYNC_INTERVAL", SyncDefaults.ELO_SYNC_INTERVAL,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Elo sync interval (seconds)")
            self._register("SYNC_LOCK_TIMEOUT", SyncDefaults.LOCK_TIMEOUT,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Sync lock timeout (seconds)")

            # Health check intervals
            self._register("HEALTH_SSH_TIMEOUT", HealthDefaults.SSH_TIMEOUT,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "SSH timeout for health checks (seconds)")
            self._register("HEALTH_CACHE_TTL", HealthDefaults.HEALTHY_CACHE_TTL,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Healthy result cache TTL (seconds)")

            # Heartbeat intervals
            self._register("HEARTBEAT_INTERVAL", HeartbeatDefaults.INTERVAL,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Heartbeat interval (seconds)")
            self._register("HEARTBEAT_TIMEOUT", HeartbeatDefaults.TIMEOUT,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Heartbeat timeout before marked dead (seconds)")

            # Retry intervals
            self._register("RETRY_BASE_DELAY", RetryDefaults.BASE_DELAY,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Base delay between retry attempts")
            self._register("RETRY_MAX_DELAY", RetryDefaults.MAX_DELAY,
                          ConfigCategory.INTERVALS, "coordination_defaults",
                          "Maximum delay between retry attempts")

        except ImportError as e:
            logger.warning(f"[ConfigRegistry] Failed to load intervals: {e}")

    def _load_limits(self) -> None:
        """Load resource limit values."""
        try:
            from app.config.coordination_defaults import (
                BandwidthDefaults,
                HealthDefaults,
                QueueDefaults,
                ResourceLimitsDefaults,
                SyncDefaults,
            )

            # Sync concurrency limits
            self._register("MAX_CONCURRENT_SYNCS_PER_HOST", SyncDefaults.MAX_CONCURRENT_PER_HOST,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Maximum concurrent sync operations per host")
            self._register("MAX_CONCURRENT_SYNCS_CLUSTER", SyncDefaults.MAX_CONCURRENT_CLUSTER,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Maximum concurrent sync operations cluster-wide")

            # Bandwidth limits
            self._register("DEFAULT_UPLOAD_MBPS", BandwidthDefaults.DEFAULT_UPLOAD_MBPS,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Default upload bandwidth limit (MB/s)")
            self._register("DEFAULT_DOWNLOAD_MBPS", BandwidthDefaults.DEFAULT_DOWNLOAD_MBPS,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Default download bandwidth limit (MB/s)")
            self._register("MAX_CONCURRENT_TRANSFERS", BandwidthDefaults.MAX_CONCURRENT_TRANSFERS,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Maximum concurrent transfers per host")

            # Health check limits
            self._register("MAX_CONCURRENT_HEALTH_CHECKS", HealthDefaults.MAX_CONCURRENT_CHECKS,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Maximum concurrent health checks")
            self._register("MIN_HEALTHY_HOSTS", HealthDefaults.MIN_HEALTHY_HOSTS,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Minimum required healthy hosts")

            # Queue limits
            self._register("TRAINING_DATA_SOFT_LIMIT", QueueDefaults.TRAINING_DATA_SOFT_LIMIT,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Training data queue soft limit")
            self._register("TRAINING_DATA_HARD_LIMIT", QueueDefaults.TRAINING_DATA_HARD_LIMIT,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Training data queue hard limit")
            self._register("PENDING_GAMES_SOFT_LIMIT", QueueDefaults.PENDING_GAMES_SOFT_LIMIT,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Pending games queue soft limit")
            self._register("PENDING_GAMES_HARD_LIMIT", QueueDefaults.PENDING_GAMES_HARD_LIMIT,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Pending games queue hard limit")

            # GPU tier limits
            self._register("CONSUMER_MAX_SELFPLAY", ResourceLimitsDefaults.CONSUMER_MAX,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Max selfplay for consumer GPU tier")
            self._register("DATACENTER_MAX_SELFPLAY", ResourceLimitsDefaults.DATACENTER_MAX,
                          ConfigCategory.LIMITS, "coordination_defaults",
                          "Max selfplay for datacenter GPU tier")

        except ImportError as e:
            logger.warning(f"[ConfigRegistry] Failed to load limits: {e}")

    def _load_quality(self) -> None:
        """Load quality threshold values."""
        try:
            from app.config.thresholds import (
                LOW_QUALITY_THRESHOLD,
                MEDIUM_QUALITY_THRESHOLD,
                QUALITY_EXCELLENT_THRESHOLD,
                QUALITY_GOOD_THRESHOLD,
            )

            self._register("QUALITY_EXCELLENT_THRESHOLD", QUALITY_EXCELLENT_THRESHOLD,
                          ConfigCategory.QUALITY, "thresholds",
                          "Quality score for 'excellent' rating")
            self._register("QUALITY_GOOD_THRESHOLD", QUALITY_GOOD_THRESHOLD,
                          ConfigCategory.QUALITY, "thresholds",
                          "Quality score for 'good' rating")
            self._register("MEDIUM_QUALITY_THRESHOLD", MEDIUM_QUALITY_THRESHOLD,
                          ConfigCategory.QUALITY, "thresholds",
                          "Quality score for 'medium' rating")
            self._register("LOW_QUALITY_THRESHOLD", LOW_QUALITY_THRESHOLD,
                          ConfigCategory.QUALITY, "thresholds",
                          "Quality score for 'low' rating")

        except ImportError as e:
            logger.warning(f"[ConfigRegistry] Failed to load quality thresholds: {e}")

    def _load_elo(self) -> None:
        """Load Elo rating configuration."""
        try:
            from app.config.thresholds import (
                ELO_TARGET_ALL_CONFIGS,
                ELO_TARGETS_BY_CONFIG,
                ELO_TIER_ADVANCED,
                ELO_TIER_EXPERT,
                ELO_TIER_GRANDMASTER,
                ELO_TIER_INTERMEDIATE,
                ELO_TIER_MASTER,
                ELO_TIER_NOVICE,
            )

            self._register("ELO_TARGET_ALL_CONFIGS", ELO_TARGET_ALL_CONFIGS,
                          ConfigCategory.ELO, "thresholds",
                          "Global Elo target for all configs")
            self._register("ELO_TARGETS_BY_CONFIG", ELO_TARGETS_BY_CONFIG,
                          ConfigCategory.ELO, "thresholds",
                          "Per-configuration Elo targets")
            self._register("ELO_TIER_NOVICE", ELO_TIER_NOVICE,
                          ConfigCategory.ELO, "thresholds",
                          "Elo tier: Novice (below heuristic)")
            self._register("ELO_TIER_INTERMEDIATE", ELO_TIER_INTERMEDIATE,
                          ConfigCategory.ELO, "thresholds",
                          "Elo tier: Intermediate (heuristic-level)")
            self._register("ELO_TIER_ADVANCED", ELO_TIER_ADVANCED,
                          ConfigCategory.ELO, "thresholds",
                          "Elo tier: Advanced (better than heuristic)")
            self._register("ELO_TIER_EXPERT", ELO_TIER_EXPERT,
                          ConfigCategory.ELO, "thresholds",
                          "Elo tier: Expert (production-ready)")
            self._register("ELO_TIER_MASTER", ELO_TIER_MASTER,
                          ConfigCategory.ELO, "thresholds",
                          "Elo tier: Master (strong model)")
            self._register("ELO_TIER_GRANDMASTER", ELO_TIER_GRANDMASTER,
                          ConfigCategory.ELO, "thresholds",
                          "Elo tier: Grandmaster (exceptional)")

        except ImportError as e:
            logger.warning(f"[ConfigRegistry] Failed to load Elo config: {e}")

    def _load_daemons(self) -> None:
        """Load daemon configuration."""
        try:
            from app.config.coordination_defaults import (
                RetryDefaults,
                TaskLifecycleDefaults,
            )

            self._register("DAEMON_MAX_RESTARTS", RetryDefaults.MAX_RETRIES,
                          ConfigCategory.DAEMONS, "coordination_defaults",
                          "Maximum restart attempts per daemon")
            self._register("DAEMON_RESTART_BACKOFF_MAX", RetryDefaults.MAX_DELAY,
                          ConfigCategory.DAEMONS, "coordination_defaults",
                          "Maximum backoff between restarts (seconds)")
            self._register("DAEMON_HEARTBEAT_TIMEOUT", TaskLifecycleDefaults.HEARTBEAT_TIMEOUT,
                          ConfigCategory.DAEMONS, "coordination_defaults",
                          "Heartbeat timeout for task orphan detection")
            self._register("DAEMON_ORPHAN_GRACE_PERIOD", TaskLifecycleDefaults.ORPHAN_GRACE_PERIOD,
                          ConfigCategory.DAEMONS, "coordination_defaults",
                          "Grace period for new tasks before orphan check")
            self._register("DAEMON_MAX_HISTORY", TaskLifecycleDefaults.MAX_HISTORY,
                          ConfigCategory.DAEMONS, "coordination_defaults",
                          "Maximum tasks to track in history")

        except ImportError as e:
            logger.warning(f"[ConfigRegistry] Failed to load daemon config: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get(self, category: ConfigCategory, key: str) -> Any:
        """Get a configuration value by category and key.

        Args:
            category: Configuration category
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found
        """
        self._ensure_initialized()

        if key not in self._values:
            raise KeyError(f"Configuration key not found: {key}")

        config = self._values[key]
        if config.category != category:
            raise KeyError(
                f"Key '{key}' is in category '{config.category.value}', "
                f"not '{category.value}'"
            )

        if config.deprecated:
            import warnings
            replacement = f" Use '{config.deprecated_replacement}' instead." \
                if config.deprecated_replacement else ""
            warnings.warn(
                f"Configuration '{key}' is deprecated.{replacement}",
                DeprecationWarning,
                stacklevel=2,
            )

        return config.value

    @overload
    def get_typed(self, key: str, expected_type: type[T]) -> T: ...

    @overload
    def get_typed(self, key: str, expected_type: type[T], default: T) -> T: ...

    def get_typed(
        self,
        key: str,
        expected_type: type[T],
        default: T | None = None,
    ) -> T:
        """Get a configuration value with type checking.

        Args:
            key: Configuration key
            expected_type: Expected type of the value
            default: Default value if key not found

        Returns:
            Configuration value of expected type

        Raises:
            KeyError: If key not found and no default provided
            TypeError: If value is not of expected type
        """
        self._ensure_initialized()

        if key not in self._values:
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key}")

        value = self._values[key].value

        if not isinstance(value, expected_type):
            raise TypeError(
                f"Configuration '{key}' is type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )

        return value

    def get_category(self, category: ConfigCategory) -> dict[str, Any]:
        """Get all values in a category.

        Args:
            category: Configuration category

        Returns:
            Dictionary of key -> value for all keys in category
        """
        self._ensure_initialized()

        return {
            key: self._values[key].value
            for key in self._categories[category]
        }

    def list_keys(self, category: ConfigCategory | None = None) -> list[str]:
        """List all configuration keys.

        Args:
            category: Optional category filter

        Returns:
            List of configuration keys
        """
        self._ensure_initialized()

        if category is not None:
            return sorted(self._categories[category])
        return sorted(self._values.keys())

    def get_info(self, key: str) -> ConfigValue | None:
        """Get full metadata for a configuration value.

        Args:
            key: Configuration key

        Returns:
            ConfigValue with full metadata, or None if not found
        """
        self._ensure_initialized()
        return self._values.get(key)

    def get_metrics(self) -> dict[str, Any]:
        """Get registry metrics for observability.

        Returns:
            Dictionary of metrics
        """
        self._ensure_initialized()

        return {
            "total_values": len(self._values),
            "categories": {
                cat.value: len(keys) for cat, keys in self._categories.items()
            },
            "sources": list({v.source for v in self._values.values()}),
            "deprecated_count": sum(1 for v in self._values.values() if v.deprecated),
        }


# =============================================================================
# Singleton
# =============================================================================

_registry: ConfigRegistry | None = None
_registry_lock = threading.Lock()


def get_config_registry() -> ConfigRegistry:
    """Get the singleton ConfigRegistry instance."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ConfigRegistry()
    return _registry


def reset_config_registry() -> None:
    """Reset the singleton (for testing)."""
    global _registry
    _registry = None


# Convenience alias
config_registry = get_config_registry


__all__ = [
    "ConfigCategory",
    "ConfigRegistry",
    "ConfigValue",
    "config_registry",
    "get_config_registry",
    "reset_config_registry",
]
