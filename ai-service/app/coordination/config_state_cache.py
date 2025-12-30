"""Cache for config-level state data (quality scores, game counts, ELO history).

December 30, 2025: Extracted from selfplay_scheduler.py to:
1. Reduce SelfplayScheduler complexity (~4,200 LOC god object)
2. Enable independent testing of cache logic
3. Allow cache strategy changes without modifying scheduler

This class handles TTL-based caching to reduce database queries during
priority calculation. The cache is designed for read-heavy workloads
where quality scores change infrequently but are queried thousands of
times during allocation cycles.

Usage:
    from app.coordination.config_state_cache import ConfigStateCache

    cache = ConfigStateCache(ttl_seconds=30.0)

    # Single config lookup
    quality = cache.get_quality("hex8_2p")
    if quality is None:
        # Cache miss - fetch from daemon
        quality = daemon.get_config_quality("hex8_2p")
        cache.set_quality("hex8_2p", quality)

    # Invalidate on external changes
    cache.invalidate("hex8_2p")  # Single config
    cache.invalidate()  # All configs
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with value and timestamp.

    Attributes:
        value: The cached value
        timestamp: Unix timestamp when entry was created
        hits: Number of times this entry was accessed (for metrics)
    """
    value: float
    timestamp: float
    hits: int = 0


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring.

    Attributes:
        total_hits: Number of cache hits (value returned from cache)
        total_misses: Number of cache misses (value not in cache or expired)
        total_invalidations: Number of invalidation calls
        entries_count: Current number of entries in cache
    """
    total_hits: int = 0
    total_misses: int = 0
    total_invalidations: int = 0
    entries_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0-1.0)."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0


class ConfigStateCache:
    """Cache for config-level state data with TTL-based expiration.

    Provides TTL-based caching for quality scores and other config state.
    Designed for integration with SelfplayScheduler's priority calculation.

    Thread Safety:
        This class is NOT thread-safe. For concurrent access, use external
        synchronization or create per-thread instances.

    Attributes:
        ttl_seconds: Time-to-live for cache entries in seconds
        default_quality: Default quality score for cache misses
    """

    # Class-level constants
    DEFAULT_TTL = 30.0  # 30 second TTL
    DEFAULT_QUALITY = 0.7  # Default medium quality

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL,
        default_quality: float = DEFAULT_QUALITY,
        quality_provider: Optional[Callable[[str], Optional[float]]] = None,
    ):
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 30s)
            default_quality: Default quality score when lookup fails (default: 0.7)
            quality_provider: Optional callback to fetch quality on cache miss.
                              Signature: (config_key: str) -> Optional[float]
        """
        self._ttl_seconds = ttl_seconds
        self._default_quality = default_quality
        self._quality_provider = quality_provider

        # Cache storage: {config_key: CacheEntry}
        self._quality_cache: dict[str, CacheEntry] = {}

        # Statistics tracking
        self._stats = CacheStats()

        # Logging prefix for consistent log messages
        self._log_prefix = "[ConfigStateCache]"

    @property
    def ttl_seconds(self) -> float:
        """Get the TTL for cache entries."""
        return self._ttl_seconds

    @ttl_seconds.setter
    def ttl_seconds(self, value: float) -> None:
        """Set the TTL for cache entries (clears cache on change)."""
        if value != self._ttl_seconds:
            self._ttl_seconds = value
            self.invalidate()  # Clear cache when TTL changes

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.entries_count = len(self._quality_cache)
        return self._stats

    def get_quality(self, config_key: str) -> Optional[float]:
        """Get quality score from cache if available and not expired.

        This is a pure cache lookup. If the value is not in cache or expired,
        returns None. Use get_quality_or_fetch() for automatic fetching.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            Quality score (0.0-1.0) if cached and valid, None otherwise
        """
        now = time.time()
        entry = self._quality_cache.get(config_key)

        if entry is not None:
            if now - entry.timestamp < self._ttl_seconds:
                # Cache hit
                entry.hits += 1
                self._stats.total_hits += 1
                return entry.value
            else:
                # Entry expired - remove it
                del self._quality_cache[config_key]

        # Cache miss
        self._stats.total_misses += 1
        return None

    def get_quality_or_fetch(
        self,
        config_key: str,
        fetch_fn: Optional[Callable[[str], Optional[float]]] = None,
    ) -> float:
        """Get quality score, fetching from provider on cache miss.

        This is the primary method for getting quality scores. It will:
        1. Check cache for valid entry
        2. On miss, call fetch_fn or self._quality_provider
        3. Cache the result and return

        Args:
            config_key: Config key like "hex8_2p"
            fetch_fn: Optional callback to fetch quality. If None, uses
                      the quality_provider from __init__ or default.

        Returns:
            Quality score (0.0-1.0), default if fetch fails
        """
        # Check cache first
        cached = self.get_quality(config_key)
        if cached is not None:
            return cached

        # Cache miss - fetch from provider
        quality = self._default_quality
        provider = fetch_fn or self._quality_provider

        if provider is not None:
            try:
                result = provider(config_key)
                if result is not None:
                    quality = result
            except Exception as e:
                logger.debug(f"{self._log_prefix} Error fetching quality for {config_key}: {e}")

        # Update cache
        self.set_quality(config_key, quality)
        return quality

    def set_quality(self, config_key: str, quality: float) -> None:
        """Set quality score in cache.

        Args:
            config_key: Config key like "hex8_2p"
            quality: Quality score (0.0-1.0)
        """
        self._quality_cache[config_key] = CacheEntry(
            value=quality,
            timestamp=time.time(),
            hits=0,
        )

    def invalidate(self, config_key: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Call this when quality data changes externally (e.g., after
        evaluation completes, after parity gate updates).

        Args:
            config_key: Specific config to invalidate, or None to clear all

        Returns:
            Number of entries invalidated
        """
        self._stats.total_invalidations += 1

        if config_key is None:
            count = len(self._quality_cache)
            self._quality_cache.clear()
            logger.debug(f"{self._log_prefix} Cleared all quality cache entries ({count} entries)")
            return count
        elif config_key in self._quality_cache:
            del self._quality_cache[config_key]
            logger.debug(f"{self._log_prefix} Invalidated quality cache for {config_key}")
            return 1

        return 0

    def get_all_qualities(
        self,
        config_keys: list[str],
        fetch_fn: Optional[Callable[[str], Optional[float]]] = None,
    ) -> dict[str, float]:
        """Get quality scores for multiple configs.

        Args:
            config_keys: List of config keys to look up
            fetch_fn: Optional callback for cache misses

        Returns:
            Dict mapping config_key to quality score
        """
        return {
            config: self.get_quality_or_fetch(config, fetch_fn)
            for config in config_keys
        }

    def get_cached_keys(self) -> list[str]:
        """Get list of currently cached config keys.

        Returns:
            List of config keys with valid cache entries
        """
        now = time.time()
        return [
            key for key, entry in self._quality_cache.items()
            if now - entry.timestamp < self._ttl_seconds
        ]

    def prune_expired(self) -> int:
        """Remove all expired entries from cache.

        Call this periodically to free memory from expired entries.

        Returns:
            Number of entries pruned
        """
        now = time.time()
        expired = [
            key for key, entry in self._quality_cache.items()
            if now - entry.timestamp >= self._ttl_seconds
        ]

        for key in expired:
            del self._quality_cache[key]

        if expired:
            logger.debug(f"{self._log_prefix} Pruned {len(expired)} expired cache entries")

        return len(expired)

    def get_status(self) -> dict[str, Any]:
        """Get cache status for health checks and monitoring.

        Returns:
            Dict with cache statistics and configuration
        """
        stats = self.stats
        return {
            "ttl_seconds": self._ttl_seconds,
            "default_quality": self._default_quality,
            "entries_count": stats.entries_count,
            "total_hits": stats.total_hits,
            "total_misses": stats.total_misses,
            "hit_rate": round(stats.hit_rate, 3),
            "total_invalidations": stats.total_invalidations,
            "cached_configs": self.get_cached_keys(),
        }


# Convenience factory function
def create_quality_cache(
    ttl_seconds: float = ConfigStateCache.DEFAULT_TTL,
    use_daemon: bool = True,
) -> ConfigStateCache:
    """Create a ConfigStateCache with optional QualityMonitorDaemon integration.

    Args:
        ttl_seconds: Cache TTL in seconds
        use_daemon: If True, configure to fetch from QualityMonitorDaemon

    Returns:
        Configured ConfigStateCache instance
    """
    quality_provider = None

    if use_daemon:
        def fetch_from_daemon(config_key: str) -> Optional[float]:
            """Fetch quality from QualityMonitorDaemon."""
            try:
                from app.coordination.quality_monitor_daemon import get_quality_daemon
                daemon = get_quality_daemon()
                if daemon:
                    return daemon.get_config_quality(config_key)
            except ImportError:
                logger.debug("[ConfigStateCache] quality_monitor_daemon not available")
            except (AttributeError, KeyError) as e:
                logger.debug(f"[ConfigStateCache] Error fetching quality: {e}")
            return None

        quality_provider = fetch_from_daemon

    return ConfigStateCache(
        ttl_seconds=ttl_seconds,
        quality_provider=quality_provider,
    )
