"""Tests for ConfigStateCache.

December 30, 2025: Unit tests for the extracted ConfigStateCache class.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.config_state_cache import (
    CacheEntry,
    CacheStats,
    ConfigStateCache,
    create_quality_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self):
        """Test creating a cache entry."""
        entry = CacheEntry(value=0.8, timestamp=1000.0)
        assert entry.value == 0.8
        assert entry.timestamp == 1000.0
        assert entry.hits == 0

    def test_entry_with_hits(self):
        """Test entry with hits counter."""
        entry = CacheEntry(value=0.9, timestamp=1000.0, hits=5)
        assert entry.hits == 5


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = CacheStats()
        assert stats.total_hits == 0
        assert stats.total_misses == 0
        assert stats.total_invalidations == 0
        assert stats.entries_count == 0

    def test_hit_rate_empty(self):
        """Test hit rate with no accesses."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        stats = CacheStats(total_hits=100, total_misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed hits and misses."""
        stats = CacheStats(total_hits=75, total_misses=25)
        assert stats.hit_rate == 0.75

    def test_hit_rate_all_misses(self):
        """Test hit rate with all misses."""
        stats = CacheStats(total_hits=0, total_misses=100)
        assert stats.hit_rate == 0.0


class TestConfigStateCache:
    """Tests for ConfigStateCache class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        cache = ConfigStateCache()
        assert cache.ttl_seconds == 30.0
        assert cache._default_quality == 0.7
        assert cache._quality_provider is None

    def test_init_custom_ttl(self):
        """Test initialization with custom TTL."""
        cache = ConfigStateCache(ttl_seconds=60.0)
        assert cache.ttl_seconds == 60.0

    def test_init_custom_default_quality(self):
        """Test initialization with custom default quality."""
        cache = ConfigStateCache(default_quality=0.5)
        assert cache._default_quality == 0.5

    def test_init_with_provider(self):
        """Test initialization with quality provider."""
        provider = MagicMock(return_value=0.9)
        cache = ConfigStateCache(quality_provider=provider)
        assert cache._quality_provider is provider

    def test_set_get_quality(self):
        """Test setting and getting quality."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)

        result = cache.get_quality("hex8_2p")
        assert result == 0.85

    def test_get_quality_miss(self):
        """Test cache miss returns None."""
        cache = ConfigStateCache()
        result = cache.get_quality("hex8_2p")
        assert result is None

    def test_get_quality_expired(self):
        """Test expired entry returns None."""
        cache = ConfigStateCache(ttl_seconds=1.0)
        cache.set_quality("hex8_2p", 0.85)

        # Wait for expiration
        time.sleep(1.1)

        result = cache.get_quality("hex8_2p")
        assert result is None

    def test_get_quality_increments_hits(self):
        """Test cache hit increments hit counter."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)

        # Multiple gets should increment hits
        cache.get_quality("hex8_2p")
        cache.get_quality("hex8_2p")
        cache.get_quality("hex8_2p")

        assert cache.stats.total_hits == 3

    def test_get_quality_increments_misses(self):
        """Test cache miss increments miss counter."""
        cache = ConfigStateCache()

        cache.get_quality("hex8_2p")
        cache.get_quality("square8_2p")

        assert cache.stats.total_misses == 2

    def test_get_quality_or_fetch_cached(self):
        """Test get_quality_or_fetch returns cached value."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)

        fetch_fn = MagicMock()
        result = cache.get_quality_or_fetch("hex8_2p", fetch_fn)

        assert result == 0.85
        fetch_fn.assert_not_called()

    def test_get_quality_or_fetch_miss_calls_provider(self):
        """Test get_quality_or_fetch calls provider on miss."""
        provider = MagicMock(return_value=0.9)
        cache = ConfigStateCache(quality_provider=provider)

        result = cache.get_quality_or_fetch("hex8_2p")

        assert result == 0.9
        provider.assert_called_once_with("hex8_2p")

    def test_get_quality_or_fetch_miss_calls_fetch_fn(self):
        """Test get_quality_or_fetch calls fetch_fn on miss."""
        cache = ConfigStateCache()
        fetch_fn = MagicMock(return_value=0.95)

        result = cache.get_quality_or_fetch("hex8_2p", fetch_fn)

        assert result == 0.95
        fetch_fn.assert_called_once_with("hex8_2p")

    def test_get_quality_or_fetch_fetch_fn_overrides_provider(self):
        """Test fetch_fn takes precedence over provider."""
        provider = MagicMock(return_value=0.8)
        fetch_fn = MagicMock(return_value=0.9)
        cache = ConfigStateCache(quality_provider=provider)

        result = cache.get_quality_or_fetch("hex8_2p", fetch_fn)

        assert result == 0.9
        fetch_fn.assert_called_once()
        provider.assert_not_called()

    def test_get_quality_or_fetch_provider_returns_none(self):
        """Test default quality when provider returns None."""
        provider = MagicMock(return_value=None)
        cache = ConfigStateCache(quality_provider=provider, default_quality=0.6)

        result = cache.get_quality_or_fetch("hex8_2p")

        assert result == 0.6

    def test_get_quality_or_fetch_provider_raises_exception(self):
        """Test default quality when provider raises exception."""
        provider = MagicMock(side_effect=RuntimeError("DB error"))
        cache = ConfigStateCache(quality_provider=provider, default_quality=0.6)

        result = cache.get_quality_or_fetch("hex8_2p")

        assert result == 0.6

    def test_get_quality_or_fetch_caches_result(self):
        """Test get_quality_or_fetch caches the fetched value."""
        provider = MagicMock(return_value=0.9)
        cache = ConfigStateCache(quality_provider=provider)

        # First call fetches
        cache.get_quality_or_fetch("hex8_2p")
        # Second call should use cache
        cache.get_quality_or_fetch("hex8_2p")

        provider.assert_called_once()

    def test_invalidate_single(self):
        """Test invalidating single config."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)
        cache.set_quality("square8_2p", 0.75)

        count = cache.invalidate("hex8_2p")

        assert count == 1
        assert cache.get_quality("hex8_2p") is None
        assert cache.get_quality("square8_2p") == 0.75

    def test_invalidate_all(self):
        """Test invalidating all configs."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)
        cache.set_quality("square8_2p", 0.75)
        cache.set_quality("hex8_4p", 0.65)

        count = cache.invalidate()

        assert count == 3
        assert cache.get_quality("hex8_2p") is None
        assert cache.get_quality("square8_2p") is None
        assert cache.get_quality("hex8_4p") is None

    def test_invalidate_nonexistent(self):
        """Test invalidating nonexistent config."""
        cache = ConfigStateCache()
        count = cache.invalidate("hex8_2p")
        assert count == 0

    def test_invalidate_increments_counter(self):
        """Test invalidation increments counter."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)

        cache.invalidate("hex8_2p")
        cache.invalidate()

        assert cache.stats.total_invalidations == 2

    def test_get_all_qualities(self):
        """Test batch quality lookup."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)
        cache.set_quality("square8_2p", 0.75)

        provider = MagicMock(return_value=0.6)
        cache._quality_provider = provider

        result = cache.get_all_qualities(["hex8_2p", "square8_2p", "hex8_4p"])

        assert result["hex8_2p"] == 0.85
        assert result["square8_2p"] == 0.75
        assert result["hex8_4p"] == 0.6

    def test_get_cached_keys(self):
        """Test getting list of cached keys."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)
        cache.set_quality("square8_2p", 0.75)

        keys = cache.get_cached_keys()
        assert set(keys) == {"hex8_2p", "square8_2p"}

    def test_get_cached_keys_excludes_expired(self):
        """Test get_cached_keys excludes expired entries."""
        cache = ConfigStateCache(ttl_seconds=1.0)
        cache.set_quality("hex8_2p", 0.85)

        time.sleep(1.1)
        cache.set_quality("square8_2p", 0.75)

        keys = cache.get_cached_keys()
        assert keys == ["square8_2p"]

    def test_prune_expired(self):
        """Test pruning expired entries."""
        cache = ConfigStateCache(ttl_seconds=1.0)
        cache.set_quality("hex8_2p", 0.85)

        time.sleep(1.1)
        cache.set_quality("square8_2p", 0.75)

        pruned = cache.prune_expired()

        assert pruned == 1
        assert cache.get_quality("hex8_2p") is None
        assert cache.get_quality("square8_2p") == 0.75

    def test_prune_expired_empty(self):
        """Test pruning with no expired entries."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)

        pruned = cache.prune_expired()
        assert pruned == 0

    def test_get_status(self):
        """Test status reporting."""
        cache = ConfigStateCache(ttl_seconds=30.0)
        cache.set_quality("hex8_2p", 0.85)
        cache.get_quality("hex8_2p")  # Hit

        status = cache.get_status()

        assert status["ttl_seconds"] == 30.0
        assert status["default_quality"] == 0.7
        assert status["entries_count"] == 1
        assert status["total_hits"] == 1
        assert "hex8_2p" in status["cached_configs"]

    def test_ttl_setter_clears_cache(self):
        """Test changing TTL clears cache."""
        cache = ConfigStateCache(ttl_seconds=30.0)
        cache.set_quality("hex8_2p", 0.85)

        cache.ttl_seconds = 60.0

        assert cache.ttl_seconds == 60.0
        assert cache.get_quality("hex8_2p") is None

    def test_stats_property_updates_entries_count(self):
        """Test stats property updates entries count."""
        cache = ConfigStateCache()
        cache.set_quality("hex8_2p", 0.85)
        cache.set_quality("square8_2p", 0.75)

        stats = cache.stats
        assert stats.entries_count == 2


class TestCreateQualityCache:
    """Tests for create_quality_cache factory function."""

    def test_create_with_defaults(self):
        """Test creating cache with defaults."""
        cache = create_quality_cache()
        assert cache.ttl_seconds == 30.0

    def test_create_with_custom_ttl(self):
        """Test creating cache with custom TTL."""
        cache = create_quality_cache(ttl_seconds=60.0)
        assert cache.ttl_seconds == 60.0

    def test_create_without_daemon(self):
        """Test creating cache without daemon integration."""
        cache = create_quality_cache(use_daemon=False)
        assert cache._quality_provider is None

    @patch("app.coordination.config_state_cache.logger")
    def test_create_with_daemon_integration(self, mock_logger):
        """Test creating cache with daemon integration."""
        cache = create_quality_cache(use_daemon=True)
        assert cache._quality_provider is not None

        # The provider should handle missing daemon gracefully
        result = cache._quality_provider("hex8_2p")
        assert result is None  # Daemon not running


class TestCacheConcurrency:
    """Tests for cache behavior under concurrent-like access patterns."""

    def test_rapid_set_get_cycles(self):
        """Test rapid set/get cycles."""
        cache = ConfigStateCache()

        for i in range(100):
            config = f"config_{i % 10}"
            cache.set_quality(config, i / 100)
            cache.get_quality(config)

        assert cache.stats.total_hits >= 90

    def test_interleaved_invalidations(self):
        """Test interleaved sets and invalidations."""
        cache = ConfigStateCache()

        for i in range(50):
            cache.set_quality(f"config_{i}", 0.5)
            if i % 5 == 0:
                cache.invalidate()

        # Should have entries only from last batch
        assert len(cache.get_cached_keys()) <= 4

    def test_provider_called_only_once_per_miss(self):
        """Test provider called exactly once per cache miss."""
        call_count = {"count": 0}

        def counting_provider(config: str) -> float:
            call_count["count"] += 1
            return 0.8

        cache = ConfigStateCache(quality_provider=counting_provider)

        # Multiple fetches for same config
        cache.get_quality_or_fetch("hex8_2p")
        cache.get_quality_or_fetch("hex8_2p")
        cache.get_quality_or_fetch("hex8_2p")

        # Provider should be called only once
        assert call_count["count"] == 1
