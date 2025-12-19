"""Unified Caching Abstraction for RingRift AI Service.

This package provides standardized caching utilities with consistent
behavior across the codebase:
- TTL-based expiration
- Size-limited LRU eviction
- Thread-safe operations
- Both in-memory and persistent backends

Usage:
    from app.caching import (
        Cache,
        MemoryCache,
        FileCache,
        TTLCache,
        cached,
    )

    # In-memory cache with TTL and size limit
    cache = MemoryCache(max_size=100, ttl_seconds=3600)
    cache.set("key", value)
    value = cache.get("key")

    # File-backed persistent cache
    cache = FileCache("/path/to/cache", ttl_seconds=86400)

    # Decorator for function memoization
    @cached(ttl_seconds=300, max_size=100)
    def expensive_function(arg):
        ...

Integration Guidelines:
    - Use MemoryCache for frequently accessed, small data
    - Use FileCache for persistent caching across restarts
    - Use TTLCache when expiration is critical
    - Prefer the @cached decorator for function memoization
"""

from app.caching.base import (
    Cache,
    CacheEntry,
    CacheStats,
    CacheConfig,
)
from app.caching.memory import (
    MemoryCache,
    TTLCache,
    LRUCache,
)
from app.caching.decorators import (
    cached,
    async_cached,
    invalidate_cache,
)

__all__ = [
    # Base
    "Cache",
    "CacheEntry",
    "CacheStats",
    "CacheConfig",
    # Memory caches
    "MemoryCache",
    "TTLCache",
    "LRUCache",
    # Decorators
    "cached",
    "async_cached",
    "invalidate_cache",
]
