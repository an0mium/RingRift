"""Tests for app.caching.decorators module.

Comprehensive tests for function caching decorators.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from app.caching.decorators import (
    _cache_registry,
    _make_key,
    async_cached,
    cached,
    get_cache_stats,
    invalidate_cache,
)


# =============================================================================
# Test _make_key Helper
# =============================================================================


class TestMakeKey:
    """Tests for the _make_key helper function."""

    def test_make_key_with_args(self):
        """Key is created from positional args."""
        key = _make_key((1, 2, 3), {})
        assert "1" in key
        assert "2" in key
        assert "3" in key

    def test_make_key_with_kwargs(self):
        """Key includes keyword arguments."""
        key = _make_key((), {"a": 1, "b": 2})
        assert "a=1" in key
        assert "b=2" in key

    def test_make_key_with_mixed_args(self):
        """Key includes both positional and keyword args."""
        key = _make_key((1,), {"x": "y"})
        assert "1" in key
        assert "x='y'" in key

    def test_make_key_kwargs_sorted(self):
        """Kwargs are sorted for consistent keys."""
        key1 = _make_key((), {"a": 1, "b": 2})
        key2 = _make_key((), {"b": 2, "a": 1})
        assert key1 == key2

    def test_make_key_long_key_hashed(self):
        """Long keys are hashed for efficiency."""
        long_args = tuple(range(100))
        key = _make_key(long_args, {})
        # MD5 hash is 32 chars
        assert len(key) <= 32 or len(key) < 200


# =============================================================================
# Test @cached Decorator
# =============================================================================


class TestCachedDecorator:
    """Tests for the @cached decorator."""

    def teardown_method(self):
        """Clear cache registry after each test."""
        _cache_registry.clear()

    def test_cached_returns_same_result(self):
        """Cached function returns same result for same args."""
        call_count = 0

        @cached()
        def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = my_func(5)
        result2 = my_func(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once

    def test_cached_different_args_cached_separately(self):
        """Different arguments are cached separately."""
        call_count = 0

        @cached()
        def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = my_func(5)
        result2 = my_func(10)

        assert result1 == 10
        assert result2 == 20
        assert call_count == 2

    def test_cached_with_ttl(self):
        """Cached results expire after TTL."""
        call_count = 0

        @cached(ttl_seconds=0.1)
        def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = my_func(5)
        time.sleep(0.15)  # Wait for expiration
        result2 = my_func(5)

        assert result1 == result2
        assert call_count == 2  # Called twice due to expiration

    def test_cached_max_size(self):
        """Cache respects max_size limit."""
        @cached(max_size=2)
        def my_func(x):
            return x * 2

        # Fill cache
        my_func(1)
        my_func(2)
        # This should evict the oldest (1)
        my_func(3)

        # Verify cache size via attached cache
        assert len(my_func._cache) == 2

    def test_cached_with_custom_key_func(self):
        """Custom key function is used."""
        @cached(key_func=lambda self, x: f"custom:{x}")
        def my_func(self, x):
            return x * 2

        result = my_func(None, 5)
        assert result == 10

        # Verify key was used
        cache = my_func._cache
        assert cache.has("custom:5")

    def test_cached_with_cache_name(self):
        """Cache name is registered in global registry."""
        @cached(cache_name="my_custom_cache")
        def my_func(x):
            return x

        my_func(1)

        assert "my_custom_cache" in _cache_registry

    def test_cached_none_not_cached(self):
        """None results are not cached."""
        call_count = 0

        @cached()
        def my_func():
            nonlocal call_count
            call_count += 1
            return None

        my_func()
        my_func()

        # Called twice because None is not cached
        assert call_count == 2

    def test_cached_preserves_metadata(self):
        """Decorator preserves function metadata."""
        @cached()
        def documented_func():
            """This is a docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    def test_cached_has_attached_cache(self):
        """Decorated function has _cache attribute."""
        @cached()
        def my_func(x):
            return x

        assert hasattr(my_func, "_cache")
        assert hasattr(my_func, "_cache_name")

    def test_cached_with_kwargs(self):
        """Works with keyword arguments."""
        call_count = 0

        @cached()
        def greet(name, greeting="Hello"):
            nonlocal call_count
            call_count += 1
            return f"{greeting}, {name}!"

        result1 = greet("World", greeting="Hi")
        result2 = greet("World", greeting="Hi")

        assert result1 == "Hi, World!"
        assert call_count == 1


# =============================================================================
# Test @async_cached Decorator
# =============================================================================


class TestAsyncCachedDecorator:
    """Tests for the @async_cached decorator."""

    def teardown_method(self):
        """Clear cache registry after each test."""
        _cache_registry.clear()

    @pytest.mark.asyncio
    async def test_async_cached_returns_same_result(self):
        """Async cached function returns same result for same args."""
        call_count = 0

        @async_cached()
        async def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await my_func(5)
        result2 = await my_func(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_cached_different_args(self):
        """Different arguments are cached separately."""
        call_count = 0

        @async_cached()
        async def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await my_func(5)
        result2 = await my_func(10)

        assert result1 == 10
        assert result2 == 20
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_cached_with_ttl(self):
        """Async cached results expire after TTL."""
        call_count = 0

        @async_cached(ttl_seconds=0.1)
        async def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await my_func(5)
        await asyncio.sleep(0.15)
        result2 = await my_func(5)

        assert result1 == result2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_cached_with_custom_key(self):
        """Custom key function works with async."""
        @async_cached(key_func=lambda x: f"async:{x}")
        async def my_func(x):
            return x * 2

        result = await my_func(5)
        assert result == 10

        cache = my_func._cache
        assert cache.has("async:5")

    @pytest.mark.asyncio
    async def test_async_cached_preserves_metadata(self):
        """Decorator preserves async function metadata."""
        @async_cached()
        async def documented_async():
            """Async docstring."""
            pass

        assert documented_async.__name__ == "documented_async"
        assert documented_async.__doc__ == "Async docstring."

    @pytest.mark.asyncio
    async def test_async_cached_concurrent_access(self):
        """Concurrent access to cached function works."""
        call_count = 0

        @async_cached()
        async def my_func(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        # Call same key concurrently
        results = await asyncio.gather(
            my_func(5),
            my_func(5),
            my_func(5),
        )

        # First call computes, others may hit cache
        assert all(r == 10 for r in results)
        # At least 1, at most 3 calls (depends on timing)
        assert 1 <= call_count <= 3


# =============================================================================
# Test invalidate_cache Function
# =============================================================================


class TestInvalidateCache:
    """Tests for the invalidate_cache function."""

    def teardown_method(self):
        """Clear cache registry after each test."""
        _cache_registry.clear()

    def test_invalidate_by_name(self):
        """Can invalidate cache by name."""
        @cached(cache_name="my_cache")
        def my_func(x):
            return x * 2

        my_func(5)
        assert my_func._cache.has("5")

        cleared = invalidate_cache("my_cache")
        assert cleared == 1
        assert not my_func._cache.has("5")

    def test_invalidate_by_func(self):
        """Can invalidate cache by function."""
        @cached()
        def my_func(x):
            return x * 2

        my_func(5)
        my_func(10)

        cleared = invalidate_cache(func=my_func)
        assert cleared == 2
        assert not my_func._cache.has("5")
        assert not my_func._cache.has("10")

    def test_invalidate_all(self):
        """Can invalidate all caches."""
        @cached(cache_name="cache1")
        def func1(x):
            return x

        @cached(cache_name="cache2")
        def func2(x):
            return x

        func1(1)
        func2(2)

        cleared = invalidate_cache()
        assert cleared == 2

    def test_invalidate_unknown_name_returns_zero(self):
        """Invalidating unknown cache name returns 0."""
        cleared = invalidate_cache("nonexistent")
        assert cleared == 0


# =============================================================================
# Test get_cache_stats Function
# =============================================================================


class TestGetCacheStats:
    """Tests for the get_cache_stats function."""

    def teardown_method(self):
        """Clear cache registry after each test."""
        _cache_registry.clear()

    def test_get_stats_for_all_caches(self):
        """get_cache_stats returns stats for all caches."""
        @cached(cache_name="cache1")
        def func1(x):
            return x

        @cached(cache_name="cache2")
        def func2(x):
            return x

        func1(1)
        func2(2)

        stats = get_cache_stats()
        assert "cache1" in stats
        assert "cache2" in stats

    def test_stats_include_hits_misses(self):
        """Stats include hit/miss counts."""
        @cached(cache_name="test_cache")
        def my_func(x):
            return x

        my_func(1)  # miss
        my_func(1)  # hit
        my_func(2)  # miss

        stats = get_cache_stats()["test_cache"]
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2

    def test_stats_include_hit_rate(self):
        """Stats include hit rate."""
        @cached(cache_name="test_cache")
        def my_func(x):
            return x

        my_func(1)  # miss
        my_func(1)  # hit

        stats = get_cache_stats()["test_cache"]
        assert stats["hit_rate"] == 0.5


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestCachingEdgeCases:
    """Edge cases and integration tests."""

    def teardown_method(self):
        """Clear cache registry after each test."""
        _cache_registry.clear()

    def test_cached_with_complex_args(self):
        """Works with complex argument types."""
        @cached()
        def my_func(data):
            return sum(data)

        result = my_func([1, 2, 3])
        assert result == 6

    def test_cached_with_method(self):
        """Works as a method decorator."""
        class MyClass:
            def __init__(self):
                self.call_count = 0

            @cached(key_func=lambda self, x: f"method:{x}")
            def compute(self, x):
                self.call_count += 1
                return x * 2

        obj = MyClass()
        result1 = obj.compute(5)
        result2 = obj.compute(5)

        assert result1 == 10
        assert result2 == 10
        assert obj.call_count == 1

    def test_multiple_decorated_functions(self):
        """Multiple decorated functions have separate caches."""
        @cached()
        def func1(x):
            return x * 2

        @cached()
        def func2(x):
            return x * 3

        func1(5)
        func2(5)

        # Each has its own cache
        assert func1._cache is not func2._cache

    def test_cached_with_exception(self):
        """Exception in cached function is not cached."""
        call_count = 0

        @cached()
        def flaky(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return x * 2

        with pytest.raises(ValueError):
            flaky(5)

        # Second call succeeds and is cached
        result = flaky(5)
        assert result == 10
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_cached_with_exception(self):
        """Exception in async cached function is not cached."""
        call_count = 0

        @async_cached()
        async def flaky(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return x * 2

        with pytest.raises(ValueError):
            await flaky(5)

        result = await flaky(5)
        assert result == 10

    def test_cache_registry_uses_qualname(self):
        """Cache uses qualified name by default."""
        @cached()
        def my_func(x):
            return x

        my_func(1)

        # Should use function's qualname
        assert my_func._cache_name == "TestCachingEdgeCases.test_cache_registry_uses_qualname.<locals>.my_func"


class TestCachingConcurrency:
    """Tests for caching under concurrent access."""

    def teardown_method(self):
        """Clear cache registry after each test."""
        _cache_registry.clear()

    @pytest.mark.asyncio
    async def test_async_cache_thread_safety(self):
        """Async cache handles concurrent access."""
        results = []

        @async_cached()
        async def compute(x):
            await asyncio.sleep(0.01)
            return x * 2

        async def worker(i):
            result = await compute(i % 5)
            results.append(result)

        # Launch many concurrent tasks
        await asyncio.gather(*[worker(i) for i in range(20)])

        # All results should be correct
        for i, r in enumerate(results):
            assert r == (i % 5) * 2
