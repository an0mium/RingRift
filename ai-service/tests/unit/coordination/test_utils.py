"""Tests for coordination utility classes (December 2025).

Tests for app/coordination/utils.py:
- BoundedHistory[T] - Fixed-size history collection
- HistoryEntry - Entry dataclass with metadata
- MetricsAccumulator - Windowed statistics
- MetricsSnapshot - Snapshot dataclass
- CallbackRegistry[T] - Type-safe callback management
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import Mock

import pytest

from app.coordination.utils import (
    BoundedHistory,
    CallbackRegistry,
    HistoryEntry,
    MetricsAccumulator,
    MetricsSnapshot,
)


# =============================================================================
# HistoryEntry Tests
# =============================================================================


class TestHistoryEntry:
    """Tests for HistoryEntry dataclass."""

    def test_create_entry_with_defaults(self):
        """Create entry with default timestamp and empty metadata."""
        entry = HistoryEntry(value="test")
        assert entry.value == "test"
        assert isinstance(entry.timestamp, float)
        assert entry.metadata == {}

    def test_create_entry_with_metadata(self):
        """Create entry with custom metadata."""
        entry = HistoryEntry(value=42, metadata={"source": "test"})
        assert entry.value == 42
        assert entry.metadata["source"] == "test"

    def test_create_entry_with_explicit_timestamp(self):
        """Create entry with explicit timestamp."""
        ts = 1735312800.0
        entry = HistoryEntry(value="x", timestamp=ts)
        assert entry.timestamp == ts


# =============================================================================
# BoundedHistory Tests
# =============================================================================


class TestBoundedHistoryBasics:
    """Basic BoundedHistory functionality."""

    def test_create_empty_history(self):
        """Create an empty history."""
        history = BoundedHistory[str](max_size=10)
        assert len(history) == 0
        assert history.oldest is None
        assert history.newest is None

    def test_append_single_item(self):
        """Append a single item."""
        history = BoundedHistory[int](max_size=10)
        history.append(42)
        assert len(history) == 1
        assert history.oldest == 42
        assert history.newest == 42

    def test_append_multiple_items(self):
        """Append multiple items in sequence."""
        history = BoundedHistory[int](max_size=10)
        for i in range(5):
            history.append(i)
        assert len(history) == 5
        assert history.oldest == 0
        assert history.newest == 4

    def test_extend_history(self):
        """Extend history with list of items."""
        history = BoundedHistory[str](max_size=10)
        history.extend(["a", "b", "c"])
        assert len(history) == 3
        assert history.get_all() == ["a", "b", "c"]

    def test_clear_history(self):
        """Clear all history."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3])
        history.clear()
        assert len(history) == 0
        assert history.oldest is None


class TestBoundedHistoryBounding:
    """Tests for max_size enforcement."""

    def test_evict_oldest_when_full(self):
        """Oldest items are evicted when max_size is reached."""
        history = BoundedHistory[int](max_size=3)
        for i in range(5):
            history.append(i)
        assert len(history) == 3
        assert history.get_all() == [2, 3, 4]

    def test_is_full_property(self):
        """is_full property is accurate."""
        history = BoundedHistory[int](max_size=3)
        assert not history.is_full
        history.extend([1, 2, 3])
        assert history.is_full

    def test_total_added_tracks_evicted(self):
        """total_added includes evicted items."""
        history = BoundedHistory[int](max_size=3)
        for i in range(10):
            history.append(i)
        assert len(history) == 3
        assert history.total_added == 10


class TestBoundedHistoryRetrieval:
    """Tests for retrieving items from history."""

    def test_get_recent_items(self):
        """Get n most recent items."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5])
        assert history.get_recent(3) == [3, 4, 5]
        assert history.get_recent(10) == [1, 2, 3, 4, 5]

    def test_get_recent_zero_or_negative(self):
        """get_recent with zero or negative returns empty list."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3])
        assert history.get_recent(0) == []
        assert history.get_recent(-1) == []

    def test_get_oldest_items(self):
        """Get n oldest items."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5])
        assert history.get_oldest(3) == [1, 2, 3]

    def test_get_all_items(self):
        """Get all items as list."""
        history = BoundedHistory[str](max_size=10)
        history.extend(["a", "b", "c"])
        assert history.get_all() == ["a", "b", "c"]

    def test_get_entries_returns_full_entries(self):
        """get_entries returns HistoryEntry objects."""
        history = BoundedHistory[str](max_size=10)
        history.append("test", source="unit_test")
        entries = history.get_entries()
        assert len(entries) == 1
        assert entries[0].value == "test"
        assert entries[0].metadata["source"] == "unit_test"


class TestBoundedHistoryTimestamps:
    """Tests for timestamp tracking."""

    def test_oldest_newest_timestamps(self):
        """oldest_timestamp and newest_timestamp are accurate."""
        history = BoundedHistory[int](max_size=10, track_timestamps=True)
        t1 = time.time()
        history.append(1)
        time.sleep(0.01)
        history.append(2)
        t2 = time.time()

        assert history.oldest_timestamp is not None
        assert history.newest_timestamp is not None
        assert t1 <= history.oldest_timestamp <= history.newest_timestamp <= t2

    def test_filter_by_time(self):
        """Filter items by timestamp range."""
        history = BoundedHistory[int](max_size=10, track_timestamps=True)
        history.append(1)
        mid_time = time.time()
        time.sleep(0.01)
        history.append(2)
        history.append(3)

        # Items after mid_time
        recent = history.filter_by_time(since=mid_time)
        assert len(recent) == 2

    def test_disable_timestamps(self):
        """Timestamps can be disabled for performance."""
        history = BoundedHistory[int](max_size=10, track_timestamps=False)
        history.append(1)
        entries = history.get_entries()
        assert entries[0].timestamp == 0.0


class TestBoundedHistoryFiltering:
    """Tests for filtering and counting."""

    def test_filter_by_predicate(self):
        """Filter items by custom predicate."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5, 6])
        evens = history.filter(lambda x: x % 2 == 0)
        assert evens == [2, 4, 6]

    def test_count_matching(self):
        """Count items matching predicate."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3, 4, 5])
        count = history.count_matching(lambda x: x > 3)
        assert count == 2


class TestBoundedHistoryIterators:
    """Tests for iteration and container protocols."""

    def test_iterate_history(self):
        """Iterate over history items."""
        history = BoundedHistory[int](max_size=10)
        history.extend([1, 2, 3])
        items = list(history)
        assert items == [1, 2, 3]

    def test_bool_conversion(self):
        """Bool conversion is accurate."""
        history = BoundedHistory[int](max_size=10)
        assert not history
        history.append(1)
        assert history

    def test_contains_check(self):
        """Item containment check works."""
        history = BoundedHistory[str](max_size=10)
        history.extend(["a", "b", "c"])
        assert "b" in history
        assert "z" not in history


# =============================================================================
# MetricsSnapshot Tests
# =============================================================================


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_create_snapshot(self):
        """Create a metrics snapshot."""
        snapshot = MetricsSnapshot(
            count=100,
            total=150.0,
            mean=1.5,
            min_value=0.5,
            max_value=3.0,
            std_dev=0.3,
            recent_mean=1.6,
            trend=0.01,
        )
        assert snapshot.count == 100
        assert snapshot.mean == 1.5
        assert snapshot.trend == 0.01


# =============================================================================
# MetricsAccumulator Tests
# =============================================================================


class TestMetricsAccumulatorBasics:
    """Basic MetricsAccumulator functionality."""

    def test_create_empty_accumulator(self):
        """Create empty accumulator."""
        metrics = MetricsAccumulator(window_size=50, name="test")
        assert metrics.count == 0
        assert metrics.mean == 0.0
        assert metrics.current is None

    def test_add_single_value(self):
        """Add single value."""
        metrics = MetricsAccumulator(window_size=50)
        metrics.add(1.0)
        assert metrics.count == 1
        assert metrics.mean == 1.0
        assert metrics.current == 1.0

    def test_add_multiple_values(self):
        """Add multiple values."""
        metrics = MetricsAccumulator(window_size=50)
        metrics.add(1.0)
        metrics.add(2.0)
        metrics.add(3.0)
        assert metrics.count == 3
        assert metrics.mean == 2.0
        assert metrics.total == 6.0

    def test_add_batch(self):
        """Add batch of values."""
        metrics = MetricsAccumulator(window_size=50)
        metrics.add_batch([1.0, 2.0, 3.0, 4.0])
        assert metrics.count == 4
        assert metrics.mean == 2.5


class TestMetricsAccumulatorStatistics:
    """Tests for statistical calculations."""

    def test_min_max_tracking(self):
        """Min and max are tracked correctly."""
        metrics = MetricsAccumulator(window_size=50)
        metrics.add_batch([3.0, 1.0, 5.0, 2.0])
        assert metrics.min_value == 1.0
        assert metrics.max_value == 5.0

    def test_std_dev_calculation(self):
        """Standard deviation is calculated correctly."""
        metrics = MetricsAccumulator(window_size=50)
        metrics.add_batch([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        # Population: [2,4,4,4,5,5,7,9], mean=5, var=4, std=2
        # Sample std_dev is slightly different
        assert 1.5 < metrics.std_dev < 2.5

    def test_std_dev_with_few_values(self):
        """std_dev returns 0 with fewer than 2 values."""
        metrics = MetricsAccumulator(window_size=50)
        assert metrics.std_dev == 0.0
        metrics.add(1.0)
        assert metrics.std_dev == 0.0

    def test_variance_calculation(self):
        """Variance is calculated correctly."""
        metrics = MetricsAccumulator(window_size=50)
        metrics.add_batch([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert 3.0 < metrics.variance < 6.0


class TestMetricsAccumulatorWindow:
    """Tests for windowed statistics."""

    def test_window_eviction(self):
        """Oldest values are evicted from window."""
        metrics = MetricsAccumulator(window_size=3)
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.window_count == 3
        assert metrics.window_mean == 4.0  # (3+4+5)/3

    def test_window_mean_vs_total_mean(self):
        """Window mean differs from total mean."""
        metrics = MetricsAccumulator(window_size=2)
        metrics.add_batch([1.0, 2.0, 10.0, 11.0])
        assert metrics.mean == 6.0  # (1+2+10+11)/4
        assert metrics.window_mean == 10.5  # (10+11)/2

    def test_get_recent_values(self):
        """Get recent values from window."""
        metrics = MetricsAccumulator(window_size=5)
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.get_recent(3) == [3.0, 4.0, 5.0]


class TestMetricsAccumulatorTrend:
    """Tests for trend calculation."""

    def test_increasing_trend(self):
        """Trend is positive for increasing values."""
        metrics = MetricsAccumulator(window_size=10)
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.trend > 0

    def test_decreasing_trend(self):
        """Trend is negative for decreasing values."""
        metrics = MetricsAccumulator(window_size=10)
        metrics.add_batch([5.0, 4.0, 3.0, 2.0, 1.0])
        assert metrics.trend < 0

    def test_flat_trend(self):
        """Trend is near zero for constant values."""
        metrics = MetricsAccumulator(window_size=10)
        metrics.add_batch([3.0, 3.0, 3.0, 3.0, 3.0])
        assert abs(metrics.trend) < 0.01

    def test_trend_with_few_values(self):
        """Trend returns 0 with fewer than 2 values."""
        metrics = MetricsAccumulator(window_size=10)
        assert metrics.trend == 0.0
        metrics.add(1.0)
        assert metrics.trend == 0.0


class TestMetricsAccumulatorImprovement:
    """Tests for improvement tracking."""

    def test_is_improving_lower_is_better(self):
        """is_improving with higher_is_better=False (default)."""
        metrics = MetricsAccumulator(window_size=10, higher_is_better=False)
        metrics.add_batch([5.0, 4.0, 3.0, 2.0, 1.0])  # Decreasing = improving
        assert metrics.is_improving

    def test_is_improving_higher_is_better(self):
        """is_improving with higher_is_better=True."""
        metrics = MetricsAccumulator(window_size=10, higher_is_better=True)
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])  # Increasing = improving
        assert metrics.is_improving

    def test_best_value_lower_is_better(self):
        """best_value with higher_is_better=False."""
        metrics = MetricsAccumulator(window_size=10, higher_is_better=False)
        metrics.add_batch([3.0, 1.0, 5.0, 2.0])
        assert metrics.best_value == 1.0

    def test_best_value_higher_is_better(self):
        """best_value with higher_is_better=True."""
        metrics = MetricsAccumulator(window_size=10, higher_is_better=True)
        metrics.add_batch([3.0, 1.0, 5.0, 2.0])
        assert metrics.best_value == 5.0


class TestMetricsAccumulatorReset:
    """Tests for reset functionality."""

    def test_full_reset(self):
        """Full reset clears everything."""
        metrics = MetricsAccumulator(window_size=10)
        metrics.add_batch([1.0, 2.0, 3.0])
        metrics.reset()
        assert metrics.count == 0
        assert metrics.total == 0.0
        assert metrics.min_value is None
        assert metrics.max_value is None

    def test_window_reset(self):
        """Window reset keeps all-time stats."""
        metrics = MetricsAccumulator(window_size=10)
        metrics.add_batch([1.0, 2.0, 3.0])
        metrics.reset_window()
        assert metrics.count == 3
        assert metrics.total == 6.0
        assert metrics.window_count == 0


class TestMetricsAccumulatorSerialization:
    """Tests for serialization."""

    def test_get_snapshot(self):
        """get_snapshot returns MetricsSnapshot."""
        metrics = MetricsAccumulator(window_size=10, name="loss")
        metrics.add_batch([1.0, 2.0, 3.0])
        snapshot = metrics.get_snapshot()
        assert isinstance(snapshot, MetricsSnapshot)
        assert snapshot.count == 3
        assert snapshot.mean == 2.0

    def test_to_dict(self):
        """to_dict returns dictionary representation."""
        metrics = MetricsAccumulator(window_size=10, name="loss")
        metrics.add_batch([1.0, 2.0, 3.0])
        d = metrics.to_dict()
        assert d["name"] == "loss"
        assert d["count"] == 3
        assert d["mean"] == 2.0


# =============================================================================
# CallbackRegistry Tests
# =============================================================================


class TestCallbackRegistryBasics:
    """Basic CallbackRegistry functionality."""

    def test_create_empty_registry(self):
        """Create empty registry."""
        registry = CallbackRegistry[str](name="test")
        assert registry.count == 0

    def test_register_callback(self):
        """Register a callback."""
        registry = CallbackRegistry[str]()

        def handler(data: str) -> None:
            pass

        registry.register(handler)
        assert registry.count == 1

    def test_register_duplicate_ignored(self):
        """Duplicate registration is ignored."""
        registry = CallbackRegistry[str]()

        def handler(data: str) -> None:
            pass

        registry.register(handler)
        registry.register(handler)
        assert registry.count == 1

    def test_unregister_callback(self):
        """Unregister a callback."""
        registry = CallbackRegistry[str]()

        def handler(data: str) -> None:
            pass

        registry.register(handler)
        assert registry.unregister(handler)
        assert registry.count == 0

    def test_unregister_nonexistent(self):
        """Unregistering nonexistent callback returns False."""
        registry = CallbackRegistry[str]()

        def handler(data: str) -> None:
            pass

        assert not registry.unregister(handler)

    def test_clear_callbacks(self):
        """Clear all callbacks."""
        registry = CallbackRegistry[str]()
        registry.register(lambda x: None)
        registry.register(lambda x: None)
        registry.clear()
        assert registry.count == 0


class TestCallbackRegistryInvocation:
    """Tests for callback invocation."""

    @pytest.mark.asyncio
    async def test_invoke_sync_callbacks(self):
        """Invoke synchronous callbacks."""
        registry = CallbackRegistry[int]()
        results = []

        def handler1(data: int) -> None:
            results.append(data * 2)

        def handler2(data: int) -> None:
            results.append(data * 3)

        registry.register(handler1)
        registry.register(handler2)
        errors = await registry.invoke_all(5)

        assert errors == []
        assert results == [10, 15]

    @pytest.mark.asyncio
    async def test_invoke_async_callbacks(self):
        """Invoke asynchronous callbacks."""
        registry = CallbackRegistry[str]()
        results = []

        async def async_handler(data: str) -> None:
            await asyncio.sleep(0.01)
            results.append(f"async:{data}")

        registry.register(async_handler)
        errors = await registry.invoke_all("test")

        assert errors == []
        assert results == ["async:test"]

    @pytest.mark.asyncio
    async def test_invoke_mixed_callbacks(self):
        """Invoke mix of sync and async callbacks."""
        registry = CallbackRegistry[str]()
        results = []

        def sync_handler(data: str) -> None:
            results.append(f"sync:{data}")

        async def async_handler(data: str) -> None:
            results.append(f"async:{data}")

        registry.register(sync_handler)
        registry.register(async_handler)
        errors = await registry.invoke_all("test")

        assert errors == []
        assert "sync:test" in results
        assert "async:test" in results

    def test_invoke_sync_only(self):
        """invoke_all_sync only invokes sync callbacks."""
        registry = CallbackRegistry[str]()
        results = []

        def sync_handler(data: str) -> None:
            results.append(f"sync:{data}")

        async def async_handler(data: str) -> None:
            results.append(f"async:{data}")

        registry.register(sync_handler)
        registry.register(async_handler)
        errors = registry.invoke_all_sync("test")

        assert errors == []
        assert results == ["sync:test"]


class TestCallbackRegistryErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_callback_error_isolated(self):
        """Errors in one callback don't affect others."""
        registry = CallbackRegistry[int]()
        results = []

        def good_handler(data: int) -> None:
            results.append(data)

        def bad_handler(data: int) -> None:
            raise ValueError("test error")

        registry.register(bad_handler)
        registry.register(good_handler)
        errors = await registry.invoke_all(42)

        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert results == [42]

    @pytest.mark.asyncio
    async def test_error_count_tracked(self):
        """Error count is tracked."""
        registry = CallbackRegistry[int]()

        def bad_handler(data: int) -> None:
            raise ValueError("test error")

        registry.register(bad_handler)
        await registry.invoke_all(1)
        await registry.invoke_all(2)

        assert registry.error_count == 2

    @pytest.mark.asyncio
    async def test_error_rate_calculated(self):
        """Error rate is calculated correctly."""
        registry = CallbackRegistry[int]()

        def bad_handler(data: int) -> None:
            if data % 2 == 0:
                raise ValueError("even error")

        registry.register(bad_handler)
        await registry.invoke_all(1)  # OK
        await registry.invoke_all(2)  # Error
        await registry.invoke_all(3)  # OK
        await registry.invoke_all(4)  # Error

        assert registry.invocation_count == 4
        assert registry.error_count == 2
        assert registry.error_rate == 0.5


class TestCallbackRegistryStats:
    """Tests for statistics."""

    @pytest.mark.asyncio
    async def test_invocation_count(self):
        """Invocation count is tracked."""
        registry = CallbackRegistry[str]()
        registry.register(lambda x: None)
        registry.register(lambda x: None)

        await registry.invoke_all("a")
        await registry.invoke_all("b")

        assert registry.invocation_count == 4  # 2 callbacks × 2 invocations

    def test_get_stats(self):
        """get_stats returns comprehensive stats."""
        registry = CallbackRegistry[str](name="events")
        registry.register(lambda x: None)

        stats = registry.get_stats()
        assert stats["name"] == "events"
        assert stats["callback_count"] == 1
        assert stats["invocation_count"] == 0
        assert stats["error_rate"] == 0.0
