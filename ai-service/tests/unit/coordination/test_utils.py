"""Tests for app.coordination.utils module.

Tests for BoundedHistory, MetricsAccumulator, and CallbackRegistry classes.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import Mock, patch

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

    def test_creates_with_defaults(self):
        """Test creating entry with default timestamp."""
        entry = HistoryEntry(value="test")
        assert entry.value == "test"
        assert entry.timestamp > 0
        assert entry.metadata == {}

    def test_creates_with_metadata(self):
        """Test creating entry with metadata."""
        entry = HistoryEntry(value=42, metadata={"key": "value"})
        assert entry.value == 42
        assert entry.metadata == {"key": "value"}

    def test_generic_type(self):
        """Test that HistoryEntry works with different types."""
        str_entry: HistoryEntry[str] = HistoryEntry(value="string")
        int_entry: HistoryEntry[int] = HistoryEntry(value=123)
        list_entry: HistoryEntry[list] = HistoryEntry(value=[1, 2, 3])

        assert str_entry.value == "string"
        assert int_entry.value == 123
        assert list_entry.value == [1, 2, 3]


# =============================================================================
# BoundedHistory Tests
# =============================================================================


class TestBoundedHistory:
    """Tests for BoundedHistory class."""

    def test_init_defaults(self):
        """Test default initialization."""
        history: BoundedHistory[int] = BoundedHistory()
        assert history.max_size == 100
        assert history.track_timestamps is True
        assert len(history) == 0

    def test_init_custom_size(self):
        """Test custom max_size."""
        history: BoundedHistory[str] = BoundedHistory(max_size=10)
        assert history.max_size == 10

    def test_append_and_len(self):
        """Test appending items and length."""
        history: BoundedHistory[int] = BoundedHistory(max_size=5)
        history.append(1)
        history.append(2)
        history.append(3)
        assert len(history) == 3

    def test_append_with_metadata(self):
        """Test appending with metadata."""
        history: BoundedHistory[str] = BoundedHistory(max_size=5)
        history.append("event", source="test", priority=1)
        entries = history.get_entries()
        assert len(entries) == 1
        assert entries[0].metadata == {"source": "test", "priority": 1}

    def test_bounded_eviction(self):
        """Test that old items are evicted when max_size reached."""
        history: BoundedHistory[int] = BoundedHistory(max_size=3)
        for i in range(5):
            history.append(i)

        assert len(history) == 3
        assert history.get_all() == [2, 3, 4]
        assert history.total_added == 5

    def test_extend(self):
        """Test extending with multiple items."""
        history: BoundedHistory[int] = BoundedHistory(max_size=10)
        history.extend([1, 2, 3, 4, 5])
        assert len(history) == 5
        assert history.get_all() == [1, 2, 3, 4, 5]

    def test_clear(self):
        """Test clearing history."""
        history: BoundedHistory[int] = BoundedHistory()
        history.extend([1, 2, 3])
        history.clear()
        assert len(history) == 0
        assert history.get_all() == []

    def test_get_recent(self):
        """Test getting recent items."""
        history: BoundedHistory[int] = BoundedHistory()
        history.extend([1, 2, 3, 4, 5])
        assert history.get_recent(3) == [3, 4, 5]
        assert history.get_recent(10) == [1, 2, 3, 4, 5]  # More than available
        assert history.get_recent(0) == []
        assert history.get_recent(-1) == []

    def test_get_oldest(self):
        """Test getting oldest items."""
        history: BoundedHistory[int] = BoundedHistory()
        history.extend([1, 2, 3, 4, 5])
        assert history.get_oldest(3) == [1, 2, 3]
        assert history.get_oldest(10) == [1, 2, 3, 4, 5]

    def test_oldest_newest_properties(self):
        """Test oldest and newest properties."""
        history: BoundedHistory[int] = BoundedHistory()
        assert history.oldest is None
        assert history.newest is None

        history.append(1)
        assert history.oldest == 1
        assert history.newest == 1

        history.append(2)
        history.append(3)
        assert history.oldest == 1
        assert history.newest == 3

    def test_timestamp_properties(self):
        """Test timestamp properties."""
        history: BoundedHistory[int] = BoundedHistory()
        assert history.oldest_timestamp is None
        assert history.newest_timestamp is None

        history.append(1)
        ts1 = history.oldest_timestamp
        assert ts1 is not None
        assert ts1 == history.newest_timestamp

        time.sleep(0.01)  # Small delay
        history.append(2)
        assert history.oldest_timestamp == ts1
        assert history.newest_timestamp > ts1

    def test_filter(self):
        """Test filtering items."""
        history: BoundedHistory[int] = BoundedHistory()
        history.extend([1, 2, 3, 4, 5, 6])
        evens = history.filter(lambda x: x % 2 == 0)
        assert evens == [2, 4, 6]

    def test_filter_by_time(self):
        """Test filtering by timestamp."""
        history: BoundedHistory[int] = BoundedHistory()
        history.append(1)
        time.sleep(0.05)
        mid_time = time.time()
        time.sleep(0.05)
        history.append(2)
        history.append(3)

        # Items after mid_time
        recent = history.filter_by_time(since=mid_time)
        assert recent == [2, 3]

        # Items before mid_time
        old = history.filter_by_time(until=mid_time)
        assert old == [1]

    def test_count_matching(self):
        """Test counting matching items."""
        history: BoundedHistory[int] = BoundedHistory()
        history.extend([1, 2, 3, 4, 5, 6])
        assert history.count_matching(lambda x: x > 3) == 3
        assert history.count_matching(lambda x: x == 10) == 0

    def test_is_full(self):
        """Test is_full property."""
        history: BoundedHistory[int] = BoundedHistory(max_size=3)
        assert not history.is_full
        history.extend([1, 2])
        assert not history.is_full
        history.append(3)
        assert history.is_full

    def test_iteration(self):
        """Test iteration over items."""
        history: BoundedHistory[int] = BoundedHistory()
        history.extend([1, 2, 3])
        items = list(history)
        assert items == [1, 2, 3]

    def test_bool(self):
        """Test boolean evaluation."""
        history: BoundedHistory[int] = BoundedHistory()
        assert not history
        history.append(1)
        assert history

    def test_contains(self):
        """Test membership testing."""
        history: BoundedHistory[int] = BoundedHistory()
        history.extend([1, 2, 3])
        assert 2 in history
        assert 10 not in history

    def test_no_timestamps(self):
        """Test with timestamp tracking disabled."""
        history: BoundedHistory[int] = BoundedHistory(track_timestamps=False)
        history.append(1)
        entries = history.get_entries()
        assert entries[0].timestamp == 0.0


# =============================================================================
# MetricsSnapshot Tests
# =============================================================================


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_creates_with_values(self):
        """Test creating snapshot with values."""
        snapshot = MetricsSnapshot(
            count=100,
            total=250.0,
            mean=2.5,
            min_value=1.0,
            max_value=5.0,
            std_dev=0.5,
            recent_mean=2.6,
            trend=0.01,
        )
        assert snapshot.count == 100
        assert snapshot.mean == 2.5
        assert snapshot.trend == 0.01
        assert snapshot.timestamp > 0


# =============================================================================
# MetricsAccumulator Tests
# =============================================================================


class TestMetricsAccumulator:
    """Tests for MetricsAccumulator class."""

    def test_init_defaults(self):
        """Test default initialization."""
        metrics = MetricsAccumulator()
        assert metrics.window_size == 100
        assert metrics.name == "metric"
        assert metrics.higher_is_better is False
        assert metrics.count == 0

    def test_init_custom(self):
        """Test custom initialization."""
        metrics = MetricsAccumulator(
            window_size=50,
            name="loss",
            higher_is_better=True,
        )
        assert metrics.window_size == 50
        assert metrics.name == "loss"
        assert metrics.higher_is_better is True

    def test_add_and_count(self):
        """Test adding values and counting."""
        metrics = MetricsAccumulator()
        metrics.add(1.0)
        metrics.add(2.0)
        metrics.add(3.0)
        assert metrics.count == 3
        assert metrics.window_count == 3

    def test_add_batch(self):
        """Test adding multiple values at once."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.count == 5

    def test_mean(self):
        """Test mean calculation."""
        metrics = MetricsAccumulator()
        metrics.add_batch([2.0, 4.0, 6.0, 8.0])
        assert metrics.mean == 5.0

    def test_mean_empty(self):
        """Test mean when empty."""
        metrics = MetricsAccumulator()
        assert metrics.mean == 0.0

    def test_window_mean(self):
        """Test window mean vs overall mean."""
        metrics = MetricsAccumulator(window_size=3)
        # Add 5 values, window only keeps last 3
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.mean == 3.0  # (1+2+3+4+5)/5
        assert metrics.window_mean == 4.0  # (3+4+5)/3

    def test_min_max(self):
        """Test min and max tracking."""
        metrics = MetricsAccumulator()
        assert metrics.min_value is None
        assert metrics.max_value is None

        metrics.add_batch([3.0, 1.0, 4.0, 1.5, 9.2])
        assert metrics.min_value == 1.0
        assert metrics.max_value == 9.2

    def test_best_value_lower_is_better(self):
        """Test best value when lower is better."""
        metrics = MetricsAccumulator(higher_is_better=False)
        metrics.add_batch([3.0, 1.0, 2.0])
        assert metrics.best_value == 1.0

    def test_best_value_higher_is_better(self):
        """Test best value when higher is better."""
        metrics = MetricsAccumulator(higher_is_better=True)
        metrics.add_batch([3.0, 1.0, 2.0])
        assert metrics.best_value == 3.0

    def test_std_dev(self):
        """Test standard deviation calculation."""
        metrics = MetricsAccumulator()
        metrics.add_batch([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        # Population std dev would be 2.0, sample std dev is ~2.14
        assert 2.0 < metrics.std_dev < 2.2

    def test_std_dev_insufficient_data(self):
        """Test std_dev with insufficient data."""
        metrics = MetricsAccumulator()
        assert metrics.std_dev == 0.0
        metrics.add(1.0)
        assert metrics.std_dev == 0.0

    def test_variance(self):
        """Test variance calculation."""
        metrics = MetricsAccumulator()
        metrics.add_batch([2.0, 4.0, 6.0])
        assert metrics.variance > 0
        assert abs(metrics.std_dev - metrics.variance ** 0.5) < 0.0001

    def test_trend_increasing(self):
        """Test trend calculation for increasing values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.trend > 0

    def test_trend_decreasing(self):
        """Test trend calculation for decreasing values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([5.0, 4.0, 3.0, 2.0, 1.0])
        assert metrics.trend < 0

    def test_trend_flat(self):
        """Test trend calculation for flat values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([3.0, 3.0, 3.0, 3.0, 3.0])
        assert abs(metrics.trend) < 0.001

    def test_trend_insufficient_data(self):
        """Test trend with insufficient data."""
        metrics = MetricsAccumulator()
        assert metrics.trend == 0.0
        metrics.add(1.0)
        assert metrics.trend == 0.0

    def test_is_improving_lower_better(self):
        """Test is_improving when lower is better."""
        metrics = MetricsAccumulator(higher_is_better=False)
        metrics.add_batch([5.0, 4.0, 3.0, 2.0, 1.0])  # Decreasing
        assert metrics.is_improving is True

    def test_is_improving_higher_better(self):
        """Test is_improving when higher is better."""
        metrics = MetricsAccumulator(higher_is_better=True)
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])  # Increasing
        assert metrics.is_improving is True

    def test_current(self):
        """Test current value."""
        metrics = MetricsAccumulator()
        assert metrics.current is None
        metrics.add(1.0)
        assert metrics.current == 1.0
        metrics.add(2.0)
        assert metrics.current == 2.0

    def test_get_recent(self):
        """Test getting recent values."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.get_recent(3) == [3.0, 4.0, 5.0]

    def test_reset(self):
        """Test full reset."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0])
        metrics.reset()
        assert metrics.count == 0
        assert metrics.window_count == 0
        assert metrics.min_value is None
        assert metrics.max_value is None
        assert metrics.best_value is None

    def test_reset_window(self):
        """Test window-only reset."""
        metrics = MetricsAccumulator()
        metrics.add_batch([1.0, 2.0, 3.0])
        total_before = metrics.total
        metrics.reset_window()
        assert metrics.count == 3  # Total count preserved
        assert metrics.total == total_before
        assert metrics.window_count == 0

    def test_get_snapshot(self):
        """Test snapshot creation."""
        metrics = MetricsAccumulator(name="test")
        metrics.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        snapshot = metrics.get_snapshot()
        assert isinstance(snapshot, MetricsSnapshot)
        assert snapshot.count == 5
        assert snapshot.mean == 3.0
        assert snapshot.min_value == 1.0
        assert snapshot.max_value == 5.0

    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = MetricsAccumulator(name="loss")
        metrics.add_batch([1.0, 2.0, 3.0])
        d = metrics.to_dict()
        assert d["name"] == "loss"
        assert d["count"] == 3
        assert "mean" in d
        assert "trend" in d
        assert "is_improving" in d


# =============================================================================
# CallbackRegistry Tests
# =============================================================================


class TestCallbackRegistry:
    """Tests for CallbackRegistry class."""

    def test_init_defaults(self):
        """Test default initialization."""
        registry: CallbackRegistry[str] = CallbackRegistry()
        assert registry.name == "callbacks"
        assert registry.count == 0

    def test_init_custom_name(self):
        """Test custom name."""
        registry: CallbackRegistry[str] = CallbackRegistry(name="events")
        assert registry.name == "events"

    def test_register(self):
        """Test registering callbacks."""

        def handler(data: str) -> None:
            pass

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(handler)
        assert registry.count == 1

    def test_register_duplicate_ignored(self):
        """Test that duplicate registration is ignored."""

        def handler(data: str) -> None:
            pass

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(handler)
        registry.register(handler)
        assert registry.count == 1

    def test_unregister(self):
        """Test unregistering callbacks."""

        def handler(data: str) -> None:
            pass

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(handler)
        result = registry.unregister(handler)
        assert result is True
        assert registry.count == 0

    def test_unregister_not_found(self):
        """Test unregistering non-existent callback."""

        def handler(data: str) -> None:
            pass

        registry: CallbackRegistry[str] = CallbackRegistry()
        result = registry.unregister(handler)
        assert result is False

    def test_clear(self):
        """Test clearing all callbacks."""

        def handler1(data: str) -> None:
            pass

        def handler2(data: str) -> None:
            pass

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(handler1)
        registry.register(handler2)
        registry.clear()
        assert registry.count == 0

    @pytest.mark.asyncio
    async def test_invoke_all_sync_handlers(self):
        """Test invoking sync handlers."""
        results = []

        def handler1(data: str) -> None:
            results.append(f"h1:{data}")

        def handler2(data: str) -> None:
            results.append(f"h2:{data}")

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(handler1)
        registry.register(handler2)

        errors = await registry.invoke_all("test")
        assert errors == []
        assert results == ["h1:test", "h2:test"]

    @pytest.mark.asyncio
    async def test_invoke_all_async_handlers(self):
        """Test invoking async handlers."""
        results = []

        async def async_handler(data: str) -> None:
            await asyncio.sleep(0.001)
            results.append(f"async:{data}")

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(async_handler)

        errors = await registry.invoke_all("test")
        assert errors == []
        assert results == ["async:test"]

    @pytest.mark.asyncio
    async def test_invoke_all_mixed_handlers(self):
        """Test invoking mixed sync and async handlers."""
        results = []

        def sync_handler(data: int) -> None:
            results.append(data * 2)

        async def async_handler(data: int) -> None:
            await asyncio.sleep(0.001)
            results.append(data * 3)

        registry: CallbackRegistry[int] = CallbackRegistry()
        registry.register(sync_handler)
        registry.register(async_handler)

        errors = await registry.invoke_all(5)
        assert errors == []
        assert 10 in results  # sync: 5*2
        assert 15 in results  # async: 5*3

    @pytest.mark.asyncio
    async def test_invoke_all_error_handling(self):
        """Test that errors are captured but don't stop other handlers."""
        results = []

        def failing_handler(data: str) -> None:
            raise ValueError("test error")

        def succeeding_handler(data: str) -> None:
            results.append(data)

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(failing_handler)
        registry.register(succeeding_handler)

        errors = await registry.invoke_all("test")
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert results == ["test"]  # Succeeding handler still ran

    def test_invoke_all_sync_method(self):
        """Test sync-only invocation method."""
        results = []

        def sync_handler(data: str) -> None:
            results.append(f"sync:{data}")

        async def async_handler(data: str) -> None:
            results.append(f"async:{data}")

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(sync_handler)
        registry.register(async_handler)

        errors = registry.invoke_all_sync("test")
        assert errors == []
        assert results == ["sync:test"]  # Async handler skipped

    def test_invoke_all_sync_error_handling(self):
        """Test sync-only invocation error handling."""

        def failing_handler(data: str) -> None:
            raise RuntimeError("sync error")

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(failing_handler)

        errors = registry.invoke_all_sync("test")
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    @pytest.mark.asyncio
    async def test_invocation_count(self):
        """Test invocation counting."""

        def handler(data: str) -> None:
            pass

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(handler)

        assert registry.invocation_count == 0
        await registry.invoke_all("test")
        assert registry.invocation_count == 1
        await registry.invoke_all("test2")
        assert registry.invocation_count == 2

    @pytest.mark.asyncio
    async def test_error_count(self):
        """Test error counting."""

        def failing_handler(data: str) -> None:
            raise ValueError("error")

        registry: CallbackRegistry[str] = CallbackRegistry()
        registry.register(failing_handler)

        assert registry.error_count == 0
        await registry.invoke_all("test")
        assert registry.error_count == 1
        await registry.invoke_all("test2")
        assert registry.error_count == 2

    @pytest.mark.asyncio
    async def test_error_rate(self):
        """Test error rate calculation."""

        call_count = 0

        def sometimes_failing(data: int) -> None:
            nonlocal call_count
            call_count += 1
            if data < 0:
                raise ValueError("negative")

        registry: CallbackRegistry[int] = CallbackRegistry()
        registry.register(sometimes_failing)

        # 2 successes, 2 failures
        await registry.invoke_all(1)
        await registry.invoke_all(2)
        await registry.invoke_all(-1)
        await registry.invoke_all(-2)

        assert registry.invocation_count == 4
        assert registry.error_count == 2
        assert registry.error_rate == 0.5

    def test_error_rate_empty(self):
        """Test error rate when no invocations."""
        registry: CallbackRegistry[str] = CallbackRegistry()
        assert registry.error_rate == 0.0

    def test_get_stats(self):
        """Test getting registry statistics."""
        registry: CallbackRegistry[str] = CallbackRegistry(name="test_registry")

        def handler(data: str) -> None:
            pass

        registry.register(handler)
        stats = registry.get_stats()

        assert stats["name"] == "test_registry"
        assert stats["callback_count"] == 1
        assert stats["invocation_count"] == 0
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0.0

    def test_generic_types(self):
        """Test that CallbackRegistry works with various types."""
        # Dict type
        dict_results = []

        def dict_handler(data: dict) -> None:
            dict_results.append(data)

        dict_registry: CallbackRegistry[dict] = CallbackRegistry()
        dict_registry.register(dict_handler)
        dict_registry.invoke_all_sync({"key": "value"})
        assert dict_results == [{"key": "value"}]

        # Custom object type
        class Event:
            def __init__(self, name: str):
                self.name = name

        event_results = []

        def event_handler(event: Event) -> None:
            event_results.append(event.name)

        event_registry: CallbackRegistry[Event] = CallbackRegistry()
        event_registry.register(event_handler)
        event_registry.invoke_all_sync(Event("test_event"))
        assert event_results == ["test_event"]
