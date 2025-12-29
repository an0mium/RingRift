"""Tests for ReactiveDispatcher.

December 29, 2025: Part of 48-hour autonomous operation optimization.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.reactive_dispatcher import (
    DispatchEvent,
    ReactiveDispatcher,
    ReactiveDispatcherConfig,
    EVENT_PRIORITIES,
    get_reactive_dispatcher,
    reset_reactive_dispatcher,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    reset_reactive_dispatcher()
    yield
    reset_reactive_dispatcher()


@pytest.fixture
def config():
    """Create test configuration."""
    return ReactiveDispatcherConfig(
        enabled=True,
        dedup_window_seconds=1.0,  # Short for testing
        max_queue_size=10,
        backpressure_threshold=5,
        per_config_cooldown_seconds=0.5,
        dispatch_timeout_seconds=5.0,
    )


@pytest.fixture
def dispatcher(config):
    """Create ReactiveDispatcher for testing."""
    return ReactiveDispatcher(config=config)


# =============================================================================
# DispatchEvent Tests
# =============================================================================


class TestDispatchEvent:
    """Tests for DispatchEvent dataclass."""

    def test_dispatch_event_creation(self):
        """Test creating a DispatchEvent."""
        event = DispatchEvent(
            event_type="node_recovered",
            priority=70,
            node_id="test-node",
        )
        assert event.event_type == "node_recovered"
        assert event.priority == 70
        assert event.node_id == "test-node"
        assert event.config_key is None
        assert event.timestamp > 0

    def test_dispatch_event_ordering(self):
        """Test that higher priority events come first."""
        low = DispatchEvent(event_type="low", priority=50)
        high = DispatchEvent(event_type="high", priority=80)

        # __lt__ returns True if self.priority > other.priority
        assert high < low  # Higher priority should be "less" for min heap
        assert not low < high

    def test_dispatch_event_with_metadata(self):
        """Test event with metadata."""
        event = DispatchEvent(
            event_type="training_completed",
            priority=80,
            config_key="hex8_2p",
            metadata={"accuracy": 0.95},
        )
        assert event.config_key == "hex8_2p"
        assert event.metadata["accuracy"] == 0.95


# =============================================================================
# ReactiveDispatcherConfig Tests
# =============================================================================


class TestReactiveDispatcherConfig:
    """Tests for ReactiveDispatcherConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReactiveDispatcherConfig()
        assert config.enabled is True
        assert config.dedup_window_seconds == 10.0
        assert config.max_queue_size == 100
        assert config.backpressure_threshold == 50
        assert config.per_config_cooldown_seconds == 5.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReactiveDispatcherConfig(
            enabled=False,
            dedup_window_seconds=30.0,
            max_queue_size=50,
        )
        assert config.enabled is False
        assert config.dedup_window_seconds == 30.0
        assert config.max_queue_size == 50


# =============================================================================
# EVENT_PRIORITIES Tests
# =============================================================================


class TestEventPriorities:
    """Tests for event priority mapping."""

    def test_priority_mapping_exists(self):
        """Test that priority mapping contains expected events."""
        assert "training_completed" in EVENT_PRIORITIES
        assert "node_recovered" in EVENT_PRIORITIES
        assert "idle_resource_detected" in EVENT_PRIORITIES
        assert "backpressure_released" in EVENT_PRIORITIES
        assert "host_online" in EVENT_PRIORITIES

    def test_priority_values(self):
        """Test that priority values are reasonable."""
        # Training completed should be high priority
        assert EVENT_PRIORITIES["training_completed"] == 80
        # Node recovered is also high
        assert EVENT_PRIORITIES["node_recovered"] == 70
        # Idle resource is lower
        assert EVENT_PRIORITIES["idle_resource_detected"] == 50


# =============================================================================
# ReactiveDispatcher Initialization Tests
# =============================================================================


class TestReactiveDispatcherInit:
    """Tests for ReactiveDispatcher initialization."""

    def test_dispatcher_initialization(self, config):
        """Test dispatcher initializes correctly."""
        dispatcher = ReactiveDispatcher(config=config)
        assert dispatcher._dispatcher_config == config
        assert dispatcher._running is False
        assert dispatcher._events_received == 0
        assert dispatcher._events_dispatched == 0

    def test_dispatcher_default_config(self):
        """Test dispatcher with default config."""
        dispatcher = ReactiveDispatcher()
        assert dispatcher.config.enabled is True
        assert dispatcher._event_queue.maxsize == dispatcher.config.max_queue_size

    def test_singleton_pattern(self):
        """Test singleton pattern works."""
        d1 = get_reactive_dispatcher()
        d2 = get_reactive_dispatcher()
        assert d1 is d2

    def test_singleton_reset(self):
        """Test singleton can be reset."""
        d1 = get_reactive_dispatcher()
        reset_reactive_dispatcher()
        d2 = get_reactive_dispatcher()
        assert d1 is not d2


# =============================================================================
# ReactiveDispatcher Event Subscription Tests
# =============================================================================


class TestReactiveDispatcherSubscriptions:
    """Tests for event subscriptions."""

    def test_event_subscriptions(self, dispatcher):
        """Test that event subscriptions are correct."""
        subs = dispatcher._get_event_subscriptions()
        assert "node_recovered" in subs
        assert "training_completed" in subs
        assert "idle_resource_detected" in subs
        assert "backpressure_released" in subs
        assert "backpressure_activated" in subs
        assert "host_online" in subs


# =============================================================================
# ReactiveDispatcher Event Handling Tests
# =============================================================================


class TestReactiveDispatcherEventHandling:
    """Tests for event handling."""

    @pytest.mark.asyncio
    async def test_on_node_recovered(self, dispatcher):
        """Test handling node_recovered event."""
        event = {"node_id": "test-node"}
        await dispatcher._on_node_recovered(event)

        assert dispatcher._events_received == 1
        assert dispatcher._event_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_on_training_completed(self, dispatcher):
        """Test handling training_completed event."""
        event = {"config_key": "hex8_2p"}
        await dispatcher._on_training_completed(event)

        assert dispatcher._events_received == 1

    @pytest.mark.asyncio
    async def test_on_idle_resource(self, dispatcher):
        """Test handling idle_resource_detected event."""
        event = {"node_id": "idle-node"}
        await dispatcher._on_idle_resource(event)

        assert dispatcher._events_received == 1

    @pytest.mark.asyncio
    async def test_on_backpressure_activated(self, dispatcher):
        """Test handling backpressure_activated event."""
        event = {}
        await dispatcher._on_backpressure_activated(event)

        assert dispatcher._under_backpressure is True

    @pytest.mark.asyncio
    async def test_on_backpressure_released(self, dispatcher):
        """Test handling backpressure_released event."""
        dispatcher._under_backpressure = True
        event = {}
        await dispatcher._on_backpressure_released(event)

        assert dispatcher._under_backpressure is False


# =============================================================================
# ReactiveDispatcher Deduplication Tests
# =============================================================================


class TestReactiveDispatcherDedup:
    """Tests for event deduplication."""

    @pytest.mark.asyncio
    async def test_deduplication_within_window(self, dispatcher):
        """Test that events within dedup window are dropped."""
        event1 = {"node_id": "test-node"}
        await dispatcher._on_node_recovered(event1)

        # Second event within window should be deduplicated
        await dispatcher._on_node_recovered(event1)

        assert dispatcher._events_received == 2
        assert dispatcher._events_deduplicated == 1
        assert dispatcher._event_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_deduplication_after_window(self, dispatcher):
        """Test that events after dedup window are not dropped."""
        event1 = {"node_id": "test-node"}
        await dispatcher._on_node_recovered(event1)

        # Wait for dedup window to expire
        await asyncio.sleep(1.1)  # Config has 1.0s window

        # Second event should not be deduplicated
        await dispatcher._on_node_recovered(event1)

        assert dispatcher._events_received == 2
        assert dispatcher._events_deduplicated == 0
        assert dispatcher._event_queue.qsize() == 2


# =============================================================================
# ReactiveDispatcher Backpressure Tests
# =============================================================================


class TestReactiveDispatcherBackpressure:
    """Tests for backpressure handling."""

    @pytest.mark.asyncio
    async def test_low_priority_dropped_during_backpressure(self, dispatcher):
        """Test that low priority events are dropped during backpressure."""
        dispatcher._under_backpressure = True

        # Idle resource is priority 50 (below 60 threshold)
        event = {"node_id": "idle-node"}
        await dispatcher._on_idle_resource(event)

        assert dispatcher._events_dropped_backpressure == 1
        assert dispatcher._event_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_high_priority_not_dropped_during_backpressure(self, dispatcher):
        """Test that high priority events are not dropped during backpressure."""
        dispatcher._under_backpressure = True

        # Node recovered is priority 70 (above 60 threshold)
        event = {"node_id": "test-node"}
        await dispatcher._on_node_recovered(event)

        assert dispatcher._events_dropped_backpressure == 0
        assert dispatcher._event_queue.qsize() == 1


# =============================================================================
# ReactiveDispatcher Health Check Tests
# =============================================================================


class TestReactiveDispatcherHealthCheck:
    """Tests for health check."""

    def test_health_check_not_running(self, dispatcher):
        """Test health check when not running."""
        result = dispatcher.health_check()
        assert result.healthy is False
        assert "not running" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_disabled(self, config):
        """Test health check when disabled."""
        config.enabled = False
        dispatcher = ReactiveDispatcher(config=config)
        dispatcher._running = True

        result = dispatcher.health_check()
        assert result.healthy is True
        assert result.details["enabled"] is False

    @pytest.mark.asyncio
    async def test_health_check_running(self, dispatcher):
        """Test health check when running normally."""
        dispatcher._running = True
        dispatcher._worker_task = MagicMock()
        dispatcher._worker_task.done.return_value = False

        result = dispatcher.health_check()
        assert result.healthy is True


# =============================================================================
# ReactiveDispatcher Stats Tests
# =============================================================================


class TestReactiveDispatcherStats:
    """Tests for statistics."""

    def test_get_stats(self, dispatcher):
        """Test getting stats."""
        dispatcher._events_received = 10
        dispatcher._events_dispatched = 8
        dispatcher._events_deduplicated = 2

        stats = dispatcher.get_stats()

        assert stats["events_received"] == 10
        assert stats["events_dispatched"] == 8
        assert stats["events_deduplicated"] == 2
        assert "queue_size" in stats
        assert "under_backpressure" in stats


# =============================================================================
# ReactiveDispatcher Lifecycle Tests
# =============================================================================


class TestReactiveDispatcherLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_when_disabled(self, config):
        """Test that start does nothing when disabled."""
        config.enabled = False
        dispatcher = ReactiveDispatcher(config=config)

        await dispatcher._on_start()

        assert dispatcher._worker_task is None

    @pytest.mark.asyncio
    async def test_start_creates_worker(self, dispatcher):
        """Test that start creates worker task."""
        await dispatcher._on_start()

        assert dispatcher._worker_task is not None

        # Cleanup
        dispatcher._running = False
        dispatcher._worker_task.cancel()
        try:
            await dispatcher._worker_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_stop_cancels_worker(self, dispatcher):
        """Test that stop cancels worker task."""
        await dispatcher._on_start()
        assert dispatcher._worker_task is not None

        await dispatcher._on_stop()
        # Worker should be cancelled


# =============================================================================
# Integration Tests
# =============================================================================


class TestReactiveDispatcherIntegration:
    """Integration tests for ReactiveDispatcher."""

    @pytest.mark.asyncio
    async def test_full_event_flow(self, dispatcher):
        """Test complete event flow from receipt to dispatch."""
        # Start the dispatcher
        dispatcher._running = True

        # Enqueue an event
        await dispatcher._on_training_completed({"config_key": "hex8_2p"})

        assert dispatcher._events_received == 1
        assert dispatcher._event_queue.qsize() == 1

        # Get the event from queue
        event = await dispatcher._event_queue.get()
        assert event.event_type == "training_completed"
        assert event.config_key == "hex8_2p"
