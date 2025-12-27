"""Tests for UnifiedEventRouter - core event system.

Tests the unified event routing layer that consolidates:
- DataEventBus (in-memory async)
- StageEventBus (pipeline stages)
- CrossProcessEventQueue (SQLite-backed)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.event_router import (
    EventSource,
    RouterEvent,
    UnifiedEventRouter,
    _compute_content_hash,
    _generate_event_id,
    get_router,
    reset_router,
)


class TestEventId:
    """Test event ID generation."""

    def test_generates_unique_ids(self):
        """Each call should generate a unique ID."""
        ids = {_generate_event_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_is_uuid_format(self):
        """ID should be valid UUID format."""
        event_id = _generate_event_id()
        # UUID format: 8-4-4-4-12 hex characters
        assert len(event_id) == 36
        assert event_id.count("-") == 4


class TestContentHash:
    """Test content hash for deduplication."""

    def test_same_content_same_hash(self):
        """Same event type and payload should produce same hash."""
        hash1 = _compute_content_hash("TRAINING_COMPLETED", {"config": "sq8_2p"})
        hash2 = _compute_content_hash("TRAINING_COMPLETED", {"config": "sq8_2p"})
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        hash1 = _compute_content_hash("TRAINING_COMPLETED", {"config": "sq8_2p"})
        hash2 = _compute_content_hash("TRAINING_COMPLETED", {"config": "hex8_2p"})
        assert hash1 != hash2

    def test_ignores_timestamp(self):
        """Timestamp should not affect hash."""
        hash1 = _compute_content_hash("EVENT", {"data": "x", "timestamp": 1.0})
        hash2 = _compute_content_hash("EVENT", {"data": "x", "timestamp": 2.0})
        assert hash1 == hash2

    def test_ignores_source(self):
        """Source should not affect hash."""
        hash1 = _compute_content_hash("EVENT", {"data": "x", "source": "a"})
        hash2 = _compute_content_hash("EVENT", {"data": "x", "source": "b"})
        assert hash1 == hash2

    def test_hash_is_16_chars(self):
        """Hash should be truncated to 16 characters."""
        content_hash = _compute_content_hash("EVENT", {"data": "x"})
        assert len(content_hash) == 16


class TestRouterEvent:
    """Test RouterEvent dataclass."""

    def test_default_values(self):
        """RouterEvent should have sensible defaults."""
        event = RouterEvent(event_type="TEST_EVENT")
        assert event.event_type == "TEST_EVENT"
        assert event.payload == {}
        assert event.timestamp > 0
        assert event.event_id is not None
        # source is a string, origin is the EventSource
        assert event.source == ""  # Default empty string
        assert event.origin == EventSource.ROUTER

    def test_custom_values(self):
        """RouterEvent should accept custom values."""
        event = RouterEvent(
            event_type="CUSTOM",
            payload={"key": "value"},
            source="test_source",
            origin=EventSource.DATA_BUS,
            event_id="custom-id",
        )
        assert event.event_type == "CUSTOM"
        assert event.payload == {"key": "value"}
        assert event.source == "test_source"
        assert event.origin == EventSource.DATA_BUS
        assert event.event_id == "custom-id"


class TestUnifiedEventRouter:
    """Test UnifiedEventRouter core functionality."""

    def setup_method(self):
        """Reset router before each test."""
        reset_router()

    def test_singleton_pattern(self):
        """get_router should return same instance."""
        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

    def test_reset_creates_new_instance(self):
        """reset_router should create new instance."""
        router1 = get_router()
        reset_router()
        router2 = get_router()
        assert router1 is not router2

    def test_subscribe_registers_callback(self):
        """subscribe should register callback for event type."""
        router = get_router()
        callback = MagicMock()

        router.subscribe("TEST_EVENT", callback)

        assert "TEST_EVENT" in router._subscribers
        assert callback in router._subscribers["TEST_EVENT"]

    def test_unsubscribe_removes_callback(self):
        """unsubscribe should remove callback."""
        router = get_router()
        callback = MagicMock()

        router.subscribe("TEST_EVENT", callback)
        router.unsubscribe("TEST_EVENT", callback)

        assert callback not in router._subscribers.get("TEST_EVENT", [])

    def test_subscribe_global(self):
        """Global subscription (None) should receive all events."""
        router = get_router()
        callback = MagicMock()

        # Pass None for global subscription
        router.subscribe(None, callback)

        assert callback in router._global_subscribers

    @pytest.mark.asyncio
    async def test_publish_calls_subscribers(self):
        """publish should call all matching subscribers."""
        router = get_router()
        callback = AsyncMock()

        router.subscribe("TEST_EVENT", callback)
        await router.publish("TEST_EVENT", {"data": "value"})

        # Give async tasks time to complete
        await asyncio.sleep(0.2)
        callback.assert_called()

    @pytest.mark.asyncio
    async def test_publish_with_global_subscriber(self):
        """Global subscriber should receive all events."""
        router = get_router()
        callback = AsyncMock()

        # Use None for global subscription
        router.subscribe(None, callback)
        await router.publish("ANY_EVENT", {"data": "value"})

        await asyncio.sleep(0.2)
        callback.assert_called()

    def test_get_stats(self):
        """get_stats should return router statistics."""
        router = get_router()
        router.subscribe("EVENT1", MagicMock())
        router.subscribe("EVENT2", MagicMock())

        stats = router.get_stats()

        # Check actual stats keys
        assert "subscriber_count" in stats
        assert "total_events_routed" in stats
        assert "duplicates_prevented" in stats or "total_duplicates_prevented" in stats


class TestEventSource:
    """Test EventSource enum."""

    def test_event_sources(self):
        """EventSource should have expected values."""
        assert EventSource.DATA_BUS == "data_bus"
        assert EventSource.STAGE_BUS == "stage_bus"
        assert EventSource.CROSS_PROCESS == "cross_process"
        assert EventSource.ROUTER == "router"


class TestIntegration:
    """Integration tests for event flow."""

    def setup_method(self):
        """Reset router before each test."""
        reset_router()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self):
        """Multiple subscribers should all receive the event."""
        router = get_router()
        callbacks = [AsyncMock() for _ in range(3)]

        for cb in callbacks:
            router.subscribe("SHARED_EVENT", cb)

        await router.publish("SHARED_EVENT", {"test": True})
        await asyncio.sleep(0.1)

        for cb in callbacks:
            cb.assert_called()

    @pytest.mark.asyncio
    async def test_subscriber_error_doesnt_block_others(self):
        """Error in one subscriber shouldn't block others."""
        router = get_router()

        error_callback = AsyncMock(side_effect=RuntimeError("test error"))
        success_callback = AsyncMock()

        router.subscribe("EVENT", error_callback)
        router.subscribe("EVENT", success_callback)

        await router.publish("EVENT", {})
        await asyncio.sleep(0.1)

        # Success callback should still be called
        success_callback.assert_called()

    def test_sync_subscribe_and_unsubscribe(self):
        """Sync subscription should work correctly."""
        router = get_router()
        callback = MagicMock()

        router.subscribe("EVENT", callback)
        # Check using internal _subscribers dict
        assert "EVENT" in router._subscribers
        assert callback in router._subscribers["EVENT"]

        router.unsubscribe("EVENT", callback)
        # After unsubscribe, callback should not be in list
        assert callback not in router._subscribers.get("EVENT", [])
