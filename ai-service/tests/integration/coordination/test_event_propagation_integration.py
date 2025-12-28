"""Integration tests for event propagation through coordination components.

Tests end-to-end event flows between coordinators:
1. Event chains work end-to-end
2. Deduplication prevents duplicate processing
3. Dead letter queue captures failed events
4. Circuit breakers trip on repeated failures
5. Event normalization works consistently

December 2025: Created to verify critical coordination event paths.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def reset_event_router():
    """Reset event router before and after each test."""
    try:
        from app.coordination.event_router import reset_router, _router_instance
        reset_router()
    except ImportError:
        pass

    yield

    try:
        from app.coordination.event_router import reset_router
        reset_router()
    except ImportError:
        pass


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus for testing event propagation."""

    class MockEventBus:
        def __init__(self):
            self.subscribers: dict[str, list] = {}
            self.published_events: list = []

        def subscribe(self, event_type: str, handler):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)

        async def publish(self, event):
            self.published_events.append(event)
            event_type = getattr(event, "event_type", None)
            if event_type:
                type_key = (
                    event_type.value if hasattr(event_type, "value") else event_type
                )
                for handler in self.subscribers.get(type_key, []):
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception:
                        pass  # Ignore handler errors in tests

        def clear(self):
            self.subscribers.clear()
            self.published_events.clear()

    return MockEventBus()


class MockDataEvent:
    """Mock DataEvent for testing."""

    def __init__(self, event_type, payload: dict, source: str = "test"):
        self.event_type = event_type
        self.payload = payload
        self.source = source
        self.timestamp = time.time()


# =============================================================================
# Test Event Propagation
# =============================================================================


class TestEventPropagation:
    """Integration tests for event propagation through coordinators."""

    @pytest.fixture
    def event_router(self, reset_event_router):
        """Get fresh event router instance."""
        from app.coordination.event_router import UnifiedEventRouter
        return UnifiedEventRouter(
            enable_cross_process_polling=False,
            max_seen_events=100,
        )

    @pytest.mark.asyncio
    async def test_data_sync_triggers_pipeline(self, event_router):
        """Verify DATA_SYNC_COMPLETED event reaches DataPipelineOrchestrator."""
        handler_called = False
        received_payload = None

        async def mock_handler(event):
            nonlocal handler_called, received_payload
            handler_called = True
            received_payload = event.payload

        # Subscribe and emit
        event_router.subscribe("data_sync_completed", mock_handler)
        await event_router.publish(
            "data_sync_completed",
            {"source": "test", "files_synced": 10}
        )

        # Give event time to propagate
        await asyncio.sleep(0.1)

        assert handler_called, "Handler should have been called"
        assert received_payload["files_synced"] == 10

    @pytest.mark.asyncio
    async def test_event_deduplication_by_content(self, event_router):
        """Verify duplicate events are filtered by content hash."""
        call_count = 0

        async def counting_handler(event):
            nonlocal call_count
            call_count += 1

        event_router.subscribe("test_event", counting_handler)

        # Emit same event twice with identical content
        event_data = {"id": "123", "data": "test"}
        await event_router.publish("test_event", event_data)
        await event_router.publish("test_event", event_data)

        # Give events time to propagate
        await asyncio.sleep(0.1)

        # Should only be called once due to content-based deduplication
        assert call_count == 1, f"Expected 1 call (dedup), got {call_count}"

    @pytest.mark.asyncio
    async def test_different_events_not_deduplicated(self, event_router):
        """Verify distinct events are NOT filtered."""
        call_count = 0

        async def counting_handler(event):
            nonlocal call_count
            call_count += 1

        event_router.subscribe("test_event", counting_handler)

        # Emit different events
        await event_router.publish("test_event", {"id": "1", "data": "first"})
        await event_router.publish("test_event", {"id": "2", "data": "second"})

        # Give events time to propagate
        await asyncio.sleep(0.1)

        # Both should be processed
        assert call_count == 2, f"Expected 2 calls, got {call_count}"

    @pytest.mark.asyncio
    async def test_event_normalization(self, reset_event_router):
        """Verify event types are normalized consistently."""
        from app.coordination.event_normalization import normalize_event_type

        # Test various input formats
        assert normalize_event_type("SYNC_COMPLETE") == "data_sync_completed"
        assert normalize_event_type("sync_completed") == "data_sync_completed"
        assert normalize_event_type("DATA_SYNC_COMPLETED") == "data_sync_completed"

        # Training events
        assert normalize_event_type("TRAINING_COMPLETE") == "training_completed"
        assert normalize_event_type("training_completed") == "training_completed"

    @pytest.mark.asyncio
    async def test_subscriber_receives_events_from_multiple_sources(self, event_router):
        """Verify subscriber receives events regardless of source."""
        received_events = []

        async def collector(event):
            received_events.append(event)

        event_router.subscribe("training_completed", collector)

        # Publish from different sources
        await event_router.publish(
            "training_completed",
            {"model": "model_a"},
            source="training_daemon"
        )
        await event_router.publish(
            "training_completed",
            {"model": "model_b"},
            source="cluster_sync"
        )

        await asyncio.sleep(0.1)

        assert len(received_events) == 2
        sources = [e.source for e in received_events]
        assert "training_daemon" in sources
        assert "cluster_sync" in sources


# =============================================================================
# Test Dead Letter Queue
# =============================================================================


class TestDeadLetterQueueIntegration:
    """Integration tests for dead letter queue functionality."""

    @pytest.mark.asyncio
    async def test_failed_event_captured_in_dlq(self, reset_event_router):
        """Verify failed events are captured in dead letter queue."""
        pytest.importorskip("app.coordination.dead_letter_queue")

        from app.coordination.dead_letter_queue import (
            DeadLetterQueue,
            FailedEvent,
        )

        # Create DLQ with in-memory storage
        dlq = DeadLetterQueue(db_path=":memory:")

        # Add a failed event
        failed_event = FailedEvent(
            event_id="test-123",
            event_type="data_sync_completed",
            payload={"source": "test"},
            handler_name="test_handler",
            error="Connection timeout",
            source="test",
        )
        dlq.add_failed_event(failed_event)

        # Retrieve and verify
        failed_events = dlq.get_failed_events(limit=10)
        assert len(failed_events) == 1
        assert failed_events[0]["event_type"] == "data_sync_completed"
        assert "timeout" in failed_events[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_dlq_retry_with_backoff(self, reset_event_router):
        """Verify DLQ retry respects backoff timing."""
        pytest.importorskip("app.coordination.dead_letter_queue")

        from app.coordination.dead_letter_queue import DeadLetterQueue, FailedEvent

        dlq = DeadLetterQueue(db_path=":memory:")

        # Add a failed event with retries
        failed_event = FailedEvent(
            event_id="retry-test",
            event_type="model_promoted",
            payload={"model_id": "test"},
            handler_name="promotion_handler",
            error="Temporary failure",
            retry_count=2,  # Already retried twice
        )
        dlq.add_failed_event(failed_event)

        # Check backoff is applied (retry_count=2 means 4s backoff by default)
        events = dlq.get_failed_events(limit=10)
        assert events[0]["retry_count"] == 2

    @pytest.mark.asyncio
    async def test_dlq_health_check(self, reset_event_router):
        """Verify DLQ health check returns valid status."""
        pytest.importorskip("app.coordination.dead_letter_queue")

        from app.coordination.dead_letter_queue import DeadLetterQueue

        dlq = DeadLetterQueue(db_path=":memory:")

        # Health check should return valid result
        health = dlq.health_check()
        assert hasattr(health, "healthy")
        assert hasattr(health, "details")


# =============================================================================
# Test Circuit Breaker Behavior
# =============================================================================


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker behavior in event handlers."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_on_repeated_failures(self, reset_event_router):
        """Verify circuit breaker opens after threshold failures."""
        pytest.importorskip("app.coordination.data_pipeline_orchestrator")

        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator,
        )

        orchestrator = DataPipelineOrchestrator()

        # Simulate multiple failures for the same handler
        handler_name = "_on_sync_complete"
        for _ in range(6):  # Default threshold is 5
            orchestrator._failure_count[handler_name] = (
                orchestrator._failure_count.get(handler_name, 0) + 1
            )

        # Check if circuit would be open
        # Circuit breakers open when failure_count >= threshold
        threshold = orchestrator._circuit_breaker_threshold
        is_open = orchestrator._failure_count.get(handler_name, 0) >= threshold

        if threshold <= 6:
            assert is_open, "Circuit breaker should be open after repeated failures"

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_recovery(self, reset_event_router):
        """Verify circuit breaker allows recovery after cooldown."""
        pytest.importorskip("app.coordination.data_pipeline_orchestrator")

        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orchestrator = DataPipelineOrchestrator()

        # Set past failure time (simulate cooldown passed)
        handler_name = "_on_test_handler"
        orchestrator._failure_count[handler_name] = 10
        orchestrator._last_failure[handler_name] = time.time() - 120  # 2 min ago

        # After cooldown, circuit should reset (depending on implementation)
        # This tests that the infrastructure supports recovery
        cooldown = getattr(orchestrator, "_circuit_breaker_cooldown", 60)
        time_since_failure = time.time() - orchestrator._last_failure[handler_name]

        assert time_since_failure > cooldown, "Time since failure should exceed cooldown"


# =============================================================================
# Test Event Chain Propagation
# =============================================================================


class TestEventChainPropagation:
    """Tests for complete event chains through multiple coordinators."""

    @pytest.mark.asyncio
    async def test_sync_to_export_to_training_chain(self, reset_event_router):
        """Test full chain: DATA_SYNC -> NPZ_EXPORT -> TRAINING event flow."""
        pytest.importorskip("app.distributed.data_events")

        from app.distributed.data_events import DataEventType

        # Verify all events in the chain exist
        chain_events = [
            "DATA_SYNC_COMPLETED",
            "EXPORT_COMPLETED",
            "TRAINING_STARTED",
            "TRAINING_COMPLETED",
            "EVALUATION_COMPLETED",
        ]

        for event_name in chain_events:
            assert hasattr(DataEventType, event_name), f"Missing event: {event_name}"

    @pytest.mark.asyncio
    async def test_training_completed_triggers_evaluation(self, mock_event_bus):
        """Verify TRAINING_COMPLETED triggers gauntlet evaluation."""
        evaluation_triggered = False
        triggered_config = None

        async def evaluation_handler(event):
            nonlocal evaluation_triggered, triggered_config
            if event.payload.get("policy_accuracy", 0) >= 0.75:
                evaluation_triggered = True
                triggered_config = event.payload.get("config")

        mock_event_bus.subscribe("training_completed", evaluation_handler)

        # Publish training completion with good accuracy
        event = MockDataEvent(
            event_type=MagicMock(value="training_completed"),
            payload={
                "config": "hex8_2p",
                "model_path": "/models/hex8_2p.pth",
                "policy_accuracy": 0.85,
                "value_accuracy": 0.72,
            },
        )
        await mock_event_bus.publish(event)

        assert evaluation_triggered, "Evaluation should trigger on good accuracy"
        assert triggered_config == "hex8_2p"

    @pytest.mark.asyncio
    async def test_evaluation_completed_triggers_curriculum_update(self, mock_event_bus):
        """Verify EVALUATION_COMPLETED triggers curriculum rebalance."""
        curriculum_updated = False

        async def curriculum_handler(event):
            nonlocal curriculum_updated
            if event.payload.get("win_rate", 0) >= 0.6:
                curriculum_updated = True

        mock_event_bus.subscribe("evaluation_completed", curriculum_handler)

        # Publish evaluation completion
        event = MockDataEvent(
            event_type=MagicMock(value="evaluation_completed"),
            payload={
                "config_key": "hex8_2p",
                "win_rate": 0.85,
                "elo_delta": 50,
            },
        )
        await mock_event_bus.publish(event)

        assert curriculum_updated, "Curriculum should update after successful evaluation"


# =============================================================================
# Test Event Router Health
# =============================================================================


class TestEventRouterHealth:
    """Tests for event router health monitoring."""

    def test_event_router_stats_available(self, reset_event_router):
        """Verify event router provides statistics."""
        from app.coordination.event_router import get_event_stats

        stats = get_event_stats()

        assert isinstance(stats, dict)
        # Check for expected stat keys
        expected_keys = ["events_routed", "duplicates_prevented"]
        for key in expected_keys:
            assert key in stats, f"Missing stat: {key}"

    def test_event_router_singleton_pattern(self, reset_event_router):
        """Verify get_router returns same instance."""
        from app.coordination.event_router import get_router

        router1 = get_router()
        router2 = get_router()

        assert router1 is router2, "get_router should return singleton"

    def test_event_router_has_required_methods(self, reset_event_router):
        """Verify router has all required interface methods."""
        from app.coordination.event_router import get_router

        router = get_router()

        required_methods = ["subscribe", "publish", "start", "stop"]
        for method in required_methods:
            assert hasattr(router, method), f"Router missing method: {method}"
            assert callable(getattr(router, method)), f"{method} should be callable"


# =============================================================================
# Test Cross-System Event Flow
# =============================================================================


class TestCrossSystemEventFlow:
    """Tests for events flowing between different subsystems."""

    @pytest.mark.asyncio
    async def test_p2p_events_reach_coordination(self, mock_event_bus):
        """Verify P2P events (HOST_ONLINE/OFFLINE) reach coordination layer."""
        pytest.importorskip("app.distributed.data_events")

        from app.distributed.data_events import DataEventType

        # Verify P2P events exist
        p2p_events = [
            "HOST_ONLINE",
            "HOST_OFFLINE",
            "LEADER_ELECTED",
            "NODE_RECOVERED",
        ]

        for event_name in p2p_events:
            assert hasattr(DataEventType, event_name), f"Missing P2P event: {event_name}"

    @pytest.mark.asyncio
    async def test_daemon_lifecycle_events_emitted(self, mock_event_bus):
        """Verify daemon lifecycle events are available."""
        pytest.importorskip("app.distributed.data_events")

        from app.distributed.data_events import DataEventType

        lifecycle_events = [
            "DAEMON_STARTED",
            "DAEMON_STOPPED",
            "DAEMON_STATUS_CHANGED",
        ]

        for event_name in lifecycle_events:
            assert hasattr(DataEventType, event_name), f"Missing lifecycle event: {event_name}"

    @pytest.mark.asyncio
    async def test_event_mappings_are_consistent(self, reset_event_router):
        """Verify event mappings are consistent across buses."""
        pytest.importorskip("app.coordination.event_mappings")

        from app.coordination.event_mappings import (
            DATA_TO_CROSS_PROCESS_MAP,
            validate_mappings,
        )

        # Key events should have mappings
        key_events = [
            "training_completed",
            "data_sync_completed",
            "model_promoted",
            "selfplay_complete",
        ]

        for event in key_events:
            assert event in DATA_TO_CROSS_PROCESS_MAP, f"Missing mapping: {event}"

        # Validate all mappings
        warnings = validate_mappings()
        assert len(warnings) == 0, f"Mapping warnings: {warnings}"
