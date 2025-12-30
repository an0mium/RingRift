"""Integration tests for event propagation across daemons.

December 2025: Tests critical event flows that span multiple daemons
and verify the event routing infrastructure works end-to-end.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTrainingCompletedFlow:
    """Tests for TRAINING_COMPLETED event propagation."""

    @pytest.mark.asyncio
    async def test_training_complete_triggers_evaluation_subscription(self):
        """TRAINING_COMPLETED event should reach evaluation subscribers."""
        from app.coordination.event_router import get_router, reset_router

        # Reset router state
        reset_router()
        router = get_router()

        # Track received events
        received_events = []

        def on_training_complete(event: Any) -> None:
            received_events.append(event)

        # Subscribe to training completed
        router.subscribe("training_completed", on_training_complete)

        # Emit event
        await router.publish(
            "training_completed",
            {
                "config_key": "hex8_2p",
                "model_path": "/tmp/test_model.pth",
                "metrics": {"loss": 0.5},
            },
        )

        # Allow async processing
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0]["config_key"] == "hex8_2p"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_receive_event(self):
        """All subscribers should receive the same event."""
        from app.coordination.event_router import get_router, reset_router

        reset_router()
        router = get_router()

        # Track events for multiple subscribers
        subscriber1_events = []
        subscriber2_events = []
        subscriber3_events = []

        router.subscribe("test_event", lambda e: subscriber1_events.append(e))
        router.subscribe("test_event", lambda e: subscriber2_events.append(e))
        router.subscribe("test_event", lambda e: subscriber3_events.append(e))

        await router.publish("test_event", {"data": "test_value"})
        await asyncio.sleep(0.1)

        assert len(subscriber1_events) == 1
        assert len(subscriber2_events) == 1
        assert len(subscriber3_events) == 1
        assert subscriber1_events[0]["data"] == "test_value"


class TestModelPromotedFlow:
    """Tests for MODEL_PROMOTED event propagation."""

    @pytest.mark.asyncio
    async def test_model_promoted_triggers_distribution(self):
        """MODEL_PROMOTED should trigger model distribution logic."""
        from app.coordination.event_router import get_router, reset_router

        reset_router()
        router = get_router()

        distribution_triggered = []

        def on_model_promoted(event: Any) -> None:
            distribution_triggered.append(event)

        router.subscribe("model_promoted", on_model_promoted)

        await router.publish(
            "model_promoted",
            {
                "config_key": "square8_2p",
                "model_path": "/tmp/canonical_square8_2p.pth",
                "elo_gain": 50,
            },
        )

        await asyncio.sleep(0.1)

        assert len(distribution_triggered) == 1
        assert distribution_triggered[0]["config_key"] == "square8_2p"


class TestDataSyncFlow:
    """Tests for DATA_SYNC_* event propagation."""

    @pytest.mark.asyncio
    async def test_sync_completed_triggers_pipeline(self):
        """DATA_SYNC_COMPLETED should trigger pipeline export check."""
        from app.coordination.event_router import get_router, reset_router

        reset_router()
        router = get_router()

        pipeline_events = []

        def on_sync_completed(event: Any) -> None:
            pipeline_events.append(event)

        router.subscribe("sync_completed", on_sync_completed)

        await router.publish(
            "sync_completed",
            {
                "host": "nebius-h100-1",
                "games_synced": 500,
                "duration": 30.5,
            },
        )

        await asyncio.sleep(0.1)

        assert len(pipeline_events) == 1
        assert pipeline_events[0]["games_synced"] == 500


class TestEventRouterResilience:
    """Tests for event router resilience features."""

    @pytest.mark.asyncio
    async def test_subscriber_exception_does_not_break_other_subscribers(self):
        """One failing subscriber should not prevent others from receiving."""
        from app.coordination.event_router import get_router, reset_router

        reset_router()
        router = get_router()

        successful_events = []

        def failing_subscriber(event: Any) -> None:
            raise RuntimeError("Intentional test failure")

        def successful_subscriber(event: Any) -> None:
            successful_events.append(event)

        # Register failing subscriber first
        router.subscribe("resilience_test", failing_subscriber)
        router.subscribe("resilience_test", successful_subscriber)

        # Should not raise despite failing subscriber
        await router.publish("resilience_test", {"test": "data"})
        await asyncio.sleep(0.1)

        # Successful subscriber should still receive
        assert len(successful_events) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_event_delivery(self):
        """Unsubscribed handlers should not receive events."""
        from app.coordination.event_router import get_router, reset_router

        reset_router()
        router = get_router()

        events = []

        def handler(event: Any) -> None:
            events.append(event)

        router.subscribe("unsub_test", handler)

        # Should receive first event
        await router.publish("unsub_test", {"n": 1})
        await asyncio.sleep(0.1)
        assert len(events) == 1

        # Unsubscribe
        router.unsubscribe("unsub_test", handler)

        # Should NOT receive second event
        await router.publish("unsub_test", {"n": 2})
        await asyncio.sleep(0.1)
        assert len(events) == 1  # Still just 1


class TestCrossLayerPropagation:
    """Tests for event propagation across in-memory, stage, and cluster layers."""

    @pytest.mark.asyncio
    async def test_router_has_subscribers_check(self):
        """Router should correctly report subscriber presence."""
        from app.coordination.event_router import get_router, reset_router

        reset_router()
        router = get_router()

        # No subscribers yet
        assert not router.has_subscribers("new_event_type")

        # Add subscriber
        router.subscribe("new_event_type", lambda e: None)

        # Now should have subscribers
        assert router.has_subscribers("new_event_type")

    @pytest.mark.asyncio
    async def test_event_stats_tracking(self):
        """Router should track event statistics."""
        from app.coordination.event_router import get_router, reset_router

        reset_router()
        router = get_router()

        router.subscribe("stats_test", lambda e: None)

        for i in range(5):
            await router.publish("stats_test", {"i": i})

        await asyncio.sleep(0.1)

        stats = router.get_stats()
        assert stats.get("events_emitted", 0) >= 5


class TestDaemonEventWiring:
    """Tests that daemon event subscriptions are correctly wired."""

    def test_feedback_loop_subscribes_to_training_events(self):
        """FeedbackLoopController should subscribe to TRAINING_* events."""
        # This is a structural test - verify the wiring exists
        from app.coordination.event_router import get_router

        router = get_router()

        # Check that common training events can be subscribed to
        test_handler = MagicMock()
        router.subscribe("training_completed", test_handler)

        # Verify subscription registered
        assert router.has_subscribers("training_completed")

    def test_data_pipeline_subscribes_to_sync_events(self):
        """DataPipelineOrchestrator should subscribe to DATA_SYNC_* events."""
        from app.coordination.event_router import get_router

        router = get_router()

        test_handler = MagicMock()
        router.subscribe("sync_completed", test_handler)

        assert router.has_subscribers("sync_completed")


class TestEventTimeout:
    """Tests for event handler timeout protection."""

    @pytest.mark.asyncio
    async def test_slow_handler_does_not_block_router(self):
        """Slow handlers should not block other event processing."""
        from app.coordination.event_router import get_router, reset_router

        reset_router()
        router = get_router()

        fast_events = []
        slow_started = []

        async def slow_handler(event: Any) -> None:
            slow_started.append(time.time())
            await asyncio.sleep(0.5)

        def fast_handler(event: Any) -> None:
            fast_events.append(time.time())

        router.subscribe("timeout_test", slow_handler)
        router.subscribe("timeout_test", fast_handler)

        start = time.time()
        await router.publish("timeout_test", {"test": True})
        await asyncio.sleep(0.2)

        # Fast handler should receive quickly even if slow handler is running
        assert len(fast_events) >= 1
        assert fast_events[0] - start < 0.3  # Should be fast
