"""Tests for stage_events - pipeline stage completion event system.

December 27, 2025: Created as part of test coverage improvement effort.
Tests the event bus for pipeline stage notifications.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def event_bus():
    """Create a fresh StageEventBus instance."""
    from app.coordination.stage_events import StageEventBus
    return StageEventBus()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    from app.coordination.stage_events import reset_event_bus
    yield
    reset_event_bus()


# =============================================================================
# StageEvent Enum Tests
# =============================================================================


class TestStageEvent:
    """Tests for StageEvent enum."""

    def test_selfplay_events(self):
        """Test selfplay-related events."""
        from app.coordination.stage_events import StageEvent

        assert StageEvent.SELFPLAY_COMPLETE.value == "selfplay_complete"
        assert StageEvent.CANONICAL_SELFPLAY_COMPLETE.value == "canonical_selfplay_complete"
        assert StageEvent.GPU_SELFPLAY_COMPLETE.value == "gpu_selfplay_complete"

    def test_training_events(self):
        """Test training-related events."""
        from app.coordination.stage_events import StageEvent

        assert StageEvent.TRAINING_STARTED.value == "training_started"
        assert StageEvent.TRAINING_COMPLETE.value == "training_complete"
        assert StageEvent.TRAINING_FAILED.value == "training_failed"

    def test_evaluation_events(self):
        """Test evaluation-related events."""
        from app.coordination.stage_events import StageEvent

        assert StageEvent.EVALUATION_COMPLETE.value == "evaluation_complete"
        assert StageEvent.SHADOW_TOURNAMENT_COMPLETE.value == "shadow_tournament_complete"

    def test_promotion_events(self):
        """Test promotion-related events."""
        from app.coordination.stage_events import StageEvent

        assert StageEvent.PROMOTION_COMPLETE.value == "promotion_complete"


# =============================================================================
# StageCompletionResult Dataclass Tests
# =============================================================================


class TestStageCompletionResult:
    """Tests for StageCompletionResult dataclass."""

    def test_defaults(self):
        """Test default values."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        assert result.event == StageEvent.SELFPLAY_COMPLETE
        assert result.success is True
        assert result.iteration == 1
        assert result.board_type == "square8"
        assert result.num_players == 2
        assert result.games_generated == 0
        assert result.model_path is None
        assert result.error is None
        assert result.metadata == {}

    def test_with_metrics(self):
        """Test with training and evaluation metrics."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        result = StageCompletionResult(
            event=StageEvent.TRAINING_COMPLETE,
            success=True,
            iteration=5,
            timestamp="2025-12-27T10:00:00",
            model_path="/path/to/model.pth",
            train_loss=0.15,
            val_loss=0.18,
        )

        assert result.model_path == "/path/to/model.pth"
        assert result.train_loss == 0.15
        assert result.val_loss == 0.18

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from app.coordination.stage_events import StageCompletionResult, StageEvent

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
            games_generated=500,
        )

        d = result.to_dict()
        assert d["event"] == "selfplay_complete"
        assert d["success"] is True
        assert d["iteration"] == 1
        assert d["games_generated"] == 500

    def test_from_dict(self):
        """Test creation from dictionary."""
        from app.coordination.stage_events import StageCompletionResult

        data = {
            "event": "training_complete",
            "success": True,
            "iteration": 3,
            "timestamp": "2025-12-27T10:00:00",
            "model_path": "/models/test.pth",
            "train_loss": 0.1,
        }

        result = StageCompletionResult.from_dict(data)
        assert result.success is True
        assert result.iteration == 3
        assert result.model_path == "/models/test.pth"
        assert result.train_loss == 0.1

    def test_from_dict_defaults(self):
        """Test from_dict with minimal data."""
        from app.coordination.stage_events import StageCompletionResult

        result = StageCompletionResult.from_dict({})
        assert result.success is False
        assert result.iteration == 0
        assert result.board_type == "square8"


# =============================================================================
# StageEventBus Initialization Tests
# =============================================================================


class TestStageEventBusInit:
    """Tests for StageEventBus initialization."""

    def test_basic_init(self, event_bus):
        """Test basic initialization."""
        assert event_bus._subscribers == {}
        assert event_bus._history == []
        assert event_bus._callback_errors == []
        assert event_bus._max_history == 100

    def test_custom_max_history(self):
        """Test custom max_history setting."""
        from app.coordination.stage_events import StageEventBus

        bus = StageEventBus(max_history=50)
        assert bus._max_history == 50

    def test_set_logger(self, event_bus):
        """Test setting a logger callback."""
        log_fn = MagicMock()
        event_bus.set_logger(log_fn)
        assert event_bus._log_callback == log_fn


# =============================================================================
# Subscription Tests
# =============================================================================


class TestSubscription:
    """Tests for event subscription."""

    def test_subscribe_single_callback(self, event_bus):
        """Test subscribing a single callback."""
        from app.coordination.stage_events import StageEvent

        async def callback(result):
            pass

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)
        assert event_bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 1

    def test_subscribe_multiple_callbacks(self, event_bus):
        """Test subscribing multiple callbacks to same event."""
        from app.coordination.stage_events import StageEvent

        async def callback1(result):
            pass

        async def callback2(result):
            pass

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback1)
        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback2)
        assert event_bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 2

    def test_subscribe_same_callback_twice(self, event_bus):
        """Test that same callback is not added twice."""
        from app.coordination.stage_events import StageEvent

        async def callback(result):
            pass

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)
        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)
        assert event_bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 1

    def test_unsubscribe(self, event_bus):
        """Test unsubscribing a callback."""
        from app.coordination.stage_events import StageEvent

        async def callback(result):
            pass

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)
        result = event_bus.unsubscribe(StageEvent.SELFPLAY_COMPLETE, callback)

        assert result is True
        assert event_bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 0

    def test_unsubscribe_not_found(self, event_bus):
        """Test unsubscribing non-existent callback."""
        from app.coordination.stage_events import StageEvent

        async def callback(result):
            pass

        result = event_bus.unsubscribe(StageEvent.SELFPLAY_COMPLETE, callback)
        assert result is False

    def test_clear_subscribers_single_event(self, event_bus):
        """Test clearing subscribers for a single event."""
        from app.coordination.stage_events import StageEvent

        async def callback(result):
            pass

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)
        event_bus.subscribe(StageEvent.TRAINING_COMPLETE, callback)

        cleared = event_bus.clear_subscribers(StageEvent.SELFPLAY_COMPLETE)

        assert cleared == 1
        assert event_bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 0
        assert event_bus.subscriber_count(StageEvent.TRAINING_COMPLETE) == 1

    def test_clear_all_subscribers(self, event_bus):
        """Test clearing all subscribers."""
        from app.coordination.stage_events import StageEvent

        async def callback(result):
            pass

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)
        event_bus.subscribe(StageEvent.TRAINING_COMPLETE, callback)

        cleared = event_bus.clear_subscribers()

        assert cleared == 2
        assert event_bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) == 0
        assert event_bus.subscriber_count(StageEvent.TRAINING_COMPLETE) == 0


# =============================================================================
# Emit Tests
# =============================================================================


class TestEmit:
    """Tests for event emission."""

    @pytest.mark.asyncio
    async def test_emit_invokes_callbacks(self, event_bus):
        """Test that emit invokes all callbacks."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        invoked = []

        async def callback1(result):
            invoked.append("callback1")

        async def callback2(result):
            invoked.append("callback2")

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback1)
        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback2)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        count = await event_bus.emit(result)

        assert count == 2
        assert "callback1" in invoked
        assert "callback2" in invoked

    @pytest.mark.asyncio
    async def test_emit_records_history(self, event_bus):
        """Test that emit records events in history."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        await event_bus.emit(result)

        history = event_bus.get_history()
        assert len(history) == 1
        assert history[0].event == StageEvent.SELFPLAY_COMPLETE

    @pytest.mark.asyncio
    async def test_emit_handles_callback_error(self, event_bus):
        """Test that emit handles callback errors gracefully."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        async def failing_callback(result):
            raise RuntimeError("Test error")

        async def passing_callback(result):
            pass

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, failing_callback)
        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, passing_callback)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        # Should not raise
        count = await event_bus.emit(result)

        assert count == 1  # Only passing callback counted
        errors = event_bus.get_callback_errors()
        assert len(errors) == 1
        assert "Test error" in errors[0]["error"]

    @pytest.mark.asyncio
    async def test_emit_no_subscribers(self, event_bus):
        """Test emit with no subscribers."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        count = await event_bus.emit(result)
        assert count == 0

    @pytest.mark.asyncio
    async def test_emit_with_logger(self, event_bus):
        """Test emit uses custom logger."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        log_messages = []
        event_bus.set_logger(lambda msg: log_messages.append(msg))

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        await event_bus.emit(result)

        assert len(log_messages) == 1
        assert "selfplay_complete" in log_messages[0]
        assert "OK" in log_messages[0]


# =============================================================================
# Emit and Wait Tests
# =============================================================================


class TestEmitAndWait:
    """Tests for emit_and_wait."""

    @pytest.mark.asyncio
    async def test_emit_and_wait_returns_results(self, event_bus):
        """Test that emit_and_wait returns callback results."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        async def callback1(result):
            return "result1"

        async def callback2(result):
            return "result2"

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback1)
        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback2)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        results = await event_bus.emit_and_wait(result)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_emit_and_wait_no_subscribers(self, event_bus):
        """Test emit_and_wait with no subscribers."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        results = await event_bus.emit_and_wait(result)
        assert results == []

    @pytest.mark.asyncio
    async def test_emit_and_wait_timeout(self, event_bus):
        """Test emit_and_wait with timeout."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        async def slow_callback(result):
            await asyncio.sleep(2)

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, slow_callback)

        result = StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        )

        results = await event_bus.emit_and_wait(result, timeout=0.1)
        assert results == []  # Timeout


# =============================================================================
# History and Stats Tests
# =============================================================================


class TestHistoryAndStats:
    """Tests for history and statistics."""

    @pytest.mark.asyncio
    async def test_history_respects_limit(self, event_bus):
        """Test that history respects max_history limit."""
        from app.coordination.stage_events import (
            StageEvent,
            StageCompletionResult,
            StageEventBus,
        )

        bus = StageEventBus(max_history=5)

        for i in range(10):
            result = StageCompletionResult(
                event=StageEvent.SELFPLAY_COMPLETE,
                success=True,
                iteration=i,
                timestamp="2025-12-27T10:00:00",
            )
            await bus.emit(result)

        history = bus.get_history()
        assert len(history) <= 5

    @pytest.mark.asyncio
    async def test_history_filter_by_event(self, event_bus):
        """Test filtering history by event type."""
        from app.coordination.stage_events import StageEvent, StageCompletionResult

        await event_bus.emit(StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        ))
        await event_bus.emit(StageCompletionResult(
            event=StageEvent.TRAINING_COMPLETE,
            success=True,
            iteration=1,
            timestamp="2025-12-27T10:00:00",
        ))

        selfplay_history = event_bus.get_history(event=StageEvent.SELFPLAY_COMPLETE)
        assert len(selfplay_history) == 1

    def test_get_stats(self, event_bus):
        """Test get_stats returns expected structure."""
        from app.coordination.stage_events import StageEvent

        async def callback(result):
            pass

        event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)

        stats = event_bus.get_stats()
        assert "total_subscribers" in stats
        assert "subscribers_by_event" in stats
        assert "history_size" in stats
        assert "callback_errors" in stats
        assert "supported_events" in stats

        assert stats["total_subscribers"] == 1


# =============================================================================
# Singleton Management Tests
# =============================================================================


class TestSingletonManagement:
    """Tests for singleton pattern."""

    def test_get_event_bus_returns_singleton(self):
        """Test that get_event_bus returns the same instance."""
        import warnings
        from app.coordination.stage_events import get_event_bus, reset_event_bus

        reset_event_bus()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bus1 = get_event_bus()
            bus2 = get_event_bus()

        assert bus1 is bus2

    def test_get_event_bus_emits_deprecation_warning(self):
        """Test that get_event_bus emits deprecation warning."""
        import warnings
        from app.coordination.stage_events import get_event_bus, reset_event_bus

        reset_event_bus()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_event_bus()

            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)


# =============================================================================
# Pipeline Callbacks Tests
# =============================================================================


class TestPipelineCallbacks:
    """Tests for pipeline callback helpers."""

    def test_create_pipeline_callbacks(self):
        """Test create_pipeline_callbacks returns expected events."""
        from app.coordination.stage_events import create_pipeline_callbacks, StageEvent

        callbacks = create_pipeline_callbacks()

        assert StageEvent.SELFPLAY_COMPLETE in callbacks
        assert StageEvent.SYNC_COMPLETE in callbacks
        assert StageEvent.TRAINING_COMPLETE in callbacks
        assert StageEvent.EVALUATION_COMPLETE in callbacks

    def test_register_standard_callbacks(self, event_bus):
        """Test register_standard_callbacks adds callbacks."""
        from app.coordination.stage_events import (
            register_standard_callbacks,
            StageEvent,
        )

        register_standard_callbacks(event_bus)

        assert event_bus.subscriber_count(StageEvent.SELFPLAY_COMPLETE) >= 1
        assert event_bus.subscriber_count(StageEvent.TRAINING_COMPLETE) >= 1
