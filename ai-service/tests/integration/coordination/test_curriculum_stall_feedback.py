"""Integration tests for curriculum stall → reset feedback loop.

January 26, 2026: Tests the P2 improvement (Elo→Curriculum feedback) that wires
PROGRESS_STALL_DETECTED → CURRICULUM_RESET_REQUESTED → curriculum weight reset.

This verifies:
1. CURRICULUM_RESET_REQUESTED event type exists
2. CurriculumFeedbackHandler subscribes to the event
3. Handler resets curriculum weights to baseline (1.0)
4. CURRICULUM_REBALANCED event is emitted after reset
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCurriculumResetEventType:
    """Test CURRICULUM_RESET_REQUESTED event type exists."""

    def test_event_type_exists(self):
        """Verify CURRICULUM_RESET_REQUESTED is in DataEventType enum."""
        from app.distributed.data_events.event_types import DataEventType

        assert hasattr(DataEventType, "CURRICULUM_RESET_REQUESTED")
        assert DataEventType.CURRICULUM_RESET_REQUESTED.value == "curriculum_reset_requested"

    def test_event_type_in_curriculum_events_section(self):
        """Verify event is with other curriculum events."""
        from app.distributed.data_events.event_types import DataEventType

        # Get enum members in order
        members = list(DataEventType)
        reset_idx = members.index(DataEventType.CURRICULUM_RESET_REQUESTED)
        rebalanced_idx = members.index(DataEventType.CURRICULUM_REBALANCED)
        rollback_idx = members.index(DataEventType.CURRICULUM_ROLLBACK_COMPLETED)

        # CURRICULUM_RESET_REQUESTED should be near other curriculum events
        # It's after CURRICULUM_ROLLBACK_COMPLETED
        assert reset_idx > rollback_idx


class TestCurriculumFeedbackHandlerSubscription:
    """Test CurriculumFeedbackHandler subscribes to CURRICULUM_RESET_REQUESTED."""

    def test_handler_subscribes_to_curriculum_reset_requested(self):
        """Verify handler has subscription for CURRICULUM_RESET_REQUESTED."""
        from app.coordination.curriculum_feedback_handler import CurriculumFeedbackHandler

        CurriculumFeedbackHandler.reset_instance()
        handler = CurriculumFeedbackHandler()

        subscriptions = handler._get_event_subscriptions()
        assert "CURRICULUM_RESET_REQUESTED" in subscriptions
        assert callable(subscriptions["CURRICULUM_RESET_REQUESTED"])

    def test_handler_method_exists(self):
        """Verify _on_curriculum_reset_requested method exists."""
        from app.coordination.curriculum_feedback_handler import CurriculumFeedbackHandler

        CurriculumFeedbackHandler.reset_instance()
        handler = CurriculumFeedbackHandler()

        assert hasattr(handler, "_on_curriculum_reset_requested")
        assert asyncio.iscoroutinefunction(handler._on_curriculum_reset_requested)


class TestCurriculumResetHandler:
    """Test curriculum reset handler behavior."""

    @pytest.fixture
    def handler(self):
        """Create a fresh handler for testing."""
        from app.coordination.curriculum_feedback_handler import CurriculumFeedbackHandler

        CurriculumFeedbackHandler.reset_instance()
        handler = CurriculumFeedbackHandler()
        yield handler
        CurriculumFeedbackHandler.reset_instance()

    @pytest.fixture
    def mock_event(self):
        """Create a mock curriculum reset event."""

        def _create_event(
            config_key: str = "hex8_2p",
            reason: str = "extended_stall",
            stall_hours: float = 96.0,
        ) -> MagicMock:
            event = MagicMock()
            event.payload = {
                "config_key": config_key,
                "reason": reason,
                "stall_duration_hours": stall_hours,
            }
            return event

        return _create_event

    @pytest.mark.asyncio
    async def test_reset_sets_weight_to_baseline(self, handler, mock_event):
        """Test that curriculum reset sets weight to 1.0 baseline."""
        # Set up a non-baseline weight
        state = handler._get_or_create_state("hex8_2p")
        state.current_curriculum_weight = 1.5

        event = mock_event(config_key="hex8_2p", reason="tier4_stall", stall_hours=168.0)

        with patch("app.coordination.curriculum_feedback_handler.safe_emit_event"):
            await handler._on_curriculum_reset_requested(event)

        # Weight should be reset to baseline
        assert state.current_curriculum_weight == 1.0

    @pytest.mark.asyncio
    async def test_reset_updates_last_reset_timestamp(self, handler, mock_event):
        """Test that curriculum reset updates the last_reset timestamp."""
        state = handler._get_or_create_state("hex8_2p")
        state.curriculum_last_reset = 0.0

        before_time = time.time()
        event = mock_event(config_key="hex8_2p")

        with patch("app.coordination.curriculum_feedback_handler.safe_emit_event"):
            await handler._on_curriculum_reset_requested(event)

        after_time = time.time()

        # Timestamp should be updated
        assert state.curriculum_last_reset >= before_time
        assert state.curriculum_last_reset <= after_time

    @pytest.mark.asyncio
    async def test_reset_increments_adjustment_counter(self, handler, mock_event):
        """Test that curriculum reset increments the adjustments counter."""
        initial_count = handler._curriculum_adjustments

        event = mock_event(config_key="hex8_2p")

        with patch("app.coordination.curriculum_feedback_handler.safe_emit_event"):
            await handler._on_curriculum_reset_requested(event)

        assert handler._curriculum_adjustments == initial_count + 1

    @pytest.mark.asyncio
    async def test_reset_emits_curriculum_rebalanced_event(self, handler, mock_event):
        """Test that CURRICULUM_REBALANCED event is emitted after reset."""
        event = mock_event(config_key="hex8_2p", reason="tier3_stall", stall_hours=96.0)

        with patch(
            "app.coordination.curriculum_feedback_handler.safe_emit_event"
        ) as mock_emit:
            await handler._on_curriculum_reset_requested(event)

        # Verify CURRICULUM_REBALANCED was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args

        assert call_args[0][0] == "CURRICULUM_REBALANCED"
        payload = call_args[0][1]
        assert payload["config_key"] == "hex8_2p"
        assert "stall_reset" in payload["reason"]
        assert payload["stall_duration_hours"] == 96.0
        assert payload["new_weight"] == 1.0

    @pytest.mark.asyncio
    async def test_reset_ignores_event_without_config_key(self, handler):
        """Test that events without config_key are ignored."""
        event = MagicMock()
        event.payload = {"reason": "test", "stall_duration_hours": 100}

        initial_count = handler._curriculum_adjustments

        with patch(
            "app.coordination.curriculum_feedback_handler.safe_emit_event"
        ) as mock_emit:
            await handler._on_curriculum_reset_requested(event)

        # Should not have made any changes
        assert handler._curriculum_adjustments == initial_count
        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_handles_curriculum_feedback_integration(self, handler, mock_event):
        """Test integration with curriculum_feedback system."""
        event = mock_event(config_key="hex8_2p")

        mock_feedback = MagicMock()
        mock_feedback._current_weights = {"hex8_2p": 1.5}

        with patch(
            "app.coordination.curriculum_feedback_handler.safe_emit_event"
        ), patch(
            "app.training.curriculum_feedback.get_curriculum_feedback",
            return_value=mock_feedback,
        ):
            await handler._on_curriculum_reset_requested(event)

        # Should have reset the weight in curriculum_feedback system too
        assert mock_feedback._current_weights["hex8_2p"] == 1.0


class TestProgressStallToCurriculumResetFlow:
    """Test the full flow from PROGRESS_STALL_DETECTED to curriculum reset.

    This tests the integration between:
    1. progress_watchdog_daemon emitting curriculum_reset_requested
    2. CurriculumFeedbackHandler receiving and processing the event
    """

    @pytest.mark.asyncio
    async def test_tier3_stall_triggers_curriculum_reset(self):
        """Test that tier3 (96+ hour) stalls trigger curriculum reset."""
        from app.coordination.curriculum_feedback_handler import (
            CurriculumFeedbackHandler,
            reset_curriculum_feedback_handler,
        )

        reset_curriculum_feedback_handler()
        handler = CurriculumFeedbackHandler()

        # Simulate a tier3 stall event (96+ hours)
        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "reason": "tier3_96h",
            "stall_duration_hours": 96.0,
            "tier": 3,
        }

        # Set up initial state with non-baseline weight
        state = handler._get_or_create_state("hex8_2p")
        state.current_curriculum_weight = 2.0

        with patch("app.coordination.curriculum_feedback_handler.safe_emit_event"):
            await handler._on_curriculum_reset_requested(event)

        # Verify reset happened
        assert state.current_curriculum_weight == 1.0
        assert state.curriculum_last_reset > 0

        reset_curriculum_feedback_handler()

    @pytest.mark.asyncio
    async def test_tier4_stall_triggers_curriculum_reset(self):
        """Test that tier4 (168+ hour) stalls trigger curriculum reset."""
        from app.coordination.curriculum_feedback_handler import (
            CurriculumFeedbackHandler,
            reset_curriculum_feedback_handler,
        )

        reset_curriculum_feedback_handler()
        handler = CurriculumFeedbackHandler()

        # Simulate a tier4 stall event (168+ hours / 7 days)
        event = MagicMock()
        event.payload = {
            "config_key": "square8_4p",
            "reason": "tier4_168h",
            "stall_duration_hours": 168.0,
            "tier": 4,
        }

        state = handler._get_or_create_state("square8_4p")
        state.current_curriculum_weight = 0.5

        with patch("app.coordination.curriculum_feedback_handler.safe_emit_event"):
            await handler._on_curriculum_reset_requested(event)

        assert state.current_curriculum_weight == 1.0

        reset_curriculum_feedback_handler()


class TestFeedbackStateFields:
    """Test that FeedbackState has the required fields for stall tracking."""

    def test_feedback_state_has_curriculum_last_reset(self):
        """Verify FeedbackState has curriculum_last_reset field."""
        from app.coordination.feedback_loop_controller import FeedbackState

        state = FeedbackState(config_key="test")
        assert hasattr(state, "curriculum_last_reset")
        assert state.curriculum_last_reset == 0.0

    def test_feedback_state_has_evaluation_in_progress(self):
        """Verify FeedbackState has evaluation_in_progress field."""
        from app.coordination.feedback_loop_controller import FeedbackState

        state = FeedbackState(config_key="test")
        assert hasattr(state, "evaluation_in_progress")
        assert state.evaluation_in_progress is False


class TestHealthCheckIncludesStallMetrics:
    """Test that health_check includes stall-related metrics."""

    def test_health_check_includes_adjustment_count(self):
        """Verify health_check reports curriculum_adjustments."""
        from app.coordination.curriculum_feedback_handler import (
            CurriculumFeedbackHandler,
            reset_curriculum_feedback_handler,
        )

        reset_curriculum_feedback_handler()
        handler = CurriculumFeedbackHandler()
        handler._curriculum_adjustments = 5

        result = handler.health_check()

        assert "curriculum_adjustments" in result.details
        assert result.details["curriculum_adjustments"] == 5

        reset_curriculum_feedback_handler()
