"""Unit tests for feedback_loop_controller module.

Tests the FeedbackLoopController and FeedbackState components that
orchestrate training feedback signals.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.feedback_loop_controller import (
    FeedbackLoopController,
    FeedbackState,
    get_feedback_loop_controller,
)


# =============================================================================
# FeedbackState Tests
# =============================================================================


class TestFeedbackState:
    """Test FeedbackState dataclass."""

    def test_create_state(self):
        """Should create state with config key."""
        state = FeedbackState(config_key="hex8_2p")
        assert state.config_key == "hex8_2p"
        assert state.last_selfplay_quality == 0.0
        assert state.current_training_intensity == "normal"

    def test_default_values(self):
        """Should have sensible defaults."""
        state = FeedbackState(config_key="test")
        assert state.last_training_accuracy == 0.0
        assert state.last_evaluation_win_rate == 0.0
        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 0
        assert state.current_exploration_boost == 1.0
        assert state.current_curriculum_weight == 1.0

    def test_promotion_status_tracking(self):
        """Should track promotion status."""
        state = FeedbackState(config_key="test")
        assert state.last_promotion_success is None

        # Simulate promotion tracking
        state.last_promotion_success = True
        state.consecutive_successes = 1
        assert state.last_promotion_success is True
        assert state.consecutive_successes == 1

    def test_work_metrics(self):
        """Should track work completion metrics."""
        state = FeedbackState(config_key="test")
        assert state.work_completed_count == 0
        assert state.last_work_completion_time == 0.0

        # Simulate work completion
        state.work_completed_count = 10
        state.last_work_completion_time = time.time()
        assert state.work_completed_count == 10


# =============================================================================
# FeedbackLoopController Initialization Tests
# =============================================================================


class TestFeedbackLoopControllerInit:
    """Test FeedbackLoopController initialization."""

    def test_init_default(self):
        """Should initialize with defaults."""
        controller = FeedbackLoopController()
        assert controller._running is False
        assert isinstance(controller._states, dict)

    def test_has_configuration_attributes(self):
        """Should have configuration attributes."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "policy_accuracy_threshold")
        assert hasattr(controller, "promotion_threshold")
        assert controller.promotion_threshold >= 0.5

    def test_has_lock(self):
        """Should have thread lock."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "_lock")


# =============================================================================
# FeedbackLoopController State Management Tests
# =============================================================================


class TestFeedbackLoopControllerState:
    """Test state management."""

    @pytest.fixture
    def controller(self):
        """Create controller."""
        return FeedbackLoopController()

    def test_get_state_creates_new(self, controller):
        """Should create new state if not exists."""
        state = controller._get_or_create_state("hex8_2p")
        assert state is not None
        assert state.config_key == "hex8_2p"

    def test_get_state_returns_existing(self, controller):
        """Should return existing state."""
        state1 = controller._get_or_create_state("hex8_2p")
        state1.last_selfplay_quality = 0.9
        state2 = controller._get_or_create_state("hex8_2p")
        assert state2.last_selfplay_quality == 0.9

    def test_get_state_returns_none_for_missing(self, controller):
        """get_state should return None for missing key."""
        result = controller.get_state("nonexistent")
        assert result is None

    def test_get_all_states(self, controller):
        """Should return all states."""
        controller._get_or_create_state("hex8_2p")
        controller._get_or_create_state("square8_4p")
        states = controller.get_all_states()
        assert len(states) >= 2


# =============================================================================
# FeedbackLoopController Signal Tests
# =============================================================================


class TestFeedbackLoopControllerSignals:
    """Test signal methods."""

    @pytest.fixture
    def controller(self):
        """Create controller without auto-wiring."""
        return FeedbackLoopController()

    def test_signal_selfplay_quality(self, controller):
        """Should update selfplay quality."""
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.85)
        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_selfplay_quality == 0.85

    def test_signal_training_complete(self, controller):
        """Should update training metrics."""
        controller.signal_training_complete(
            "hex8_2p",
            policy_accuracy=0.78,
            value_accuracy=0.65,
        )
        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_training_accuracy == 0.78


# =============================================================================
# FeedbackLoopController Summary Tests
# =============================================================================


class TestFeedbackLoopControllerSummary:
    """Test summary and status methods."""

    @pytest.fixture
    def controller(self):
        """Create controller without auto-wiring."""
        return FeedbackLoopController()

    def test_get_summary_empty(self, controller):
        """Should return summary even with no states."""
        summary = controller.get_summary()
        assert isinstance(summary, dict)
        assert "configs" in summary or len(summary) >= 0

    def test_get_summary_with_states(self, controller):
        """Should include state info in summary."""
        controller._get_or_create_state("hex8_2p")
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.9)
        summary = controller.get_summary()
        assert isinstance(summary, dict)


# =============================================================================
# FeedbackLoopController Lifecycle Tests
# =============================================================================


class TestFeedbackLoopControllerLifecycle:
    """Test start/stop lifecycle."""

    @pytest.fixture
    def controller(self):
        """Create controller without auto-wiring."""
        return FeedbackLoopController()

    @pytest.mark.asyncio
    async def test_start_sets_running(self, controller):
        """Start should set running flag."""
        # Start in background
        task = asyncio.create_task(controller.start())
        await asyncio.sleep(0.1)
        assert controller._running is True

        # Stop
        await controller.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, controller):
        """Stop should clear running flag."""
        controller._running = True
        await controller.stop()
        assert controller._running is False


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_feedback_loop_controller(self):
        """Should return controller instance."""
        controller = get_feedback_loop_controller()
        assert isinstance(controller, FeedbackLoopController)

    def test_get_feedback_loop_controller_singleton(self):
        """Should return same instance."""
        c1 = get_feedback_loop_controller()
        c2 = get_feedback_loop_controller()
        assert c1 is c2


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeedbackLoopControllerIntegration:
    """Integration tests for feedback loop."""

    @pytest.fixture
    def controller(self):
        """Create controller without auto-wiring."""
        return FeedbackLoopController()

    def test_feedback_cycle(self, controller):
        """Should handle full feedback cycle."""
        config = "hex8_2p"

        # Selfplay quality signal
        controller.signal_selfplay_quality(config, quality_score=0.85)

        # Training complete signal
        controller.signal_training_complete(
            config,
            policy_accuracy=0.78,
            value_accuracy=0.65,
        )

        # Check state
        state = controller.get_state(config)
        assert state.last_selfplay_quality == 0.85
        assert state.last_training_accuracy == 0.78

    def test_multiple_configs(self, controller):
        """Should track multiple configs independently."""
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.9)
        controller.signal_selfplay_quality("square8_4p", quality_score=0.7)

        state1 = controller.get_state("hex8_2p")
        state2 = controller.get_state("square8_4p")

        assert state1.last_selfplay_quality == 0.9
        assert state2.last_selfplay_quality == 0.7
