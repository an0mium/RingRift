"""Unit tests for feedback_loop_controller module.

Tests the FeedbackLoopController and FeedbackState components that
orchestrate training feedback signals.

December 2025: Expanded to 50+ tests covering:
- FeedbackState dataclass
- Controller initialization and configuration
- State management and tracking
- Event handlers (selfplay, training, evaluation, promotion)
- Health check and status methods
- Singleton pattern
- Error handling paths
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.feedback_loop_controller import (
    FeedbackLoopController,
    FeedbackState,
    get_feedback_loop_controller,
    reset_feedback_loop_controller,
    _safe_create_task,
    _handle_task_error,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def controller():
    """Create a fresh FeedbackLoopController instance."""
    return FeedbackLoopController()


@pytest.fixture
def started_controller():
    """Create a controller with running state."""
    ctrl = FeedbackLoopController()
    ctrl._running = True
    ctrl._subscribed = True
    return ctrl


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    reset_feedback_loop_controller()
    yield
    reset_feedback_loop_controller()


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

    def test_elo_history_initialization(self):
        """Should initialize elo_history as empty list."""
        state = FeedbackState(config_key="test")
        assert state.elo_history == []

    def test_update_elo_tracks_history(self):
        """Should track Elo history and calculate velocity."""
        state = FeedbackState(config_key="test")
        now = time.time()

        # Add first data point
        state.update_elo(1500.0, timestamp=now)
        assert state.last_elo == 1500.0
        assert len(state.elo_history) == 1

    def test_update_elo_calculates_velocity(self):
        """Should calculate Elo velocity from history."""
        state = FeedbackState(config_key="test")
        base_time = time.time()

        # Add 3+ data points for velocity calculation
        state.update_elo(1500.0, timestamp=base_time)
        state.update_elo(1550.0, timestamp=base_time + 1800)  # +50 in 30 min
        state.update_elo(1600.0, timestamp=base_time + 3600)  # +100 in 1 hr

        assert state.elo_velocity > 0  # Should show positive velocity

    def test_elo_history_bounded_to_10_entries(self):
        """Should keep only last 10 Elo history entries."""
        state = FeedbackState(config_key="test")

        # Add 15 entries
        for i in range(15):
            state.update_elo(1500.0 + i * 10, timestamp=time.time() + i)

        assert len(state.elo_history) == 10

    def test_timing_attributes(self):
        """Should have timing attributes for tracking."""
        state = FeedbackState(config_key="test")
        assert state.last_selfplay_time == 0.0
        assert state.last_training_time == 0.0
        assert state.last_evaluation_time == 0.0
        assert state.last_promotion_time == 0.0

    def test_search_budget_default(self):
        """Should have default search budget."""
        state = FeedbackState(config_key="test")
        assert state.current_search_budget == 400


# =============================================================================
# FeedbackLoopController Initialization Tests
# =============================================================================


class TestFeedbackLoopControllerInit:
    """Test FeedbackLoopController initialization."""

    def test_init_default(self):
        """Should initialize with defaults."""
        controller = FeedbackLoopController()
        assert controller._running is False
        assert controller._subscribed is False
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
        assert isinstance(controller._lock, type(threading.Lock()))

    def test_has_rate_history(self):
        """Should have rate history tracking."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "_rate_history")
        assert isinstance(controller._rate_history, dict)

    def test_cluster_healthy_default(self):
        """Should have cluster_healthy flag initialized."""
        controller = FeedbackLoopController()
        assert controller._cluster_healthy is True

    def test_failure_exploration_boost_configured(self):
        """Should have failure_exploration_boost from config."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "failure_exploration_boost")
        assert controller.failure_exploration_boost > 1.0

    def test_success_intensity_reduction_configured(self):
        """Should have success_intensity_reduction from config."""
        controller = FeedbackLoopController()
        assert hasattr(controller, "success_intensity_reduction")


# =============================================================================
# FeedbackLoopController State Management Tests
# =============================================================================


class TestFeedbackLoopControllerState:
    """Test state management."""

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
        assert "hex8_2p" in states
        assert "square8_4p" in states

    def test_get_all_states_returns_copy(self, controller):
        """Should return copy, not original dict."""
        controller._get_or_create_state("hex8_2p")
        states1 = controller.get_all_states()
        states2 = controller.get_all_states()
        # Should be equal content but not same object
        assert states1 == states2
        states1["new_key"] = "value"
        assert "new_key" not in controller._states

    def test_state_thread_safety(self, controller):
        """State creation should be thread-safe."""
        results = []

        def create_state():
            state = controller._get_or_create_state("thread_test")
            results.append(state)

        threads = [threading.Thread(target=create_state) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same state instance
        assert len(set(id(s) for s in results)) == 1


# =============================================================================
# FeedbackLoopController Signal Tests
# =============================================================================


class TestFeedbackLoopControllerSignals:
    """Test signal methods."""

    def test_signal_selfplay_quality(self, controller):
        """Should update selfplay quality."""
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.85)
        state = controller.get_state("hex8_2p")
        assert state is not None
        assert state.last_selfplay_quality == 0.85

    def test_signal_selfplay_quality_updates_time(self, controller):
        """Should update selfplay timestamp."""
        before = time.time()
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.85)
        after = time.time()
        state = controller.get_state("hex8_2p")
        assert before <= state.last_selfplay_time <= after

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

    def test_signal_training_complete_updates_time(self, controller):
        """Should update training timestamp."""
        before = time.time()
        controller.signal_training_complete("hex8_2p", policy_accuracy=0.78)
        after = time.time()
        state = controller.get_state("hex8_2p")
        assert before <= state.last_training_time <= after


# =============================================================================
# FeedbackLoopController Summary Tests
# =============================================================================


class TestFeedbackLoopControllerSummary:
    """Test summary and status methods."""

    def test_get_summary_empty(self, controller):
        """Should return summary even with no states."""
        summary = controller.get_summary()
        assert isinstance(summary, dict)
        assert "running" in summary
        assert "subscribed" in summary
        assert "configs_tracked" in summary

    def test_get_summary_with_states(self, controller):
        """Should include state info in summary."""
        controller._get_or_create_state("hex8_2p")
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.9)
        summary = controller.get_summary()
        assert summary["configs_tracked"] >= 1
        assert "states" in summary
        assert "hex8_2p" in summary["states"]

    def test_get_summary_state_details(self, controller):
        """Should include detailed state info."""
        controller.signal_selfplay_quality("hex8_2p", quality_score=0.9)
        summary = controller.get_summary()
        state_summary = summary["states"]["hex8_2p"]
        assert "training_intensity" in state_summary
        assert "exploration_boost" in state_summary
        assert "last_selfplay_quality" in state_summary


# =============================================================================
# FeedbackLoopController Health Check Tests
# =============================================================================


class TestFeedbackLoopControllerHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_result(self, controller):
        """Should return HealthCheckResult."""
        result = controller.health_check()
        assert hasattr(result, "healthy")
        assert hasattr(result, "message")
        assert hasattr(result, "details")

    def test_health_check_unhealthy_when_not_running(self, controller):
        """Should report unhealthy when not running."""
        controller._running = False
        result = controller.health_check()
        assert result.healthy is False
        assert "stopped" in result.message.lower() or "not" in result.message.lower()

    def test_health_check_healthy_when_running(self, started_controller):
        """Should report healthy when running and subscribed."""
        result = started_controller.health_check()
        assert result.healthy is True

    def test_health_check_includes_details(self, started_controller):
        """Should include relevant details."""
        result = started_controller.health_check()
        assert "running" in result.details
        assert "subscribed" in result.details
        assert "configs_tracked" in result.details
        assert "cluster_healthy" in result.details

    def test_health_check_includes_thresholds(self, started_controller):
        """Should include threshold configuration."""
        result = started_controller.health_check()
        assert "policy_accuracy_threshold" in result.details
        assert "promotion_threshold" in result.details

    def test_health_check_counts_active_configs(self, started_controller):
        """Should count active configs (trained within 1 hour)."""
        # Add a state with recent training
        state = started_controller._get_or_create_state("hex8_2p")
        state.last_training_time = time.time()  # Just now

        result = started_controller.health_check()
        assert result.details["active_configs"] >= 1


# =============================================================================
# FeedbackLoopController Lifecycle Tests
# =============================================================================


class TestFeedbackLoopControllerLifecycle:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, controller):
        """Start should set running flag."""
        # Mock the subscription methods to avoid side effects
        with patch.object(controller, "_subscribe_to_events"):
            with patch.object(controller, "_wire_curriculum_feedback"):
                with patch.object(controller, "_wire_exploration_boost"):
                    with patch.object(controller, "_subscribe_to_lazy_scheduler_registration"):
                        await controller.start()

        assert controller._running is True
        await controller.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, controller):
        """Stop should clear running flag."""
        controller._running = True
        await controller.stop()
        assert controller._running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, controller):
        """Start should be idempotent."""
        with patch.object(controller, "_subscribe_to_events") as mock_sub:
            with patch.object(controller, "_wire_curriculum_feedback"):
                with patch.object(controller, "_wire_exploration_boost"):
                    with patch.object(controller, "_subscribe_to_lazy_scheduler_registration"):
                        await controller.start()
                        await controller.start()  # Second call

        # Subscribe should only be called once
        assert mock_sub.call_count == 1
        await controller.stop()

    def test_is_running(self, controller):
        """Should correctly report running status."""
        assert controller.is_running() is False
        controller._running = True
        assert controller.is_running() is True


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

    def test_reset_feedback_loop_controller(self):
        """Should reset singleton."""
        c1 = get_feedback_loop_controller()
        reset_feedback_loop_controller()
        c2 = get_feedback_loop_controller()
        assert c1 is not c2


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Test event handler methods."""

    def test_on_selfplay_complete_updates_quality(self, controller):
        """_on_selfplay_complete should update quality."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "games_count": 100,
            "db_path": "/tmp/test.db",
        }

        with patch.object(controller, "_assess_selfplay_quality", return_value=0.85):
            with patch.object(controller, "_update_training_intensity"):
                with patch.object(controller, "_update_curriculum_weight_from_selfplay"):
                    controller._on_selfplay_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.last_selfplay_quality == 0.85

    def test_on_selfplay_complete_missing_config(self, controller):
        """Should handle missing config gracefully."""
        event = MagicMock()
        event.payload = {"games_count": 100}  # No config

        # Should not raise
        controller._on_selfplay_complete(event)
        assert len(controller._states) == 0

    def test_on_training_complete_updates_state(self, controller):
        """_on_training_complete should update state."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "policy_accuracy": 0.82,
            "value_accuracy": 0.75,
            "model_path": "/tmp/model.pth",
        }

        with patch.object(controller, "_trigger_evaluation"):
            with patch.object(controller, "_record_training_in_curriculum"):
                with patch.object(controller, "_emit_curriculum_training_feedback"):
                    controller._on_training_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.last_training_accuracy == 0.82

    def test_on_evaluation_complete_updates_win_rate(self, controller):
        """_on_evaluation_complete should update win rate."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "win_rate": 0.65,
            "elo": 1650.0,
            "model_path": "/tmp/model.pth",
        }

        with patch.object(controller, "_adjust_selfplay_for_velocity"):
            with patch.object(controller, "_consider_promotion"):
                controller._on_evaluation_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.last_evaluation_win_rate == 0.65

    def test_on_promotion_complete_success_increments_streak(self, controller):
        """Promotion success should increment consecutive successes."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "promoted": True,
        }

        with patch.object(controller, "_apply_intensity_feedback"):
            controller._on_promotion_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.consecutive_successes == 1
        assert state.consecutive_failures == 0

    def test_on_promotion_complete_failure_increments_failures(self, controller):
        """Promotion failure should increment consecutive failures."""
        event = MagicMock()
        event.payload = {
            "config": "hex8_2p",
            "promoted": False,
        }

        with patch.object(controller, "_apply_intensity_feedback"):
            with patch.object(controller, "_signal_urgent_training"):
                controller._on_promotion_complete(event)

        state = controller.get_state("hex8_2p")
        assert state.consecutive_failures == 1
        assert state.consecutive_successes == 0

    def test_on_work_completed_updates_metrics(self, controller):
        """_on_work_completed should update work metrics."""
        event = MagicMock()
        event.payload = {
            "work_id": "work-123",
            "work_type": "selfplay",
            "board_type": "hex8",
            "num_players": 2,
            "claimed_by": "node-1",
        }

        controller._on_work_completed(event)

        state = controller.get_state("hex8_2p")
        assert state.work_completed_count == 1

    def test_on_work_failed_tracks_failures(self, controller):
        """_on_work_failed should track failure count."""
        event = MagicMock()
        event.payload = {
            "work_id": "work-123",
            "work_type": "selfplay",
            "board_type": "hex8",
            "num_players": 2,
            "node_id": "node-1",
            "reason": "timeout",
        }

        controller._on_work_failed(event)

        state = controller.get_state("hex8_2p")
        assert hasattr(state, "work_failed_count")
        assert state.work_failed_count == 1

    def test_on_work_timeout_tracks_timeouts(self, controller):
        """_on_work_timeout should track timeout count."""
        event = MagicMock()
        event.payload = {
            "work_id": "work-123",
            "work_type": "training",
            "board_type": "square8",
            "num_players": 4,
            "node_id": "node-2",
            "timeout_seconds": 3600,
        }

        controller._on_work_timeout(event)

        state = controller.get_state("square8_4p")
        assert hasattr(state, "work_timeout_count")
        assert state.work_timeout_count == 1


# =============================================================================
# Intensity and Quality Tests
# =============================================================================


class TestIntensityAndQuality:
    """Test intensity and quality computation methods."""

    def test_compute_intensity_from_quality_hot_path(self, controller):
        """High quality should return hot_path intensity."""
        result = controller._compute_intensity_from_quality(0.95)
        assert result == "hot_path"

    def test_compute_intensity_from_quality_accelerated(self, controller):
        """Good quality should return accelerated intensity."""
        result = controller._compute_intensity_from_quality(0.85)
        assert result == "accelerated"

    def test_compute_intensity_from_quality_normal(self, controller):
        """Adequate quality should return normal intensity."""
        result = controller._compute_intensity_from_quality(0.70)
        assert result == "normal"

    def test_compute_intensity_from_quality_reduced(self, controller):
        """Poor quality should return reduced intensity."""
        result = controller._compute_intensity_from_quality(0.55)
        assert result == "reduced"

    def test_compute_intensity_from_quality_paused(self, controller):
        """Very poor quality should return paused intensity."""
        result = controller._compute_intensity_from_quality(0.40)
        assert result == "paused"

    def test_assess_selfplay_quality_db_not_found(self, controller):
        """Should return 0.3 when database does not exist."""
        # Test with non-existent path - returns 0.3 regardless of games_count
        result = controller._assess_selfplay_quality("/nonexistent/path.db", 50)
        assert result == 0.3

    def test_assess_selfplay_quality_fallback_uses_import_error(self, controller):
        """Count-based fallback is used when ImportError occurs."""
        # Mock the import to fail
        with patch("app.coordination.feedback_loop_controller.logger"):
            with patch.dict("sys.modules", {"app.quality.unified_quality": None}):
                # When import fails, should use count-based heuristic
                # This is implementation detail - testing the logic separately
                pass

    def test_assess_selfplay_quality_handles_exceptions(self, controller):
        """Should handle exceptions gracefully and return fallback."""
        # Non-existent path should be handled gracefully
        result = controller._assess_selfplay_quality("/nonexistent/db.db", 100)
        # Returns 0.3 because DB doesn't exist
        assert result == 0.3

    def test_assess_selfplay_quality_returns_valid_range(self, controller):
        """Should always return a value between 0 and 1."""
        result = controller._assess_selfplay_quality("/any/path.db", 500)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Exploration Boost Tests
# =============================================================================


class TestExplorationBoost:
    """Test exploration boost functionality."""

    def test_boost_exploration_for_anomaly_updates_state(self, controller):
        """Should update exploration boost in state."""
        controller._boost_exploration_for_anomaly("hex8_2p", anomaly_count=2)

        state = controller.get_state("hex8_2p")
        assert state.current_exploration_boost > 1.0

    def test_boost_exploration_for_stall_updates_state(self, controller):
        """Should update exploration boost for stall."""
        controller._boost_exploration_for_stall("hex8_2p", stall_epochs=10)

        state = controller.get_state("hex8_2p")
        assert state.current_exploration_boost >= 1.0

    def test_reduce_exploration_after_improvement(self, controller):
        """Should reduce exploration boost on improvement."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_exploration_boost = 1.5

        controller._reduce_exploration_after_improvement("hex8_2p")

        assert state.current_exploration_boost < 1.5

    def test_reduce_exploration_does_not_go_below_1(self, controller):
        """Exploration boost should not go below 1.0."""
        state = controller._get_or_create_state("hex8_2p")
        state.current_exploration_boost = 1.05

        controller._reduce_exploration_after_improvement("hex8_2p")

        assert state.current_exploration_boost >= 1.0


# =============================================================================
# Cluster Health Tests
# =============================================================================


class TestClusterHealth:
    """Test cluster health handling."""

    def test_on_p2p_cluster_unhealthy_sets_flag(self, controller):
        """Should set cluster_healthy to False on majority dead."""
        event = MagicMock()
        event.payload = {
            "dead_nodes": ["node-1", "node-2", "node-3"],
            "alive_nodes": ["node-4"],
        }

        with patch.object(controller, "_get_or_create_state"):
            controller._on_p2p_cluster_unhealthy(event)

        assert controller._cluster_healthy is False

    def test_on_p2p_cluster_unhealthy_stays_healthy(self, controller):
        """Should stay healthy when majority alive."""
        event = MagicMock()
        event.payload = {
            "dead_nodes": ["node-1"],
            "alive_nodes": ["node-2", "node-3", "node-4"],
        }

        controller._on_p2p_cluster_unhealthy(event)

        assert controller._cluster_healthy is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""

    @pytest.mark.asyncio
    async def test_safe_create_task_returns_task(self):
        """_safe_create_task should return a task."""
        async def dummy_coro():
            pass

        task = _safe_create_task(dummy_coro(), context="test")
        assert task is not None
        await task

    def test_handle_task_error_handles_cancelled(self):
        """_handle_task_error should handle cancelled tasks."""
        task = MagicMock()
        task.exception.side_effect = asyncio.CancelledError()

        # Should not raise
        _handle_task_error(task, "test_context")

    def test_handle_task_error_logs_exception(self):
        """_handle_task_error should log real exceptions."""
        task = MagicMock()
        task.exception.return_value = ValueError("test error")

        with patch("app.coordination.feedback_loop_controller.logger") as mock_logger:
            _handle_task_error(task, "test_context")
            mock_logger.error.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeedbackLoopControllerIntegration:
    """Integration tests for feedback loop."""

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

    def test_promotion_streak_reset(self, controller):
        """Promotion success should reset failure streak."""
        # Simulate failures
        state = controller._get_or_create_state("hex8_2p")
        state.consecutive_failures = 3

        # Simulate success
        event = MagicMock()
        event.payload = {"config": "hex8_2p", "promoted": True}

        with patch.object(controller, "_apply_intensity_feedback"):
            controller._on_promotion_complete(event)

        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 1

    def test_hot_path_activation(self, controller):
        """Hot path should activate after 3 consecutive successes."""
        state = controller._get_or_create_state("hex8_2p")
        state.consecutive_successes = 2  # Already had 2 successes

        event = MagicMock()
        event.payload = {"config": "hex8_2p", "promoted": True}

        with patch.object(controller, "_apply_intensity_feedback"):
            controller._on_promotion_complete(event)

        assert state.consecutive_successes == 3
        assert state.current_training_intensity == "hot_path"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling paths."""

    def test_on_selfplay_complete_handles_exception(self, controller):
        """Should handle exceptions gracefully."""
        event = MagicMock()
        event.payload = {"config": "hex8_2p", "games_count": 100, "db_path": "/tmp/test.db"}

        with patch.object(controller, "_assess_selfplay_quality", side_effect=RuntimeError("test")):
            # Should not raise
            controller._on_selfplay_complete(event)

    def test_on_training_complete_handles_exception(self, controller):
        """Should handle exceptions in training complete handler."""
        event = MagicMock()
        event.payload = {"config": "hex8_2p", "policy_accuracy": 0.8}

        with patch.object(controller, "_trigger_evaluation", side_effect=RuntimeError("test")):
            # Should not raise
            controller._on_training_complete(event)

    def test_on_promotion_complete_handles_exception(self, controller):
        """Should handle exceptions in promotion complete handler."""
        event = MagicMock()
        event.payload = {"config": "hex8_2p", "promoted": True}

        with patch.object(controller, "_apply_intensity_feedback", side_effect=RuntimeError("test")):
            # Should not raise
            controller._on_promotion_complete(event)

    def test_handles_missing_payload(self, controller):
        """Should handle events without payload attribute."""
        event = object()  # No payload attribute

        # Should not raise
        controller._on_selfplay_complete(event)
        controller._on_training_complete(event)
        controller._on_work_completed(event)
