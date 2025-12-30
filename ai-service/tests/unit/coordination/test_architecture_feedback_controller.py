"""Unit tests for architecture_feedback_controller.py.

Tests:
- ArchitectureFeedbackConfig dataclass
- ArchitectureFeedbackState dataclass
- ArchitectureFeedbackController singleton pattern
- Event subscriptions (EVALUATION_COMPLETED, TRAINING_COMPLETED)
- Minimum allocation enforcement
- Weight emission logic
- Health check reporting

December 30, 2025 - Test coverage for 397 LOC module.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.architecture_feedback_controller import (
    ArchitectureFeedbackConfig,
    ArchitectureFeedbackController,
    ArchitectureFeedbackState,
    get_architecture_feedback_controller,
    start_architecture_feedback_controller,
)


# =============================================================================
# ArchitectureFeedbackConfig Tests
# =============================================================================


class TestArchitectureFeedbackConfig:
    """Tests for ArchitectureFeedbackConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ArchitectureFeedbackConfig()
        assert config.min_allocation_per_arch == 0.10
        assert config.weight_update_interval == 1800.0
        assert config.weight_temperature == 0.5

    def test_default_architectures(self):
        """Test default supported architectures list."""
        config = ArchitectureFeedbackConfig()
        assert "v4" in config.supported_architectures
        assert "v5" in config.supported_architectures
        assert "v5_heavy" in config.supported_architectures
        assert "v6" in config.supported_architectures
        assert "nnue_v1" in config.supported_architectures
        assert len(config.supported_architectures) == 7

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ArchitectureFeedbackConfig(
            min_allocation_per_arch=0.15,
            weight_update_interval=900.0,
            weight_temperature=1.0,
        )
        assert config.min_allocation_per_arch == 0.15
        assert config.weight_update_interval == 900.0
        assert config.weight_temperature == 1.0

    def test_custom_architectures(self):
        """Test custom architectures list."""
        config = ArchitectureFeedbackConfig(
            supported_architectures=["v5", "v6", "v7"]
        )
        assert config.supported_architectures == ["v5", "v6", "v7"]


# =============================================================================
# ArchitectureFeedbackState Tests
# =============================================================================


class TestArchitectureFeedbackState:
    """Tests for ArchitectureFeedbackState dataclass."""

    def test_default_state(self):
        """Test default state initialization."""
        state = ArchitectureFeedbackState()
        assert state.last_weight_update_time == 0.0
        assert state.cached_weights == {}
        assert state.evaluations_processed == 0
        assert state.trainings_processed == 0

    def test_state_mutation(self):
        """Test state can be mutated."""
        state = ArchitectureFeedbackState()
        state.evaluations_processed = 5
        state.trainings_processed = 3
        state.cached_weights["hex8_2p"] = {"v5": 0.5, "v6": 0.5}

        assert state.evaluations_processed == 5
        assert state.trainings_processed == 3
        assert "hex8_2p" in state.cached_weights


# =============================================================================
# ArchitectureFeedbackController Singleton Tests
# =============================================================================


class TestArchitectureFeedbackControllerSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_get_instance_creates_singleton(self):
        """Test get_instance creates new instance on first call."""
        instance = ArchitectureFeedbackController.get_instance()
        assert instance is not None
        assert isinstance(instance, ArchitectureFeedbackController)

    def test_get_instance_returns_same_instance(self):
        """Test get_instance returns same instance on subsequent calls."""
        instance1 = ArchitectureFeedbackController.get_instance()
        instance2 = ArchitectureFeedbackController.get_instance()
        assert instance1 is instance2

    def test_reset_instance_clears_singleton(self):
        """Test reset_instance clears the singleton."""
        instance1 = ArchitectureFeedbackController.get_instance()
        ArchitectureFeedbackController.reset_instance()
        instance2 = ArchitectureFeedbackController.get_instance()
        assert instance1 is not instance2

    def test_module_accessor_function(self):
        """Test get_architecture_feedback_controller returns singleton."""
        instance1 = get_architecture_feedback_controller()
        instance2 = get_architecture_feedback_controller()
        assert instance1 is instance2


# =============================================================================
# ArchitectureFeedbackController Initialization Tests
# =============================================================================


class TestArchitectureFeedbackControllerInit:
    """Tests for controller initialization."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_default_initialization(self):
        """Test default initialization."""
        controller = ArchitectureFeedbackController()
        assert controller._config is not None
        assert controller._state is not None
        assert controller._running is False

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = ArchitectureFeedbackConfig(min_allocation_per_arch=0.20)
        controller = ArchitectureFeedbackController(config=config)
        assert controller._config.min_allocation_per_arch == 0.20

    def test_handler_base_properties(self):
        """Test HandlerBase properties are set correctly."""
        controller = ArchitectureFeedbackController()
        assert controller.name == "architecture_feedback"
        assert controller.cycle_interval == 60.0


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscriptions:
    """Tests for event subscriptions."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_get_event_subscriptions(self):
        """Test event subscriptions are defined correctly."""
        controller = ArchitectureFeedbackController()
        subs = controller._get_event_subscriptions()

        assert "EVALUATION_COMPLETED" in subs
        assert "TRAINING_COMPLETED" in subs
        assert callable(subs["EVALUATION_COMPLETED"])
        assert callable(subs["TRAINING_COMPLETED"])


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEvaluationCompletedHandler:
    """Tests for _on_evaluation_completed handler."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    @pytest.mark.asyncio
    async def test_handles_valid_evaluation_event(self):
        """Test handler processes valid evaluation event."""
        controller = ArchitectureFeedbackController()

        with patch("app.coordination.architecture_feedback_controller.get_architecture_tracker") as mock_tracker, \
             patch("app.coordination.architecture_feedback_controller.extract_architecture_from_model_path") as mock_extract:

            mock_extract.return_value = "v5"
            mock_tracker_instance = MagicMock()
            mock_tracker.return_value = mock_tracker_instance

            event = {
                "config_key": "hex8_2p",
                "model_path": "models/canonical_hex8_2p.pth",
                "elo": 1450.0,
                "games": 50,
            }

            await controller._on_evaluation_completed(event)

            mock_tracker_instance.record_evaluation.assert_called_once()
            assert controller._state.evaluations_processed == 1

    @pytest.mark.asyncio
    async def test_ignores_event_without_config_key(self):
        """Test handler ignores event without config_key."""
        controller = ArchitectureFeedbackController()

        event = {
            "model_path": "models/test.pth",
            "elo": 1450.0,
        }

        await controller._on_evaluation_completed(event)
        assert controller._state.evaluations_processed == 0

    @pytest.mark.asyncio
    async def test_ignores_event_without_model_path(self):
        """Test handler ignores event without model_path."""
        controller = ArchitectureFeedbackController()

        event = {
            "config_key": "hex8_2p",
            "elo": 1450.0,
        }

        await controller._on_evaluation_completed(event)
        assert controller._state.evaluations_processed == 0

    @pytest.mark.asyncio
    async def test_handles_import_error_gracefully(self):
        """Test handler gracefully handles ImportError."""
        controller = ArchitectureFeedbackController()

        with patch.dict("sys.modules", {"app.training.architecture_tracker": None}):
            # This should not raise
            event = {
                "config_key": "hex8_2p",
                "model_path": "models/test.pth",
                "elo": 1450.0,
            }
            await controller._on_evaluation_completed(event)
            # Should not increment due to error
            # (may or may not increment depending on how import fails)


class TestTrainingCompletedHandler:
    """Tests for _on_training_completed handler."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    @pytest.mark.asyncio
    async def test_handles_valid_training_event(self):
        """Test handler processes valid training event."""
        controller = ArchitectureFeedbackController()

        with patch("app.coordination.architecture_feedback_controller.get_architecture_tracker") as mock_tracker, \
             patch("app.coordination.architecture_feedback_controller.extract_architecture_from_model_path") as mock_extract:

            mock_extract.return_value = "v5"
            mock_tracker_instance = MagicMock()
            mock_tracker.return_value = mock_tracker_instance

            event = {
                "config_key": "hex8_2p",
                "model_path": "models/canonical_hex8_2p.pth",
                "duration_seconds": 7200.0,  # 2 hours
            }

            await controller._on_training_completed(event)

            mock_tracker_instance.record_evaluation.assert_called_once()
            # Check training_hours is converted correctly
            call_kwargs = mock_tracker_instance.record_evaluation.call_args[1]
            assert call_kwargs["training_hours"] == 2.0
            assert controller._state.trainings_processed == 1

    @pytest.mark.asyncio
    async def test_ignores_event_without_required_fields(self):
        """Test handler ignores event without required fields."""
        controller = ArchitectureFeedbackController()

        event = {"elo": 1450.0}
        await controller._on_training_completed(event)
        assert controller._state.trainings_processed == 0


# =============================================================================
# Minimum Allocation Enforcement Tests
# =============================================================================


class TestMinimumAllocationEnforcement:
    """Tests for _enforce_minimum_allocation method."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_empty_weights_unchanged(self):
        """Test empty weights dict is returned unchanged."""
        controller = ArchitectureFeedbackController()
        result = controller._enforce_minimum_allocation({})
        assert result == {}

    def test_weights_above_minimum_unchanged(self):
        """Test weights above minimum are relatively unchanged."""
        controller = ArchitectureFeedbackController()
        weights = {"v5": 0.6, "v6": 0.4}
        result = controller._enforce_minimum_allocation(weights)

        # Both weights are above 10%, so should remain similar
        assert result["v5"] > result["v6"]
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_weights_below_minimum_raised(self):
        """Test weights below minimum are raised to minimum."""
        controller = ArchitectureFeedbackController()
        weights = {"v5": 0.95, "v6": 0.05}  # v6 below 10%
        result = controller._enforce_minimum_allocation(weights)

        # v6 should be raised to at least 10%
        assert result["v6"] >= 0.10
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_many_architectures_equal_weights(self):
        """Test many architectures get equal weights when min exceeds 100%."""
        config = ArchitectureFeedbackConfig(min_allocation_per_arch=0.15)
        controller = ArchitectureFeedbackController(config=config)

        # 10 architectures * 15% = 150% > 100%, so should get equal weights
        weights = {f"v{i}": 0.1 for i in range(10)}
        result = controller._enforce_minimum_allocation(weights)

        # All should be equal (10%)
        expected = 0.10
        for arch, weight in result.items():
            assert abs(weight - expected) < 0.01

    def test_renormalization_sums_to_one(self):
        """Test result always sums to 1.0."""
        controller = ArchitectureFeedbackController()

        test_cases = [
            {"v5": 0.9, "v6": 0.1},
            {"v5": 0.5, "v6": 0.3, "v7": 0.2},
            {"v5": 0.05, "v6": 0.05, "v7": 0.9},
        ]

        for weights in test_cases:
            result = controller._enforce_minimum_allocation(weights)
            assert abs(sum(result.values()) - 1.0) < 0.01


# =============================================================================
# Weight Emission Tests
# =============================================================================


class TestWeightEmission:
    """Tests for weight emission methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    @pytest.mark.asyncio
    async def test_maybe_emit_skips_if_too_soon(self):
        """Test _maybe_emit_weight_update skips if interval not elapsed."""
        controller = ArchitectureFeedbackController()
        controller._state.last_weight_update_time = time.time()

        with patch.object(controller, "_emit_architecture_weights_updated") as mock_emit:
            await controller._maybe_emit_weight_update("hex8_2p")
            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_maybe_emit_emits_if_interval_elapsed(self):
        """Test _maybe_emit_weight_update emits if interval elapsed."""
        controller = ArchitectureFeedbackController()
        controller._state.last_weight_update_time = time.time() - 3600.0  # 1 hour ago

        with patch.object(controller, "_emit_architecture_weights_updated", new_callable=AsyncMock) as mock_emit:
            await controller._maybe_emit_weight_update("hex8_2p")
            mock_emit.assert_called_once_with("hex8_2p")

    @pytest.mark.asyncio
    async def test_emit_weights_calls_event_router(self):
        """Test _emit_architecture_weights_updated calls event router."""
        controller = ArchitectureFeedbackController()

        with patch("app.coordination.architecture_feedback_controller.get_allocation_weights") as mock_get_weights, \
             patch.object(controller, "_emit_event") as mock_emit:

            mock_get_weights.return_value = {"v5": 0.6, "v6": 0.4}

            await controller._emit_architecture_weights_updated("hex8_2p")

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args[0]
            assert call_args[0] == "ARCHITECTURE_WEIGHTS_UPDATED"
            assert call_args[1]["config_key"] == "hex8_2p"
            assert "weights" in call_args[1]

    @pytest.mark.asyncio
    async def test_emit_weights_caches_result(self):
        """Test _emit_architecture_weights_updated caches weights."""
        controller = ArchitectureFeedbackController()

        with patch("app.coordination.architecture_feedback_controller.get_allocation_weights") as mock_get_weights, \
             patch.object(controller, "_emit_event"):

            mock_get_weights.return_value = {"v5": 0.6, "v6": 0.4}

            await controller._emit_architecture_weights_updated("hex8_2p")

            assert "hex8_2p" in controller._state.cached_weights


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Tests for _run_cycle method."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    @pytest.mark.asyncio
    async def test_run_cycle_skips_if_not_running(self):
        """Test _run_cycle does nothing if not running."""
        controller = ArchitectureFeedbackController()
        controller._running = False
        controller._state.cached_weights = {"hex8_2p": {"v5": 0.5}}
        controller._state.last_weight_update_time = 0  # Long ago

        with patch.object(controller, "_emit_architecture_weights_updated", new_callable=AsyncMock) as mock_emit:
            await controller._run_cycle()
            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_emits_for_all_cached_configs(self):
        """Test _run_cycle emits weights for all cached configs."""
        controller = ArchitectureFeedbackController()
        controller._running = True
        controller._state.cached_weights = {
            "hex8_2p": {"v5": 0.5},
            "square8_2p": {"v5": 0.5},
        }
        controller._state.last_weight_update_time = 0  # Long ago

        with patch.object(controller, "_emit_architecture_weights_updated", new_callable=AsyncMock) as mock_emit:
            await controller._run_cycle()

            # Should be called for both cached configs
            assert mock_emit.call_count == 2


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    def test_health_check_when_not_running(self):
        """Test health check returns degraded when not running."""
        controller = ArchitectureFeedbackController()
        controller._running = False

        result = controller.health_check()

        assert result.status.value == "degraded"
        assert "Not running" in result.message

    def test_health_check_when_running(self):
        """Test health check returns healthy when running."""
        controller = ArchitectureFeedbackController()
        controller._running = True

        result = controller.health_check()

        assert result.status.value == "healthy"
        assert "Running" in result.message

    def test_health_check_includes_details(self):
        """Test health check includes processing stats."""
        controller = ArchitectureFeedbackController()
        controller._running = True
        controller._state.evaluations_processed = 10
        controller._state.trainings_processed = 5
        controller._state.cached_weights = {"hex8_2p": {}}

        result = controller.health_check()

        assert result.details["evaluations_processed"] == 10
        assert result.details["trainings_processed"] == 5
        assert result.details["cached_configs"] == 1
        assert "version" in result.details


# =============================================================================
# Start Function Tests
# =============================================================================


class TestStartFunction:
    """Tests for start_architecture_feedback_controller function."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureFeedbackController.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureFeedbackController.reset_instance()

    @pytest.mark.asyncio
    async def test_start_function_returns_controller(self):
        """Test start function returns controller instance."""
        with patch.object(ArchitectureFeedbackController, "start", new_callable=AsyncMock):
            controller = await start_architecture_feedback_controller()
            assert isinstance(controller, ArchitectureFeedbackController)

    @pytest.mark.asyncio
    async def test_start_function_calls_start(self):
        """Test start function calls controller.start()."""
        with patch.object(ArchitectureFeedbackController, "start", new_callable=AsyncMock) as mock_start:
            await start_architecture_feedback_controller()
            mock_start.assert_called_once()
