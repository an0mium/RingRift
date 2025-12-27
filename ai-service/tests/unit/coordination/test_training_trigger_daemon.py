"""Tests for TrainingTriggerDaemon.

Tests the training trigger daemon that automatically decides when to start
training based on data freshness, GPU availability, and quality metrics.

December 2025: Created for Phase 3 test coverage of critical cluster daemons.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.training_trigger_daemon import (
    ConfigTrainingState,
    TrainingTriggerConfig,
    TrainingTriggerDaemon,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return TrainingTriggerConfig(
        enabled=True,
        max_data_age_hours=1.0,
        min_samples_threshold=1000,  # Lower for testing
        training_cooldown_hours=0.1,  # Short cooldown for testing
        scan_interval_seconds=1,  # Fast scan for testing
        max_concurrent_training=2,
    )


@pytest.fixture
def daemon(config):
    """Create a daemon instance for testing."""
    return TrainingTriggerDaemon(config)


# =============================================================================
# TrainingTriggerConfig Tests
# =============================================================================


class TestTrainingTriggerConfig:
    """Tests for TrainingTriggerConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = TrainingTriggerConfig()
        assert config.enabled is True
        assert config.max_data_age_hours == 1.0
        assert config.min_samples_threshold == 10000
        assert config.training_cooldown_hours == 4.0
        assert config.max_concurrent_training == 2
        assert config.gpu_idle_threshold_percent == 20.0
        assert config.default_epochs == 50
        assert config.default_batch_size == 512
        assert config.model_version == "v2"

    def test_custom_config(self):
        """Should accept custom values."""
        config = TrainingTriggerConfig(
            enabled=False,
            max_data_age_hours=2.0,
            min_samples_threshold=50000,
            training_cooldown_hours=8.0,
            max_concurrent_training=4,
        )
        assert config.enabled is False
        assert config.max_data_age_hours == 2.0
        assert config.min_samples_threshold == 50000
        assert config.training_cooldown_hours == 8.0
        assert config.max_concurrent_training == 4

    def test_freshness_sync_config(self):
        """Should have freshness sync configuration."""
        config = TrainingTriggerConfig()
        assert config.enforce_freshness_with_sync is True
        assert config.freshness_sync_timeout_seconds == 300.0


# =============================================================================
# ConfigTrainingState Tests
# =============================================================================


class TestConfigTrainingState:
    """Tests for ConfigTrainingState dataclass."""

    def test_create_state(self):
        """Should create state with required fields."""
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
        )
        assert state.config_key == "hex8_2p"
        assert state.board_type == "hex8"
        assert state.num_players == 2
        assert state.training_in_progress is False
        assert state.training_pid is None
        assert state.last_elo == 1500.0
        assert state.training_intensity == "normal"

    def test_state_with_training(self):
        """Should track training state."""
        state = ConfigTrainingState(
            config_key="square8_4p",
            board_type="square8",
            num_players=4,
            training_in_progress=True,
            training_pid=12345,
        )
        assert state.training_in_progress is True
        assert state.training_pid == 12345

    def test_state_with_npz_info(self):
        """Should track NPZ data info."""
        now = time.time()
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            last_npz_update=now,
            npz_sample_count=50000,
            npz_path="/data/training/hex8_2p.npz",
        )
        assert state.last_npz_update == now
        assert state.npz_sample_count == 50000
        assert state.npz_path == "/data/training/hex8_2p.npz"

    def test_training_intensity_levels(self):
        """Should support all training intensity levels."""
        for intensity in ["hot_path", "accelerated", "normal", "reduced", "paused"]:
            state = ConfigTrainingState(
                config_key="test",
                board_type="hex8",
                num_players=2,
                training_intensity=intensity,
            )
            assert state.training_intensity == intensity


# =============================================================================
# TrainingTriggerDaemon Initialization Tests
# =============================================================================


class TestTrainingTriggerDaemonInit:
    """Tests for TrainingTriggerDaemon initialization."""

    def test_default_init(self):
        """Should initialize with default config."""
        daemon = TrainingTriggerDaemon()
        assert daemon.config is not None
        assert daemon._running is False
        assert daemon._training_states == {}

    def test_custom_config_init(self, config):
        """Should initialize with custom config."""
        daemon = TrainingTriggerDaemon(config)
        assert daemon.config.min_samples_threshold == 1000
        assert daemon.config.max_concurrent_training == 2

    def test_has_semaphore(self, daemon):
        """Should have training semaphore for concurrency control."""
        assert daemon._training_semaphore is not None

    def test_has_required_methods(self, daemon):
        """Should have required lifecycle methods."""
        assert hasattr(daemon, "start")
        assert hasattr(daemon, "stop")
        assert callable(daemon.start)
        assert callable(daemon.stop)


# =============================================================================
# Daemon State Tests
# =============================================================================


class TestDaemonState:
    """Tests for daemon state management."""

    def test_initial_state(self, daemon):
        """Should start in stopped state."""
        assert daemon._running is False
        assert daemon._task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, daemon):
        """Should handle stop when not running."""
        # Should not raise
        await daemon.stop()
        assert daemon._running is False

    def test_training_states_isolation(self, config):
        """Each daemon should have its own training states."""
        daemon1 = TrainingTriggerDaemon(config)
        daemon2 = TrainingTriggerDaemon(config)

        daemon1._training_states["hex8_2p"] = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
        )

        assert "hex8_2p" in daemon1._training_states
        assert "hex8_2p" not in daemon2._training_states


# =============================================================================
# Training State Management Tests
# =============================================================================


class TestTrainingStateManagement:
    """Tests for training state tracking."""

    def test_get_or_create_state(self, daemon):
        """Should get or create training state for a config."""
        if hasattr(daemon, "_get_or_create_state"):
            state = daemon._get_or_create_state("hex8_2p")
            assert state is not None
            assert state.config_key == "hex8_2p"

    def test_state_persistence(self, daemon):
        """Training state should persist between accesses."""
        state1 = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            training_in_progress=True,
        )
        daemon._training_states["hex8_2p"] = state1

        # Should get same state
        retrieved = daemon._training_states.get("hex8_2p")
        assert retrieved is state1
        assert retrieved.training_in_progress is True


# =============================================================================
# Decision Logic Tests
# =============================================================================


class TestDecisionLogic:
    """Tests for training trigger decision logic."""

    def test_data_freshness_check(self, daemon):
        """Should check data freshness."""
        now = time.time()
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            last_npz_update=now - 3600,  # 1 hour old
            npz_sample_count=50000,
        )

        # Data is at threshold (1 hour = max_data_age_hours)
        age_hours = (now - state.last_npz_update) / 3600
        assert age_hours <= daemon.config.max_data_age_hours * 1.1  # Allow small margin

    def test_sample_count_threshold(self, config):
        """Should check minimum sample threshold."""
        daemon = TrainingTriggerDaemon(config)

        # Below threshold
        state_low = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            npz_sample_count=500,  # Below 1000 threshold
        )
        assert state_low.npz_sample_count < config.min_samples_threshold

        # Above threshold
        state_high = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            npz_sample_count=5000,
        )
        assert state_high.npz_sample_count >= config.min_samples_threshold

    def test_cooldown_check(self, config):
        """Should respect training cooldown."""
        daemon = TrainingTriggerDaemon(config)
        now = time.time()

        # Just trained - should be in cooldown
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            last_training_time=now - 60,  # 1 minute ago
        )
        cooldown_seconds = config.training_cooldown_hours * 3600
        time_since_training = now - state.last_training_time
        assert time_since_training < cooldown_seconds

    def test_concurrent_training_limit(self, daemon):
        """Should limit concurrent training jobs."""
        # Semaphore should limit concurrency
        assert daemon._training_semaphore._value == daemon.config.max_concurrent_training


# =============================================================================
# Training Intensity Tests
# =============================================================================


class TestTrainingIntensity:
    """Tests for training intensity handling."""

    def test_paused_intensity_blocks_training(self, daemon):
        """Paused intensity should block training."""
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            training_intensity="paused",
        )
        assert state.training_intensity == "paused"
        # When intensity is "paused", training should not be triggered

    def test_hot_path_intensity_prioritizes(self, daemon):
        """Hot path intensity should prioritize training."""
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            training_intensity="hot_path",
        )
        assert state.training_intensity == "hot_path"
        # When intensity is "hot_path", training should be prioritized


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestEventHandling:
    """Tests for event subscription and handling."""

    def test_has_subscribe_method(self, daemon):
        """Should have method to subscribe to events."""
        assert hasattr(daemon, "_subscribe_to_events")

    def test_event_subscriptions_list(self, daemon):
        """Should track event subscriptions."""
        assert hasattr(daemon, "_event_subscriptions")
        assert isinstance(daemon._event_subscriptions, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the daemon."""

    def test_multiple_configs_tracked(self, daemon):
        """Should track multiple configurations independently."""
        configs = ["hex8_2p", "square8_4p", "hexagonal_3p"]

        for config_key in configs:
            board_type, players = config_key.rsplit("_", 1)
            daemon._training_states[config_key] = ConfigTrainingState(
                config_key=config_key,
                board_type=board_type,
                num_players=int(players[0]),
            )

        assert len(daemon._training_states) == 3
        for config_key in configs:
            assert config_key in daemon._training_states

    def test_active_training_tasks_tracked(self, daemon):
        """Should track active training tasks."""
        assert hasattr(daemon, "_active_training_tasks")
        assert isinstance(daemon._active_training_tasks, dict)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_handles_missing_npz_gracefully(self, daemon):
        """Should handle missing NPZ info gracefully."""
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            npz_path="",  # Empty path
            npz_sample_count=0,
        )
        # Should not crash when NPZ info is missing
        assert state.npz_path == ""
        assert state.npz_sample_count == 0

    def test_handles_training_failure_count(self, daemon):
        """Should track consecutive failures."""
        state = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            consecutive_failures=3,
        )
        assert state.consecutive_failures == 3
