"""Unit tests for Training Trigger RPC endpoints (December 30, 2025).

Tests the new training trigger RPC functionality:
- TrainingDecision dataclass
- get_training_decision() method
- P2P orchestrator handlers (handle_training_trigger, handle_training_trigger_decision)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.training_trigger_daemon import (
    ConfigTrainingState,
    TrainingDecision,
    TrainingTriggerConfig,
    TrainingTriggerDaemon,
)


class TestTrainingDecision:
    """Tests for TrainingDecision dataclass."""

    def test_default_values(self):
        """Test TrainingDecision default values."""
        decision = TrainingDecision(
            config_key="hex8_2p",
            can_trigger=True,
            reason="all conditions met",
        )

        assert decision.config_key == "hex8_2p"
        assert decision.can_trigger is True
        assert decision.reason == "all conditions met"
        assert decision.training_in_progress is False
        assert decision.sample_count == 0
        assert decision.sample_threshold == 5000
        assert decision.gpu_available is True
        assert decision.current_elo == 1500.0

    def test_to_dict(self):
        """Test TrainingDecision.to_dict() serialization."""
        decision = TrainingDecision(
            config_key="hex8_4p",
            can_trigger=False,
            reason="cooldown active",
            training_in_progress=False,
            cooldown_remaining_hours=2.345,
            data_age_hours=0.567,
            sample_count=12000,
            sample_threshold=5000,
            current_elo=1650.5,
            elo_velocity=0.123,
            elo_velocity_trend="accelerating",
        )

        result = decision.to_dict()

        assert result["config_key"] == "hex8_4p"
        assert result["can_trigger"] is False
        assert result["reason"] == "cooldown active"
        assert result["conditions"]["cooldown_remaining_hours"] == 2.35  # rounded
        assert result["conditions"]["sample_count"] == 12000
        assert result["data_info"]["samples"] == 12000
        assert result["data_info"]["age_hours"] == 0.57  # rounded
        assert result["elo_info"]["current_elo"] == 1650.5
        assert result["elo_info"]["elo_velocity"] == 0.123
        assert result["elo_info"]["elo_velocity_trend"] == "accelerating"

    def test_to_dict_nested_structure(self):
        """Test to_dict() produces correct nested structure."""
        decision = TrainingDecision(
            config_key="square8_2p",
            can_trigger=True,
            reason="conditions met",
        )

        result = decision.to_dict()

        # Check nested structure
        assert "conditions" in result
        assert "data_info" in result
        assert "elo_info" in result

        # Check conditions dict
        conditions = result["conditions"]
        assert "training_in_progress" in conditions
        assert "intensity_paused" in conditions
        assert "evaluation_backpressure" in conditions
        assert "circuit_breaker_open" in conditions
        assert "cooldown_remaining_hours" in conditions
        assert "gpu_available" in conditions

        # Check data_info dict
        data_info = result["data_info"]
        assert "npz_path" in data_info
        assert "samples" in data_info
        assert "age_hours" in data_info


class TestTrainingTriggerDaemonAPI:
    """Tests for TrainingTriggerDaemon get_training_decision() method."""

    @pytest.fixture
    def daemon(self):
        """Create a daemon instance for testing."""
        # Reset singleton
        TrainingTriggerDaemon._instance = None
        config = TrainingTriggerConfig(
            enabled=True,
            scan_interval_seconds=60,
            max_concurrent_training=20,
        )
        daemon = TrainingTriggerDaemon(config)
        return daemon

    def test_get_tracked_configs_empty(self, daemon):
        """Test get_tracked_configs() with no configs."""
        configs = daemon.get_tracked_configs()
        assert configs == []

    def test_get_tracked_configs_with_data(self, daemon):
        """Test get_tracked_configs() with tracked configs."""
        # Add some configs
        daemon._training_states["hex8_2p"] = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
        )
        daemon._training_states["square8_4p"] = ConfigTrainingState(
            config_key="square8_4p",
            board_type="square8",
            num_players=4,
        )

        configs = daemon.get_tracked_configs()
        assert len(configs) == 2
        assert "hex8_2p" in configs
        assert "square8_4p" in configs

    @pytest.mark.asyncio
    async def test_get_training_decision_config_not_tracked(self, daemon):
        """Test get_training_decision() for untracked config."""
        decision = await daemon.get_training_decision("unknown_2p")

        assert decision.config_key == "unknown_2p"
        assert decision.can_trigger is False
        assert decision.reason == "config not tracked"

    @pytest.mark.asyncio
    async def test_get_training_decision_training_in_progress(self, daemon):
        """Test get_training_decision() when training is in progress."""
        daemon._training_states["hex8_2p"] = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            training_in_progress=True,
            last_npz_update=time.time(),
            npz_sample_count=10000,
        )

        decision = await daemon.get_training_decision("hex8_2p")

        assert decision.config_key == "hex8_2p"
        assert decision.can_trigger is False
        assert "training already in progress" in decision.reason
        assert decision.training_in_progress is True

    @pytest.mark.asyncio
    async def test_get_training_decision_insufficient_samples(self, daemon):
        """Test get_training_decision() with insufficient samples."""
        daemon._training_states["hex8_2p"] = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            training_in_progress=False,
            last_npz_update=time.time(),
            npz_sample_count=100,  # Below threshold
            last_training_time=0,  # No cooldown
        )

        decision = await daemon.get_training_decision("hex8_2p")

        assert decision.config_key == "hex8_2p"
        assert decision.can_trigger is False
        assert "insufficient samples" in decision.reason
        assert decision.sample_count == 100

    @pytest.mark.asyncio
    async def test_get_training_decision_paused_intensity(self, daemon):
        """Test get_training_decision() when training intensity is paused."""
        daemon._training_states["hex8_2p"] = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            training_in_progress=False,
            training_intensity="paused",
            last_npz_update=time.time(),
            npz_sample_count=10000,
        )

        decision = await daemon.get_training_decision("hex8_2p")

        assert decision.can_trigger is False
        assert "paused" in decision.reason
        assert decision.intensity_paused is True

    @pytest.mark.asyncio
    async def test_get_training_decision_conditions_met(self, daemon):
        """Test get_training_decision() when all conditions are met."""
        daemon._training_states["hex8_2p"] = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            training_in_progress=False,
            training_intensity="normal",
            last_npz_update=time.time(),  # Fresh data
            npz_sample_count=10000,  # Above threshold
            last_training_time=0,  # No cooldown
            last_elo=1600.0,
            elo_velocity=0.5,
            elo_velocity_trend="accelerating",
        )

        decision = await daemon.get_training_decision("hex8_2p")

        assert decision.config_key == "hex8_2p"
        assert decision.can_trigger is True
        assert decision.reason == "all conditions met"
        assert decision.sample_count == 10000
        assert decision.current_elo == 1600.0
        assert decision.elo_velocity == 0.5
        assert decision.elo_velocity_trend == "accelerating"

    @pytest.mark.asyncio
    async def test_get_training_decision_includes_data_age(self, daemon):
        """Test that get_training_decision() calculates data age correctly."""
        # Set last NPZ update to 2 hours ago
        two_hours_ago = time.time() - 7200
        daemon._training_states["hex8_2p"] = ConfigTrainingState(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            training_in_progress=False,
            training_intensity="normal",
            last_npz_update=two_hours_ago,
            npz_sample_count=10000,
            last_training_time=0,
        )

        decision = await daemon.get_training_decision("hex8_2p")

        # Data age should be approximately 2 hours
        assert 1.9 < decision.data_age_hours < 2.1


class TestGetTrainingTriggerDaemon:
    """Tests for get_training_trigger_daemon() singleton accessor."""

    def test_singleton_pattern(self):
        """Test that get_training_trigger_daemon returns singleton."""
        # Reset singleton
        TrainingTriggerDaemon._instance = None

        from app.coordination.training_trigger_daemon import get_training_trigger_daemon

        daemon1 = get_training_trigger_daemon()
        daemon2 = get_training_trigger_daemon()

        assert daemon1 is daemon2

    def teardown_method(self):
        """Reset singleton after each test."""
        TrainingTriggerDaemon._instance = None
