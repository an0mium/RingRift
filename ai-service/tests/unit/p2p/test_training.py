"""Tests for app.p2p.training - P2P Training Coordination.

This module tests the training coordination utilities.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.p2p.training import (
    TrainingThresholds,
    calculate_training_priority,
    should_trigger_training,
)


# =============================================================================
# TrainingThresholds Tests
# =============================================================================


class TestTrainingThresholds:
    """Tests for TrainingThresholds dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        thresholds = TrainingThresholds()
        assert thresholds.min_games == 1000
        assert thresholds.batch_size == 256
        assert thresholds.min_epochs == 5
        assert thresholds.max_epochs == 50
        assert thresholds.elo_improvement_threshold == 10.0
        assert thresholds.max_stagnant_runs == 3

    def test_epoch_range(self):
        """Min epochs should be less than max epochs."""
        thresholds = TrainingThresholds()
        assert thresholds.min_epochs < thresholds.max_epochs

    def test_env_override_min_games(self):
        """Should respect environment variable override."""
        with patch.dict(os.environ, {"RINGRIFT_MIN_TRAINING_GAMES": "500"}):
            thresholds = TrainingThresholds()
            assert thresholds.min_games == 500

    def test_env_override_batch_size(self):
        """Should respect batch size override."""
        with patch.dict(os.environ, {"RINGRIFT_TRAINING_BATCH_SIZE": "512"}):
            thresholds = TrainingThresholds()
            assert thresholds.batch_size == 512

    def test_to_dict(self):
        """Should convert to dict correctly."""
        thresholds = TrainingThresholds()
        d = thresholds.to_dict()
        assert d["min_games"] == 1000
        assert d["batch_size"] == 256
        assert d["min_epochs"] == 5
        assert d["max_epochs"] == 50
        assert "elo_improvement_threshold" in d
        assert "max_stagnant_runs" in d


# =============================================================================
# calculate_training_priority Tests
# =============================================================================


class TestCalculateTrainingPriority:
    """Tests for calculate_training_priority function."""

    def test_basic_priority(self):
        """Should calculate priority for basic node."""
        priority = calculate_training_priority(
            gpu_type="4090",
            memory_gb=64.0,
            current_load=50.0,
            has_training_data=True,
        )
        assert priority > 0

    def test_h100_higher_than_4090(self):
        """H100 should have higher priority than 4090."""
        h100_priority = calculate_training_priority(
            gpu_type="H100", memory_gb=64.0, current_load=50.0
        )
        rtx_priority = calculate_training_priority(
            gpu_type="4090", memory_gb=64.0, current_load=50.0
        )
        assert h100_priority > rtx_priority

    def test_more_memory_higher_priority(self):
        """More memory should increase priority."""
        low_mem = calculate_training_priority(
            gpu_type="4090", memory_gb=32.0, current_load=50.0
        )
        high_mem = calculate_training_priority(
            gpu_type="4090", memory_gb=128.0, current_load=50.0
        )
        assert high_mem > low_mem

    def test_lower_load_higher_priority(self):
        """Lower load should increase priority."""
        low_load = calculate_training_priority(
            gpu_type="4090", memory_gb=64.0, current_load=20.0
        )
        high_load = calculate_training_priority(
            gpu_type="4090", memory_gb=64.0, current_load=80.0
        )
        assert low_load > high_load

    def test_data_locality_bonus(self):
        """Having training data should increase priority."""
        with_data = calculate_training_priority(
            gpu_type="4090", memory_gb=64.0, current_load=50.0, has_training_data=True
        )
        without_data = calculate_training_priority(
            gpu_type="4090", memory_gb=64.0, current_load=50.0, has_training_data=False
        )
        assert with_data > without_data

    def test_unknown_gpu(self):
        """Unknown GPU should still get a priority."""
        priority = calculate_training_priority(
            gpu_type="Unknown", memory_gb=64.0, current_load=50.0
        )
        assert priority >= 0


# =============================================================================
# should_trigger_training Tests
# =============================================================================


class TestShouldTriggerTraining:
    """Tests for should_trigger_training function."""

    def test_sufficient_games_triggers(self):
        """Should trigger with sufficient games."""
        should_train, reason = should_trigger_training(
            games_available=2000,
            hours_since_last_training=5.0,
        )
        assert should_train is True
        assert "Ready to train" in reason

    def test_insufficient_games_blocks(self):
        """Should not trigger with insufficient games."""
        should_train, reason = should_trigger_training(
            games_available=500,
            hours_since_last_training=5.0,
        )
        assert should_train is False
        assert "Insufficient games" in reason

    def test_rate_limited(self):
        """Should not trigger too soon after last training."""
        should_train, reason = should_trigger_training(
            games_available=2000,
            hours_since_last_training=0.5,  # Only 30 minutes
        )
        assert should_train is False
        assert "Rate limited" in reason

    def test_stagnant_runs_blocks(self):
        """Should pause after too many stagnant runs."""
        should_train, reason = should_trigger_training(
            games_available=2000,
            hours_since_last_training=5.0,
            consecutive_stagnant=5,
        )
        assert should_train is False
        assert "stagnant" in reason

    def test_custom_thresholds(self):
        """Should respect custom thresholds."""
        thresholds = TrainingThresholds(min_games=100, min_hours_between_training=0.1)
        should_train, reason = should_trigger_training(
            games_available=150,
            hours_since_last_training=0.2,
            thresholds=thresholds,
        )
        assert should_train is True

    def test_boundary_games(self):
        """Exactly min_games should trigger."""
        thresholds = TrainingThresholds(min_games=1000)
        should_train, _ = should_trigger_training(
            games_available=1000,
            hours_since_last_training=5.0,
            thresholds=thresholds,
        )
        assert should_train is True

    def test_boundary_games_minus_one(self):
        """min_games - 1 should not trigger."""
        thresholds = TrainingThresholds(min_games=1000)
        should_train, _ = should_trigger_training(
            games_available=999,
            hours_since_last_training=5.0,
            thresholds=thresholds,
        )
        assert should_train is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Integration tests for training coordination."""

    def test_training_decision_flow(self):
        """Test realistic training decision flow."""
        thresholds = TrainingThresholds(
            min_games=500,
            min_hours_between_training=1.0,
            max_stagnant_runs=2,
        )

        # Initially not enough games
        should_train, _ = should_trigger_training(
            games_available=300,
            hours_since_last_training=10.0,
            thresholds=thresholds,
        )
        assert should_train is False

        # Enough games, should trigger
        should_train, _ = should_trigger_training(
            games_available=600,
            hours_since_last_training=2.0,
            thresholds=thresholds,
        )
        assert should_train is True

        # Right after training, rate limited
        should_train, _ = should_trigger_training(
            games_available=600,
            hours_since_last_training=0.5,
            thresholds=thresholds,
        )
        assert should_train is False

    def test_node_ranking_for_training(self):
        """Test that nodes can be ranked for training."""
        # Simulate ranking multiple nodes
        nodes = [
            {"gpu": "H100", "mem": 80.0, "load": 30.0},
            {"gpu": "4090", "mem": 64.0, "load": 50.0},
            {"gpu": "3080", "mem": 32.0, "load": 20.0},
        ]

        priorities = [
            calculate_training_priority(n["gpu"], n["mem"], n["load"])
            for n in nodes
        ]

        # H100 should be first choice
        assert priorities[0] == max(priorities)
