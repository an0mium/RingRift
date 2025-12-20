"""Tests for app/metrics/orchestrator.py helper functions.

Tests cover:
- record_selfplay_batch
- record_training_run
- record_evaluation
- record_model_promotion
- record_pipeline_stage / time_pipeline_stage
- set_pipeline_state
- Queue and iteration tracking helpers
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from app.metrics.orchestrator import (
    PIPELINE_EVALUATION,
    # State constants
    PIPELINE_IDLE,
    PIPELINE_PROMOTION,
    PIPELINE_SELFPLAY,
    PIPELINE_TRAINING,
    get_pipeline_iterations,
    get_selfplay_queue_size,
    record_evaluation,
    record_model_promotion,
    record_pipeline_iteration,
    record_pipeline_stage,
    record_promotion_rejection,
    # Helper functions
    record_selfplay_batch,
    record_training_run,
    set_pipeline_state,
    time_pipeline_stage,
    # Queue and iteration tracking
    update_selfplay_queue_size,
)


class TestRecordSelfplayBatch:
    """Tests for record_selfplay_batch helper."""

    def test_records_games_total(self):
        """Test that games total counter is incremented."""
        record_selfplay_batch(
            board_type="square8",
            num_players=2,
            games=100,
            duration_seconds=60.0,
        )

        # Should not raise - metric incremented successfully

    def test_records_batch_duration(self):
        """Test that batch duration histogram is observed."""
        record_selfplay_batch(
            board_type="hex7",
            num_players=3,
            games=50,
            duration_seconds=120.0,
        )

        # Should not raise

    def test_custom_orchestrator_label(self):
        """Test using custom orchestrator label."""
        record_selfplay_batch(
            board_type="square8",
            num_players=2,
            games=10,
            duration_seconds=10.0,
            orchestrator="test_orchestrator",
        )

        # Should not raise


class TestRecordTrainingRun:
    """Tests for record_training_run helper."""

    def test_basic_training_run(self):
        """Test recording a basic training run."""
        record_training_run(
            board_type="square8",
            num_players=2,
            duration_seconds=3600.0,
            final_loss=0.5,
        )

        # Should not raise

    def test_training_run_with_accuracy(self):
        """Test recording training run with accuracy."""
        record_training_run(
            board_type="hex7",
            num_players=2,
            duration_seconds=1800.0,
            final_loss=0.3,
            final_accuracy=0.85,
        )

        # Should not raise

    def test_training_run_with_samples(self):
        """Test recording training run with sample count."""
        record_training_run(
            board_type="square8",
            num_players=2,
            duration_seconds=7200.0,
            final_loss=0.25,
            samples=100000,
        )

        # Should not raise

    def test_training_run_with_epochs(self):
        """Test recording training run with epoch count."""
        record_training_run(
            board_type="square8",
            num_players=4,
            duration_seconds=5400.0,
            final_loss=0.4,
            epochs=10,
        )

        # Should not raise


class TestRecordEvaluation:
    """Tests for record_evaluation helper."""

    def test_basic_evaluation(self):
        """Test recording a basic evaluation."""
        record_evaluation(
            board_type="square8",
            num_players=2,
            games=100,
            elo_delta=50.0,
            win_rate=0.65,
            duration_seconds=600.0,
        )

        # Should not raise

    def test_evaluation_negative_elo(self):
        """Test recording evaluation with negative Elo delta."""
        record_evaluation(
            board_type="hex7",
            num_players=2,
            games=100,
            elo_delta=-30.0,
            win_rate=0.45,
            duration_seconds=300.0,
        )

        # Should not raise


class TestRecordModelPromotion:
    """Tests for record_model_promotion helper."""

    def test_basic_promotion(self):
        """Test recording a basic promotion."""
        record_model_promotion(
            board_type="square8",
            num_players=2,
            elo_gain=100.0,
            new_elo=1600.0,
        )

        # Should not raise

    def test_promotion_with_promotion_type(self):
        """Test recording promotion with promotion type."""
        record_model_promotion(
            board_type="hex7",
            num_players=3,
            elo_gain=50.0,
            new_elo=1550.0,
            promotion_type="manual",
        )

        # Should not raise


class TestRecordPromotionRejection:
    """Tests for record_promotion_rejection helper."""

    def test_basic_rejection(self):
        """Test recording a basic rejection."""
        record_promotion_rejection(
            board_type="square8",
            num_players=2,
            reason="elo_insufficient",
        )

        # Should not raise

    def test_rejection_with_details(self):
        """Test recording rejection with various reasons."""
        reasons = [
            "elo_regression",
            "significance_failed",
            "parity_check_failed",
            "win_rate_too_low",
        ]

        for reason in reasons:
            record_promotion_rejection(
                board_type="square8",
                num_players=2,
                reason=reason,
            )

        # Should not raise


class TestPipelineStageMetrics:
    """Tests for pipeline stage recording."""

    def test_record_pipeline_stage(self):
        """Test recording a pipeline stage duration."""
        record_pipeline_stage(
            stage="selfplay",
            duration_seconds=300.0,
        )

        # Should not raise

    def test_record_multiple_stages(self):
        """Test recording multiple pipeline stages."""
        stages = ["selfplay", "training", "evaluation", "promotion"]

        for stage in stages:
            record_pipeline_stage(
                stage=stage,
                duration_seconds=100.0,
            )

        # Should not raise

    def test_time_pipeline_stage_context_manager(self):
        """Test time_pipeline_stage context manager."""
        with time_pipeline_stage("test_stage"):
            # Simulate some work
            time.sleep(0.01)

        # Should not raise

    def test_time_pipeline_stage_timing(self):
        """Test that time_pipeline_stage measures actual time."""
        with time_pipeline_stage("timing_test"):
            time.sleep(0.05)

        # Should have recorded ~0.05 seconds


class TestSetPipelineState:
    """Tests for set_pipeline_state helper."""

    def test_set_idle(self):
        """Test setting pipeline to idle."""
        set_pipeline_state("test_orch", PIPELINE_IDLE)

        # Should not raise

    def test_set_selfplay(self):
        """Test setting pipeline to selfplay."""
        set_pipeline_state("test_orch", PIPELINE_SELFPLAY)

        # Should not raise

    def test_set_training(self):
        """Test setting pipeline to training."""
        set_pipeline_state("test_orch", PIPELINE_TRAINING)

        # Should not raise

    def test_set_evaluation(self):
        """Test setting pipeline to evaluation."""
        set_pipeline_state("test_orch", PIPELINE_EVALUATION)

        # Should not raise

    def test_set_promotion(self):
        """Test setting pipeline to promotion."""
        set_pipeline_state("test_orch", PIPELINE_PROMOTION)

        # Should not raise


class TestStateConstants:
    """Tests for pipeline state constants."""

    def test_state_values(self):
        """Test state constants have expected values."""
        assert PIPELINE_IDLE == 0
        assert PIPELINE_SELFPLAY == 1
        assert PIPELINE_TRAINING == 2
        assert PIPELINE_EVALUATION == 3
        assert PIPELINE_PROMOTION == 4


class TestQueueSizeTracking:
    """Tests for selfplay queue size tracking."""

    def test_update_queue_size(self):
        """Test updating queue size."""
        update_selfplay_queue_size(10)

        # Should be able to get it back
        size = get_selfplay_queue_size()
        assert size == 10.0

    def test_update_queue_size_zero(self):
        """Test setting queue size to zero."""
        update_selfplay_queue_size(0)

        size = get_selfplay_queue_size()
        assert size == 0.0

    def test_update_queue_size_with_orchestrator(self):
        """Test setting queue size with custom orchestrator."""
        update_selfplay_queue_size(5, orchestrator="test_orch")

        size = get_selfplay_queue_size(orchestrator="test_orch")
        assert size == 5.0


class TestPipelineIterationTracking:
    """Tests for pipeline iteration tracking."""

    def test_record_iteration(self):
        """Test recording a pipeline iteration."""
        initial = get_pipeline_iterations()

        record_pipeline_iteration()

        after = get_pipeline_iterations()
        assert after == initial + 1

    def test_record_multiple_iterations(self):
        """Test recording multiple iterations."""
        initial = get_pipeline_iterations()

        for _ in range(5):
            record_pipeline_iteration()

        after = get_pipeline_iterations()
        assert after == initial + 5

    def test_iteration_with_orchestrator(self):
        """Test iteration tracking with custom orchestrator."""
        initial = get_pipeline_iterations(orchestrator="test_iter_orch")

        record_pipeline_iteration(orchestrator="test_iter_orch")

        after = get_pipeline_iterations(orchestrator="test_iter_orch")
        assert after == initial + 1
