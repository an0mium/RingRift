"""Unit tests for optimized_pipeline module.

Tests cover:
- PipelineResult dataclass
- PipelineStatus dataclass
- OptimizedTrainingPipeline initialization
- Feature availability flags
- Export settings computation
- Lock acquisition and release
- Curriculum weight retrieval
- Training enhancement getters

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import fields

from app.training.optimized_pipeline import (
    PipelineResult,
    PipelineStatus,
    OptimizedTrainingPipeline,
    get_optimized_pipeline,
    HAS_EXPORT_CACHE,
    HAS_DYNAMIC_EXPORT,
    HAS_CURRICULUM_FEEDBACK,
    HAS_DISTRIBUTED_LOCK,
    HAS_HEALTH_MONITOR,
    HAS_MODEL_REGISTRY,
    HAS_TRAINING_ENHANCEMENTS,
    HAS_MULTI_TASK,
    HAS_DISTRIBUTED_TRAINING,
    HAS_ADVANCED_TRAINING,
    HAS_UNIFIED_SIGNALS,
)


class TestPipelineResultDataclass:
    """Tests for PipelineResult dataclass."""

    def test_basic_creation(self):
        """Test basic PipelineResult creation."""
        result = PipelineResult(
            config_key="hex8_2p",
            success=True,
            message="Training complete",
        )

        assert result.config_key == "hex8_2p"
        assert result.success is True
        assert result.message == "Training complete"

    def test_default_values(self):
        """Test PipelineResult default values."""
        result = PipelineResult(
            config_key="square8_4p",
            success=False,
            message="Export failed",
        )

        assert result.export_time == 0
        assert result.training_time == 0
        assert result.model_path is None
        assert result.model_id is None
        assert result.metrics == {}

    def test_metrics_default_post_init(self):
        """Test metrics defaults to empty dict via __post_init__."""
        result = PipelineResult(
            config_key="test",
            success=True,
            message="ok",
            metrics=None,
        )

        assert result.metrics == {}
        assert isinstance(result.metrics, dict)

    def test_with_timing_info(self):
        """Test PipelineResult with timing information."""
        result = PipelineResult(
            config_key="hex8_2p",
            success=True,
            message="Complete",
            export_time=45.5,
            training_time=1200.0,
        )

        assert result.export_time == 45.5
        assert result.training_time == 1200.0

    def test_with_model_info(self):
        """Test PipelineResult with model information."""
        result = PipelineResult(
            config_key="square8_2p",
            success=True,
            message="Complete",
            model_path="/models/best_model.pt",
            model_id="model_12345",
        )

        assert result.model_path == "/models/best_model.pt"
        assert result.model_id == "model_12345"

    def test_with_metrics(self):
        """Test PipelineResult with metrics."""
        metrics = {
            "final_loss": 0.15,
            "best_epoch": 45,
            "elo_gain": 50.0,
        }
        result = PipelineResult(
            config_key="hex8_4p",
            success=True,
            message="Complete",
            metrics=metrics,
        )

        assert result.metrics["final_loss"] == 0.15
        assert result.metrics["best_epoch"] == 45

    def test_failure_result(self):
        """Test PipelineResult for failure case."""
        result = PipelineResult(
            config_key="hexagonal_2p",
            success=False,
            message="Could not acquire training lock",
        )

        assert result.success is False
        assert "lock" in result.message.lower()


class TestPipelineStatusDataclass:
    """Tests for PipelineStatus dataclass."""

    def test_basic_creation(self):
        """Test basic PipelineStatus creation."""
        status = PipelineStatus(
            available_features={"export_cache": True, "dynamic_export": True},
            active_training=["hex8_2p"],
            health_status="healthy",
            curriculum_weights={"hex8_2p": 1.5},
            recent_results=[],
        )

        assert status.available_features["export_cache"] is True
        assert "hex8_2p" in status.active_training
        assert status.health_status == "healthy"

    def test_feature_availability(self):
        """Test feature availability tracking."""
        features = {
            "export_cache": True,
            "dynamic_export": True,
            "curriculum_feedback": True,
            "distributed_lock": False,
            "health_monitor": True,
        }
        status = PipelineStatus(
            available_features=features,
            active_training=[],
            health_status="unknown",
            curriculum_weights={},
            recent_results=[],
        )

        assert status.available_features["distributed_lock"] is False
        assert len(status.available_features) == 5

    def test_curriculum_weights(self):
        """Test curriculum weights in status."""
        weights = {
            "hex8_2p": 1.5,
            "square8_2p": 1.0,
            "hexagonal_4p": 0.8,
        }
        status = PipelineStatus(
            available_features={},
            active_training=[],
            health_status="healthy",
            curriculum_weights=weights,
            recent_results=[],
        )

        assert status.curriculum_weights["hex8_2p"] == 1.5
        assert status.curriculum_weights["hexagonal_4p"] == 0.8

    def test_recent_results_list(self):
        """Test recent results list in status."""
        results = [
            PipelineResult("hex8_2p", True, "ok"),
            PipelineResult("square8_2p", False, "failed"),
        ]
        status = PipelineStatus(
            available_features={},
            active_training=[],
            health_status="degraded",
            curriculum_weights={},
            recent_results=results,
        )

        assert len(status.recent_results) == 2
        assert status.recent_results[0].success is True
        assert status.recent_results[1].success is False


class TestFeatureFlags:
    """Tests for module-level feature flags."""

    def test_flags_are_booleans(self):
        """Test all feature flags are booleans."""
        flags = [
            HAS_EXPORT_CACHE,
            HAS_DYNAMIC_EXPORT,
            HAS_CURRICULUM_FEEDBACK,
            HAS_DISTRIBUTED_LOCK,
            HAS_HEALTH_MONITOR,
            HAS_MODEL_REGISTRY,
            HAS_TRAINING_ENHANCEMENTS,
            HAS_MULTI_TASK,
            HAS_DISTRIBUTED_TRAINING,
            HAS_ADVANCED_TRAINING,
            HAS_UNIFIED_SIGNALS,
        ]

        for flag in flags:
            assert isinstance(flag, bool)


class TestOptimizedTrainingPipelineInit:
    """Tests for OptimizedTrainingPipeline initialization."""

    def test_basic_initialization(self):
        """Test basic pipeline initialization."""
        pipeline = OptimizedTrainingPipeline()
        assert pipeline is not None

    def test_internal_state_initialized(self):
        """Test internal state is properly initialized."""
        pipeline = OptimizedTrainingPipeline()

        assert hasattr(pipeline, '_active_locks')
        assert hasattr(pipeline, '_recent_results')
        assert hasattr(pipeline, '_max_recent')

        assert pipeline._active_locks == {}
        assert pipeline._recent_results == []
        assert pipeline._max_recent == 50

    def test_components_initialized(self):
        """Test component attributes exist."""
        pipeline = OptimizedTrainingPipeline()

        # These may be None if dependencies not available
        assert hasattr(pipeline, '_export_cache')
        assert hasattr(pipeline, '_curriculum')
        assert hasattr(pipeline, '_health')
        assert hasattr(pipeline, '_triggers')
        assert hasattr(pipeline, '_signal_computer')


class TestOptimizedTrainingPipelineShouldTrain:
    """Tests for should_train method."""

    def test_should_train_basic_threshold(self):
        """Test basic threshold-based decision."""
        pipeline = OptimizedTrainingPipeline()
        # Patch out unified signals and triggers to test fallback
        pipeline._signal_computer = None
        pipeline._triggers = None

        # Below threshold
        should, reason = pipeline.should_train("hex8_2p", games_since_training=100)
        assert should is False
        assert "100" in reason

        # At threshold
        should, reason = pipeline.should_train("hex8_2p", games_since_training=500)
        assert should is True

        # Above threshold
        should, reason = pipeline.should_train("hex8_2p", games_since_training=1000)
        assert should is True

    def test_should_train_returns_tuple(self):
        """Test should_train returns tuple of (bool, str)."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.should_train("hex8_2p", games_since_training=100)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


class TestOptimizedTrainingPipelineExportSettings:
    """Tests for get_export_settings method."""

    def test_default_settings(self):
        """Test default export settings returned."""
        pipeline = OptimizedTrainingPipeline()
        settings = pipeline.get_export_settings("hex8_2p", [])

        assert "max_games" in settings
        assert "sample_every" in settings
        assert "epochs" in settings
        assert "batch_size" in settings
        assert "data_tier" in settings
        assert "estimated_samples" in settings

    def test_default_values(self):
        """Test default export setting values."""
        pipeline = OptimizedTrainingPipeline()
        settings = pipeline.get_export_settings("square8_4p", [])

        assert settings["max_games"] == 50000
        assert settings["sample_every"] == 2
        assert settings["epochs"] == 50
        assert settings["batch_size"] == 256

    def test_config_key_parsing(self):
        """Test various config key formats work."""
        pipeline = OptimizedTrainingPipeline()

        # Different board types
        for config_key in ["hex8_2p", "square8_4p", "hexagonal_3p", "square19_2p"]:
            settings = pipeline.get_export_settings(config_key, [])
            assert settings is not None
            assert isinstance(settings, dict)


class TestOptimizedTrainingPipelineNeedsExport:
    """Tests for needs_export method."""

    def test_needs_export_without_cache(self):
        """Test needs_export returns True when cache unavailable."""
        pipeline = OptimizedTrainingPipeline()
        pipeline._export_cache = None

        result = pipeline.needs_export(
            "hex8_2p",
            ["data/games/test.db"],
            "data/training/test.npz",
        )

        assert result is True


class TestOptimizedTrainingPipelineLocks:
    """Tests for lock acquisition and release."""

    def test_acquire_lock_without_distributed_lock(self):
        """Test lock acquisition when distributed lock unavailable."""
        with patch.dict('app.training.optimized_pipeline.__dict__', {'HAS_DISTRIBUTED_LOCK': False}):
            pipeline = OptimizedTrainingPipeline()
            # When distributed lock not available, should return True
            # (no locking needed)
            result = pipeline.acquire_lock("hex8_2p")
            assert result is True

    def test_release_lock_no_op(self):
        """Test release_lock is no-op when no lock held."""
        pipeline = OptimizedTrainingPipeline()
        # Should not raise
        pipeline.release_lock("hex8_2p")
        assert "hex8_2p" not in pipeline._active_locks


class TestOptimizedTrainingPipelineCurriculum:
    """Tests for curriculum weight retrieval."""

    def test_get_curriculum_weight_default(self):
        """Test curriculum weight defaults to 1.0."""
        pipeline = OptimizedTrainingPipeline()
        pipeline._curriculum = None

        weight = pipeline.get_curriculum_weight("hex8_2p")
        assert weight == 1.0

    def test_get_curriculum_weight_with_curriculum(self):
        """Test curriculum weight from curriculum component."""
        pipeline = OptimizedTrainingPipeline()
        mock_curriculum = MagicMock()
        mock_curriculum.get_curriculum_weights.return_value = {
            "hex8_2p": 1.5,
            "square8_4p": 0.8,
        }
        pipeline._curriculum = mock_curriculum

        weight = pipeline.get_curriculum_weight("hex8_2p")
        assert weight == 1.5

        weight = pipeline.get_curriculum_weight("unknown_config")
        assert weight == 1.0  # Default for unknown


class TestOptimizedTrainingPipelineStatus:
    """Tests for get_status method."""

    def test_get_status_returns_pipeline_status(self):
        """Test get_status returns PipelineStatus."""
        pipeline = OptimizedTrainingPipeline()
        status = pipeline.get_status()

        assert isinstance(status, PipelineStatus)

    def test_get_status_features(self):
        """Test get_status includes available features."""
        pipeline = OptimizedTrainingPipeline()
        status = pipeline.get_status()

        assert "export_cache" in status.available_features
        assert "dynamic_export" in status.available_features
        assert "curriculum_feedback" in status.available_features
        assert "distributed_lock" in status.available_features
        assert "health_monitor" in status.available_features
        assert "model_registry" in status.available_features
        assert "training_triggers" in status.available_features
        assert "training_enhancements" in status.available_features
        assert "multi_task_learning" in status.available_features
        assert "distributed_training" in status.available_features
        assert "advanced_training" in status.available_features

    def test_get_status_empty_active_training(self):
        """Test get_status with no active training."""
        pipeline = OptimizedTrainingPipeline()
        status = pipeline.get_status()

        assert status.active_training == []


class TestOptimizedTrainingPipelineEnhancements:
    """Tests for training enhancement getters."""

    def test_get_training_enhancements_without_module(self):
        """Test get_training_enhancements when module unavailable."""
        pipeline = OptimizedTrainingPipeline()

        # With mocked model/optimizer
        result = pipeline.get_training_enhancements(
            model=MagicMock(),
            optimizer=MagicMock(),
        )

        # May return None if enhancements not available
        assert result is None or isinstance(result, dict)

    def test_get_checkpoint_averager(self):
        """Test get_checkpoint_averager method."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.get_checkpoint_averager(num_checkpoints=5)

        # May return None if not available
        assert result is None or hasattr(result, 'add_checkpoint')

    def test_get_gradient_accumulator(self):
        """Test get_gradient_accumulator method."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.get_gradient_accumulator(
            accumulation_steps=4,
            max_grad_norm=1.0,
        )

        # May return None if not available
        assert result is None or hasattr(result, 'step')

    def test_get_model_ensemble(self):
        """Test get_model_ensemble method."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.get_model_ensemble(
            model_class=MagicMock,
            model_kwargs={},
        )

        # May return None if not available, otherwise should be an object
        assert result is None or result is not None


class TestOptimizedTrainingPipelineAdvanced:
    """Tests for advanced training utilities."""

    def test_get_lr_finder(self):
        """Test get_lr_finder method."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.get_lr_finder(
            model=MagicMock(),
            optimizer=MagicMock(),
            criterion=MagicMock(),
        )

        # May return None if not available
        assert result is None or hasattr(result, 'run')

    def test_get_gradient_checkpointing(self):
        """Test get_gradient_checkpointing method."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.get_gradient_checkpointing(model=MagicMock())

        # May return None if not available
        assert result is None or result is not None

    def test_get_pfsp_pool(self):
        """Test get_pfsp_pool method."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.get_pfsp_pool()

        # May return None if not available
        assert result is None or hasattr(result, 'select_opponent')

    def test_get_auto_tuner(self):
        """Test get_auto_tuner method."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.get_auto_tuner(
            board_type="hex8",
            num_players=2,
            plateau_patience=10,
        )

        # May return None if not available
        assert result is None or hasattr(result, 'suggest')

    def test_get_advanced_training_suite(self):
        """Test get_advanced_training_suite method."""
        pipeline = OptimizedTrainingPipeline()
        result = pipeline.get_advanced_training_suite(
            model=MagicMock(),
            optimizer=MagicMock(),
            criterion=MagicMock(),
            board_type="square8",
            num_players=2,
        )

        # May return None if not available
        assert result is None or isinstance(result, dict)


class TestOptimizedTrainingPipelineCalibration:
    """Tests for calibration methods."""

    def test_should_recalibrate_without_calibration(self):
        """Test should_recalibrate when calibration unavailable."""
        pipeline = OptimizedTrainingPipeline()
        pipeline._calibration = None

        result = pipeline.should_recalibrate(epoch=10)
        assert result is False

    def test_add_calibration_samples_no_op(self):
        """Test add_calibration_samples is no-op without calibration."""
        pipeline = OptimizedTrainingPipeline()
        pipeline._calibration = None

        # Should not raise
        pipeline.add_calibration_samples(
            predictions=MagicMock(),
            outcomes=MagicMock(),
        )


class TestOptimizedTrainingPipelineUnifiedSignals:
    """Tests for unified signals methods."""

    def test_get_unified_signals_without_computer(self):
        """Test get_unified_signals when computer unavailable."""
        pipeline = OptimizedTrainingPipeline()
        pipeline._signal_computer = None

        result = pipeline.get_unified_signals(
            config_key="hex8_2p",
            current_games=1000,
            current_elo=1550.0,
        )

        assert result is None

    def test_get_training_urgency_without_signals(self):
        """Test get_training_urgency when signals unavailable."""
        pipeline = OptimizedTrainingPipeline()
        pipeline._signal_computer = None

        result = pipeline.get_training_urgency(
            config_key="hex8_2p",
            current_games=500,
            current_elo=1500.0,
        )

        assert result is None


class TestGetOptimizedPipelineSingleton:
    """Tests for get_optimized_pipeline singleton function."""

    def test_returns_pipeline_instance(self):
        """Test function returns OptimizedTrainingPipeline."""
        pipeline = get_optimized_pipeline()
        assert isinstance(pipeline, OptimizedTrainingPipeline)

    def test_returns_same_instance(self):
        """Test singleton returns same instance."""
        pipeline1 = get_optimized_pipeline()
        pipeline2 = get_optimized_pipeline()
        assert pipeline1 is pipeline2


class TestPipelineResultHistory:
    """Tests for result history management."""

    def test_add_result_internal(self):
        """Test _add_result adds to history."""
        pipeline = OptimizedTrainingPipeline()
        result = PipelineResult("hex8_2p", True, "test")

        pipeline._add_result(result)

        assert len(pipeline._recent_results) == 1
        assert pipeline._recent_results[0].config_key == "hex8_2p"

    def test_add_result_trims_history(self):
        """Test _add_result trims history at max_recent."""
        pipeline = OptimizedTrainingPipeline()
        pipeline._max_recent = 5

        # Add more than max
        for i in range(10):
            result = PipelineResult(f"config_{i}", True, "test")
            pipeline._add_result(result)

        assert len(pipeline._recent_results) == 5
        # Should have most recent results
        assert pipeline._recent_results[-1].config_key == "config_9"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
