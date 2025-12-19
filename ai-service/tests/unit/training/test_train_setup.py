"""Tests for app/training/train_setup.py - Training setup utilities.

Tests cover:
- FaultToleranceConfig and FaultToleranceComponents dataclasses
- setup_fault_tolerance factory function
- Device selection helpers
- TrainingState management
"""

import pytest
import torch

from app.training.train_setup import (
    FaultToleranceConfig,
    FaultToleranceComponents,
    setup_fault_tolerance,
    get_device,
    compute_effective_lr,
    TrainingState,
)


class TestFaultToleranceConfig:
    """Tests for FaultToleranceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FaultToleranceConfig()

        assert config.enable_circuit_breaker is True
        assert config.enable_anomaly_detection is True
        assert config.enable_graceful_shutdown is True
        assert config.gradient_clip_mode == 'adaptive'
        assert config.gradient_clip_max_norm == 1.0
        assert config.anomaly_spike_threshold == 3.0
        assert config.anomaly_gradient_threshold == 100.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FaultToleranceConfig(
            enable_circuit_breaker=False,
            gradient_clip_mode='fixed',
            gradient_clip_max_norm=0.5,
        )

        assert config.enable_circuit_breaker is False
        assert config.gradient_clip_mode == 'fixed'
        assert config.gradient_clip_max_norm == 0.5


class TestFaultToleranceComponents:
    """Tests for FaultToleranceComponents dataclass."""

    def test_default_values(self):
        """Test default component values."""
        components = FaultToleranceComponents()

        assert components.training_breaker is None
        assert components.anomaly_detector is None
        assert components.adaptive_clipper is None
        assert components.shutdown_handler is None
        assert components.gradient_clip_mode == 'fixed'
        assert components.fixed_clip_norm == 1.0


class TestSetupFaultTolerance:
    """Tests for setup_fault_tolerance function."""

    def test_with_all_disabled(self):
        """Test setup with all components disabled."""
        config = FaultToleranceConfig(
            enable_circuit_breaker=False,
            enable_anomaly_detection=False,
            gradient_clip_mode='fixed',
        )

        components = setup_fault_tolerance(config)

        assert components.training_breaker is None
        assert components.anomaly_detector is None
        assert components.adaptive_clipper is None
        assert components.gradient_clip_mode == 'fixed'

    def test_with_defaults(self):
        """Test setup with default configuration."""
        config = FaultToleranceConfig()

        components = setup_fault_tolerance(config)

        # Components may or may not be available depending on imports
        # Just verify the function runs without error
        assert isinstance(components, FaultToleranceComponents)

    def test_fixed_gradient_clipping(self):
        """Test fixed gradient clipping mode."""
        config = FaultToleranceConfig(
            gradient_clip_mode='fixed',
            gradient_clip_max_norm=2.0,
        )

        components = setup_fault_tolerance(config)

        assert components.gradient_clip_mode == 'fixed'
        assert components.fixed_clip_norm == 2.0
        assert components.adaptive_clipper is None


class TestGetDevice:
    """Tests for get_device function."""

    def test_cpu_fallback(self):
        """Test CPU device selection."""
        device = get_device(local_rank=-1)

        # Should return a valid device
        assert device.type in ['cpu', 'cuda', 'mps']

    def test_distributed_rank(self):
        """Test device selection with distributed rank."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = get_device(local_rank=0)
        assert device.type == 'cuda'
        assert device.index == 0


class TestComputeEffectiveLR:
    """Tests for compute_effective_lr function."""

    def test_no_scaling(self):
        """Test LR without scaling."""
        lr = compute_effective_lr(
            base_lr=0.001,
            world_size=4,
            scale_lr=False,
        )

        assert lr == 0.001

    def test_linear_scaling(self):
        """Test linear LR scaling."""
        lr = compute_effective_lr(
            base_lr=0.001,
            world_size=4,
            scale_lr=True,
            lr_scale_mode='linear',
        )

        assert lr == pytest.approx(0.004)

    def test_sqrt_scaling(self):
        """Test sqrt LR scaling."""
        lr = compute_effective_lr(
            base_lr=0.001,
            world_size=4,
            scale_lr=True,
            lr_scale_mode='sqrt',
        )

        assert lr == pytest.approx(0.002)

    def test_single_process(self):
        """Test LR scaling with single process."""
        lr = compute_effective_lr(
            base_lr=0.001,
            world_size=1,
            scale_lr=True,
        )

        # Should not scale with world_size=1
        assert lr == 0.001


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = TrainingState()

        assert state.epoch == 0
        assert state.best_val_loss == float('inf')
        assert state.last_good_checkpoint_path is None
        assert state.circuit_breaker_rollbacks == 0

    def test_can_rollback_no_checkpoint(self):
        """Test rollback check with no checkpoint."""
        state = TrainingState()

        assert state.can_rollback() is False

    def test_can_rollback_with_checkpoint(self):
        """Test rollback check with checkpoint."""
        state = TrainingState(
            last_good_checkpoint_path="/path/to/checkpoint.pth",
            last_good_epoch=5,
        )

        assert state.can_rollback() is True

    def test_can_rollback_max_reached(self):
        """Test rollback check when max rollbacks reached."""
        state = TrainingState(
            last_good_checkpoint_path="/path/to/checkpoint.pth",
            circuit_breaker_rollbacks=3,
            max_circuit_breaker_rollbacks=3,
        )

        assert state.can_rollback() is False

    def test_record_rollback(self):
        """Test recording a rollback."""
        state = TrainingState()
        assert state.circuit_breaker_rollbacks == 0

        state.record_rollback()
        assert state.circuit_breaker_rollbacks == 1

        state.record_rollback()
        assert state.circuit_breaker_rollbacks == 2

    def test_update_good_checkpoint(self):
        """Test updating good checkpoint."""
        state = TrainingState()

        state.update_good_checkpoint("/path/to/new_checkpoint.pth", 10)

        assert state.last_good_checkpoint_path == "/path/to/new_checkpoint.pth"
        assert state.last_good_epoch == 10
