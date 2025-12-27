"""
Tests for learning rate scheduling functionality.

December 2025: Added comprehensive test coverage for
app/training/enhancements/learning_rate_scheduling.py
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from app.training.enhancements.learning_rate_scheduling import (
    AdaptiveLRScheduler,
    WarmRestartsScheduler,
)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 2)


@pytest.fixture
def optimizer(simple_model):
    """Create an optimizer for testing."""
    return optim.SGD(simple_model.parameters(), lr=0.001)


class TestAdaptiveLRSchedulerInit:
    """Tests for AdaptiveLRScheduler initialization."""

    def test_default_init(self, optimizer):
        """Test default initialization."""
        scheduler = AdaptiveLRScheduler(optimizer)

        assert scheduler.base_lr == 0.001
        assert scheduler.min_lr == 1e-6
        assert scheduler.max_lr == 0.01
        assert scheduler.elo_lookback == 5
        assert scheduler.loss_lookback == 10
        assert scheduler.warmup_epochs == 5
        assert scheduler.current_lr == 0.001

    def test_custom_init(self, optimizer):
        """Test initialization with custom parameters."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.01,
            min_lr=1e-5,
            max_lr=0.1,
            elo_lookback=10,
            loss_lookback=20,
            warmup_epochs=10,
        )

        assert scheduler.base_lr == 0.01
        assert scheduler.min_lr == 1e-5
        assert scheduler.max_lr == 0.1
        assert scheduler.elo_lookback == 10
        assert scheduler.loss_lookback == 20
        assert scheduler.warmup_epochs == 10

    def test_sets_optimizer_lr(self, optimizer):
        """Test that initialization sets optimizer LR."""
        AdaptiveLRScheduler(optimizer, base_lr=0.005)

        for param_group in optimizer.param_groups:
            assert param_group['lr'] == 0.005


class TestAdaptiveLRSchedulerStep:
    """Tests for AdaptiveLRScheduler step functionality."""

    def test_step_during_warmup(self, optimizer):
        """Test step does not adjust LR during warmup."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=5,
        )

        # Step through warmup
        for _ in range(5):
            lr = scheduler.step(train_loss=0.5)
            assert lr == 0.001

    def test_step_after_warmup(self, optimizer):
        """Test step can adjust LR after warmup."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=2,
            elo_lookback=3,
        )

        # Pass warmup
        scheduler.step(train_loss=0.5)
        scheduler.step(train_loss=0.5)

        # After warmup, step should still work
        lr = scheduler.step(train_loss=0.5)
        assert lr == 0.001  # No change without stagnation/oscillation

    def test_step_tracks_elo(self, optimizer):
        """Test that step tracks Elo history."""
        scheduler = AdaptiveLRScheduler(optimizer, warmup_epochs=0)

        scheduler.step(train_loss=0.5, current_elo=1500)
        scheduler.step(train_loss=0.5, current_elo=1510)
        scheduler.step(train_loss=0.5, current_elo=1520)

        assert len(scheduler._elo_history) == 3
        assert list(scheduler._elo_history) == [1500, 1510, 1520]


class TestEloStagnation:
    """Tests for Elo stagnation detection."""

    def test_no_stagnation_with_progress(self, optimizer):
        """Test no stagnation detected with Elo progress."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=0,
            elo_lookback=3,
            elo_stagnation_threshold=10.0,
        )

        # Good Elo progress (30 points over 3 steps)
        scheduler.step(train_loss=0.5, current_elo=1500)
        scheduler.step(train_loss=0.5, current_elo=1510)
        scheduler.step(train_loss=0.5, current_elo=1530)

        assert not scheduler._check_elo_stagnation()

    def test_stagnation_with_flat_elo(self, optimizer):
        """Test stagnation detected with flat Elo."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=0,
            elo_lookback=3,
            elo_stagnation_threshold=10.0,
        )

        # Flat Elo (5 points over 3 steps < 10 threshold)
        scheduler.step(train_loss=0.5, current_elo=1500)
        scheduler.step(train_loss=0.5, current_elo=1502)
        scheduler.step(train_loss=0.5, current_elo=1505)

        assert scheduler._check_elo_stagnation()

    def test_lr_increase_on_stagnation(self, optimizer):
        """Test LR increases on Elo stagnation."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=0,
            elo_lookback=3,
            elo_stagnation_threshold=10.0,
            lr_increase_factor=1.5,
        )

        # Fill Elo history with stagnating values
        scheduler.step(train_loss=0.5, current_elo=1500)
        scheduler.step(train_loss=0.5, current_elo=1501)
        lr = scheduler.step(train_loss=0.5, current_elo=1502)

        # Should have increased LR
        assert lr == pytest.approx(0.0015, rel=0.01)


class TestLossOscillation:
    """Tests for loss oscillation detection."""

    def test_no_oscillation_stable_loss(self, optimizer):
        """Test no oscillation detected with stable loss."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=0,
            loss_lookback=5,
            loss_oscillation_threshold=0.1,
        )

        # Stable decreasing loss
        for loss in [0.5, 0.48, 0.46, 0.44, 0.42]:
            scheduler.step(train_loss=loss)

        assert not scheduler._check_loss_oscillation()

    def test_oscillation_detected(self, optimizer):
        """Test oscillation detected with high-variance loss."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=0,
            loss_lookback=5,
            loss_oscillation_threshold=0.01,  # Low threshold
        )

        # High variance loss (oscillating)
        for loss in [0.5, 1.0, 0.3, 0.9, 0.4]:
            scheduler.step(train_loss=loss)

        assert scheduler._check_loss_oscillation()

    def test_lr_decrease_on_oscillation(self, optimizer):
        """Test LR decreases on oscillation."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=0,
            loss_lookback=5,
            loss_oscillation_threshold=0.001,  # Very low threshold
            lr_decrease_factor=0.5,
        )

        # High variance loss
        for loss in [0.5, 1.0, 0.3, 1.5, 0.2]:
            lr = scheduler.step(train_loss=loss)

        assert lr == pytest.approx(0.0005, rel=0.01)


class TestWarmRestart:
    """Tests for warm restart functionality."""

    def test_warm_restart(self, optimizer):
        """Test warm restart resets state."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            warmup_epochs=0,
        )

        # Build up history
        for _ in range(5):
            scheduler.step(train_loss=0.5, current_elo=1500)

        assert len(scheduler._loss_history) == 5

        # Warm restart
        scheduler.warm_restart(lr_factor=1.0)

        assert len(scheduler._loss_history) == 0
        assert len(scheduler._elo_history) == 0
        assert scheduler.current_lr == 0.001

    def test_warm_restart_with_factor(self, optimizer):
        """Test warm restart with LR factor."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            max_lr=0.01,
        )

        scheduler.warm_restart(lr_factor=2.0)
        assert scheduler.current_lr == 0.002

        # Factor is capped at max_lr
        scheduler.warm_restart(lr_factor=100.0)
        assert scheduler.current_lr == 0.01


class TestWarmRestartsSchedulerInit:
    """Tests for WarmRestartsScheduler initialization."""

    def test_default_init(self, optimizer):
        """Test default initialization."""
        scheduler = WarmRestartsScheduler(optimizer)

        assert scheduler.T_0 == 10
        assert scheduler.T_mult == 2
        assert scheduler.eta_min == 1e-6
        assert scheduler.warmup_steps == 0

    def test_custom_init(self, optimizer):
        """Test initialization with custom parameters."""
        scheduler = WarmRestartsScheduler(
            optimizer,
            T_0=5,
            T_mult=3,
            eta_min=1e-5,
            eta_max=0.01,
            warmup_steps=100,
        )

        assert scheduler.T_0 == 5
        assert scheduler.T_mult == 3
        assert scheduler.eta_min == 1e-5
        assert scheduler.eta_max == 0.01
        assert scheduler.warmup_steps == 100


class TestWarmRestartsSchedulerStep:
    """Tests for WarmRestartsScheduler step functionality."""

    def test_step_cosine_annealing(self, optimizer):
        """Test cosine annealing behavior."""
        scheduler = WarmRestartsScheduler(
            optimizer,
            T_0=10,
            eta_min=0.0,
            eta_max=1.0,
            warmup_steps=0,
        )

        # At T_cur = 0, LR should be eta_max
        lr = scheduler.step(epoch=0)
        assert lr == pytest.approx(1.0, rel=0.01)

        # At T_cur = 5 (halfway), LR should be eta_max/2
        scheduler._T_cur = 5
        scheduler._step_count = 5
        lr = scheduler.step()
        # cos(pi * 5/10) = cos(0.5*pi) = 0
        # lr = 0 + 0.5 * (1 - 0) * (1 + 0) = 0.5
        assert lr == pytest.approx(0.5, rel=0.01)

    def test_step_warmup(self, optimizer):
        """Test linear warmup."""
        scheduler = WarmRestartsScheduler(
            optimizer,
            T_0=10,
            eta_min=0.0,
            eta_max=1.0,
            warmup_steps=10,
        )

        # During warmup, LR should increase linearly
        for i in range(1, 11):
            lr = scheduler.step()
            expected = i / 10 * 1.0
            assert lr == pytest.approx(expected, rel=0.01)

    def test_warm_restart_occurs(self, optimizer):
        """Test that warm restart occurs at period end."""
        scheduler = WarmRestartsScheduler(
            optimizer,
            T_0=5,
            T_mult=2,
            warmup_steps=0,
        )

        # Step through first period
        for _ in range(5):
            scheduler.step()

        # Should have restarted
        assert scheduler.num_restarts == 1
        # Next period should be 10 (5 * 2)
        assert scheduler._T_i == 10


class TestWarmRestartsSchedulerStateDict:
    """Tests for state dict save/load."""

    def test_state_dict(self, optimizer):
        """Test state dict contains all required keys."""
        scheduler = WarmRestartsScheduler(optimizer)

        # Step a few times
        for _ in range(5):
            scheduler.step()

        state = scheduler.state_dict()

        assert 'step_count' in state
        assert 'epoch' in state
        assert 'restart_count' in state
        assert 'T_cur' in state
        assert 'T_i' in state
        assert 'current_lr' in state
        assert 'restart_epochs' in state

    def test_load_state_dict(self, optimizer):
        """Test loading state dict."""
        scheduler1 = WarmRestartsScheduler(optimizer)

        # Step a few times
        for _ in range(5):
            scheduler1.step()

        state = scheduler1.state_dict()

        # Create new scheduler and load state
        scheduler2 = WarmRestartsScheduler(optimizer)
        scheduler2.load_state_dict(state)

        assert scheduler2._step_count == scheduler1._step_count
        assert scheduler2._epoch == scheduler1._epoch
        assert scheduler2._current_lr == scheduler1._current_lr


class TestWarmRestartsSchedulerInfo:
    """Tests for schedule info methods."""

    def test_get_last_lr(self, optimizer):
        """Test get_last_lr returns correct values."""
        scheduler = WarmRestartsScheduler(optimizer)
        scheduler.step()

        last_lr = scheduler.get_last_lr()
        assert len(last_lr) == len(optimizer.param_groups)
        assert all(lr == scheduler.current_lr for lr in last_lr)

    def test_get_schedule_info(self, optimizer):
        """Test get_schedule_info returns all info."""
        scheduler = WarmRestartsScheduler(optimizer)
        for _ in range(5):
            scheduler.step()

        info = scheduler.get_schedule_info()

        assert 'current_lr' in info
        assert 'epoch' in info
        assert 'step' in info
        assert 'restart_count' in info
        assert 'current_period' in info
        assert 'position_in_period' in info
        assert 'progress_in_period' in info
        assert 'restart_epochs' in info

        assert info['step'] == 5

    def test_step_epoch(self, optimizer):
        """Test step_epoch increments epoch."""
        scheduler = WarmRestartsScheduler(optimizer, last_epoch=-1)

        scheduler.step_epoch()
        assert scheduler._epoch == 1

        scheduler.step_epoch()
        assert scheduler._epoch == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_adaptive_with_very_small_loss(self, optimizer):
        """Test adaptive scheduler handles very small losses."""
        scheduler = AdaptiveLRScheduler(optimizer, warmup_epochs=0)

        # Very small losses
        for _ in range(20):
            scheduler.step(train_loss=1e-10)

        # Should not crash
        assert scheduler.current_lr > 0

    def test_warm_restarts_with_zero_t0(self, optimizer):
        """Test warm restarts with T_0=1 (immediate restarts)."""
        scheduler = WarmRestartsScheduler(optimizer, T_0=1, T_mult=1)

        # Should restart every step
        scheduler.step()
        assert scheduler.num_restarts == 1

        scheduler.step()
        assert scheduler.num_restarts == 2

    def test_lr_bounds_respected(self, optimizer):
        """Test LR bounds are respected."""
        scheduler = AdaptiveLRScheduler(
            optimizer,
            base_lr=0.001,
            min_lr=0.0001,
            max_lr=0.01,
            warmup_epochs=0,
            elo_lookback=3,
            elo_stagnation_threshold=1000,  # Always stagnating
            lr_increase_factor=100,  # Large factor
        )

        # Force many increases
        for _ in range(10):
            scheduler.step(train_loss=0.5, current_elo=1500)

        # LR should be capped at max
        assert scheduler.current_lr <= 0.01
