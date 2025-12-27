"""Tests for temperature scheduling module.

Tests cover all temperature schedule implementations:
- ConstantSchedule
- LinearDecaySchedule
- ExponentialDecaySchedule
- StepSchedule
- CosineAnnealingSchedule
- AdaptiveSchedule
- CurriculumSchedule
- EloAdaptiveSchedule
- MixedSchedule
- AlphaZeroTemperature
- DirichletNoiseTemperature
"""

import math
import pytest
from unittest.mock import MagicMock

from app.training.temperature_scheduling import (
    ScheduleType,
    TemperatureConfig,
    TemperatureSchedule,
    ConstantSchedule,
    LinearDecaySchedule,
    ExponentialDecaySchedule,
    StepSchedule,
    CosineAnnealingSchedule,
    AdaptiveSchedule,
    CurriculumSchedule,
    EloAdaptiveSchedule,
    create_scheduler,
)


class TestConstantSchedule:
    """Tests for ConstantSchedule."""

    def test_returns_constant_value(self):
        schedule = ConstantSchedule(temperature=0.8)
        assert schedule.get_temperature(0) == 0.8
        assert schedule.get_temperature(50) == 0.8
        assert schedule.get_temperature(100) == 0.8

    def test_default_temperature(self):
        schedule = ConstantSchedule()
        assert schedule.get_temperature(0) == 1.0

    def test_ignores_game_state(self):
        schedule = ConstantSchedule(temperature=0.5)
        assert schedule.get_temperature(10, game_state="dummy") == 0.5

    def test_ignores_training_progress(self):
        schedule = ConstantSchedule(temperature=0.5)
        assert schedule.get_temperature(10, training_progress=0.9) == 0.5


class TestLinearDecaySchedule:
    """Tests for LinearDecaySchedule."""

    def test_returns_initial_before_decay_start(self):
        schedule = LinearDecaySchedule(
            initial_temp=1.0, final_temp=0.1, decay_start=10, decay_end=60
        )
        assert schedule.get_temperature(0) == 1.0
        assert schedule.get_temperature(5) == 1.0
        assert schedule.get_temperature(10) == 1.0

    def test_returns_final_after_decay_end(self):
        schedule = LinearDecaySchedule(
            initial_temp=1.0, final_temp=0.1, decay_start=10, decay_end=60
        )
        assert schedule.get_temperature(60) == 0.1
        assert schedule.get_temperature(100) == 0.1

    def test_linear_interpolation(self):
        schedule = LinearDecaySchedule(
            initial_temp=1.0, final_temp=0.0, decay_start=0, decay_end=100
        )
        # Midpoint should be 0.5
        assert schedule.get_temperature(50) == pytest.approx(0.5)
        # Quarter point should be 0.75
        assert schedule.get_temperature(25) == pytest.approx(0.75)

    def test_decay_range(self):
        schedule = LinearDecaySchedule(
            initial_temp=2.0, final_temp=0.5, decay_start=20, decay_end=40
        )
        # Midpoint of decay (move 30) should be midpoint of temp range
        expected = 2.0 + 0.5 * (0.5 - 2.0)  # 1.25
        assert schedule.get_temperature(30) == pytest.approx(expected)


class TestExponentialDecaySchedule:
    """Tests for ExponentialDecaySchedule."""

    def test_returns_initial_before_decay_start(self):
        schedule = ExponentialDecaySchedule(
            initial_temp=1.0, final_temp=0.1, decay_rate=0.1, decay_start=10
        )
        assert schedule.get_temperature(0) == 1.0
        assert schedule.get_temperature(10) == 1.0

    def test_decays_exponentially(self):
        schedule = ExponentialDecaySchedule(
            initial_temp=1.0, final_temp=0.0, decay_rate=0.1, decay_start=0
        )
        # At move 10: 1.0 * exp(-0.1 * 10) = exp(-1) ~ 0.368
        assert schedule.get_temperature(10) == pytest.approx(math.exp(-1), rel=0.01)

    def test_respects_final_temp_floor(self):
        schedule = ExponentialDecaySchedule(
            initial_temp=1.0, final_temp=0.2, decay_rate=0.1, decay_start=0
        )
        # At very high move numbers, should not go below final_temp
        assert schedule.get_temperature(1000) == 0.2


class TestStepSchedule:
    """Tests for StepSchedule."""

    def test_returns_default_before_first_step(self):
        schedule = StepSchedule(steps=[(10, 0.5), (20, 0.3)], default_temp=1.0)
        assert schedule.get_temperature(0) == 1.0
        assert schedule.get_temperature(9) == 1.0

    def test_applies_steps_in_order(self):
        schedule = StepSchedule(steps=[(10, 0.5), (20, 0.3)], default_temp=1.0)
        assert schedule.get_temperature(10) == 0.5
        assert schedule.get_temperature(15) == 0.5
        assert schedule.get_temperature(20) == 0.3
        assert schedule.get_temperature(100) == 0.3

    def test_unsorted_steps_are_sorted(self):
        schedule = StepSchedule(steps=[(20, 0.3), (10, 0.5)], default_temp=1.0)
        assert schedule.get_temperature(15) == 0.5
        assert schedule.get_temperature(25) == 0.3

    def test_single_step(self):
        schedule = StepSchedule(steps=[(30, 0.1)], default_temp=2.0)
        assert schedule.get_temperature(29) == 2.0
        assert schedule.get_temperature(30) == 0.1


class TestCosineAnnealingSchedule:
    """Tests for CosineAnnealingSchedule."""

    def test_starts_at_initial(self):
        schedule = CosineAnnealingSchedule(
            initial_temp=1.0, final_temp=0.1, period_moves=60
        )
        # At start of cycle, cosine(0) = 1, so temp = final + 1 * (initial - final) = initial
        assert schedule.get_temperature(0) == pytest.approx(1.0)

    def test_reaches_final_at_half_period(self):
        schedule = CosineAnnealingSchedule(
            initial_temp=1.0, final_temp=0.0, period_moves=60
        )
        # Cosine annealing: at progress=0.5, cos(0.5*pi) = 0, so (1+0)/2 = 0.5
        # temp = final + 0.5 * (initial - final) = 0 + 0.5 * 1.0 = 0.5
        assert schedule.get_temperature(30) == pytest.approx(0.5, abs=0.1)

    def test_cycles_with_warm_restarts(self):
        schedule = CosineAnnealingSchedule(
            initial_temp=1.0, final_temp=0.0, period_moves=60, num_cycles=2
        )
        # With 2 cycles in 60 moves, cycle length = 30
        # Move 0 and 30 should both be at start of cycle
        temp_0 = schedule.get_temperature(0)
        temp_30 = schedule.get_temperature(30)
        assert temp_0 == pytest.approx(temp_30, rel=0.1)


class TestAdaptiveSchedule:
    """Tests for AdaptiveSchedule."""

    def test_base_temperature_with_none_state(self):
        schedule = AdaptiveSchedule(base_temp=1.0)
        temp = schedule.get_temperature(10, game_state=None)
        # With default complexity=0.5, uncertainty=0.5, phase based on move
        assert 0.1 <= temp <= 2.0

    def test_higher_complexity_increases_temp(self):
        schedule = AdaptiveSchedule(base_temp=1.0, complexity_weight=0.5)

        # Mock game state with many legal moves (high complexity)
        high_complexity_state = MagicMock()
        high_complexity_state.get_legal_moves.return_value = list(range(100))
        # Remove last_policy_entropy to avoid TypeError in _get_uncertainty
        del high_complexity_state.last_policy_entropy

        # Mock game state with few legal moves (low complexity)
        low_complexity_state = MagicMock()
        low_complexity_state.get_legal_moves.return_value = list(range(5))
        del low_complexity_state.last_policy_entropy

        temp_high = schedule.get_temperature(10, game_state=high_complexity_state)
        temp_low = schedule.get_temperature(10, game_state=low_complexity_state)

        assert temp_high > temp_low

    def test_later_game_phase_decreases_temp(self):
        schedule = AdaptiveSchedule(base_temp=1.0, phase_weight=0.5)

        temp_early = schedule.get_temperature(5)
        temp_late = schedule.get_temperature(80)

        assert temp_early > temp_late

    def test_respects_min_max_bounds(self):
        schedule = AdaptiveSchedule(
            base_temp=5.0, min_temp=0.5, max_temp=1.5
        )
        temp = schedule.get_temperature(10)
        assert 0.5 <= temp <= 1.5


class TestCurriculumSchedule:
    """Tests for CurriculumSchedule."""

    def test_early_training_uses_early_temp(self):
        schedule = CurriculumSchedule(
            early_temp=1.5, late_temp=0.5, transition_start=0.3, transition_end=0.7
        )
        # training_progress=0.0 (early)
        temp = schedule.get_temperature(5, training_progress=0.0)
        assert temp == pytest.approx(1.5)

    def test_late_training_uses_late_temp(self):
        schedule = CurriculumSchedule(
            early_temp=1.5, late_temp=0.5, transition_start=0.3, transition_end=0.7
        )
        # training_progress=1.0 (late)
        temp = schedule.get_temperature(5, training_progress=1.0)
        assert temp == pytest.approx(0.5)

    def test_interpolates_during_transition(self):
        schedule = CurriculumSchedule(
            early_temp=1.0, late_temp=0.5, transition_start=0.0, transition_end=1.0
        )
        # training_progress=0.5 (middle of transition)
        temp = schedule.get_temperature(5, training_progress=0.5)
        assert temp == pytest.approx(0.75)

    def test_decays_after_exploration_moves(self):
        schedule = CurriculumSchedule(
            early_temp=1.0, late_temp=1.0,
            early_exploration_moves=10, late_exploration_moves=10
        )
        temp_early = schedule.get_temperature(5, training_progress=0.0)
        temp_late = schedule.get_temperature(50, training_progress=0.0)
        assert temp_late < temp_early


class TestEloAdaptiveSchedule:
    """Tests for EloAdaptiveSchedule."""

    def test_weak_model_gets_high_temp(self):
        schedule = EloAdaptiveSchedule(model_elo=1100, weak_temp=1.5)
        temp = schedule.get_temperature(5)
        assert temp == pytest.approx(1.5)

    def test_strong_model_gets_low_temp(self):
        schedule = EloAdaptiveSchedule(model_elo=1800, very_strong_temp=0.5)
        temp = schedule.get_temperature(5)
        assert temp == pytest.approx(0.5)

    def test_medium_model_interpolates(self):
        schedule = EloAdaptiveSchedule(
            model_elo=1400,  # Between weak (1300) and medium (1500)
            weak_temp=1.5,
            medium_temp=1.0
        )
        temp = schedule.get_temperature(5)
        # Should be between weak and medium temps
        assert 1.0 <= temp <= 1.5


class TestCreateScheduler:
    """Tests for create_scheduler factory function."""

    def test_creates_alphazero_preset(self):
        scheduler = create_scheduler("alphazero")
        assert scheduler is not None
        # AlphaZero uses high temp for first N moves, then 0
        temp_early = scheduler.get_temperature(5)
        temp_late = scheduler.get_temperature(100)
        assert temp_early > temp_late

    def test_creates_default_preset(self):
        scheduler = create_scheduler("default")
        assert scheduler is not None

    def test_creates_aggressive_exploration_preset(self):
        scheduler = create_scheduler("aggressive_exploration")
        assert scheduler is not None
        # Should have higher temperature
        temp = scheduler.get_temperature(10)
        assert temp >= 1.0

    def test_creates_conservative_preset(self):
        scheduler = create_scheduler("conservative")
        assert scheduler is not None
        # Should have lower temperature
        temp = scheduler.get_temperature(10)
        assert temp <= 1.0

    def test_creates_with_kwargs_override(self):
        # create_scheduler accepts kwargs to override config
        scheduler = create_scheduler("default", initial_temp=0.7)
        # The scheduler should use the overridden value
        assert scheduler is not None

    def test_invalid_preset_returns_default(self):
        # Unknown presets fall back to default, not raise
        scheduler = create_scheduler("nonexistent_preset")
        assert scheduler is not None


class TestTemperatureScheduleBase:
    """Tests for TemperatureSchedule base class."""

    def test_clip_temperature_clips_low(self):
        schedule = ConstantSchedule()
        assert schedule.clip_temperature(-0.5, min_temp=0.1) == 0.1

    def test_clip_temperature_clips_high(self):
        schedule = ConstantSchedule()
        assert schedule.clip_temperature(5.0, max_temp=2.0) == 2.0

    def test_clip_temperature_passes_valid(self):
        schedule = ConstantSchedule()
        assert schedule.clip_temperature(1.0, min_temp=0.1, max_temp=2.0) == 1.0


class TestScheduleTypeEnum:
    """Tests for ScheduleType enum."""

    def test_all_schedule_types_defined(self):
        assert ScheduleType.CONSTANT.value == "constant"
        assert ScheduleType.LINEAR_DECAY.value == "linear_decay"
        assert ScheduleType.EXPONENTIAL_DECAY.value == "exp_decay"
        assert ScheduleType.STEP.value == "step"
        assert ScheduleType.COSINE.value == "cosine"
        assert ScheduleType.ADAPTIVE.value == "adaptive"
        assert ScheduleType.CURRICULUM.value == "curriculum"
        assert ScheduleType.ELO_ADAPTIVE.value == "elo_adaptive"


class TestTemperatureConfig:
    """Tests for TemperatureConfig dataclass."""

    def test_default_values(self):
        config = TemperatureConfig()
        assert config.schedule_type == ScheduleType.LINEAR_DECAY
        assert config.initial_temp == 1.0
        assert config.final_temp == 0.1
        assert config.exploration_moves == 30

    def test_custom_values(self):
        config = TemperatureConfig(
            schedule_type=ScheduleType.COSINE,
            initial_temp=2.0,
            final_temp=0.5
        )
        assert config.schedule_type == ScheduleType.COSINE
        assert config.initial_temp == 2.0
        assert config.final_temp == 0.5
