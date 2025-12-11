"""Tests for Shadow Validation Infrastructure.

These tests verify that the shadow validation system correctly:
1. Samples moves at the configured rate
2. Detects divergences between GPU and CPU move generation
3. Reports statistics accurately
4. Halts when threshold exceeded (if configured)

This is Phase 2 infrastructure - the safety net for GPU move generation.
"""

import pytest
import random
from typing import List, Tuple
from unittest.mock import MagicMock, patch

from app.models import MoveType

# Import shadow validation
from app.ai.shadow_validation import (
    ShadowValidator,
    ValidationStats,
    DivergenceType,
    DivergenceRecord,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DIVERGENCE_THRESHOLD,
    create_shadow_validator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def validator():
    """Create a shadow validator with 100% sample rate for testing."""
    return ShadowValidator(sample_rate=1.0, threshold=0.1, halt_on_threshold=False)


@pytest.fixture
def strict_validator():
    """Create a strict validator that halts on any divergence."""
    return ShadowValidator(sample_rate=1.0, threshold=0.0, halt_on_threshold=True)


@pytest.fixture
def mock_game_state():
    """Create a mock game state for testing."""
    state = MagicMock()
    state.move_count = 10
    state.current_player = 1
    return state


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestShadowValidatorBasics:
    """Test basic shadow validator functionality."""

    def test_creation_with_defaults(self):
        """Validator creates with default settings."""
        validator = ShadowValidator()
        assert validator.sample_rate == DEFAULT_SAMPLE_RATE
        assert validator.threshold == DEFAULT_DIVERGENCE_THRESHOLD
        assert validator.halt_on_threshold is True

    def test_creation_with_custom_settings(self):
        """Validator creates with custom settings."""
        validator = ShadowValidator(
            sample_rate=0.1,
            threshold=0.05,
            halt_on_threshold=False,
        )
        assert validator.sample_rate == 0.1
        assert validator.threshold == 0.05
        assert validator.halt_on_threshold is False

    def test_initial_stats_are_zero(self, validator):
        """Initial validation stats are all zero."""
        assert validator.stats.total_validations == 0
        assert validator.stats.total_divergences == 0
        assert validator.stats.divergence_rate == 0.0

    def test_seed_reproducibility(self):
        """Setting seed produces reproducible sampling."""
        validator1 = ShadowValidator(sample_rate=0.5)
        validator1.set_seed(42)

        validator2 = ShadowValidator(sample_rate=0.5)
        validator2.set_seed(42)

        # Same seed should produce same sampling decisions
        results1 = [validator1.should_validate() for _ in range(100)]
        results2 = [validator2.should_validate() for _ in range(100)]
        assert results1 == results2


class TestSampling:
    """Test probabilistic sampling behavior."""

    def test_zero_sample_rate_never_validates(self):
        """Sample rate 0 never triggers validation."""
        validator = ShadowValidator(sample_rate=0.0)
        for _ in range(100):
            assert validator.should_validate() is False

    def test_full_sample_rate_always_validates(self):
        """Sample rate 1.0 always triggers validation."""
        validator = ShadowValidator(sample_rate=1.0)
        for _ in range(100):
            assert validator.should_validate() is True

    def test_partial_sample_rate_approximate(self):
        """Sample rate 0.5 validates approximately half the time."""
        validator = ShadowValidator(sample_rate=0.5)
        validator.set_seed(42)

        validations = sum(1 for _ in range(1000) if validator.should_validate())
        # Should be roughly 500, allow 10% margin
        assert 400 < validations < 600


# =============================================================================
# Placement Move Validation Tests
# =============================================================================


class TestPlacementValidation:
    """Test placement move validation."""

    def test_matching_placements_pass(self, validator, mock_game_state):
        """Matching placement positions pass validation."""
        gpu_positions = [(0, 0), (1, 1), (2, 2)]

        # Mock CPU to return same positions (Position uses x, y)
        mock_moves = []
        for x, y in gpu_positions:
            m = MagicMock()
            m.type = MoveType.PLACE_RING
            m.player = 1
            m.to = MagicMock(x=x, y=y)
            mock_moves.append(m)

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = mock_moves

            result = validator.validate_placement_moves(
                gpu_positions, mock_game_state, player=1
            )

            assert result is True
            assert validator.stats.placement_validations == 1
            assert validator.stats.placement_divergences == 0

    def test_missing_placement_fails(self, validator, mock_game_state):
        """Missing placement position fails validation."""
        gpu_positions = [(0, 0), (1, 1)]  # Missing (2, 2)

        # CPU has one more position (Position uses x, y)
        mock_moves = []
        for x, y in [(0, 0), (1, 1), (2, 2)]:
            m = MagicMock()
            m.type = MoveType.PLACE_RING
            m.player = 1
            m.to = MagicMock(x=x, y=y)
            mock_moves.append(m)

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = mock_moves

            result = validator.validate_placement_moves(
                gpu_positions, mock_game_state, player=1
            )

            assert result is False
            assert validator.stats.placement_divergences == 1
            assert len(validator.divergence_log) == 1

    def test_extra_placement_fails(self, validator, mock_game_state):
        """Extra placement position fails validation."""
        gpu_positions = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Extra (3, 3)

        mock_moves = []
        for r, c in [(0, 0), (1, 1), (2, 2)]:
            m = MagicMock()
            m.type = "placement"
            m.player = 1
            m.to = MagicMock(row=r, col=c)
            mock_moves.append(m)

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = mock_moves

            result = validator.validate_placement_moves(
                gpu_positions, mock_game_state, player=1
            )

            assert result is False
            assert validator.stats.placement_divergences == 1


# =============================================================================
# Movement/Capture/Recovery Validation Tests
# =============================================================================


class TestMovementValidation:
    """Test movement move validation."""

    def test_matching_movements_pass(self, validator, mock_game_state):
        """Matching movement moves pass validation."""
        gpu_moves = [((0, 0), (2, 0)), ((1, 1), (3, 1))]

        mock_moves = []
        for (fr, fc), (tr, tc) in gpu_moves:
            m = MagicMock()
            m.type = "movement"
            m.player = 1
            m.from_pos = MagicMock(row=fr, col=fc)
            m.to = MagicMock(row=tr, col=tc)
            mock_moves.append(m)

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = mock_moves

            result = validator.validate_movement_moves(
                gpu_moves, mock_game_state, player=1
            )

            assert result is True
            assert validator.stats.movement_validations == 1
            assert validator.stats.movement_divergences == 0


class TestCaptureValidation:
    """Test capture move validation."""

    def test_matching_captures_pass(self, validator, mock_game_state):
        """Matching capture moves pass validation."""
        gpu_moves = [((0, 0), (2, 0))]

        mock_moves = []
        for (fr, fc), (tr, tc) in gpu_moves:
            m = MagicMock()
            m.type = "capture"
            m.player = 1
            m.from_pos = MagicMock(row=fr, col=fc)
            m.to = MagicMock(row=tr, col=tc)
            mock_moves.append(m)

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = mock_moves

            result = validator.validate_capture_moves(
                gpu_moves, mock_game_state, player=1
            )

            assert result is True
            assert validator.stats.capture_validations == 1


class TestRecoveryValidation:
    """Test recovery move validation."""

    def test_matching_recovery_pass(self, validator, mock_game_state):
        """Matching recovery moves pass validation."""
        gpu_moves = [((3, 3), (3, 4))]

        mock_moves = []
        for (fr, fc), (tr, tc) in gpu_moves:
            m = MagicMock()
            m.type = "recovery_slide"
            m.player = 1
            m.from_pos = MagicMock(row=fr, col=fc)
            m.to = MagicMock(row=tr, col=tc)
            mock_moves.append(m)

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = mock_moves

            result = validator.validate_recovery_moves(
                gpu_moves, mock_game_state, player=1
            )

            assert result is True
            assert validator.stats.recovery_validations == 1


# =============================================================================
# Threshold and Halting Tests
# =============================================================================


class TestThresholdBehavior:
    """Test divergence threshold behavior."""

    def test_halt_when_threshold_exceeded(self, mock_game_state):
        """Validator halts when divergence threshold exceeded."""
        validator = ShadowValidator(
            sample_rate=1.0,
            threshold=0.0,  # Any divergence triggers halt
            halt_on_threshold=True,
        )

        gpu_positions = [(0, 0)]
        mock_moves = []  # CPU has no moves - will diverge

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = mock_moves

            with pytest.raises(RuntimeError, match="exceeds threshold"):
                validator.validate_placement_moves(
                    gpu_positions, mock_game_state, player=1
                )

    def test_no_halt_when_disabled(self, mock_game_state):
        """Validator doesn't halt when halt_on_threshold=False."""
        validator = ShadowValidator(
            sample_rate=1.0,
            threshold=0.0,
            halt_on_threshold=False,  # Don't halt
        )

        gpu_positions = [(0, 0)]
        mock_moves = []

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = mock_moves

            # Should not raise
            result = validator.validate_placement_moves(
                gpu_positions, mock_game_state, player=1
            )
            assert result is False


# =============================================================================
# Statistics and Reporting Tests
# =============================================================================


class TestStatisticsReporting:
    """Test statistics tracking and reporting."""

    def test_divergence_rate_calculation(self, validator, mock_game_state):
        """Divergence rate calculated correctly."""
        # First: passing validation
        with patch("app.game_engine.GameEngine") as mock_engine:
            m = MagicMock()
            m.type = "placement"
            m.player = 1
            m.to = MagicMock(row=0, col=0)
            mock_engine.get_valid_moves.return_value = [m]

            validator.validate_placement_moves([(0, 0)], mock_game_state, player=1)

        assert validator.stats.divergence_rate == 0.0

        # Second: failing validation
        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = []

            validator.validate_placement_moves([(1, 1)], mock_game_state, player=1)

        assert validator.stats.total_validations == 2
        assert validator.stats.total_divergences == 1
        assert validator.stats.divergence_rate == 0.5

    def test_report_structure(self, validator, mock_game_state):
        """Report contains expected fields."""
        report = validator.get_report()

        assert "total_validations" in report
        assert "total_divergences" in report
        assert "divergence_rate" in report
        assert "threshold" in report
        assert "status" in report
        assert "by_move_type" in report
        assert "timing" in report
        assert "recent_divergences" in report

        assert report["status"] == "PASS"

    def test_reset_stats(self, validator, mock_game_state):
        """Reset clears all statistics."""
        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.get_valid_moves.return_value = []
            validator.validate_placement_moves([(0, 0)], mock_game_state, player=1)

        assert validator.stats.total_validations > 0

        validator.reset_stats()

        assert validator.stats.total_validations == 0
        assert validator.stats.total_divergences == 0
        assert len(validator.divergence_log) == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Test create_shadow_validator factory function."""

    def test_create_enabled_validator(self):
        """Factory creates enabled validator."""
        validator = create_shadow_validator(enabled=True)
        assert validator is not None
        assert isinstance(validator, ShadowValidator)

    def test_create_disabled_returns_none(self):
        """Factory returns None when disabled."""
        validator = create_shadow_validator(enabled=False)
        assert validator is None

    def test_create_with_custom_settings(self):
        """Factory respects custom settings."""
        validator = create_shadow_validator(
            sample_rate=0.1,
            threshold=0.05,
            enabled=True,
        )
        assert validator is not None
        assert validator.sample_rate == 0.1
        assert validator.threshold == 0.05
