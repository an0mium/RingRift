"""Tests for app.quality.thresholds - Quality Thresholds Module.

This module tests the quality threshold utilities that provide convenience
functions for checking game quality against canonical thresholds.
"""

from __future__ import annotations

import pytest

from app.quality.thresholds import (
    HIGH_QUALITY_THRESHOLD,
    MIN_QUALITY_FOR_PRIORITY_SYNC,
    MIN_QUALITY_FOR_TRAINING,
    QualityThresholds,
    get_quality_thresholds,
    is_high_quality,
    is_priority_sync_worthy,
    is_training_worthy,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestThresholdConstants:
    """Tests for threshold constants."""

    def test_thresholds_are_floats(self):
        """All thresholds should be floats."""
        assert isinstance(MIN_QUALITY_FOR_TRAINING, float)
        assert isinstance(MIN_QUALITY_FOR_PRIORITY_SYNC, float)
        assert isinstance(HIGH_QUALITY_THRESHOLD, float)

    def test_thresholds_in_valid_range(self):
        """All thresholds should be in [0, 1]."""
        assert 0.0 <= MIN_QUALITY_FOR_TRAINING <= 1.0
        assert 0.0 <= MIN_QUALITY_FOR_PRIORITY_SYNC <= 1.0
        assert 0.0 <= HIGH_QUALITY_THRESHOLD <= 1.0

    def test_threshold_ordering(self):
        """Thresholds should be in logical order."""
        assert MIN_QUALITY_FOR_TRAINING <= MIN_QUALITY_FOR_PRIORITY_SYNC
        assert MIN_QUALITY_FOR_PRIORITY_SYNC <= HIGH_QUALITY_THRESHOLD

    def test_expected_default_values(self):
        """Default values should match expected values."""
        assert MIN_QUALITY_FOR_TRAINING == 0.3
        assert MIN_QUALITY_FOR_PRIORITY_SYNC == 0.5
        assert HIGH_QUALITY_THRESHOLD == 0.7


# =============================================================================
# QualityThresholds Dataclass Tests
# =============================================================================


class TestQualityThresholds:
    """Tests for QualityThresholds dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        thresholds = QualityThresholds()

        assert thresholds.min_quality_for_training == MIN_QUALITY_FOR_TRAINING
        assert thresholds.min_quality_for_priority_sync == MIN_QUALITY_FOR_PRIORITY_SYNC
        assert thresholds.high_quality_threshold == HIGH_QUALITY_THRESHOLD

    def test_custom_values(self):
        """Should accept custom values."""
        thresholds = QualityThresholds(
            min_quality_for_training=0.4,
            min_quality_for_priority_sync=0.6,
            high_quality_threshold=0.8,
        )

        assert thresholds.min_quality_for_training == 0.4
        assert thresholds.min_quality_for_priority_sync == 0.6
        assert thresholds.high_quality_threshold == 0.8

    def test_is_frozen(self):
        """Dataclass should be frozen (immutable)."""
        thresholds = QualityThresholds()

        with pytest.raises(AttributeError):
            thresholds.min_quality_for_training = 0.5

    def test_is_training_worthy_method(self):
        """is_training_worthy method should work correctly."""
        thresholds = QualityThresholds()

        assert thresholds.is_training_worthy(0.5) is True
        assert thresholds.is_training_worthy(0.3) is True  # Boundary
        assert thresholds.is_training_worthy(0.2) is False

    def test_is_priority_sync_worthy_method(self):
        """is_priority_sync_worthy method should work correctly."""
        thresholds = QualityThresholds()

        assert thresholds.is_priority_sync_worthy(0.6) is True
        assert thresholds.is_priority_sync_worthy(0.5) is True  # Boundary
        assert thresholds.is_priority_sync_worthy(0.4) is False

    def test_is_high_quality_method(self):
        """is_high_quality method should work correctly."""
        thresholds = QualityThresholds()

        assert thresholds.is_high_quality(0.8) is True
        assert thresholds.is_high_quality(0.7) is True  # Boundary
        assert thresholds.is_high_quality(0.6) is False


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetQualityThresholds:
    """Tests for get_quality_thresholds function."""

    def test_returns_thresholds_instance(self):
        """Should return QualityThresholds instance."""
        thresholds = get_quality_thresholds()
        assert isinstance(thresholds, QualityThresholds)

    def test_uses_current_constants(self):
        """Should use current constant values."""
        thresholds = get_quality_thresholds()

        assert thresholds.min_quality_for_training == MIN_QUALITY_FOR_TRAINING
        assert thresholds.min_quality_for_priority_sync == MIN_QUALITY_FOR_PRIORITY_SYNC
        assert thresholds.high_quality_threshold == HIGH_QUALITY_THRESHOLD


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestIsTrainingWorthy:
    """Tests for is_training_worthy convenience function."""

    def test_above_threshold(self):
        """Scores above threshold should return True."""
        assert is_training_worthy(0.5) is True
        assert is_training_worthy(0.8) is True
        assert is_training_worthy(1.0) is True

    def test_at_threshold(self):
        """Score exactly at threshold should return True."""
        assert is_training_worthy(MIN_QUALITY_FOR_TRAINING) is True

    def test_below_threshold(self):
        """Scores below threshold should return False."""
        assert is_training_worthy(0.2) is False
        assert is_training_worthy(0.1) is False
        assert is_training_worthy(0.0) is False

    def test_edge_case_negative(self):
        """Negative scores should return False."""
        assert is_training_worthy(-0.1) is False


class TestIsPrioritySyncWorthy:
    """Tests for is_priority_sync_worthy convenience function."""

    def test_above_threshold(self):
        """Scores above threshold should return True."""
        assert is_priority_sync_worthy(0.6) is True
        assert is_priority_sync_worthy(0.8) is True
        assert is_priority_sync_worthy(1.0) is True

    def test_at_threshold(self):
        """Score exactly at threshold should return True."""
        assert is_priority_sync_worthy(MIN_QUALITY_FOR_PRIORITY_SYNC) is True

    def test_below_threshold(self):
        """Scores below threshold should return False."""
        assert is_priority_sync_worthy(0.4) is False
        assert is_priority_sync_worthy(0.3) is False
        assert is_priority_sync_worthy(0.0) is False


class TestIsHighQuality:
    """Tests for is_high_quality convenience function."""

    def test_above_threshold(self):
        """Scores above threshold should return True."""
        assert is_high_quality(0.8) is True
        assert is_high_quality(0.9) is True
        assert is_high_quality(1.0) is True

    def test_at_threshold(self):
        """Score exactly at threshold should return True."""
        assert is_high_quality(HIGH_QUALITY_THRESHOLD) is True

    def test_below_threshold(self):
        """Scores below threshold should return False."""
        assert is_high_quality(0.6) is False
        assert is_high_quality(0.5) is False
        assert is_high_quality(0.0) is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestThresholdIntegration:
    """Integration tests for threshold checks."""

    def test_quality_progression(self):
        """Test that thresholds define a clear quality progression."""
        # A score of 0.2 is not training worthy
        assert is_training_worthy(0.2) is False
        assert is_priority_sync_worthy(0.2) is False
        assert is_high_quality(0.2) is False

        # A score of 0.35 is training worthy only
        assert is_training_worthy(0.35) is True
        assert is_priority_sync_worthy(0.35) is False
        assert is_high_quality(0.35) is False

        # A score of 0.55 is training and priority sync worthy
        assert is_training_worthy(0.55) is True
        assert is_priority_sync_worthy(0.55) is True
        assert is_high_quality(0.55) is False

        # A score of 0.75 passes all thresholds
        assert is_training_worthy(0.75) is True
        assert is_priority_sync_worthy(0.75) is True
        assert is_high_quality(0.75) is True

    def test_threshold_boundaries(self):
        """Test exact boundary values."""
        # Just below training threshold
        just_below_training = MIN_QUALITY_FOR_TRAINING - 0.001
        assert is_training_worthy(just_below_training) is False

        # Exactly at training threshold
        assert is_training_worthy(MIN_QUALITY_FOR_TRAINING) is True

        # Just above training threshold
        just_above_training = MIN_QUALITY_FOR_TRAINING + 0.001
        assert is_training_worthy(just_above_training) is True

    def test_threshold_object_vs_functions(self):
        """QualityThresholds methods should match convenience functions."""
        thresholds = get_quality_thresholds()
        test_scores = [0.2, 0.35, 0.5, 0.55, 0.7, 0.75, 0.9]

        for score in test_scores:
            assert thresholds.is_training_worthy(score) == is_training_worthy(score)
            assert thresholds.is_priority_sync_worthy(score) == is_priority_sync_worthy(score)
            assert thresholds.is_high_quality(score) == is_high_quality(score)
