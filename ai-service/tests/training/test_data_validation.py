"""Tests for training data validation module.

Tests DataValidator and GameDeduplicator functionality.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from app.training.data_validation import (
    DataValidator,
    DataValidatorConfig,
    GameDeduplicator,
    ValidationIssueType,
    ValidationResult,
    validate_npz_file,
    validate_training_data,
)


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_valid_data_passes(self):
        """Valid training data should pass validation."""
        data = {
            'features': np.random.randn(100, 8, 8, 10).astype(np.float32),
            'values': np.random.uniform(-1, 1, 100).astype(np.float32),
            'policy_values': np.random.dirichlet(np.ones(64), 100).astype(np.float32),
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert result.valid
        assert result.total_samples == 100
        assert len(result.issues) == 0

    def test_missing_values_array_fails(self):
        """Missing values array should fail validation."""
        data = {
            'features': np.random.randn(100, 8, 8, 10).astype(np.float32),
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert not result.valid
        assert any(i.issue_type == ValidationIssueType.MISSING_ARRAY for i in result.issues)

    def test_value_out_of_range_detected(self):
        """Values outside [-1, 1] should be flagged."""
        data = {
            'values': np.array([0.5, -0.3, 1.5, -1.2, 0.0]),  # 1.5 and -1.2 out of range
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert not result.valid
        out_of_range_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.VALUE_OUT_OF_RANGE
        ]
        assert len(out_of_range_issues) == 2

    def test_nan_values_detected(self):
        """NaN values should be flagged."""
        data = {
            'values': np.array([0.5, np.nan, -0.3]),
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert not result.valid
        nan_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.FEATURE_NAN
        ]
        assert len(nan_issues) >= 1

    def test_policy_sum_validation(self):
        """Policies not summing to 1 should be flagged."""
        data = {
            'values': np.array([0.5, 0.3, -0.2]),
            'policy_values': np.array([
                [0.25, 0.25, 0.25, 0.25],  # Sum = 1.0 ✓
                [0.5, 0.5, 0.5, 0.5],      # Sum = 2.0 ✗
                [0.1, 0.1, 0.1, 0.1],      # Sum = 0.4 ✗
            ]),
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert not result.valid
        policy_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.POLICY_SUM_INVALID
        ]
        assert len(policy_issues) == 2

    def test_empty_policy_allowed_by_default(self):
        """Empty policies (terminal states) should be allowed by default."""
        data = {
            'values': np.array([0.5, 1.0]),  # Second is terminal
            'policy_values': np.array([
                [0.25, 0.25, 0.25, 0.25],  # Normal
                [0.0, 0.0, 0.0, 0.0],      # Empty (terminal)
            ]),
        }

        config = DataValidatorConfig(allow_empty_policies=True)
        validator = DataValidator(config)
        result = validator.validate_arrays(data)

        # Should not flag the empty policy
        policy_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.POLICY_SUM_INVALID
        ]
        assert len(policy_issues) == 0

    def test_negative_policy_detected(self):
        """Negative probabilities in policy should be flagged."""
        data = {
            'values': np.array([0.5]),
            'policy_values': np.array([[0.5, 0.3, -0.1, 0.3]]),  # -0.1 is invalid
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert not result.valid
        negative_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.POLICY_NEGATIVE
        ]
        assert len(negative_issues) >= 1

    def test_feature_nan_detected(self):
        """NaN in features should be flagged."""
        features = np.random.randn(10, 8, 8, 5).astype(np.float32)
        features[3, 4, 4, 2] = np.nan

        data = {
            'values': np.random.uniform(-1, 1, 10).astype(np.float32),
            'features': features,
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert not result.valid
        nan_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.FEATURE_NAN
        ]
        assert len(nan_issues) >= 1

    def test_feature_inf_detected(self):
        """Inf in features should be flagged."""
        features = np.random.randn(10, 8, 8, 5).astype(np.float32)
        features[5, 2, 3, 1] = np.inf

        data = {
            'values': np.random.uniform(-1, 1, 10).astype(np.float32),
            'features': features,
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert not result.valid
        inf_issues = [
            i for i in result.issues
            if i.issue_type == ValidationIssueType.FEATURE_INF
        ]
        assert len(inf_issues) >= 1

    def test_value_stats_computed(self):
        """Value statistics should be computed."""
        data = {
            'values': np.array([-0.5, 0.0, 0.5, 1.0]),
        }

        validator = DataValidator()
        result = validator.validate_arrays(data)

        assert result.value_stats is not None
        assert result.value_stats['min'] == -0.5
        assert result.value_stats['max'] == 1.0
        assert abs(result.value_stats['mean'] - 0.25) < 0.01

    def test_max_issues_limit(self):
        """Should not report more than max_issues_to_report."""
        data = {
            'values': np.full(1000, 5.0),  # All out of range
        }

        config = DataValidatorConfig(max_issues_to_report=10)
        validator = DataValidator(config)
        result = validator.validate_arrays(data)

        assert not result.valid
        assert len(result.issues) <= 10


class TestGameDeduplicator:
    """Tests for GameDeduplicator class."""

    def test_first_game_not_duplicate(self):
        """First game should never be a duplicate."""
        dedup = GameDeduplicator()
        moves = ['a1', 'b2', 'c3']

        assert not dedup.is_duplicate(moves)

    def test_same_game_is_duplicate(self):
        """Same move sequence should be detected as duplicate."""
        dedup = GameDeduplicator()
        moves = ['a1', 'b2', 'c3']

        assert not dedup.is_duplicate(moves)
        assert dedup.is_duplicate(moves)  # Second time

    def test_different_games_not_duplicate(self):
        """Different games should not be duplicates."""
        dedup = GameDeduplicator()

        assert not dedup.is_duplicate(['a1', 'b2', 'c3'])
        assert not dedup.is_duplicate(['a1', 'b2', 'd4'])
        assert not dedup.is_duplicate(['x1', 'y2', 'z3'])

    def test_stats_tracking(self):
        """Statistics should be tracked correctly."""
        dedup = GameDeduplicator()

        dedup.is_duplicate(['a1', 'b2'])
        dedup.is_duplicate(['a1', 'b2'])  # Duplicate
        dedup.is_duplicate(['c3', 'd4'])
        dedup.is_duplicate(['a1', 'b2'])  # Duplicate again

        stats = dedup.stats()
        assert stats['total_checked'] == 4
        assert stats['duplicates_found'] == 2
        assert stats['unique_games'] == 2
        assert stats['duplicate_rate'] == 0.5

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        dedup = GameDeduplicator()

        dedup.is_duplicate(['a1', 'b2'])
        dedup.is_duplicate(['a1', 'b2'])

        dedup.reset()

        # Same game should not be duplicate after reset
        assert not dedup.is_duplicate(['a1', 'b2'])
        assert dedup.stats()['total_checked'] == 1

    def test_hash_based_dedup(self):
        """Hash-based deduplication should work."""
        dedup = GameDeduplicator()

        hash1 = "abc123def456"
        hash2 = "xyz789uvw012"

        assert not dedup.is_duplicate_by_hash(hash1)
        assert dedup.is_duplicate_by_hash(hash1)  # Duplicate
        assert not dedup.is_duplicate_by_hash(hash2)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_summary_valid(self):
        """Valid result should show success summary."""
        result = ValidationResult(valid=True, total_samples=100)
        summary = result.summary()

        assert "✓ Valid" in summary
        assert "100 samples" in summary

    def test_summary_invalid(self):
        """Invalid result should show issue counts."""
        result = ValidationResult(valid=False, total_samples=100, samples_with_issues=5)
        result.issues = [
            type('Issue', (), {'issue_type': ValidationIssueType.VALUE_OUT_OF_RANGE})(),
            type('Issue', (), {'issue_type': ValidationIssueType.VALUE_OUT_OF_RANGE})(),
            type('Issue', (), {'issue_type': ValidationIssueType.POLICY_SUM_INVALID})(),
        ]

        summary = result.summary()

        assert "✗ Invalid" in summary
        assert "5/100" in summary


class TestNPZValidation:
    """Tests for NPZ file validation."""

    def test_validate_npz_file(self):
        """Should validate NPZ file correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.npz"

            # Create valid test data
            np.savez_compressed(
                path,
                features=np.random.randn(50, 8, 8, 5).astype(np.float32),
                values=np.random.uniform(-1, 1, 50).astype(np.float32),
            )

            result = validate_npz_file(str(path))

            assert result.valid
            assert result.total_samples == 50

    def test_validate_corrupt_npz(self):
        """Should handle corrupt NPZ file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.npz"
            path.write_bytes(b"not a valid npz file")

            result = validate_npz_file(str(path))

            assert not result.valid
            assert any(i.issue_type == ValidationIssueType.CORRUPT_FILE for i in result.issues)
