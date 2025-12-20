"""Tests for Shadow Validation: GPU vs CPU Parity Verification.

Tests the shadow validation system that verifies GPU-generated moves
against the canonical CPU rules engine.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.ai.shadow_validation import (
    DEFAULT_DIVERGENCE_THRESHOLD,
    DEFAULT_SAMPLE_RATE,
    DivergenceRecord,
    DivergenceType,
    ShadowValidator,
    ValidationStats,
)


class TestDivergenceType:
    """Tests for DivergenceType enum."""

    def test_all_divergence_types_defined(self):
        """All divergence types should be defined."""
        assert DivergenceType.MISSING_IN_GPU.value == "missing_in_gpu"
        assert DivergenceType.EXTRA_IN_GPU.value == "extra_in_gpu"
        assert DivergenceType.MOVE_COUNT_MISMATCH.value == "move_count_mismatch"
        assert DivergenceType.MOVE_DETAILS_MISMATCH.value == "move_details_mismatch"

    def test_divergence_type_count(self):
        """Should have exactly 4 divergence types."""
        assert len(DivergenceType) == 4


class TestDivergenceRecord:
    """Tests for DivergenceRecord dataclass."""

    def test_create_record(self):
        """Should create DivergenceRecord with all fields."""
        record = DivergenceRecord(
            timestamp=time.time(),
            game_index=0,
            move_number=5,
            divergence_type=DivergenceType.MISSING_IN_GPU,
            cpu_move_count=10,
            gpu_move_count=8,
            missing_moves=["(3,4)", "(5,6)"],
            extra_moves=[],
        )
        assert record.game_index == 0
        assert record.move_number == 5
        assert record.cpu_move_count == 10
        assert record.gpu_move_count == 8
        assert len(record.missing_moves) == 2

    def test_record_with_hash(self):
        """Should support optional game_state_hash."""
        record = DivergenceRecord(
            timestamp=time.time(),
            game_index=1,
            move_number=10,
            divergence_type=DivergenceType.EXTRA_IN_GPU,
            cpu_move_count=5,
            gpu_move_count=7,
            missing_moves=[],
            extra_moves=["(1,2)", "(3,4)"],
            game_state_hash="abc123",
        )
        assert record.game_state_hash == "abc123"


class TestValidationStats:
    """Tests for ValidationStats dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        stats = ValidationStats()
        assert stats.total_validations == 0
        assert stats.total_divergences == 0
        assert stats.divergence_by_type == {}
        assert stats.total_validation_time_ms == 0.0

    def test_divergence_rate_zero_validations(self):
        """Divergence rate should be 0 with no validations."""
        stats = ValidationStats()
        assert stats.divergence_rate == 0.0

    def test_divergence_rate_calculation(self):
        """Should calculate divergence rate correctly."""
        stats = ValidationStats()
        stats.total_validations = 100
        stats.total_divergences = 5
        assert stats.divergence_rate == 0.05

    def test_avg_validation_time_zero(self):
        """Avg validation time should be 0 with no validations."""
        stats = ValidationStats()
        assert stats.avg_validation_time_ms == 0.0

    def test_avg_validation_time_calculation(self):
        """Should calculate average validation time correctly."""
        stats = ValidationStats()
        stats.total_validations = 10
        stats.total_validation_time_ms = 100.0
        assert stats.avg_validation_time_ms == 10.0


class TestShadowValidator:
    """Tests for ShadowValidator class."""

    def test_init_default_values(self):
        """Should initialize with default values."""
        validator = ShadowValidator()
        assert validator.sample_rate == DEFAULT_SAMPLE_RATE
        assert validator.threshold == DEFAULT_DIVERGENCE_THRESHOLD
        assert validator.halt_on_threshold is True
        assert validator.stats.total_validations == 0

    def test_init_custom_values(self):
        """Should initialize with custom values."""
        validator = ShadowValidator(
            sample_rate=0.1,
            threshold=0.01,
            halt_on_threshold=False,
        )
        assert validator.sample_rate == 0.1
        assert validator.threshold == 0.01
        assert validator.halt_on_threshold is False

    def test_set_seed_reproducible(self):
        """set_seed should make sampling reproducible."""
        validator1 = ShadowValidator(sample_rate=0.5)
        validator2 = ShadowValidator(sample_rate=0.5)

        validator1.set_seed(42)
        validator2.set_seed(42)

        results1 = [validator1.should_validate() for _ in range(10)]
        results2 = [validator2.should_validate() for _ in range(10)]

        assert results1 == results2

    def test_should_validate_probabilistic(self):
        """should_validate should respect sample rate."""
        validator = ShadowValidator(sample_rate=0.5)
        validator.set_seed(42)

        # Run many times and check distribution
        results = [validator.should_validate() for _ in range(1000)]
        true_count = sum(results)

        # With 50% rate, expect ~500 true (allow ±10%)
        assert 400 < true_count < 600

    def test_should_validate_always_with_rate_1(self):
        """should_validate should always return True with rate 1.0."""
        validator = ShadowValidator(sample_rate=1.0)

        results = [validator.should_validate() for _ in range(100)]
        assert all(results)

    def test_should_validate_never_with_rate_0(self):
        """should_validate should always return False with rate 0.0."""
        validator = ShadowValidator(sample_rate=0.0)

        results = [validator.should_validate() for _ in range(100)]
        assert not any(results)


class TestShadowValidatorStats:
    """Tests for validation statistics."""

    def test_divergence_log_limited(self):
        """Divergence log should be limited to max size."""
        validator = ShadowValidator(sample_rate=1.0, max_divergence_log=5)

        # Record more divergences than limit
        for i in range(10):
            validator._record_divergence(
                game_index=i,
                move_number=0,
                divergence_type=DivergenceType.MISSING_IN_GPU,
                cpu_count=1,
                gpu_count=0,
                missing=["test"],
                extra=[],
            )

        assert len(validator.divergence_log) == 5


class TestValidatorReport:
    """Tests for validation reporting."""

    def test_get_report_empty(self):
        """Should return report with zero stats."""
        validator = ShadowValidator()
        report = validator.get_report()

        assert "total_validations" in report
        assert report["total_validations"] == 0
        assert report["divergence_rate"] == 0.0

    def test_get_report_with_data(self):
        """Should return report with validation data."""
        validator = ShadowValidator()
        validator.stats.total_validations = 100
        validator.stats.total_divergences = 2
        validator.stats.placement_validations = 50
        validator.stats.placement_divergences = 1

        report = validator.get_report()

        assert report["total_validations"] == 100
        assert report["divergence_rate"] == 0.02
        # Placement stats are nested under by_move_type
        assert report["by_move_type"]["placement"]["validations"] == 50

    def test_reset_stats(self):
        """Should reset all statistics."""
        validator = ShadowValidator()
        validator.stats.total_validations = 100
        validator.stats.total_divergences = 5
        validator.divergence_log.append(MagicMock())

        validator.reset_stats()

        assert validator.stats.total_validations == 0
        assert validator.stats.total_divergences == 0
        assert len(validator.divergence_log) == 0

    def test_report_status_pass(self):
        """Report status should be PASS when below threshold."""
        validator = ShadowValidator(threshold=0.1)
        validator.stats.total_validations = 100
        validator.stats.total_divergences = 5  # 5% < 10%

        report = validator.get_report()
        assert report["status"] == "PASS"

    def test_report_status_fail(self):
        """Report status should be FAIL when above threshold."""
        validator = ShadowValidator(threshold=0.01)  # 1%
        validator.stats.total_validations = 100
        validator.stats.total_divergences = 5  # 5% > 1%

        report = validator.get_report()
        assert report["status"] == "FAIL"


class TestDefaultConfiguration:
    """Tests for default configuration values."""

    def test_default_sample_rate(self):
        """Default sample rate should be 5%."""
        assert DEFAULT_SAMPLE_RATE == 0.05

    def test_default_threshold(self):
        """Default divergence threshold should be 0.1%."""
        assert DEFAULT_DIVERGENCE_THRESHOLD == 0.001
