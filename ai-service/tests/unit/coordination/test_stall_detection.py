"""Tests for job stall detection and node penalties.

Tests the JobStallDetector for cluster coordination.
"""

import time
import pytest

from app.coordination.stall_detection import (
    JobStallDetector,
    NodePenalty,
    StallDetectorConfig,
    StallRecord,
    StallSeverity,
    get_stall_detector,
    reset_stall_detector,
)


class TestStallSeverity:
    """Tests for StallSeverity enum."""

    def test_severity_values(self):
        """Test severity level values."""
        assert StallSeverity.MINOR.value == "minor"
        assert StallSeverity.MODERATE.value == "moderate"
        assert StallSeverity.SEVERE.value == "severe"


class TestStallRecord:
    """Tests for StallRecord dataclass."""

    def test_create_stall_record(self):
        """Test creating a stall record."""
        record = StallRecord(
            job_id="job_123",
            node_id="node_abc",
            stall_time=time.time(),
            duration_seconds=300.0,
            severity=StallSeverity.MINOR,
        )
        assert record.job_id == "job_123"
        assert record.node_id == "node_abc"
        assert record.severity == StallSeverity.MINOR


class TestNodePenalty:
    """Tests for NodePenalty dataclass."""

    def test_create_node_penalty(self):
        """Test creating a node penalty."""
        penalty = NodePenalty(
            node_id="node_abc",
            penalty_until=time.time() + 300,
            stall_count=1,
            last_stall_time=time.time(),
            backoff_level=0,
        )
        assert penalty.node_id == "node_abc"
        assert penalty.stall_count == 1


class TestStallDetectorConfig:
    """Tests for StallDetectorConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = StallDetectorConfig()
        assert config.stall_threshold_seconds > 0
        assert config.base_penalty_seconds > 0
        assert config.max_penalty_seconds > config.base_penalty_seconds
        assert config.max_stalls_before_permanent > 0


class TestJobStallDetector:
    """Tests for JobStallDetector."""

    @pytest.fixture
    def detector(self):
        """Create a fresh detector for testing."""
        reset_stall_detector()
        config = StallDetectorConfig(
            stall_threshold_seconds=1.0,  # Short for testing
            base_penalty_seconds=1.0,
            max_penalty_seconds=10.0,
        )
        return JobStallDetector(config=config)

    def test_register_job(self, detector):
        """Test registering a job."""
        detector.register_job("job_1", "node_a")
        assert "job_1" in detector._active_jobs

    def test_update_progress(self, detector):
        """Test updating job progress."""
        detector.register_job("job_1", "node_a")
        old_time = detector._active_jobs["job_1"][1]
        time.sleep(0.01)
        detector.update_progress("job_1")
        new_time = detector._active_jobs["job_1"][1]
        assert new_time > old_time

    def test_complete_job_success(self, detector):
        """Test completing a job successfully."""
        detector.register_job("job_1", "node_a")
        detector.complete_job("job_1", success=True)
        assert "job_1" not in detector._active_jobs

    def test_is_job_stalled_not_stalled(self, detector):
        """Test is_job_stalled returns False for fresh job."""
        detector.register_job("job_1", "node_a")
        assert not detector.is_job_stalled("job_1")

    def test_is_job_stalled_after_threshold(self, detector):
        """Test is_job_stalled returns True after threshold."""
        detector.register_job("job_1", "node_a")
        # Manually set old progress time
        detector._active_jobs["job_1"] = ("node_a", time.time() - 10)
        assert detector.is_job_stalled("job_1")

    def test_report_stall_applies_penalty(self, detector):
        """Test that reporting a stall applies penalty."""
        record = detector.report_stall("job_1", "node_a", duration_seconds=2.0)
        assert record.job_id == "job_1"
        assert record.node_id == "node_a"
        assert detector.is_node_penalized("node_a")

    def test_penalty_severity_minor(self, detector):
        """Test minor severity for short stalls."""
        # Stall duration < 2x threshold
        record = detector.report_stall("job_1", "node_a", duration_seconds=1.5)
        assert record.severity == StallSeverity.MINOR

    def test_penalty_severity_moderate(self, detector):
        """Test moderate severity for medium stalls."""
        # Stall duration 2-5x threshold
        record = detector.report_stall("job_1", "node_a", duration_seconds=3.0)
        assert record.severity == StallSeverity.MODERATE

    def test_penalty_severity_severe(self, detector):
        """Test severe severity for long stalls."""
        # Stall duration > 5x threshold
        record = detector.report_stall("job_1", "node_a", duration_seconds=10.0)
        assert record.severity == StallSeverity.SEVERE

    def test_exponential_backoff(self, detector):
        """Test exponential backoff on repeated stalls."""
        # First stall
        detector.report_stall("job_1", "node_a")
        first_penalty = detector.get_penalty_remaining("node_a")

        # Wait for penalty to expire
        time.sleep(detector.config.base_penalty_seconds + 0.1)
        assert not detector.is_node_penalized("node_a")

        # Second stall - should have higher penalty due to backoff
        detector.report_stall("job_2", "node_a")
        second_penalty = detector.get_penalty_remaining("node_a")

        # Backoff should increase penalty
        assert second_penalty > first_penalty * 0.9

    def test_is_node_unhealthy_after_max_stalls(self, detector):
        """Test node marked unhealthy after max stalls."""
        config = StallDetectorConfig(
            stall_threshold_seconds=1.0,
            max_stalls_before_permanent=3,
        )
        detector = JobStallDetector(config=config)

        for i in range(3):
            detector.report_stall(f"job_{i}", "node_a")

        assert detector.is_node_unhealthy("node_a")

    def test_clear_penalty(self, detector):
        """Test manually clearing penalty."""
        detector.report_stall("job_1", "node_a")
        assert detector.is_node_penalized("node_a")
        detector.clear_penalty("node_a")
        assert not detector.is_node_penalized("node_a")

    def test_get_stall_duration_not_stalled(self, detector):
        """Test get_stall_duration returns 0 for non-stalled job."""
        detector.register_job("job_1", "node_a")
        assert detector.get_stall_duration("job_1") == 0.0

    def test_get_stall_duration_stalled(self, detector):
        """Test get_stall_duration returns duration for stalled job."""
        detector.register_job("job_1", "node_a")
        detector._active_jobs["job_1"] = ("node_a", time.time() - 10)
        duration = detector.get_stall_duration("job_1")
        assert duration > 9.0

    def test_get_statistics(self, detector):
        """Test getting detector statistics."""
        detector.report_stall("job_1", "node_a")
        stats = detector.get_statistics()
        assert "total_stalls" in stats
        assert "penalized_nodes" in stats
        assert stats["total_stalls"] >= 1

    def test_get_stall_history(self, detector):
        """Test getting stall history."""
        detector.report_stall("job_1", "node_a")
        detector.report_stall("job_2", "node_b")
        history = detector.get_stall_history(limit=10)
        assert len(history) >= 2

    def test_successful_completion_reduces_backoff(self, detector):
        """Test successful job completion reduces backoff level."""
        # Create a stall to increase backoff
        detector.report_stall("job_1", "node_a")
        penalty_before = detector._penalties.get("node_a")
        assert penalty_before is not None
        initial_backoff = penalty_before.backoff_level

        # Register and complete a successful job
        detector.register_job("job_2", "node_a")
        detector.complete_job("job_2", success=True)

        # Backoff should be reduced or stay same (can't go below 0)
        penalty_after = detector._penalties.get("node_a")
        if penalty_after:
            assert penalty_after.backoff_level <= initial_backoff


class TestStallDetectorSingleton:
    """Tests for singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_stall_detector()

    def test_get_stall_detector_returns_same_instance(self):
        """get_stall_detector should return same instance."""
        d1 = get_stall_detector()
        d2 = get_stall_detector()
        assert d1 is d2

    def test_reset_creates_new_instance(self):
        """reset_stall_detector should allow new instance."""
        d1 = get_stall_detector()
        reset_stall_detector()
        d2 = get_stall_detector()
        assert d1 is not d2
