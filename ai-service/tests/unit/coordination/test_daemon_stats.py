"""Tests for daemon_stats.py base classes.

December 2025: Tests for consolidated daemon stats infrastructure.
"""

import time

import pytest

from app.coordination.daemon_stats import (
    CleanupDaemonStats,
    DaemonStatsBase,
    EvaluationDaemonStats,
    JobDaemonStats,
    SyncDaemonStats,
)


class TestDaemonStatsBase:
    """Tests for the base DaemonStatsBase class."""

    def test_initialization(self):
        """Test default initialization."""
        stats = DaemonStatsBase()
        assert stats.operations_attempted == 0
        assert stats.operations_completed == 0
        assert stats.operations_failed == 0
        assert stats.errors_count == 0
        assert stats.last_error is None
        assert stats.consecutive_failures == 0

    def test_record_attempt(self):
        """Test recording an attempt."""
        stats = DaemonStatsBase()
        stats.record_attempt()

        assert stats.operations_attempted == 1
        assert stats.last_check_time > 0

    def test_record_success(self):
        """Test recording a successful operation."""
        stats = DaemonStatsBase()
        stats.record_success(duration_seconds=1.5)

        assert stats.operations_completed == 1
        assert stats.operations_failed == 0
        assert stats.consecutive_failures == 0
        assert stats.last_check_time > 0
        assert stats.avg_operation_duration == 1.5

    def test_record_success_running_average(self):
        """Test that avg_operation_duration uses running average."""
        stats = DaemonStatsBase()
        stats.record_success(duration_seconds=10.0)
        stats.record_success(duration_seconds=2.0)

        # Running average: 0.9 * 10.0 + 0.1 * 2.0 = 9.2
        assert abs(stats.avg_operation_duration - 9.2) < 0.01

    def test_record_success_zero_duration_keeps_average(self):
        """Test that 0 duration doesn't update average."""
        stats = DaemonStatsBase()
        stats.record_success(duration_seconds=10.0)
        stats.record_success(duration_seconds=0.0)  # 0 duration is skipped

        # Average should remain at 10.0
        assert stats.avg_operation_duration == 10.0

    def test_record_failure(self):
        """Test recording a failed operation."""
        stats = DaemonStatsBase()
        stats.record_failure("Connection error")

        assert stats.operations_failed == 1
        assert stats.errors_count == 1
        assert stats.consecutive_failures == 1
        assert stats.last_error == "Connection error"

    def test_record_failure_with_exception(self):
        """Test recording failure with an exception."""
        stats = DaemonStatsBase()
        stats.record_failure(ValueError("Bad value"))

        assert stats.last_error == "Bad value"

    def test_consecutive_failures_reset_on_success(self):
        """Test that consecutive failures reset after success."""
        stats = DaemonStatsBase()
        stats.record_failure("Error 1")
        stats.record_failure("Error 2")
        assert stats.consecutive_failures == 2

        stats.record_success()
        assert stats.consecutive_failures == 0

    def test_critical_error_increments_critical_errors(self):
        """Test that critical=True increments critical_errors."""
        stats = DaemonStatsBase()
        stats.record_failure("Critical error", critical=True)

        assert stats.critical_errors == 1
        assert stats.errors_count == 1

    def test_is_healthy_default(self):
        """Test is_healthy with default thresholds."""
        stats = DaemonStatsBase()
        assert stats.is_healthy() is True

    def test_is_healthy_with_failures(self):
        """Test is_healthy detects excessive failures."""
        stats = DaemonStatsBase()
        # Use record_attempt to track attempts, then record_failure
        # to get proper error_rate calculation
        for _ in range(10):
            stats.record_attempt()
            stats.record_success()
        stats.record_attempt()
        stats.record_failure("Error 1")
        stats.record_attempt()
        stats.record_failure("Error 2")

        # 12 attempts, 2 failures = 16.7% error rate
        # Default max_error_rate is 0.1 (10%), we're above that
        assert stats.is_healthy(max_error_rate=0.1) is False
        assert stats.is_healthy(max_error_rate=0.2) is True

    def test_is_healthy_consecutive_failures(self):
        """Test is_healthy detects consecutive failures."""
        stats = DaemonStatsBase()
        # Add some successful operations first to keep error_rate reasonable
        for _ in range(50):
            stats.record_attempt()
            stats.record_success()
        # Then add consecutive failures
        for _ in range(5):
            stats.record_attempt()
            stats.record_failure("Error")

        # 5 consecutive failures >= max=5, so unhealthy
        assert stats.is_healthy(max_consecutive_failures=5, max_error_rate=0.2) is False
        # 5 consecutive failures < max=6, but we need to pass error_rate check too
        assert stats.is_healthy(max_consecutive_failures=6, max_error_rate=0.2) is True

    def test_to_dict(self):
        """Test serialization to dict."""
        stats = DaemonStatsBase()
        stats.record_attempt()
        stats.record_success(duration_seconds=1.0)
        stats.record_attempt()
        stats.record_failure("Test error")

        data = stats.to_dict()
        assert data["operations_attempted"] == 2
        assert data["operations_completed"] == 1
        assert data["operations_failed"] == 1
        assert data["last_error"] == "Test error"
        assert "avg_operation_duration" in data

    def test_success_rate(self):
        """Test success_rate property."""
        stats = DaemonStatsBase()
        assert stats.success_rate == 1.0  # No operations = 100% success

        stats.record_attempt()
        stats.record_success()
        stats.record_attempt()
        stats.record_failure("Error")

        # 1 success, 2 attempts
        assert stats.success_rate == 0.5

    def test_error_rate(self):
        """Test error_rate property."""
        stats = DaemonStatsBase()
        assert stats.error_rate == 0.0  # No operations = 0% error

        stats.record_attempt()
        stats.record_success()
        stats.record_attempt()
        stats.record_failure("Error")

        # 1 error, 2 attempts
        assert stats.error_rate == 0.5

    def test_reset_counters(self):
        """Test reset_counters clears all metrics."""
        stats = DaemonStatsBase()
        stats.record_attempt()
        stats.record_success(duration_seconds=5.0)
        stats.record_failure("Error")

        stats.reset_counters()

        assert stats.operations_attempted == 0
        assert stats.operations_completed == 0
        assert stats.operations_failed == 0
        assert stats.errors_count == 0
        assert stats.last_error is None
        assert stats.avg_operation_duration == 0.0


class TestSyncDaemonStats:
    """Tests for SyncDaemonStats."""

    def test_initialization(self):
        """Test default initialization."""
        stats = SyncDaemonStats()
        assert stats.syncs_completed == 0
        assert stats.syncs_failed == 0
        assert stats.bytes_synced == 0
        assert stats.files_synced == 0

    def test_record_sync_success(self):
        """Test recording a successful sync."""
        stats = SyncDaemonStats()
        stats.record_sync_success(duration=2.5, files=10, bytes_count=1024)

        assert stats.syncs_completed == 1
        assert stats.files_synced == 10
        assert stats.bytes_synced == 1024
        assert stats.operations_completed == 1
        assert stats.last_sync_duration == 2.5

    def test_record_sync_success_with_nodes(self):
        """Test recording a successful sync with node count."""
        stats = SyncDaemonStats()
        stats.record_sync_success(duration=2.5, nodes=3, files=10, bytes_count=1024)

        assert stats.nodes_synced == 3

    def test_record_sync_failure(self):
        """Test recording a failed sync."""
        stats = SyncDaemonStats()
        stats.record_sync_failure("Disk full")

        assert stats.syncs_failed == 1
        assert stats.operations_failed == 1
        assert stats.last_error == "Disk full"

    def test_to_dict_includes_sync_fields(self):
        """Test that to_dict includes sync-specific fields."""
        stats = SyncDaemonStats()
        stats.record_sync_success(duration=1.0, files=5, bytes_count=512)

        data = stats.to_dict()
        assert "syncs_completed" in data
        assert "files_synced" in data
        assert "bytes_synced" in data
        assert data["syncs_completed"] == 1


class TestCleanupDaemonStats:
    """Tests for CleanupDaemonStats."""

    def test_initialization(self):
        """Test default initialization."""
        stats = CleanupDaemonStats()
        assert stats.items_scanned == 0
        assert stats.items_cleaned == 0
        assert stats.items_quarantined == 0
        assert stats.bytes_reclaimed == 0

    def test_record_cleanup(self):
        """Test recording a cleanup operation."""
        stats = CleanupDaemonStats()
        stats.record_cleanup(scanned=100, cleaned=50, bytes_reclaimed=1024 * 1024)

        assert stats.items_scanned == 100
        assert stats.items_cleaned == 50
        assert stats.bytes_reclaimed == 1024 * 1024
        assert stats.total_items_processed == 50

    def test_record_cleanup_with_quarantine(self):
        """Test recording cleanup with quarantined items."""
        stats = CleanupDaemonStats()
        stats.record_cleanup(scanned=100, cleaned=45, quarantined=5)

        assert stats.items_quarantined == 5


class TestJobDaemonStats:
    """Tests for JobDaemonStats."""

    def test_initialization(self):
        """Test default initialization."""
        stats = JobDaemonStats()
        assert stats.jobs_processed == 0
        assert stats.jobs_succeeded == 0
        assert stats.jobs_failed == 0
        assert stats.jobs_timed_out == 0
        assert stats.jobs_reassigned == 0

    def test_record_job_success(self):
        """Test recording a successful job."""
        stats = JobDaemonStats()
        stats.record_job_success(duration=10.0)

        assert stats.jobs_processed == 1
        assert stats.jobs_succeeded == 1
        assert stats.operations_completed == 1
        assert stats.last_job_time > 0

    def test_record_job_failure(self):
        """Test recording a failed job."""
        stats = JobDaemonStats()
        stats.record_job_failure("Out of memory")

        assert stats.jobs_processed == 1
        assert stats.jobs_failed == 1
        assert stats.operations_failed == 1
        assert stats.last_error == "Out of memory"

    def test_record_job_timeout(self):
        """Test recording a job timeout."""
        stats = JobDaemonStats()
        stats.record_job_timeout()

        assert stats.jobs_timed_out == 1
        assert stats.jobs_processed == 1

    def test_record_job_reassigned(self):
        """Test recording a job reassignment."""
        stats = JobDaemonStats()
        stats.record_job_reassigned()

        assert stats.jobs_reassigned == 1


class TestEvaluationDaemonStats:
    """Tests for EvaluationDaemonStats."""

    def test_initialization(self):
        """Test default initialization."""
        stats = EvaluationDaemonStats()
        assert stats.evaluations_triggered == 0
        assert stats.evaluations_completed == 0
        assert stats.evaluations_failed == 0
        assert stats.games_played == 0
        assert stats.models_evaluated == 0
        assert stats.promotions_triggered == 0

    def test_record_evaluation_success(self):
        """Test recording a successful evaluation."""
        stats = EvaluationDaemonStats()
        stats.record_evaluation_success(duration=30.0, games=50, promoted=True)

        assert stats.evaluations_completed == 1
        assert stats.models_evaluated == 1
        assert stats.games_played == 50
        assert stats.promotions_triggered == 1
        assert stats.avg_evaluation_duration == 30.0

    def test_record_evaluation_success_without_promotion(self):
        """Test recording evaluation without promotion."""
        stats = EvaluationDaemonStats()
        stats.record_evaluation_success(duration=20.0, games=30, promoted=False)

        assert stats.evaluations_completed == 1
        assert stats.promotions_triggered == 0

    def test_record_evaluation_failure(self):
        """Test recording a failed evaluation."""
        stats = EvaluationDaemonStats()
        stats.record_evaluation_failure("Model not found")

        assert stats.evaluations_failed == 1
        assert stats.operations_failed == 1


class TestDaemonStatsIntegration:
    """Integration tests for daemon stats migration."""

    def test_sync_stats_from_auto_sync_daemon(self):
        """Test SyncStats from auto_sync_daemon works correctly."""
        from app.coordination.auto_sync_daemon import SyncStats

        stats = SyncStats()
        stats.record_sync_success(duration=1.5, files=10, bytes_count=1024)

        # Inherited methods work
        assert stats.syncs_completed == 1
        assert stats.bytes_synced == 1024

        # Backward-compat aliases
        assert stats.successful_syncs == 1

        # AutoSync-specific fields
        stats.games_synced = 50
        assert stats.games_synced == 50

    def test_evaluation_stats_from_evaluation_daemon(self):
        """Test EvaluationStats from evaluation_daemon works correctly."""
        from app.coordination.evaluation_daemon import EvaluationStats

        stats = EvaluationStats()
        stats.record_evaluation_success(duration=30.0, games=50, promoted=True)

        # Inherited methods work
        assert stats.evaluations_completed == 1
        assert stats.games_played == 50

        # Backward-compat aliases
        assert stats.total_games_played == 50
        assert stats.average_evaluation_time == 30.0

    def test_stats_are_dataclass_compatible(self):
        """Test that all stats classes work as dataclasses."""
        from dataclasses import asdict

        stats = DaemonStatsBase()
        stats.record_success(duration_seconds=1.0)

        # Should be convertible to dict via dataclass
        data = asdict(stats)
        assert "operations_completed" in data
        assert data["operations_completed"] == 1
