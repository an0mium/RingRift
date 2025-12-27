"""Tests for PerNodeSyncStats dataclass.

December 2025: Added as part of per-node reliability tracking implementation.
"""

import pytest
import time

from app.coordination.daemon_stats import PerNodeSyncStats


class TestPerNodeSyncStats:
    """Tests for PerNodeSyncStats dataclass."""

    def test_initial_state(self):
        """Test initial state of a new PerNodeSyncStats instance."""
        stats = PerNodeSyncStats(node_id="test-node")
        assert stats.node_id == "test-node"
        assert stats.syncs_attempted == 0
        assert stats.syncs_successful == 0
        assert stats.syncs_failed == 0
        assert stats.consecutive_failures == 0
        assert stats.avg_sync_duration == 0.0

    def test_initial_success_rate_is_one(self):
        """With no attempts, success rate should be 1.0 (optimistic default)."""
        stats = PerNodeSyncStats(node_id="test-node")
        assert stats.success_rate == 1.0

    def test_record_success_updates_stats(self):
        """Test that record_success properly updates all relevant fields."""
        stats = PerNodeSyncStats(node_id="test-node")
        before = time.time()
        stats.record_success(duration=1.5)
        after = time.time()

        assert stats.syncs_attempted == 1
        assert stats.syncs_successful == 1
        assert stats.syncs_failed == 0
        assert stats.consecutive_failures == 0
        assert stats.avg_sync_duration == 1.5
        assert before <= stats.last_sync_success_time <= after

    def test_record_failure_updates_stats(self):
        """Test that record_failure properly updates all relevant fields."""
        stats = PerNodeSyncStats(node_id="test-node")
        before = time.time()
        stats.record_failure(reason="connection_timeout")
        after = time.time()

        assert stats.syncs_attempted == 1
        assert stats.syncs_successful == 0
        assert stats.syncs_failed == 1
        assert stats.consecutive_failures == 1
        assert stats.last_failure_reason == "connection_timeout"
        assert before <= stats.last_sync_failure_time <= after

    def test_success_rate_calculation(self):
        """Test success rate calculation with mixed results."""
        stats = PerNodeSyncStats(node_id="test-node")
        stats.record_success(1.0)
        stats.record_success(1.0)
        stats.record_failure("test")

        assert stats.syncs_attempted == 3
        assert stats.syncs_successful == 2
        assert stats.syncs_failed == 1
        assert stats.success_rate == pytest.approx(0.666, rel=0.01)

    def test_consecutive_failures_increments(self):
        """Test that consecutive failures increment properly."""
        stats = PerNodeSyncStats(node_id="test-node")
        stats.record_failure("err1")
        assert stats.consecutive_failures == 1
        stats.record_failure("err2")
        assert stats.consecutive_failures == 2
        stats.record_failure("err3")
        assert stats.consecutive_failures == 3

    def test_consecutive_failures_reset_on_success(self):
        """Test that consecutive failures reset to 0 after success."""
        stats = PerNodeSyncStats(node_id="test-node")
        stats.record_failure("err1")
        stats.record_failure("err2")
        assert stats.consecutive_failures == 2

        stats.record_success(1.0)
        assert stats.consecutive_failures == 0

        # And then increment again on next failure
        stats.record_failure("err3")
        assert stats.consecutive_failures == 1

    def test_avg_duration_first_record(self):
        """Test that first duration sets avg directly (no EMA)."""
        stats = PerNodeSyncStats(node_id="test-node")
        stats.record_success(10.0)
        assert stats.avg_sync_duration == 10.0

    def test_avg_duration_ema_update(self):
        """Test EMA update for average duration (90% old, 10% new)."""
        stats = PerNodeSyncStats(node_id="test-node")
        stats.record_success(10.0)  # First: set directly
        assert stats.avg_sync_duration == 10.0

        stats.record_success(20.0)  # EMA: 0.9*10 + 0.1*20 = 11
        assert stats.avg_sync_duration == pytest.approx(11.0, rel=0.01)

        stats.record_success(20.0)  # EMA: 0.9*11 + 0.1*20 = 11.9
        assert stats.avg_sync_duration == pytest.approx(11.9, rel=0.01)

    def test_to_dict_serialization(self):
        """Test that to_dict returns expected structure."""
        stats = PerNodeSyncStats(node_id="test-node")
        stats.record_success(1.5)
        stats.record_failure("test error")

        d = stats.to_dict()

        assert d["node_id"] == "test-node"
        assert d["syncs_attempted"] == 2
        assert d["syncs_successful"] == 1
        assert d["syncs_failed"] == 1
        assert d["success_rate"] == 0.5
        assert d["avg_sync_duration"] == 1.5
        assert d["consecutive_failures"] == 1
        assert d["last_failure_reason"] == "test error"
        assert "last_sync_success_time" in d
        assert "last_sync_failure_time" in d

    def test_to_dict_empty_stats(self):
        """Test to_dict with no operations recorded."""
        stats = PerNodeSyncStats(node_id="empty-node")
        d = stats.to_dict()

        assert d["node_id"] == "empty-node"
        assert d["syncs_attempted"] == 0
        assert d["success_rate"] == 1.0  # Optimistic default

    def test_failure_reason_updated_on_each_failure(self):
        """Test that last_failure_reason is updated on each failure."""
        stats = PerNodeSyncStats(node_id="test-node")
        stats.record_failure("first_error")
        assert stats.last_failure_reason == "first_error"

        stats.record_failure("second_error")
        assert stats.last_failure_reason == "second_error"

    def test_failure_reason_empty_string_allowed(self):
        """Test that empty failure reason is allowed."""
        stats = PerNodeSyncStats(node_id="test-node")
        stats.record_failure()  # No reason provided
        assert stats.last_failure_reason == ""
        assert stats.syncs_failed == 1

    def test_high_volume_tracking(self):
        """Test stats remain accurate with many operations."""
        stats = PerNodeSyncStats(node_id="busy-node")

        # 80 successes, 20 failures
        for _ in range(80):
            stats.record_success(1.0)
        for _ in range(20):
            stats.record_failure("err")

        assert stats.syncs_attempted == 100
        assert stats.syncs_successful == 80
        assert stats.syncs_failed == 20
        assert stats.success_rate == 0.8
        assert stats.consecutive_failures == 20  # Last 20 were failures
