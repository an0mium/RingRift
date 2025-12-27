"""Tests for SyncStallHandler - automatic failover from stalled syncs.

Tests cover:
1. Stall detection (within timeout, exceeded timeout)
2. Stall recording and penalty tracking
3. Alternative source selection (with exclusions)
4. Recovery recording
5. Statistics tracking
6. Edge cases (no alternatives, all hosts stalled)
7. Penalty expiry
8. Source availability checks
9. Module-level singleton management
"""

import time
from unittest.mock import patch

import pytest

from app.coordination.sync_stall_handler import (
    SyncStallHandler,
    get_stall_handler,
    reset_stall_handler,
)


class TestSyncStallDetection:
    """Test stall detection logic."""

    def test_no_stall_within_timeout(self):
        """Should return False when sync is within timeout."""
        handler = SyncStallHandler()
        sync_id = "sync-123"
        started_at = time.time()
        timeout = 600.0

        assert not handler.check_stall(sync_id, started_at, timeout)

    def test_stall_detected_when_exceeded_timeout(self):
        """Should return True when sync exceeds timeout."""
        handler = SyncStallHandler()
        sync_id = "sync-123"
        started_at = time.time() - 700.0  # Started 700 seconds ago
        timeout = 600.0

        assert handler.check_stall(sync_id, started_at, timeout)

    def test_stall_detected_at_boundary(self):
        """Should detect stall exactly at timeout boundary."""
        handler = SyncStallHandler()
        sync_id = "sync-123"
        started_at = time.time() - 600.1  # Just over timeout
        timeout = 600.0

        assert handler.check_stall(sync_id, started_at, timeout)

    def test_no_stall_just_before_timeout(self):
        """Should not detect stall just before timeout."""
        handler = SyncStallHandler()
        sync_id = "sync-123"
        started_at = time.time() - 599.9  # Just under timeout
        timeout = 600.0

        assert not handler.check_stall(sync_id, started_at, timeout)

    def test_stall_detection_with_different_timeouts(self):
        """Should respect different timeout values."""
        handler = SyncStallHandler()
        sync_id = "sync-123"
        started_at = time.time() - 100.0

        # Should not stall with longer timeout
        assert not handler.check_stall(sync_id, started_at, timeout=200.0)

        # Should stall with shorter timeout
        assert handler.check_stall(sync_id, started_at, timeout=50.0)


class TestStallRecording:
    """Test stall recording and penalty tracking."""

    def test_record_stall_increments_count(self):
        """Should increment stall count when recording stall."""
        handler = SyncStallHandler()
        assert handler._stall_count == 0

        handler.record_stall("node-5", "sync-123")
        assert handler._stall_count == 1

        handler.record_stall("node-6", "sync-456")
        assert handler._stall_count == 2

    def test_record_stall_applies_penalty(self):
        """Should apply penalty to stalled source."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-5", "sync-123")

        assert "node-5" in handler._stalled_sources
        penalty_until = handler._stalled_sources["node-5"]
        expected_penalty = time.time() + 300.0
        assert abs(penalty_until - expected_penalty) < 1.0  # Within 1 second

    def test_record_stall_updates_existing_penalty(self):
        """Should update penalty for already-penalized source."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)

        # First stall
        handler.record_stall("node-5", "sync-123")
        first_penalty = handler._stalled_sources["node-5"]

        time.sleep(0.1)

        # Second stall - should update penalty
        handler.record_stall("node-5", "sync-456")
        second_penalty = handler._stalled_sources["node-5"]

        assert second_penalty > first_penalty

    def test_record_stall_with_empty_host(self):
        """Should handle empty host gracefully."""
        handler = SyncStallHandler()
        handler.record_stall("", "sync-123")

        # Should not record penalty
        assert "" not in handler._stalled_sources
        # Should not increment count
        assert handler._stall_count == 0

    def test_record_stall_multiple_hosts(self):
        """Should track penalties for multiple hosts independently."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)

        handler.record_stall("node-5", "sync-123")
        handler.record_stall("node-6", "sync-456")
        handler.record_stall("node-7", "sync-789")

        assert len(handler._stalled_sources) == 3
        assert "node-5" in handler._stalled_sources
        assert "node-6" in handler._stalled_sources
        assert "node-7" in handler._stalled_sources

    def test_record_stall_custom_penalty_duration(self):
        """Should respect custom penalty duration."""
        handler = SyncStallHandler(stall_penalty_seconds=600.0)
        handler.record_stall("node-5", "sync-123")

        penalty_until = handler._stalled_sources["node-5"]
        expected_penalty = time.time() + 600.0
        assert abs(penalty_until - expected_penalty) < 1.0


class TestAlternativeSourceSelection:
    """Test alternative source selection logic."""

    def test_get_alternative_source_without_exclusions(self):
        """Should return first available source when no exclusions."""
        handler = SyncStallHandler()
        all_sources = ["node-1", "node-2", "node-3"]

        alt = handler.get_alternative_source(exclude=None, all_sources=all_sources)
        assert alt == "node-1"

    def test_get_alternative_source_with_exclusions(self):
        """Should skip excluded sources."""
        handler = SyncStallHandler()
        all_sources = ["node-1", "node-2", "node-3"]

        alt = handler.get_alternative_source(
            exclude=["node-1"], all_sources=all_sources
        )
        assert alt == "node-2"

    def test_get_alternative_source_with_penalized_sources(self):
        """Should skip penalized sources."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-1", "sync-123")

        all_sources = ["node-1", "node-2", "node-3"]
        alt = handler.get_alternative_source(exclude=None, all_sources=all_sources)

        # Should skip node-1 (penalized) and return node-2
        assert alt == "node-2"

    def test_get_alternative_source_with_exclusions_and_penalties(self):
        """Should skip both excluded and penalized sources."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-1", "sync-123")

        all_sources = ["node-1", "node-2", "node-3"]
        alt = handler.get_alternative_source(
            exclude=["node-2"], all_sources=all_sources
        )

        # Should skip node-1 (penalized) and node-2 (excluded)
        assert alt == "node-3"

    def test_get_alternative_source_no_pool_provided(self):
        """Should return None when no source pool provided."""
        handler = SyncStallHandler()
        alt = handler.get_alternative_source(exclude=["node-1"], all_sources=None)
        assert alt is None

    def test_get_alternative_source_all_excluded(self):
        """Should return None when all sources excluded."""
        handler = SyncStallHandler()
        all_sources = ["node-1", "node-2"]

        alt = handler.get_alternative_source(
            exclude=["node-1", "node-2"], all_sources=all_sources
        )
        assert alt is None

    def test_get_alternative_source_all_penalized(self):
        """Should return None when all sources penalized."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-1", "sync-123")
        handler.record_stall("node-2", "sync-456")

        all_sources = ["node-1", "node-2"]
        alt = handler.get_alternative_source(exclude=None, all_sources=all_sources)
        assert alt is None

    def test_get_alternative_source_all_excluded_and_penalized(self):
        """Should return None when sources are excluded and penalized."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-1", "sync-123")

        all_sources = ["node-1", "node-2"]
        alt = handler.get_alternative_source(
            exclude=["node-2"], all_sources=all_sources
        )
        assert alt is None

    def test_get_alternative_source_expired_penalty(self):
        """Should include sources with expired penalties."""
        handler = SyncStallHandler(stall_penalty_seconds=0.1)
        handler.record_stall("node-1", "sync-123")

        # Wait for penalty to expire
        time.sleep(0.2)

        all_sources = ["node-1", "node-2"]
        alt = handler.get_alternative_source(exclude=None, all_sources=all_sources)

        # node-1 penalty expired, should be available
        assert alt == "node-1"

    def test_get_alternative_source_empty_pool(self):
        """Should return None for empty source pool."""
        handler = SyncStallHandler()
        alt = handler.get_alternative_source(exclude=None, all_sources=[])
        assert alt is None


class TestRecoveryRecording:
    """Test recovery recording."""

    def test_record_recovery_increments_count(self):
        """Should increment recovery count."""
        handler = SyncStallHandler()
        assert handler._recovery_count == 0

        handler.record_recovery("sync-123", "node-6")
        assert handler._recovery_count == 1

        handler.record_recovery("sync-456", "node-7")
        assert handler._recovery_count == 2

    def test_record_recovery_logs_new_source(self):
        """Should record which source was used for recovery."""
        handler = SyncStallHandler()
        handler.record_recovery("sync-123", "node-6")
        assert handler._recovery_count == 1

    def test_record_failed_recovery_increments_count(self):
        """Should increment failed recovery count."""
        handler = SyncStallHandler()
        assert handler._failed_recoveries == 0

        handler.record_failed_recovery("sync-123", "No alternatives available")
        assert handler._failed_recoveries == 1

        handler.record_failed_recovery("sync-456", "All hosts stalled")
        assert handler._failed_recoveries == 2

    def test_record_failed_recovery_with_reason(self):
        """Should record failure reason."""
        handler = SyncStallHandler()
        handler.record_failed_recovery("sync-123", "No alternatives available")
        assert handler._failed_recoveries == 1


class TestStatisticsTracking:
    """Test statistics retrieval."""

    def test_get_stats_initial_state(self):
        """Should return zero stats for new handler."""
        handler = SyncStallHandler()
        stats = handler.get_stats()

        assert stats["stall_count"] == 0
        assert stats["recovery_count"] == 0
        assert stats["failed_recoveries"] == 0
        assert stats["active_penalties"] == 0
        assert stats["penalized_sources"] == []
        assert stats["penalty_details"] == {}

    def test_get_stats_after_stalls(self):
        """Should reflect stall counts in stats."""
        handler = SyncStallHandler()
        handler.record_stall("node-5", "sync-123")
        handler.record_stall("node-6", "sync-456")

        stats = handler.get_stats()
        assert stats["stall_count"] == 2
        assert stats["active_penalties"] == 2
        assert "node-5" in stats["penalized_sources"]
        assert "node-6" in stats["penalized_sources"]

    def test_get_stats_after_recoveries(self):
        """Should reflect recovery counts in stats."""
        handler = SyncStallHandler()
        handler.record_recovery("sync-123", "node-6")
        handler.record_recovery("sync-456", "node-7")

        stats = handler.get_stats()
        assert stats["recovery_count"] == 2

    def test_get_stats_after_failed_recoveries(self):
        """Should reflect failed recovery counts in stats."""
        handler = SyncStallHandler()
        handler.record_failed_recovery("sync-123", "No alternatives")
        handler.record_failed_recovery("sync-456", "All stalled")

        stats = handler.get_stats()
        assert stats["failed_recoveries"] == 2

    def test_get_stats_penalty_details(self):
        """Should include penalty details with remaining time."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-5", "sync-123")

        stats = handler.get_stats()
        assert "node-5" in stats["penalty_details"]

        # Remaining time should be close to 300 seconds
        remaining = stats["penalty_details"]["node-5"]
        assert "299." in remaining or "300." in remaining

    def test_get_stats_excludes_expired_penalties(self):
        """Should not include expired penalties in stats."""
        handler = SyncStallHandler(stall_penalty_seconds=0.1)
        handler.record_stall("node-5", "sync-123")

        # Wait for penalty to expire
        time.sleep(0.2)

        stats = handler.get_stats()
        assert stats["active_penalties"] == 0
        assert "node-5" not in stats["penalized_sources"]

    def test_get_stats_config(self):
        """Should include handler configuration in stats."""
        handler = SyncStallHandler(stall_penalty_seconds=600.0, max_retries=5)
        stats = handler.get_stats()

        assert stats["config"]["stall_penalty_seconds"] == 600.0
        assert stats["config"]["max_retries"] == 5

    def test_get_stats_mixed_scenario(self):
        """Should handle mixed scenario with stalls and recoveries."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)

        # Record some stalls
        handler.record_stall("node-5", "sync-123")
        handler.record_stall("node-6", "sync-456")

        # Record some recoveries
        handler.record_recovery("sync-123", "node-7")
        handler.record_failed_recovery("sync-789", "No alternatives")

        stats = handler.get_stats()
        assert stats["stall_count"] == 2
        assert stats["recovery_count"] == 1
        assert stats["failed_recoveries"] == 1
        assert stats["active_penalties"] == 2


class TestSourceAvailability:
    """Test source availability checks."""

    def test_is_source_available_for_new_source(self):
        """Should return True for source without penalty."""
        handler = SyncStallHandler()
        assert handler.is_source_available("node-5")

    def test_is_source_available_for_penalized_source(self):
        """Should return False for penalized source."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-5", "sync-123")

        assert not handler.is_source_available("node-5")

    def test_is_source_available_after_penalty_expires(self):
        """Should return True after penalty expires."""
        handler = SyncStallHandler(stall_penalty_seconds=0.1)
        handler.record_stall("node-5", "sync-123")

        # Initially unavailable
        assert not handler.is_source_available("node-5")

        # Wait for penalty to expire
        time.sleep(0.2)

        # Now available
        assert handler.is_source_available("node-5")

    def test_is_source_available_multiple_sources(self):
        """Should track availability independently per source."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-5", "sync-123")

        assert not handler.is_source_available("node-5")
        assert handler.is_source_available("node-6")
        assert handler.is_source_available("node-7")


class TestPenaltyCleaning:
    """Test penalty expiry and cleanup."""

    def test_clear_penalties_removes_expired(self):
        """Should remove expired penalties."""
        handler = SyncStallHandler(stall_penalty_seconds=0.1)
        handler.record_stall("node-5", "sync-123")
        handler.record_stall("node-6", "sync-456")

        # Wait for penalties to expire
        time.sleep(0.2)

        cleared = handler.clear_penalties()
        assert cleared == 2
        assert len(handler._stalled_sources) == 0

    def test_clear_penalties_keeps_active(self):
        """Should keep active penalties."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        handler.record_stall("node-5", "sync-123")
        handler.record_stall("node-6", "sync-456")

        cleared = handler.clear_penalties()
        assert cleared == 0
        assert len(handler._stalled_sources) == 2

    def test_clear_penalties_mixed_expiry(self):
        """Should clear only expired penalties, keep active ones."""
        handler = SyncStallHandler(stall_penalty_seconds=0.2)
        handler.record_stall("node-5", "sync-123")

        time.sleep(0.1)

        handler.record_stall("node-6", "sync-456")

        # Wait for first penalty to expire but not second
        time.sleep(0.15)

        cleared = handler.clear_penalties()
        assert cleared == 1
        assert "node-5" not in handler._stalled_sources
        assert "node-6" in handler._stalled_sources

    def test_clear_penalties_when_empty(self):
        """Should handle empty penalty list gracefully."""
        handler = SyncStallHandler()
        cleared = handler.clear_penalties()
        assert cleared == 0


class TestHandlerReset:
    """Test handler state reset."""

    def test_reset_clears_all_state(self):
        """Should clear all handler state."""
        handler = SyncStallHandler()

        # Set up some state
        handler.record_stall("node-5", "sync-123")
        handler.record_recovery("sync-123", "node-6")
        handler.record_failed_recovery("sync-456", "No alternatives")

        # Reset
        handler.reset()

        # Verify all state cleared
        assert handler._stall_count == 0
        assert handler._recovery_count == 0
        assert handler._failed_recoveries == 0
        assert len(handler._stalled_sources) == 0

        stats = handler.get_stats()
        assert stats["stall_count"] == 0
        assert stats["recovery_count"] == 0
        assert stats["failed_recoveries"] == 0
        assert stats["active_penalties"] == 0

    def test_reset_preserves_config(self):
        """Should preserve handler configuration after reset."""
        handler = SyncStallHandler(stall_penalty_seconds=600.0, max_retries=5)
        handler.reset()

        assert handler.stall_penalty_seconds == 600.0
        assert handler.max_retries == 5

    def test_reset_multiple_times(self):
        """Should handle multiple resets."""
        handler = SyncStallHandler()

        for i in range(3):
            handler.record_stall(f"node-{i}", f"sync-{i}")
            handler.reset()
            assert handler._stall_count == 0


class TestSingletonManagement:
    """Test module-level singleton management."""

    def test_get_stall_handler_creates_instance(self):
        """Should create singleton instance on first call."""
        reset_stall_handler()  # Ensure clean state

        handler = get_stall_handler()
        assert handler is not None
        assert isinstance(handler, SyncStallHandler)

    def test_get_stall_handler_returns_same_instance(self):
        """Should return same instance on subsequent calls."""
        reset_stall_handler()

        handler1 = get_stall_handler()
        handler2 = get_stall_handler()

        assert handler1 is handler2

    def test_get_stall_handler_with_custom_params(self):
        """Should initialize singleton with custom parameters."""
        reset_stall_handler()

        handler = get_stall_handler(stall_penalty_seconds=600.0, max_retries=5)
        assert handler.stall_penalty_seconds == 600.0
        assert handler.max_retries == 5

    def test_get_stall_handler_ignores_params_after_init(self):
        """Should ignore parameters after singleton is created."""
        reset_stall_handler()

        handler1 = get_stall_handler(stall_penalty_seconds=300.0, max_retries=3)
        handler2 = get_stall_handler(stall_penalty_seconds=600.0, max_retries=5)

        # Second call parameters ignored
        assert handler2.stall_penalty_seconds == 300.0
        assert handler2.max_retries == 3

    def test_reset_stall_handler_resets_state(self):
        """Should reset singleton state."""
        reset_stall_handler()

        handler = get_stall_handler()
        handler.record_stall("node-5", "sync-123")

        reset_stall_handler()

        # After reset, new instance should have clean state
        handler2 = get_stall_handler()
        assert handler2._stall_count == 0

    def test_reset_stall_handler_allows_new_params(self):
        """Should allow new parameters after reset."""
        reset_stall_handler()

        handler1 = get_stall_handler(stall_penalty_seconds=300.0, max_retries=3)
        assert handler1.stall_penalty_seconds == 300.0

        reset_stall_handler()

        handler2 = get_stall_handler(stall_penalty_seconds=600.0, max_retries=5)
        assert handler2.stall_penalty_seconds == 600.0
        assert handler2.max_retries == 5

    def test_reset_stall_handler_when_not_initialized(self):
        """Should handle reset when singleton not yet created."""
        reset_stall_handler()
        # Should not raise error
        reset_stall_handler()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_penalty_duration(self):
        """Should handle zero penalty duration."""
        handler = SyncStallHandler(stall_penalty_seconds=0.0)
        handler.record_stall("node-5", "sync-123")

        # Penalty should expire immediately
        assert handler.is_source_available("node-5")

    def test_negative_penalty_duration(self):
        """Should handle negative penalty duration (expires immediately)."""
        handler = SyncStallHandler(stall_penalty_seconds=-1.0)
        handler.record_stall("node-5", "sync-123")

        # Penalty should be expired
        assert handler.is_source_available("node-5")

    def test_very_long_penalty_duration(self):
        """Should handle very long penalty durations."""
        handler = SyncStallHandler(stall_penalty_seconds=86400.0)  # 24 hours
        handler.record_stall("node-5", "sync-123")

        assert not handler.is_source_available("node-5")

    def test_zero_timeout(self):
        """Should handle zero timeout."""
        handler = SyncStallHandler()
        started_at = time.time()

        # Even just-started sync should stall with zero timeout
        assert handler.check_stall("sync-123", started_at, timeout=0.0)

    def test_negative_timeout(self):
        """Should handle negative timeout."""
        handler = SyncStallHandler()
        started_at = time.time()

        # Negative timeout should always stall
        assert handler.check_stall("sync-123", started_at, timeout=-1.0)

    def test_large_number_of_penalties(self):
        """Should handle large number of penalties."""
        handler = SyncStallHandler()

        # Record many penalties
        for i in range(100):
            handler.record_stall(f"node-{i}", f"sync-{i}")

        assert len(handler._stalled_sources) == 100
        assert handler._stall_count == 100

    def test_alternative_source_with_duplicate_exclusions(self):
        """Should handle duplicate entries in exclusion list."""
        handler = SyncStallHandler()
        all_sources = ["node-1", "node-2", "node-3"]

        alt = handler.get_alternative_source(
            exclude=["node-1", "node-1", "node-1"], all_sources=all_sources
        )
        assert alt == "node-2"

    def test_alternative_source_with_invalid_exclusions(self):
        """Should handle exclusions not in source pool."""
        handler = SyncStallHandler()
        all_sources = ["node-1", "node-2", "node-3"]

        alt = handler.get_alternative_source(
            exclude=["node-99", "node-100"], all_sources=all_sources
        )
        assert alt == "node-1"

    def test_stall_check_with_future_start_time(self):
        """Should handle start time in the future."""
        handler = SyncStallHandler()
        started_at = time.time() + 100.0  # Future time
        timeout = 600.0

        # Negative elapsed time should not stall
        assert not handler.check_stall("sync-123", started_at, timeout)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_stall_and_recovery_workflow(self):
        """Should handle complete stall and recovery workflow."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0, max_retries=3)
        all_sources = ["node-1", "node-2", "node-3"]

        # 1. Detect stall
        sync_id = "sync-123"
        started_at = time.time() - 700.0
        assert handler.check_stall(sync_id, started_at, timeout=600.0)

        # 2. Record stall
        handler.record_stall("node-1", sync_id)
        assert handler._stall_count == 1

        # 3. Get alternative
        alt = handler.get_alternative_source(
            exclude=["node-1"], all_sources=all_sources
        )
        assert alt == "node-2"

        # 4. Record recovery
        handler.record_recovery(sync_id, alt)
        assert handler._recovery_count == 1

        # 5. Verify stats
        stats = handler.get_stats()
        assert stats["stall_count"] == 1
        assert stats["recovery_count"] == 1
        assert stats["active_penalties"] == 1

    def test_cascading_failures(self):
        """Should handle cascading failures across multiple nodes."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)
        all_sources = ["node-1", "node-2", "node-3"]

        # First stall on node-1
        handler.record_stall("node-1", "sync-123")
        alt1 = handler.get_alternative_source(
            exclude=["node-1"], all_sources=all_sources
        )
        assert alt1 == "node-2"

        # Second stall on node-2
        handler.record_stall("node-2", "sync-456")
        alt2 = handler.get_alternative_source(
            exclude=["node-1", "node-2"], all_sources=all_sources
        )
        assert alt2 == "node-3"

        # Third stall on node-3 - no alternatives left
        handler.record_stall("node-3", "sync-789")
        alt3 = handler.get_alternative_source(
            exclude=["node-1", "node-2", "node-3"], all_sources=all_sources
        )
        assert alt3 is None

        # Record failed recovery
        handler.record_failed_recovery("sync-789", "No alternatives available")
        assert handler._failed_recoveries == 1

    def test_penalty_expiry_restores_availability(self):
        """Should restore source availability after penalty expires."""
        handler = SyncStallHandler(stall_penalty_seconds=0.2)
        all_sources = ["node-1", "node-2"]

        # Stall node-1
        handler.record_stall("node-1", "sync-123")

        # Initially, node-2 is the only alternative
        alt1 = handler.get_alternative_source(
            exclude=["node-1"], all_sources=all_sources
        )
        assert alt1 == "node-2"

        # Wait for penalty to expire
        time.sleep(0.3)

        # Now node-1 should be available again
        alt2 = handler.get_alternative_source(
            exclude=["node-2"], all_sources=all_sources
        )
        assert alt2 == "node-1"

    def test_multiple_concurrent_syncs(self):
        """Should handle multiple concurrent sync operations."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0)

        # Multiple syncs can stall at the same time
        sync_ids = [f"sync-{i}" for i in range(10)]
        for i, sync_id in enumerate(sync_ids):
            handler.record_stall(f"node-{i % 3}", sync_id)

        assert handler._stall_count == 10

        # Each node should have penalties
        stats = handler.get_stats()
        assert stats["active_penalties"] == 3  # node-0, node-1, node-2

    def test_retry_limit_tracking(self):
        """Should track retry attempts against max_retries."""
        handler = SyncStallHandler(stall_penalty_seconds=300.0, max_retries=3)

        # max_retries is configured but not enforced by handler
        # (enforcement is caller's responsibility)
        assert handler.max_retries == 3

        # Handler provides the config in stats for callers to check
        stats = handler.get_stats()
        assert stats["config"]["max_retries"] == 3
