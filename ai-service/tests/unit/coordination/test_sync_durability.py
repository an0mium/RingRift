"""Comprehensive tests for sync_durability.py module.

Tests for SyncWAL (Write-Ahead Log) and DeadLetterQueue (DLQ) functionality.

Test categories:
1. SyncWAL basic operations (append, recover, mark_complete, mark_failed)
2. SyncWAL crash recovery simulation
3. DeadLetterQueue operations
4. Edge cases (empty WAL, duplicate entries, concurrent access)
5. Integration between WAL and DLQ
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from app.coordination.sync_durability import (
    DLQStats,
    DeadLetterEntry,
    DeadLetterQueue,
    SyncStatus,
    SyncWAL,
    SyncWALEntry,
    WALStats,
    get_dlq,
    get_sync_wal,
    reset_instances,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_sync.db"


@pytest.fixture
def wal(temp_db_path: Path) -> SyncWAL:
    """Create a SyncWAL instance for testing."""
    return SyncWAL(db_path=temp_db_path)


@pytest.fixture
def dlq(tmp_path: Path) -> DeadLetterQueue:
    """Create a DeadLetterQueue instance for testing."""
    return DeadLetterQueue(db_path=tmp_path / "test_dlq.db")


@pytest.fixture
def sample_game_data() -> dict[str, Any]:
    """Sample game data for testing."""
    return {
        "moves": [{"from": [0, 0], "to": [0, 1]}, {"from": [1, 1], "to": [1, 2]}],
        "players": ["alice", "bob"],
        "board_type": "hex8",
        "num_players": 2,
    }


# =============================================================================
# Test: SyncWAL Basic Operations
# =============================================================================


class TestSyncWALBasicOperations:
    """Test basic SyncWAL operations."""

    def test_initialization_creates_database(self, temp_db_path: Path):
        """Test that initialization creates the database and schema."""
        assert not temp_db_path.exists()

        wal = SyncWAL(db_path=temp_db_path)

        assert temp_db_path.exists()
        assert temp_db_path.is_file()

        # Verify schema
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Check sync_wal table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sync_wal'"
        )
        assert cursor.fetchone() is not None

        # Check indexes exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_sync_wal_status'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_sync_wal_game_id'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_sync_wal_dedup'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_append_creates_entry(self, wal: SyncWAL, sample_game_data: dict):
        """Test that append creates a new entry."""
        entry_id = wal.append(
            game_id="game_001",
            source="node_a",
            target="node_b",
            data=sample_game_data,
        )

        assert isinstance(entry_id, int)
        assert entry_id > 0

        # Verify entry in database
        stats = wal.get_stats()
        assert stats.total_entries == 1
        assert stats.pending == 1

    def test_append_returns_incremental_ids(self, wal: SyncWAL, sample_game_data: dict):
        """Test that append returns incremental entry IDs."""
        id1 = wal.append("game_001", "node_a", "node_b", sample_game_data)
        id2 = wal.append("game_002", "node_a", "node_b", sample_game_data)
        id3 = wal.append("game_003", "node_a", "node_b", sample_game_data)

        assert id2 == id1 + 1
        assert id3 == id2 + 1

    def test_append_idempotency(self, wal: SyncWAL, sample_game_data: dict):
        """Test that duplicate entries return the same ID (idempotent)."""
        id1 = wal.append("game_001", "node_a", "node_b", sample_game_data)
        id2 = wal.append("game_001", "node_a", "node_b", sample_game_data)

        assert id1 == id2

        # Should still only have one entry
        stats = wal.get_stats()
        assert stats.total_entries == 1

    def test_append_different_data_creates_new_entry(
        self, wal: SyncWAL, sample_game_data: dict
    ):
        """Test that same game with different data creates new entry."""
        id1 = wal.append("game_001", "node_a", "node_b", sample_game_data)

        # Different data
        different_data = {**sample_game_data, "moves": [{"from": [2, 2], "to": [3, 3]}]}
        id2 = wal.append("game_001", "node_a", "node_b", different_data)

        assert id2 != id1

        # Should have two entries
        stats = wal.get_stats()
        assert stats.total_entries == 2

    def test_mark_complete_updates_status(self, wal: SyncWAL, sample_game_data: dict):
        """Test that mark_complete updates entry status."""
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)

        result = wal.mark_complete(entry_id)
        assert result is True

        stats = wal.get_stats()
        assert stats.pending == 0
        assert stats.completed == 1

    def test_mark_complete_nonexistent_entry(self, wal: SyncWAL):
        """Test that mark_complete returns False for nonexistent entry."""
        result = wal.mark_complete(99999)
        assert result is False

    def test_mark_complete_already_completed(
        self, wal: SyncWAL, sample_game_data: dict
    ):
        """Test that mark_complete on already completed entry returns False."""
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)

        wal.mark_complete(entry_id)
        result = wal.mark_complete(entry_id)

        # Should return False because it was already complete
        assert result is False

    def test_mark_failed_updates_status(self, wal: SyncWAL, sample_game_data: dict):
        """Test that mark_failed updates entry status."""
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)

        result = wal.mark_failed(entry_id, "Connection timeout")
        assert result is True

        stats = wal.get_stats()
        assert stats.pending == 0
        assert stats.failed == 1

    def test_mark_failed_increments_retry_count(
        self, wal: SyncWAL, sample_game_data: dict
    ):
        """Test that mark_failed increments retry count."""
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)

        wal.mark_failed(entry_id, "First failure")
        wal.mark_failed(entry_id, "Second failure")
        wal.mark_failed(entry_id, "Third failure")

        entries = wal.get_pending(limit=10)
        # Entry should be failed, not pending
        assert len(entries) == 0

        # Check directly in database
        with wal._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT retry_count FROM sync_wal WHERE entry_id = ?", (entry_id,)
            )
            retry_count = cursor.fetchone()[0]
            assert retry_count == 3

    def test_get_pending_returns_only_pending(self, wal: SyncWAL, sample_game_data: dict):
        """Test that get_pending returns only pending entries."""
        # Create entries with different statuses
        id1 = wal.append("game_001", "node_a", "node_b", sample_game_data)
        id2 = wal.append("game_002", "node_a", "node_b", sample_game_data)
        id3 = wal.append("game_003", "node_a", "node_b", sample_game_data)

        wal.mark_complete(id1)
        wal.mark_failed(id2, "Error")
        # id3 stays pending

        pending = wal.get_pending(limit=10)
        assert len(pending) == 1
        assert pending[0].entry_id == id3

    def test_get_pending_respects_limit(self, wal: SyncWAL, sample_game_data: dict):
        """Test that get_pending respects the limit parameter."""
        # Create 5 pending entries
        for i in range(5):
            wal.append(f"game_{i:03d}", "node_a", "node_b", sample_game_data)

        pending = wal.get_pending(limit=3)
        assert len(pending) == 3

    def test_get_pending_returns_oldest_first(self, wal: SyncWAL, sample_game_data: dict):
        """Test that get_pending returns oldest entries first."""
        ids = []
        for i in range(5):
            entry_id = wal.append(f"game_{i:03d}", "node_a", "node_b", sample_game_data)
            ids.append(entry_id)
            time.sleep(0.01)  # Small delay to ensure different timestamps

        pending = wal.get_pending(limit=10)
        pending_ids = [e.entry_id for e in pending]

        assert pending_ids == ids  # Should be in order of creation


# =============================================================================
# Test: SyncWAL Crash Recovery
# =============================================================================


class TestSyncWALCrashRecovery:
    """Test crash recovery scenarios."""

    def test_recover_returns_pending_entries(
        self, temp_db_path: Path, sample_game_data: dict
    ):
        """Test that recover() returns all pending entries after restart."""
        # Create initial WAL and add entries
        wal1 = SyncWAL(db_path=temp_db_path)
        id1 = wal1.append("game_001", "node_a", "node_b", sample_game_data)
        id2 = wal1.append("game_002", "node_a", "node_b", sample_game_data)
        id3 = wal1.append("game_003", "node_a", "node_b", sample_game_data)

        # Mark one as complete
        wal1.mark_complete(id1)

        # Simulate crash - close connection pool
        if wal1._conn_pool:
            wal1._conn_pool.close_all()

        # Create new WAL instance (simulates restart)
        wal2 = SyncWAL(db_path=temp_db_path)
        recovered = wal2.recover()

        assert len(recovered) == 2
        recovered_ids = {e.entry_id for e in recovered}
        assert recovered_ids == {id2, id3}

    def test_recover_returns_in_progress_entries(
        self, temp_db_path: Path, sample_game_data: dict
    ):
        """Test that recover() returns entries that were in-progress during crash."""
        wal = SyncWAL(db_path=temp_db_path)
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)

        # Manually set to in_progress (simulate crash during sync)
        with wal._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sync_wal SET status = ? WHERE entry_id = ?",
                (SyncStatus.IN_PROGRESS.value, entry_id),
            )
            conn.commit()

        # Close and reopen
        if wal._conn_pool:
            wal._conn_pool.close_all()

        wal2 = SyncWAL(db_path=temp_db_path)
        recovered = wal2.recover()

        assert len(recovered) == 1
        assert recovered[0].entry_id == entry_id
        assert recovered[0].status == SyncStatus.IN_PROGRESS

    def test_recover_empty_wal(self, temp_db_path: Path):
        """Test that recover() returns empty list for empty WAL."""
        wal = SyncWAL(db_path=temp_db_path)
        recovered = wal.recover()

        assert recovered == []

    def test_recover_all_completed(self, temp_db_path: Path, sample_game_data: dict):
        """Test that recover() returns empty list when all entries completed."""
        wal = SyncWAL(db_path=temp_db_path)
        id1 = wal.append("game_001", "node_a", "node_b", sample_game_data)
        id2 = wal.append("game_002", "node_a", "node_b", sample_game_data)

        wal.mark_complete(id1)
        wal.mark_complete(id2)

        recovered = wal.recover()
        assert recovered == []

    def test_sync_wal_entry_data_property(self, wal: SyncWAL, sample_game_data: dict):
        """Test that SyncWALEntry.data property correctly parses JSON."""
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)
        pending = wal.get_pending(limit=1)

        assert len(pending) == 1
        entry = pending[0]

        # Test data property
        assert entry.data == sample_game_data
        assert isinstance(entry.data, dict)

    def test_sync_wal_entry_to_dict(self, wal: SyncWAL, sample_game_data: dict):
        """Test that SyncWALEntry.to_dict() returns correct dictionary."""
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)
        pending = wal.get_pending(limit=1)

        entry_dict = pending[0].to_dict()

        assert entry_dict["entry_id"] == entry_id
        assert entry_dict["game_id"] == "game_001"
        assert entry_dict["source_host"] == "node_a"
        assert entry_dict["target_host"] == "node_b"
        assert entry_dict["status"] == SyncStatus.PENDING.value
        assert isinstance(entry_dict["created_at"], float)


# =============================================================================
# Test: SyncWAL Statistics
# =============================================================================


class TestSyncWALStatistics:
    """Test WAL statistics functionality."""

    def test_get_stats_empty_wal(self, wal: SyncWAL):
        """Test get_stats on empty WAL."""
        stats = wal.get_stats()

        assert stats.total_entries == 0
        assert stats.pending == 0
        assert stats.in_progress == 0
        assert stats.completed == 0
        assert stats.failed == 0
        assert stats.oldest_pending is None
        assert stats.newest_entry is None

    def test_get_stats_with_entries(self, wal: SyncWAL, sample_game_data: dict):
        """Test get_stats with various entry states."""
        id1 = wal.append("game_001", "node_a", "node_b", sample_game_data)
        id2 = wal.append("game_002", "node_a", "node_b", sample_game_data)
        id3 = wal.append("game_003", "node_a", "node_b", sample_game_data)
        id4 = wal.append("game_004", "node_a", "node_b", sample_game_data)

        wal.mark_complete(id1)
        wal.mark_complete(id2)
        wal.mark_failed(id3, "Error")
        # id4 stays pending

        stats = wal.get_stats()

        assert stats.total_entries == 4
        assert stats.pending == 1
        assert stats.in_progress == 0
        assert stats.completed == 2
        assert stats.failed == 1
        assert stats.oldest_pending is not None
        assert stats.newest_entry is not None

    def test_clear_completed_removes_old_entries(
        self, wal: SyncWAL, sample_game_data: dict
    ):
        """Test that clear_completed removes old completed entries."""
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)
        wal.mark_complete(entry_id)

        # Manually set completed_at to old timestamp
        old_timestamp = time.time() - (25 * 3600)  # 25 hours ago
        with wal._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sync_wal SET completed_at = ? WHERE entry_id = ?",
                (old_timestamp, entry_id),
            )
            conn.commit()

        # Clear entries older than 24 hours
        removed = wal.clear_completed(older_than_hours=24)

        assert removed == 1

        stats = wal.get_stats()
        assert stats.total_entries == 0

    def test_clear_completed_keeps_recent_entries(
        self, wal: SyncWAL, sample_game_data: dict
    ):
        """Test that clear_completed keeps recent completed entries."""
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)
        wal.mark_complete(entry_id)

        removed = wal.clear_completed(older_than_hours=24)

        assert removed == 0

        stats = wal.get_stats()
        assert stats.total_entries == 1


# =============================================================================
# Test: DeadLetterQueue Operations
# =============================================================================


class TestDeadLetterQueueOperations:
    """Test DeadLetterQueue operations."""

    def test_dlq_initialization_creates_database(self, tmp_path: Path):
        """Test that DLQ initialization creates the database and schema."""
        db_path = tmp_path / "test_dlq.db"
        assert not db_path.exists()

        dlq = DeadLetterQueue(db_path=db_path)

        assert db_path.exists()

        # Verify schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='dead_letter_queue'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_add_creates_entry(self, dlq: DeadLetterQueue):
        """Test that add creates a new DLQ entry."""
        entry_id = dlq.add(
            game_id="game_001",
            source="node_a",
            target="node_b",
            error="Connection timeout",
            error_type="timeout",
            retry_count=3,
        )

        assert isinstance(entry_id, int)
        assert entry_id > 0

        stats = dlq.get_stats()
        assert stats.total_entries == 1
        assert stats.unresolved == 1

    def test_get_pending_returns_unresolved(self, dlq: DeadLetterQueue):
        """Test that get_pending returns only unresolved entries."""
        id1 = dlq.add(
            "game_001", "node_a", "node_b", "Error 1", "timeout", retry_count=3
        )
        id2 = dlq.add(
            "game_002", "node_a", "node_b", "Error 2", "validation", retry_count=2
        )

        dlq.resolve("game_001")

        pending = dlq.get_pending(limit=10)

        assert len(pending) == 1
        assert pending[0].game_id == "game_002"

    def test_get_pending_respects_limit(self, dlq: DeadLetterQueue):
        """Test that get_pending respects limit parameter."""
        for i in range(5):
            dlq.add(f"game_{i:03d}", "node_a", "node_b", "Error", "timeout", retry_count=1)

        pending = dlq.get_pending(limit=3)
        assert len(pending) == 3

    def test_resolve_marks_entry_resolved(self, dlq: DeadLetterQueue):
        """Test that resolve marks entry as resolved."""
        dlq.add("game_001", "node_a", "node_b", "Error", "timeout", retry_count=3)

        updated = dlq.resolve("game_001")

        assert updated == 1

        stats = dlq.get_stats()
        assert stats.unresolved == 0
        assert stats.resolved == 1

    def test_resolve_nonexistent_entry(self, dlq: DeadLetterQueue):
        """Test that resolve returns 0 for nonexistent entry."""
        updated = dlq.resolve("nonexistent_game")
        assert updated == 0

    def test_resolve_already_resolved(self, dlq: DeadLetterQueue):
        """Test that resolve on already resolved entry returns 0."""
        dlq.add("game_001", "node_a", "node_b", "Error", "timeout", retry_count=3)

        dlq.resolve("game_001")
        updated = dlq.resolve("game_001")

        assert updated == 0

    def test_update_retry_increments_count(self, dlq: DeadLetterQueue):
        """Test that update_retry increments retry count."""
        dlq.add("game_001", "node_a", "node_b", "Error 1", "timeout", retry_count=3)

        result = dlq.update_retry("game_001", error="Error 2")
        assert result is True

        # Check retry count increased
        with dlq._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT retry_count, error_message FROM dead_letter_queue "
                "WHERE game_id = ?",
                ("game_001",),
            )
            row = cursor.fetchone()
            assert row[0] == 4  # 3 + 1
            assert row[1] == "Error 2"

    def test_update_retry_without_error_message(self, dlq: DeadLetterQueue):
        """Test that update_retry works without updating error message."""
        dlq.add("game_001", "node_a", "node_b", "Original error", "timeout", retry_count=1)

        result = dlq.update_retry("game_001")
        assert result is True

        with dlq._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT retry_count, error_message FROM dead_letter_queue "
                "WHERE game_id = ?",
                ("game_001",),
            )
            row = cursor.fetchone()
            assert row[0] == 2
            assert row[1] == "Original error"

    def test_dead_letter_entry_to_dict(self, dlq: DeadLetterQueue):
        """Test that DeadLetterEntry.to_dict() returns correct dictionary."""
        dlq.add(
            "game_001",
            "node_a",
            "node_b",
            "Connection timeout",
            "timeout",
            retry_count=3,
        )

        pending = dlq.get_pending(limit=1)
        entry_dict = pending[0].to_dict()

        assert entry_dict["game_id"] == "game_001"
        assert entry_dict["source_host"] == "node_a"
        assert entry_dict["target_host"] == "node_b"
        assert entry_dict["error_message"] == "Connection timeout"
        assert entry_dict["error_type"] == "timeout"
        assert entry_dict["retry_count"] == 3
        assert entry_dict["resolved"] is False


# =============================================================================
# Test: DeadLetterQueue Statistics
# =============================================================================


class TestDeadLetterQueueStatistics:
    """Test DLQ statistics functionality."""

    def test_get_stats_empty_dlq(self, dlq: DeadLetterQueue):
        """Test get_stats on empty DLQ."""
        stats = dlq.get_stats()

        assert stats.total_entries == 0
        assert stats.unresolved == 0
        assert stats.resolved == 0
        assert stats.by_error_type == {}
        assert stats.oldest_unresolved is None
        assert stats.avg_retry_count == 0.0

    def test_get_stats_with_entries(self, dlq: DeadLetterQueue):
        """Test get_stats with various entries."""
        dlq.add("game_001", "node_a", "node_b", "Error 1", "timeout", retry_count=3)
        dlq.add("game_002", "node_a", "node_b", "Error 2", "timeout", retry_count=2)
        dlq.add("game_003", "node_a", "node_b", "Error 3", "validation", retry_count=1)

        dlq.resolve("game_001")

        stats = dlq.get_stats()

        assert stats.total_entries == 3
        assert stats.unresolved == 2
        assert stats.resolved == 1
        assert stats.by_error_type == {"timeout": 1, "validation": 1}
        assert stats.oldest_unresolved is not None
        assert stats.avg_retry_count == 2.0  # (3 + 2 + 1) / 3

    def test_cleanup_resolved_removes_old_entries(self, dlq: DeadLetterQueue):
        """Test that cleanup_resolved removes old resolved entries."""
        dlq.add("game_001", "node_a", "node_b", "Error", "timeout", retry_count=3)
        dlq.resolve("game_001")

        # Manually set resolved_at to old timestamp
        old_timestamp = time.time() - (8 * 86400)  # 8 days ago
        with dlq._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE dead_letter_queue SET resolved_at = ? WHERE game_id = ?",
                (old_timestamp, "game_001"),
            )
            conn.commit()

        removed = dlq.cleanup_resolved(older_than_days=7)

        assert removed == 1

        stats = dlq.get_stats()
        assert stats.total_entries == 0

    def test_cleanup_resolved_keeps_recent_entries(self, dlq: DeadLetterQueue):
        """Test that cleanup_resolved keeps recent resolved entries."""
        dlq.add("game_001", "node_a", "node_b", "Error", "timeout", retry_count=3)
        dlq.resolve("game_001")

        removed = dlq.cleanup_resolved(older_than_days=7)

        assert removed == 0

        stats = dlq.get_stats()
        assert stats.total_entries == 1


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_wal_max_pending_limit(self, wal: SyncWAL, sample_game_data: dict):
        """Test that WAL enforces max_pending limit."""
        # Create WAL with low limit
        temp_db = wal.db_path.parent / "limited_wal.db"
        limited_wal = SyncWAL(db_path=temp_db, max_pending=3)

        # Add 3 entries (at limit)
        limited_wal.append("game_001", "node_a", "node_b", sample_game_data)
        limited_wal.append("game_002", "node_a", "node_b", sample_game_data)
        limited_wal.append("game_003", "node_a", "node_b", sample_game_data)

        # 4th should fail
        with pytest.raises(RuntimeError, match="SyncWAL full"):
            limited_wal.append("game_004", "node_a", "node_b", sample_game_data)

    def test_wal_invalid_json_in_data_property(self, temp_db_path: Path):
        """Test that invalid JSON in data property returns empty dict."""
        wal = SyncWAL(db_path=temp_db_path)

        # Manually insert entry with invalid JSON
        with wal._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sync_wal
                (game_id, source_host, target_host, data_json, data_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("game_001", "node_a", "node_b", "invalid{json", "hash123", time.time()),
            )
            conn.commit()

        pending = wal.get_pending(limit=1)
        assert len(pending) == 1
        assert pending[0].data == {}  # Should return empty dict for invalid JSON

    def test_concurrent_append(self, temp_db_path: Path, sample_game_data: dict):
        """Test concurrent append operations."""
        wal = SyncWAL(db_path=temp_db_path)
        results = []

        def append_entries(count: int):
            for i in range(count):
                entry_id = wal.append(
                    f"game_{threading.current_thread().name}_{i:03d}",
                    "node_a",
                    "node_b",
                    sample_game_data,
                )
                results.append(entry_id)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=append_entries, args=(10,), name=f"T{i}")
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert len(results) == 50

        # All IDs should be unique
        assert len(set(results)) == 50

        stats = wal.get_stats()
        assert stats.total_entries == 50

    def test_connection_pool_without_pooling(self, tmp_path: Path, sample_game_data: dict):
        """Test WAL without connection pooling."""
        wal = SyncWAL(
            db_path=tmp_path / "no_pool.db",
            use_connection_pool=False,
        )

        # Should still work
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)
        assert entry_id > 0

        wal.mark_complete(entry_id)

        stats = wal.get_stats()
        assert stats.completed == 1

    def test_automatic_cleanup_after_operations(
        self, temp_db_path: Path, sample_game_data: dict
    ):
        """Test that automatic cleanup runs after cleanup_interval operations."""
        wal = SyncWAL(db_path=temp_db_path, cleanup_interval=10)

        # Add and complete 5 entries
        for i in range(5):
            entry_id = wal.append(f"game_{i:03d}", "node_a", "node_b", sample_game_data)
            wal.mark_complete(entry_id)

        # Manually set completed_at to old timestamp for all
        old_timestamp = time.time() - (25 * 3600)
        with wal._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE sync_wal SET completed_at = ?", (old_timestamp,))
            conn.commit()

        # Add 10 more entries to trigger cleanup
        for i in range(10):
            wal.append(f"game_new_{i:03d}", "node_a", "node_b", sample_game_data)

        # Old completed entries should be cleaned up
        stats = wal.get_stats()
        assert stats.completed == 0  # Old entries cleaned
        assert stats.pending == 10  # New entries remain


# =============================================================================
# Test: Integration WAL + DLQ
# =============================================================================


class TestWALDLQIntegration:
    """Test integration between WAL and DLQ."""

    def test_failed_sync_workflow(
        self, temp_db_path: Path, tmp_path: Path, sample_game_data: dict
    ):
        """Test complete workflow: WAL -> failure -> DLQ."""
        wal = SyncWAL(db_path=temp_db_path)
        dlq = DeadLetterQueue(db_path=tmp_path / "integration_dlq.db")

        # 1. Add entry to WAL
        entry_id = wal.append("game_001", "node_a", "node_b", sample_game_data)

        # 2. Simulate sync failure
        wal.mark_failed(entry_id, "Connection timeout after 3 retries")

        # 3. Move to DLQ
        pending = wal.get_pending(limit=10)
        # Entry should be failed, not pending
        assert len(pending) == 0

        # Get failed entry from WAL
        with wal._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT game_id, source_host, target_host, error_message, retry_count "
                "FROM sync_wal WHERE entry_id = ? AND status = ?",
                (entry_id, SyncStatus.FAILED.value),
            )
            row = cursor.fetchone()

        assert row is not None
        game_id, source, target, error_msg, retry_count = row

        # 4. Add to DLQ
        dlq.add(
            game_id=game_id,
            source=source,
            target=target,
            error=error_msg,
            error_type="timeout",
            retry_count=retry_count,
        )

        # 5. Verify in DLQ
        dlq_entries = dlq.get_pending(limit=10)
        assert len(dlq_entries) == 1
        assert dlq_entries[0].game_id == "game_001"

    def test_retry_from_dlq_to_wal(self, temp_db_path: Path, tmp_path: Path):
        """Test retrying a DLQ entry by re-adding to WAL."""
        wal = SyncWAL(db_path=temp_db_path)
        dlq = DeadLetterQueue(db_path=tmp_path / "retry_dlq.db")

        # 1. Add to DLQ
        dlq.add(
            "game_001",
            "node_a",
            "node_b",
            "Original error",
            "timeout",
            retry_count=3,
        )

        # 2. Decide to retry - add back to WAL
        retry_data = {"moves": [], "retry": True}
        entry_id = wal.append("game_001", "node_a", "node_b", retry_data)

        # 3. Simulate successful sync
        wal.mark_complete(entry_id)

        # 4. Resolve DLQ entry
        dlq.resolve("game_001")

        # Verify
        wal_stats = wal.get_stats()
        dlq_stats = dlq.get_stats()

        assert wal_stats.completed == 1
        assert dlq_stats.resolved == 1


# =============================================================================
# Test: Singleton Instances
# =============================================================================


class TestSingletonInstances:
    """Test singleton instance management."""

    def test_get_sync_wal_creates_instance(self, tmp_path: Path):
        """Test that get_sync_wal creates singleton instance."""
        reset_instances()

        wal1 = get_sync_wal(db_path=tmp_path / "singleton_wal.db")
        wal2 = get_sync_wal()

        assert wal1 is wal2

    def test_get_dlq_creates_instance(self, tmp_path: Path):
        """Test that get_dlq creates singleton instance."""
        reset_instances()

        dlq1 = get_dlq(db_path=tmp_path / "singleton_dlq.db")
        dlq2 = get_dlq()

        assert dlq1 is dlq2

    def test_reset_instances_clears_singletons(self, tmp_path: Path):
        """Test that reset_instances clears singleton instances."""
        reset_instances()

        wal1 = get_sync_wal(db_path=tmp_path / "reset_wal.db")
        dlq1 = get_dlq(db_path=tmp_path / "reset_dlq.db")

        reset_instances()

        wal2 = get_sync_wal(db_path=tmp_path / "reset_wal2.db")
        dlq2 = get_dlq(db_path=tmp_path / "reset_dlq2.db")

        assert wal1 is not wal2
        assert dlq1 is not dlq2


# =============================================================================
# Test: Connection Pool
# =============================================================================


class TestConnectionPool:
    """Test connection pool functionality."""

    def test_connection_pool_reuses_connections(self, temp_db_path: Path):
        """Test that connection pool reuses connections in same thread."""
        wal = SyncWAL(db_path=temp_db_path, use_connection_pool=True)

        # Make multiple operations
        for i in range(10):
            wal.append(f"game_{i:03d}", "node_a", "node_b", {"data": i})

        # Check pool stats
        if wal._conn_pool:
            stats = wal._conn_pool.get_stats()
            assert stats["connections_created"] > 0
            assert stats["connections_reused"] > 0
            assert stats["reuse_ratio"] > 0

    def test_connection_pool_per_thread(self, temp_db_path: Path, sample_game_data: dict):
        """Test that connection pool creates separate connections per thread."""
        wal = SyncWAL(db_path=temp_db_path, use_connection_pool=True)

        def thread_operations():
            for i in range(5):
                wal.append(
                    f"game_{threading.current_thread().name}_{i}",
                    "node_a",
                    "node_b",
                    sample_game_data,
                )

        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_operations, name=f"T{i}")
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        stats = wal.get_stats()
        assert stats.total_entries == 15  # 3 threads * 5 entries
