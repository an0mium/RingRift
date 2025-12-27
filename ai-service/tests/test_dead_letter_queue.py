#!/usr/bin/env python3
"""Unit tests for DeadLetterQueue (December 2025).

Tests the SQLite-backed dead letter queue for failed event recovery.
"""

import asyncio
import os
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip("app.coordination.dead_letter_queue")


class TestFailedEvent:
    """Tests for FailedEvent dataclass."""

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        from app.coordination.dead_letter_queue import FailedEvent

        event = FailedEvent(
            event_id="test-123",
            event_type="SELFPLAY_COMPLETE",
            payload={"games": 100, "config": "hex8_2p"},
            handler_name="DataPipelineOrchestrator._on_selfplay_complete",
            error="Connection timeout",
            retry_count=2,
            source="selfplay_runner",
        )

        event_dict = event.to_dict()
        restored = FailedEvent.from_dict(event_dict)

        assert restored.event_id == event.event_id
        assert restored.event_type == event.event_type
        assert restored.payload == event.payload
        assert restored.handler_name == event.handler_name
        assert restored.error == event.error
        assert restored.retry_count == event.retry_count
        assert restored.source == event.source

    def test_from_dict_with_json_payload(self):
        """Test deserializing JSON string payload."""
        from app.coordination.dead_letter_queue import FailedEvent
        import json

        data = {
            "event_id": "test-456",
            "event_type": "NPZ_EXPORT_COMPLETE",
            "payload": json.dumps({"path": "/data/test.npz"}),
            "handler_name": "test_handler",
            "error": "File not found",
        }

        event = FailedEvent.from_dict(data)
        assert event.payload == {"path": "/data/test.npz"}

    def test_default_values(self):
        """Test default field values."""
        from app.coordination.dead_letter_queue import FailedEvent

        event = FailedEvent(
            event_id="test-789",
            event_type="TEST",
            payload={},
            handler_name="test",
            error="test error",
        )

        assert event.retry_count == 0
        assert event.last_retry_at is None
        assert event.source == "unknown"
        assert event.created_at is not None


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue class."""

    def test_init_creates_database(self):
        """Test that initialization creates the database and schema."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            # Verify database exists
            assert db_path.exists()

            # Verify schema
            with sqlite3.connect(db_path) as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = [t[0] for t in tables]
                assert "dead_letter" in table_names

    def test_capture_stores_event(self):
        """Test capturing a failed event."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            event_id = dlq.capture(
                event_type="TEST_EVENT",
                payload={"key": "value"},
                handler_name="test_handler",
                error="Test error message",
                source="test_source",
            )

            # Verify event was stored
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM dead_letter WHERE event_id = ?",
                    (event_id,)
                ).fetchone()

            assert row is not None
            assert row["event_type"] == "TEST_EVENT"
            assert row["handler_name"] == "test_handler"
            assert row["error"] == "Test error message"
            assert row["source"] == "test_source"
            assert row["status"] == "pending"
            assert row["retry_count"] == 0

    def test_get_pending_events(self):
        """Test retrieving pending events."""
        from app.coordination.dead_letter_queue import DeadLetterQueue

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            # Capture multiple events
            for i in range(5):
                dlq.capture(
                    event_type=f"EVENT_{i % 2}",  # Two types
                    payload={"index": i},
                    handler_name="test_handler",
                    error=f"Error {i}",
                )

            # Get all pending
            pending = dlq.get_pending_events()
            assert len(pending) == 5

            # Get filtered by type
            type_0 = dlq.get_pending_events(event_type="EVENT_0")
            assert len(type_0) == 3  # 0, 2, 4

            type_1 = dlq.get_pending_events(event_type="EVENT_1")
            assert len(type_1) == 2  # 1, 3

            # Test limit
            limited = dlq.get_pending_events(limit=2)
            assert len(limited) == 2

    def test_get_failed_events(self):
        """Test retrieving failed events for inspection."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            # Capture events
            event_id = dlq.capture(
                event_type="TEST",
                payload={},
                handler_name="test",
                error="error",
            )

            # Mark one as abandoned
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE dead_letter SET status = 'abandoned' WHERE event_id = ?",
                    (event_id,)
                )
                conn.commit()

            # Capture another pending event
            dlq.capture(
                event_type="TEST2",
                payload={},
                handler_name="test",
                error="error2",
            )

            # Without abandoned
            events = dlq.get_failed_events()
            assert len(events) == 1
            assert events[0]["event_type"] == "TEST2"

            # With abandoned
            events_all = dlq.get_failed_events(include_abandoned=True)
            assert len(events_all) == 2

    def test_register_handler(self):
        """Test handler registration."""
        from app.coordination.dead_letter_queue import DeadLetterQueue

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            async def handler1(payload):
                pass

            async def handler2(payload):
                pass

            dlq.register_handler("EVENT_A", handler1)
            dlq.register_handler("EVENT_A", handler2)
            dlq.register_handler("EVENT_B", handler1)

            assert len(dlq._handlers["EVENT_A"]) == 2
            assert len(dlq._handlers["EVENT_B"]) == 1

    @pytest.mark.asyncio
    async def test_retry_event_success(self):
        """Test successful event retry."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            # Capture an event
            event_id = dlq.capture(
                event_type="RETRY_TEST",
                payload={"data": "test"},
                handler_name="test",
                error="original error",
            )

            # Register a successful handler
            handler_called = []

            async def success_handler(payload):
                handler_called.append(payload)

            dlq.register_handler("RETRY_TEST", success_handler)

            # Retry
            success = await dlq.retry_event(event_id)
            assert success
            assert len(handler_called) == 1
            assert handler_called[0] == {"data": "test"}

            # Verify status updated
            with sqlite3.connect(db_path) as conn:
                status = conn.execute(
                    "SELECT status FROM dead_letter WHERE event_id = ?",
                    (event_id,)
                ).fetchone()[0]
            assert status == "recovered"

    @pytest.mark.asyncio
    async def test_retry_event_failure(self):
        """Test failed event retry with increment."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path, max_retries=3)

            # Capture an event
            event_id = dlq.capture(
                event_type="FAIL_TEST",
                payload={},
                handler_name="test",
                error="original error",
            )

            # Register a failing handler
            async def failing_handler(payload):
                raise RuntimeError("Still failing")

            dlq.register_handler("FAIL_TEST", failing_handler)

            # Retry should fail
            success = await dlq.retry_event(event_id)
            assert not success

            # Verify retry count incremented
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT retry_count, status FROM dead_letter WHERE event_id = ?",
                    (event_id,)
                ).fetchone()
            assert row[0] == 1
            assert row[1] == "pending"

    @pytest.mark.asyncio
    async def test_retry_event_abandonment(self):
        """Test event abandonment after max retries."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path, max_retries=2)

            # Capture an event with 1 retry already
            event_id = dlq.capture(
                event_type="ABANDON_TEST",
                payload={},
                handler_name="test",
                error="error",
            )

            # Manually set retry count to max_retries - 1
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE dead_letter SET retry_count = 1 WHERE event_id = ?",
                    (event_id,)
                )
                conn.commit()

            # Register failing handler
            async def failing_handler(payload):
                raise RuntimeError("Still failing")

            dlq.register_handler("ABANDON_TEST", failing_handler)

            # Retry should abandon
            success = await dlq.retry_event(event_id)
            assert not success

            # Verify status is abandoned
            with sqlite3.connect(db_path) as conn:
                status = conn.execute(
                    "SELECT status FROM dead_letter WHERE event_id = ?",
                    (event_id,)
                ).fetchone()[0]
            assert status == "abandoned"

    @pytest.mark.asyncio
    async def test_retry_event_no_handler(self):
        """Test retry with no registered handler."""
        from app.coordination.dead_letter_queue import DeadLetterQueue

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            event_id = dlq.capture(
                event_type="NO_HANDLER",
                payload={},
                handler_name="test",
                error="error",
            )

            # No handler registered
            success = await dlq.retry_event(event_id)
            assert not success

    @pytest.mark.asyncio
    async def test_retry_failed_events_batch(self):
        """Test batch retry with backoff respect."""
        from app.coordination.dead_letter_queue import DeadLetterQueue

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(
                db_path=db_path,
                base_backoff_seconds=0.1,  # Short for testing
            )

            # Capture events
            for i in range(3):
                dlq.capture(
                    event_type="BATCH_TEST",
                    payload={"index": i},
                    handler_name="test",
                    error="error",
                )

            # Register successful handler
            handled = []

            async def handler(payload):
                handled.append(payload["index"])

            dlq.register_handler("BATCH_TEST", handler)

            # Retry batch
            stats = await dlq.retry_failed_events(max_events=10)

            assert stats["recovered"] == 3
            assert stats["failed"] == 0
            assert len(handled) == 3

    def test_purge_old_events(self):
        """Test purging old events."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        import sqlite3
        from datetime import datetime, timedelta

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            # Capture event
            event_id = dlq.capture(
                event_type="OLD_EVENT",
                payload={},
                handler_name="test",
                error="error",
            )

            # Manually age the event and mark as recovered
            old_date = (datetime.now() - timedelta(days=10)).isoformat()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE dead_letter SET created_at = ?, status = 'recovered' "
                    "WHERE event_id = ?",
                    (old_date, event_id)
                )
                conn.commit()

            # Add a recent event (pending, won't be purged)
            dlq.capture(
                event_type="RECENT",
                payload={},
                handler_name="test",
                error="error",
            )

            # Purge events older than 7 days
            deleted = dlq.purge_old_events(days=7)
            assert deleted == 1

            # Verify only recent remains
            with sqlite3.connect(db_path) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM dead_letter"
                ).fetchone()[0]
            assert count == 1

    def test_get_stats(self):
        """Test statistics gathering."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)

            # Capture events of different types
            for i in range(3):
                dlq.capture(
                    event_type="TYPE_A",
                    payload={},
                    handler_name="test",
                    error="error",
                )
            for i in range(2):
                dlq.capture(
                    event_type="TYPE_B",
                    payload={},
                    handler_name="test",
                    error="error",
                )

            stats = dlq.get_stats()

            assert stats["pending"] == 5
            assert stats["total"] == 5
            assert stats["recovered"] == 0
            assert stats["abandoned"] == 0
            assert stats["by_event_type"]["TYPE_A"] == 3
            assert stats["by_event_type"]["TYPE_B"] == 2
            assert stats["session_captured"] == 5


class TestGlobalInstance:
    """Tests for global DLQ instance."""

    def test_get_dead_letter_queue_singleton(self):
        """Test that get_dead_letter_queue returns singleton."""
        from app.coordination import dead_letter_queue as dlq_module

        # Reset global instance
        dlq_module._dead_letter_queue = None

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_global.db"

            dlq1 = dlq_module.get_dead_letter_queue(db_path=db_path)
            dlq2 = dlq_module.get_dead_letter_queue()

            assert dlq1 is dlq2

            # Reset for other tests
            dlq_module._dead_letter_queue = None

    def test_enable_dead_letter_queue(self):
        """Test enabling DLQ on event bus."""
        from app.coordination.dead_letter_queue import (
            DeadLetterQueue,
            enable_dead_letter_queue,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_enable.db"
            dlq = DeadLetterQueue(db_path=db_path)

            # Mock event bus
            mock_bus = MagicMock()
            mock_bus.emit = AsyncMock(return_value=1)

            enable_dead_letter_queue(dlq, mock_bus)

            # Verify DLQ reference stored
            assert mock_bus._dlq is dlq


class TestConfigurableBackoff:
    """Tests for backoff configuration."""

    def test_custom_backoff_settings(self):
        """Test custom backoff configuration."""
        from app.coordination.dead_letter_queue import DeadLetterQueue

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_backoff.db"
            dlq = DeadLetterQueue(
                db_path=db_path,
                max_retries=10,
                base_backoff_seconds=30.0,
                max_backoff_seconds=1800.0,
            )

            assert dlq.max_retries == 10
            assert dlq.base_backoff == 30.0
            assert dlq.max_backoff == 1800.0

    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self):
        """Test that backoff increases exponentially."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        from datetime import datetime
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_exp_backoff.db"
            dlq = DeadLetterQueue(
                db_path=db_path,
                base_backoff_seconds=1.0,
                max_backoff_seconds=100.0,
            )

            # Capture event
            event_id = dlq.capture(
                event_type="BACKOFF_TEST",
                payload={},
                handler_name="test",
                error="error",
            )

            # Set retry count and recent last_retry_at
            now = datetime.now()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE dead_letter SET retry_count = 3, last_retry_at = ? "
                    "WHERE event_id = ?",
                    (now.isoformat(), event_id)
                )
                conn.commit()

            # Backoff for retry 3 = 1.0 * 2^3 = 8 seconds
            # Since we just set last_retry_at to now, it should be skipped
            stats = await dlq.retry_failed_events(max_events=1)
            assert stats["skipped"] == 1

    @pytest.mark.asyncio
    async def test_max_backoff_cap(self):
        """Test that backoff is capped at max_backoff_seconds."""
        from app.coordination.dead_letter_queue import DeadLetterQueue
        from datetime import datetime, timedelta
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_max_backoff.db"
            dlq = DeadLetterQueue(
                db_path=db_path,
                base_backoff_seconds=1.0,
                max_backoff_seconds=10.0,  # Cap at 10 seconds
            )

            # Capture event
            event_id = dlq.capture(
                event_type="MAX_BACKOFF_TEST",
                payload={},
                handler_name="test",
                error="error",
            )

            # Set high retry count (would be 1.0 * 2^10 = 1024s without cap)
            # But with cap, should be 10s
            past = datetime.now() - timedelta(seconds=15)  # 15s ago > 10s cap
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE dead_letter SET retry_count = 10, last_retry_at = ? "
                    "WHERE event_id = ?",
                    (past.isoformat(), event_id)
                )
                conn.commit()

            # Register handler
            async def handler(payload):
                pass

            dlq.register_handler("MAX_BACKOFF_TEST", handler)

            # Should not be skipped since enough time passed
            stats = await dlq.retry_failed_events(max_events=1)
            assert stats["recovered"] == 1
            assert stats["skipped"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
