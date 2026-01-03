"""Tests for work queue backend Strategy pattern (Jan 2, 2026).

Tests the WorkQueueBackend abstraction and its implementations:
- SQLiteBackend - Local persistence
- RaftBackend - Cluster-wide with fallback
"""

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.work_queue_backends import (
    BackendResult,
    BackendType,
    RaftBackend,
    SQLiteBackend,
    WorkQueueBackend,
    create_backend,
)


class TestBackendResult:
    """Tests for BackendResult dataclass."""

    def test_success_result(self):
        result = BackendResult(success=True, data={"work_id": "test123"})
        assert result.success is True
        assert result.data == {"work_id": "test123"}
        assert result.error is None
        assert result.fallback_used is False

    def test_failure_result(self):
        result = BackendResult(success=False, error="Connection failed")
        assert result.success is False
        assert result.error == "Connection failed"

    def test_fallback_result(self):
        result = BackendResult(success=True, fallback_used=True)
        assert result.fallback_used is True


class TestBackendType:
    """Tests for BackendType enum."""

    def test_raft_value(self):
        assert BackendType.RAFT.value == "raft"

    def test_sqlite_value(self):
        assert BackendType.SQLITE.value == "sqlite"


class TestSQLiteBackend:
    """Tests for SQLiteBackend implementation."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # Initialize schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS work_items (
                work_id TEXT PRIMARY KEY,
                work_type TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 50,
                config TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                claimed_at REAL NOT NULL DEFAULT 0.0,
                started_at REAL NOT NULL DEFAULT 0.0,
                completed_at REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'pending',
                claimed_by TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                timeout_seconds REAL NOT NULL DEFAULT 3600.0,
                result TEXT NOT NULL DEFAULT '{}',
                error TEXT NOT NULL DEFAULT '',
                depends_on TEXT NOT NULL DEFAULT '[]'
            )
        """)
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        db_path.unlink(missing_ok=True)

    @pytest.fixture
    def sqlite_backend(self, temp_db):
        """Create SQLiteBackend with temp database."""

        def get_connection():
            return sqlite3.connect(str(temp_db), timeout=10.0)

        return SQLiteBackend(
            db_path=temp_db,
            get_connection=get_connection,
            readonly_mode=False,
        )

    def test_backend_type(self, sqlite_backend):
        assert sqlite_backend.backend_type == BackendType.SQLITE

    def test_is_available(self, sqlite_backend):
        assert sqlite_backend.is_available is True

    def test_add_item(self, sqlite_backend):
        work_data = {
            "work_type": "selfplay",
            "priority": 100,
            "config": {"board_type": "hex8"},
            "created_at": time.time(),
            "status": "pending",
        }

        result = sqlite_backend.add_item("work-001", work_data)
        assert result.success is True
        assert result.data == "work-001"

    def test_add_item_readonly(self, temp_db):
        """Test that add_item fails in readonly mode."""

        def get_connection():
            return sqlite3.connect(str(temp_db))

        backend = SQLiteBackend(
            db_path=temp_db,
            get_connection=get_connection,
            readonly_mode=True,
        )

        result = backend.add_item("work-001", {})
        assert result.success is False
        assert "readonly" in result.error.lower()

    def test_get_item(self, sqlite_backend):
        # Add an item first
        work_data = {
            "work_type": "training",
            "priority": 90,
            "config": {"epochs": 10},
            "created_at": time.time(),
            "status": "pending",
        }
        sqlite_backend.add_item("work-002", work_data)

        # Get it
        item = sqlite_backend.get_item("work-002")
        assert item is not None
        assert item["work_id"] == "work-002"
        assert item["work_type"] == "training"
        assert item["config"] == {"epochs": 10}

    def test_get_item_not_found(self, sqlite_backend):
        item = sqlite_backend.get_item("nonexistent")
        assert item is None

    def test_get_pending_items(self, sqlite_backend):
        # Add multiple items
        for i in range(5):
            work_data = {
                "work_type": "selfplay",
                "priority": 50 + i,
                "config": {},
                "created_at": time.time(),
                "status": "pending",
            }
            sqlite_backend.add_item(f"work-{i:03d}", work_data)

        pending = sqlite_backend.get_pending_items(limit=3)
        assert len(pending) == 3
        # Should be sorted by priority DESC
        assert pending[0]["priority"] >= pending[1]["priority"]

    def test_claim_item(self, sqlite_backend):
        # Add pending item
        work_data = {
            "work_type": "selfplay",
            "priority": 50,
            "config": {},
            "created_at": time.time(),
            "status": "pending",
        }
        sqlite_backend.add_item("work-claim-001", work_data)

        # Claim it
        result = sqlite_backend.claim_item("work-claim-001", "node-1")
        assert result.success is True
        assert result.data["node_id"] == "node-1"

        # Verify status changed
        item = sqlite_backend.get_item("work-claim-001")
        assert item["status"] == "claimed"
        assert item["claimed_by"] == "node-1"

    def test_claim_item_already_claimed(self, sqlite_backend):
        # Add and claim item
        work_data = {
            "work_type": "selfplay",
            "priority": 50,
            "config": {},
            "created_at": time.time(),
            "status": "claimed",
            "claimed_by": "node-0",
        }
        sqlite_backend.add_item("work-claimed", work_data)

        # Try to claim again
        result = sqlite_backend.claim_item("work-claimed", "node-1")
        assert result.success is False

    def test_start_item(self, sqlite_backend):
        # Add and claim item
        work_data = {
            "work_type": "selfplay",
            "priority": 50,
            "config": {},
            "created_at": time.time(),
            "status": "claimed",
        }
        sqlite_backend.add_item("work-start-001", work_data)

        # Start it
        result = sqlite_backend.start_item("work-start-001")
        assert result.success is True

        # Verify status changed
        item = sqlite_backend.get_item("work-start-001")
        assert item["status"] == "running"
        assert item["started_at"] > 0

    def test_complete_item(self, sqlite_backend):
        # Add claimed item
        work_data = {
            "work_type": "training",
            "priority": 100,
            "config": {},
            "created_at": time.time(),
            "status": "claimed",
        }
        sqlite_backend.add_item("work-complete-001", work_data)

        # Complete it
        result_data = {"model_path": "/path/to/model.pth"}
        result = sqlite_backend.complete_item("work-complete-001", result_data)
        assert result.success is True

        # Verify status changed
        item = sqlite_backend.get_item("work-complete-001")
        assert item["status"] == "completed"
        assert item["result"]["model_path"] == "/path/to/model.pth"

    def test_fail_item_permanent(self, sqlite_backend):
        work_data = {
            "work_type": "selfplay",
            "priority": 50,
            "config": {},
            "created_at": time.time(),
            "status": "running",
        }
        sqlite_backend.add_item("work-fail-001", work_data)

        result = sqlite_backend.fail_item(
            "work-fail-001", error="OOM error", permanent=True
        )
        assert result.success is True

        item = sqlite_backend.get_item("work-fail-001")
        assert item["status"] == "failed"
        assert item["error"] == "OOM error"

    def test_fail_item_retry(self, sqlite_backend):
        work_data = {
            "work_type": "selfplay",
            "priority": 50,
            "config": {},
            "created_at": time.time(),
            "status": "running",
        }
        sqlite_backend.add_item("work-fail-002", work_data)

        result = sqlite_backend.fail_item(
            "work-fail-002", error="Timeout", permanent=False
        )
        assert result.success is True

        item = sqlite_backend.get_item("work-fail-002")
        assert item["status"] == "pending"  # Reset for retry
        assert item["error"] == "Timeout"

    def test_get_stats(self, sqlite_backend):
        # Add items with different statuses
        for i, status in enumerate(["pending", "pending", "claimed", "completed"]):
            work_data = {
                "work_type": "selfplay",
                "priority": 50,
                "config": {},
                "created_at": time.time(),
                "status": status,
            }
            sqlite_backend.add_item(f"work-stats-{i}", work_data)

        stats = sqlite_backend.get_stats()
        assert stats["total"] == 4
        assert stats["pending"] == 2
        assert stats["claimed"] == 1
        assert stats["completed"] == 1
        assert stats["backend"] == "sqlite"

    def test_update_item_config(self, sqlite_backend):
        work_data = {
            "work_type": "selfplay",
            "priority": 50,
            "config": {"target_node": "old-node"},
            "created_at": time.time(),
            "status": "pending",
        }
        sqlite_backend.add_item("work-config-001", work_data)

        # Update config
        new_config = {"target_node": "new-node", "extra_field": True}
        result = sqlite_backend.update_item_config("work-config-001", new_config)
        assert result.success is True

        # Verify update
        item = sqlite_backend.get_item("work-config-001")
        assert item["config"]["target_node"] == "new-node"
        assert item["config"]["extra_field"] is True


class TestRaftBackend:
    """Tests for RaftBackend with fallback."""

    @pytest.fixture
    def mock_sqlite_backend(self):
        """Create a mock SQLite backend."""
        backend = MagicMock(spec=SQLiteBackend)
        backend.backend_type = BackendType.SQLITE
        backend.is_available = True
        return backend

    @pytest.fixture
    def mock_raft_queue(self):
        """Create a mock Raft work queue."""
        raft_wq = MagicMock()
        raft_wq.add_work.return_value = True
        raft_wq.claim_work.return_value = True
        raft_wq.start_work.return_value = True
        raft_wq.complete_work.return_value = True
        raft_wq.fail_work.return_value = True
        raft_wq.get_pending_work.return_value = []
        raft_wq.get_work.return_value = None
        raft_wq.get_queue_stats.return_value = {
            "total": 10,
            "pending": 5,
            "claimed": 2,
            "running": 1,
            "completed": 2,
            "is_leader": True,
        }
        return raft_wq

    def test_backend_type_raft(self, mock_sqlite_backend, mock_raft_queue):
        backend = RaftBackend(
            sqlite_backend=mock_sqlite_backend,
            get_raft_queue=lambda: mock_raft_queue,
        )
        assert backend.backend_type == BackendType.RAFT

    def test_backend_type_fallback(self, mock_sqlite_backend):
        backend = RaftBackend(
            sqlite_backend=mock_sqlite_backend,
            get_raft_queue=lambda: None,
        )
        # Trigger availability check
        _ = backend.is_available
        assert backend.backend_type == BackendType.SQLITE

    def test_add_item_raft(self, mock_sqlite_backend, mock_raft_queue):
        backend = RaftBackend(
            sqlite_backend=mock_sqlite_backend,
            get_raft_queue=lambda: mock_raft_queue,
        )

        result = backend.add_item("work-001", {"work_type": "selfplay"})
        assert result.success is True
        mock_raft_queue.add_work.assert_called_once()
        mock_sqlite_backend.add_item.assert_not_called()

    def test_add_item_fallback(self, mock_sqlite_backend):
        mock_sqlite_backend.add_item.return_value = BackendResult(success=True)

        backend = RaftBackend(
            sqlite_backend=mock_sqlite_backend,
            get_raft_queue=lambda: None,  # Raft unavailable
        )

        result = backend.add_item("work-001", {"work_type": "selfplay"})
        assert result.success is True
        assert result.fallback_used is True
        mock_sqlite_backend.add_item.assert_called_once()

    def test_claim_item_raft(self, mock_sqlite_backend, mock_raft_queue):
        backend = RaftBackend(
            sqlite_backend=mock_sqlite_backend,
            get_raft_queue=lambda: mock_raft_queue,
        )

        result = backend.claim_item("work-001", "node-1")
        assert result.success is True
        mock_raft_queue.claim_work.assert_called_once_with("work-001", "node-1")

    def test_get_stats_raft(self, mock_sqlite_backend, mock_raft_queue):
        backend = RaftBackend(
            sqlite_backend=mock_sqlite_backend,
            get_raft_queue=lambda: mock_raft_queue,
        )

        stats = backend.get_stats()
        assert stats["backend"] == "raft"
        assert stats["total"] == 10
        assert stats["pending"] == 5
        assert stats["is_leader"] is True

    def test_get_stats_fallback(self, mock_sqlite_backend):
        mock_sqlite_backend.get_stats.return_value = {
            "total": 5,
            "pending": 3,
            "backend": "sqlite",
        }

        backend = RaftBackend(
            sqlite_backend=mock_sqlite_backend,
            get_raft_queue=lambda: None,
        )

        stats = backend.get_stats()
        assert stats["fallback_active"] is True
        mock_sqlite_backend.get_stats.assert_called_once()

    def test_exception_triggers_fallback(self, mock_sqlite_backend, mock_raft_queue):
        mock_raft_queue.add_work.side_effect = Exception("Raft error")
        mock_sqlite_backend.add_item.return_value = BackendResult(success=True)

        backend = RaftBackend(
            sqlite_backend=mock_sqlite_backend,
            get_raft_queue=lambda: mock_raft_queue,
        )

        result = backend.add_item("work-001", {})
        assert result.success is True
        assert result.fallback_used is True
        mock_sqlite_backend.add_item.assert_called_once()


class TestCreateBackend:
    """Tests for create_backend factory function."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # Initialize schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS work_items (
                work_id TEXT PRIMARY KEY,
                work_type TEXT,
                priority INTEGER DEFAULT 50,
                config TEXT DEFAULT '{}',
                created_at REAL,
                claimed_at REAL DEFAULT 0.0,
                started_at REAL DEFAULT 0.0,
                completed_at REAL DEFAULT 0.0,
                status TEXT DEFAULT 'pending',
                claimed_by TEXT DEFAULT '',
                attempts INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 3,
                timeout_seconds REAL DEFAULT 3600.0,
                result TEXT DEFAULT '{}',
                error TEXT DEFAULT '',
                depends_on TEXT DEFAULT '[]'
            )
        """)
        conn.commit()
        conn.close()

        yield db_path
        db_path.unlink(missing_ok=True)

    def test_create_sqlite_backend_no_raft(self, temp_db):
        def get_connection():
            return sqlite3.connect(str(temp_db))

        backend = create_backend(
            db_path=temp_db,
            get_connection=get_connection,
            use_raft=False,
        )

        assert isinstance(backend, SQLiteBackend)
        assert backend.backend_type == BackendType.SQLITE

    def test_create_backend_raft_unavailable(self, temp_db):
        def get_connection():
            return sqlite3.connect(str(temp_db))

        # Patch in work_queue module where get_raft_work_queue is defined
        with patch(
            "app.coordination.work_queue.get_raft_work_queue",
            return_value=None,
        ):
            backend = create_backend(
                db_path=temp_db,
                get_connection=get_connection,
                use_raft=True,
            )

        # Should fall back to SQLite when Raft unavailable
        assert isinstance(backend, SQLiteBackend)

    def test_create_backend_raft_available(self, temp_db):
        def get_connection():
            return sqlite3.connect(str(temp_db))

        mock_raft = MagicMock()
        # Patch in work_queue module where get_raft_work_queue is defined
        with patch(
            "app.coordination.work_queue.get_raft_work_queue",
            return_value=mock_raft,
        ):
            backend = create_backend(
                db_path=temp_db,
                get_connection=get_connection,
                use_raft=True,
            )

        assert isinstance(backend, RaftBackend)
        assert backend.backend_type == BackendType.RAFT
