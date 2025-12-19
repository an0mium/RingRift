"""Tests for CoordinatorBase and related mixins."""

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.coordinator_base import (
    CoordinatorBase,
    CoordinatorProtocol,
    CoordinatorStatus,
    CoordinatorStats,
    SQLitePersistenceMixin,
    SingletonMixin,
    CallbackMixin,
    is_coordinator,
)


class TestCoordinatorStatus:
    """Tests for CoordinatorStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert CoordinatorStatus.INITIALIZING.value == "initializing"
        assert CoordinatorStatus.READY.value == "ready"
        assert CoordinatorStatus.RUNNING.value == "running"
        assert CoordinatorStatus.PAUSED.value == "paused"
        assert CoordinatorStatus.DRAINING.value == "draining"
        assert CoordinatorStatus.ERROR.value == "error"
        assert CoordinatorStatus.STOPPED.value == "stopped"


class TestCoordinatorStats:
    """Tests for CoordinatorStats dataclass."""

    def test_default_stats(self):
        """Test default stats values."""
        stats = CoordinatorStats()
        assert stats.status == CoordinatorStatus.INITIALIZING
        assert stats.uptime_seconds == 0.0
        assert stats.operations_count == 0
        assert stats.errors_count == 0
        assert stats.last_error is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = CoordinatorStats(
            status=CoordinatorStatus.RUNNING,
            operations_count=10,
            errors_count=2,
            extra={"custom_key": "custom_value"},
        )
        data = stats.to_dict()
        assert data["status"] == "running"
        assert data["operations_count"] == 10
        assert data["custom_key"] == "custom_value"


class SimpleCoordinator(CoordinatorBase):
    """Simple coordinator for testing."""

    def __init__(self, name: str = "test"):
        super().__init__(name=name)
        self.started = False
        self.stopped = False

    async def _do_start(self) -> None:
        self.started = True

    async def _do_stop(self) -> None:
        self.stopped = True

    async def get_stats(self) -> Dict[str, Any]:
        base_stats = await super().get_stats()
        base_stats["custom_field"] = "test_value"
        return base_stats


class TestCoordinatorBase:
    """Tests for CoordinatorBase class."""

    def test_initialization(self):
        """Test coordinator initialization."""
        coord = SimpleCoordinator("my_coordinator")
        assert coord.name == "my_coordinator"
        assert coord.status == CoordinatorStatus.INITIALIZING
        assert not coord.is_running
        assert coord.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_lifecycle(self):
        """Test coordinator lifecycle: initialize -> start -> stop."""
        coord = SimpleCoordinator()

        # Initialize
        await coord.initialize()
        assert coord.status == CoordinatorStatus.READY

        # Start
        await coord.start()
        assert coord.status == CoordinatorStatus.RUNNING
        assert coord.is_running
        assert coord.started
        assert coord.uptime_seconds >= 0

        # Stop
        await coord.stop()
        assert coord.status == CoordinatorStatus.STOPPED
        assert not coord.is_running
        assert coord.stopped

    @pytest.mark.asyncio
    async def test_pause_resume(self):
        """Test pausing and resuming."""
        coord = SimpleCoordinator()
        await coord.start()

        await coord.pause()
        assert coord.status == CoordinatorStatus.PAUSED
        assert coord.is_running  # Still running, just paused

        await coord.resume()
        assert coord.status == CoordinatorStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dependency_injection(self):
        """Test dependency injection pattern."""
        coord = SimpleCoordinator()

        # Set dependencies
        coord.set_dependency("work_queue", MagicMock())
        coord.set_dependency("notifier", MagicMock())

        assert coord.has_dependency("work_queue")
        assert coord.has_dependency("notifier")
        assert not coord.has_dependency("unknown")

        work_queue = coord.get_dependency("work_queue")
        assert work_queue is not None

        unknown = coord.get_dependency("unknown", default="default_value")
        assert unknown == "default_value"

    def test_record_operations(self):
        """Test operation and error recording."""
        coord = SimpleCoordinator()

        coord.record_operation()
        coord.record_operation()
        assert coord._operations_count == 2

        coord.record_error(Exception("Test error"))
        assert coord._errors_count == 1
        assert coord._last_error == "Test error"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting stats."""
        coord = SimpleCoordinator("stats_test")
        await coord.start()

        stats = await coord.get_stats()
        assert stats["name"] == "stats_test"
        assert stats["status"] == "running"
        assert stats["is_running"] is True
        assert stats["custom_field"] == "test_value"

    @pytest.mark.asyncio
    async def test_start_auto_initializes(self):
        """Test that start() auto-initializes if needed."""
        coord = SimpleCoordinator()
        assert coord.status == CoordinatorStatus.INITIALIZING

        await coord.start()
        assert coord.status == CoordinatorStatus.RUNNING
        assert coord.started


class SQLiteCoordinator(CoordinatorBase, SQLitePersistenceMixin):
    """Coordinator with SQLite persistence for testing."""

    def __init__(self, db_path: Path):
        super().__init__(name="sqlite_test")
        self.init_db(db_path)

    def _get_schema(self) -> str:
        return """
            CREATE TABLE IF NOT EXISTS test_data (
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            );
        """

    async def get_stats(self) -> Dict[str, Any]:
        return await super().get_stats()


class TestSQLitePersistenceMixin:
    """Tests for SQLitePersistenceMixin."""

    def test_db_initialization(self):
        """Test database initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            coord = SQLiteCoordinator(db_path)

            assert db_path.exists()

            # Verify table was created
            conn = coord._get_connection()
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_data'"
            )
            assert cursor.fetchone() is not None

    def test_thread_local_connection(self):
        """Test that connections are thread-local."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            coord = SQLiteCoordinator(db_path)

            conn1 = coord._get_connection()
            conn2 = coord._get_connection()
            assert conn1 is conn2  # Same thread, same connection

    def test_wal_mode(self):
        """Test that WAL mode is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            coord = SQLiteCoordinator(db_path)

            conn = coord._get_connection()
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"


class SingletonCoordinator(CoordinatorBase, SingletonMixin):
    """Singleton coordinator for testing."""

    def __init__(self, value: str = "default"):
        super().__init__(name="singleton_test")
        self.value = value

    @classmethod
    def get_instance(cls, value: str = "default") -> "SingletonCoordinator":
        return cls._get_or_create_instance(value)

    async def get_stats(self) -> Dict[str, Any]:
        return await super().get_stats()


class TestSingletonMixin:
    """Tests for SingletonMixin."""

    def teardown_method(self):
        """Clean up singleton instances after each test."""
        SingletonCoordinator._clear_instance()

    def test_singleton_pattern(self):
        """Test that only one instance is created."""
        inst1 = SingletonCoordinator.get_instance("first")
        inst2 = SingletonCoordinator.get_instance("second")

        assert inst1 is inst2
        assert inst1.value == "first"  # First value is preserved

    def test_clear_instance(self):
        """Test clearing the singleton instance."""
        inst1 = SingletonCoordinator.get_instance("first")
        SingletonCoordinator._clear_instance()

        inst2 = SingletonCoordinator.get_instance("second")
        assert inst1 is not inst2
        assert inst2.value == "second"


class CallbackCoordinator(CoordinatorBase, CallbackMixin):
    """Coordinator with callbacks for testing."""

    def __init__(self):
        super().__init__(name="callback_test")
        self.__init_callbacks__()

    async def get_stats(self) -> Dict[str, Any]:
        return await super().get_stats()


class TestCallbackMixin:
    """Tests for CallbackMixin."""

    @pytest.mark.asyncio
    async def test_sync_callback(self):
        """Test synchronous callback invocation."""
        coord = CallbackCoordinator()
        results = []

        def on_event(value):
            results.append(value)
            return "sync_result"

        coord.register_callback("test_event", on_event)
        returned = await coord.invoke_callbacks("test_event", "test_value")

        assert results == ["test_value"]
        assert returned == ["sync_result"]

    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Test asynchronous callback invocation."""
        coord = CallbackCoordinator()
        results = []

        async def on_event(value):
            results.append(value)
            return "async_result"

        coord.register_callback("test_event", on_event)
        returned = await coord.invoke_callbacks("test_event", "test_value")

        assert results == ["test_value"]
        assert returned == ["async_result"]

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        """Test multiple callbacks for same event."""
        coord = CallbackCoordinator()
        results = []

        coord.register_callback("test_event", lambda x: results.append(f"1:{x}"))
        coord.register_callback("test_event", lambda x: results.append(f"2:{x}"))

        await coord.invoke_callbacks("test_event", "value")

        assert results == ["1:value", "2:value"]

    @pytest.mark.asyncio
    async def test_unregister_callback(self):
        """Test unregistering a callback."""
        coord = CallbackCoordinator()
        results = []

        def my_callback(x):
            results.append(x)

        coord.register_callback("test_event", my_callback)
        coord.unregister_callback("test_event", my_callback)

        await coord.invoke_callbacks("test_event", "value")

        assert results == []

    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """Test that errors in callbacks don't break invocation."""
        coord = CallbackCoordinator()
        results = []

        def failing_callback(x):
            raise ValueError("Intentional error")

        def working_callback(x):
            results.append(x)
            return "worked"

        coord.register_callback("test_event", failing_callback)
        coord.register_callback("test_event", working_callback)

        returned = await coord.invoke_callbacks("test_event", "value")

        assert results == ["value"]
        assert returned == [None, "worked"]  # First is None due to error


class TestCoordinatorProtocol:
    """Tests for CoordinatorProtocol."""

    def test_protocol_check(self):
        """Test that protocol checking works."""
        coord = SimpleCoordinator()
        assert isinstance(coord, CoordinatorProtocol)

    def test_is_coordinator_function(self):
        """Test is_coordinator helper function."""
        coord = SimpleCoordinator()
        assert is_coordinator(coord)

        not_coord = {"status": "running"}
        assert not is_coordinator(not_coord)
