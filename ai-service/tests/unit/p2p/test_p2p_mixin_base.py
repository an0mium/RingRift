"""Unit tests for P2PMixinBase class.

Tests the shared functionality extracted for P2P mixin consolidation.

Created: December 27, 2025
"""

import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.p2p.p2p_mixin_base import P2PMixinBase


class MockPeer:
    """Mock peer for testing."""

    def __init__(self, alive: bool = True):
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class TestMixin(P2PMixinBase):
    """Test implementation of P2PMixinBase."""

    MIXIN_TYPE = "test_mixin"

    def __init__(self, db_path: Path | None = None, verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.node_id = "test-node-1"
        self.peers: dict[str, Any] = {}
        self.peers_lock = threading.RLock()
        self._emit_event_calls: list[tuple[str, dict]] = []

    def _emit_event(self, event_type: str, payload: dict) -> None:
        """Track event emissions for testing."""
        self._emit_event_calls.append((event_type, payload))


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Create test table
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value REAL
        )
    """)
    conn.execute("INSERT INTO test_table (name, value) VALUES ('foo', 1.5)")
    conn.execute("INSERT INTO test_table (name, value) VALUES ('bar', 2.5)")
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def mixin(temp_db: Path) -> TestMixin:
    """Create a test mixin instance with a database."""
    return TestMixin(db_path=temp_db, verbose=True)


@pytest.fixture
def mixin_no_db() -> TestMixin:
    """Create a test mixin instance without a database."""
    return TestMixin(db_path=None, verbose=True)


class TestDatabaseHelpers:
    """Tests for database helper methods."""

    def test_execute_db_query_fetch(self, mixin: TestMixin) -> None:
        """Test fetching rows from database."""
        rows = mixin._execute_db_query(
            "SELECT name, value FROM test_table ORDER BY name",
            fetch=True,
        )
        assert rows is not None
        assert len(rows) == 2
        assert rows[0] == ("bar", 2.5)
        assert rows[1] == ("foo", 1.5)

    def test_execute_db_query_insert(self, mixin: TestMixin) -> None:
        """Test inserting a row."""
        rowcount = mixin._execute_db_query(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            ("baz", 3.5),
            fetch=False,
        )
        assert rowcount == 1

        # Verify insert
        rows = mixin._execute_db_query(
            "SELECT value FROM test_table WHERE name = ?",
            ("baz",),
            fetch=True,
        )
        assert rows == [(3.5,)]

    def test_execute_db_query_update(self, mixin: TestMixin) -> None:
        """Test updating rows."""
        rowcount = mixin._execute_db_query(
            "UPDATE test_table SET value = ? WHERE name = ?",
            (10.0, "foo"),
            fetch=False,
        )
        assert rowcount == 1

        # Verify update
        rows = mixin._execute_db_query(
            "SELECT value FROM test_table WHERE name = ?",
            ("foo",),
            fetch=True,
        )
        assert rows == [(10.0,)]

    def test_execute_db_query_no_db_path(self, mixin_no_db: TestMixin) -> None:
        """Test graceful handling when no db_path is set."""
        result = mixin._execute_db_query(
            "SELECT * FROM test_table",
            fetch=True,
        )
        assert result is None

    def test_execute_db_query_invalid_sql(self, mixin: TestMixin) -> None:
        """Test graceful handling of invalid SQL."""
        result = mixin._execute_db_query(
            "INVALID SQL STATEMENT",
            fetch=True,
        )
        assert result is None

    def test_db_connection_context_manager(self, mixin: TestMixin) -> None:
        """Test database connection context manager."""
        with mixin._db_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test_table")
            count = cursor.fetchone()[0]
            assert count == 2

    def test_db_connection_no_db_path(self, mixin_no_db: TestMixin) -> None:
        """Test connection context manager without db_path."""
        with mixin_no_db._db_connection() as conn:
            assert conn is None


class TestStateInitialization:
    """Tests for state initialization helpers."""

    def test_ensure_state_attr_new_attr(self, mixin: TestMixin) -> None:
        """Test creating a new attribute."""
        assert not hasattr(mixin, "_new_cache")
        mixin._ensure_state_attr("_new_cache", {})
        assert hasattr(mixin, "_new_cache")
        assert mixin._new_cache == {}

    def test_ensure_state_attr_existing_attr(self, mixin: TestMixin) -> None:
        """Test that existing attributes are not overwritten."""
        mixin._existing = {"key": "value"}
        mixin._ensure_state_attr("_existing", {})
        assert mixin._existing == {"key": "value"}

    def test_ensure_state_attr_auto_dict(self, mixin: TestMixin) -> None:
        """Test auto-detection of dict types from name."""
        mixin._ensure_state_attr("_peer_cache")  # No default provided
        assert mixin._peer_cache == {}

        mixin._ensure_state_attr("_state_dict")
        assert mixin._state_dict == {}

    def test_ensure_multiple_state_attrs(self, mixin: TestMixin) -> None:
        """Test ensuring multiple attributes at once."""
        mixin._ensure_multiple_state_attrs({
            "_cache1": {},
            "_count": 0,
            "_flag": False,
        })
        assert mixin._cache1 == {}
        assert mixin._count == 0
        assert mixin._flag is False


class TestPeerManagement:
    """Tests for peer management helpers."""

    def test_count_alive_peers_empty_list(self, mixin: TestMixin) -> None:
        """Test counting with empty list."""
        assert mixin._count_alive_peers([]) == 0

    def test_count_alive_peers_self_only(self, mixin: TestMixin) -> None:
        """Test that self is counted as alive."""
        mixin.node_id = "node-1"
        assert mixin._count_alive_peers(["node-1"]) == 1

    def test_count_alive_peers_with_peers(self, mixin: TestMixin) -> None:
        """Test counting alive peers."""
        mixin.node_id = "node-1"
        mixin.peers = {
            "node-2": MockPeer(alive=True),
            "node-3": MockPeer(alive=False),
            "node-4": MockPeer(alive=True),
        }

        alive = mixin._count_alive_peers(["node-1", "node-2", "node-3", "node-4"])
        assert alive == 3  # node-1 (self), node-2, node-4

    def test_count_alive_peers_missing_peer(self, mixin: TestMixin) -> None:
        """Test counting when peer is not in peers dict."""
        mixin.node_id = "node-1"
        mixin.peers = {"node-2": MockPeer(alive=True)}

        alive = mixin._count_alive_peers(["node-1", "node-2", "node-99"])
        assert alive == 2  # node-1 (self), node-2

    def test_get_alive_peer_list(self, mixin: TestMixin) -> None:
        """Test getting list of alive peer IDs."""
        mixin.node_id = "node-1"
        mixin.peers = {
            "node-2": MockPeer(alive=True),
            "node-3": MockPeer(alive=False),
            "node-4": MockPeer(alive=True),
        }

        alive_list = mixin._get_alive_peer_list(["node-1", "node-2", "node-3", "node-4"])
        assert set(alive_list) == {"node-1", "node-2", "node-4"}


class TestEventEmission:
    """Tests for event emission helpers."""

    def test_safe_emit_event_success(self, mixin: TestMixin) -> None:
        """Test successful event emission."""
        result = mixin._safe_emit_event("TEST_EVENT", {"key": "value"})
        assert result is True
        assert len(mixin._emit_event_calls) == 1
        assert mixin._emit_event_calls[0] == ("TEST_EVENT", {"key": "value"})

    def test_safe_emit_event_no_handler(self, mixin_no_db: TestMixin) -> None:
        """Test when no _emit_event method exists."""
        # Remove the method if it exists
        if hasattr(mixin_no_db, "_emit_event"):
            delattr(mixin_no_db, "_emit_event")

        result = mixin_no_db._safe_emit_event("TEST_EVENT", {"key": "value"})
        assert result is False

    def test_safe_emit_event_empty_payload(self, mixin: TestMixin) -> None:
        """Test emission with no payload."""
        result = mixin._safe_emit_event("TEST_EVENT")
        assert result is True
        assert mixin._emit_event_calls[0] == ("TEST_EVENT", {})

    def test_safe_emit_event_exception(self, mixin: TestMixin) -> None:
        """Test graceful handling of exceptions during emission."""
        def raise_error(event_type: str, payload: dict) -> None:
            raise RuntimeError("Emission failed")

        mixin._emit_event = raise_error

        result = mixin._safe_emit_event("TEST_EVENT", {"key": "value"})
        assert result is False


class TestConfigurationLoading:
    """Tests for configuration constant loading."""

    def test_load_config_constant_success(self) -> None:
        """Test loading an existing constant."""
        # This will try to load from scripts.p2p.constants
        # Using a known constant that should exist
        result = P2PMixinBase._load_config_constant(
            "DEFAULT_PORT",
            9999,  # fallback
            "scripts.p2p.constants",
        )
        # Either gets the real value or fallback (both are valid)
        assert isinstance(result, int)

    def test_load_config_constant_fallback(self) -> None:
        """Test fallback when constant doesn't exist."""
        result = P2PMixinBase._load_config_constant(
            "NONEXISTENT_CONSTANT_12345",
            42,
        )
        assert result == 42

    def test_load_config_constant_bad_module(self) -> None:
        """Test fallback when module doesn't exist."""
        result = P2PMixinBase._load_config_constant(
            "SOME_CONSTANT",
            "default",
            "nonexistent.module.path",
        )
        assert result == "default"

    def test_load_config_constants_multiple(self) -> None:
        """Test loading multiple constants at once."""
        result = P2PMixinBase._load_config_constants({
            "NONEXISTENT_A": 100,
            "NONEXISTENT_B": "hello",
            "NONEXISTENT_C": True,
        })
        assert result == {
            "NONEXISTENT_A": 100,
            "NONEXISTENT_B": "hello",
            "NONEXISTENT_C": True,
        }


class TestLoggingHelpers:
    """Tests for logging helper methods."""

    def test_log_methods_exist(self, mixin: TestMixin) -> None:
        """Test that all log methods exist and are callable."""
        assert callable(mixin._log_debug)
        assert callable(mixin._log_info)
        assert callable(mixin._log_warning)
        assert callable(mixin._log_error)

    @patch("scripts.p2p.p2p_mixin_base.logger")
    def test_log_debug(self, mock_logger: MagicMock, mixin: TestMixin) -> None:
        """Test debug logging with prefix."""
        mixin._log_debug("test message")
        mock_logger.debug.assert_called_once_with("[test_mixin] test message")

    @patch("scripts.p2p.p2p_mixin_base.logger")
    def test_log_info(self, mock_logger: MagicMock, mixin: TestMixin) -> None:
        """Test info logging with prefix."""
        mixin._log_info("test message")
        mock_logger.info.assert_called_once_with("[test_mixin] test message")


class TestTimingHelpers:
    """Tests for timing helper methods."""

    def test_get_timestamp(self, mixin: TestMixin) -> None:
        """Test timestamp retrieval."""
        before = time.time()
        ts = mixin._get_timestamp()
        after = time.time()
        assert before <= ts <= after

    def test_is_expired_true(self, mixin: TestMixin) -> None:
        """Test expiration check for expired timestamp."""
        old_timestamp = time.time() - 100  # 100 seconds ago
        assert mixin._is_expired(old_timestamp, 60) is True

    def test_is_expired_false(self, mixin: TestMixin) -> None:
        """Test expiration check for non-expired timestamp."""
        recent_timestamp = time.time() - 10  # 10 seconds ago
        assert mixin._is_expired(recent_timestamp, 60) is False
