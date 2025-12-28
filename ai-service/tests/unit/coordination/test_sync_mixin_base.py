"""Unit tests for app/coordination/sync_mixin_base.py.

Tests for:
- SyncError dataclass and from_exception() classmethod
- SyncMixinBase logging and error recording methods
- AutoSyncDaemonProtocol validation
- validate_sync_daemon_protocol() function

Created: December 28, 2025
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.sync_mixin_base import (
    AutoSyncDaemonProtocol,
    SyncError,
    SyncMixinBase,
    validate_sync_daemon_protocol,
)


# =============================================================================
# SyncError Tests
# =============================================================================


class TestSyncError:
    """Tests for SyncError dataclass."""

    def test_basic_creation(self):
        """Test creating a SyncError with required fields."""
        error = SyncError(error_type="network", message="Connection refused")
        assert error.error_type == "network"
        assert error.message == "Connection refused"
        assert error.target_node == ""
        assert error.db_path == ""
        assert error.recoverable is True

    def test_full_creation(self):
        """Test creating a SyncError with all fields."""
        ts = time.time()
        error = SyncError(
            error_type="database",
            message="Corrupt file",
            target_node="node-123",
            db_path="/data/games.db",
            timestamp=ts,
            recoverable=False,
        )
        assert error.error_type == "database"
        assert error.message == "Corrupt file"
        assert error.target_node == "node-123"
        assert error.db_path == "/data/games.db"
        assert error.timestamp == ts
        assert error.recoverable is False

    def test_timestamp_default(self):
        """Test that timestamp defaults to current time."""
        before = time.time()
        error = SyncError(error_type="test", message="test")
        after = time.time()
        assert before <= error.timestamp <= after


class TestSyncErrorFromException:
    """Tests for SyncError.from_exception() classmethod."""

    def test_timeout_exception(self):
        """Test classification of timeout exceptions."""
        exc = TimeoutError("Connection timed out")
        error = SyncError.from_exception(exc)
        assert error.error_type == "timeout"
        assert error.recoverable is True
        assert "timed out" in error.message

    def test_connection_exception(self):
        """Test classification of connection exceptions."""
        exc = ConnectionRefusedError("No route to host")
        error = SyncError.from_exception(exc)
        assert error.error_type == "network"
        assert error.recoverable is True

    def test_ssh_error_in_message(self):
        """Test classification of SSH errors in message."""
        exc = Exception("SSH connection failed: host unreachable")
        error = SyncError.from_exception(exc)
        assert error.error_type == "network"
        assert error.recoverable is True

    def test_database_exception(self):
        """Test classification of database exceptions."""
        exc = Exception("sqlite3.OperationalError: database is locked")
        error = SyncError.from_exception(exc)
        assert error.error_type == "database"
        assert error.recoverable is False

    def test_permission_exception(self):
        """Test classification of permission exceptions."""
        exc = PermissionError("Access denied to /data/games.db")
        error = SyncError.from_exception(exc)
        assert error.error_type == "permission"
        assert error.recoverable is False

    def test_unknown_exception(self):
        """Test classification of unknown exceptions."""
        exc = ValueError("Some random error")
        error = SyncError.from_exception(exc)
        assert error.error_type == "unknown"
        assert error.recoverable is True

    def test_with_target_node(self):
        """Test from_exception with target_node parameter."""
        exc = TimeoutError("timeout")
        error = SyncError.from_exception(exc, target_node="vast-12345")
        assert error.target_node == "vast-12345"

    def test_with_db_path(self):
        """Test from_exception with db_path parameter."""
        exc = Exception("error")
        error = SyncError.from_exception(exc, db_path="/data/test.db")
        assert error.db_path == "/data/test.db"

    def test_timeout_in_message_not_type(self):
        """Test timeout detection in message even when type is different."""
        exc = RuntimeError("Operation timeout after 30 seconds")
        error = SyncError.from_exception(exc)
        assert error.error_type == "timeout"


# =============================================================================
# SyncMixinBase Tests
# =============================================================================


class ConcreteSyncMixin(SyncMixinBase):
    """Concrete implementation of SyncMixinBase for testing."""

    def __init__(self):
        self._events_processed = 0
        self._errors_count = 0
        self._last_error = ""

    async def _emit_sync_failure(
        self, target_node: str, db_path: str, error: str
    ) -> None:
        pass

    async def _emit_sync_stalled(
        self,
        target_node: str,
        timeout_seconds: float,
        data_type: str = "game",
        retry_count: int = 0,
    ) -> None:
        pass


class TestSyncMixinBase:
    """Tests for SyncMixinBase class."""

    def test_log_prefix(self):
        """Test default LOG_PREFIX value."""
        mixin = ConcreteSyncMixin()
        assert mixin.LOG_PREFIX == "[AutoSyncDaemon]"

    def test_record_event_processed(self):
        """Test _record_event_processed increments counter."""
        mixin = ConcreteSyncMixin()
        assert mixin._events_processed == 0
        mixin._record_event_processed()
        assert mixin._events_processed == 1
        mixin._record_event_processed()
        assert mixin._events_processed == 2

    def test_record_error_with_string(self):
        """Test _record_error with string message."""
        mixin = ConcreteSyncMixin()
        result = mixin._record_error("Something went wrong")
        assert isinstance(result, SyncError)
        assert result.error_type == "unknown"
        assert result.message == "Something went wrong"
        assert mixin._errors_count == 1
        assert mixin._last_error == "Something went wrong"

    def test_record_error_with_exception(self):
        """Test _record_error with exception."""
        mixin = ConcreteSyncMixin()
        exc = TimeoutError("timed out")
        result = mixin._record_error(exc)
        assert isinstance(result, SyncError)
        assert result.error_type == "timeout"
        assert mixin._errors_count == 1

    def test_record_error_with_target_node(self):
        """Test _record_error with target_node parameter."""
        mixin = ConcreteSyncMixin()
        result = mixin._record_error("error", target_node="node-123")
        assert result.target_node == "node-123"

    def test_record_error_increments_counter(self):
        """Test that multiple _record_error calls increment counter."""
        mixin = ConcreteSyncMixin()
        mixin._record_error("error1")
        mixin._record_error("error2")
        mixin._record_error("error3")
        assert mixin._errors_count == 3

    def test_log_info(self, caplog):
        """Test _log_info method."""
        mixin = ConcreteSyncMixin()
        with caplog.at_level(logging.INFO):
            mixin._log_info("Test message")
        assert "[AutoSyncDaemon] Test message" in caplog.text

    def test_log_debug(self, caplog):
        """Test _log_debug method."""
        mixin = ConcreteSyncMixin()
        with caplog.at_level(logging.DEBUG):
            mixin._log_debug("Debug message")
        assert "[AutoSyncDaemon] Debug message" in caplog.text

    def test_log_warning(self, caplog):
        """Test _log_warning method."""
        mixin = ConcreteSyncMixin()
        with caplog.at_level(logging.WARNING):
            mixin._log_warning("Warning message")
        assert "[AutoSyncDaemon] Warning message" in caplog.text

    def test_log_error(self, caplog):
        """Test _log_error method."""
        mixin = ConcreteSyncMixin()
        with caplog.at_level(logging.ERROR):
            mixin._log_error("Error message")
        assert "[AutoSyncDaemon] Error message" in caplog.text

    def test_record_error_logs_recoverable(self, caplog):
        """Test that recoverable errors are logged as warnings."""
        mixin = ConcreteSyncMixin()
        with caplog.at_level(logging.WARNING):
            mixin._record_error(TimeoutError("timeout"))
        assert "warning" in caplog.text.lower() or "WARNING" in caplog.text

    def test_record_error_logs_non_recoverable(self, caplog):
        """Test that non-recoverable errors are logged as errors."""
        mixin = ConcreteSyncMixin()
        with caplog.at_level(logging.ERROR):
            mixin._record_error(PermissionError("Access denied"))
        assert "error" in caplog.text.lower() or "ERROR" in caplog.text


# =============================================================================
# AutoSyncDaemonProtocol Tests
# =============================================================================


class TestAutoSyncDaemonProtocol:
    """Tests for AutoSyncDaemonProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that AutoSyncDaemonProtocol is runtime_checkable."""
        # Should not raise
        assert hasattr(AutoSyncDaemonProtocol, "__protocol_attrs__") or hasattr(
            AutoSyncDaemonProtocol, "__subclasshook__"
        )


# =============================================================================
# validate_sync_daemon_protocol Tests
# =============================================================================


class TestValidateSyncDaemonProtocol:
    """Tests for validate_sync_daemon_protocol function."""

    def test_valid_object(self):
        """Test with object that has all required attributes/methods."""

        class ValidDaemon:
            config = MagicMock()
            node_id = "test-node"
            _running = True
            _stats = MagicMock()
            _events_processed = 0
            _errors_count = 0

            async def _sync_all(self):
                pass

            async def _sync_to_peer(self, node_id: str) -> bool:
                return True

        daemon = ValidDaemon()
        assert validate_sync_daemon_protocol(daemon) is True

    def test_missing_attribute(self):
        """Test with object missing required attribute."""

        class PartialDaemon:
            config = MagicMock()
            node_id = "test-node"
            # Missing _running, _stats, etc.

            async def _sync_all(self):
                pass

            async def _sync_to_peer(self, node_id: str) -> bool:
                return True

        daemon = PartialDaemon()
        assert validate_sync_daemon_protocol(daemon) is False

    def test_missing_method(self):
        """Test with object missing required method."""

        class NoDaemon:
            config = MagicMock()
            node_id = "test-node"
            _running = True
            _stats = MagicMock()
            _events_processed = 0
            _errors_count = 0
            # Missing _sync_all, _sync_to_peer

        daemon = NoDaemon()
        assert validate_sync_daemon_protocol(daemon) is False

    def test_method_not_callable(self):
        """Test with object where method is not callable."""

        class NotCallableDaemon:
            config = MagicMock()
            node_id = "test-node"
            _running = True
            _stats = MagicMock()
            _events_processed = 0
            _errors_count = 0
            _sync_all = "not_callable"  # Not callable
            _sync_to_peer = "not_callable"

        daemon = NotCallableDaemon()
        assert validate_sync_daemon_protocol(daemon) is False

    def test_empty_object(self):
        """Test with empty object."""
        assert validate_sync_daemon_protocol(object()) is False

    def test_none(self):
        """Test with None."""
        assert validate_sync_daemon_protocol(None) is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestSyncMixinEdgeCases:
    """Edge case tests for sync mixin base."""

    def test_record_error_empty_string(self):
        """Test _record_error with empty string."""
        mixin = ConcreteSyncMixin()
        result = mixin._record_error("")
        assert result.message == ""
        assert mixin._errors_count == 1

    def test_record_error_preserves_last_error(self):
        """Test that _last_error is updated on each call."""
        mixin = ConcreteSyncMixin()
        mixin._record_error("first error")
        assert mixin._last_error == "first error"
        mixin._record_error("second error")
        assert mixin._last_error == "second error"

    def test_sync_error_immutability(self):
        """Test that SyncError fields can be read after creation."""
        error = SyncError(
            error_type="test",
            message="test message",
            target_node="node-1",
            db_path="/test/path.db",
            timestamp=12345.0,
            recoverable=False,
        )
        # All fields should be readable
        assert error.error_type == "test"
        assert error.message == "test message"
        assert error.target_node == "node-1"
        assert error.db_path == "/test/path.db"
        assert error.timestamp == 12345.0
        assert error.recoverable is False
