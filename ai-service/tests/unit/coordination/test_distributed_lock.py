"""Tests for app.coordination.distributed_lock module.

Comprehensive tests for distributed locking infrastructure including:
- LockProtocol interface
- DistributedLock class (Redis + file fallback)
- Convenience functions (training_lock, acquire_training_lock, etc.)
- Stale lock cleanup
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.distributed_lock import (
    # Constants
    DEFAULT_ACQUIRE_TIMEOUT,
    DEFAULT_LOCK_TIMEOUT,
    LOCK_DIR,
    # Main class
    DistributedLock,
    # Protocol
    LockProtocol,
    # Functions
    acquire_training_lock,
    cleanup_stale_locks,
    get_appropriate_lock,
    release_training_lock,
    training_lock,
)


class TestLockProtocol:
    """Tests for LockProtocol interface."""

    def test_lock_protocol_runtime_checkable(self):
        """Test LockProtocol is runtime checkable."""
        # DistributedLock should satisfy the protocol
        lock = DistributedLock("test_protocol", use_redis=False)
        assert isinstance(lock, LockProtocol)

    def test_lock_protocol_methods(self):
        """Test LockProtocol has required methods."""
        # Check protocol has required methods
        assert hasattr(LockProtocol, "acquire")
        assert hasattr(LockProtocol, "release")
        assert hasattr(LockProtocol, "name")


class TestDistributedLockInit:
    """Tests for DistributedLock initialization."""

    def test_default_initialization(self):
        """Test DistributedLock with default values."""
        lock = DistributedLock("test_lock", use_redis=False)
        assert lock.name == "test_lock"
        assert lock.lock_timeout == DEFAULT_LOCK_TIMEOUT
        assert lock._acquired is False
        assert lock._file_fd is None

    def test_custom_timeout(self):
        """Test DistributedLock with custom timeout."""
        lock = DistributedLock("test_lock", lock_timeout=7200, use_redis=False)
        assert lock.lock_timeout == 7200

    def test_lock_id_format(self):
        """Test lock ID contains hostname, pid, and uuid."""
        lock = DistributedLock("test_lock", use_redis=False)
        parts = lock._lock_id.split(":")
        assert len(parts) == 3
        # Hostname
        assert len(parts[0]) > 0
        # PID
        assert parts[1].isdigit()
        # UUID hex
        assert len(parts[2]) == 8

    def test_force_file_based_lock(self):
        """Test forcing file-based lock by disabling Redis."""
        lock = DistributedLock("test_lock", use_redis=False)
        assert lock._redis_client is None
        assert lock._use_redis is False


class TestDistributedLockFileBased:
    """Tests for file-based locking functionality."""

    def test_acquire_and_release(self):
        """Test basic acquire and release cycle."""
        lock = DistributedLock("test_acquire", use_redis=False)
        try:
            assert lock.acquire(timeout=5)
            assert lock._acquired is True
            assert lock.is_held() is True
        finally:
            lock.release()
            assert lock._acquired is False
            assert lock.is_held() is False

    def test_acquire_non_blocking_success(self):
        """Test non-blocking acquire when lock is free."""
        lock = DistributedLock("test_nonblock", use_redis=False)
        try:
            assert lock.acquire(blocking=False) is True
        finally:
            lock.release()

    def test_acquire_non_blocking_failure(self):
        """Test non-blocking acquire when lock is held."""
        lock1 = DistributedLock("test_nonblock_fail", use_redis=False)
        lock2 = DistributedLock("test_nonblock_fail", use_redis=False)
        try:
            assert lock1.acquire()
            # Second lock should fail immediately with non-blocking
            assert lock2.acquire(blocking=False) is False
        finally:
            lock1.release()

    def test_acquire_already_acquired(self):
        """Test acquire returns True if already acquired."""
        lock = DistributedLock("test_reacquire", use_redis=False)
        try:
            assert lock.acquire()
            # Second acquire should return True immediately
            assert lock.acquire() is True
        finally:
            lock.release()

    def test_release_without_acquire(self):
        """Test release without prior acquire is safe."""
        lock = DistributedLock("test_no_release", use_redis=False)
        lock.release()  # Should not raise
        assert lock._acquired is False

    def test_is_locked_when_held(self):
        """Test is_locked() when lock is held."""
        lock = DistributedLock("test_is_locked", use_redis=False)
        try:
            lock.acquire()
            assert lock.is_locked() is True
        finally:
            lock.release()

    def test_is_locked_when_free(self):
        """Test is_locked() when lock is free."""
        lock = DistributedLock("test_is_locked_free", use_redis=False)
        # Acquire and release to create lock file, then check
        lock.acquire()
        lock.release()
        # After release, should not be locked
        assert lock.is_locked() is False

    def test_lock_path_creation(self):
        """Test lock file path is created correctly."""
        lock = DistributedLock("test:path:lock", use_redis=False)
        path = lock._get_lock_path()
        assert "test_path_lock.lock" in str(path)
        assert LOCK_DIR in path.parents or path.parent == LOCK_DIR

    def test_timeout_returns_false(self):
        """Test acquire returns False on timeout."""
        lock1 = DistributedLock("test_timeout", use_redis=False)
        lock2 = DistributedLock("test_timeout", use_redis=False)
        try:
            lock1.acquire()
            # Second lock should timeout (use very short timeout)
            start = time.time()
            result = lock2.acquire(timeout=1, blocking=True)
            elapsed = time.time() - start
            assert result is False
            assert elapsed >= 1.0
        finally:
            lock1.release()


class TestDistributedLockContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_success(self):
        """Test context manager acquires and releases lock."""
        lock = DistributedLock("test_cm", use_redis=False)
        with lock:
            assert lock.is_held() is True
        assert lock.is_held() is False

    def test_context_manager_releases_on_exception(self):
        """Test context manager releases lock on exception."""
        lock = DistributedLock("test_cm_exception", use_redis=False)
        try:
            with lock:
                assert lock.is_held() is True
                raise ValueError("test error")
        except ValueError:
            pass
        assert lock.is_held() is False

    def test_context_manager_raises_on_failure(self):
        """Test context manager raises RuntimeError on acquire failure."""
        lock1 = DistributedLock("test_cm_fail", use_redis=False)
        lock2 = DistributedLock("test_cm_fail", use_redis=False)
        try:
            lock1.acquire()
            with patch.object(lock2, "acquire", return_value=False):
                with pytest.raises(RuntimeError, match="Could not acquire lock"):
                    with lock2:
                        pass
        finally:
            lock1.release()


class TestDistributedLockExpiry:
    """Tests for lock expiry handling."""

    def test_expired_lock_takeover(self):
        """Test taking over an expired lock."""
        lock_name = f"test_expiry_{time.time()}"
        lock1 = DistributedLock(lock_name, lock_timeout=1, use_redis=False)

        try:
            # Acquire first lock
            lock1.acquire()
            lock_path = lock1._get_lock_path()

            # Manually modify lock file to make it expired
            with open(lock_path, 'w') as f:
                f.write(f"{lock1._lock_id}\n")
                f.write(f"{time.time() - 100}\n")  # 100 seconds ago
                f.write("1\n")  # 1 second timeout

            # Release without cleanup
            lock1._acquired = False
            lock1._file_fd = None

            # Second lock should be able to take over
            lock2 = DistributedLock(lock_name, use_redis=False)
            assert lock2._is_file_lock_expired() is True
            assert lock2.acquire(timeout=2)
            lock2.release()
        finally:
            # Cleanup
            with patch.object(lock1, "_file_fd", None):
                lock1.release()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_acquire_training_lock_success(self):
        """Test acquire_training_lock when successful."""
        lock = acquire_training_lock("test_config", timeout=5)
        assert lock is not None
        assert lock.is_held() is True
        release_training_lock(lock)
        assert lock.is_held() is False

    def test_acquire_training_lock_failure(self):
        """Test acquire_training_lock returns None on failure."""
        # First acquire the lock
        first_lock = DistributedLock("training:test_fail_config", use_redis=False)
        first_lock.acquire()
        try:
            # Second acquire should fail
            result = acquire_training_lock("test_fail_config", timeout=1)
            assert result is None
        finally:
            first_lock.release()

    def test_release_training_lock_with_none(self):
        """Test release_training_lock handles None."""
        release_training_lock(None)  # Should not raise

    def test_training_lock_context_manager_success(self):
        """Test training_lock context manager success path."""
        with training_lock("test_cm_config", timeout=5) as lock:
            assert lock is not None
            assert lock.is_held() is True
        # After context, lock should be released
        assert not lock.is_held()

    def test_training_lock_context_manager_failure(self):
        """Test training_lock context manager when lock unavailable."""
        # First acquire the lock
        first_lock = DistributedLock("training:test_cm_fail", use_redis=False)
        first_lock.acquire()
        try:
            with training_lock("test_cm_fail", timeout=1) as lock:
                assert lock is None
        finally:
            first_lock.release()

    def test_get_appropriate_lock(self):
        """Test get_appropriate_lock factory."""
        lock = get_appropriate_lock("test_factory", scope="distributed", timeout=1800)
        assert isinstance(lock, DistributedLock)
        assert lock.name == "test_factory"
        assert lock.lock_timeout == 1800


class TestCleanupStaleLocks:
    """Tests for stale lock cleanup functionality."""

    def test_cleanup_empty_directory(self):
        """Test cleanup with no lock files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = cleanup_stale_locks(lock_dir=Path(tmpdir))
            assert stats["scanned"] == 0
            assert stats["removed_expired"] == 0

    def test_cleanup_nonexistent_directory(self):
        """Test cleanup with non-existent directory."""
        stats = cleanup_stale_locks(lock_dir=Path("/nonexistent/path"))
        assert stats["scanned"] == 0

    def test_cleanup_expired_lock(self):
        """Test cleanup removes expired lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_file = lock_dir / "test_expired.lock"

            # Create expired lock file
            with open(lock_file, 'w') as f:
                f.write("hostname:12345:abc123\n")
                f.write(f"{time.time() - 100}\n")  # 100 seconds ago
                f.write("1\n")  # 1 second timeout (already expired)

            stats = cleanup_stale_locks(lock_dir=lock_dir)
            assert stats["scanned"] == 1
            assert stats["removed_expired"] == 1
            assert not lock_file.exists()

    def test_cleanup_old_lock(self):
        """Test cleanup removes old lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_file = lock_dir / "test_old.lock"

            # Create old lock file (older than max_age)
            with open(lock_file, 'w') as f:
                f.write("hostname:12345:abc123\n")
                f.write(f"{time.time() - 100000}\n")  # ~27 hours ago
                f.write("999999\n")  # High timeout (not technically expired)

            stats = cleanup_stale_locks(max_age_hours=24, lock_dir=lock_dir)
            assert stats["scanned"] == 1
            assert stats["removed_old"] == 1
            assert not lock_file.exists()

    def test_cleanup_preserves_valid_lock(self):
        """Test cleanup preserves valid, active lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_file = lock_dir / "test_valid.lock"

            # Create valid lock file
            with open(lock_file, 'w') as f:
                f.write("other_host:12345:abc123\n")  # Different host
                f.write(f"{time.time()}\n")  # Just now
                f.write("3600\n")  # 1 hour timeout

            stats = cleanup_stale_locks(lock_dir=lock_dir)
            assert stats["scanned"] == 1
            assert stats["removed_expired"] == 0
            assert stats["removed_old"] == 0
            assert lock_file.exists()

    def test_cleanup_dead_process_same_host(self):
        """Test cleanup removes lock from dead process on same host."""
        import socket
        hostname = socket.gethostname()

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_file = lock_dir / "test_dead.lock"

            # Create lock file with dead PID on same host
            # Use a PID that definitely doesn't exist
            dead_pid = 999999999
            with open(lock_file, 'w') as f:
                f.write(f"{hostname}:{dead_pid}:abc123\n")
                f.write(f"{time.time()}\n")
                f.write("3600\n")

            stats = cleanup_stale_locks(lock_dir=lock_dir)
            assert stats["scanned"] == 1
            assert stats["removed_dead_process"] == 1
            assert not lock_file.exists()


class TestConcurrency:
    """Tests for concurrent locking behavior."""

    def test_concurrent_acquire_exclusive(self):
        """Test only one thread can hold the lock."""
        lock_name = f"concurrent_test_{time.time()}"
        acquired_by = []
        released = threading.Event()

        def try_acquire(thread_id: int):
            lock = DistributedLock(lock_name, use_redis=False)
            if lock.acquire(timeout=5):
                acquired_by.append(thread_id)
                released.wait()  # Hold lock until signaled
                lock.release()

        threads = [threading.Thread(target=try_acquire, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()

        # Give threads time to try acquiring
        time.sleep(0.5)

        # Only one should have acquired
        assert len(acquired_by) == 1

        # Release and let others try
        released.set()
        for t in threads:
            t.join(timeout=5)

    def test_lock_passes_between_threads(self):
        """Test lock can be acquired by different threads sequentially."""
        lock_name = f"sequential_test_{time.time()}"
        results = []

        def acquire_and_record(thread_id: int):
            lock = DistributedLock(lock_name, use_redis=False)
            if lock.acquire(timeout=10):
                results.append(thread_id)
                time.sleep(0.1)  # Hold briefly
                lock.release()

        threads = [threading.Thread(target=acquire_and_record, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        # All threads should have acquired the lock
        assert len(results) == 3


class TestConstants:
    """Tests for module constants."""

    def test_default_timeouts_reasonable(self):
        """Test default timeouts are reasonable values."""
        assert DEFAULT_LOCK_TIMEOUT >= 60  # At least 1 minute
        assert DEFAULT_LOCK_TIMEOUT <= 7200  # At most 2 hours
        assert DEFAULT_ACQUIRE_TIMEOUT >= 10  # At least 10 seconds
        assert DEFAULT_ACQUIRE_TIMEOUT <= 300  # At most 5 minutes

    def test_lock_dir_is_path(self):
        """Test LOCK_DIR is a Path object."""
        assert isinstance(LOCK_DIR, Path)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_special_characters_in_name(self):
        """Test lock name with special characters."""
        lock = DistributedLock("test:special/chars@name", use_redis=False)
        path = lock._get_lock_path()
        # Path should have sanitized name
        assert ":" not in path.name
        assert "/" not in path.name

    def test_empty_lock_file(self):
        """Test handling of empty lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_file = lock_dir / "empty.lock"
            lock_file.touch()  # Create empty file

            stats = cleanup_stale_locks(lock_dir=lock_dir)
            assert stats["scanned"] == 1
            # Empty file should be preserved (can't determine if expired)

    def test_malformed_lock_file(self):
        """Test handling of malformed lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_file = lock_dir / "malformed.lock"
            with open(lock_file, 'w') as f:
                f.write("not_enough_lines\n")

            stats = cleanup_stale_locks(lock_dir=lock_dir)
            assert stats["scanned"] == 1
            # Malformed should be preserved (not enough lines)

    def test_lock_file_with_invalid_timestamp(self):
        """Test handling of invalid timestamp in lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            lock_file = lock_dir / "invalid_time.lock"
            with open(lock_file, 'w') as f:
                f.write("hostname:12345:abc123\n")
                f.write("not_a_number\n")
                f.write("3600\n")

            stats = cleanup_stale_locks(lock_dir=lock_dir)
            assert stats["scanned"] == 1
            assert stats["errors"] == 1


class TestRedisIntegration:
    """Tests for Redis integration (when available)."""

    def test_redis_not_available_fallback(self):
        """Test fallback to file lock when Redis unavailable."""
        # Create a mock Redis module
        mock_redis_module = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("Redis not available")
        mock_redis_module.Redis.from_url.return_value = mock_client
        mock_redis_module.RedisError = Exception  # Base exception type

        with patch.dict("sys.modules", {"redis": mock_redis_module}):
            with patch("app.coordination.distributed_lock.HAS_REDIS", True):
                with patch("app.coordination.distributed_lock.redis", mock_redis_module):
                    lock = DistributedLock("test_fallback", use_redis=True)
                    # Should fallback to file-based (redis_client set to None on error)
                    assert lock._redis_client is None

    def test_redis_disabled_explicitly(self):
        """Test Redis disabled with use_redis=False."""
        lock = DistributedLock("test_no_redis", use_redis=False)
        assert lock._redis_client is None
        assert lock._use_redis is False


class TestLockInfo:
    """Tests for lock information in lock files."""

    def test_lock_file_contains_info(self):
        """Test lock file contains expected information."""
        lock = DistributedLock("test_info", use_redis=False)
        try:
            lock.acquire()
            lock_path = lock._get_lock_path()

            with open(lock_path) as f:
                lines = f.readlines()

            assert len(lines) >= 3
            # Line 0: lock_id (hostname:pid:uuid)
            assert lock._lock_id in lines[0]
            # Line 1: timestamp (float)
            timestamp = float(lines[1].strip())
            assert timestamp > 0
            # Line 2: timeout
            timeout = float(lines[2].strip())
            assert timeout == lock.lock_timeout
        finally:
            lock.release()
