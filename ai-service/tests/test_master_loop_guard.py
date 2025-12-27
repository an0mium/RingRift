"""Tests for master loop guard functionality (December 2025)."""

import os
import tempfile
from pathlib import Path

import pytest

from app.coordination.master_loop_guard import (
    PID_FILE_PATH,
    check_or_warn,
    ensure_master_loop_running,
    is_master_loop_running,
)


@pytest.fixture
def temp_pid_file():
    """Create a temporary PID file for testing."""
    # Use a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        pid_file = Path(tmpdir) / "test_master_loop.pid"
        yield pid_file


def test_is_master_loop_running_no_file(temp_pid_file):
    """Test that is_master_loop_running returns False when PID file doesn't exist."""
    assert not is_master_loop_running(temp_pid_file)


def test_is_master_loop_running_valid_process(temp_pid_file):
    """Test that is_master_loop_running returns True for valid process."""
    # Write current process PID
    with open(temp_pid_file, "w") as f:
        f.write(str(os.getpid()))

    assert is_master_loop_running(temp_pid_file)


def test_is_master_loop_running_stale_pid(temp_pid_file):
    """Test that is_master_loop_running removes stale PID file."""
    # Write a PID that definitely doesn't exist (99999999)
    with open(temp_pid_file, "w") as f:
        f.write("99999999")

    assert not is_master_loop_running(temp_pid_file)
    # PID file should be cleaned up
    assert not temp_pid_file.exists()


def test_is_master_loop_running_invalid_pid(temp_pid_file):
    """Test that is_master_loop_running handles invalid PID gracefully."""
    # Write invalid PID
    with open(temp_pid_file, "w") as f:
        f.write("not_a_number")

    assert not is_master_loop_running(temp_pid_file)


def test_ensure_master_loop_running_no_check(temp_pid_file):
    """Test that ensure_master_loop_running doesn't raise when requirement disabled."""
    # Should not raise even though no PID file exists
    ensure_master_loop_running(require_for_automation=False)


def test_ensure_master_loop_running_raises(temp_pid_file):
    """Test that ensure_master_loop_running raises when master loop is not running."""
    with pytest.raises(RuntimeError, match="Master loop must be running"):
        ensure_master_loop_running(
            require_for_automation=True,
            operation_name="test operation",
        )


def test_ensure_master_loop_running_passes(temp_pid_file):
    """Test that ensure_master_loop_running passes when master loop is running."""
    # Create a valid PID file
    with open(temp_pid_file, "w") as f:
        f.write(str(os.getpid()))

    # Temporarily override PID_FILE_PATH for this test
    import app.coordination.master_loop_guard as guard_module

    original_path = guard_module.PID_FILE_PATH
    try:
        guard_module.PID_FILE_PATH = temp_pid_file
        # Should not raise
        ensure_master_loop_running(require_for_automation=True)
    finally:
        guard_module.PID_FILE_PATH = original_path


def test_check_or_warn_returns_bool(temp_pid_file):
    """Test that check_or_warn returns correct boolean."""
    assert not check_or_warn("test operation")

    # With valid PID
    with open(temp_pid_file, "w") as f:
        f.write(str(os.getpid()))

    import app.coordination.master_loop_guard as guard_module

    original_path = guard_module.PID_FILE_PATH
    try:
        guard_module.PID_FILE_PATH = temp_pid_file
        assert check_or_warn("test operation")
    finally:
        guard_module.PID_FILE_PATH = original_path
