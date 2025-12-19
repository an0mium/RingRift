"""Tests for scripts/lib/process.py module.

Tests cover:
- SingletonLock for preventing duplicate processes
- SignalHandler for graceful shutdown
- Process discovery and control functions
- Command execution utilities
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.lib.process import (
    SingletonLock,
    SignalHandler,
    ProcessInfo,
    CommandOutput,
    is_process_running,
    find_processes_by_pattern,
    count_processes_by_pattern,
    kill_process,
    kill_processes_by_pattern,
    run_command,
    daemon_context,
    wait_for_process_exit,
)


class TestSingletonLock:
    """Tests for SingletonLock class."""

    def test_acquire_and_release(self, tmp_path):
        """Test basic acquire and release."""
        lock = SingletonLock("test-lock", lock_dir=tmp_path)
        assert lock.acquire() is True
        assert lock.acquired is True
        assert lock.lock_path.exists()

        lock.release()
        assert lock.acquired is False

    def test_context_manager(self, tmp_path):
        """Test using lock as context manager."""
        with SingletonLock("test-lock", lock_dir=tmp_path) as lock:
            assert lock.acquired is True
            assert lock.lock_path.exists()

        assert lock.acquired is False

    def test_prevents_duplicate(self, tmp_path):
        """Test that lock prevents duplicate acquisition."""
        lock1 = SingletonLock("test-lock", lock_dir=tmp_path)
        lock2 = SingletonLock("test-lock", lock_dir=tmp_path)

        assert lock1.acquire() is True
        assert lock2.acquire() is False

        lock1.release()
        # Now lock2 should be able to acquire
        assert lock2.acquire() is True
        lock2.release()

    def test_writes_pid(self, tmp_path):
        """Test that PID is written to lock file."""
        lock = SingletonLock("test-lock", lock_dir=tmp_path, write_pid=True)
        lock.acquire()

        content = lock.lock_path.read_text()
        assert content == str(os.getpid())

        lock.release()

    def test_get_holder_pid(self, tmp_path):
        """Test getting PID of lock holder."""
        lock = SingletonLock("test-lock", lock_dir=tmp_path)
        lock.acquire()

        assert lock.get_holder_pid() == os.getpid()

        lock.release()

    def test_different_names_independent(self, tmp_path):
        """Test that locks with different names are independent."""
        lock1 = SingletonLock("lock-a", lock_dir=tmp_path)
        lock2 = SingletonLock("lock-b", lock_dir=tmp_path)

        assert lock1.acquire() is True
        assert lock2.acquire() is True

        lock1.release()
        lock2.release()


class TestSignalHandler:
    """Tests for SignalHandler class."""

    def test_initial_state(self):
        """Test initial state of signal handler."""
        handler = SignalHandler()
        assert handler.running is True
        assert handler.shutdown_requested is False
        handler.restore_handlers()

    def test_with_callback(self):
        """Test shutdown callback is invoked."""
        callback_called = []

        def on_shutdown():
            callback_called.append(True)

        handler = SignalHandler(on_shutdown=on_shutdown)

        # Simulate signal
        handler._handle_signal(signal.SIGTERM, None)

        assert handler.running is False
        assert handler.shutdown_requested is True
        assert callback_called == [True]

        handler.restore_handlers()


class TestProcessInfo:
    """Tests for ProcessInfo dataclass."""

    def test_full_command(self):
        """Test full_command property."""
        proc = ProcessInfo(pid=123, command="python", args=["script.py", "--arg"])
        assert proc.full_command == "python script.py --arg"

    def test_full_command_empty_args(self):
        """Test full_command with no args."""
        proc = ProcessInfo(pid=123, command="python")
        assert proc.full_command == "python"


class TestCommandOutput:
    """Tests for CommandOutput dataclass."""

    def test_bool_success(self):
        """Test bool conversion for successful command."""
        output = CommandOutput(
            success=True, stdout="ok", stderr="", exit_code=0,
            duration_seconds=0.1, command="echo ok"
        )
        assert bool(output) is True

    def test_bool_failure(self):
        """Test bool conversion for failed command."""
        output = CommandOutput(
            success=False, stdout="", stderr="error", exit_code=1,
            duration_seconds=0.1, command="false"
        )
        assert bool(output) is False

    def test_output_prefers_stdout(self):
        """Test output property prefers stdout."""
        output = CommandOutput(
            success=True, stdout="stdout text", stderr="stderr text",
            exit_code=0, duration_seconds=0.1, command="cmd"
        )
        assert output.output == "stdout text"

    def test_output_falls_back_to_stderr(self):
        """Test output property falls back to stderr."""
        output = CommandOutput(
            success=False, stdout="", stderr="error message",
            exit_code=1, duration_seconds=0.1, command="cmd"
        )
        assert output.output == "error message"


class TestIsProcessRunning:
    """Tests for is_process_running function."""

    def test_current_process(self):
        """Test that current process is running."""
        assert is_process_running(os.getpid()) is True

    def test_nonexistent_process(self):
        """Test detecting non-existent process."""
        # Use a PID that's very unlikely to exist
        fake_pid = 999999999
        assert is_process_running(fake_pid) is False

    def test_init_process(self):
        """Test that init (PID 1) is always running."""
        # PID 1 should always exist on Unix
        assert is_process_running(1) is True


class TestFindProcessesByPattern:
    """Tests for find_processes_by_pattern function."""

    @patch("scripts.lib.process.subprocess.run")
    def test_finds_processes(self, mock_run):
        """Test finding processes by pattern."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345 python script.py\n12346 python other.py\n"
        )

        processes = find_processes_by_pattern("python")

        assert len(processes) == 2
        assert processes[0].pid == 12345
        assert "python script.py" in processes[0].command

    @patch("scripts.lib.process.subprocess.run")
    def test_no_matches(self, mock_run):
        """Test when no processes match."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        processes = find_processes_by_pattern("nonexistent-process-xyz")
        assert processes == []

    @patch("scripts.lib.process.subprocess.run")
    def test_excludes_self(self, mock_run):
        """Test that current process is excluded."""
        current_pid = os.getpid()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"{current_pid} python test.py\n12345 python other.py\n"
        )

        processes = find_processes_by_pattern("python", exclude_self=True)

        pids = [p.pid for p in processes]
        assert current_pid not in pids
        assert 12345 in pids


class TestCountProcessesByPattern:
    """Tests for count_processes_by_pattern function."""

    @patch("scripts.lib.process.subprocess.run")
    def test_counts_processes(self, mock_run):
        """Test counting processes."""
        mock_run.return_value = MagicMock(returncode=0, stdout="3\n")

        # Use a pattern that won't match our own process
        count = count_processes_by_pattern("some-unique-pattern-xyz")
        assert count == 3

    @patch("scripts.lib.process.subprocess.run")
    def test_returns_zero_on_no_match(self, mock_run):
        """Test returns zero when no matches."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        count = count_processes_by_pattern("nonexistent")
        assert count == 0


class TestKillProcess:
    """Tests for kill_process function."""

    def test_kill_nonexistent_returns_true(self):
        """Test killing non-existent process returns True."""
        fake_pid = 999999999
        result = kill_process(fake_pid)
        assert result is True

    @patch("scripts.lib.process.os.kill")
    def test_permission_error(self, mock_kill):
        """Test handling permission error."""
        mock_kill.side_effect = PermissionError()

        result = kill_process(123)
        assert result is False


class TestKillProcessesByPattern:
    """Tests for kill_processes_by_pattern function."""

    @patch("scripts.lib.process.find_processes_by_pattern")
    def test_no_matching_processes(self, mock_find):
        """Test when no processes match."""
        mock_find.return_value = []

        killed = kill_processes_by_pattern("nonexistent")
        assert killed == 0


class TestRunCommand:
    """Tests for run_command function."""

    def test_successful_command(self):
        """Test running a successful command."""
        result = run_command(["echo", "hello"])

        assert result.success is True
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0

    def test_failed_command(self):
        """Test running a failed command."""
        result = run_command(["false"])

        assert result.success is False
        assert result.exit_code != 0

    def test_with_env(self):
        """Test command with environment variables."""
        result = run_command(
            ["sh", "-c", "echo $TEST_VAR"],
            env={"TEST_VAR": "test_value"}
        )

        assert result.success is True
        assert "test_value" in result.stdout

    def test_with_cwd(self, tmp_path):
        """Test command with working directory."""
        result = run_command(["pwd"], cwd=tmp_path)

        assert result.success is True
        assert str(tmp_path) in result.stdout

    def test_timeout(self):
        """Test command timeout."""
        result = run_command(
            ["sleep", "10"],
            timeout=0.1,
            check=False
        )

        assert result.success is False
        assert result.exit_code == -1

    def test_check_raises_on_failure(self):
        """Test check=True raises on failure."""
        with pytest.raises(subprocess.CalledProcessError):
            run_command(["false"], check=True)

    def test_shell_command(self):
        """Test running shell command."""
        result = run_command("echo hello && echo world", shell=True)

        assert result.success is True
        assert "hello" in result.stdout
        assert "world" in result.stdout

    def test_command_string_parsing(self):
        """Test parsing command string."""
        result = run_command("echo test")

        assert result.success is True
        assert "test" in result.stdout


class TestDaemonContext:
    """Tests for daemon_context context manager."""

    def test_provides_signal_handler(self, tmp_path):
        """Test daemon context provides signal handler."""
        with patch.object(SingletonLock, "acquire", return_value=True):
            with daemon_context("test-daemon", lock_dir=tmp_path) as handler:
                assert handler.running is True
                assert hasattr(handler, "shutdown_requested")

    def test_exits_if_already_running(self, tmp_path):
        """Test daemon exits if another instance running."""
        # First acquire the lock
        lock = SingletonLock("test-daemon", lock_dir=tmp_path)
        lock.acquire()

        with pytest.raises(SystemExit) as exc_info:
            with daemon_context("test-daemon", lock_dir=tmp_path, exit_if_running=True):
                pass

        assert exc_info.value.code == 0
        lock.release()


class TestWaitForProcessExit:
    """Tests for wait_for_process_exit function."""

    def test_already_dead(self):
        """Test waiting for already-dead process."""
        fake_pid = 999999999
        result = wait_for_process_exit(fake_pid, timeout=1.0)
        assert result is True

    def test_timeout_on_running_process(self):
        """Test timeout when process keeps running."""
        # Current process will keep running
        result = wait_for_process_exit(os.getpid(), timeout=0.1, poll_interval=0.05)
        assert result is False
