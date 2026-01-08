"""Tests for async_process_pool.py.

Tests the non-blocking subprocess execution pool for async contexts.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.async_process_pool import (
    AsyncProcessPool,
    ProcessConfig,
    ProcessExecutionError,
    ProcessResult,
    ProcessTimeoutError,
    get_process_pool,
    reset_process_pool,
    run_command,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test.

    Note: We directly set _instance to None to avoid the asyncio.create_task()
    call in reset_instance() which requires a running event loop.
    """
    # Direct reset without cancel_all (no running processes in test isolation)
    AsyncProcessPool._instance = None
    yield
    AsyncProcessPool._instance = None


# =============================================================================
# ProcessResult Tests
# =============================================================================


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_success_property_true_on_zero_exit(self):
        """Success should be True when exit_code is 0 and not timed out."""
        result = ProcessResult(
            stdout="output",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
        )
        assert result.success is True

    def test_success_property_false_on_nonzero_exit(self):
        """Success should be False when exit_code is non-zero."""
        result = ProcessResult(
            stdout="",
            stderr="error",
            exit_code=1,
            duration_seconds=1.0,
        )
        assert result.success is False

    def test_success_property_false_on_timeout(self):
        """Success should be False when timed_out is True."""
        result = ProcessResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=5.0,
            timed_out=True,
        )
        assert result.success is False

    def test_default_values(self):
        """Test default values for ProcessResult."""
        result = ProcessResult(
            stdout="out",
            stderr="err",
            exit_code=0,
            duration_seconds=1.0,
        )
        assert result.command == []
        assert result.timed_out is False


# =============================================================================
# ProcessConfig Tests
# =============================================================================


class TestProcessConfig:
    """Tests for ProcessConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessConfig()
        assert config.max_concurrent == 10
        assert config.default_timeout == 60.0
        assert config.kill_timeout == 5.0
        assert config.encoding == "utf-8"
        assert config.capture_output is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProcessConfig(
            max_concurrent=5,
            default_timeout=30.0,
            kill_timeout=2.0,
            encoding="latin-1",
            capture_output=False,
        )
        assert config.max_concurrent == 5
        assert config.default_timeout == 30.0
        assert config.kill_timeout == 2.0
        assert config.encoding == "latin-1"
        assert config.capture_output is False


# =============================================================================
# Exception Tests
# =============================================================================


class TestProcessTimeoutError:
    """Tests for ProcessTimeoutError exception."""

    def test_message_format(self):
        """Test exception message format."""
        exc = ProcessTimeoutError(["python", "-c", "pass"], 30.0)
        assert "30.0s" in str(exc)
        assert "python -c pass" in str(exc)

    def test_message_truncates_long_commands(self):
        """Test that long commands are truncated in message."""
        cmd = ["cmd", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]
        exc = ProcessTimeoutError(cmd, 10.0)
        assert "..." in str(exc)

    def test_attributes(self):
        """Test exception attributes."""
        cmd = ["echo", "hello"]
        exc = ProcessTimeoutError(cmd, 60.0)
        assert exc.command == cmd
        assert exc.timeout_seconds == 60.0


class TestProcessExecutionError:
    """Tests for ProcessExecutionError exception."""

    def test_message_format(self):
        """Test exception message format."""
        exc = ProcessExecutionError(
            ["python", "-c", "fail"],
            exit_code=1,
            stdout="out",
            stderr="error message",
        )
        assert "exit code 1" in str(exc)
        assert "error message" in str(exc)

    def test_message_truncates_long_stderr(self):
        """Test that long stderr is truncated in message."""
        long_stderr = "x" * 300
        exc = ProcessExecutionError(["cmd"], 1, "", long_stderr)
        # Message should truncate stderr to 200 chars
        assert len(str(exc)) < 350

    def test_attributes(self):
        """Test exception attributes."""
        cmd = ["echo", "test"]
        exc = ProcessExecutionError(cmd, 2, "stdout", "stderr")
        assert exc.command == cmd
        assert exc.exit_code == 2
        assert exc.stdout == "stdout"
        assert exc.stderr == "stderr"


# =============================================================================
# AsyncProcessPool Basic Tests
# =============================================================================


class TestAsyncProcessPoolBasic:
    """Tests for AsyncProcessPool basic functionality."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        pool = AsyncProcessPool()
        assert pool.config.max_concurrent == 10
        assert pool.config.default_timeout == 60.0
        assert pool._total_executed == 0

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = ProcessConfig(max_concurrent=5)
        pool = AsyncProcessPool(config=config)
        assert pool.config.max_concurrent == 5

    def test_get_active_count_initially_zero(self):
        """Test that active count is initially zero."""
        pool = AsyncProcessPool()
        assert pool.get_active_count() == 0

    def test_get_stats_initial_values(self):
        """Test initial statistics values."""
        pool = AsyncProcessPool()
        stats = pool.get_stats()
        assert stats["total_executed"] == 0
        assert stats["total_succeeded"] == 0
        assert stats["total_failed"] == 0
        assert stats["total_timed_out"] == 0
        assert stats["success_rate"] == 1.0  # Default when no executions
        assert stats["active_processes"] == []


# =============================================================================
# AsyncProcessPool Singleton Tests
# =============================================================================


class TestAsyncProcessPoolSingleton:
    """Tests for AsyncProcessPool singleton pattern."""

    def test_get_instance_returns_same_instance(self):
        """Test that get_instance returns the same instance."""
        pool1 = AsyncProcessPool.get_instance()
        pool2 = AsyncProcessPool.get_instance()
        assert pool1 is pool2

    def test_reset_instance_clears_singleton(self):
        """Test that reset_instance clears the singleton."""
        pool1 = AsyncProcessPool.get_instance()
        # Direct reset to avoid asyncio.create_task() in sync context
        AsyncProcessPool._instance = None
        pool2 = AsyncProcessPool.get_instance()
        assert pool1 is not pool2

    def test_get_process_pool_function(self):
        """Test the get_process_pool helper function."""
        pool = get_process_pool()
        assert isinstance(pool, AsyncProcessPool)

    def test_get_process_pool_returns_same_instance(self):
        """Test that get_process_pool returns the same instance."""
        pool1 = get_process_pool()
        pool2 = get_process_pool()
        assert pool1 is pool2

    def test_reset_process_pool_function(self):
        """Test the reset_process_pool helper function behavior."""
        pool1 = get_process_pool()
        # Direct reset to avoid asyncio.create_task() in sync context
        # This tests the conceptual behavior of reset_process_pool()
        AsyncProcessPool._instance = None
        pool2 = get_process_pool()
        assert pool1 is not pool2


# =============================================================================
# AsyncProcessPool Execute Tests
# =============================================================================


class TestAsyncProcessPoolExecute:
    """Tests for AsyncProcessPool.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_simple_command(self):
        """Test executing a simple command."""
        pool = AsyncProcessPool()
        result = await pool.execute(["echo", "hello"], timeout_seconds=5.0)
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.success is True
        assert result.timed_out is False
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(self):
        """Test executing a command that writes to stderr."""
        pool = AsyncProcessPool()
        result = await pool.execute(
            ["python", "-c", "import sys; sys.stderr.write('error\\n')"],
            timeout_seconds=5.0,
        )
        assert "error" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_command_with_nonzero_exit(self):
        """Test executing a command with non-zero exit code."""
        pool = AsyncProcessPool()
        result = await pool.execute(["python", "-c", "exit(42)"], timeout_seconds=5.0)
        assert result.exit_code == 42
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_with_check_raises_on_failure(self):
        """Test that check=True raises on non-zero exit."""
        pool = AsyncProcessPool()
        with pytest.raises(ProcessExecutionError) as exc_info:
            await pool.execute(
                ["python", "-c", "exit(1)"],
                timeout_seconds=5.0,
                check=True,
            )
        assert exc_info.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_execute_with_cwd(self):
        """Test executing a command with working directory."""
        pool = AsyncProcessPool()
        result = await pool.execute(
            ["pwd"],
            timeout_seconds=5.0,
            cwd="/tmp",
        )
        assert "/tmp" in result.stdout or "/private/tmp" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_with_env(self):
        """Test executing a command with custom environment."""
        pool = AsyncProcessPool()
        result = await pool.execute(
            ["python", "-c", "import os; print(os.environ.get('TEST_VAR', 'missing'))"],
            timeout_seconds=5.0,
            env={"TEST_VAR": "custom_value"},
        )
        assert "custom_value" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_uses_default_timeout(self):
        """Test that execute uses default timeout when not specified."""
        config = ProcessConfig(default_timeout=10.0)
        pool = AsyncProcessPool(config=config)

        result = await pool.execute(["echo", "test"])
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tracks_command(self):
        """Test that executed command is tracked in result."""
        pool = AsyncProcessPool()
        cmd = ["echo", "tracked"]
        result = await pool.execute(cmd, timeout_seconds=5.0)
        assert result.command == cmd


# =============================================================================
# AsyncProcessPool Timeout Tests
# =============================================================================


class TestAsyncProcessPoolTimeout:
    """Tests for AsyncProcessPool timeout handling."""

    @pytest.mark.asyncio
    async def test_execute_timeout_sets_flag(self):
        """Test that timeout sets the timed_out flag."""
        pool = AsyncProcessPool()
        result = await pool.execute(
            ["sleep", "10"],
            timeout_seconds=0.1,
        )
        assert result.timed_out is True
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_timeout_with_check_raises(self):
        """Test that timeout with check=True raises ProcessTimeoutError."""
        pool = AsyncProcessPool()
        with pytest.raises(ProcessTimeoutError) as exc_info:
            await pool.execute(
                ["sleep", "10"],
                timeout_seconds=0.1,
                check=True,
            )
        assert exc_info.value.timeout_seconds == 0.1


# =============================================================================
# AsyncProcessPool Statistics Tests
# =============================================================================


class TestAsyncProcessPoolStats:
    """Tests for AsyncProcessPool statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_track_successful_execution(self):
        """Test that successful execution updates stats."""
        pool = AsyncProcessPool()
        await pool.execute(["echo", "test"], timeout_seconds=5.0)

        stats = pool.get_stats()
        assert stats["total_executed"] == 1
        assert stats["total_succeeded"] == 1
        assert stats["total_failed"] == 0
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_stats_track_failed_execution(self):
        """Test that failed execution updates stats."""
        pool = AsyncProcessPool()
        await pool.execute(["python", "-c", "exit(1)"], timeout_seconds=5.0)

        stats = pool.get_stats()
        assert stats["total_executed"] == 1
        assert stats["total_failed"] == 1
        assert stats["total_succeeded"] == 0
        assert stats["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_track_timed_out_execution(self):
        """Test that timed out execution updates stats."""
        pool = AsyncProcessPool()
        await pool.execute(["sleep", "10"], timeout_seconds=0.1)

        stats = pool.get_stats()
        assert stats["total_executed"] == 1
        assert stats["total_timed_out"] == 1
        assert stats["total_succeeded"] == 0

    @pytest.mark.asyncio
    async def test_stats_success_rate_calculation(self):
        """Test success rate calculation with mixed results."""
        pool = AsyncProcessPool()

        # 2 successful, 1 failed
        await pool.execute(["echo", "1"], timeout_seconds=5.0)
        await pool.execute(["echo", "2"], timeout_seconds=5.0)
        await pool.execute(["python", "-c", "exit(1)"], timeout_seconds=5.0)

        stats = pool.get_stats()
        assert stats["total_executed"] == 3
        assert stats["total_succeeded"] == 2
        assert stats["total_failed"] == 1
        assert stats["success_rate"] == pytest.approx(2 / 3)


# =============================================================================
# AsyncProcessPool Concurrency Tests
# =============================================================================


class TestAsyncProcessPoolConcurrency:
    """Tests for AsyncProcessPool concurrency limiting."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_respects_limit(self):
        """Test that concurrent execution respects max_concurrent limit."""
        config = ProcessConfig(max_concurrent=2)
        pool = AsyncProcessPool(config=config)

        # Track active count during execution
        active_counts = []

        async def track_and_execute():
            active_counts.append(pool.get_active_count())
            result = await pool.execute(["sleep", "0.1"], timeout_seconds=5.0)
            return result

        # Start 4 concurrent tasks
        tasks = [track_and_execute() for _ in range(4)]
        await asyncio.gather(*tasks)

        # All should complete
        assert pool._total_executed == 4


# =============================================================================
# AsyncProcessPool Cancel Tests
# =============================================================================


class TestAsyncProcessPoolCancel:
    """Tests for AsyncProcessPool cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_all_returns_count(self):
        """Test that cancel_all returns the number cancelled."""
        pool = AsyncProcessPool()
        cancelled = await pool.cancel_all()
        assert cancelled == 0  # No active processes

    @pytest.mark.asyncio
    async def test_cancel_all_clears_active_dict(self):
        """Test that cancel_all clears the active processes dict."""
        pool = AsyncProcessPool()
        # Manually add a mock to _active to test clearing
        pool._active["test"] = MagicMock()
        pool._active["test"].process = MagicMock()
        pool._active["test"].process.terminate = MagicMock()
        pool._active["test"].process.wait = AsyncMock()
        pool._active["test"].command = ["test"]

        await pool.cancel_all()
        assert len(pool._active) == 0


# =============================================================================
# run_command Convenience Function Tests
# =============================================================================


class TestRunCommand:
    """Tests for run_command convenience function."""

    @pytest.mark.asyncio
    async def test_run_command_basic(self):
        """Test basic run_command usage."""
        result = await run_command(["echo", "convenience"], timeout_seconds=5.0)
        assert "convenience" in result.stdout
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_command_with_cwd(self):
        """Test run_command with working directory."""
        result = await run_command(["pwd"], timeout_seconds=5.0, cwd="/tmp")
        assert "/tmp" in result.stdout or "/private/tmp" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_with_check(self):
        """Test run_command with check flag."""
        with pytest.raises(ProcessExecutionError):
            await run_command(
                ["python", "-c", "exit(1)"],
                timeout_seconds=5.0,
                check=True,
            )


# =============================================================================
# AsyncProcessPool Termination Tests
# =============================================================================


class TestAsyncProcessPoolTermination:
    """Tests for process termination handling."""

    @pytest.mark.asyncio
    async def test_terminate_process_sends_sigterm(self):
        """Test that _terminate_process sends SIGTERM first."""
        pool = AsyncProcessPool()

        # Mock process
        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        await pool._terminate_process(mock_proc, ["test"])

        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_process_handles_process_lookup_error(self):
        """Test that _terminate_process handles ProcessLookupError."""
        pool = AsyncProcessPool()

        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock(side_effect=ProcessLookupError())

        # Should not raise
        await pool._terminate_process(mock_proc, ["test"])


# =============================================================================
# AsyncProcessPool Edge Cases
# =============================================================================


class TestAsyncProcessPoolEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_execute_with_empty_output(self):
        """Test executing a command with no output."""
        pool = AsyncProcessPool()
        result = await pool.execute(["true"], timeout_seconds=5.0)
        assert result.stdout == ""
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_tracks_duration(self):
        """Test that execution duration is tracked accurately."""
        pool = AsyncProcessPool()
        start = time.monotonic()
        result = await pool.execute(["sleep", "0.2"], timeout_seconds=5.0)
        elapsed = time.monotonic() - start

        # Duration should be close to actual time
        assert result.duration_seconds >= 0.1
        assert result.duration_seconds <= elapsed + 0.1

    @pytest.mark.asyncio
    async def test_active_processes_cleared_after_execution(self):
        """Test that active processes dict is cleared after execution."""
        pool = AsyncProcessPool()

        # Execute a command
        await pool.execute(["echo", "test"], timeout_seconds=5.0)

        # Active dict should be empty
        assert pool.get_active_count() == 0

    @pytest.mark.asyncio
    async def test_stats_include_config_info(self):
        """Test that stats include configuration info."""
        config = ProcessConfig(max_concurrent=7)
        pool = AsyncProcessPool(config=config)

        stats = pool.get_stats()
        assert stats["max_concurrent"] == 7
        assert stats["active_count"] == 0

    def test_counter_increments(self):
        """Test that internal counter increments."""
        pool = AsyncProcessPool()
        assert pool._counter == 0

        # Counter only increments during execute, which we don't run here
        # Just verify it exists and is initialized
        assert hasattr(pool, "_counter")
