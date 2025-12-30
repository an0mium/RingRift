"""Unit tests for async_executor_base.py.

Tests the unified async executor framework that provides:
- AsyncExecutor: Managed async task execution with lifecycle
- TaskGroup: Grouped task management with cancellation
- TimeoutExecutor: Tasks with timeout enforcement
- RetryExecutor: Tasks with retry and backoff

This is critical infrastructure (866 LOC) used by all 77 daemons.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.async_executor_base import (
    AsyncExecutor,
    RetryConfig,
    RetryExecutor,
    TaskGroup,
    TaskInfo,
    TaskState,
    TimeoutExecutor,
    execute_with_retry,
    execute_with_timeout,
    managed_executor,
)


# =============================================================================
# TaskState Tests
# =============================================================================


class TestTaskState:
    """Tests for TaskState enum."""

    def test_task_state_values(self) -> None:
        """Test that all task states have expected values."""
        assert TaskState.PENDING.value == "pending"
        assert TaskState.RUNNING.value == "running"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELLED.value == "cancelled"
        assert TaskState.TIMEOUT.value == "timeout"

    def test_task_state_count(self) -> None:
        """Test that we have all expected states."""
        assert len(TaskState) == 6


# =============================================================================
# TaskInfo Tests
# =============================================================================


class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_task_info_defaults(self) -> None:
        """Test TaskInfo with default values."""
        info = TaskInfo(task_id="abc123", name="test_task")

        assert info.task_id == "abc123"
        assert info.name == "test_task"
        assert info.state == TaskState.PENDING
        assert info.created_at > 0
        assert info.started_at is None
        assert info.completed_at is None
        assert info.timeout_seconds is None
        assert info.error is None
        assert info.result is None

    def test_task_info_duration_not_started(self) -> None:
        """Test duration is None when not started."""
        info = TaskInfo(task_id="abc", name="test")
        assert info.duration is None

    def test_task_info_duration_running(self) -> None:
        """Test duration when task is running."""
        info = TaskInfo(task_id="abc", name="test")
        info.started_at = time.time() - 5.0  # Started 5 seconds ago

        duration = info.duration
        assert duration is not None
        assert 5.0 <= duration <= 6.0

    def test_task_info_duration_completed(self) -> None:
        """Test duration when task is completed."""
        info = TaskInfo(task_id="abc", name="test")
        info.started_at = 1000.0
        info.completed_at = 1005.0

        assert info.duration == 5.0

    def test_task_info_to_dict(self) -> None:
        """Test serialization to dictionary."""
        info = TaskInfo(
            task_id="abc123",
            name="test_task",
            state=TaskState.COMPLETED,
            timeout_seconds=30.0,
        )
        info.started_at = 1000.0
        info.completed_at = 1005.0
        info.error = ValueError("test error")

        result = info.to_dict()

        assert result["task_id"] == "abc123"
        assert result["name"] == "test_task"
        assert result["state"] == "completed"
        assert result["timeout_seconds"] == 30.0
        assert result["duration"] == 5.0
        assert result["error"] == "test error"


# =============================================================================
# AsyncExecutor Tests
# =============================================================================


class TestAsyncExecutor:
    """Tests for AsyncExecutor class."""

    @pytest.fixture
    def executor(self) -> AsyncExecutor:
        """Create a test executor."""
        return AsyncExecutor(name="test_executor", max_concurrent=10)

    @pytest.mark.asyncio
    async def test_executor_start(self, executor: AsyncExecutor) -> None:
        """Test executor can be started."""
        await executor.start()
        assert executor._running is True
        assert executor._semaphore is not None
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_executor_double_start(self, executor: AsyncExecutor) -> None:
        """Test that starting twice is a no-op."""
        await executor.start()
        await executor.start()  # Should not raise
        assert executor._running is True
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_executor_shutdown(self, executor: AsyncExecutor) -> None:
        """Test executor shutdown."""
        await executor.start()
        await executor.shutdown()
        assert executor._running is False

    @pytest.mark.asyncio
    async def test_executor_double_shutdown(self, executor: AsyncExecutor) -> None:
        """Test that shutting down twice is safe."""
        await executor.start()
        await executor.shutdown()
        await executor.shutdown()  # Should not raise
        assert executor._running is False

    @pytest.mark.asyncio
    async def test_submit_not_running_raises(self, executor: AsyncExecutor) -> None:
        """Test that submit raises when executor not running."""
        async def dummy() -> None:
            pass

        with pytest.raises(RuntimeError, match="not running"):
            await executor.submit(dummy())

    @pytest.mark.asyncio
    async def test_submit_task(self, executor: AsyncExecutor) -> None:
        """Test submitting a task."""
        await executor.start()

        async def my_task() -> int:
            return 42

        task_id = await executor.submit(my_task(), name="my_task")
        result = await executor.wait_for(task_id)

        assert result == 42
        assert executor._total_completed == 1
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_task_with_error(self, executor: AsyncExecutor) -> None:
        """Test submitting a task that raises an error."""
        await executor.start()

        async def failing_task() -> None:
            raise ValueError("test error")

        error_callback = MagicMock()
        task_id = await executor.submit(failing_task(), on_error=error_callback)

        with pytest.raises(ValueError):
            await executor.wait_for(task_id)

        error_callback.assert_called_once()
        assert executor._total_failed == 1
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_task_with_timeout(self, executor: AsyncExecutor) -> None:
        """Test task timeout enforcement."""
        await executor.start()

        async def slow_task() -> None:
            await asyncio.sleep(10)

        task_id = await executor.submit(slow_task(), timeout=0.1)

        with pytest.raises(asyncio.TimeoutError):
            await executor.wait_for(task_id)

        assert executor._total_timeout == 1
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_with_completion_callback(self, executor: AsyncExecutor) -> None:
        """Test completion callback is called."""
        await executor.start()

        async def my_task() -> str:
            return "result"

        callback = MagicMock()
        task_id = await executor.submit(my_task(), on_complete=callback)
        await executor.wait_for(task_id)

        callback.assert_called_once_with("result")
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_fire_and_forget_not_running(self, executor: AsyncExecutor) -> None:
        """Test fire_and_forget returns None when not running."""
        async def dummy() -> None:
            pass

        result = executor.fire_and_forget(dummy())
        assert result is None

    @pytest.mark.asyncio
    async def test_fire_and_forget(self, executor: AsyncExecutor) -> None:
        """Test fire_and_forget task execution."""
        await executor.start()

        completed = asyncio.Event()

        async def my_task() -> None:
            completed.set()

        task_id = executor.fire_and_forget(my_task())
        assert task_id is not None

        await asyncio.wait_for(completed.wait(), timeout=1.0)
        await asyncio.sleep(0.1)  # Let cleanup happen

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_fire_and_forget_with_error(self, executor: AsyncExecutor) -> None:
        """Test fire_and_forget handles errors gracefully."""
        await executor.start()

        error_callback = MagicMock()

        async def failing_task() -> None:
            raise ValueError("test")

        task_id = executor.fire_and_forget(failing_task(), on_error=error_callback)
        assert task_id is not None

        await asyncio.sleep(0.1)  # Let task run

        error_callback.assert_called_once()
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_wait_for_unknown_task(self, executor: AsyncExecutor) -> None:
        """Test wait_for raises for unknown task."""
        await executor.start()

        with pytest.raises(KeyError):
            await executor.wait_for("unknown_id")

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_wait_for_completed_task(self, executor: AsyncExecutor) -> None:
        """Test wait_for returns result for already completed task."""
        await executor.start()

        async def quick_task() -> int:
            return 123

        task_id = await executor.submit(quick_task())
        result1 = await executor.wait_for(task_id)

        # Task should be tracked even after completion
        info = executor.get_task_info(task_id)
        assert info is not None
        assert info.result == 123

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_wait_all(self, executor: AsyncExecutor) -> None:
        """Test waiting for all tasks."""
        await executor.start()

        results: list[int] = []

        async def task(n: int) -> int:
            results.append(n)
            return n

        for i in range(5):
            await executor.submit(task(i))

        await executor.wait_all()

        assert len(results) == 5
        assert executor._total_completed == 5
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_wait_all_empty(self, executor: AsyncExecutor) -> None:
        """Test wait_all with no tasks."""
        await executor.start()
        await executor.wait_all()  # Should not hang
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_get_pending_count(self, executor: AsyncExecutor) -> None:
        """Test getting pending task count."""
        await executor.start()

        async def slow_task() -> None:
            await asyncio.sleep(1)

        await executor.submit(slow_task())
        await executor.submit(slow_task())

        assert executor.get_pending_count() == 2
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_get_stats(self, executor: AsyncExecutor) -> None:
        """Test getting executor statistics."""
        await executor.start()

        async def task() -> int:
            return 1

        await executor.submit(task())
        await executor.wait_all()

        stats = executor.get_stats()

        assert stats["name"] == "test_executor"
        assert stats["running"] is True
        assert stats["total_submitted"] == 1
        assert stats["total_completed"] == 1
        assert stats["max_concurrent"] == 10

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_global_error_callback(self) -> None:
        """Test global error callback."""
        callback = MagicMock()
        executor = AsyncExecutor(name="test", on_task_error=callback)
        await executor.start()

        async def failing() -> None:
            raise ValueError("fail")

        task_id = await executor.submit(failing())

        with pytest.raises(ValueError):
            await executor.wait_for(task_id)

        callback.assert_called_once()
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_global_complete_callback(self) -> None:
        """Test global completion callback."""
        callback = MagicMock()
        executor = AsyncExecutor(name="test", on_task_complete=callback)
        await executor.start()

        async def task() -> str:
            return "done"

        task_id = await executor.submit(task())
        await executor.wait_for(task_id)

        callback.assert_called_once()
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending(self, executor: AsyncExecutor) -> None:
        """Test shutdown cancels pending tasks."""
        await executor.start()

        async def slow_task() -> None:
            await asyncio.sleep(10)

        await executor.submit(slow_task())
        await executor.submit(slow_task())

        assert executor.get_pending_count() == 2

        await executor.shutdown(timeout=1.0)

        assert executor._total_cancelled == 2


# =============================================================================
# TaskGroup Tests
# =============================================================================


class TestTaskGroup:
    """Tests for TaskGroup class."""

    @pytest.mark.asyncio
    async def test_task_group_basic(self) -> None:
        """Test basic task group usage."""
        results: list[int] = []

        async with TaskGroup(name="test_group") as group:
            for i in range(3):
                async def task(n: int = i) -> int:
                    results.append(n)
                    return n

                group.create_task(task())

        assert len(results) == 3
        assert group.success_count == 3
        assert group.error_count == 0

    @pytest.mark.asyncio
    async def test_task_group_results(self) -> None:
        """Test task group collects results."""
        async with TaskGroup(name="test_group") as group:
            async def task1() -> int:
                return 1

            async def task2() -> int:
                return 2

            group.create_task(task1())
            group.create_task(task2())

        assert set(group.results) == {1, 2}

    @pytest.mark.asyncio
    async def test_task_group_errors(self) -> None:
        """Test task group collects errors."""
        async with TaskGroup(name="test_group", suppress_errors=True) as group:
            async def failing() -> None:
                raise ValueError("fail")

            async def success() -> int:
                return 1

            group.create_task(failing())
            group.create_task(success())

        assert group.error_count == 1
        assert group.success_count == 1
        assert len(group.errors) == 1
        assert isinstance(group.errors[0], ValueError)

    @pytest.mark.asyncio
    async def test_task_group_timeout(self) -> None:
        """Test task group timeout."""
        async with TaskGroup(name="test", timeout=0.1) as group:
            async def slow() -> None:
                await asyncio.sleep(10)

            group.create_task(slow())

        assert group.error_count == 1
        assert isinstance(group.errors[0], asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_task_group_not_started_error(self) -> None:
        """Test error when creating task without context manager."""
        group = TaskGroup(name="test")

        async def task() -> None:
            pass

        with pytest.raises(RuntimeError, match="not started"):
            group.create_task(task())

    @pytest.mark.asyncio
    async def test_task_group_finished_error(self) -> None:
        """Test error when creating task after group finished."""
        group: TaskGroup

        async with TaskGroup(name="test") as group:
            pass

        async def task() -> None:
            pass

        with pytest.raises(RuntimeError, match="finished"):
            group.create_task(task())

    @pytest.mark.asyncio
    async def test_task_group_empty(self) -> None:
        """Test empty task group."""
        async with TaskGroup(name="empty") as group:
            pass

        assert group.success_count == 0
        assert group.error_count == 0


# =============================================================================
# TimeoutExecutor Tests
# =============================================================================


class TestTimeoutExecutor:
    """Tests for TimeoutExecutor class."""

    @pytest.mark.asyncio
    async def test_timeout_executor_success(self) -> None:
        """Test successful execution within timeout."""
        executor = TimeoutExecutor(default_timeout=10.0)

        async def quick() -> int:
            return 42

        result = await executor.run(quick())
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout_executor_timeout(self) -> None:
        """Test timeout is enforced."""
        executor = TimeoutExecutor(default_timeout=0.1)

        async def slow() -> None:
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            await executor.run(slow())

    @pytest.mark.asyncio
    async def test_timeout_executor_override(self) -> None:
        """Test timeout can be overridden per call."""
        executor = TimeoutExecutor(default_timeout=10.0)

        async def slow() -> None:
            await asyncio.sleep(1)

        with pytest.raises(asyncio.TimeoutError):
            await executor.run(slow(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_timeout_executor_callback(self) -> None:
        """Test timeout callback is called."""
        callback = MagicMock()
        executor = TimeoutExecutor(default_timeout=0.1, on_timeout=callback)

        async def slow() -> None:
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            await executor.run(slow(), name="my_task")

        callback.assert_called_once_with("my_task")


# =============================================================================
# RetryExecutor Tests
# =============================================================================


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_retry_config_defaults(self) -> None:
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retry_on == (Exception,)

    def test_retry_config_custom(self) -> None:
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            retry_on=(ValueError, TypeError),
        )

        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.retry_on == (ValueError, TypeError)


class TestRetryExecutor:
    """Tests for RetryExecutor class."""

    @pytest.mark.asyncio
    async def test_retry_success_first_try(self) -> None:
        """Test successful execution on first try."""
        executor = RetryExecutor()

        async def task() -> int:
            return 42

        result = await executor.run(lambda: task())
        assert result == 42

    @pytest.mark.asyncio
    async def test_retry_success_after_failure(self) -> None:
        """Test success after initial failures."""
        attempt = 0

        async def flaky() -> str:
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ValueError("not yet")
            return "success"

        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        executor = RetryExecutor(config=config)

        result = await executor.run(lambda: flaky())
        assert result == "success"
        assert attempt == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        """Test failure after all retries exhausted."""
        async def always_fail() -> None:
            raise ValueError("always fails")

        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        executor = RetryExecutor(config=config)

        with pytest.raises(ValueError, match="always fails"):
            await executor.run(lambda: always_fail())

    @pytest.mark.asyncio
    async def test_retry_callback(self) -> None:
        """Test retry callback is called."""
        callback = MagicMock()
        attempt = 0

        async def flaky() -> str:
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise ValueError("retry")
            return "done"

        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        executor = RetryExecutor(config=config, on_retry=callback)

        await executor.run(lambda: flaky())

        callback.assert_called_once()
        call_args = callback.call_args
        assert call_args[0][0] == 1  # First retry
        assert isinstance(call_args[0][1], ValueError)

    @pytest.mark.asyncio
    async def test_retry_only_on_specified_exceptions(self) -> None:
        """Test retry only on specified exception types."""
        attempt = 0

        async def task() -> None:
            nonlocal attempt
            attempt += 1
            raise TypeError("wrong type")

        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            retry_on=(ValueError,),  # Only retry ValueError
        )
        executor = RetryExecutor(config=config)

        with pytest.raises(TypeError):
            await executor.run(lambda: task())

        # Should not retry for TypeError
        assert attempt == 1


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestExecuteWithTimeout:
    """Tests for execute_with_timeout function."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """Test successful execution."""
        async def task() -> int:
            return 42

        result = await execute_with_timeout(task(), timeout=10.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout_returns_default(self) -> None:
        """Test timeout returns default value."""
        async def slow() -> int:
            await asyncio.sleep(10)
            return 42

        result = await execute_with_timeout(slow(), timeout=0.05, default=-1)
        assert result == -1

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self) -> None:
        """Test timeout returns None when no default specified."""
        async def slow() -> int:
            await asyncio.sleep(10)
            return 42

        result = await execute_with_timeout(slow(), timeout=0.05)
        assert result is None


class TestExecuteWithRetry:
    """Tests for execute_with_retry function."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """Test successful execution."""
        async def task() -> str:
            return "done"

        result = await execute_with_retry(lambda: task())
        assert result == "done"

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        """Test retry until success."""
        attempt = 0

        async def flaky() -> str:
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise ValueError("retry")
            return "done"

        result = await execute_with_retry(lambda: flaky(), max_attempts=3, delay=0.01)
        assert result == "done"
        assert attempt == 2


class TestManagedExecutor:
    """Tests for managed_executor context manager."""

    @pytest.mark.asyncio
    async def test_managed_executor_lifecycle(self) -> None:
        """Test executor is properly started and shut down."""
        async with managed_executor(name="test", max_concurrent=5) as executor:
            assert executor._running is True
            assert executor.name == "test"
            assert executor.max_concurrent == 5

            async def task() -> int:
                return 42

            task_id = await executor.submit(task())
            result = await executor.wait_for(task_id)
            assert result == 42

        # After context exit
        assert executor._running is False

    @pytest.mark.asyncio
    async def test_managed_executor_with_timeout(self) -> None:
        """Test managed executor with default timeout."""
        async with managed_executor(name="test", default_timeout=5.0) as executor:
            assert executor.default_timeout == 5.0


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent execution."""

    @pytest.mark.asyncio
    async def test_max_concurrent_limit(self) -> None:
        """Test that max_concurrent is enforced."""
        executor = AsyncExecutor(name="test", max_concurrent=2)
        await executor.start()

        running: list[bool] = []
        max_running = 0

        async def task() -> None:
            nonlocal max_running
            running.append(True)
            current = len(running)
            if current > max_running:
                max_running = current
            await asyncio.sleep(0.1)
            running.pop()

        # Submit 5 tasks but only 2 should run concurrently
        for _ in range(5):
            await executor.submit(task())

        await executor.wait_all()

        assert max_running <= 2
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_task_group(self) -> None:
        """Test tasks in group run concurrently."""
        start = time.time()

        async with TaskGroup(name="test") as group:
            for _ in range(3):
                async def task() -> None:
                    await asyncio.sleep(0.1)

                group.create_task(task())

        elapsed = time.time() - start

        # All tasks should run concurrently, so total time ~0.1s not 0.3s
        assert elapsed < 0.25
