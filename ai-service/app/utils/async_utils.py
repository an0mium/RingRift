"""Async utilities for safe task management and subprocess execution.

This module provides utilities for:
- Managing asyncio tasks safely (preventing GC and unhandled exceptions)
- Running subprocesses asynchronously without blocking the event loop

Use `async_subprocess_run()` instead of `subprocess.run()` in async contexts.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Global set to hold references to background tasks
# This prevents garbage collection from cancelling them
_background_tasks: set[asyncio.Task[Any]] = set()

T = TypeVar("T")


def fire_and_forget(coro: Coroutine[Any, Any, T], name: str | None = None) -> asyncio.Task[T]:
    """Schedule a coroutine to run in the background without awaiting it.

    This is a safe alternative to `asyncio.create_task()` for fire-and-forget
    scenarios. The task is stored in a global set to prevent garbage collection,
    and exceptions are logged instead of being silently dropped.

    Args:
        coro: The coroutine to run in the background.
        name: Optional name for the task (for debugging).

    Returns:
        The created task (can be ignored for fire-and-forget usage).

    Example:
        # Instead of:
        asyncio.create_task(some_async_function())  # Risky: may be GC'd

        # Use:
        fire_and_forget(some_async_function())  # Safe: tracked and exceptions logged
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_task_done_callback)
    return task


def _task_done_callback(task: asyncio.Task[Any]) -> None:
    """Callback to handle task completion and cleanup."""
    _background_tasks.discard(task)
    try:
        # Check if task raised an exception
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Background task %s raised exception: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )
    except asyncio.CancelledError:
        # Task was cancelled - this is expected in shutdown scenarios
        logger.debug("Background task %s was cancelled", task.get_name())


def get_pending_tasks() -> int:
    """Get the number of pending background tasks."""
    return len(_background_tasks)


async def wait_all_background_tasks(timeout: float | None = None) -> None:
    """Wait for all background tasks to complete.

    Useful for graceful shutdown scenarios.

    Args:
        timeout: Maximum time to wait in seconds. None means wait indefinitely.
    """
    if not _background_tasks:
        return

    logger.info("Waiting for %d background tasks to complete...", len(_background_tasks))
    _done, pending = await asyncio.wait(
        _background_tasks,
        timeout=timeout,
        return_when=asyncio.ALL_COMPLETED,
    )

    if pending:
        logger.warning("Timed out waiting for %d tasks, cancelling...", len(pending))
        for task in pending:
            task.cancel()


def cancel_all_background_tasks() -> None:
    """Cancel all pending background tasks.

    Useful for immediate shutdown scenarios.
    """
    for task in _background_tasks:
        task.cancel()
    logger.info("Cancelled %d background tasks", len(_background_tasks))


# =============================================================================
# Async Subprocess Utilities
# =============================================================================


class SubprocessError(Exception):
    """Error raised when async subprocess execution fails."""

    def __init__(
        self,
        message: str,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class SubprocessTimeoutError(SubprocessError):
    """Error raised when subprocess times out."""

    pass


@dataclass
class SubprocessResult:
    """Result of async subprocess execution.

    Attributes:
        returncode: Exit code from the process (0 = success).
        stdout: Captured standard output as string.
        stderr: Captured standard error as string.
    """

    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if process exited successfully (returncode == 0)."""
        return self.returncode == 0


async def async_subprocess_run(
    cmd: Sequence[str],
    *,
    timeout: float = 60.0,
    check: bool = False,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Run a subprocess asynchronously without blocking the event loop.

    This is the async equivalent of `subprocess.run()`. Use this instead of
    `run_in_executor(subprocess.run)` for better efficiency and cleaner code.

    Args:
        cmd: Command and arguments as a sequence (e.g., ["ls", "-la"]).
        timeout: Maximum time to wait in seconds (default: 60).
        check: If True, raise SubprocessError on non-zero exit code.
        cwd: Working directory for the subprocess.
        env: Environment variables for the subprocess.

    Returns:
        SubprocessResult with returncode, stdout, and stderr.

    Raises:
        SubprocessTimeoutError: If the process exceeds the timeout.
        SubprocessError: If check=True and process exits with non-zero code.
        OSError: If the command cannot be executed.

    Example:
        # Basic usage
        result = await async_subprocess_run(["ls", "-la"])
        if result.success:
            print(result.stdout)

        # With timeout and error checking
        result = await async_subprocess_run(
            ["vastai", "show", "instances", "--raw"],
            timeout=30.0,
            check=True,
        )
    """
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        returncode = proc.returncode or 0

        result = SubprocessResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

        if check and returncode != 0:
            raise SubprocessError(
                f"Command {cmd[0]} failed with exit code {returncode}: {stderr}",
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )

        return result

    except asyncio.TimeoutError:
        # Kill the process on timeout
        if proc is not None:
            proc.kill()
            await proc.wait()
        raise SubprocessTimeoutError(
            f"Command {cmd[0]} timed out after {timeout}s",
            returncode=-1,
            stdout="",
            stderr="",
        )


async def async_subprocess_shell(
    cmd: str,
    *,
    timeout: float = 60.0,
    check: bool = False,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Run a shell command asynchronously.

    Similar to async_subprocess_run but executes via shell. Use with caution
    as shell commands can be vulnerable to injection attacks.

    Args:
        cmd: Shell command as a string.
        timeout: Maximum time to wait in seconds.
        check: If True, raise SubprocessError on non-zero exit code.
        cwd: Working directory for the subprocess.
        env: Environment variables for the subprocess.

    Returns:
        SubprocessResult with returncode, stdout, and stderr.

    Example:
        result = await async_subprocess_shell("ls -la | grep py")
    """
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        returncode = proc.returncode or 0

        result = SubprocessResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

        if check and returncode != 0:
            raise SubprocessError(
                f"Shell command failed with exit code {returncode}: {stderr}",
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )

        return result

    except asyncio.TimeoutError:
        if proc is not None:
            proc.kill()
            await proc.wait()
        raise SubprocessTimeoutError(
            f"Shell command timed out after {timeout}s",
            returncode=-1,
            stdout="",
            stderr="",
        )
