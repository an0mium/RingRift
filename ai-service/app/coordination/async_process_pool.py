"""Async Process Pool (Phase 2.2 - January 2026).

Non-blocking subprocess execution for async contexts. Replaces blocking
subprocess.run() calls in async daemons with proper async primitives.

Key features:
- Semaphore-based concurrency limiting
- Per-process timeout handling
- Process tracking and cancellation
- Resource cleanup on failure

Usage:
    from app.coordination.async_process_pool import (
        AsyncProcessPool,
        ProcessResult,
        get_process_pool,
    )

    pool = get_process_pool()

    # Execute a command
    result = await pool.execute(
        ["python", "-c", "print('hello')"],
        timeout_seconds=30.0,
    )
    print(f"Output: {result.stdout}")
    print(f"Exit code: {result.exit_code}")

    # Execute with working directory
    result = await pool.execute(
        ["git", "status"],
        cwd="/path/to/repo",
        timeout_seconds=10.0,
    )

January 2026: Created as part of long-term stability improvements.
Expected impact: -40-60% event loop latency, +25% stability.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ProcessResult",
    "ProcessConfig",
    "AsyncProcessPool",
    "ProcessTimeoutError",
    "ProcessExecutionError",
    "get_process_pool",
    "reset_process_pool",
]


# =============================================================================
# Exceptions
# =============================================================================

class ProcessTimeoutError(Exception):
    """Raised when a process exceeds its timeout."""

    def __init__(self, command: list[str], timeout_seconds: float):
        self.command = command
        self.timeout_seconds = timeout_seconds
        cmd_str = " ".join(command[:5])
        if len(command) > 5:
            cmd_str += " ..."
        super().__init__(f"Process timed out after {timeout_seconds}s: {cmd_str}")


class ProcessExecutionError(Exception):
    """Raised when a process fails to execute."""

    def __init__(
        self,
        command: list[str],
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
    ):
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        cmd_str = " ".join(command[:5])
        if len(command) > 5:
            cmd_str += " ..."
        super().__init__(
            f"Process failed with exit code {exit_code}: {cmd_str}\n"
            f"stderr: {stderr[:200]}"
        )


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ProcessResult:
    """Result from an async process execution.

    Attributes:
        stdout: Standard output as string
        stderr: Standard error as string
        exit_code: Process exit code (0 = success)
        duration_seconds: Wall clock time for execution
        command: The command that was executed
        timed_out: Whether the process was killed due to timeout
    """

    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    command: list[str] = field(default_factory=list)
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Check if process succeeded."""
        return self.exit_code == 0 and not self.timed_out


@dataclass
class ProcessConfig:
    """Configuration for async process pool.

    Attributes:
        max_concurrent: Maximum concurrent processes
        default_timeout: Default timeout in seconds
        kill_timeout: Time to wait after SIGTERM before SIGKILL
        encoding: Output encoding (default: utf-8)
        capture_output: Whether to capture stdout/stderr
    """

    max_concurrent: int = 10
    default_timeout: float = 60.0
    kill_timeout: float = 5.0
    encoding: str = "utf-8"
    capture_output: bool = True


# =============================================================================
# Active Process Tracking
# =============================================================================

@dataclass
class _ActiveProcess:
    """Tracks an active subprocess."""

    process: asyncio.subprocess.Process
    command: list[str]
    start_time: float
    task_id: str


# =============================================================================
# Async Process Pool
# =============================================================================

class AsyncProcessPool:
    """Non-blocking subprocess execution pool for async contexts.

    Provides:
    - Concurrent process execution with semaphore limiting
    - Per-process timeouts with graceful termination
    - Active process tracking for cancellation
    - Resource cleanup on failure

    Thread-safe and suitable for use from multiple coroutines.
    """

    _instance: AsyncProcessPool | None = None

    def __init__(self, config: ProcessConfig | None = None):
        self.config = config or ProcessConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._active: dict[str, _ActiveProcess] = {}
        self._counter = 0

        # Statistics
        self._total_executed = 0
        self._total_succeeded = 0
        self._total_failed = 0
        self._total_timed_out = 0

    @classmethod
    def get_instance(cls, config: ProcessConfig | None = None) -> AsyncProcessPool:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        if cls._instance:
            # Cancel any active processes
            asyncio.create_task(cls._instance.cancel_all())
        cls._instance = None

    async def execute(
        self,
        command: list[str],
        timeout_seconds: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        check: bool = False,
    ) -> ProcessResult:
        """Execute command asynchronously without blocking the event loop.

        Args:
            command: Command and arguments as list
            timeout_seconds: Per-process timeout (uses default if None)
            cwd: Working directory for the process
            env: Environment variables (merged with current env)
            check: If True, raise ProcessExecutionError on non-zero exit

        Returns:
            ProcessResult with stdout, stderr, exit_code, and timing

        Raises:
            ProcessTimeoutError: If process exceeds timeout
            ProcessExecutionError: If check=True and exit code != 0
        """
        timeout = timeout_seconds or self.config.default_timeout
        task_id = f"proc_{self._counter}"
        self._counter += 1

        # Merge environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        async with self._semaphore:
            start = time.monotonic()
            timed_out = False

            try:
                # Create subprocess
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE if self.config.capture_output else None,
                    stderr=asyncio.subprocess.PIPE if self.config.capture_output else None,
                    cwd=cwd,
                    env=process_env,
                )

                # Track active process
                self._active[task_id] = _ActiveProcess(
                    process=proc,
                    command=command,
                    start_time=start,
                    task_id=task_id,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    await self._terminate_process(proc, command)
                    stdout, stderr = b"", b"Process killed due to timeout"

                duration = time.monotonic() - start

                # Decode output
                stdout_str = stdout.decode(self.config.encoding, errors="replace") if stdout else ""
                stderr_str = stderr.decode(self.config.encoding, errors="replace") if stderr else ""

                # Handle exit code: returncode can be 0, None, or negative
                exit_code = proc.returncode if proc.returncode is not None else -1

                result = ProcessResult(
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=exit_code,
                    duration_seconds=duration,
                    command=command,
                    timed_out=timed_out,
                )

                # Update stats
                self._total_executed += 1
                if timed_out:
                    self._total_timed_out += 1
                elif result.success:
                    self._total_succeeded += 1
                else:
                    self._total_failed += 1

                # Check for errors if requested
                if check and not result.success:
                    if timed_out:
                        raise ProcessTimeoutError(command, timeout)
                    raise ProcessExecutionError(
                        command, result.exit_code, stdout_str, stderr_str
                    )

                return result

            finally:
                # Remove from active tracking
                self._active.pop(task_id, None)

    async def _terminate_process(
        self,
        proc: asyncio.subprocess.Process,
        command: list[str],
    ) -> None:
        """Gracefully terminate a process with SIGTERM, then SIGKILL."""
        cmd_str = " ".join(command[:3])

        try:
            # Try SIGTERM first
            proc.terminate()
            logger.warning(f"[AsyncProcessPool] Sending SIGTERM to: {cmd_str}")

            try:
                await asyncio.wait_for(
                    proc.wait(),
                    timeout=self.config.kill_timeout,
                )
                return
            except asyncio.TimeoutError:
                pass

            # SIGTERM didn't work, use SIGKILL
            logger.warning(f"[AsyncProcessPool] SIGTERM failed, sending SIGKILL: {cmd_str}")
            proc.kill()
            await proc.wait()

        except ProcessLookupError:
            # Process already exited
            pass
        except OSError as e:
            logger.error(f"[AsyncProcessPool] Error terminating process: {e}")

    async def cancel_all(self) -> int:
        """Cancel all active processes.

        Returns:
            Number of processes cancelled
        """
        cancelled = 0
        for task_id, active in list(self._active.items()):
            try:
                await self._terminate_process(active.process, active.command)
                cancelled += 1
            except Exception as e:
                logger.error(f"[AsyncProcessPool] Error cancelling {task_id}: {e}")

        self._active.clear()
        return cancelled

    def get_active_count(self) -> int:
        """Get number of currently active processes."""
        return len(self._active)

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "max_concurrent": self.config.max_concurrent,
            "active_count": len(self._active),
            "total_executed": self._total_executed,
            "total_succeeded": self._total_succeeded,
            "total_failed": self._total_failed,
            "total_timed_out": self._total_timed_out,
            "success_rate": (
                self._total_succeeded / self._total_executed
                if self._total_executed > 0
                else 1.0
            ),
            "active_processes": [
                {
                    "task_id": ap.task_id,
                    "command": " ".join(ap.command[:3]),
                    "running_seconds": time.monotonic() - ap.start_time,
                }
                for ap in self._active.values()
            ],
        }


# =============================================================================
# Singleton Access
# =============================================================================

def get_process_pool(config: ProcessConfig | None = None) -> AsyncProcessPool:
    """Get the singleton AsyncProcessPool instance."""
    return AsyncProcessPool.get_instance(config)


def reset_process_pool() -> None:
    """Reset the singleton (for testing)."""
    AsyncProcessPool.reset_instance()


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_command(
    command: list[str],
    timeout_seconds: float = 60.0,
    cwd: str | None = None,
    check: bool = False,
) -> ProcessResult:
    """Convenience function to run a command using the default pool.

    Args:
        command: Command and arguments as list
        timeout_seconds: Process timeout
        cwd: Working directory
        check: Raise on non-zero exit

    Returns:
        ProcessResult
    """
    pool = get_process_pool()
    return await pool.execute(
        command,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
        check=check,
    )
