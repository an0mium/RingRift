"""Async utilities for safe task management, subprocess execution, and SQLite.

This module provides utilities for:
- Managing asyncio tasks safely (preventing GC and unhandled exceptions)
- Running subprocesses asynchronously without blocking the event loop
- Executing SQLite operations asynchronously without blocking the event loop

Use `async_subprocess_run()` instead of `subprocess.run()` in async contexts.
Use `async_sqlite_execute()` instead of `sqlite3.connect().execute()` in async contexts.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import subprocess
from collections.abc import Callable, Coroutine, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
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


# =============================================================================
# Async SQLite Utilities
# =============================================================================


class SqliteError(Exception):
    """Error raised when async SQLite execution fails."""

    def __init__(self, message: str, db_path: str | Path | None = None):
        super().__init__(message)
        self.db_path = str(db_path) if db_path else None


@dataclass
class SqliteResult:
    """Result of async SQLite execution.

    Attributes:
        rows: List of result rows (for SELECT queries).
        rowcount: Number of affected rows (for INSERT/UPDATE/DELETE).
        lastrowid: ID of last inserted row (for INSERT).
    """

    rows: list[tuple[Any, ...]]
    rowcount: int
    lastrowid: int | None

    @property
    def first(self) -> tuple[Any, ...] | None:
        """Get the first row, or None if empty."""
        return self.rows[0] if self.rows else None

    @property
    def scalar(self) -> Any:
        """Get single value from first row, first column."""
        if self.rows and self.rows[0]:
            return self.rows[0][0]
        return None


def _execute_sqlite_sync(
    db_path: str | Path,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
    timeout: float = 30.0,
    row_factory: Callable[[sqlite3.Cursor, tuple[Any, ...]], Any] | None = None,
) -> SqliteResult:
    """Synchronous SQLite execution (runs in thread pool)."""
    with sqlite3.connect(str(db_path), timeout=timeout) as conn:
        if row_factory:
            conn.row_factory = row_factory
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        rows = cursor.fetchall()
        conn.commit()
        return SqliteResult(
            rows=rows,
            rowcount=cursor.rowcount,
            lastrowid=cursor.lastrowid,
        )


def _executemany_sqlite_sync(
    db_path: str | Path,
    query: str,
    params_seq: Sequence[tuple[Any, ...] | dict[str, Any]],
    timeout: float = 30.0,
) -> SqliteResult:
    """Synchronous SQLite executemany (runs in thread pool)."""
    with sqlite3.connect(str(db_path), timeout=timeout) as conn:
        cursor = conn.cursor()
        cursor.executemany(query, params_seq)
        conn.commit()
        return SqliteResult(
            rows=[],
            rowcount=cursor.rowcount,
            lastrowid=cursor.lastrowid,
        )


async def async_sqlite_execute(
    db_path: str | Path,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
    row_factory: Callable[[sqlite3.Cursor, tuple[Any, ...]], Any] | None = None,
) -> SqliteResult:
    """Execute a SQLite query asynchronously without blocking the event loop.

    This is the async equivalent of `sqlite3.connect().execute()`. Use this
    instead of blocking SQLite calls in async contexts.

    Args:
        db_path: Path to the SQLite database file.
        query: SQL query to execute.
        params: Query parameters (tuple for positional, dict for named).
        timeout: SQLite connection timeout in seconds.
        row_factory: Optional row factory (e.g., sqlite3.Row for dict-like access).

    Returns:
        SqliteResult with rows, rowcount, and lastrowid.

    Raises:
        SqliteError: If the query fails.
        sqlite3.Error: For SQLite-specific errors.

    Example:
        # SELECT query
        result = await async_sqlite_execute(
            "data/games.db",
            "SELECT * FROM games WHERE board_type = ?",
            ("hex8",),
        )
        for row in result.rows:
            print(row)

        # INSERT query
        result = await async_sqlite_execute(
            "data/games.db",
            "INSERT INTO games (board_type, num_players) VALUES (?, ?)",
            ("hex8", 2),
        )
        print(f"Inserted row ID: {result.lastrowid}")

        # With dict-like row access
        result = await async_sqlite_execute(
            "data/games.db",
            "SELECT * FROM games LIMIT 1",
            row_factory=sqlite3.Row,
        )
        if result.first:
            print(result.first["board_type"])
    """
    try:
        return await asyncio.to_thread(
            _execute_sqlite_sync,
            db_path,
            query,
            params,
            timeout,
            row_factory,
        )
    except sqlite3.Error as e:
        raise SqliteError(f"SQLite error: {e}", db_path) from e


async def async_sqlite_executemany(
    db_path: str | Path,
    query: str,
    params_seq: Sequence[tuple[Any, ...] | dict[str, Any]],
    *,
    timeout: float = 30.0,
) -> SqliteResult:
    """Execute a SQLite query with multiple parameter sets asynchronously.

    This is the async equivalent of `cursor.executemany()`. Use for bulk
    INSERT/UPDATE operations.

    Args:
        db_path: Path to the SQLite database file.
        query: SQL query to execute.
        params_seq: Sequence of parameter tuples/dicts.
        timeout: SQLite connection timeout in seconds.

    Returns:
        SqliteResult with rowcount.

    Example:
        # Bulk insert
        result = await async_sqlite_executemany(
            "data/games.db",
            "INSERT INTO games (board_type, num_players) VALUES (?, ?)",
            [("hex8", 2), ("square8", 4), ("hexagonal", 3)],
        )
        print(f"Inserted {result.rowcount} rows")
    """
    try:
        return await asyncio.to_thread(
            _executemany_sqlite_sync,
            db_path,
            query,
            params_seq,
            timeout,
        )
    except sqlite3.Error as e:
        raise SqliteError(f"SQLite error: {e}", db_path) from e


async def async_sqlite_fetchone(
    db_path: str | Path,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[Any, ...] | None:
    """Execute a query and return the first row.

    Convenience wrapper around async_sqlite_execute for single-row queries.

    Args:
        db_path: Path to the SQLite database file.
        query: SQL query to execute.
        params: Query parameters.
        timeout: SQLite connection timeout in seconds.

    Returns:
        First row as tuple, or None if no results.

    Example:
        row = await async_sqlite_fetchone(
            "data/games.db",
            "SELECT COUNT(*) FROM games WHERE board_type = ?",
            ("hex8",),
        )
        if row:
            print(f"Count: {row[0]}")
    """
    result = await async_sqlite_execute(db_path, query, params, timeout=timeout)
    return result.first


async def async_sqlite_fetchall(
    db_path: str | Path,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> list[tuple[Any, ...]]:
    """Execute a query and return all rows.

    Convenience wrapper around async_sqlite_execute for multi-row queries.

    Args:
        db_path: Path to the SQLite database file.
        query: SQL query to execute.
        params: Query parameters.
        timeout: SQLite connection timeout in seconds.

    Returns:
        List of rows as tuples.

    Example:
        rows = await async_sqlite_fetchall(
            "data/games.db",
            "SELECT board_type, num_players FROM games",
        )
        for board_type, num_players in rows:
            print(f"{board_type}: {num_players}p")
    """
    result = await async_sqlite_execute(db_path, query, params, timeout=timeout)
    return result.rows


async def async_sqlite_scalar(
    db_path: str | Path,
    query: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
    default: Any = None,
) -> Any:
    """Execute a query and return a single scalar value.

    Convenience wrapper for queries that return a single value (e.g., COUNT).

    Args:
        db_path: Path to the SQLite database file.
        query: SQL query to execute.
        params: Query parameters.
        timeout: SQLite connection timeout in seconds.
        default: Value to return if no results.

    Returns:
        Single value from first row, first column, or default.

    Example:
        count = await async_sqlite_scalar(
            "data/games.db",
            "SELECT COUNT(*) FROM games",
            default=0,
        )
        print(f"Total games: {count}")
    """
    result = await async_sqlite_execute(db_path, query, params, timeout=timeout)
    return result.scalar if result.scalar is not None else default
