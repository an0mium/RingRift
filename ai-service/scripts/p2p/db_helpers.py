"""Centralized SQLite connection helpers for scripts/p2p/.

February 2026: Phase 1.1 - Prevents fd exhaustion by routing all P2P layer
SQLite connections through the global SQLiteConnectionLimiter.

Usage:
    from scripts.p2p.db_helpers import p2p_db_connection, p2p_db_query

    # Context manager for multi-statement operations
    with p2p_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM peers")
        conn.commit()

    # One-shot convenience query
    rows = p2p_db_query(db_path, "SELECT COUNT(*) FROM games")
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import the centralized connection limiter
from app.distributed.db_utils import get_connection_limiter


@contextmanager
def p2p_db_connection(
    db_path: str | Path,
    timeout: float = 30.0,
    row_factory: bool = False,
):
    """Context manager for SQLite connections with global fd limiting.

    Acquires a slot from SQLiteConnectionLimiter before opening the connection,
    sets WAL mode and NORMAL synchronous pragmas, and releases the slot on exit.

    Args:
        db_path: Path to SQLite database file.
        timeout: Timeout for both slot acquisition and SQLite busy_timeout.
        row_factory: If True, set row_factory to sqlite3.Row.

    Yields:
        sqlite3.Connection with WAL mode and NORMAL synchronous.

    Raises:
        TimeoutError: If a connection slot cannot be acquired.
    """
    db_path_str = str(db_path)
    limiter = get_connection_limiter()
    acquired = False
    conn = None

    try:
        if not limiter.acquire(db_path_str, timeout=timeout):
            raise TimeoutError(
                f"Could not acquire SQLite connection slot for {db_path_str} "
                f"(limit: {limiter.MAX_CONNECTIONS})"
            )
        acquired = True

        conn = sqlite3.connect(db_path_str, timeout=timeout)
        if row_factory:
            conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
    finally:
        if conn is not None:
            conn.close()
        if acquired:
            limiter.release()


def p2p_db_query(
    db_path: str | Path,
    query: str,
    params: tuple[Any, ...] | None = None,
    timeout: float = 30.0,
) -> list[tuple[Any, ...]]:
    """Execute a one-shot read query and return all rows.

    Args:
        db_path: Path to SQLite database file.
        query: SQL query to execute.
        params: Optional query parameters.
        timeout: Timeout for slot acquisition and SQLite.

    Returns:
        List of result rows as tuples.
    """
    with p2p_db_connection(db_path, timeout=timeout) as conn:
        cursor = conn.execute(query, params or ())
        return cursor.fetchall()
