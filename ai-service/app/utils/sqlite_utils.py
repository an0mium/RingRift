"""SQLite connection utilities with memory-safe defaults.

February 2026: Added after diagnosing kernel panics caused by unbounded SQLite
memory usage on the coordinator node (128 GB Mac Studio). Without pragmas,
multi-GB game databases (up to 11 GB) caused 34+ GB of disk writes and OOM.

IMPORTANT: In Python 3.11, sqlite3.Connection used as a context manager
(``with conn:``) only commits/rolls back but does NOT close the connection.
This caused massive connection leaks (1900+ open FDs) when ``connect_safe()``
was used with ``with`` statements. _SafeConnection wraps the connection to
auto-close on context exit.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union


class _SafeConnection:
    """Wrapper around sqlite3.Connection that closes on context manager exit.

    In Python < 3.12, sqlite3.Connection.__exit__ only commits/rolls back
    but does NOT close the connection. This wrapper ensures close() is called,
    which is critical because connect_safe() enables mmap_size=64MB per
    connection — leaked connections accumulate memory-mapped file handles.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # Proxy attribute access to the underlying connection
    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()

    def close(self) -> None:
        self._conn.close()


def connect_safe(
    db_path: Union[str, Path],
    *,
    timeout: float = 30.0,
    cache_size_kb: int = 65536,  # 64 MB
    mmap_size: int = 67108864,   # 64 MB
    wal_mode: bool = True,
    row_factory: Optional[type] = sqlite3.Row,
) -> _SafeConnection:
    """Open a SQLite connection with memory-limiting pragmas.

    Returns a _SafeConnection wrapper that auto-closes on context manager exit.
    Can also be used without ``with`` — call ``.close()`` explicitly.

    Default pragmas prevent unbounded cache/mmap growth that caused OOM:
    - cache_size: 64 MB (default is 2 MB but can balloon with large result sets)
    - mmap_size: 64 MB (prevents OS from mapping entire multi-GB databases)
    - wal_autocheckpoint: 4000 pages (~64 MB) to bound WAL file growth
    - synchronous=NORMAL for WAL mode performance

    Args:
        db_path: Path to the SQLite database file.
        timeout: Connection timeout in seconds.
        cache_size_kb: Maximum cache size in KB (negative PRAGMA value).
        mmap_size: Maximum memory-mapped I/O size in bytes.
        wal_mode: Whether to enable WAL journal mode.
        row_factory: Row factory for the connection (default: sqlite3.Row).

    Returns:
        _SafeConnection wrapping configured sqlite3.Connection.
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    if row_factory is not None:
        conn.row_factory = row_factory
    if wal_mode:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA cache_size=-{cache_size_kb}")
    conn.execute(f"PRAGMA mmap_size={mmap_size}")
    conn.execute("PRAGMA wal_autocheckpoint=4000")
    return _SafeConnection(conn)
