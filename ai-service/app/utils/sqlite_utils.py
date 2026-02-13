"""SQLite connection utilities with memory-safe defaults.

February 2026: Added after diagnosing kernel panics caused by unbounded SQLite
memory usage on the coordinator node (128 GB Mac Studio). Without pragmas,
multi-GB game databases (up to 11 GB) caused 34+ GB of disk writes and OOM.
"""
import sqlite3
from pathlib import Path


def connect_safe(
    db_path: str | Path,
    *,
    timeout: float = 30.0,
    cache_size_kb: int = 65536,  # 64 MB
    mmap_size: int = 67108864,   # 64 MB
    wal_mode: bool = True,
    row_factory: type | None = sqlite3.Row,
) -> sqlite3.Connection:
    """Open a SQLite connection with memory-limiting pragmas.

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
        Configured sqlite3.Connection.
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
    return conn
