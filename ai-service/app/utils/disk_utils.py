"""Disk space utilities for preventing data loss when disk fills up.

This module provides utilities for checking available disk space before
performing write operations. Use these checks before database writes,
file saves, or other operations that could fail and lose data if the
disk is full.

Usage:
    from app.utils.disk_utils import check_disk_space_available, ensure_disk_space

    # Check if space is available (returns True/False)
    if check_disk_space_available("/path/to/db.sqlite", min_bytes=100_000_000):
        db.write_data(...)

    # Raise DiskSpaceError if insufficient space
    ensure_disk_space("/path/to/db.sqlite")  # Uses default 100MB threshold
    db.write_data(...)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default minimum free space: 100 MB
DEFAULT_MIN_BYTES = 100 * 1024 * 1024

__all__ = [
    "check_disk_space_available",
    "ensure_disk_space",
    "get_available_disk_space",
    "is_enospc_error",
    "handle_enospc_error",
    "DEFAULT_MIN_BYTES",
]

# Patterns that indicate ENOSPC/disk full errors in sqlite3 messages
ENOSPC_PATTERNS = (
    "disk full",
    "no space left",
    "database or disk is full",
    "disk i/o error",
    "enospc",
    "errno 28",  # ENOSPC on Linux
    "cannot allocate",
)


def get_available_disk_space(path: str | Path) -> int:
    """Get available disk space in bytes for the filesystem containing path.

    Args:
        path: File or directory path. If file doesn't exist, uses parent directory.

    Returns:
        Available space in bytes.

    Raises:
        OSError: If the path doesn't exist and parent directory can't be found.
    """
    path = Path(path)

    # Find the actual path to check (file, directory, or nearest existing parent)
    check_path = path
    while not check_path.exists():
        parent = check_path.parent
        if parent == check_path:
            # Reached root without finding existing path
            raise OSError(f"Cannot determine disk space: no existing path found for {path}")
        check_path = parent

    # Use os.statvfs on Unix-like systems (including macOS and Linux)
    # This is more portable than shutil.disk_usage for our use case
    try:
        stat = os.statvfs(check_path)
        # f_bavail = free blocks available to non-superuser
        # f_frsize = fragment size (usually same as block size)
        return stat.f_bavail * stat.f_frsize
    except AttributeError:
        # Windows fallback using shutil
        import shutil
        _, _, free = shutil.disk_usage(check_path)
        return free


def check_disk_space_available(
    path: str | Path,
    min_bytes: int = DEFAULT_MIN_BYTES,
) -> bool:
    """Check if at least min_bytes of disk space is available.

    This is a non-throwing check suitable for conditional logic.

    Args:
        path: File or directory path to check disk space for.
              If file doesn't exist, checks parent directory.
        min_bytes: Minimum required bytes (default: 100MB).

    Returns:
        True if at least min_bytes is available, False otherwise.
        Also returns False if space cannot be determined (with warning logged).

    Example:
        if check_disk_space_available(db_path):
            db.store_game(...)
        else:
            logger.error("Disk full, cannot store game")
    """
    try:
        available = get_available_disk_space(path)
        if available < min_bytes:
            logger.warning(
                "Low disk space: %d bytes available, %d bytes required at %s",
                available,
                min_bytes,
                path,
            )
            return False
        return True
    except OSError as e:
        logger.warning("Could not check disk space for %s: %s", path, e)
        return False


def ensure_disk_space(
    path: str | Path,
    min_bytes: int = DEFAULT_MIN_BYTES,
    operation: str = "write",
) -> None:
    """Ensure sufficient disk space is available, raising DiskSpaceError if not.

    This is a pre-condition check that should be called before write operations.
    It raises an exception that callers can handle gracefully.

    Args:
        path: File or directory path to check disk space for.
              If file doesn't exist, checks parent directory.
        min_bytes: Minimum required bytes (default: 100MB).
        operation: Description of the operation for error messages.

    Raises:
        DiskSpaceError: If insufficient disk space is available.
        OSError: If the path doesn't exist and cannot be checked.

    Example:
        from app.utils.disk_utils import ensure_disk_space

        def store_game(self, game_data):
            ensure_disk_space(self.db_path, operation="store game")
            # Proceed with database write...
    """
    from app.errors import DiskSpaceError

    try:
        available = get_available_disk_space(path)
    except OSError as e:
        raise DiskSpaceError(
            f"Cannot verify disk space before {operation}: {e}",
            context={"path": str(path), "operation": operation},
        ) from e

    if available < min_bytes:
        available_mb = available / (1024 * 1024)
        required_mb = min_bytes / (1024 * 1024)
        raise DiskSpaceError(
            f"Insufficient disk space for {operation}: "
            f"{available_mb:.1f}MB available, {required_mb:.1f}MB required",
            context={
                "path": str(path),
                "available_bytes": available,
                "required_bytes": min_bytes,
                "available_mb": round(available_mb, 1),
                "required_mb": round(required_mb, 1),
                "operation": operation,
            },
        )


def is_enospc_error(error: Exception) -> bool:
    """Check if an exception is caused by ENOSPC (disk full).

    This detects disk full errors from sqlite3.OperationalError and other
    low-level I/O errors. Use this to determine if an operation failed
    due to insufficient disk space rather than other causes.

    Args:
        error: The exception to check.

    Returns:
        True if the error indicates disk full, False otherwise.

    Example:
        try:
            conn.execute("INSERT INTO games ...")
        except sqlite3.OperationalError as e:
            if is_enospc_error(e):
                # Handle disk full specifically
                emit_disk_full_event()
                raise DiskSpaceError(...) from e
            raise  # Re-raise other errors
    """
    error_msg = str(error).lower()
    return any(pattern in error_msg for pattern in ENOSPC_PATTERNS)


def handle_enospc_error(
    error: Exception,
    path: str | Path,
    operation: str = "write",
    emit_event: bool = True,
) -> None:
    """Convert an ENOSPC error to DiskSpaceError and optionally emit event.

    This function should be called when you catch a sqlite3.OperationalError
    or OSError and determine it's caused by disk full. It:
    1. Logs the error with full context
    2. Optionally emits a DISK_FULL event for monitoring
    3. Raises DiskSpaceError with proper context

    Args:
        error: The original exception that triggered this.
        path: The file path where the error occurred.
        operation: Description of the operation that failed.
        emit_event: Whether to emit DISK_FULL event (default True).

    Raises:
        DiskSpaceError: Always raises with context from the original error.

    Example:
        try:
            conn.execute("INSERT INTO games ...")
            conn.commit()
        except sqlite3.OperationalError as e:
            if is_enospc_error(e):
                handle_enospc_error(e, db_path, "store game")
            raise
    """
    from app.errors import DiskSpaceError

    path = Path(path)

    # Get current disk state for context
    try:
        available = get_available_disk_space(path)
        available_mb = available / (1024 * 1024)
    except OSError:
        available = 0
        available_mb = 0.0

    logger.error(
        "ENOSPC error during %s at %s: %s (%.1f MB available)",
        operation,
        path,
        error,
        available_mb,
    )

    # Emit DISK_FULL event for monitoring/alerting
    # Dec 28, 2025: Added to enable cluster-wide disk monitoring
    if emit_event:
        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            bus.publish(
                "DISK_FULL",
                {
                    "path": str(path),
                    "operation": operation,
                    "available_bytes": available,
                    "available_mb": round(available_mb, 1),
                    "error": str(error),
                    "timestamp": time.time(),
                },
            )
        except Exception as emit_err:
            # Don't fail if event emission fails - the DiskSpaceError is what matters
            logger.debug("Failed to emit DISK_FULL event: %s", emit_err)

    raise DiskSpaceError(
        f"Disk full during {operation}: {error}",
        context={
            "path": str(path),
            "operation": operation,
            "available_bytes": available,
            "available_mb": round(available_mb, 1),
            "original_error": str(error),
        },
    ) from error
