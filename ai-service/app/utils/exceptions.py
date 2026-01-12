"""Exception utilities for narrow exception catching.

Jan 12, 2026: Created as part of P2P Network & Master Loop Fundamental Fixes.

This module provides exception type tuples for use in narrow exception handlers,
replacing broad `except Exception:` with specific exception types. This allows
programming errors (NameError, AttributeError, etc.) to bubble up immediately
while still handling expected operational errors gracefully.

Usage:
    from app.utils.exceptions import NETWORK_ERRORS, PARSE_ERRORS, DB_ERRORS

    # Network operations
    try:
        await fetch_data(url)
    except NETWORK_ERRORS as e:
        logger.warning(f"Network error: {e}")

    # JSON parsing
    try:
        data = json.loads(raw)
    except PARSE_ERRORS as e:
        logger.warning(f"Parse error: {e}")

    # Database operations
    try:
        conn.execute(sql)
    except DB_ERRORS as e:
        logger.error(f"Database error: {e}")
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

# =============================================================================
# Exception Type Tuples
# =============================================================================

# Network-related exceptions to catch for I/O operations
# Use for: HTTP requests, socket operations, P2P communication
NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,      # Connection refused, reset, aborted
    TimeoutError,         # Socket/connect timeout
    OSError,              # Low-level I/O errors (includes socket.error)
    asyncio.TimeoutError, # Async operation timeout
)

# JSON/parsing exceptions for data deserialization
# Use for: JSON decoding, config parsing, message handling
PARSE_ERRORS: tuple[type[BaseException], ...] = (
    json.JSONDecodeError,  # Malformed JSON
    KeyError,              # Missing expected key
    TypeError,             # Wrong type in data structure
    ValueError,            # Invalid value format
)

# Database exceptions for SQLite operations
# Use for: Database queries, connections, transactions
DB_ERRORS: tuple[type[BaseException], ...] = (
    sqlite3.Error,              # Base SQLite error (catches all subtypes)
    sqlite3.OperationalError,   # Database locked, I/O error
    sqlite3.IntegrityError,     # Constraint violation
)

# File system exceptions for disk operations
# Use for: File reads, writes, path operations
FS_ERRORS: tuple[type[BaseException], ...] = (
    OSError,           # File not found, permission denied, etc.
    IOError,           # I/O failures
    PermissionError,   # Access denied
    FileNotFoundError, # Missing file
)

# Import exceptions for optional module loading
# Use for: Optional dependency checks, lazy imports
IMPORT_ERRORS: tuple[type[BaseException], ...] = (
    ImportError,
    ModuleNotFoundError,
)

# Process/subprocess exceptions
# Use for: External command execution, process management
PROCESS_ERRORS: tuple[type[BaseException], ...] = (
    OSError,                   # Process-related OS errors
    PermissionError,           # Permission to execute
    FileNotFoundError,         # Command not found
)


# =============================================================================
# Utility Functions
# =============================================================================

def log_and_continue(
    e: BaseException,
    context: str,
    logger_instance: logging.Logger,
    level: int = logging.WARNING,
) -> None:
    """Log exception with context, allowing the caller to continue.

    Use this for expected errors that should not crash the program.

    Args:
        e: The exception that was caught
        context: Short description of the operation (e.g., "peer_sync")
        logger_instance: Logger to use for logging
        level: Logging level (default: WARNING)

    Example:
        try:
            sync_data()
        except NETWORK_ERRORS as e:
            log_and_continue(e, "peer_sync", logger)
    """
    logger_instance.log(
        level,
        f"[{context}] Caught {type(e).__name__}: {e}",
    )


def is_transient_error(e: BaseException) -> bool:
    """Check if an exception is likely transient and worth retrying.

    Returns True for network issues, timeouts, and temporary I/O errors
    that might succeed on retry. Returns False for programming errors,
    permission issues, and other persistent problems.

    Args:
        e: The exception to check

    Returns:
        True if the error is likely transient
    """
    if isinstance(e, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    if isinstance(e, OSError):
        # Some OSError codes are transient (connection reset, temp unavailable)
        # Others are persistent (permission denied, file not found)
        import errno
        transient_codes = {
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.ETIMEDOUT,
            errno.EAGAIN,
            errno.EWOULDBLOCK,
            errno.EINTR,
        }
        return getattr(e, 'errno', None) in transient_codes
    if isinstance(e, sqlite3.OperationalError):
        # "database is locked" is transient
        return "locked" in str(e).lower()
    return False


class ExceptionContext:
    """Context manager for exception handling with automatic logging.

    Usage:
        with ExceptionContext("peer_sync", logger, NETWORK_ERRORS):
            sync_data()
        # Exceptions in NETWORK_ERRORS are logged and suppressed
        # Other exceptions propagate normally

        # Or with re-raise:
        with ExceptionContext("peer_sync", logger, NETWORK_ERRORS, reraise=True):
            sync_data()
        # Exceptions are logged then re-raised
    """

    def __init__(
        self,
        context: str,
        logger_instance: logging.Logger,
        exception_types: tuple[type[BaseException], ...],
        reraise: bool = False,
        level: int = logging.WARNING,
    ) -> None:
        self.context = context
        self.logger = logger_instance
        self.exception_types = exception_types
        self.reraise = reraise
        self.level = level

    def __enter__(self) -> "ExceptionContext":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None and issubclass(exc_type, self.exception_types):
            log_and_continue(exc_val, self.context, self.logger, self.level)
            return not self.reraise
        return False
