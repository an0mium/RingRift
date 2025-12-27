#!/usr/bin/env python3
"""Sync Integrity - Checksum validation and data integrity verification.

This module provides comprehensive integrity checking for sync operations:

1. File checksum computation (SHA256, streaming for large files)
2. SQLite database integrity verification (PRAGMA integrity_check)
3. Full sync integrity reports comparing source and target
4. Support for multiple hash algorithms
5. Structured error reporting

This consolidates checksum validation functionality from:
- app.distributed.unified_data_sync._compute_file_checksum
- app.distributed.p2p_sync_client checksum verification
- app.coordination.transfer_verification (higher-level wrapper)

Usage:
    from app.coordination.sync_integrity import (
        compute_file_checksum,
        compute_db_checksum,
        verify_checksum,
        check_sqlite_integrity,
        verify_sync_integrity,
        IntegrityReport,
    )

    # Basic file checksum
    checksum = compute_file_checksum(Path("data.db"))

    # Verify against expected
    if verify_checksum(Path("data.db"), expected_checksum):
        print("File is valid")

    # Check SQLite database integrity
    is_valid, errors = check_sqlite_integrity(Path("data.db"))

    # Full sync verification
    report = verify_sync_integrity(source_path, target_path)
    if report.is_valid:
        print("Sync successful")
    else:
        print(f"Errors: {report.errors}")
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "LARGE_CHUNK_SIZE",
    "IntegrityCheckResult",
    "IntegrityReport",
    "check_sqlite_integrity",
    "compute_db_checksum",
    "compute_file_checksum",
    "verify_checksum",
    "verify_sync_integrity",
]

logger = logging.getLogger(__name__)

# Chunk sizes for streaming checksum computation
DEFAULT_CHUNK_SIZE = 8192  # 8KB - standard for most files
LARGE_CHUNK_SIZE = 65536  # 64KB - better for large files

# Supported hash algorithms
HashAlgorithm = Literal["sha256", "sha1", "md5", "blake2b"]


@dataclass
class IntegrityCheckResult:
    """Result of a SQLite integrity check."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    check_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "check_time": self.check_time,
        }


@dataclass
class IntegrityReport:
    """Comprehensive integrity verification report for sync operations."""

    source_path: str
    target_path: str
    is_valid: bool
    source_checksum: str
    target_checksum: str
    source_size: int
    target_size: int
    checksum_match: bool
    size_match: bool
    db_integrity_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verification_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "is_valid": self.is_valid,
            "source_checksum": self.source_checksum[:16] + "..." if self.source_checksum else "",
            "target_checksum": self.target_checksum[:16] + "..." if self.target_checksum else "",
            "source_size": self.source_size,
            "target_size": self.target_size,
            "checksum_match": self.checksum_match,
            "size_match": self.size_match,
            "db_integrity_valid": self.db_integrity_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "verification_time": round(self.verification_time, 3),
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.is_valid:
            return f"✓ Valid sync: {self.target_path} ({self.target_size} bytes)"
        else:
            error_summary = "; ".join(self.errors[:3])
            return f"✗ Invalid sync: {self.target_path} - {error_summary}"


# =============================================================================
# Core Checksum Functions
# =============================================================================


def compute_file_checksum(
    path: Path,
    algorithm: HashAlgorithm = "sha256",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Compute checksum of a file using streaming read.

    Uses chunk-based reading to handle large files efficiently without
    loading the entire file into memory.

    Args:
        path: Path to the file
        algorithm: Hash algorithm to use (sha256, sha1, md5, blake2b)
        chunk_size: Size of chunks to read (default: 8KB)
                   Use LARGE_CHUNK_SIZE (64KB) for files > 100MB

    Returns:
        Hex-encoded hash string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm is not supported
        PermissionError: If file cannot be read

    Example:
        # Standard usage
        checksum = compute_file_checksum(Path("data.db"))

        # Large file with bigger chunks
        checksum = compute_file_checksum(
            Path("huge.db"),
            chunk_size=LARGE_CHUNK_SIZE
        )
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
    except PermissionError as e:
        raise PermissionError(f"Cannot read file: {path}") from e

    return hasher.hexdigest()


def compute_db_checksum(
    db_path: Path,
    algorithm: HashAlgorithm = "sha256",
) -> str:
    """Compute checksum of a SQLite database file.

    This is a specialized version of compute_file_checksum that:
    1. Uses large chunks for better performance on database files
    2. Logs warnings if database is locked or corrupted
    3. Falls back gracefully on errors

    Args:
        db_path: Path to SQLite database file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hex-encoded hash string, or empty string on error

    Example:
        checksum = compute_db_checksum(Path("games.db"))
    """
    if not db_path.exists():
        logger.warning(f"[SyncIntegrity] Database not found: {db_path}")
        return ""

    try:
        # Use large chunks for database files (typically several MB to GB)
        return compute_file_checksum(db_path, algorithm=algorithm, chunk_size=LARGE_CHUNK_SIZE)
    except Exception as e:
        logger.warning(f"[SyncIntegrity] Failed to compute checksum for {db_path}: {e}")
        return ""


def verify_checksum(
    path: Path,
    expected: str,
    algorithm: HashAlgorithm = "sha256",
) -> bool:
    """Verify a file matches an expected checksum.

    Args:
        path: Path to file to verify
        expected: Expected hex-encoded checksum
        algorithm: Hash algorithm used for expected checksum

    Returns:
        True if checksum matches, False otherwise

    Example:
        if verify_checksum(Path("data.db"), expected_hash):
            print("File is valid")
        else:
            print("Checksum mismatch - file may be corrupted")
    """
    if not expected:
        logger.warning(f"[SyncIntegrity] No expected checksum provided for {path}")
        return False

    if not path.exists():
        logger.warning(f"[SyncIntegrity] File not found for verification: {path}")
        return False

    try:
        actual = compute_file_checksum(path, algorithm=algorithm)
        match = actual == expected

        if not match:
            logger.warning(
                f"[SyncIntegrity] Checksum mismatch for {path}: "
                f"expected {expected[:16]}..., got {actual[:16]}..."
            )

        return match

    except Exception as e:
        logger.error(f"[SyncIntegrity] Error verifying checksum for {path}: {e}")
        return False


# =============================================================================
# SQLite Integrity Checking
# =============================================================================


def check_sqlite_integrity(db_path: Path) -> tuple[bool, list[str]]:
    """Run SQLite PRAGMA integrity_check on a database.

    This performs a comprehensive integrity check of the database file,
    including:
    - B-tree structure validation
    - Page consistency checks
    - Index verification
    - Foreign key constraint validation (if enabled)

    Args:
        db_path: Path to SQLite database file

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if database passes integrity check
        - error_messages: List of error messages (empty if valid)

    Example:
        is_valid, errors = check_sqlite_integrity(Path("games.db"))
        if not is_valid:
            print(f"Database corrupted: {errors}")
    """
    if not db_path.exists():
        return False, [f"Database file not found: {db_path}"]

    if not db_path.is_file():
        return False, [f"Path is not a file: {db_path}"]

    errors = []

    try:
        # Open read-only to avoid locking issues
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        cursor = conn.cursor()

        # Run integrity check
        cursor.execute("PRAGMA integrity_check")
        results = cursor.fetchall()

        conn.close()

        # Check results
        # SQLite returns a single row with "ok" if everything is fine
        # Otherwise, returns multiple rows describing errors
        if len(results) == 1 and results[0][0] == "ok":
            return True, []
        else:
            errors = [str(row[0]) for row in results]
            logger.warning(f"[SyncIntegrity] Database {db_path} integrity check failed: {errors}")
            return False, errors

    except sqlite3.DatabaseError as e:
        error_msg = f"Database error: {e}"
        logger.error(f"[SyncIntegrity] {error_msg} for {db_path}")
        return False, [error_msg]

    except sqlite3.OperationalError as e:
        error_msg = f"Database locked or inaccessible: {e}"
        logger.warning(f"[SyncIntegrity] {error_msg} for {db_path}")
        return False, [error_msg]

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(f"[SyncIntegrity] {error_msg} for {db_path}")
        return False, [error_msg]


# =============================================================================
# Full Sync Integrity Verification
# =============================================================================


def verify_sync_integrity(
    source: Path,
    target: Path,
    algorithm: HashAlgorithm = "sha256",
    check_db: bool = True,
) -> IntegrityReport:
    """Perform full integrity verification for a sync operation.

    This comprehensive check verifies:
    1. Both files exist
    2. File sizes match
    3. Checksums match
    4. SQLite database integrity (if target is .db file and check_db=True)

    Args:
        source: Source file path
        target: Target file path
        algorithm: Hash algorithm to use (default: sha256)
        check_db: If True, run PRAGMA integrity_check on .db files

    Returns:
        IntegrityReport with detailed verification results

    Example:
        report = verify_sync_integrity(
            source=Path("remote/games.db"),
            target=Path("local/games.db")
        )

        if report.is_valid:
            print("Sync verified successfully")
        else:
            print(f"Sync failed: {report.summary()}")
            for error in report.errors:
                print(f"  - {error}")
    """
    start_time = time.time()
    errors = []
    warnings = []

    # Initialize with defaults
    source_checksum = ""
    target_checksum = ""
    source_size = 0
    target_size = 0
    checksum_match = False
    size_match = False
    db_integrity_valid = True  # Default to True if not checked

    # Check source exists
    if not source.exists():
        errors.append(f"Source file not found: {source}")
    else:
        try:
            source_size = source.stat().st_size
        except Exception as e:
            errors.append(f"Cannot stat source file: {e}")

    # Check target exists
    if not target.exists():
        errors.append(f"Target file not found: {target}")
    else:
        try:
            target_size = target.stat().st_size
        except Exception as e:
            errors.append(f"Cannot stat target file: {e}")

    # Compare sizes
    if source_size > 0 and target_size > 0:
        size_match = source_size == target_size
        if not size_match:
            errors.append(
                f"Size mismatch: source={source_size} bytes, target={target_size} bytes "
                f"(diff: {abs(source_size - target_size)} bytes)"
            )

    # Compute checksums
    if source.exists():
        try:
            # Use large chunks for better performance
            chunk_size = LARGE_CHUNK_SIZE if source_size > 1_000_000 else DEFAULT_CHUNK_SIZE
            source_checksum = compute_file_checksum(source, algorithm=algorithm, chunk_size=chunk_size)
        except Exception as e:
            errors.append(f"Cannot compute source checksum: {e}")

    if target.exists():
        try:
            # Use large chunks for better performance
            chunk_size = LARGE_CHUNK_SIZE if target_size > 1_000_000 else DEFAULT_CHUNK_SIZE
            target_checksum = compute_file_checksum(target, algorithm=algorithm, chunk_size=chunk_size)
        except Exception as e:
            errors.append(f"Cannot compute target checksum: {e}")

    # Compare checksums
    if source_checksum and target_checksum:
        checksum_match = source_checksum == target_checksum
        if not checksum_match:
            errors.append(
                f"Checksum mismatch: source={source_checksum[:16]}..., "
                f"target={target_checksum[:16]}..."
            )

    # Check SQLite database integrity
    is_db_file = target.suffix.lower() == ".db"
    if check_db and is_db_file and target.exists():
        try:
            db_integrity_valid, db_errors = check_sqlite_integrity(target)
            if not db_integrity_valid:
                errors.extend(f"SQLite integrity error: {err}" for err in db_errors)
        except Exception as e:
            warnings.append(f"Could not check database integrity: {e}")
            # Don't fail the whole verification if integrity check fails
            # (database might be locked, read-only, etc.)

    # Determine overall validity
    is_valid = (
        len(errors) == 0
        and size_match
        and checksum_match
        and db_integrity_valid
    )

    verification_time = time.time() - start_time

    report = IntegrityReport(
        source_path=str(source),
        target_path=str(target),
        is_valid=is_valid,
        source_checksum=source_checksum,
        target_checksum=target_checksum,
        source_size=source_size,
        target_size=target_size,
        checksum_match=checksum_match,
        size_match=size_match,
        db_integrity_valid=db_integrity_valid,
        errors=errors,
        warnings=warnings,
        verification_time=verification_time,
    )

    if not is_valid:
        logger.warning(f"[SyncIntegrity] {report.summary()}")
    else:
        logger.debug(f"[SyncIntegrity] ✓ {target} verified ({target_size} bytes, {verification_time:.2f}s)")

    return report


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "LARGE_CHUNK_SIZE",
    "IntegrityCheckResult",
    "IntegrityReport",
    "check_sqlite_integrity",
    "compute_db_checksum",
    "compute_file_checksum",
    "verify_checksum",
    "verify_sync_integrity",
]
