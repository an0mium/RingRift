"""ImportDaemonMixin - Consolidated import/download patterns for data daemons.

January 2026: Created as part of code consolidation to reduce duplication
across S3ImportDaemon, OWCImportDaemon, and NodeDataAgent (~200 LOC savings).

This mixin provides:
- _download_with_progress(): Download with progress tracking and verification
- _validate_import(): Validate imported file integrity
- _atomic_replace(): Atomically replace file after successful download
- _compute_checksum(): SHA256 checksum computation

Usage:
    from app.coordination.mixins.import_mixin import ImportDaemonMixin

    class MyImportDaemon(HandlerBase, ImportDaemonMixin):
        async def import_data(self, source_url: str, dest_path: Path) -> bool:
            success = await self._download_with_progress(
                source_url=source_url,
                dest_path=dest_path,
                verify_checksum=expected_checksum,
                progress_callback=self._on_progress,
            )
            if success:
                result = await self._validate_import(dest_path, "npz")
                return result.valid
            return False
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "ImportDaemonMixin",
    "ImportValidationResult",
    "DownloadProgress",
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ImportValidationResult:
    """Result of import validation.

    Attributes:
        valid: Whether the file passed validation
        file_type: Detected file type (db, npz, pth, unknown)
        size_bytes: File size in bytes
        checksum: SHA256 checksum (if computed)
        error: Error message if validation failed
        details: Additional validation details
    """

    valid: bool
    file_type: str
    size_bytes: int = 0
    checksum: str = ""
    error: str = ""
    details: dict[str, Any] | None = None


@dataclass
class DownloadProgress:
    """Progress information for downloads.

    Attributes:
        bytes_downloaded: Bytes downloaded so far
        total_bytes: Total bytes to download (0 if unknown)
        percent_complete: Percentage complete (0-100)
        elapsed_seconds: Time elapsed since start
        speed_bytes_per_sec: Current download speed
    """

    bytes_downloaded: int
    total_bytes: int
    percent_complete: float
    elapsed_seconds: float
    speed_bytes_per_sec: float


# =============================================================================
# ImportDaemonMixin
# =============================================================================


class ImportDaemonMixin:
    """Mixin providing common import/download functionality.

    Subclasses should have logging set up and may override:
    - IMPORT_LOG_PREFIX: Prefix for log messages (default: "[Import]")
    - IMPORT_CHUNK_SIZE: Chunk size for streaming downloads (default: 8192)
    - IMPORT_VERIFY_CHECKSUMS: Whether to verify checksums (default: True)
    """

    # Configuration (can be overridden in subclasses)
    IMPORT_LOG_PREFIX: str = "[Import]"
    IMPORT_CHUNK_SIZE: int = 8192
    IMPORT_VERIFY_CHECKSUMS: bool = True

    # Supported file types and their validation methods
    _VALIDATION_METHODS: dict[str, str] = {
        "db": "_validate_sqlite_db",
        "npz": "_validate_npz_file",
        "pth": "_validate_pytorch_model",
        "pt": "_validate_pytorch_model",
    }

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    async def _download_with_progress(
        self,
        source_url: str,
        dest_path: Path,
        verify_checksum: str | None = None,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
        timeout: int = 600,
    ) -> bool:
        """Download file with progress tracking and verification.

        Supports various source types:
        - s3://bucket/key - AWS S3 downloads
        - http(s)://... - HTTP downloads (via curl)
        - ssh://user@host:path - SSH/SCP downloads
        - file:///path - Local file copy

        Args:
            source_url: Source URL (s3://, http://, ssh://, file://)
            dest_path: Local destination path
            verify_checksum: Expected SHA256 checksum (optional)
            progress_callback: Callback for progress updates (optional)
            timeout: Download timeout in seconds

        Returns:
            True if download succeeded and checksum matches (if provided)
        """
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Download to temporary file first
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=dest_path.parent,
                suffix=dest_path.suffix,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

            start_time = time.time()
            success = False

            try:
                if source_url.startswith("s3://"):
                    success = await self._download_s3(source_url, tmp_path, timeout)
                elif source_url.startswith(("http://", "https://")):
                    success = await self._download_http(
                        source_url, tmp_path, timeout, progress_callback
                    )
                elif source_url.startswith("ssh://"):
                    success = await self._download_ssh(source_url, tmp_path, timeout)
                elif source_url.startswith("file://"):
                    local_path = Path(source_url[7:])  # Remove "file://"
                    success = await self._copy_local(local_path, tmp_path)
                else:
                    # Assume local path
                    success = await self._copy_local(Path(source_url), tmp_path)

            except Exception as e:
                logger.warning(f"{self.IMPORT_LOG_PREFIX} Download error: {e}")
                tmp_path.unlink(missing_ok=True)
                return False

            if not success:
                tmp_path.unlink(missing_ok=True)
                return False

            # Verify checksum if provided
            if verify_checksum and self.IMPORT_VERIFY_CHECKSUMS:
                actual_checksum = await self._compute_checksum(tmp_path)
                if actual_checksum != verify_checksum:
                    logger.warning(
                        f"{self.IMPORT_LOG_PREFIX} Checksum mismatch: "
                        f"expected {verify_checksum[:16]}..., got {actual_checksum[:16]}..."
                    )
                    tmp_path.unlink(missing_ok=True)
                    return False

            # Atomic move to final destination
            success = await self._atomic_replace(tmp_path, dest_path)

            elapsed = time.time() - start_time
            if success:
                size_mb = dest_path.stat().st_size / (1024 * 1024)
                speed_mb = size_mb / max(elapsed, 0.001)
                logger.info(
                    f"{self.IMPORT_LOG_PREFIX} Downloaded {dest_path.name} "
                    f"({size_mb:.1f} MB in {elapsed:.1f}s, {speed_mb:.1f} MB/s)"
                )

            return success

        except Exception as e:
            logger.warning(f"{self.IMPORT_LOG_PREFIX} Download failed: {e}")
            return False

    async def _validate_import(
        self,
        file_path: Path,
        expected_type: str | None = None,
    ) -> ImportValidationResult:
        """Validate imported file integrity.

        Runs type-specific validation:
        - db: SQLite PRAGMA integrity_check
        - npz: NumPy load test, array shape validation
        - pth/pt: PyTorch checkpoint structure validation

        Args:
            file_path: Path to the file to validate
            expected_type: Expected file type (db, npz, pth, pt)
                          If None, inferred from extension

        Returns:
            ImportValidationResult with validation details
        """
        if not file_path.exists():
            return ImportValidationResult(
                valid=False,
                file_type="unknown",
                error="File does not exist",
            )

        # Determine file type
        if expected_type:
            file_type = expected_type
        else:
            file_type = file_path.suffix.lstrip(".").lower()

        size_bytes = file_path.stat().st_size

        # Run type-specific validation
        validation_method = self._VALIDATION_METHODS.get(file_type)
        if validation_method and hasattr(self, validation_method):
            method = getattr(self, validation_method)
            try:
                result = await asyncio.to_thread(method, file_path)
                if isinstance(result, ImportValidationResult):
                    result.size_bytes = size_bytes
                    return result
            except Exception as e:
                return ImportValidationResult(
                    valid=False,
                    file_type=file_type,
                    size_bytes=size_bytes,
                    error=str(e),
                )

        # Basic existence check for unknown types
        return ImportValidationResult(
            valid=True,
            file_type=file_type,
            size_bytes=size_bytes,
        )

    async def _atomic_replace(
        self,
        temp_path: Path,
        final_path: Path,
    ) -> bool:
        """Atomically replace file after successful download.

        Uses os.replace() for atomic rename on POSIX systems.
        Creates backup of existing file if present.

        Args:
            temp_path: Temporary file path
            final_path: Final destination path

        Returns:
            True if replacement succeeded
        """
        try:
            # Ensure parent directory exists
            final_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if file exists
            backup_path = None
            if final_path.exists():
                backup_path = final_path.with_suffix(final_path.suffix + ".bak")
                try:
                    os.replace(final_path, backup_path)
                except (OSError, IOError) as e:
                    logger.debug(f"{self.IMPORT_LOG_PREFIX} Backup failed: {e}")
                    # Continue anyway - atomic replace will overwrite

            # Atomic replace
            os.replace(temp_path, final_path)

            # Remove backup on success
            if backup_path and backup_path.exists():
                try:
                    backup_path.unlink()
                except (OSError, IOError):
                    pass  # Non-critical

            return True

        except (OSError, IOError) as e:
            logger.warning(f"{self.IMPORT_LOG_PREFIX} Atomic replace failed: {e}")
            return False

    async def _compute_checksum(
        self,
        file_path: Path,
        algorithm: str = "sha256",
    ) -> str:
        """Compute file checksum.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, md5, etc.)

        Returns:
            Hex digest of the checksum
        """
        def _compute() -> str:
            hasher = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(self.IMPORT_CHUNK_SIZE), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        return await asyncio.to_thread(_compute)

    # -------------------------------------------------------------------------
    # Download Methods (Internal)
    # -------------------------------------------------------------------------

    async def _download_s3(
        self,
        s3_url: str,
        dest_path: Path,
        timeout: int,
    ) -> bool:
        """Download from S3 using AWS CLI.

        Args:
            s3_url: S3 URL (s3://bucket/key)
            dest_path: Local destination
            timeout: Timeout in seconds

        Returns:
            True if download succeeded
        """
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["aws", "s3", "cp", s3_url, str(dest_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning(f"{self.IMPORT_LOG_PREFIX} S3 download timed out")
            return False

    async def _download_http(
        self,
        url: str,
        dest_path: Path,
        timeout: int,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> bool:
        """Download from HTTP/HTTPS using curl.

        Args:
            url: HTTP(S) URL
            dest_path: Local destination
            timeout: Timeout in seconds
            progress_callback: Progress callback (optional)

        Returns:
            True if download succeeded
        """
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "curl", "-fsSL",
                    "--connect-timeout", "30",
                    "--max-time", str(timeout),
                    "-o", str(dest_path),
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 30,  # Extra buffer for curl
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning(f"{self.IMPORT_LOG_PREFIX} HTTP download timed out")
            return False

    async def _download_ssh(
        self,
        ssh_url: str,
        dest_path: Path,
        timeout: int,
    ) -> bool:
        """Download via SSH/SCP.

        Args:
            ssh_url: SSH URL (ssh://user@host:path)
            dest_path: Local destination
            timeout: Timeout in seconds

        Returns:
            True if download succeeded
        """
        try:
            # Parse ssh://user@host:path or ssh://user@host/path
            url_part = ssh_url[6:]  # Remove "ssh://"
            if ":" in url_part.split("@")[-1]:
                # ssh://user@host:path format
                host_part, remote_path = url_part.rsplit(":", 1)
            else:
                # ssh://user@host/path format
                host_part, remote_path = url_part.split("/", 1)
                remote_path = "/" + remote_path

            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "scp", "-o", "StrictHostKeyChecking=no",
                    "-o", f"ConnectTimeout={min(30, timeout)}",
                    f"{host_part}:{remote_path}",
                    str(dest_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning(f"{self.IMPORT_LOG_PREFIX} SSH download timed out")
            return False

    async def _copy_local(
        self,
        source_path: Path,
        dest_path: Path,
    ) -> bool:
        """Copy local file.

        Args:
            source_path: Source file path
            dest_path: Destination path

        Returns:
            True if copy succeeded
        """
        try:
            await asyncio.to_thread(shutil.copy2, source_path, dest_path)
            return True
        except (OSError, IOError) as e:
            logger.warning(f"{self.IMPORT_LOG_PREFIX} Local copy failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Validation Methods (Internal)
    # -------------------------------------------------------------------------

    def _validate_sqlite_db(self, file_path: Path) -> ImportValidationResult:
        """Validate SQLite database integrity."""
        import sqlite3

        try:
            conn = sqlite3.connect(str(file_path))
            cursor = conn.cursor()

            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()

            if result and result[0] == "ok":
                # Get table count
                cursor.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                )
                table_count = cursor.fetchone()[0]
                conn.close()

                return ImportValidationResult(
                    valid=True,
                    file_type="db",
                    details={"table_count": table_count},
                )
            else:
                conn.close()
                return ImportValidationResult(
                    valid=False,
                    file_type="db",
                    error=f"Integrity check failed: {result}",
                )

        except sqlite3.Error as e:
            return ImportValidationResult(
                valid=False,
                file_type="db",
                error=str(e),
            )

    def _validate_npz_file(self, file_path: Path) -> ImportValidationResult:
        """Validate NumPy NPZ file."""
        try:
            import numpy as np

            with np.load(str(file_path), allow_pickle=False) as data:
                arrays = list(data.keys())
                shapes = {k: data[k].shape for k in arrays}

                return ImportValidationResult(
                    valid=True,
                    file_type="npz",
                    details={
                        "arrays": arrays,
                        "shapes": shapes,
                    },
                )

        except Exception as e:
            return ImportValidationResult(
                valid=False,
                file_type="npz",
                error=str(e),
            )

    def _validate_pytorch_model(self, file_path: Path) -> ImportValidationResult:
        """Validate PyTorch checkpoint."""
        try:
            import torch

            # Use safe_load if available
            try:
                from app.utils.torch_utils import safe_load_checkpoint
                checkpoint = safe_load_checkpoint(str(file_path))
            except ImportError:
                checkpoint = torch.load(
                    str(file_path),
                    map_location="cpu",
                    weights_only=True,
                )

            # Check for common checkpoint keys
            keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
            has_state_dict = "state_dict" in keys or "model_state_dict" in keys

            return ImportValidationResult(
                valid=True,
                file_type="pth",
                details={
                    "keys": keys[:10],  # First 10 keys
                    "has_state_dict": has_state_dict,
                },
            )

        except Exception as e:
            return ImportValidationResult(
                valid=False,
                file_type="pth",
                error=str(e),
            )
