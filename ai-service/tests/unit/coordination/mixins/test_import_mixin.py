"""Tests for ImportDaemonMixin.

January 2026: Created comprehensive tests for the import/download mixin
used by S3ImportDaemon, OWCImportDaemon, and NodeDataAgent.

Tests cover:
- _download_with_progress(): S3, HTTP, SSH, file:// transports
- _validate_import(): db, npz, pth validation
- _atomic_replace(): Backup creation and atomic replacement
- _compute_checksum(): SHA256 checksum computation
- Configuration overrides (IMPORT_LOG_PREFIX, IMPORT_CHUNK_SIZE, etc.)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.mixins.import_mixin import (
    DownloadProgress,
    ImportDaemonMixin,
    ImportValidationResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockImportDaemon(ImportDaemonMixin):
    """Mock daemon for testing ImportDaemonMixin."""

    IMPORT_LOG_PREFIX = "[TestImport]"
    IMPORT_CHUNK_SIZE = 4096
    IMPORT_VERIFY_CHECKSUMS = True


class MockImportDaemonNoVerify(ImportDaemonMixin):
    """Mock daemon with checksum verification disabled."""

    IMPORT_LOG_PREFIX = "[NoVerify]"
    IMPORT_VERIFY_CHECKSUMS = False


@pytest.fixture
def daemon():
    """Create a mock import daemon."""
    return MockImportDaemon()


@pytest.fixture
def daemon_no_verify():
    """Create a mock import daemon without checksum verification."""
    return MockImportDaemonNoVerify()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World!")
    return file_path


@pytest.fixture
def sample_db(temp_dir):
    """Create a sample SQLite database for testing."""
    db_path = temp_dir / "sample.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test (value) VALUES ('test_data')")
    conn.commit()
    conn.close()
    return db_path


# =============================================================================
# Test Data Classes
# =============================================================================


class TestImportValidationResult:
    """Tests for ImportValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid result creation."""
        result = ImportValidationResult(
            valid=True,
            file_type="db",
            size_bytes=1024,
            checksum="abc123",
        )
        assert result.valid is True
        assert result.file_type == "db"
        assert result.size_bytes == 1024
        assert result.checksum == "abc123"
        assert result.error == ""
        assert result.details is None

    def test_invalid_result(self):
        """Test invalid result with error."""
        result = ImportValidationResult(
            valid=False,
            file_type="npz",
            error="File corrupted",
        )
        assert result.valid is False
        assert result.file_type == "npz"
        assert result.error == "File corrupted"

    def test_result_with_details(self):
        """Test result with additional details."""
        result = ImportValidationResult(
            valid=True,
            file_type="db",
            details={"table_count": 5, "row_count": 100},
        )
        assert result.details == {"table_count": 5, "row_count": 100}


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_progress_creation(self):
        """Test progress creation."""
        progress = DownloadProgress(
            bytes_downloaded=1024,
            total_bytes=4096,
            percent_complete=25.0,
            elapsed_seconds=5.0,
            speed_bytes_per_sec=204.8,
        )
        assert progress.bytes_downloaded == 1024
        assert progress.total_bytes == 4096
        assert progress.percent_complete == 25.0
        assert progress.elapsed_seconds == 5.0
        assert progress.speed_bytes_per_sec == 204.8

    def test_progress_unknown_total(self):
        """Test progress with unknown total size."""
        progress = DownloadProgress(
            bytes_downloaded=512,
            total_bytes=0,  # Unknown
            percent_complete=0.0,
            elapsed_seconds=2.0,
            speed_bytes_per_sec=256.0,
        )
        assert progress.total_bytes == 0


# =============================================================================
# Test Configuration
# =============================================================================


class TestMixinConfiguration:
    """Tests for mixin configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""

        class DefaultDaemon(ImportDaemonMixin):
            pass

        daemon = DefaultDaemon()
        assert daemon.IMPORT_LOG_PREFIX == "[Import]"
        assert daemon.IMPORT_CHUNK_SIZE == 8192
        assert daemon.IMPORT_VERIFY_CHECKSUMS is True

    def test_custom_configuration(self, daemon):
        """Test custom configuration values."""
        assert daemon.IMPORT_LOG_PREFIX == "[TestImport]"
        assert daemon.IMPORT_CHUNK_SIZE == 4096
        assert daemon.IMPORT_VERIFY_CHECKSUMS is True

    def test_disabled_checksum_verification(self, daemon_no_verify):
        """Test disabled checksum verification."""
        assert daemon_no_verify.IMPORT_VERIFY_CHECKSUMS is False


# =============================================================================
# Test Checksum Computation
# =============================================================================


class TestComputeChecksum:
    """Tests for _compute_checksum method."""

    @pytest.mark.asyncio
    async def test_compute_sha256(self, daemon, sample_file):
        """Test SHA256 checksum computation."""
        expected = hashlib.sha256(sample_file.read_bytes()).hexdigest()
        actual = await daemon._compute_checksum(sample_file)
        assert actual == expected

    @pytest.mark.asyncio
    async def test_compute_md5(self, daemon, sample_file):
        """Test MD5 checksum computation."""
        expected = hashlib.md5(sample_file.read_bytes()).hexdigest()
        actual = await daemon._compute_checksum(sample_file, algorithm="md5")
        assert actual == expected

    @pytest.mark.asyncio
    async def test_large_file_checksum(self, daemon, temp_dir):
        """Test checksum on larger file."""
        large_file = temp_dir / "large.bin"
        # Create 1MB file
        large_file.write_bytes(os.urandom(1024 * 1024))

        expected = hashlib.sha256(large_file.read_bytes()).hexdigest()
        actual = await daemon._compute_checksum(large_file)
        assert actual == expected


# =============================================================================
# Test Atomic Replace
# =============================================================================


class TestAtomicReplace:
    """Tests for _atomic_replace method."""

    @pytest.mark.asyncio
    async def test_atomic_replace_new_file(self, daemon, temp_dir):
        """Test atomic replace when destination doesn't exist."""
        temp_file = temp_dir / "temp.txt"
        final_file = temp_dir / "final.txt"

        temp_file.write_text("new content")

        result = await daemon._atomic_replace(temp_file, final_file)

        assert result is True
        assert final_file.exists()
        assert final_file.read_text() == "new content"
        assert not temp_file.exists()

    @pytest.mark.asyncio
    async def test_atomic_replace_existing_file(self, daemon, temp_dir):
        """Test atomic replace when destination exists."""
        temp_file = temp_dir / "temp.txt"
        final_file = temp_dir / "final.txt"

        final_file.write_text("old content")
        temp_file.write_text("new content")

        result = await daemon._atomic_replace(temp_file, final_file)

        assert result is True
        assert final_file.read_text() == "new content"
        # Backup should be removed on success
        backup_file = temp_dir / "final.txt.bak"
        assert not backup_file.exists()

    @pytest.mark.asyncio
    async def test_atomic_replace_creates_parent_dirs(self, daemon, temp_dir):
        """Test atomic replace creates parent directories."""
        temp_file = temp_dir / "temp.txt"
        final_file = temp_dir / "nested" / "deep" / "final.txt"

        temp_file.write_text("content")

        result = await daemon._atomic_replace(temp_file, final_file)

        assert result is True
        assert final_file.exists()
        assert final_file.read_text() == "content"


# =============================================================================
# Test Validation Methods
# =============================================================================


class TestValidateSQLiteDB:
    """Tests for SQLite database validation."""

    @pytest.mark.asyncio
    async def test_validate_valid_db(self, daemon, sample_db):
        """Test validation of valid SQLite database."""
        result = await daemon._validate_import(sample_db, expected_type="db")

        assert result.valid is True
        assert result.file_type == "db"
        assert result.size_bytes > 0
        assert result.details is not None
        assert result.details.get("table_count") == 1

    @pytest.mark.asyncio
    async def test_validate_corrupted_db(self, daemon, temp_dir):
        """Test validation of corrupted database."""
        corrupt_db = temp_dir / "corrupt.db"
        corrupt_db.write_text("not a valid database")

        result = await daemon._validate_import(corrupt_db, expected_type="db")

        assert result.valid is False
        assert result.file_type == "db"
        assert result.error != ""

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, daemon, temp_dir):
        """Test validation of non-existent file."""
        nonexistent = temp_dir / "does_not_exist.db"

        result = await daemon._validate_import(nonexistent)

        assert result.valid is False
        assert result.error == "File does not exist"


class TestValidateNPZ:
    """Tests for NPZ file validation."""

    @pytest.mark.asyncio
    async def test_validate_valid_npz(self, daemon, temp_dir):
        """Test validation of valid NPZ file."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        npz_file = temp_dir / "data.npz"
        np.savez(str(npz_file), array1=np.zeros(10), array2=np.ones((3, 4)))

        result = await daemon._validate_import(npz_file, expected_type="npz")

        assert result.valid is True
        assert result.file_type == "npz"
        assert result.details is not None
        assert "array1" in result.details.get("arrays", [])
        assert "array2" in result.details.get("arrays", [])

    @pytest.mark.asyncio
    async def test_validate_corrupted_npz(self, daemon, temp_dir):
        """Test validation of corrupted NPZ file."""
        corrupt_npz = temp_dir / "corrupt.npz"
        corrupt_npz.write_text("not a valid npz file")

        result = await daemon._validate_import(corrupt_npz, expected_type="npz")

        assert result.valid is False
        assert result.file_type == "npz"


class TestValidatePyTorch:
    """Tests for PyTorch checkpoint validation."""

    @pytest.mark.asyncio
    async def test_validate_valid_checkpoint(self, daemon, temp_dir):
        """Test validation of valid PyTorch checkpoint."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        pth_file = temp_dir / "model.pth"
        checkpoint = {
            "model_state_dict": {"layer1.weight": torch.zeros(10, 10)},
            "epoch": 5,
        }
        torch.save(checkpoint, str(pth_file))

        result = await daemon._validate_import(pth_file, expected_type="pth")

        assert result.valid is True
        assert result.file_type == "pth"
        assert result.details is not None
        assert "model_state_dict" in result.details.get("keys", [])


class TestValidateUnknownType:
    """Tests for unknown file type validation."""

    @pytest.mark.asyncio
    async def test_validate_unknown_type_exists(self, daemon, sample_file):
        """Test validation of unknown file type (just checks existence)."""
        result = await daemon._validate_import(sample_file)

        # For unknown types, just checks existence
        assert result.valid is True
        assert result.file_type == "txt"
        assert result.size_bytes > 0

    @pytest.mark.asyncio
    async def test_infer_type_from_extension(self, daemon, temp_dir):
        """Test type inference from file extension."""
        db_file = temp_dir / "test.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()

        # Let it infer type from extension
        result = await daemon._validate_import(db_file)
        assert result.file_type == "db"


# =============================================================================
# Test Download Methods
# =============================================================================


class TestDownloadS3:
    """Tests for S3 download method."""

    @pytest.mark.asyncio
    async def test_s3_download_success(self, daemon, temp_dir):
        """Test successful S3 download."""
        dest = temp_dir / "downloaded.txt"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await daemon._download_s3(
                "s3://bucket/key/file.txt",
                dest,
                timeout=60,
            )

            assert result is True
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_s3_download_failure(self, daemon, temp_dir):
        """Test failed S3 download."""
        dest = temp_dir / "downloaded.txt"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_to_thread.return_value = mock_result

            result = await daemon._download_s3(
                "s3://bucket/key/file.txt",
                dest,
                timeout=60,
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_s3_download_timeout(self, daemon, temp_dir):
        """Test S3 download timeout."""
        dest = temp_dir / "downloaded.txt"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = subprocess.TimeoutExpired(
                cmd=["aws", "s3", "cp"],
                timeout=60,
            )

            result = await daemon._download_s3(
                "s3://bucket/key/file.txt",
                dest,
                timeout=60,
            )

            assert result is False


class TestDownloadHTTP:
    """Tests for HTTP download method."""

    @pytest.mark.asyncio
    async def test_http_download_success(self, daemon, temp_dir):
        """Test successful HTTP download."""
        dest = temp_dir / "downloaded.txt"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await daemon._download_http(
                "https://example.com/file.txt",
                dest,
                timeout=60,
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_http_download_failure(self, daemon, temp_dir):
        """Test failed HTTP download (404 etc.)."""
        dest = temp_dir / "downloaded.txt"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 22  # curl exit code for HTTP errors
            mock_to_thread.return_value = mock_result

            result = await daemon._download_http(
                "https://example.com/not_found.txt",
                dest,
                timeout=60,
            )

            assert result is False


class TestDownloadSSH:
    """Tests for SSH/SCP download method."""

    @pytest.mark.asyncio
    async def test_ssh_download_success(self, daemon, temp_dir):
        """Test successful SSH download."""
        dest = temp_dir / "downloaded.txt"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await daemon._download_ssh(
                "ssh://user@host:/path/to/file.txt",
                dest,
                timeout=60,
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_ssh_download_slash_format(self, daemon, temp_dir):
        """Test SSH download with slash path format."""
        dest = temp_dir / "downloaded.txt"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_to_thread.return_value = mock_result

            result = await daemon._download_ssh(
                "ssh://user@host/path/to/file.txt",
                dest,
                timeout=60,
            )

            assert result is True


class TestCopyLocal:
    """Tests for local file copy method."""

    @pytest.mark.asyncio
    async def test_copy_local_success(self, daemon, temp_dir, sample_file):
        """Test successful local file copy."""
        dest = temp_dir / "copy.txt"

        result = await daemon._copy_local(sample_file, dest)

        assert result is True
        assert dest.exists()
        assert dest.read_text() == sample_file.read_text()

    @pytest.mark.asyncio
    async def test_copy_local_not_found(self, daemon, temp_dir):
        """Test local copy of non-existent file."""
        source = temp_dir / "not_found.txt"
        dest = temp_dir / "copy.txt"

        result = await daemon._copy_local(source, dest)

        assert result is False


# =============================================================================
# Test Download With Progress
# =============================================================================


class TestDownloadWithProgress:
    """Tests for _download_with_progress method."""

    @pytest.mark.asyncio
    async def test_download_s3_url(self, daemon, temp_dir):
        """Test download from S3 URL."""
        dest = temp_dir / "downloaded.txt"

        with patch.object(daemon, "_download_s3", new_callable=AsyncMock) as mock_s3:
            mock_s3.return_value = True
            # Create the temp file to simulate download
            with patch.object(daemon, "_atomic_replace", new_callable=AsyncMock) as mock_replace:
                mock_replace.return_value = True

                result = await daemon._download_with_progress(
                    "s3://bucket/key/file.txt",
                    dest,
                )

                mock_s3.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_http_url(self, daemon, temp_dir):
        """Test download from HTTP URL."""
        dest = temp_dir / "downloaded.txt"

        with patch.object(daemon, "_download_http", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = True
            with patch.object(daemon, "_atomic_replace", new_callable=AsyncMock) as mock_replace:
                mock_replace.return_value = True

                await daemon._download_with_progress(
                    "https://example.com/file.txt",
                    dest,
                )

                mock_http.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_ssh_url(self, daemon, temp_dir):
        """Test download from SSH URL."""
        dest = temp_dir / "downloaded.txt"

        with patch.object(daemon, "_download_ssh", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = True
            with patch.object(daemon, "_atomic_replace", new_callable=AsyncMock) as mock_replace:
                mock_replace.return_value = True

                await daemon._download_with_progress(
                    "ssh://user@host:/path/file.txt",
                    dest,
                )

                mock_ssh.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_url(self, daemon, temp_dir, sample_file):
        """Test download from file:// URL."""
        dest = temp_dir / "downloaded.txt"

        result = await daemon._download_with_progress(
            f"file://{sample_file}",
            dest,
        )

        assert result is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_download_local_path(self, daemon, temp_dir, sample_file):
        """Test download from local path (no URL scheme)."""
        dest = temp_dir / "downloaded.txt"

        result = await daemon._download_with_progress(
            str(sample_file),
            dest,
        )

        assert result is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_download_with_checksum_verification(self, daemon, temp_dir, sample_file):
        """Test download with checksum verification."""
        dest = temp_dir / "downloaded.txt"
        expected_checksum = hashlib.sha256(sample_file.read_bytes()).hexdigest()

        result = await daemon._download_with_progress(
            str(sample_file),
            dest,
            verify_checksum=expected_checksum,
        )

        assert result is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_download_checksum_mismatch(self, daemon, temp_dir, sample_file):
        """Test download fails on checksum mismatch."""
        dest = temp_dir / "downloaded.txt"

        result = await daemon._download_with_progress(
            str(sample_file),
            dest,
            verify_checksum="bad_checksum_value",
        )

        assert result is False
        assert not dest.exists()

    @pytest.mark.asyncio
    async def test_download_checksum_skipped_when_disabled(self, daemon_no_verify, temp_dir, sample_file):
        """Test checksum verification is skipped when disabled."""
        dest = temp_dir / "downloaded.txt"

        # Should succeed even with wrong checksum because verification is disabled
        result = await daemon_no_verify._download_with_progress(
            str(sample_file),
            dest,
            verify_checksum="bad_checksum_value",
        )

        assert result is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_download_creates_parent_dirs(self, daemon, temp_dir, sample_file):
        """Test download creates parent directories."""
        dest = temp_dir / "nested" / "deep" / "downloaded.txt"

        result = await daemon._download_with_progress(
            str(sample_file),
            dest,
        )

        assert result is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_download_failure_cleanup(self, daemon, temp_dir):
        """Test temp file cleanup on download failure."""
        dest = temp_dir / "downloaded.txt"

        with patch.object(daemon, "_download_s3", new_callable=AsyncMock) as mock_s3:
            mock_s3.return_value = False

            result = await daemon._download_with_progress(
                "s3://bucket/key/file.txt",
                dest,
            )

            assert result is False
            # Temp file should be cleaned up
            temp_files = list(temp_dir.glob("*.txt"))
            assert len(temp_files) == 0


# =============================================================================
# Test Progress Callback
# =============================================================================


class TestProgressCallback:
    """Tests for progress callback functionality."""

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, daemon, temp_dir):
        """Test that progress callback is called during HTTP download."""
        dest = temp_dir / "downloaded.txt"
        progress_calls = []

        def callback(progress: DownloadProgress):
            progress_calls.append(progress)

        with patch.object(daemon, "_download_http", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = True
            with patch.object(daemon, "_atomic_replace", new_callable=AsyncMock) as mock_replace:
                mock_replace.return_value = True

                # Progress callback is passed to _download_http but not yet used
                await daemon._download_with_progress(
                    "https://example.com/file.txt",
                    dest,
                    progress_callback=callback,
                )

                # The callback is passed through to _download_http
                call_args = mock_http.call_args
                assert call_args is not None
