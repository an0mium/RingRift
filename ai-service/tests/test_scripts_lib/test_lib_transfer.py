"""Tests for scripts/lib/transfer.py module.

Tests cover:
- TransferConfig and TransferResult dataclasses
- Checksum computation
- Local file copy operations
- File compression/decompression
"""

import gzip
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.lib.transfer import (
    TransferConfig,
    TransferResult,
    compress_file,
    compute_checksum,
    copy_local,
    decompress_file,
    rsync_pull,
    rsync_push,
    scp_pull,
    scp_push,
    verify_transfer,
)


class TestTransferConfig:
    """Tests for TransferConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TransferConfig()
        assert config.ssh_user == "root"
        assert config.ssh_port == 22
        assert config.connect_timeout == 30
        assert config.max_retries == 3
        assert config.compress is True
        assert config.verify_checksum is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TransferConfig(
            ssh_key="/path/to/key",
            ssh_user="ubuntu",
            ssh_port=2222,
            max_retries=5,
        )
        assert config.ssh_key == "/path/to/key"
        assert config.ssh_user == "ubuntu"
        assert config.ssh_port == 2222
        assert config.max_retries == 5

    def test_get_ssh_options(self):
        """Test SSH options generation."""
        config = TransferConfig(connect_timeout=15)
        opts = config.get_ssh_options()

        assert "-o" in opts
        assert "StrictHostKeyChecking=no" in opts
        assert "ConnectTimeout=15" in opts

    def test_get_ssh_options_with_key(self, tmp_path):
        """Test SSH options include key when specified."""
        key_file = tmp_path / "id_rsa"
        key_file.write_text("fake key")

        config = TransferConfig(ssh_key=str(key_file))
        opts = config.get_ssh_options()

        assert "-i" in opts
        assert str(key_file) in opts


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_bool_success(self):
        """Test bool conversion for successful transfer."""
        result = TransferResult(success=True, bytes_transferred=1000)
        assert bool(result) is True

    def test_bool_failure(self):
        """Test bool conversion for failed transfer."""
        result = TransferResult(success=False, error="Connection refused")
        assert bool(result) is False

    def test_speed_mbps(self):
        """Test speed calculation."""
        result = TransferResult(
            success=True,
            bytes_transferred=10 * 1024 * 1024,  # 10 MB
            duration_seconds=2.0,  # 2 seconds
        )
        # 10 MB / 2 sec = 5 MB/s
        assert abs(result.speed_mbps - 5.0) < 0.1

    def test_speed_zero_duration(self):
        """Test speed calculation with zero duration."""
        result = TransferResult(
            success=True,
            bytes_transferred=1000,
            duration_seconds=0.0,
        )
        assert result.speed_mbps == 0.0


class TestComputeChecksum:
    """Tests for compute_checksum function."""

    def test_md5_checksum(self, tmp_path):
        """Test MD5 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\n")

        checksum = compute_checksum(test_file, algorithm="md5")
        assert len(checksum) == 32  # MD5 produces 32 hex chars
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_sha256_checksum(self, tmp_path):
        """Test SHA256 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\n")

        checksum = compute_checksum(test_file, algorithm="sha256")
        assert len(checksum) == 64  # SHA256 produces 64 hex chars

    def test_consistent_checksum(self, tmp_path):
        """Test that same content produces same checksum."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        content = "same content here"
        file1.write_text(content)
        file2.write_text(content)

        assert compute_checksum(file1) == compute_checksum(file2)

    def test_different_content_different_checksum(self, tmp_path):
        """Test that different content produces different checksum."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        assert compute_checksum(file1) != compute_checksum(file2)

    def test_invalid_algorithm(self, tmp_path):
        """Test that invalid algorithm raises error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            compute_checksum(test_file, algorithm="invalid")


class TestCopyLocal:
    """Tests for copy_local function."""

    def test_copy_file(self, tmp_path):
        """Test copying a file."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("test content")

        result = copy_local(source, dest)

        assert result.success is True
        assert dest.exists()
        assert dest.read_text() == "test content"
        assert result.bytes_transferred > 0

    def test_copy_with_checksum(self, tmp_path):
        """Test copying with checksum verification."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        source.write_text("test content")

        result = copy_local(source, dest, verify_checksum=True)

        assert result.success is True
        assert result.checksum_verified is True

    def test_copy_creates_parent_dirs(self, tmp_path):
        """Test that copy creates parent directories."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "subdir" / "nested" / "dest.txt"
        source.write_text("test")

        result = copy_local(source, dest)

        assert result.success is True
        assert dest.exists()

    def test_copy_nonexistent_source(self, tmp_path):
        """Test copying nonexistent source fails."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        result = copy_local(source, dest)

        assert result.success is False
        assert "not found" in result.error.lower()


class TestCompressFile:
    """Tests for compress_file function."""

    def test_compress_file(self, tmp_path):
        """Test file compression."""
        source = tmp_path / "source.txt"
        source.write_text("a" * 10000)  # Compressible content

        compressed, size = compress_file(source)

        assert compressed.exists()
        assert compressed.suffix == ".gz"
        assert size < source.stat().st_size  # Should be smaller

    def test_compress_to_custom_dest(self, tmp_path):
        """Test compression to custom destination."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "custom.gz"
        source.write_text("test content")

        compressed, _ = compress_file(source, dest)

        assert compressed == dest
        assert dest.exists()

    def test_compressed_can_be_decompressed(self, tmp_path):
        """Test that compressed file can be read with gzip."""
        source = tmp_path / "source.txt"
        original_content = "test content for compression"
        source.write_text(original_content)

        compressed, _ = compress_file(source)

        with gzip.open(compressed, "rt") as f:
            assert f.read() == original_content


class TestDecompressFile:
    """Tests for decompress_file function."""

    def test_decompress_file(self, tmp_path):
        """Test file decompression."""
        source = tmp_path / "source.txt"
        original_content = "decompression test content"
        source.write_text(original_content)

        compressed, _ = compress_file(source)
        decompressed, _ = decompress_file(compressed)

        assert decompressed.exists()
        assert decompressed.read_text() == original_content

    def test_decompress_to_custom_dest(self, tmp_path):
        """Test decompression to custom destination."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "custom_output.txt"
        source.write_text("test content")

        compressed, _ = compress_file(source)
        decompressed, _ = decompress_file(compressed, dest)

        assert decompressed == dest
        assert dest.exists()


class TestScpPush:
    """Tests for scp_push function."""

    def test_source_not_found(self, tmp_path):
        """Test pushing nonexistent file fails."""
        config = TransferConfig()
        result = scp_push(
            tmp_path / "nonexistent.txt",
            "host.example.com",
            22,
            "/remote/path/",
            config,
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @patch("scripts.lib.transfer.subprocess.run")
    def test_successful_push(self, mock_run, tmp_path):
        """Test successful SCP push."""
        source = tmp_path / "source.txt"
        source.write_text("test content")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = TransferConfig(verify_checksum=False)
        result = scp_push(source, "host", 22, "/remote/", config)

        assert result.success is True
        assert result.method == "scp"
        assert mock_run.called

    @patch("scripts.lib.transfer.subprocess.run")
    def test_retry_on_failure(self, mock_run, tmp_path):
        """Test that SCP retries on failure."""
        source = tmp_path / "source.txt"
        source.write_text("test")

        # Fail twice, succeed on third
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr="Connection refused"),
            MagicMock(returncode=1, stderr="Timeout"),
            MagicMock(returncode=0),
        ]

        config = TransferConfig(max_retries=3, retry_delay=0.01, verify_checksum=False)
        result = scp_push(source, "host", 22, "/remote/", config)

        assert result.success is True
        assert result.attempts == 3


class TestScpPull:
    """Tests for scp_pull function."""

    @patch("scripts.lib.transfer.subprocess.run")
    def test_successful_pull(self, mock_run, tmp_path):
        """Test successful SCP pull."""
        dest = tmp_path / "dest.txt"

        # Create file after "scp" runs
        def create_file(*args, **kwargs):
            dest.write_text("pulled content")
            return MagicMock(returncode=0)

        mock_run.side_effect = create_file

        config = TransferConfig(verify_checksum=False)
        result = scp_pull("host", 22, "/remote/file.txt", dest, config)

        assert result.success is True
        assert result.method == "scp"


class TestRsyncPush:
    """Tests for rsync_push function."""

    def test_source_not_found(self, tmp_path):
        """Test pushing nonexistent path fails."""
        config = TransferConfig()
        result = rsync_push(
            tmp_path / "nonexistent",
            "host",
            22,
            "/remote/",
            config,
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @patch("scripts.lib.transfer.subprocess.run")
    def test_successful_rsync(self, mock_run, tmp_path):
        """Test successful rsync push."""
        source = tmp_path / "source_dir"
        source.mkdir()
        (source / "file.txt").write_text("content")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="sent 1,000 bytes",
            stderr="",
        )

        config = TransferConfig()
        result = rsync_push(source, "host", 22, "/remote/", config)

        assert result.success is True
        assert result.method == "rsync"


class TestRsyncPull:
    """Tests for rsync_pull function."""

    @patch("scripts.lib.transfer.subprocess.run")
    def test_successful_rsync_pull(self, mock_run, tmp_path):
        """Test successful rsync pull."""
        dest = tmp_path / "dest.txt"

        def create_file(*args, **kwargs):
            dest.write_text("pulled content")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = create_file

        config = TransferConfig()
        result = rsync_pull("host", 22, "/remote/file.txt", dest, config)

        assert result.success is True
        assert result.method == "rsync"


class TestVerifyTransfer:
    """Tests for verify_transfer function."""

    def test_local_file_not_found(self, tmp_path):
        """Test verification fails when local file doesn't exist."""
        config = TransferConfig()
        result = verify_transfer(
            tmp_path / "nonexistent.txt",
            "host",
            "/remote/file.txt",
            config,
        )

        assert result is False

    @patch("scripts.lib.transfer.get_remote_checksum")
    def test_matching_checksums(self, mock_remote, tmp_path):
        """Test verification succeeds when checksums match."""
        local_file = tmp_path / "file.txt"
        local_file.write_text("test content")

        local_checksum = compute_checksum(local_file)
        mock_remote.return_value = local_checksum

        config = TransferConfig()
        result = verify_transfer(local_file, "host", "/remote/file.txt", config)

        assert result is True

    @patch("scripts.lib.transfer.get_remote_checksum")
    def test_mismatched_checksums(self, mock_remote, tmp_path):
        """Test verification fails when checksums don't match."""
        local_file = tmp_path / "file.txt"
        local_file.write_text("test content")

        mock_remote.return_value = "different_checksum"

        config = TransferConfig()
        result = verify_transfer(local_file, "host", "/remote/file.txt", config)

        assert result is False

    @patch("scripts.lib.transfer.get_remote_checksum")
    def test_remote_checksum_fails(self, mock_remote, tmp_path):
        """Test verification fails when remote checksum unavailable."""
        local_file = tmp_path / "file.txt"
        local_file.write_text("test content")

        mock_remote.return_value = None

        config = TransferConfig()
        result = verify_transfer(local_file, "host", "/remote/file.txt", config)

        assert result is False
