"""Tests for checksum utilities."""

import hashlib
from pathlib import Path

import pytest

from app.utils.checksum_utils import (
    DEFAULT_CHUNK_SIZE,
    LARGE_CHUNK_SIZE,
    compute_bytes_checksum,
    compute_content_id,
    compute_file_checksum,
    compute_string_checksum,
    verify_file_checksum,
)


class TestComputeFileChecksum:
    """Tests for compute_file_checksum()."""

    def test_basic_file(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello world")

        result = compute_file_checksum(filepath)

        # Verify it's a valid SHA256 hex string
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_known_content(self, tmp_path):
        filepath = tmp_path / "test.txt"
        content = "test content"
        filepath.write_text(content)

        result = compute_file_checksum(filepath)
        expected = hashlib.sha256(content.encode()).hexdigest()

        assert result == expected

    def test_binary_file(self, tmp_path):
        filepath = tmp_path / "test.bin"
        content = b"\x00\x01\x02\xff\xfe"
        filepath.write_bytes(content)

        result = compute_file_checksum(filepath)
        expected = hashlib.sha256(content).hexdigest()

        assert result == expected

    def test_truncate(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")

        result = compute_file_checksum(filepath, truncate=16)

        assert len(result) == 16

    def test_different_algorithms(self, tmp_path):
        filepath = tmp_path / "test.txt"
        content = "test"
        filepath.write_text(content)

        sha256 = compute_file_checksum(filepath, algorithm="sha256")
        sha1 = compute_file_checksum(filepath, algorithm="sha1")
        md5 = compute_file_checksum(filepath, algorithm="md5")

        # All should be different lengths
        assert len(sha256) == 64
        assert len(sha1) == 40
        assert len(md5) == 32

    def test_invalid_algorithm_raises(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("test")

        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            compute_file_checksum(filepath, algorithm="invalid")

    def test_missing_file_returns_empty(self, tmp_path):
        filepath = tmp_path / "nonexistent.txt"

        result = compute_file_checksum(filepath)

        assert result == ""

    def test_missing_file_raises_when_configured(self, tmp_path):
        filepath = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            compute_file_checksum(filepath, return_empty_for_missing=False)

    def test_accepts_string_path(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")

        result = compute_file_checksum(str(filepath))

        assert len(result) == 64

    def test_large_file(self, tmp_path):
        filepath = tmp_path / "large.bin"
        # Create a file larger than chunk size
        content = b"x" * (DEFAULT_CHUNK_SIZE * 3 + 100)
        filepath.write_bytes(content)

        result = compute_file_checksum(filepath)
        expected = hashlib.sha256(content).hexdigest()

        assert result == expected

    def test_custom_chunk_size(self, tmp_path):
        filepath = tmp_path / "test.txt"
        content = b"hello world"
        filepath.write_bytes(content)

        # Should get same result with different chunk sizes
        result1 = compute_file_checksum(filepath, chunk_size=1)
        result2 = compute_file_checksum(filepath, chunk_size=LARGE_CHUNK_SIZE)

        assert result1 == result2


class TestComputeBytesChecksum:
    """Tests for compute_bytes_checksum()."""

    def test_basic_bytes(self):
        data = b"hello world"
        result = compute_bytes_checksum(data)

        expected = hashlib.sha256(data).hexdigest()
        assert result == expected

    def test_empty_bytes(self):
        data = b""
        result = compute_bytes_checksum(data)

        expected = hashlib.sha256(data).hexdigest()
        assert result == expected

    def test_truncate(self):
        data = b"test"
        result = compute_bytes_checksum(data, truncate=16)

        assert len(result) == 16

    def test_different_algorithms(self):
        data = b"test"

        sha256 = compute_bytes_checksum(data, algorithm="sha256")
        md5 = compute_bytes_checksum(data, algorithm="md5")

        assert len(sha256) == 64
        assert len(md5) == 32

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            compute_bytes_checksum(b"test", algorithm="invalid")


class TestComputeStringChecksum:
    """Tests for compute_string_checksum()."""

    def test_basic_string(self):
        content = "hello world"
        result = compute_string_checksum(content)

        expected = hashlib.sha256(content.encode()).hexdigest()
        assert result == expected

    def test_unicode_string(self):
        content = "h\u00e9llo w\u00f6rld \u65e5\u672c\u8a9e"
        result = compute_string_checksum(content)

        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert result == expected

    def test_truncate(self):
        content = "test"
        result = compute_string_checksum(content, truncate=32)

        assert len(result) == 32

    def test_custom_encoding(self):
        content = "test"
        result_utf8 = compute_string_checksum(content, encoding="utf-8")
        result_latin1 = compute_string_checksum(content, encoding="latin-1")

        # For ASCII content, results should be the same
        assert result_utf8 == result_latin1


class TestVerifyFileChecksum:
    """Tests for verify_file_checksum()."""

    def test_matching_checksum(self, tmp_path):
        filepath = tmp_path / "test.txt"
        content = "hello world"
        filepath.write_text(content)

        expected = hashlib.sha256(content.encode()).hexdigest()
        assert verify_file_checksum(filepath, expected) is True

    def test_non_matching_checksum(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello world")

        assert verify_file_checksum(filepath, "invalid_hash") is False

    def test_truncated_checksum(self, tmp_path):
        filepath = tmp_path / "test.txt"
        content = "hello world"
        filepath.write_text(content)

        full_hash = hashlib.sha256(content.encode()).hexdigest()
        truncated = full_hash[:16]

        assert verify_file_checksum(filepath, truncated) is True

    def test_empty_expected_returns_false(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")

        assert verify_file_checksum(filepath, "") is False

    def test_missing_file_returns_false(self, tmp_path):
        filepath = tmp_path / "nonexistent.txt"

        assert verify_file_checksum(filepath, "somehash") is False


class TestComputeContentId:
    """Tests for compute_content_id()."""

    def test_basic(self):
        result = compute_content_id("game123", '{"moves": []}')

        assert len(result) == 32  # default truncate

    def test_custom_truncate(self):
        result = compute_content_id("game123", '{"moves": []}', truncate=16)

        assert len(result) == 16

    def test_deterministic(self):
        result1 = compute_content_id("game123", '{"moves": []}')
        result2 = compute_content_id("game123", '{"moves": []}')

        assert result1 == result2

    def test_different_game_ids(self):
        result1 = compute_content_id("game1", '{"moves": []}')
        result2 = compute_content_id("game2", '{"moves": []}')

        assert result1 != result2

    def test_different_data(self):
        result1 = compute_content_id("game1", '{"a": 1}')
        result2 = compute_content_id("game1", '{"a": 2}')

        assert result1 != result2


class TestConstants:
    """Tests for module constants."""

    def test_default_chunk_size(self):
        assert DEFAULT_CHUNK_SIZE == 8192

    def test_large_chunk_size(self):
        assert LARGE_CHUNK_SIZE == 65536

    def test_large_is_bigger(self):
        assert LARGE_CHUNK_SIZE > DEFAULT_CHUNK_SIZE
