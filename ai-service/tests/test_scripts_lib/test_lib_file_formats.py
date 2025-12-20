"""Tests for scripts/lib/file_formats.py module.

Tests cover:
- Gzip detection
- JSONL file handling (read/write)
- JSON file handling (load/save)
- File information utilities
"""

import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.lib.file_formats import (
    count_jsonl_lines,
    get_file_size_mb,
    get_uncompressed_size_estimate,
    is_gzip_file,
    load_json,
    load_json_if_exists,
    load_json_strict,
    open_jsonl_file,
    read_jsonl_lines,
    save_json,
    save_json_compact,
    update_json,
    write_jsonl_lines,
)


class TestIsGzipFile:
    """Tests for is_gzip_file function."""

    def test_detects_gzip_file(self, tmp_path):
        """Test detection of gzip file."""
        gzip_file = tmp_path / "test.gz"
        with gzip.open(gzip_file, "wt") as f:
            f.write("test content")

        assert is_gzip_file(gzip_file) is True

    def test_detects_non_gzip_file(self, tmp_path):
        """Test detection of non-gzip file."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("test content")

        assert is_gzip_file(text_file) is False

    def test_handles_nonexistent_file(self, tmp_path):
        """Test handling of nonexistent file."""
        assert is_gzip_file(tmp_path / "nonexistent.gz") is False

    def test_handles_empty_file(self, tmp_path):
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()

        assert is_gzip_file(empty_file) is False


class TestOpenJsonlFile:
    """Tests for open_jsonl_file context manager."""

    def test_opens_plain_jsonl(self, tmp_path):
        """Test opening plain JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"key": "value1"}\n{"key": "value2"}\n')

        with open_jsonl_file(jsonl_file) as f:
            lines = list(f)

        assert len(lines) == 2
        assert '"value1"' in lines[0]

    def test_opens_gzipped_jsonl(self, tmp_path):
        """Test opening gzipped JSONL file."""
        gzip_file = tmp_path / "test.jsonl.gz"
        with gzip.open(gzip_file, "wt") as f:
            f.write('{"key": "value1"}\n{"key": "value2"}\n')

        with open_jsonl_file(gzip_file) as f:
            lines = list(f)

        assert len(lines) == 2

    def test_handles_encoding_errors(self, tmp_path):
        """Test handling of encoding errors."""
        # Write file with invalid UTF-8 bytes
        bin_file = tmp_path / "test.jsonl"
        with open(bin_file, "wb") as f:
            f.write(b'{"key": "\xff\xfe"}\n')

        # Should not raise due to errors="replace"
        with open_jsonl_file(bin_file) as f:
            lines = list(f)
        assert len(lines) == 1


class TestReadJsonlLines:
    """Tests for read_jsonl_lines function."""

    def test_reads_all_lines(self, tmp_path):
        """Test reading all lines."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": 1}\n{"id": 2}\n{"id": 3}\n')

        records = list(read_jsonl_lines(jsonl_file))

        assert len(records) == 3
        assert records[0] == {"id": 1}
        assert records[2] == {"id": 3}

    def test_respects_limit(self, tmp_path):
        """Test limit parameter."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": 1}\n{"id": 2}\n{"id": 3}\n{"id": 4}\n')

        records = list(read_jsonl_lines(jsonl_file, limit=2))

        assert len(records) == 2
        assert records[0] == {"id": 1}

    def test_skips_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": 1}\n\n{"id": 2}\n   \n{"id": 3}\n')

        records = list(read_jsonl_lines(jsonl_file))

        assert len(records) == 3

    def test_skips_invalid_json(self, tmp_path):
        """Test skipping invalid JSON lines."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": 1}\ninvalid json\n{"id": 2}\n')

        records = list(read_jsonl_lines(jsonl_file, skip_invalid=True))

        assert len(records) == 2

    def test_raises_on_invalid_json(self, tmp_path):
        """Test raising on invalid JSON when skip_invalid=False."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": 1}\ninvalid json\n')

        with pytest.raises(ValueError, match="Invalid JSON"):
            list(read_jsonl_lines(jsonl_file, skip_invalid=False))


class TestCountJsonlLines:
    """Tests for count_jsonl_lines function."""

    def test_counts_non_empty_lines(self, tmp_path):
        """Test counting non-empty lines."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": 1}\n{"id": 2}\n\n{"id": 3}\n')

        count = count_jsonl_lines(jsonl_file)

        assert count == 3

    def test_counts_gzipped_file(self, tmp_path):
        """Test counting lines in gzipped file."""
        gzip_file = tmp_path / "test.jsonl.gz"
        with gzip.open(gzip_file, "wt") as f:
            f.write('{"id": 1}\n{"id": 2}\n')

        count = count_jsonl_lines(gzip_file)

        assert count == 2


class TestWriteJsonlLines:
    """Tests for write_jsonl_lines function."""

    def test_writes_records(self, tmp_path):
        """Test writing records to file."""
        jsonl_file = tmp_path / "output.jsonl"
        records = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]

        count = write_jsonl_lines(jsonl_file, iter(records))

        assert count == 2
        assert jsonl_file.exists()

        # Verify content
        with open(jsonl_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1, "name": "a"}

    def test_writes_gzipped(self, tmp_path):
        """Test writing gzipped output."""
        gzip_file = tmp_path / "output.jsonl.gz"
        records = [{"id": 1}, {"id": 2}]

        count = write_jsonl_lines(gzip_file, iter(records), compress=True)

        assert count == 2
        assert is_gzip_file(gzip_file)

    def test_append_mode(self, tmp_path):
        """Test appending to existing file."""
        jsonl_file = tmp_path / "output.jsonl"
        jsonl_file.write_text('{"id": 1}\n')

        write_jsonl_lines(jsonl_file, iter([{"id": 2}]), append=True)

        count = count_jsonl_lines(jsonl_file)
        assert count == 2

    def test_creates_parent_dirs(self, tmp_path):
        """Test creating parent directories."""
        nested_file = tmp_path / "nested" / "dir" / "output.jsonl"

        write_jsonl_lines(nested_file, iter([{"id": 1}]))

        assert nested_file.exists()


class TestLoadJson:
    """Tests for load_json function."""

    def test_loads_valid_json(self, tmp_path):
        """Test loading valid JSON."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value", "num": 42}')

        data = load_json(json_file)

        assert data == {"key": "value", "num": 42}

    def test_returns_default_for_missing_file(self, tmp_path):
        """Test returning default for missing file."""
        data = load_json(tmp_path / "missing.json", default={"default": True})

        assert data == {"default": True}

    def test_returns_default_for_invalid_json(self, tmp_path):
        """Test returning default for invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json {")

        data = load_json(json_file, default=[])

        assert data == []

    def test_default_is_none(self, tmp_path):
        """Test default default is None."""
        data = load_json(tmp_path / "missing.json")

        assert data is None


class TestLoadJsonStrict:
    """Tests for load_json_strict function."""

    def test_loads_valid_json(self, tmp_path):
        """Test loading valid JSON."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')

        data = load_json_strict(json_file)

        assert data == {"key": "value"}

    def test_raises_for_missing_file(self, tmp_path):
        """Test raising for missing file."""
        with pytest.raises(FileNotFoundError):
            load_json_strict(tmp_path / "missing.json")

    def test_raises_for_invalid_json(self, tmp_path):
        """Test raising for invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid")

        with pytest.raises(json.JSONDecodeError):
            load_json_strict(json_file)


class TestLoadJsonIfExists:
    """Tests for load_json_if_exists function."""

    def test_loads_existing_file(self, tmp_path):
        """Test loading existing file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"exists": true}')

        data = load_json_if_exists(json_file, default={"exists": False})

        assert data == {"exists": True}

    def test_returns_default_for_missing(self, tmp_path):
        """Test returning default for missing file."""
        data = load_json_if_exists(tmp_path / "missing.json", default={"missing": True})

        assert data == {"missing": True}


class TestSaveJson:
    """Tests for save_json function."""

    def test_saves_json(self, tmp_path):
        """Test saving JSON."""
        json_file = tmp_path / "output.json"
        data = {"key": "value", "list": [1, 2, 3]}

        save_json(json_file, data)

        assert json_file.exists()
        loaded = json.loads(json_file.read_text())
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        """Test creating parent directories."""
        nested_file = tmp_path / "nested" / "dir" / "output.json"

        save_json(nested_file, {"nested": True})

        assert nested_file.exists()

    def test_atomic_write(self, tmp_path):
        """Test atomic write creates complete file."""
        json_file = tmp_path / "atomic.json"

        save_json(json_file, {"atomic": True}, atomic=True)

        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert data == {"atomic": True}

    def test_non_atomic_write(self, tmp_path):
        """Test non-atomic write."""
        json_file = tmp_path / "direct.json"

        save_json(json_file, {"direct": True}, atomic=False)

        assert json_file.exists()

    def test_indentation(self, tmp_path):
        """Test indentation parameter."""
        json_file = tmp_path / "indented.json"

        save_json(json_file, {"key": "value"}, indent=4)

        content = json_file.read_text()
        assert "    " in content  # 4-space indent

    def test_trailing_newline(self, tmp_path):
        """Test trailing newline is added."""
        json_file = tmp_path / "newline.json"

        save_json(json_file, {"key": "value"})

        content = json_file.read_text()
        assert content.endswith("\n")


class TestSaveJsonCompact:
    """Tests for save_json_compact function."""

    def test_saves_compact(self, tmp_path):
        """Test saving compact JSON."""
        json_file = tmp_path / "compact.json"

        save_json_compact(json_file, {"key": "value", "num": 42})

        content = json_file.read_text()
        # No indentation, single line
        assert '{"key": "value", "num": 42}' in content


class TestUpdateJson:
    """Tests for update_json function."""

    def test_updates_existing_file(self, tmp_path):
        """Test updating existing file."""
        json_file = tmp_path / "config.json"
        json_file.write_text('{"version": 1, "name": "test"}')

        result = update_json(json_file, {"version": 2})

        assert result["version"] == 2
        assert result["name"] == "test"

        # Verify persisted
        loaded = load_json(json_file)
        assert loaded["version"] == 2

    def test_creates_new_file(self, tmp_path):
        """Test creating new file with updates."""
        json_file = tmp_path / "new.json"

        result = update_json(json_file, {"new_key": "value"})

        assert result == {"new_key": "value"}
        assert json_file.exists()

    def test_raises_for_non_dict(self, tmp_path):
        """Test raising for non-dict JSON."""
        json_file = tmp_path / "list.json"
        json_file.write_text("[1, 2, 3]")

        with pytest.raises(TypeError, match="Cannot update non-dict"):
            update_json(json_file, {"key": "value"})


class TestGetFileSizeMb:
    """Tests for get_file_size_mb function."""

    def test_returns_size(self, tmp_path):
        """Test returning file size."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"x" * 1024 * 1024)  # 1 MB

        size = get_file_size_mb(test_file)

        assert 0.9 < size < 1.1  # ~1 MB

    def test_returns_zero_for_missing(self, tmp_path):
        """Test returning 0 for missing file."""
        size = get_file_size_mb(tmp_path / "missing.txt")

        assert size == 0.0


class TestGetUncompressedSizeEstimate:
    """Tests for get_uncompressed_size_estimate function."""

    def test_returns_file_size_for_non_gzip(self, tmp_path):
        """Test returning file size for non-gzip file."""
        text_file = tmp_path / "test.txt"
        content = "test content" * 100
        text_file.write_text(content)

        size = get_uncompressed_size_estimate(text_file)

        assert size == len(content)

    def test_estimates_uncompressed_size(self, tmp_path):
        """Test estimating uncompressed size of gzip file."""
        gzip_file = tmp_path / "test.gz"
        content = "test content " * 1000

        with gzip.open(gzip_file, "wt") as f:
            f.write(content)

        size = get_uncompressed_size_estimate(gzip_file)

        # Should be close to original size
        assert abs(size - len(content)) < 100
