"""Tests for scripts/lib/datetime_utils.py module.

Tests cover:
- File age operations
- Elapsed time formatting
- Timestamp generation
- Timestamp parsing
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.lib.datetime_utils import (
    ElapsedTimer,
    find_files_older_than,
    format_elapsed_time,
    format_elapsed_time_short,
    get_file_age,
    get_file_age_days,
    get_file_age_hours,
    is_file_older_than,
    iter_files_by_age,
    parse_timestamp,
    parse_timestamp_safe,
    timestamp_age,
    timestamp_for_log,
    timestamp_id,
    timestamp_id_ms,
    timestamp_iso,
    timestamp_iso_utc,
)


class TestGetFileAge:
    """Tests for file age functions."""

    def test_get_file_age(self, tmp_path):
        """Test getting file age."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # File was just created, age should be very small
        age = get_file_age(test_file)
        assert isinstance(age, timedelta)
        assert age.total_seconds() < 5  # Less than 5 seconds

    def test_get_file_age_hours(self, tmp_path):
        """Test getting file age in hours."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        age_hours = get_file_age_hours(test_file)
        assert isinstance(age_hours, float)
        assert age_hours < 1  # Less than 1 hour

    def test_get_file_age_days(self, tmp_path):
        """Test getting file age in days."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        age_days = get_file_age_days(test_file)
        assert isinstance(age_days, float)
        assert age_days < 1  # Less than 1 day

    def test_get_file_age_nonexistent(self, tmp_path):
        """Test file age for nonexistent file raises."""
        with pytest.raises(FileNotFoundError):
            get_file_age(tmp_path / "nonexistent.txt")


class TestIsFileOlderThan:
    """Tests for is_file_older_than function."""

    def test_new_file_not_older(self, tmp_path):
        """Test that new file is not older than threshold."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        assert is_file_older_than(test_file, hours=1) is False
        assert is_file_older_than(test_file, days=1) is False
        assert is_file_older_than(test_file, minutes=1) is False

    def test_nonexistent_file(self, tmp_path):
        """Test nonexistent file returns False."""
        assert is_file_older_than(tmp_path / "missing.txt", hours=1) is False

    def test_no_threshold_raises(self, tmp_path):
        """Test that missing threshold raises."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Must specify"):
            is_file_older_than(test_file)


class TestFindFilesOlderThan:
    """Tests for find_files_older_than function."""

    def test_finds_old_files(self, tmp_path):
        """Test finding old files."""
        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Set mtime to 2 days ago
        old_time = time.time() - (2 * 86400)
        import os
        os.utime(test_file, (old_time, old_time))

        old_files = find_files_older_than(tmp_path, days=1)
        assert len(old_files) == 1
        assert old_files[0] == test_file

    def test_excludes_new_files(self, tmp_path):
        """Test that new files are excluded."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        old_files = find_files_older_than(tmp_path, days=1)
        assert len(old_files) == 0

    def test_with_pattern(self, tmp_path):
        """Test finding with glob pattern."""
        (tmp_path / "test.log").write_text("log")
        (tmp_path / "test.txt").write_text("txt")

        # Set both to old
        old_time = time.time() - (2 * 86400)
        import os
        os.utime(tmp_path / "test.log", (old_time, old_time))
        os.utime(tmp_path / "test.txt", (old_time, old_time))

        old_logs = find_files_older_than(tmp_path, days=1, pattern="*.log")
        assert len(old_logs) == 1
        assert old_logs[0].suffix == ".log"

    def test_recursive(self, tmp_path):
        """Test recursive search."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        nested_file = subdir / "nested.txt"
        nested_file.write_text("content")

        # Set to old
        old_time = time.time() - (2 * 86400)
        import os
        os.utime(nested_file, (old_time, old_time))

        # Non-recursive should not find it
        old_files = find_files_older_than(tmp_path, days=1, recursive=False)
        assert len(old_files) == 0

        # Recursive should find it
        old_files = find_files_older_than(tmp_path, days=1, recursive=True)
        assert len(old_files) == 1

    def test_no_threshold_raises(self, tmp_path):
        """Test that missing threshold raises."""
        with pytest.raises(ValueError, match="Must specify"):
            find_files_older_than(tmp_path)


class TestIterFilesByAge:
    """Tests for iter_files_by_age function."""

    def test_iterates_files(self, tmp_path):
        """Test iterating files by age."""
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        files = list(iter_files_by_age(tmp_path))
        assert len(files) == 2

    def test_sorts_by_age(self, tmp_path):
        """Test files are sorted by age."""
        import os

        old_file = tmp_path / "old.txt"
        new_file = tmp_path / "new.txt"

        old_file.write_text("old")
        old_time = time.time() - 3600
        os.utime(old_file, (old_time, old_time))

        new_file.write_text("new")

        # Default: oldest first
        files = list(iter_files_by_age(tmp_path))
        assert files[0].name == "old.txt"
        assert files[1].name == "new.txt"

        # newest_first=True
        files = list(iter_files_by_age(tmp_path, newest_first=True))
        assert files[0].name == "new.txt"
        assert files[1].name == "old.txt"


class TestFormatElapsedTime:
    """Tests for elapsed time formatting."""

    def test_seconds_only(self):
        """Test formatting seconds."""
        assert format_elapsed_time(5) == "5s"
        assert format_elapsed_time(45) == "45s"

    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        assert format_elapsed_time(65) == "1m 5s"
        assert format_elapsed_time(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        assert format_elapsed_time(3665) == "1h 1m 5s"

    def test_with_precision(self):
        """Test formatting with decimal precision."""
        result = format_elapsed_time(5.5, precision=1)
        assert "5.5s" in result

    def test_negative_returns_zero(self):
        """Test negative values return 0s."""
        assert format_elapsed_time(-5) == "0s"


class TestFormatElapsedTimeShort:
    """Tests for short elapsed time formatting."""

    def test_minutes_seconds(self):
        """Test MM:SS format."""
        assert format_elapsed_time_short(65) == "1:05"
        assert format_elapsed_time_short(125) == "2:05"

    def test_hours_minutes_seconds(self):
        """Test H:MM:SS format."""
        assert format_elapsed_time_short(3665) == "1:01:05"


class TestElapsedTimer:
    """Tests for ElapsedTimer context manager."""

    def test_basic_timing(self):
        """Test basic timing."""
        with ElapsedTimer() as timer:
            time.sleep(0.1)

        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.5

    def test_elapsed_str(self):
        """Test elapsed string property."""
        with ElapsedTimer() as timer:
            pass

        assert isinstance(timer.elapsed_str, str)
        assert "s" in timer.elapsed_str

    def test_with_description(self, caplog):
        """Test logging with description."""
        import logging
        caplog.set_level(logging.INFO)

        with ElapsedTimer("Test operation"):
            pass

        assert "Test operation completed" in caplog.text

    def test_without_logging(self, caplog):
        """Test without logging."""
        import logging
        caplog.set_level(logging.INFO)

        with ElapsedTimer("Test", log_on_exit=False):
            pass

        assert "Test" not in caplog.text


class TestTimestampGeneration:
    """Tests for timestamp generation functions."""

    def test_timestamp_id_format(self):
        """Test timestamp_id format."""
        ts = timestamp_id()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert ts[8] == "_"
        assert ts[:8].isdigit()
        assert ts[9:].isdigit()

    def test_timestamp_id_ms_format(self):
        """Test timestamp_id_ms format."""
        ts = timestamp_id_ms()
        parts = ts.split("_")
        assert len(parts) == 3
        assert len(parts[2]) == 3  # milliseconds

    def test_timestamp_for_log_format(self):
        """Test timestamp_for_log format."""
        ts = timestamp_for_log()
        assert len(ts) == 8  # HH:MM:SS
        assert ts.count(":") == 2

    def test_timestamp_iso_format(self):
        """Test timestamp_iso format."""
        ts = timestamp_iso()
        # Should be parseable
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)

    def test_timestamp_iso_utc_format(self):
        """Test timestamp_iso_utc format."""
        ts = timestamp_iso_utc()
        assert ts.endswith("Z")


class TestParseTimestamp:
    """Tests for timestamp parsing."""

    def test_parse_iso_string(self):
        """Test parsing ISO format string."""
        result = parse_timestamp("2024-12-19T14:30:00")
        assert result.year == 2024
        assert result.month == 12
        assert result.day == 19

    def test_parse_iso_with_z(self):
        """Test parsing ISO format with Z suffix."""
        result = parse_timestamp("2024-12-19T14:30:00Z")
        assert result.year == 2024

    def test_parse_unix_timestamp(self):
        """Test parsing Unix timestamp."""
        ts = time.time()
        result = parse_timestamp(ts)
        assert isinstance(result, datetime)

    def test_parse_datetime_passthrough(self):
        """Test datetime objects pass through."""
        dt = datetime.now()
        result = parse_timestamp(dt)
        assert result is dt

    def test_parse_common_formats(self):
        """Test parsing common date formats."""
        # YYYY-MM-DD HH:MM:SS
        result = parse_timestamp("2024-12-19 14:30:00")
        assert result.year == 2024

        # YYYYMMDD_HHMMSS
        result = parse_timestamp("20241219_143000")
        assert result.year == 2024

    def test_parse_invalid_raises(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_timestamp("not a timestamp")

    def test_parse_invalid_type_raises(self):
        """Test invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            parse_timestamp([])


class TestParseTimestampSafe:
    """Tests for parse_timestamp_safe function."""

    def test_returns_parsed(self):
        """Test successful parsing."""
        result = parse_timestamp_safe("2024-12-19T14:30:00")
        assert result.year == 2024

    def test_returns_default_on_failure(self):
        """Test returning default on failure."""
        default = datetime(2000, 1, 1)
        result = parse_timestamp_safe("invalid", default=default)
        assert result == default

    def test_returns_none_on_none(self):
        """Test returning None for None input."""
        result = parse_timestamp_safe(None)
        assert result is None

    def test_returns_default_for_none(self):
        """Test returning default for None input."""
        default = datetime(2000, 1, 1)
        result = parse_timestamp_safe(None, default=default)
        assert result == default


class TestTimestampAge:
    """Tests for timestamp_age function."""

    def test_recent_timestamp(self):
        """Test age of recent timestamp."""
        recent = datetime.now() - timedelta(hours=1)
        age = timestamp_age(recent)

        # Should be about 1 hour
        assert 0.9 < age.total_seconds() / 3600 < 1.1

    def test_old_timestamp(self):
        """Test age of old timestamp."""
        old = datetime.now() - timedelta(days=7)
        age = timestamp_age(old)

        # Should be about 7 days
        assert 6.9 < age.total_seconds() / 86400 < 7.1

    def test_unix_timestamp(self):
        """Test age of Unix timestamp."""
        # 1 hour ago
        ts = time.time() - 3600
        age = timestamp_age(ts)

        assert 0.9 < age.total_seconds() / 3600 < 1.1
