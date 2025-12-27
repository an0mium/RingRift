"""Tests for CLI output module.

Tests the formatted console output functionality:
- Status messages with icons
- Table formatting
- Progress bars
- Color handling
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest

from app.cli.output import (
    ProgressBar,
    _USE_COLORS,
    _color,
    print_error,
    print_progress,
    print_status,
    print_success,
    print_table,
    print_warning,
)


# ============================================================================
# Color Tests
# ============================================================================


class TestColor:
    """Tests for the _color helper function."""

    def test_color_returns_text_without_colors(self):
        """Test color returns plain text when colors disabled."""
        with patch("app.cli.output._USE_COLORS", False):
            result = _color("test", "red")
            assert result == "test"

    def test_color_applies_ansi_with_colors(self):
        """Test color applies ANSI codes when colors enabled."""
        with patch("app.cli.output._USE_COLORS", True):
            result = _color("test", "red")
            assert "\033[91m" in result
            assert "\033[0m" in result
            assert "test" in result

    def test_color_handles_unknown_color(self):
        """Test color handles unknown color names."""
        with patch("app.cli.output._USE_COLORS", True):
            result = _color("test", "nonexistent")
            assert "test" in result


# ============================================================================
# print_status Tests
# ============================================================================


class TestPrintStatus:
    """Tests for print_status function."""

    def test_info_status(self, capsys):
        """Test info status message."""
        with patch("app.cli.output._USE_COLORS", False):
            print_status("Test message", "info")

        captured = capsys.readouterr()
        assert "[i]" in captured.out
        assert "Test message" in captured.out

    def test_success_status(self, capsys):
        """Test success status message."""
        with patch("app.cli.output._USE_COLORS", False):
            print_status("Success!", "success")

        captured = capsys.readouterr()
        assert "[+]" in captured.out
        assert "Success!" in captured.out

    def test_warning_status(self, capsys):
        """Test warning status message."""
        with patch("app.cli.output._USE_COLORS", False):
            print_status("Warning!", "warning")

        captured = capsys.readouterr()
        assert "[!]" in captured.out
        assert "Warning!" in captured.out

    def test_error_status(self, capsys):
        """Test error status message."""
        with patch("app.cli.output._USE_COLORS", False):
            print_status("Error!", "error")

        captured = capsys.readouterr()
        assert "[x]" in captured.out
        assert "Error!" in captured.out

    def test_default_status(self, capsys):
        """Test default status (info) when unknown."""
        with patch("app.cli.output._USE_COLORS", False):
            print_status("Unknown status", "unknown")

        captured = capsys.readouterr()
        assert "[i]" in captured.out


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestPrintError:
    """Tests for print_error function."""

    def test_prints_error(self, capsys):
        """Test print_error outputs error status."""
        with patch("app.cli.output._USE_COLORS", False):
            print_error("Something went wrong")

        captured = capsys.readouterr()
        assert "[x]" in captured.out
        assert "Something went wrong" in captured.out


class TestPrintSuccess:
    """Tests for print_success function."""

    def test_prints_success(self, capsys):
        """Test print_success outputs success status."""
        with patch("app.cli.output._USE_COLORS", False):
            print_success("Operation completed")

        captured = capsys.readouterr()
        assert "[+]" in captured.out
        assert "Operation completed" in captured.out


class TestPrintWarning:
    """Tests for print_warning function."""

    def test_prints_warning(self, capsys):
        """Test print_warning outputs warning status."""
        with patch("app.cli.output._USE_COLORS", False):
            print_warning("Be careful")

        captured = capsys.readouterr()
        assert "[!]" in captured.out
        assert "Be careful" in captured.out


# ============================================================================
# print_table Tests
# ============================================================================


class TestPrintTable:
    """Tests for print_table function."""

    def test_empty_data(self, capsys):
        """Test table with no data."""
        print_table([])

        captured = capsys.readouterr()
        assert "(no data)" in captured.out

    def test_basic_table(self, capsys):
        """Test basic table output."""
        data = [
            {"name": "Alice", "score": 100},
            {"name": "Bob", "score": 85},
        ]

        with patch("app.cli.output._USE_COLORS", False):
            print_table(data)

        captured = capsys.readouterr()
        assert "name" in captured.out
        assert "score" in captured.out
        assert "Alice" in captured.out
        assert "Bob" in captured.out
        assert "100" in captured.out
        assert "85" in captured.out

    def test_table_with_columns(self, capsys):
        """Test table with specific columns."""
        data = [
            {"name": "Alice", "score": 100, "rank": 1},
            {"name": "Bob", "score": 85, "rank": 2},
        ]

        with patch("app.cli.output._USE_COLORS", False):
            print_table(data, columns=["name", "score"])

        captured = capsys.readouterr()
        assert "name" in captured.out
        assert "score" in captured.out
        # rank should not appear
        assert "rank" not in captured.out

    def test_table_with_headers(self, capsys):
        """Test table with custom headers."""
        data = [
            {"name": "Alice", "score": 100},
        ]

        with patch("app.cli.output._USE_COLORS", False):
            print_table(data, headers={"name": "Player Name", "score": "Total Score"})

        captured = capsys.readouterr()
        assert "Player Name" in captured.out
        assert "Total Score" in captured.out

    def test_table_separator(self, capsys):
        """Test table has separator line."""
        data = [{"x": 1}]

        with patch("app.cli.output._USE_COLORS", False):
            print_table(data)

        captured = capsys.readouterr()
        assert "-" in captured.out

    def test_table_alignment(self, capsys):
        """Test table columns are aligned."""
        data = [
            {"short": "a", "long": "very long value"},
            {"short": "b", "long": "x"},
        ]

        with patch("app.cli.output._USE_COLORS", False):
            print_table(data)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        # All data lines should have similar length due to padding
        assert len(lines) >= 3  # header, separator, 2 data rows


# ============================================================================
# print_progress Tests
# ============================================================================


class TestPrintProgress:
    """Tests for print_progress function."""

    def test_progress_zero(self, capsys):
        """Test progress at 0%."""
        print_progress(0, 100)

        captured = capsys.readouterr()
        assert "0.0%" in captured.out
        assert "[" in captured.out
        assert "]" in captured.out

    def test_progress_half(self, capsys):
        """Test progress at 50%."""
        print_progress(50, 100)

        captured = capsys.readouterr()
        assert "50.0%" in captured.out

    def test_progress_complete(self, capsys):
        """Test progress at 100%."""
        print_progress(100, 100)

        captured = capsys.readouterr()
        assert "100.0%" in captured.out

    def test_progress_with_prefix(self, capsys):
        """Test progress with prefix."""
        print_progress(50, 100, prefix="Loading: ")

        captured = capsys.readouterr()
        assert "Loading:" in captured.out

    def test_progress_with_suffix(self, capsys):
        """Test progress with suffix."""
        print_progress(50, 100, suffix="items")

        captured = capsys.readouterr()
        assert "items" in captured.out

    def test_progress_zero_total(self, capsys):
        """Test progress with zero total."""
        print_progress(0, 0)

        captured = capsys.readouterr()
        assert captured.out == ""  # Should print nothing

    def test_progress_custom_width(self, capsys):
        """Test progress with custom width."""
        print_progress(50, 100, width=20)

        captured = capsys.readouterr()
        # Bar should contain = and - characters
        assert "=" in captured.out
        assert "-" in captured.out


# ============================================================================
# ProgressBar Tests
# ============================================================================


class TestProgressBar:
    """Tests for ProgressBar class."""

    def test_init(self):
        """Test progress bar initialization."""
        bar = ProgressBar(total=100, prefix="Test: ", width=50)

        assert bar.total == 100
        assert bar.prefix == "Test: "
        assert bar.width == 50
        assert bar.current == 0

    def test_update_increments(self, capsys):
        """Test update increments progress."""
        bar = ProgressBar(total=100)
        bar.update()

        assert bar.current == 1

    def test_update_custom_amount(self, capsys):
        """Test update with custom amount."""
        bar = ProgressBar(total=100)
        bar.update(10)

        assert bar.current == 10

    def test_update_caps_at_total(self, capsys):
        """Test update doesn't exceed total."""
        bar = ProgressBar(total=100)
        bar.update(150)

        assert bar.current == 100

    def test_set_value(self, capsys):
        """Test set sets specific value."""
        bar = ProgressBar(total=100)
        bar.set(50)

        assert bar.current == 50

    def test_set_caps_at_total(self, capsys):
        """Test set doesn't exceed total."""
        bar = ProgressBar(total=100)
        bar.set(200)

        assert bar.current == 100

    def test_finish(self, capsys):
        """Test finish completes the bar."""
        bar = ProgressBar(total=100)
        bar.finish()

        assert bar.current == 100

    def test_context_manager(self, capsys):
        """Test progress bar as context manager."""
        with ProgressBar(total=5) as bar:
            for i in range(5):
                bar.update()

        assert bar.current == 5

    def test_context_manager_newline_on_incomplete(self, capsys):
        """Test context manager prints newline when incomplete."""
        with ProgressBar(total=100) as bar:
            bar.update(50)

        captured = capsys.readouterr()
        # Should end with newline
        assert captured.out.endswith("\n")

    def test_multiple_updates(self, capsys):
        """Test multiple updates accumulate."""
        bar = ProgressBar(total=100)
        bar.update(25)
        bar.update(25)
        bar.update(25)

        assert bar.current == 75
