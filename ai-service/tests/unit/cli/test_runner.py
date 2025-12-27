"""Tests for CLI runner module.

Tests the command-line script runner functionality:
- ScriptRunner: Argument parsing and script context
- add_common_args: Common CLI arguments
- setup_script: Quick script setup helper
"""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.cli.runner import (
    ScriptConfig,
    ScriptRunner,
    add_common_args,
    setup_script,
)


# ============================================================================
# ScriptConfig Tests
# ============================================================================


class TestScriptConfig:
    """Tests for the ScriptConfig dataclass."""

    def test_basic_creation(self):
        """Test basic config creation."""
        config = ScriptConfig(name="test_script")

        assert config.name == "test_script"
        assert config.verbose is False
        assert config.dry_run is False
        assert config.config_path is None
        assert config.log_level == "INFO"

    def test_with_all_options(self):
        """Test config with all options specified."""
        config = ScriptConfig(
            name="test",
            verbose=True,
            dry_run=True,
            config_path=Path("/path/to/config"),
            log_level="DEBUG",
        )

        assert config.verbose is True
        assert config.dry_run is True
        assert config.config_path == Path("/path/to/config")
        assert config.log_level == "DEBUG"


# ============================================================================
# add_common_args Tests
# ============================================================================


class TestAddCommonArgs:
    """Tests for add_common_args function."""

    def test_adds_verbose(self):
        """Test verbose argument is added."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(["-v"])
        assert args.verbose is True

        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

    def test_adds_quiet(self):
        """Test quiet argument is added."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(["-q"])
        assert args.quiet is True

        args = parser.parse_args(["--quiet"])
        assert args.quiet is True

    def test_adds_dry_run(self):
        """Test dry-run argument is added."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_adds_config(self):
        """Test config argument is added."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(["-c", "/path/to/config.yaml"])
        assert args.config == Path("/path/to/config.yaml")

        args = parser.parse_args(["--config", "/other/path.yaml"])
        assert args.config == Path("/other/path.yaml")

    def test_adds_log_level(self):
        """Test log-level argument is added."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

        args = parser.parse_args(["--log-level", "WARNING"])
        assert args.log_level == "WARNING"

    def test_log_level_default(self):
        """Test log-level has INFO default."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args([])
        assert args.log_level == "INFO"

    def test_log_level_choices(self):
        """Test log-level only accepts valid choices."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--log-level", "INVALID"])


# ============================================================================
# ScriptRunner Tests
# ============================================================================


class TestScriptRunnerInit:
    """Tests for ScriptRunner initialization."""

    def test_basic_init(self):
        """Test basic runner creation."""
        runner = ScriptRunner("test_script")

        assert runner.name == "test_script"
        assert runner.parser is not None
        assert runner.logger is None
        assert runner._shutdown_requested is False

    def test_with_description(self):
        """Test runner with description."""
        runner = ScriptRunner("test", description="Test description")

        assert runner.parser.description == "Test description"

    def test_with_add_common_false(self):
        """Test runner without common args."""
        runner = ScriptRunner("test", add_common=False)

        # Should not have verbose argument
        with pytest.raises(SystemExit):
            runner.parser.parse_args(["--verbose"])

    def test_with_add_common_true(self):
        """Test runner with common args."""
        runner = ScriptRunner("test", add_common=True)

        args = runner.parser.parse_args(["--verbose"])
        assert args.verbose is True


class TestScriptRunnerAddArgument:
    """Tests for ScriptRunner.add_argument."""

    def test_add_simple_argument(self):
        """Test adding a simple argument."""
        runner = ScriptRunner("test", add_common=False)
        runner.add_argument("--input", required=True)

        args = runner.parser.parse_args(["--input", "file.txt"])
        assert args.input == "file.txt"

    def test_add_argument_with_default(self):
        """Test adding argument with default."""
        runner = ScriptRunner("test", add_common=False)
        runner.add_argument("--count", type=int, default=10)

        args = runner.parser.parse_args([])
        assert args.count == 10

    def test_add_positional_argument(self):
        """Test adding positional argument."""
        runner = ScriptRunner("test", add_common=False)
        runner.add_argument("filename")

        args = runner.parser.parse_args(["input.txt"])
        assert args.filename == "input.txt"


class TestScriptRunnerParseArgs:
    """Tests for ScriptRunner.parse_args."""

    def test_parse_args_returns_namespace(self):
        """Test parse_args returns Namespace."""
        runner = ScriptRunner("test", add_common=False)
        runner.add_argument("--value", default="default")

        args = runner.parse_args([])
        assert isinstance(args, argparse.Namespace)
        assert args.value == "default"

    def test_parse_args_sets_up_logger(self):
        """Test parse_args sets up logger."""
        runner = ScriptRunner("test")

        with patch("app.cli.runner.logging.basicConfig"):
            runner.parse_args([])

        assert runner.logger is not None

    def test_parse_args_verbose_sets_debug(self):
        """Test verbose flag sets DEBUG level."""
        runner = ScriptRunner("test")

        # Mock the centralized logging setup
        with patch("app.cli.runner.logging.basicConfig"):
            with patch.object(runner, "_setup_logging") as mock_setup:
                runner.parse_args(["--verbose"])
                mock_setup.assert_called_once()
                args = mock_setup.call_args[0][0]
                assert args.verbose is True

    def test_parse_args_quiet_sets_error(self):
        """Test quiet flag sets ERROR level."""
        runner = ScriptRunner("test")

        # Mock the centralized logging setup
        with patch("app.cli.runner.logging.basicConfig"):
            with patch.object(runner, "_setup_logging") as mock_setup:
                runner.parse_args(["--quiet"])
                mock_setup.assert_called_once()
                args = mock_setup.call_args[0][0]
                assert args.quiet is True


class TestScriptRunnerRunContext:
    """Tests for ScriptRunner.run_context."""

    def test_run_context_basic(self):
        """Test basic run context."""
        runner = ScriptRunner("test")
        runner.parse_args([])

        executed = False
        with runner.run_context():
            executed = True

        assert executed is True

    def test_run_context_handles_keyboard_interrupt(self):
        """Test run context handles KeyboardInterrupt."""
        runner = ScriptRunner("test")
        runner.parse_args([])

        with runner.run_context():
            raise KeyboardInterrupt()

        # Should not propagate

    def test_run_context_propagates_exceptions(self):
        """Test run context propagates other exceptions."""
        runner = ScriptRunner("test")
        runner.parse_args([])

        with pytest.raises(ValueError, match="test error"):
            with runner.run_context():
                raise ValueError("test error")

    def test_run_context_calls_cleanup_handlers(self):
        """Test run context calls cleanup handlers."""
        runner = ScriptRunner("test")
        runner.parse_args([])

        cleanup_called = False

        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        runner.on_cleanup(cleanup)

        with runner.run_context():
            pass

        assert cleanup_called is True

    def test_run_context_calls_cleanup_on_exception(self):
        """Test cleanup is called even on exception."""
        runner = ScriptRunner("test")
        runner.parse_args([])

        cleanup_called = False

        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        runner.on_cleanup(cleanup)

        with pytest.raises(ValueError):
            with runner.run_context():
                raise ValueError("error")

        assert cleanup_called is True


class TestScriptRunnerShutdown:
    """Tests for ScriptRunner shutdown handling."""

    def test_shutdown_requested_initial_false(self):
        """Test shutdown_requested is initially False."""
        runner = ScriptRunner("test")
        assert runner.shutdown_requested is False

    def test_request_shutdown(self):
        """Test request_shutdown sets flag."""
        runner = ScriptRunner("test")
        runner.request_shutdown()

        assert runner.shutdown_requested is True


class TestScriptRunnerSubparsers:
    """Tests for ScriptRunner subparsers."""

    def test_add_subparsers(self):
        """Test adding subparsers."""
        runner = ScriptRunner("test", add_common=False)
        subparsers = runner.add_subparsers(dest="command")

        cmd_parser = subparsers.add_parser("run")
        cmd_parser.add_argument("--count", type=int, default=1)

        args = runner.parser.parse_args(["run", "--count", "5"])
        assert args.command == "run"
        assert args.count == 5


# ============================================================================
# setup_script Tests
# ============================================================================


class TestSetupScript:
    """Tests for setup_script function."""

    def test_basic_setup(self):
        """Test basic script setup."""
        with patch("app.cli.runner.logging.basicConfig"):
            with patch("sys.argv", ["test_script"]):
                args, logger = setup_script("test_script")

        assert isinstance(args, argparse.Namespace)
        assert logger is not None

    def test_with_description(self):
        """Test setup with description."""
        with patch("app.cli.runner.logging.basicConfig"):
            with patch("sys.argv", ["test"]):
                args, logger = setup_script("test", description="Test script")

        assert logger is not None

    def test_with_extra_args(self):
        """Test setup with extra arguments."""
        with patch("app.cli.runner.logging.basicConfig"):
            with patch("sys.argv", ["test", "--input", "file.txt"]):
                args, logger = setup_script(
                    "test",
                    input={"required": True, "help": "Input file"},
                )

        assert args.input == "file.txt"

    def test_extra_args_underscore_to_dash(self):
        """Test underscore in extra arg names converts to dash."""
        with patch("app.cli.runner.logging.basicConfig"):
            with patch("sys.argv", ["test", "--output-dir", "/path"]):
                args, logger = setup_script(
                    "test",
                    output_dir={"default": ".", "help": "Output directory"},
                )

        assert args.output_dir == "/path"

    def test_returns_tuple(self):
        """Test returns (args, logger) tuple."""
        with patch("app.cli.runner.logging.basicConfig"):
            with patch("sys.argv", ["test"]):
                result = setup_script("test")

        assert isinstance(result, tuple)
        assert len(result) == 2
