"""Tests for app/core/ssh.py - Unified SSH helper module.

Tests the SSHClient class, SSHConfig, SSHResult dataclasses,
and convenience functions for SSH operations.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.ssh import (
    SSHClient,
    SSHConfig,
    SSHResult,
    get_ssh_client,
    run_ssh_command,
    run_ssh_command_async,
    run_ssh_command_sync,
)


class TestSSHConfig:
    """Tests for SSHConfig dataclass."""

    def test_default_values(self):
        """SSHConfig should have sensible defaults."""
        config = SSHConfig(host="test-host")
        assert config.host == "test-host"
        assert config.port == 22
        assert config.user is None
        assert config.key_path is None
        assert config.connect_timeout == 10
        assert config.command_timeout == 60
        assert config.use_control_master is True
        assert config.control_persist == 300
        assert config.server_alive_interval == 30
        assert config.server_alive_count_max == 4

    def test_custom_values(self):
        """SSHConfig should accept custom values."""
        config = SSHConfig(
            host="custom-host",
            port=2222,
            user="admin",
            key_path="/path/to/key",
            connect_timeout=30,
            command_timeout=120,
            tailscale_ip="100.1.2.3",
            work_dir="/workspace",
            venv_activate="source /path/to/venv/bin/activate",
        )
        assert config.host == "custom-host"
        assert config.port == 2222
        assert config.user == "admin"
        assert config.key_path == "/path/to/key"
        assert config.connect_timeout == 30
        assert config.command_timeout == 120
        assert config.tailscale_ip == "100.1.2.3"
        assert config.work_dir == "/workspace"
        assert config.venv_activate == "source /path/to/venv/bin/activate"


class TestSSHResult:
    """Tests for SSHResult dataclass."""

    def test_success_result(self):
        """SSHResult should represent successful command."""
        result = SSHResult(
            success=True,
            returncode=0,
            stdout="Hello World",
            stderr="",
            elapsed_ms=500.0,
            command="echo 'Hello World'",
            host="test-host",
        )
        assert result.success is True
        assert result.stdout == "Hello World"
        assert result.stderr == ""
        assert result.returncode == 0
        assert result.command == "echo 'Hello World'"
        assert result.elapsed_ms == 500.0
        assert result.host == "test-host"

    def test_failure_result(self):
        """SSHResult should represent failed command."""
        result = SSHResult(
            success=False,
            returncode=127,
            stdout="",
            stderr="Command not found",
            elapsed_ms=100.0,
            command="nonexistent",
            host="test-host",
        )
        assert result.success is False
        assert result.stderr == "Command not found"
        assert result.returncode == 127

    def test_error_result(self):
        """SSHResult with error should not be success."""
        result = SSHResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="",
            elapsed_ms=0.0,
            command="test",
            host="test-host",
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"

    def test_timed_out_result(self):
        """SSHResult should track timeout status."""
        result = SSHResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="",
            elapsed_ms=30000.0,
            command="slow-command",
            host="test-host",
            timed_out=True,
        )
        assert result.timed_out is True
        assert result.success is False

    def test_bool_conversion(self):
        """SSHResult should be truthy when successful."""
        success = SSHResult(
            success=True, returncode=0, stdout="", stderr="",
            elapsed_ms=0, command="", host="",
        )
        failure = SSHResult(
            success=False, returncode=1, stdout="", stderr="",
            elapsed_ms=0, command="", host="",
        )
        assert bool(success) is True
        assert bool(failure) is False

    def test_output_property(self):
        """output property should combine stdout and stderr."""
        result = SSHResult(
            success=True, returncode=0,
            stdout="stdout line",
            stderr="stderr line",
            elapsed_ms=0, command="", host="",
        )
        assert "stdout line" in result.output
        assert "stderr line" in result.output


class TestSSHClient:
    """Tests for SSHClient class."""

    def test_init_with_config(self):
        """SSHClient should accept SSHConfig."""
        config = SSHConfig(host="test-host", port=2222)
        client = SSHClient(config)
        assert client.config.host == "test-host"
        assert client.config.port == 2222

    def test_build_ssh_command(self):
        """SSHClient should build correct SSH command."""
        config = SSHConfig(
            host="test-host",
            port=2222,
            user="ubuntu",
            connect_timeout=30,
        )
        client = SSHClient(config)
        cmd = client._build_ssh_command("echo test")

        # Should include ssh binary
        assert cmd[0] == "ssh"
        # Should include port
        assert "-p" in cmd
        assert "2222" in cmd
        # Command should be at the end
        assert cmd[-1] == "echo test"
        # Should include user@host target
        assert "ubuntu@test-host" in cmd

    def test_build_ssh_command_with_key(self):
        """SSHClient should include -i flag when key file exists."""
        import tempfile
        import os

        # Create a temp key file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            f.write(b"fake key content")
            key_path = f.name

        try:
            config = SSHConfig(
                host="test-host",
                user="ubuntu",
                key_path=key_path,
            )
            client = SSHClient(config)
            cmd = client._build_ssh_command("echo test")

            # Should include -i flag when key exists
            assert "-i" in cmd
            assert key_path in cmd
        finally:
            os.unlink(key_path)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_ssh_client(self):
        """get_ssh_client should return SSHClient instance."""
        client = get_ssh_client("test-host")
        assert isinstance(client, SSHClient)
        assert client.config.host == "test-host"

    def test_get_ssh_client_caches_clients(self):
        """get_ssh_client should return cached client for same host."""
        client1 = get_ssh_client("cache-test-host")
        client2 = get_ssh_client("cache-test-host")
        assert client1 is client2

    def test_run_ssh_command_sync_accepts_port(self):
        """run_ssh_command_sync should accept port parameter."""
        # Test that the sync function accepts port/user/key_path kwargs
        # (We don't actually run SSH, just verify the API)
        from app.core.ssh import run_ssh_command_sync
        import inspect
        sig = inspect.signature(run_ssh_command_sync)
        params = list(sig.parameters.keys())
        assert "port" in params
        assert "user" in params
        assert "key_path" in params
