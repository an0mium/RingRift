"""Tests for app.execution.executor module.

Tests the unified execution framework including ExecutionResult, SSHConfig,
LocalExecutor, and SSHExecutor.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.execution.executor import (
    ExecutionResult,
    SSHConfig,
    LocalExecutor,
    SSHExecutor,
)


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = ExecutionResult(
            success=True,
            returncode=0,
            stdout="Hello, World!",
            stderr="",
            duration_seconds=1.5,
            command="echo 'Hello, World!'",
        )
        assert result.success is True
        assert result.returncode == 0
        assert result.stdout == "Hello, World!"
        assert result.duration_seconds == 1.5
        assert result.host == "local"  # default
        assert result.timed_out is False  # default

    def test_failed_result(self):
        """Test creating a failed result."""
        result = ExecutionResult(
            success=False,
            returncode=1,
            stdout="",
            stderr="Command not found",
            duration_seconds=0.1,
            command="nonexistent_command",
        )
        assert result.success is False
        assert result.returncode == 1
        assert result.stderr == "Command not found"

    def test_timed_out_result(self):
        """Test creating a timed out result."""
        result = ExecutionResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="Timed out after 30s",
            duration_seconds=30.0,
            command="sleep 60",
            timed_out=True,
        )
        assert result.timed_out is True
        assert result.success is False

    def test_output_property_combined(self):
        """Test output property combines stdout and stderr."""
        result = ExecutionResult(
            success=True,
            returncode=0,
            stdout="Standard output",
            stderr="Standard error",
            duration_seconds=1.0,
            command="test",
        )
        assert "Standard output" in result.output
        assert "Standard error" in result.output

    def test_output_property_empty_stderr(self):
        """Test output property with empty stderr."""
        result = ExecutionResult(
            success=True,
            returncode=0,
            stdout="Output only",
            stderr="",
            duration_seconds=1.0,
            command="test",
        )
        assert result.output == "Output only"

    def test_bool_true_for_success(self):
        """Test bool() returns True for successful result."""
        result = ExecutionResult(
            success=True,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
            command="true",
        )
        assert bool(result) is True

    def test_bool_false_for_failure(self):
        """Test bool() returns False for failed result."""
        result = ExecutionResult(
            success=False,
            returncode=1,
            stdout="",
            stderr="",
            duration_seconds=1.0,
            command="false",
        )
        assert bool(result) is False

    def test_remote_host(self):
        """Test result with remote host."""
        result = ExecutionResult(
            success=True,
            returncode=0,
            stdout="Remote output",
            stderr="",
            duration_seconds=2.0,
            command="hostname",
            host="worker-1",
        )
        assert result.host == "worker-1"


class TestSSHConfig:
    """Tests for SSHConfig dataclass."""

    def test_basic_config(self):
        """Test basic SSH config creation."""
        config = SSHConfig(host="192.168.1.10")
        assert config.host == "192.168.1.10"
        assert config.user is None
        assert config.port == 22
        assert config.key_path is None
        assert config.connect_timeout == 10

    def test_config_with_user(self):
        """Test SSH config with user."""
        config = SSHConfig(host="192.168.1.10", user="ubuntu")
        assert config.user == "ubuntu"
        assert config.ssh_target == "ubuntu@192.168.1.10"

    def test_config_without_user(self):
        """Test SSH target without user."""
        config = SSHConfig(host="192.168.1.10")
        assert config.ssh_target == "192.168.1.10"

    def test_config_with_custom_port(self):
        """Test SSH config with custom port."""
        config = SSHConfig(host="192.168.1.10", port=2222)
        assert config.port == 2222

    def test_config_with_key_path(self):
        """Test SSH config with key path."""
        config = SSHConfig(
            host="192.168.1.10",
            key_path="/home/user/.ssh/id_rsa",
        )
        assert config.key_path == "/home/user/.ssh/id_rsa"

    def test_build_ssh_command_basic(self):
        """Test building basic SSH command."""
        config = SSHConfig(host="192.168.1.10")
        cmd = config.build_ssh_command()

        assert cmd[0] == "ssh"
        assert "-o" in cmd
        assert "BatchMode=yes" in cmd
        assert "192.168.1.10" in cmd

    def test_build_ssh_command_with_user(self):
        """Test building SSH command with user."""
        config = SSHConfig(host="192.168.1.10", user="ubuntu")
        cmd = config.build_ssh_command()

        assert "ubuntu@192.168.1.10" in cmd

    def test_build_ssh_command_with_port(self):
        """Test building SSH command with custom port."""
        config = SSHConfig(host="192.168.1.10", port=2222)
        cmd = config.build_ssh_command()

        port_idx = cmd.index("-p")
        assert cmd[port_idx + 1] == "2222"

    def test_build_ssh_command_with_key(self):
        """Test building SSH command with key file."""
        config = SSHConfig(
            host="192.168.1.10",
            key_path="/path/to/key",
        )
        cmd = config.build_ssh_command()

        key_idx = cmd.index("-i")
        assert cmd[key_idx + 1] == "/path/to/key"

    def test_build_ssh_command_with_options(self):
        """Test building SSH command with custom options."""
        config = SSHConfig(
            host="192.168.1.10",
            options={"ServerAliveInterval": "60"},
        )
        cmd = config.build_ssh_command()

        assert "ServerAliveInterval=60" in cmd

    def test_connect_timeout_in_command(self):
        """Test connect timeout is included in command."""
        config = SSHConfig(host="192.168.1.10", connect_timeout=30)
        cmd = config.build_ssh_command()

        assert "ConnectTimeout=30" in cmd


class TestLocalExecutor:
    """Tests for LocalExecutor class."""

    def test_name(self):
        """Test executor name property."""
        executor = LocalExecutor()
        assert executor.name == "local"

    def test_init_with_working_dir(self):
        """Test initialization with working directory."""
        executor = LocalExecutor(working_dir="/tmp")
        assert executor.working_dir == "/tmp"

    def test_init_with_resource_check(self):
        """Test initialization with resource checking."""
        executor = LocalExecutor(
            check_resources=True,
            required_mem_gb=4.0,
        )
        assert executor.check_resources is True
        assert executor.required_mem_gb == 4.0

    @pytest.mark.asyncio
    async def test_check_available(self):
        """Test local executor is always available."""
        executor = LocalExecutor()
        assert await executor.check_available() is True

    @pytest.mark.asyncio
    async def test_run_simple_command(self):
        """Test running a simple command."""
        executor = LocalExecutor()
        result = await executor.run("echo 'hello'")

        assert result.success is True
        assert result.returncode == 0
        assert "hello" in result.stdout
        assert result.host == "local"

    @pytest.mark.asyncio
    async def test_run_failing_command(self):
        """Test running a failing command."""
        executor = LocalExecutor()
        result = await executor.run("exit 1")

        assert result.success is False
        assert result.returncode == 1

    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        """Test command with timeout."""
        executor = LocalExecutor()
        result = await executor.run("sleep 10", timeout=1)

        assert result.success is False
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_run_with_env(self):
        """Test running command with environment variables."""
        executor = LocalExecutor()
        result = await executor.run(
            "echo $MY_VAR",
            env={"MY_VAR": "test_value"},
        )

        assert result.success is True
        assert "test_value" in result.stdout

    @pytest.mark.asyncio
    async def test_run_with_cwd(self):
        """Test running command with working directory."""
        executor = LocalExecutor()
        result = await executor.run("pwd", cwd="/tmp")

        assert result.success is True
        assert "/tmp" in result.stdout

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        """Test that duration is recorded."""
        executor = LocalExecutor()
        result = await executor.run("sleep 0.1")

        assert result.duration_seconds >= 0.1


class TestSSHExecutor:
    """Tests for SSHExecutor class."""

    def test_name(self):
        """Test executor name property."""
        executor = SSHExecutor(host="192.168.1.10")
        assert "ssh:" in executor.name
        assert "192.168.1.10" in executor.name

    def test_name_with_user(self):
        """Test executor name with user."""
        executor = SSHExecutor(host="192.168.1.10", user="ubuntu")
        assert "ubuntu@192.168.1.10" in executor.name

    def test_init_with_config(self):
        """Test initialization creates correct config."""
        executor = SSHExecutor(
            host="192.168.1.10",
            user="ubuntu",
            port=2222,
            key_path="/path/to/key",
            connect_timeout=30,
        )

        assert executor.config.host == "192.168.1.10"
        assert executor.config.user == "ubuntu"
        assert executor.config.port == 2222
        assert executor.config.key_path == "/path/to/key"
        assert executor.config.connect_timeout == 30

    def test_init_retry_settings(self):
        """Test initialization with retry settings."""
        executor = SSHExecutor(
            host="192.168.1.10",
            max_retries=5,
            retry_delay=5.0,
        )

        assert executor.max_retries == 5
        assert executor.retry_delay == 5.0

    def test_default_retry_settings(self):
        """Test default retry settings."""
        executor = SSHExecutor(host="192.168.1.10")

        assert executor.max_retries == 3
        assert executor.retry_delay == 2.0


class TestExecutorIntegration:
    """Integration tests for executors."""

    @pytest.mark.asyncio
    async def test_local_executor_captures_stderr(self):
        """Test local executor captures stderr."""
        executor = LocalExecutor()
        result = await executor.run("echo 'error' >&2")

        assert "error" in result.stderr or "error" in result.output

    @pytest.mark.asyncio
    async def test_local_executor_command_in_result(self):
        """Test command is preserved in result."""
        executor = LocalExecutor()
        cmd = "echo 'test command'"
        result = await executor.run(cmd)

        assert result.command == cmd

    @pytest.mark.asyncio
    async def test_local_executor_multiline_output(self):
        """Test handling multiline output."""
        executor = LocalExecutor()
        result = await executor.run("echo 'line1' && echo 'line2'")

        assert "line1" in result.stdout
        assert "line2" in result.stdout
