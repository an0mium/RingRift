"""Tests for rsync_command_builder.py.

December 2025: Tests for the unified rsync command building utilities.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.rsync_command_builder import (
    RsyncCommand,
    RsyncCommandBuilder,
    SSHOptions,
    TimeoutCalculator,
    build_ssh_options,
    get_timeout,
)


# =============================================================================
# SSHOptions Tests
# =============================================================================


class TestSSHOptions:
    """Tests for SSHOptions dataclass."""

    def test_default_options(self):
        """Test default SSH options."""
        opts = SSHOptions(key_path="/path/to/key")
        assert opts.connect_timeout == 10
        assert opts.strict_host_key_checking is False
        assert opts.tcp_keepalive is True

    def test_to_string_basic(self):
        """Test SSH options string generation."""
        opts = SSHOptions(key_path="/path/to/key")
        result = opts.to_string()
        assert "ssh -i /path/to/key" in result
        assert "-o StrictHostKeyChecking=no" in result
        assert "-o ConnectTimeout=10" in result
        assert "-o TCPKeepAlive=yes" in result

    def test_to_string_custom_timeout(self):
        """Test custom connect timeout."""
        opts = SSHOptions(key_path="/path/to/key", connect_timeout=30)
        result = opts.to_string()
        assert "-o ConnectTimeout=30" in result

    def test_to_string_strict_host_key(self):
        """Test with strict host key checking enabled."""
        opts = SSHOptions(key_path="/path/to/key", strict_host_key_checking=True)
        result = opts.to_string()
        assert "StrictHostKeyChecking=no" not in result

    def test_to_string_no_keepalive(self):
        """Test without TCP keepalive."""
        opts = SSHOptions(key_path="/path/to/key", tcp_keepalive=False)
        result = opts.to_string()
        assert "TCPKeepAlive" not in result

    def test_frozen_dataclass(self):
        """Test that SSHOptions is immutable."""
        opts = SSHOptions(key_path="/path/to/key")
        with pytest.raises(AttributeError):
            opts.key_path = "/different/path"


# =============================================================================
# TimeoutCalculator Tests
# =============================================================================


class TestTimeoutCalculator:
    """Tests for TimeoutCalculator."""

    def test_standard_strategy_minimum(self):
        """Test standard strategy respects minimum timeout."""
        result = TimeoutCalculator.calculate(0, strategy="standard")
        assert result == 120  # Minimum for standard

    def test_standard_strategy_maximum(self):
        """Test standard strategy respects maximum timeout."""
        result = TimeoutCalculator.calculate(10000, strategy="standard")
        assert result == 1800  # Maximum for standard

    def test_standard_strategy_calculation(self):
        """Test standard strategy timeout calculation."""
        # 100 MB: 60 + 100*2 = 260, within bounds
        result = TimeoutCalculator.calculate(100, strategy="standard")
        assert result == 260

    def test_ephemeral_strategy_minimum(self):
        """Test ephemeral strategy respects minimum timeout."""
        result = TimeoutCalculator.calculate(0, strategy="ephemeral")
        assert result == 60  # Minimum for ephemeral

    def test_ephemeral_strategy_maximum(self):
        """Test ephemeral strategy respects maximum timeout."""
        result = TimeoutCalculator.calculate(10000, strategy="ephemeral")
        assert result == 900  # Maximum for ephemeral

    def test_ephemeral_strategy_calculation(self):
        """Test ephemeral strategy timeout calculation."""
        # 100 MB: 40 + 100*1.5 = 190, within bounds
        result = TimeoutCalculator.calculate(100, strategy="ephemeral")
        assert result == 190

    def test_pull_strategy_minimum(self):
        """Test pull strategy respects minimum timeout."""
        result = TimeoutCalculator.calculate(0, strategy="pull")
        assert result == 120  # Minimum for pull

    def test_pull_strategy_maximum(self):
        """Test pull strategy respects maximum timeout."""
        result = TimeoutCalculator.calculate(10000, strategy="pull")
        assert result == 2400  # Maximum for pull

    def test_pull_strategy_calculation(self):
        """Test pull strategy timeout calculation."""
        # 100 MB: 60 + 100*2.5 = 310, within bounds
        result = TimeoutCalculator.calculate(100, strategy="pull")
        assert result == 310

    def test_unknown_strategy_uses_standard(self):
        """Test unknown strategy falls back to standard."""
        result = TimeoutCalculator.calculate(100, strategy="unknown")
        assert result == TimeoutCalculator.calculate(100, strategy="standard")

    def test_from_path_existing_file(self, tmp_path):
        """Test timeout calculation from existing file."""
        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"x" * (10 * 1024 * 1024))  # 10 MB
        result = TimeoutCalculator.from_path(test_file, strategy="standard")
        # 10 MB: 60 + 10*2 = 80, but min is 120
        assert result == 120

    def test_from_path_nonexistent_file(self, tmp_path):
        """Test timeout calculation for nonexistent file uses default."""
        test_file = tmp_path / "nonexistent.db"
        result = TimeoutCalculator.from_path(test_file, strategy="standard", default_mb=100)
        # Default 100 MB: 60 + 100*2 = 260
        assert result == 260


# =============================================================================
# RsyncCommand Tests
# =============================================================================


class TestRsyncCommand:
    """Tests for RsyncCommand dataclass."""

    def test_command_creation(self):
        """Test basic command creation."""
        cmd = RsyncCommand(
            args=["rsync", "-avz", "source", "dest"],
            timeout=120,
            description="Test sync",
        )
        assert cmd.args == ["rsync", "-avz", "source", "dest"]
        assert cmd.timeout == 120
        assert cmd.description == "Test sync"

    @pytest.mark.asyncio
    async def test_execute_async_success(self):
        """Test async execution success."""
        cmd = RsyncCommand(
            args=["echo", "hello"],
            timeout=10,
        )
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))
            mock_exec.return_value = mock_proc

            returncode, stdout, stderr = await cmd.execute_async()

            assert returncode == 0
            assert "hello" in stdout

    def test_execute_sync_success(self):
        """Test sync execution."""
        cmd = RsyncCommand(
            args=["echo", "test"],
            timeout=10,
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="test\n",
                stderr="",
            )
            returncode, stdout, stderr = cmd.execute_sync()

            assert returncode == 0
            assert "test" in stdout


# =============================================================================
# RsyncCommandBuilder Tests
# =============================================================================


class TestRsyncCommandBuilder:
    """Tests for RsyncCommandBuilder."""

    @pytest.fixture
    def ssh_options(self):
        """Create default SSH options."""
        return SSHOptions(key_path="/path/to/key")

    def test_build_for_push_basic(self, ssh_options, tmp_path):
        """Test basic push command building."""
        source = tmp_path / "test.db"
        source.touch()
        target = "user@host:/path/to/dest"

        cmd = RsyncCommandBuilder.build_for_push(
            source=source,
            target_path=target,
            ssh_options=ssh_options,
        )

        assert "rsync" in cmd.args
        assert "-avz" in cmd.args
        assert "--delay-updates" in cmd.args
        assert "--checksum" in cmd.args
        assert "--progress" in cmd.args
        assert str(source) in cmd.args
        assert target in cmd.args
        assert cmd.timeout >= 120  # Minimum timeout

    def test_build_for_push_bandwidth_limit(self, ssh_options, tmp_path):
        """Test push command includes bandwidth limit."""
        source = tmp_path / "test.db"
        source.touch()

        cmd = RsyncCommandBuilder.build_for_push(
            source=source,
            target_path="user@host:/dest",
            ssh_options=ssh_options,
            bandwidth_kbps=10000,
        )

        assert "--bwlimit=10000" in cmd.args

    def test_build_for_push_no_checksum(self, ssh_options, tmp_path):
        """Test push command without checksum."""
        source = tmp_path / "test.db"
        source.touch()

        cmd = RsyncCommandBuilder.build_for_push(
            source=source,
            target_path="user@host:/dest",
            ssh_options=ssh_options,
            use_checksum=False,
        )

        assert "--checksum" not in cmd.args

    def test_build_for_push_no_progress(self, ssh_options, tmp_path):
        """Test push command without progress."""
        source = tmp_path / "test.db"
        source.touch()

        cmd = RsyncCommandBuilder.build_for_push(
            source=source,
            target_path="user@host:/dest",
            ssh_options=ssh_options,
            use_progress=False,
        )

        assert "--progress" not in cmd.args

    def test_build_for_pull_basic(self, ssh_options, tmp_path):
        """Test basic pull command building."""
        remote = "user@host:/path/to/source.db"
        local = tmp_path / "dest.db"

        cmd = RsyncCommandBuilder.build_for_pull(
            remote_full=remote,
            local_path=local,
            ssh_options=ssh_options,
        )

        assert "rsync" in cmd.args
        assert "-az" in cmd.args
        assert "--timeout=60" in cmd.args
        assert remote in cmd.args
        assert str(local) in cmd.args

    def test_build_for_pull_custom_timeout(self, ssh_options, tmp_path):
        """Test pull command with custom timeout."""
        cmd = RsyncCommandBuilder.build_for_pull(
            remote_full="user@host:/src.db",
            local_path=tmp_path / "dest.db",
            ssh_options=ssh_options,
            timeout=300,
        )

        assert cmd.timeout == 300

    def test_build_for_ephemeral_basic(self, ssh_options):
        """Test basic ephemeral command building."""
        cmd = RsyncCommandBuilder.build_for_ephemeral(
            source_dir="/data/games/",
            target_path="user@host:/dest/",
            db_name="test.db",
            ssh_options=ssh_options,
        )

        assert "rsync" in cmd.args
        assert "-avz" in cmd.args
        assert "--compress" in cmd.args
        assert "--include=test.db" in cmd.args
        assert "--include=test.db-wal" in cmd.args
        assert "--include=test.db-shm" in cmd.args
        assert "--exclude=*" in cmd.args

    def test_build_for_ephemeral_bandwidth(self, ssh_options):
        """Test ephemeral command with bandwidth limit."""
        cmd = RsyncCommandBuilder.build_for_ephemeral(
            source_dir="/data/games/",
            target_path="user@host:/dest/",
            db_name="test.db",
            ssh_options=ssh_options,
            bandwidth_kbps=5000,
        )

        assert "--bwlimit=5000" in cmd.args

    def test_build_for_ephemeral_no_wal(self, ssh_options):
        """Test ephemeral command without WAL files."""
        cmd = RsyncCommandBuilder.build_for_ephemeral(
            source_dir="/data/games/",
            target_path="user@host:/dest/",
            db_name="test.db",
            ssh_options=ssh_options,
            include_wal=False,
        )

        assert "--include=test.db-wal" not in cmd.args
        assert "--include=test.db-shm" not in cmd.args

    def test_get_wal_include_args(self):
        """Test WAL include args generation."""
        args = RsyncCommandBuilder._get_wal_include_args("mydb.db")
        assert args == [
            "--include=mydb.db",
            "--include=mydb.db-wal",
            "--include=mydb.db-shm",
        ]


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_build_ssh_options(self):
        """Test build_ssh_options convenience function."""
        opts = build_ssh_options(
            key_path="/path/to/key",
            connect_timeout=30,
            strict_host_key_checking=True,
        )

        assert opts.key_path == "/path/to/key"
        assert opts.connect_timeout == 30
        assert opts.strict_host_key_checking is True

    def test_get_timeout(self):
        """Test get_timeout convenience function."""
        result = get_timeout(100, strategy="standard")
        assert result == TimeoutCalculator.calculate(100, strategy="standard")

    def test_get_timeout_default_strategy(self):
        """Test get_timeout with default strategy."""
        result = get_timeout(100)
        assert result == TimeoutCalculator.calculate(100, strategy="standard")


# =============================================================================
# Integration Tests
# =============================================================================


class TestRsyncCommandBuilderIntegration:
    """Integration tests for full workflow."""

    def test_full_push_workflow(self, tmp_path):
        """Test complete push workflow."""
        # Create test file
        source = tmp_path / "test.db"
        source.write_bytes(b"x" * 1024)  # 1 KB

        # Build SSH options
        ssh_opts = build_ssh_options("/path/to/key", connect_timeout=5)

        # Build command
        cmd = RsyncCommandBuilder.build_for_push(
            source=source,
            target_path="user@host:/dest/test.db",
            ssh_options=ssh_opts,
            bandwidth_kbps=10000,
        )

        # Verify command structure
        assert cmd.args[0] == "rsync"
        assert "-e" in cmd.args
        ssh_e_idx = cmd.args.index("-e")
        assert "/path/to/key" in cmd.args[ssh_e_idx + 1]
        assert cmd.timeout >= 120

    def test_full_pull_workflow(self, tmp_path):
        """Test complete pull workflow."""
        ssh_opts = build_ssh_options("/path/to/key")

        cmd = RsyncCommandBuilder.build_for_pull(
            remote_full="user@host:/data/source.db",
            local_path=tmp_path / "dest.db",
            ssh_options=ssh_opts,
        )

        assert cmd.args[0] == "rsync"
        assert "user@host:/data/source.db" in cmd.args

    def test_full_ephemeral_workflow(self):
        """Test complete ephemeral workflow."""
        ssh_opts = build_ssh_options("/path/to/key")

        cmd = RsyncCommandBuilder.build_for_ephemeral(
            source_dir="/data/games/",
            target_path="user@host:/backup/",
            db_name="selfplay.db",
            ssh_options=ssh_opts,
            bandwidth_kbps=5000,
        )

        # Verify includes WAL files
        assert "--include=selfplay.db" in cmd.args
        assert "--include=selfplay.db-wal" in cmd.args
        assert "--include=selfplay.db-shm" in cmd.args
        assert "--exclude=*" in cmd.args
