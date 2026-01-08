"""Tests for OWCPushDaemon.

Tests the OWC push daemon for backing up training data to external drive.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.coordination.owc_push_daemon import (
    OWCPushConfig,
    OWCPushStats,
    OWCPushDaemon,
    get_owc_push_daemon,
    reset_owc_push_daemon,
    _is_running_on_owc_host,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    OWCPushDaemon._instance = None
    yield
    OWCPushDaemon._instance = None


@pytest.fixture
def tmp_base_path(tmp_path):
    """Create temporary base path with required directories."""
    games_dir = tmp_path / "data" / "games"
    training_dir = tmp_path / "data" / "training"
    models_dir = tmp_path / "models"

    games_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    return tmp_path


@pytest.fixture
def daemon_with_tmp_path(tmp_base_path):
    """Create daemon with temporary base path."""
    daemon = OWCPushDaemon()
    daemon._base_path = tmp_base_path
    daemon._is_local = True  # Avoid SSH in tests
    return daemon


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestIsRunningOnOWCHost:
    """Tests for _is_running_on_owc_host helper function."""

    def test_exact_hostname_match(self):
        """Test exact hostname match."""
        with patch("socket.gethostname", return_value="mac-studio"):
            assert _is_running_on_owc_host("mac-studio") is True

    def test_hostname_with_local_suffix(self):
        """Test hostname with .local suffix."""
        with patch("socket.gethostname", return_value="mac-studio.local"):
            assert _is_running_on_owc_host("mac-studio") is True

    def test_hostname_case_insensitive(self):
        """Test case insensitive matching."""
        with patch("socket.gethostname", return_value="MAC-STUDIO"):
            assert _is_running_on_owc_host("mac-studio") is True

    def test_hostname_without_dashes(self):
        """Test hostname with dashes removed."""
        with patch("socket.gethostname", return_value="macstudio"):
            assert _is_running_on_owc_host("mac-studio") is True

    def test_localhost(self):
        """Test localhost detection."""
        with patch("socket.gethostname", return_value="anyhost"):
            assert _is_running_on_owc_host("localhost") is True
            assert _is_running_on_owc_host("127.0.0.1") is True
            assert _is_running_on_owc_host("::1") is True

    def test_different_hostname(self):
        """Test non-matching hostname."""
        with patch("socket.gethostname", return_value="nebius-h100"):
            assert _is_running_on_owc_host("mac-studio") is False


# ============================================================================
# OWCPushConfig Tests
# ============================================================================


class TestOWCPushConfig:
    """Tests for OWCPushConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OWCPushConfig()

        assert config.enabled is True
        assert config.push_interval == 21600  # 6 hours
        assert config.owc_base_path == "/Volumes/RingRift-Data"
        assert config.owc_host == "mac-studio"
        assert config.owc_user == "armand"
        assert config.ssh_timeout == 60
        assert config.rsync_timeout == 1800

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OWCPushConfig(
            enabled=False,
            push_interval=3600,
            owc_base_path="/custom/path",
            owc_host="custom-host",
            owc_user="custom-user",
        )

        assert config.enabled is False
        assert config.push_interval == 3600
        assert config.owc_base_path == "/custom/path"
        assert config.owc_host == "custom-host"
        assert config.owc_user == "custom-user"

    def test_env_var_override(self):
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "RINGRIFT_OWC_PUSH_ENABLED": "false",
                "RINGRIFT_OWC_PUSH_INTERVAL": "7200",
                "RINGRIFT_OWC_PUSH_PATH": "/env/path",
                "OWC_HOST": "env-host",
                "OWC_USER": "env-user",
            },
        ):
            config = OWCPushConfig()

            assert config.enabled is False
            assert config.push_interval == 7200
            assert config.owc_base_path == "/env/path"
            assert config.owc_host == "env-host"
            assert config.owc_user == "env-user"

    def test_subdirectory_defaults(self):
        """Test subdirectory path defaults."""
        config = OWCPushConfig()

        assert config.canonical_db_subdir == "consolidated/games"
        assert config.training_subdir == "consolidated/training"
        assert config.models_subdir == "models"


# ============================================================================
# OWCPushStats Tests
# ============================================================================


class TestOWCPushStats:
    """Tests for OWCPushStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = OWCPushStats()

        assert stats.total_files_pushed == 0
        assert stats.total_bytes_pushed == 0
        assert stats.last_push_time == 0.0
        assert stats.push_errors == 0
        assert stats.last_error is None
        assert stats.consecutive_failures == 0

    def test_custom_values(self):
        """Test custom statistics values."""
        stats = OWCPushStats(
            total_files_pushed=10,
            total_bytes_pushed=100 * 1024 * 1024,
            push_errors=2,
            last_error="Test error",
            consecutive_failures=3,
        )

        assert stats.total_files_pushed == 10
        assert stats.total_bytes_pushed == 104857600
        assert stats.push_errors == 2
        assert stats.last_error == "Test error"
        assert stats.consecutive_failures == 3


# ============================================================================
# OWCPushDaemon Basic Tests
# ============================================================================


class TestOWCPushDaemonBasic:
    """Basic tests for OWCPushDaemon."""

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        daemon1 = OWCPushDaemon.get_instance()
        daemon2 = OWCPushDaemon.get_instance()

        assert daemon1 is daemon2

    def test_singleton_reset(self):
        """Test singleton reset."""
        daemon1 = OWCPushDaemon.get_instance()
        OWCPushDaemon.reset_instance()
        daemon2 = OWCPushDaemon.get_instance()

        assert daemon1 is not daemon2

    def test_initialization_default(self):
        """Test initialization with default config."""
        with patch("socket.gethostname", return_value="nebius-h100"):
            daemon = OWCPushDaemon()

            assert daemon.config is not None
            assert daemon.config.push_interval == 21600
            assert daemon._owc_available is True
            assert daemon._is_local is False

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = OWCPushConfig(push_interval=1800, owc_host="custom-host")
        daemon = OWCPushDaemon(config=config)

        assert daemon.config.push_interval == 1800
        assert daemon.config.owc_host == "custom-host"

    def test_initialization_local_mode(self):
        """Test initialization when running on OWC host."""
        with patch("socket.gethostname", return_value="mac-studio"):
            daemon = OWCPushDaemon()

            assert daemon._is_local is True

    def test_module_functions(self):
        """Test module-level helper functions."""
        daemon1 = get_owc_push_daemon()
        daemon2 = get_owc_push_daemon()

        assert daemon1 is daemon2

        reset_owc_push_daemon()
        daemon3 = get_owc_push_daemon()

        assert daemon1 is not daemon3


# ============================================================================
# OWC Availability Tests
# ============================================================================


class TestOWCAvailability:
    """Tests for OWC availability checking."""

    @pytest.mark.asyncio
    async def test_check_available_local_exists(self, tmp_path):
        """Test local availability check when path exists."""
        config = OWCPushConfig(owc_base_path=str(tmp_path))
        daemon = OWCPushDaemon(config=config)
        daemon._is_local = True

        result = await daemon._check_owc_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_available_local_missing(self):
        """Test local availability check when path doesn't exist."""
        config = OWCPushConfig(owc_base_path="/nonexistent/path")
        daemon = OWCPushDaemon(config=config)
        daemon._is_local = True

        result = await daemon._check_owc_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_available_remote_success(self):
        """Test remote availability check success."""
        daemon = OWCPushDaemon()
        daemon._is_local = False

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("asyncio.to_thread", return_value=mock_result):
            result = await daemon._check_owc_available()

            assert result is True

    @pytest.mark.asyncio
    async def test_check_available_remote_failure(self):
        """Test remote availability check failure."""
        daemon = OWCPushDaemon()
        daemon._is_local = False

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("asyncio.to_thread", return_value=mock_result):
            result = await daemon._check_owc_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_check_available_remote_timeout(self):
        """Test remote availability check timeout."""
        import subprocess

        daemon = OWCPushDaemon()
        daemon._is_local = False

        with patch("asyncio.to_thread", side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=60)):
            result = await daemon._check_owc_available()

            assert result is False


# ============================================================================
# File Checksum Tests
# ============================================================================


class TestFileChecksum:
    """Tests for file checksum computation."""

    def test_compute_checksum(self, tmp_path):
        """Test checksum computation."""
        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"test content for checksum")

        daemon = OWCPushDaemon()
        checksum = daemon._compute_file_checksum(test_file)

        assert len(checksum) == 64  # SHA256 hex digest
        assert checksum.isalnum()

    def test_checksum_consistency(self, tmp_path):
        """Test checksum is consistent for same content."""
        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"consistent content")

        daemon = OWCPushDaemon()
        checksum1 = daemon._compute_file_checksum(test_file)
        checksum2 = daemon._compute_file_checksum(test_file)

        assert checksum1 == checksum2

    def test_checksum_different_content(self, tmp_path):
        """Test checksum differs for different content."""
        file1 = tmp_path / "file1.db"
        file2 = tmp_path / "file2.db"
        file1.write_bytes(b"content 1")
        file2.write_bytes(b"content 2")

        daemon = OWCPushDaemon()
        checksum1 = daemon._compute_file_checksum(file1)
        checksum2 = daemon._compute_file_checksum(file2)

        assert checksum1 != checksum2


# ============================================================================
# Push If Modified Tests
# ============================================================================


class TestPushIfModified:
    """Tests for push_if_modified method."""

    @pytest.mark.asyncio
    async def test_push_nonexistent_file(self, daemon_with_tmp_path):
        """Test pushing nonexistent file returns False."""
        result = await daemon_with_tmp_path._push_if_modified(
            Path("/nonexistent/file.db"), "dest/path"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_push_new_file(self, daemon_with_tmp_path, tmp_base_path):
        """Test pushing a new file."""
        test_file = tmp_base_path / "test.db"
        test_file.write_bytes(b"new file content")

        # Mock local push
        with patch.object(daemon_with_tmp_path, "_push_local", return_value=True):
            result = await daemon_with_tmp_path._push_if_modified(
                test_file, "consolidated/games/test.db"
            )

            assert result is True
            assert daemon_with_tmp_path._push_stats.total_files_pushed == 1

    @pytest.mark.asyncio
    async def test_skip_unmodified_file(self, daemon_with_tmp_path, tmp_base_path):
        """Test skipping unmodified file."""
        test_file = tmp_base_path / "test.db"
        test_file.write_bytes(b"file content")

        # Record the file as already pushed
        mtime = test_file.stat().st_mtime
        daemon_with_tmp_path._last_push_times[str(test_file)] = mtime + 1  # Future mtime

        result = await daemon_with_tmp_path._push_if_modified(
            test_file, "consolidated/games/test.db"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_skip_checksum_match(self, daemon_with_tmp_path, tmp_base_path):
        """Test skipping file with matching checksum."""
        test_file = tmp_base_path / "test.db"
        test_file.write_bytes(b"file content")

        # Compute and store checksum
        checksum = daemon_with_tmp_path._compute_file_checksum(test_file)
        daemon_with_tmp_path._file_checksums[str(test_file)] = checksum
        daemon_with_tmp_path._last_push_times[str(test_file)] = 0  # Old mtime

        result = await daemon_with_tmp_path._push_if_modified(
            test_file, "consolidated/games/test.db"
        )

        assert result is False


# ============================================================================
# Local Push Tests
# ============================================================================


class TestPushLocal:
    """Tests for local push method."""

    @pytest.mark.asyncio
    async def test_push_local_success(self, daemon_with_tmp_path, tmp_base_path):
        """Test successful local push."""
        source = tmp_base_path / "source.db"
        dest = tmp_base_path / "dest" / "target.db"

        source.write_bytes(b"source content")

        result = await daemon_with_tmp_path._push_local(source, str(dest))

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"source content"

    @pytest.mark.asyncio
    async def test_push_local_creates_directories(self, daemon_with_tmp_path, tmp_base_path):
        """Test local push creates destination directories."""
        source = tmp_base_path / "source.db"
        dest = tmp_base_path / "deep" / "nested" / "dir" / "target.db"

        source.write_bytes(b"content")

        result = await daemon_with_tmp_path._push_local(source, str(dest))

        assert result is True
        assert dest.parent.exists()

    @pytest.mark.asyncio
    async def test_push_local_failure(self, daemon_with_tmp_path, tmp_base_path):
        """Test local push failure handling."""
        source = tmp_base_path / "source.db"
        source.write_bytes(b"content")

        with patch("shutil.copy2", side_effect=OSError("Disk full")):
            result = await daemon_with_tmp_path._push_local(source, "/invalid/dest")

            assert result is False


# ============================================================================
# Remote Push Tests
# ============================================================================


class TestPushRemote:
    """Tests for remote push method."""

    @pytest.mark.asyncio
    async def test_push_remote_success(self, tmp_base_path):
        """Test successful remote push."""
        daemon = OWCPushDaemon()
        daemon._is_local = False

        source = tmp_base_path / "source.db"
        source.write_bytes(b"content")

        # Mock successful SSH mkdir and rsync
        mock_mkdir_result = MagicMock()
        mock_mkdir_result.returncode = 0

        mock_rsync_result = MagicMock()
        mock_rsync_result.returncode = 0

        results = [mock_mkdir_result, mock_rsync_result]

        with patch("asyncio.to_thread", side_effect=results):
            result = await daemon._push_remote(source, "/dest/target.db")

            assert result is True

    @pytest.mark.asyncio
    async def test_push_remote_rsync_failure(self, tmp_base_path):
        """Test remote push with rsync failure."""
        daemon = OWCPushDaemon()
        daemon._is_local = False

        source = tmp_base_path / "source.db"
        source.write_bytes(b"content")

        mock_mkdir_result = MagicMock()
        mock_mkdir_result.returncode = 0

        mock_rsync_result = MagicMock()
        mock_rsync_result.returncode = 1
        mock_rsync_result.stderr = "rsync error"

        results = [mock_mkdir_result, mock_rsync_result]

        with patch("asyncio.to_thread", side_effect=results):
            result = await daemon._push_remote(source, "/dest/target.db")

            assert result is False

    @pytest.mark.asyncio
    async def test_push_remote_timeout(self, tmp_base_path):
        """Test remote push timeout."""
        import subprocess

        daemon = OWCPushDaemon()
        daemon._is_local = False

        source = tmp_base_path / "source.db"
        source.write_bytes(b"content")

        with patch("asyncio.to_thread", side_effect=subprocess.TimeoutExpired(cmd="rsync", timeout=1800)):
            result = await daemon._push_remote(source, "/dest/target.db")

            assert result is False


# ============================================================================
# Push Category Tests
# ============================================================================


class TestPushCategories:
    """Tests for push category methods."""

    @pytest.mark.asyncio
    async def test_push_canonical_databases(self, daemon_with_tmp_path, tmp_base_path):
        """Test pushing canonical databases."""
        games_dir = tmp_base_path / "data" / "games"
        (games_dir / "canonical_hex8_2p.db").write_bytes(b"db1")
        (games_dir / "canonical_hex8_4p.db").write_bytes(b"db2")
        (games_dir / "selfplay.db").write_bytes(b"ignored")  # Non-canonical

        with patch.object(daemon_with_tmp_path, "_push_if_modified", return_value=True) as mock_push:
            count = await daemon_with_tmp_path._push_canonical_databases()

            assert count == 2
            assert mock_push.call_count == 2

    @pytest.mark.asyncio
    async def test_push_training_files(self, daemon_with_tmp_path, tmp_base_path):
        """Test pushing training NPZ files."""
        training_dir = tmp_base_path / "data" / "training"
        (training_dir / "hex8_2p.npz").write_bytes(b"npz1")
        (training_dir / "hex8_4p.npz").write_bytes(b"npz2")

        with patch.object(daemon_with_tmp_path, "_push_if_modified", return_value=True) as mock_push:
            count = await daemon_with_tmp_path._push_training_files()

            assert count == 2
            assert mock_push.call_count == 2

    @pytest.mark.asyncio
    async def test_push_models(self, daemon_with_tmp_path, tmp_base_path):
        """Test pushing model checkpoints."""
        models_dir = tmp_base_path / "models"
        (models_dir / "canonical_hex8_2p.pth").write_bytes(b"model1")
        (models_dir / "canonical_hex8_4p.pth").write_bytes(b"model2")
        (models_dir / "other.pth").write_bytes(b"ignored")  # Non-canonical

        with patch.object(daemon_with_tmp_path, "_push_if_modified", return_value=True) as mock_push:
            count = await daemon_with_tmp_path._push_models()

            assert count == 2
            assert mock_push.call_count == 2

    @pytest.mark.asyncio
    async def test_push_empty_directory(self, daemon_with_tmp_path, tmp_base_path):
        """Test pushing from empty directory."""
        # Remove files (directory exists but is empty)
        count = await daemon_with_tmp_path._push_canonical_databases()
        assert count == 0


class TestShouldPushModels:
    """Tests for model push frequency logic."""

    def test_push_on_first_cycle(self):
        """Test models pushed on first cycle."""
        daemon = OWCPushDaemon()
        daemon._cycle_count = 0

        assert daemon._should_push_models() is True

    def test_skip_intermediate_cycles(self):
        """Test models skipped on intermediate cycles."""
        daemon = OWCPushDaemon()
        daemon._cycle_count = 1

        assert daemon._should_push_models() is False
        assert daemon._should_push_models() is False
        assert daemon._should_push_models() is False

    def test_push_every_fourth_cycle(self):
        """Test models pushed every 4th cycle."""
        daemon = OWCPushDaemon()
        daemon._cycle_count = 4

        assert daemon._should_push_models() is True


# ============================================================================
# Run Cycle Tests
# ============================================================================


class TestRunCycle:
    """Tests for daemon run cycle."""

    @pytest.mark.asyncio
    async def test_run_cycle_disabled(self):
        """Test run cycle when daemon is disabled."""
        config = OWCPushConfig(enabled=False)
        daemon = OWCPushDaemon(config=config)

        with patch.object(daemon, "_check_owc_available") as mock_check:
            await daemon._run_cycle()

            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_owc_unavailable(self, daemon_with_tmp_path):
        """Test run cycle when OWC is unavailable."""
        with patch.object(daemon_with_tmp_path, "_check_owc_available", return_value=False):
            await daemon_with_tmp_path._run_cycle()

            assert daemon_with_tmp_path._owc_available is False
            assert daemon_with_tmp_path._push_stats.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_run_cycle_success(self, daemon_with_tmp_path, tmp_base_path):
        """Test successful run cycle."""
        # Create some test files
        games_dir = tmp_base_path / "data" / "games"
        (games_dir / "canonical_hex8_2p.db").write_bytes(b"db")

        with patch.object(daemon_with_tmp_path, "_check_owc_available", return_value=True):
            with patch.object(daemon_with_tmp_path, "_push_if_modified", return_value=True):
                await daemon_with_tmp_path._run_cycle()

                assert daemon_with_tmp_path._owc_available is True
                assert daemon_with_tmp_path._push_stats.last_push_time > 0


# ============================================================================
# Event Subscription Tests
# ============================================================================


class TestEventSubscriptions:
    """Tests for event subscriptions."""

    def test_get_event_subscriptions(self):
        """Test event subscription registration."""
        daemon = OWCPushDaemon()

        subs = daemon._get_event_subscriptions()

        assert "DATA_SYNC_COMPLETED" in subs
        assert "TRAINING_COMPLETED" in subs
        assert "NPZ_EXPORT_COMPLETE" in subs
        assert "CONSOLIDATION_COMPLETE" in subs

    @pytest.mark.asyncio
    async def test_on_data_sync_completed(self, daemon_with_tmp_path, tmp_base_path):
        """Test data sync completed handler."""
        db_path = tmp_base_path / "data" / "games" / "canonical_hex8_2p.db"
        db_path.write_bytes(b"synced db")

        event = {"needs_owc_backup": True, "db_path": str(db_path)}

        with patch.object(daemon_with_tmp_path, "_push_if_modified") as mock_push:
            await daemon_with_tmp_path._on_data_sync_completed(event)

            mock_push.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_data_sync_completed_no_backup_needed(self, daemon_with_tmp_path):
        """Test data sync handler when backup not needed."""
        event = {"needs_owc_backup": False}

        with patch.object(daemon_with_tmp_path, "_push_if_modified") as mock_push:
            await daemon_with_tmp_path._on_data_sync_completed(event)

            mock_push.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_training_completed(self, daemon_with_tmp_path, tmp_base_path):
        """Test training completed handler."""
        model_path = tmp_base_path / "models" / "canonical_hex8_2p.pth"
        model_path.write_bytes(b"model")

        event = {"model_path": str(model_path)}

        with patch.object(daemon_with_tmp_path, "_push_if_modified") as mock_push:
            await daemon_with_tmp_path._on_training_completed(event)

            mock_push.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_npz_export_complete(self, daemon_with_tmp_path, tmp_base_path):
        """Test NPZ export completed handler."""
        npz_path = tmp_base_path / "data" / "training" / "hex8_2p.npz"
        npz_path.write_bytes(b"npz")

        event = {"npz_path": str(npz_path)}

        with patch.object(daemon_with_tmp_path, "_push_if_modified") as mock_push:
            await daemon_with_tmp_path._on_npz_export_complete(event)

            mock_push.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_consolidation_complete(self, daemon_with_tmp_path, tmp_base_path):
        """Test consolidation completed handler."""
        db_path = tmp_base_path / "data" / "games" / "canonical_hex8_2p.db"
        db_path.write_bytes(b"consolidated db")

        event = {"canonical_db_path": str(db_path)}

        with patch.object(daemon_with_tmp_path, "_push_if_modified") as mock_push:
            await daemon_with_tmp_path._on_consolidation_complete(event)

            mock_push.assert_called_once()


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health check."""

    def test_health_check_not_running(self):
        """Test health check when not running."""
        daemon = OWCPushDaemon()
        daemon._running = False

        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()

    def test_health_check_owc_unavailable(self):
        """Test health check when OWC unavailable."""
        daemon = OWCPushDaemon()
        daemon._running = True
        daemon._owc_available = False
        daemon._push_stats.consecutive_failures = 3

        result = daemon.health_check()

        assert result.healthy is True  # Still healthy, just OWC unavailable
        assert "not available" in result.message.lower()
        assert result.details["consecutive_failures"] == 3

    def test_health_check_healthy_local(self):
        """Test health check when healthy (local mode)."""
        daemon = OWCPushDaemon()
        daemon._running = True
        daemon._owc_available = True
        daemon._is_local = True
        daemon._push_stats.total_files_pushed = 15
        daemon._stats.cycles_completed = 3

        result = daemon.health_check()

        assert result.healthy is True
        assert "local" in result.message
        assert "15 files" in result.message
        assert result.details["is_local"] is True

    def test_health_check_healthy_remote(self):
        """Test health check when healthy (remote mode)."""
        daemon = OWCPushDaemon()
        daemon._running = True
        daemon._owc_available = True
        daemon._is_local = False
        daemon._push_stats.total_files_pushed = 10

        result = daemon.health_check()

        assert result.healthy is True
        assert "remote" in result.message
        assert result.details["is_local"] is False


# ============================================================================
# Statistics Tests
# ============================================================================


class TestGetStats:
    """Tests for statistics retrieval."""

    def test_get_stats(self):
        """Test statistics retrieval."""
        daemon = OWCPushDaemon()
        daemon._push_stats.total_files_pushed = 20
        daemon._push_stats.total_bytes_pushed = 50 * 1024 * 1024
        daemon._push_stats.push_errors = 2
        daemon._push_stats.last_error = "Test error"
        daemon._last_push_times = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
        daemon._owc_available = True
        daemon._is_local = True

        stats = daemon.get_stats()

        assert stats["total_files_pushed"] == 20
        assert stats["total_mb_pushed"] == 50.0
        assert stats["push_errors"] == 2
        assert stats["last_error"] == "Test error"
        assert stats["tracked_files"] == 3
        assert stats["owc_available"] is True
        assert stats["is_local"] is True
