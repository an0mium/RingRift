"""Tests for OWCImportDaemon.

December 29, 2025: Comprehensive tests for OWC data import daemon.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.owc_import_daemon import (
    ImportStats,
    OWCDatabaseInfo,
    OWCImportConfig,
    OWCImportDaemon,
    get_owc_import_daemon,
    reset_owc_import_daemon,
)


# =============================================================================
# OWCImportConfig Tests
# =============================================================================


class TestOWCImportConfig:
    """Tests for OWCImportConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OWCImportConfig()
        assert config.check_interval_seconds == 3600
        assert config.min_games_for_import == 50
        assert config.owc_host == "mac-studio"
        assert config.ssh_timeout == 60
        assert config.rsync_timeout == 600

    def test_from_env_defaults(self):
        """Test from_env with default values."""
        with patch.dict("os.environ", {}, clear=True):
            config = OWCImportConfig.from_env()
            assert config.enabled is True
            assert config.check_interval_seconds == 3600
            assert config.min_games_for_import == 50

    def test_from_env_overrides(self):
        """Test from_env with environment overrides."""
        env = {
            "RINGRIFT_OWC_IMPORT_ENABLED": "false",
            "RINGRIFT_OWC_IMPORT_INTERVAL": "1800",
            "RINGRIFT_OWC_IMPORT_MIN_GAMES": "100",
        }
        with patch.dict("os.environ", env, clear=True):
            config = OWCImportConfig.from_env()
            assert config.enabled is False
            assert config.check_interval_seconds == 1800
            assert config.min_games_for_import == 100


# =============================================================================
# OWCDatabaseInfo Tests
# =============================================================================


class TestOWCDatabaseInfo:
    """Tests for OWCDatabaseInfo dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        info = OWCDatabaseInfo(path="test/db.db")
        assert info.path == "test/db.db"
        assert info.configs == {}
        assert info.synced is False

    def test_with_configs(self):
        """Test with config data."""
        info = OWCDatabaseInfo(
            path="test/db.db",
            configs={"hex8_2p": 500, "square8_2p": 300},
        )
        assert info.configs["hex8_2p"] == 500
        assert info.configs["square8_2p"] == 300


# =============================================================================
# ImportStats Tests
# =============================================================================


class TestImportStats:
    """Tests for ImportStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = ImportStats()
        assert stats.cycle_start == 0.0
        assert stats.cycle_end == 0.0
        assert stats.databases_scanned == 0
        assert stats.databases_synced == 0
        assert stats.games_imported == 0
        assert stats.configs_updated == []
        assert stats.errors == []

    def test_duration_calculation(self):
        """Test duration property."""
        stats = ImportStats(
            cycle_start=1000.0,
            cycle_end=1005.0,
        )
        assert stats.duration_seconds == 5.0

    def test_with_data(self):
        """Test with populated data."""
        stats = ImportStats(
            cycle_start=1000.0,
            cycle_end=1010.0,
            databases_scanned=5,
            databases_synced=2,
            games_imported=1500,
            configs_updated=["hex8_2p", "square8_2p"],
        )
        assert stats.databases_scanned == 5
        assert stats.databases_synced == 2
        assert stats.games_imported == 1500
        assert "hex8_2p" in stats.configs_updated


# =============================================================================
# OWCImportDaemon Init Tests
# =============================================================================


class TestOWCImportDaemonInit:
    """Tests for daemon initialization."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    def test_default_init(self):
        """Test default initialization."""
        daemon = OWCImportDaemon()
        assert daemon._last_import == {}
        assert daemon._import_history == []
        assert daemon._total_games_imported == 0
        assert daemon._owc_available is True

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = OWCImportConfig(
            min_games_for_import=200,
            owc_host="custom-host",
        )
        daemon = OWCImportDaemon(config=config)
        assert daemon.config.min_games_for_import == 200
        assert daemon.config.owc_host == "custom-host"

    def test_daemon_name(self):
        """Test daemon name."""
        daemon = OWCImportDaemon()
        assert daemon._get_daemon_name() == "OWCImport"


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    def test_get_instance(self):
        """Test get_instance returns same instance."""
        d1 = OWCImportDaemon.get_instance()
        d2 = OWCImportDaemon.get_instance()
        assert d1 is d2

    def test_reset_instance(self):
        """Test reset_instance clears singleton."""
        d1 = OWCImportDaemon.get_instance()
        OWCImportDaemon.reset_instance()
        d2 = OWCImportDaemon.get_instance()
        assert d1 is not d2

    def test_get_owc_import_daemon(self):
        """Test module-level getter."""
        daemon = get_owc_import_daemon()
        assert isinstance(daemon, OWCImportDaemon)


# =============================================================================
# SSH Command Tests
# =============================================================================


class TestSSHCommands:
    """Tests for SSH command execution."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    @pytest.mark.asyncio
    async def test_run_ssh_command_success(self):
        """Test successful SSH command."""
        daemon = OWCImportDaemon()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success, output = await daemon._run_ssh_command("echo test")

        assert success is True
        assert output == "output"

    @pytest.mark.asyncio
    async def test_run_ssh_command_failure(self):
        """Test failed SSH command."""
        daemon = OWCImportDaemon()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error message"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success, output = await daemon._run_ssh_command("bad command")

        assert success is False
        assert output == "error message"

    @pytest.mark.asyncio
    async def test_run_ssh_command_timeout(self):
        """Test SSH command timeout."""
        daemon = OWCImportDaemon(config=OWCImportConfig(ssh_timeout=1))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            success, output = await daemon._run_ssh_command("sleep 100")

        assert success is False
        assert "timed out" in output

    @pytest.mark.asyncio
    async def test_check_owc_available_success(self):
        """Test OWC availability check - available."""
        daemon = OWCImportDaemon()

        with patch.object(
            daemon, "_run_ssh_command",
            return_value=(True, "/Volumes/RingRift-Data")
        ):
            available = await daemon._check_owc_available()

        assert available is True

    @pytest.mark.asyncio
    async def test_check_owc_available_not_mounted(self):
        """Test OWC availability check - not mounted."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_run_ssh_command", return_value=(False, "")):
            available = await daemon._check_owc_available()

        assert available is False


# =============================================================================
# Database Scanning Tests
# =============================================================================


class TestDatabaseScanning:
    """Tests for OWC database scanning."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    @pytest.mark.asyncio
    async def test_scan_owc_database_success(self):
        """Test successful database scan."""
        daemon = OWCImportDaemon()

        ssh_output = "hex8_2p|500\nsquare8_2p|300\nhex8_4p|200"

        with patch.object(daemon, "_run_ssh_command", return_value=(True, ssh_output)):
            info = await daemon._scan_owc_database("test/db.db")

        assert info is not None
        assert info.path == "test/db.db"
        assert info.configs["hex8_2p"] == 500
        assert info.configs["square8_2p"] == 300
        assert info.configs["hex8_4p"] == 200

    @pytest.mark.asyncio
    async def test_scan_owc_database_failure(self):
        """Test database scan failure."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_run_ssh_command", return_value=(False, "error")):
            info = await daemon._scan_owc_database("test/db.db")

        assert info is None

    @pytest.mark.asyncio
    async def test_scan_owc_database_empty(self):
        """Test database scan with no games."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_run_ssh_command", return_value=(True, "")):
            info = await daemon._scan_owc_database("test/db.db")

        assert info is None  # No configs found


# =============================================================================
# Database Sync Tests
# =============================================================================


class TestDatabaseSync:
    """Tests for database synchronization."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    @pytest.mark.asyncio
    async def test_sync_database_success(self):
        """Test successful database sync."""
        daemon = OWCImportDaemon()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_size = 10 * 1024 * 1024  # 10MB

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(Path, "mkdir"):
                with patch.object(
                    daemon.config.staging_dir.__class__,
                    "__truediv__",
                    return_value=mock_path,
                ):
                    result = await daemon._sync_database("test/db.db")

        # Result is the mock path
        assert result is not None

    @pytest.mark.asyncio
    async def test_sync_database_timeout(self):
        """Test database sync timeout."""
        daemon = OWCImportDaemon(config=OWCImportConfig(rsync_timeout=1))

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(Path, "mkdir"):
                result = await daemon._sync_database("test/db.db")

        assert result is None


# =============================================================================
# Local Game Count Tests
# =============================================================================


class TestLocalGameCount:
    """Tests for local game count retrieval."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    def test_get_local_game_count_invalid_config(self):
        """Test with invalid config key."""
        daemon = OWCImportDaemon()

        count = daemon._get_local_game_count("invalid")

        assert count == 0

    def test_get_local_game_count_no_db(self):
        """Test when database doesn't exist."""
        daemon = OWCImportDaemon()

        with patch.object(Path, "exists", return_value=False):
            count = daemon._get_local_game_count("hex8_2p")

        assert count == 0


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    def test_emit_new_games_available(self):
        """Test NEW_GAMES_AVAILABLE event emission."""
        daemon = OWCImportDaemon()

        with patch("app.distributed.data_events.emit_data_event") as mock_emit:
            daemon._emit_new_games_available("hex8_2p", 500, "owc:test.db")

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[1]["config_key"] == "hex8_2p"
            assert call_args[1]["new_games"] == 500

    def test_emit_data_sync_completed(self):
        """Test DATA_SYNC_COMPLETED event emission."""
        daemon = OWCImportDaemon()

        with patch("app.distributed.data_events.emit_data_event") as mock_emit:
            daemon._emit_data_sync_completed(["hex8_2p", "square8_2p"], 800)

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[1]["sync_type"] == "owc_import"
            assert call_args[1]["games_imported"] == 800


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    def test_health_check_not_running(self):
        """Test health check when not running."""
        daemon = OWCImportDaemon()
        daemon._running = False

        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message

    def test_health_check_owc_unavailable(self):
        """Test health check when OWC unavailable."""
        daemon = OWCImportDaemon()
        daemon._running = True
        daemon._owc_available = False

        result = daemon.health_check()

        assert result.healthy is True  # Still healthy
        assert "not available" in result.message

    def test_health_check_healthy(self):
        """Test healthy state."""
        daemon = OWCImportDaemon()
        daemon._running = True
        daemon._owc_available = True
        daemon._total_games_imported = 1000

        result = daemon.health_check()

        assert result.healthy is True
        assert "1000 games imported" in result.message


# =============================================================================
# Status Tests
# =============================================================================


class TestStatus:
    """Tests for status retrieval."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    def test_get_status(self):
        """Test get_status returns expected fields."""
        daemon = OWCImportDaemon()
        daemon._owc_available = True
        daemon._total_games_imported = 500

        # Add some import history
        daemon._import_history.append(
            ImportStats(
                cycle_start=1000.0,
                cycle_end=1005.0,
                games_imported=500,
                configs_updated=["hex8_2p"],
            )
        )

        status = daemon.get_status()

        assert "owc_host" in status
        assert status["owc_available"] is True
        assert status["total_games_imported"] == 500
        assert "recent_imports" in status
        assert len(status["recent_imports"]) == 1


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Tests for main run cycle."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_owc_import_daemon()

    @pytest.mark.asyncio
    async def test_run_cycle_owc_unavailable(self):
        """Test run cycle when OWC unavailable."""
        daemon = OWCImportDaemon()
        daemon._owc_available = True

        with patch.object(daemon, "_check_owc_available", return_value=False):
            await daemon._run_cycle()

        assert daemon._owc_available is False

    @pytest.mark.asyncio
    async def test_run_cycle_all_configs_served(self):
        """Test run cycle when all configs have data."""
        daemon = OWCImportDaemon()

        with patch.object(daemon, "_check_owc_available", return_value=True):
            with patch.object(daemon, "_get_underserved_configs", return_value=[]):
                await daemon._run_cycle()

        # Should return early, no imports
        assert len(daemon._import_history) == 0
