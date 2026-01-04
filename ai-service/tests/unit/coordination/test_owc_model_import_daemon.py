"""Tests for OWCModelImportDaemon.

Sprint 13 Session 4 (January 3, 2026): Part of model evaluation automation.

Comprehensive test suite covering:
- OWCModelImportConfig configuration
- OWCModelInfo dataclass
- OWCModelImportDaemon functionality
- Factory functions and singleton pattern
"""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.owc_model_import_daemon import (
    MIN_MODEL_SIZE_BYTES,
    OWC_BASE_PATH,
    OWC_HOST,
    OWC_MODEL_PATHS,
    OWC_SSH_KEY,
    OWC_USER,
    OWCModelImportConfig,
    OWCModelImportDaemon,
    OWCModelInfo,
    _is_running_on_owc_host,
    get_owc_model_import_daemon,
    reset_owc_model_import_daemon,
)


# =============================================================================
# OWCModelImportConfig Tests
# =============================================================================


class TestOWCModelImportConfig:
    """Tests for OWCModelImportConfig dataclass."""

    def test_defaults(self):
        """Default configuration values are correct."""
        config = OWCModelImportConfig()
        assert config.check_interval_seconds == 7200
        assert config.enabled is True
        assert config.max_models_per_cycle == 10
        assert config.min_model_size_bytes == MIN_MODEL_SIZE_BYTES
        assert config.import_dir == Path("models/owc_imports")

    def test_owc_connection_defaults(self):
        """OWC connection defaults from environment."""
        config = OWCModelImportConfig()
        assert config.owc_host == OWC_HOST
        assert config.owc_user == OWC_USER
        assert config.owc_base_path == OWC_BASE_PATH
        assert config.owc_ssh_key == OWC_SSH_KEY

    def test_custom_values(self):
        """Can override configuration values."""
        config = OWCModelImportConfig(
            check_interval_seconds=3600,
            max_models_per_cycle=20,
            min_model_size_bytes=500_000,
            import_dir=Path("/tmp/models"),
        )
        assert config.check_interval_seconds == 3600
        assert config.max_models_per_cycle == 20
        assert config.min_model_size_bytes == 500_000
        assert config.import_dir == Path("/tmp/models")

    def test_from_env_defaults(self):
        """from_env() uses defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = OWCModelImportConfig.from_env()
            assert config.enabled is True
            assert config.check_interval_seconds == 7200
            assert config.max_models_per_cycle == 10

    def test_from_env_custom(self):
        """from_env() reads environment variables."""
        env_vars = {
            "RINGRIFT_OWC_MODEL_IMPORT_ENABLED": "false",
            "RINGRIFT_OWC_MODEL_IMPORT_INTERVAL": "3600",
            "RINGRIFT_OWC_MODEL_IMPORT_MAX_PER_CYCLE": "5",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = OWCModelImportConfig.from_env()
            assert config.enabled is False
            assert config.check_interval_seconds == 3600
            assert config.max_models_per_cycle == 5

    def test_timeout_defaults(self):
        """SSH and rsync timeout defaults are correct."""
        config = OWCModelImportConfig()
        assert config.ssh_timeout == 60
        assert config.rsync_timeout == 600


# =============================================================================
# OWCModelInfo Tests
# =============================================================================


class TestOWCModelInfo:
    """Tests for OWCModelInfo dataclass."""

    def test_basic_creation(self):
        """Can create OWCModelInfo with all fields."""
        info = OWCModelInfo(
            path="models/archived/hex8_2p_v2.pth",
            file_name="hex8_2p_v2.pth",
            board_type="hex8",
            num_players=2,
            architecture_version="v2",
            file_size=35_000_000,
            has_elo=False,
        )
        assert info.path == "models/archived/hex8_2p_v2.pth"
        assert info.file_name == "hex8_2p_v2.pth"
        assert info.board_type == "hex8"
        assert info.num_players == 2
        assert info.architecture_version == "v2"
        assert info.file_size == 35_000_000
        assert info.has_elo is False

    def test_optional_fields(self):
        """Optional fields can be None."""
        info = OWCModelInfo(
            path="models/unknown.pth",
            file_name="unknown.pth",
            board_type=None,
            num_players=None,
            architecture_version=None,
            file_size=10_000_000,
            has_elo=False,
        )
        assert info.board_type is None
        assert info.num_players is None
        assert info.architecture_version is None


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestIsRunningOnOwcHost:
    """Tests for _is_running_on_owc_host helper."""

    def test_localhost(self):
        """Localhost is recognized as OWC host."""
        assert _is_running_on_owc_host("localhost") is True
        assert _is_running_on_owc_host("127.0.0.1") is True

    def test_exact_match(self):
        """Exact hostname match is recognized."""
        with patch.object(socket, "gethostname", return_value="mac-studio"):
            assert _is_running_on_owc_host("mac-studio") is True

    def test_with_local_suffix(self):
        """Hostname with .local suffix is recognized."""
        with patch.object(socket, "gethostname", return_value="mac-studio.local"):
            assert _is_running_on_owc_host("mac-studio") is True

    def test_different_host(self):
        """Different hostname is not matched."""
        with patch.object(socket, "gethostname", return_value="other-host"):
            assert _is_running_on_owc_host("mac-studio") is False


# =============================================================================
# OWCModelImportDaemon Tests
# =============================================================================


class TestOWCModelImportDaemon:
    """Tests for OWCModelImportDaemon."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OWCModelImportConfig(
            check_interval_seconds=60,
            enabled=True,
            max_models_per_cycle=5,
            import_dir=Path("/tmp/test_owc_imports"),
        )

    @pytest.fixture
    def daemon(self, config):
        """Create daemon instance for testing."""
        reset_owc_model_import_daemon()  # Ensure clean state
        daemon = OWCModelImportDaemon(config)
        return daemon

    def test_init_with_config(self, daemon, config):
        """Daemon initializes with config."""
        # config is accessed via _config (internal)
        assert hasattr(daemon, '_config')

    def test_daemon_inherits_handler_base(self, daemon):
        """Daemon inherits from HandlerBase."""
        from app.coordination.handler_base import HandlerBase
        assert isinstance(daemon, HandlerBase)

    def test_cycle_interval_from_config(self, config):
        """Daemon uses check_interval_seconds from config."""
        daemon = OWCModelImportDaemon(config)
        # The cycle interval should match the config
        assert daemon._cycle_interval == config.check_interval_seconds

    def test_health_check_returns_result(self, daemon):
        """health_check returns HealthCheckResult."""
        result = daemon.health_check()
        assert hasattr(result, "healthy")

    def test_health_check_is_healthy(self, daemon):
        """health_check reports healthy state."""
        result = daemon.health_check()
        assert result is not None
        assert result.healthy is True

    def test_has_required_methods(self, daemon):
        """Daemon has required methods."""
        # Verify key methods exist
        assert hasattr(daemon, '_run_cycle')
        assert hasattr(daemon, 'health_check')
        assert hasattr(daemon, '_discover_owc_models')

    def test_daemon_name(self, daemon):
        """Daemon has a name."""
        assert daemon.name is not None
        assert len(daemon.name) > 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory and singleton functions."""

    def test_get_owc_model_import_daemon_returns_daemon(self):
        """get_owc_model_import_daemon returns a daemon."""
        reset_owc_model_import_daemon()
        daemon = get_owc_model_import_daemon()
        assert isinstance(daemon, OWCModelImportDaemon)

    def test_get_owc_model_import_daemon_singleton(self):
        """get_owc_model_import_daemon returns same instance."""
        reset_owc_model_import_daemon()
        daemon1 = get_owc_model_import_daemon()
        daemon2 = get_owc_model_import_daemon()
        assert daemon1 is daemon2

    def test_reset_owc_model_import_daemon(self):
        """reset_owc_model_import_daemon clears singleton."""
        reset_owc_model_import_daemon()
        daemon1 = get_owc_model_import_daemon()
        reset_owc_model_import_daemon()
        daemon2 = get_owc_model_import_daemon()
        assert daemon1 is not daemon2


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_model_paths_not_empty(self):
        """OWC_MODEL_PATHS contains expected paths."""
        assert len(OWC_MODEL_PATHS) > 0
        assert "models/archived" in OWC_MODEL_PATHS

    def test_min_model_size(self):
        """MIN_MODEL_SIZE_BYTES is reasonable."""
        assert MIN_MODEL_SIZE_BYTES == 1_000_000  # 1MB
