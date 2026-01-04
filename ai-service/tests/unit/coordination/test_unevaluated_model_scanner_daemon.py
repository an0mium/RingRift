"""Tests for UnevaluatedModelScannerDaemon.

Sprint 13 Session 4 (January 3, 2026): Part of model evaluation automation.

Comprehensive test suite covering:
- UnevaluatedModelScannerConfig configuration
- UnevaluatedModel dataclass
- UnevaluatedModelScannerDaemon functionality
- Priority computation
- Factory functions and singleton pattern
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.unevaluated_model_scanner_daemon import (
    CURRICULUM_WEIGHTS,
    MODEL_SCAN_PATHS,
    PRIORITY_BOOST_4_PLAYER,
    PRIORITY_BOOST_CANONICAL,
    PRIORITY_BOOST_DIVERSITY,
    PRIORITY_BOOST_RECENT,
    PRIORITY_BOOST_UNDERSERVED,
    RECENT_THRESHOLD_SECONDS,
    UnevaluatedModel,
    UnevaluatedModelScannerConfig,
    UnevaluatedModelScannerDaemon,
    get_unevaluated_model_scanner_daemon,
    reset_unevaluated_model_scanner_daemon,
)


# =============================================================================
# UnevaluatedModelScannerConfig Tests
# =============================================================================


class TestUnevaluatedModelScannerConfig:
    """Tests for UnevaluatedModelScannerConfig dataclass."""

    def test_defaults(self):
        """Default configuration values are correct."""
        config = UnevaluatedModelScannerConfig()
        assert config.scan_interval_seconds == 3600
        assert config.enabled is True
        assert config.max_queue_per_cycle == 20
        assert config.base_priority == 50

    def test_scan_paths_default(self):
        """Default scan paths include expected directories."""
        config = UnevaluatedModelScannerConfig()
        assert "models" in config.scan_paths
        assert "models/owc_imports" in config.scan_paths

    def test_custom_values(self):
        """Can override configuration values."""
        config = UnevaluatedModelScannerConfig(
            scan_interval_seconds=1800,
            enabled=False,
            max_queue_per_cycle=10,
            base_priority=75,
        )
        assert config.scan_interval_seconds == 1800
        assert config.enabled is False
        assert config.max_queue_per_cycle == 10
        assert config.base_priority == 75

    def test_from_env_defaults(self):
        """from_env() uses defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = UnevaluatedModelScannerConfig.from_env()
            assert config.enabled is True
            assert config.scan_interval_seconds == 3600
            assert config.max_queue_per_cycle == 20

    def test_from_env_custom(self):
        """from_env() reads environment variables."""
        env_vars = {
            "RINGRIFT_SCANNER_ENABLED": "false",
            "RINGRIFT_SCANNER_INTERVAL": "7200",
            "RINGRIFT_SCANNER_MAX_QUEUE_PER_CYCLE": "50",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = UnevaluatedModelScannerConfig.from_env()
            assert config.enabled is False
            assert config.scan_interval_seconds == 7200
            assert config.max_queue_per_cycle == 50


# =============================================================================
# UnevaluatedModel Tests
# =============================================================================


class TestUnevaluatedModel:
    """Tests for UnevaluatedModel dataclass."""

    def test_basic_creation(self):
        """Can create UnevaluatedModel with required fields."""
        model = UnevaluatedModel(
            path="/path/to/model.pth",
            source="local",
            board_type="hex8",
            num_players=2,
            architecture_version="v2",
        )
        assert model.path == "/path/to/model.pth"
        assert model.source == "local"
        assert model.board_type == "hex8"
        assert model.num_players == 2
        assert model.architecture_version == "v2"

    def test_default_values(self):
        """Default values are set correctly."""
        model = UnevaluatedModel(
            path="/path/to/model.pth",
            source="local",
            board_type="hex8",
            num_players=2,
            architecture_version=None,
        )
        assert model.priority == 50
        assert model.file_size == 0
        assert model.is_canonical is False
        assert model.discovered_at > 0

    def test_config_key_property(self):
        """config_key property returns correct key."""
        model = UnevaluatedModel(
            path="/path/to/model.pth",
            source="local",
            board_type="hex8",
            num_players=4,
            architecture_version="v2",
        )
        assert model.config_key == "hex8_4p"

    def test_config_key_none_when_missing_info(self):
        """config_key returns None when board_type or num_players is None."""
        model = UnevaluatedModel(
            path="/path/to/model.pth",
            source="local",
            board_type=None,
            num_players=2,
            architecture_version=None,
        )
        assert model.config_key is None

        model2 = UnevaluatedModel(
            path="/path/to/model.pth",
            source="local",
            board_type="hex8",
            num_players=None,
            architecture_version=None,
        )
        assert model2.config_key is None

    def test_canonical_flag(self):
        """is_canonical flag can be set."""
        model = UnevaluatedModel(
            path="models/canonical_hex8_2p.pth",
            source="local",
            board_type="hex8",
            num_players=2,
            architecture_version="v2",
            is_canonical=True,
        )
        assert model.is_canonical is True


# =============================================================================
# UnevaluatedModelScannerDaemon Tests
# =============================================================================


class TestUnevaluatedModelScannerDaemon:
    """Tests for UnevaluatedModelScannerDaemon."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnevaluatedModelScannerConfig(
            scan_interval_seconds=60,
            enabled=True,
            max_queue_per_cycle=5,
        )

    @pytest.fixture
    def daemon(self, config):
        """Create daemon instance for testing."""
        reset_unevaluated_model_scanner_daemon()
        daemon = UnevaluatedModelScannerDaemon(config)
        return daemon

    def test_init_with_config(self, daemon, config):
        """Daemon initializes with config."""
        assert daemon.config == config
        assert daemon.name is not None
        assert len(daemon.name) > 0

    def test_daemon_inherits_handler_base(self, daemon):
        """Daemon inherits from HandlerBase."""
        from app.coordination.handler_base import HandlerBase
        assert isinstance(daemon, HandlerBase)

    def test_cycle_interval_from_config(self, config):
        """Daemon uses scan_interval_seconds from config."""
        daemon = UnevaluatedModelScannerDaemon(config)
        assert daemon._cycle_interval == config.scan_interval_seconds

    @pytest.mark.asyncio
    async def test_run_cycle_disabled(self):
        """run_cycle does nothing when disabled."""
        reset_unevaluated_model_scanner_daemon()
        config = UnevaluatedModelScannerConfig(enabled=False)
        daemon = UnevaluatedModelScannerDaemon(config)
        # Should complete without error
        await daemon._run_cycle()

    def test_health_check_returns_result(self, daemon):
        """health_check returns HealthCheckResult."""
        result = daemon.health_check()
        assert hasattr(result, "healthy")

    def test_has_config(self, daemon):
        """Daemon has config attribute."""
        assert daemon.config is not None
        assert daemon.config.enabled is True

    def test_compute_priority_base(self, daemon):
        """compute_priority returns base priority for standard model."""
        model = UnevaluatedModel(
            path="/path/to/model.pth",
            source="local",
            board_type="hex8",
            num_players=2,
            architecture_version="v2",
        )
        priority = daemon._compute_priority(model)
        assert priority >= daemon.config.base_priority

    def test_compute_priority_4player_boost(self, daemon):
        """compute_priority gives boost to 4-player models."""
        model_2p = UnevaluatedModel(
            path="/path/to/model.pth",
            source="local",
            board_type="hex8",
            num_players=2,
            architecture_version="v2",
        )
        model_4p = UnevaluatedModel(
            path="/path/to/model.pth",
            source="local",
            board_type="hex8",
            num_players=4,
            architecture_version="v2",
        )
        priority_2p = daemon._compute_priority(model_2p)
        priority_4p = daemon._compute_priority(model_4p)
        assert priority_4p > priority_2p

    def test_compute_priority_canonical_boost(self, daemon):
        """compute_priority gives boost to canonical models."""
        model = UnevaluatedModel(
            path="models/canonical_hex8_2p.pth",
            source="local",
            board_type="hex8",
            num_players=2,
            architecture_version="v2",
            is_canonical=True,
        )
        priority = daemon._compute_priority(model)
        assert priority >= daemon.config.base_priority + PRIORITY_BOOST_CANONICAL

    def test_parse_model_file_canonical(self, daemon):
        """Can parse canonical model from path."""
        # Test the _parse_model_file method exists
        assert hasattr(daemon, '_parse_model_file')

    def test_extract_config_from_name(self, daemon):
        """Can extract config from model filename."""
        # Test the _extract_config_from_name method
        board_type, num_players = daemon._extract_config_from_name("canonical_hex8_2p.pth")
        assert board_type == "hex8"
        assert num_players == 2


# =============================================================================
# Priority Computation Tests
# =============================================================================


class TestPriorityComputation:
    """Tests for priority computation logic."""

    def test_curriculum_weights_exist(self):
        """Curriculum weights are defined for all configs."""
        assert "hex8_2p" in CURRICULUM_WEIGHTS
        assert "hex8_4p" in CURRICULUM_WEIGHTS
        assert "square8_2p" in CURRICULUM_WEIGHTS
        assert "hexagonal_4p" in CURRICULUM_WEIGHTS

    def test_4player_weight_higher(self):
        """4-player configs have higher curriculum weights."""
        assert CURRICULUM_WEIGHTS["hex8_4p"] > CURRICULUM_WEIGHTS["hex8_2p"]
        assert CURRICULUM_WEIGHTS["square8_4p"] > CURRICULUM_WEIGHTS["square8_2p"]

    def test_priority_boosts_defined(self):
        """Priority boost constants are defined."""
        assert PRIORITY_BOOST_4_PLAYER == 30
        assert PRIORITY_BOOST_UNDERSERVED == 50
        assert PRIORITY_BOOST_CANONICAL == 20
        assert PRIORITY_BOOST_RECENT == 30
        assert PRIORITY_BOOST_DIVERSITY == 10

    def test_recent_threshold(self):
        """Recent threshold is 24 hours."""
        assert RECENT_THRESHOLD_SECONDS == 86400


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory and singleton functions."""

    def test_get_unevaluated_model_scanner_daemon_returns_daemon(self):
        """get_unevaluated_model_scanner_daemon returns a daemon."""
        reset_unevaluated_model_scanner_daemon()
        daemon = get_unevaluated_model_scanner_daemon()
        assert isinstance(daemon, UnevaluatedModelScannerDaemon)

    def test_get_unevaluated_model_scanner_daemon_singleton(self):
        """get_unevaluated_model_scanner_daemon returns same instance."""
        reset_unevaluated_model_scanner_daemon()
        daemon1 = get_unevaluated_model_scanner_daemon()
        daemon2 = get_unevaluated_model_scanner_daemon()
        assert daemon1 is daemon2

    def test_reset_unevaluated_model_scanner_daemon(self):
        """reset_unevaluated_model_scanner_daemon clears singleton."""
        reset_unevaluated_model_scanner_daemon()
        daemon1 = get_unevaluated_model_scanner_daemon()
        reset_unevaluated_model_scanner_daemon()
        daemon2 = get_unevaluated_model_scanner_daemon()
        assert daemon1 is not daemon2


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_model_scan_paths_not_empty(self):
        """MODEL_SCAN_PATHS contains expected paths."""
        assert len(MODEL_SCAN_PATHS) > 0
        assert "models" in MODEL_SCAN_PATHS

    def test_curriculum_weights_coverage(self):
        """Curriculum weights cover all 12 canonical configs."""
        expected_configs = [
            "hex8_2p", "hex8_3p", "hex8_4p",
            "square8_2p", "square8_3p", "square8_4p",
            "square19_2p", "square19_3p", "square19_4p",
            "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
        ]
        for config in expected_configs:
            assert config in CURRICULUM_WEIGHTS, f"Missing weight for {config}"
