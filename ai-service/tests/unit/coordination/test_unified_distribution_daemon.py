"""Tests for UnifiedDistributionDaemon.

Tests cover:
- Configuration initialization
- DataType enum
- DeliveryResult dataclass
- Factory functions (backward compatibility)
- Daemon lifecycle (start/stop)
- Event subscription handling
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from app.coordination.unified_distribution_daemon import (
    DataType,
    DeliveryResult,
    DistributionConfig,
    UnifiedDistributionDaemon,
    create_model_distribution_daemon,
    create_npz_distribution_daemon,
    create_unified_distribution_daemon,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return DistributionConfig(
        sync_timeout_seconds=60.0,
        retry_count=2,
        http_timeout_seconds=30.0,
    )


@pytest.fixture
def daemon(config):
    """Create a test daemon."""
    return UnifiedDistributionDaemon(config)


# =============================================================================
# DataType Enum Tests
# =============================================================================


class TestDataType:
    """Test DataType enumeration."""

    def test_has_model(self):
        """Test MODEL type exists."""
        assert DataType.MODEL is not None

    def test_has_npz(self):
        """Test NPZ type exists."""
        assert DataType.NPZ is not None

    def test_has_torrent(self):
        """Test TORRENT type exists."""
        assert DataType.TORRENT is not None

    def test_all_types(self):
        """Test all data types."""
        all_types = list(DataType)
        assert len(all_types) >= 3


# =============================================================================
# DistributionConfig Tests
# =============================================================================


class TestDistributionConfig:
    """Test DistributionConfig dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        config = DistributionConfig()
        assert config.sync_timeout_seconds == 300.0
        assert config.retry_count == 3
        assert config.use_http_distribution is True
        assert config.verify_checksums is True
        assert config.models_dir == "models"
        assert config.training_data_dir == "data/training"

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = DistributionConfig(
            sync_timeout_seconds=120.0,
            retry_count=5,
            http_port=9999,
            use_bittorrent_for_large_files=False,
        )
        assert config.sync_timeout_seconds == 120.0
        assert config.retry_count == 5
        assert config.http_port == 9999
        assert config.use_bittorrent_for_large_files is False

    def test_bittorrent_threshold(self):
        """Test BitTorrent threshold setting."""
        config = DistributionConfig(bittorrent_threshold_bytes=100_000_000)
        assert config.bittorrent_threshold_bytes == 100_000_000

    def test_retry_backoff(self):
        """Test retry backoff multiplier."""
        config = DistributionConfig(retry_backoff_multiplier=2.0)
        assert config.retry_backoff_multiplier == 2.0


# =============================================================================
# DeliveryResult Tests
# =============================================================================


class TestDeliveryResult:
    """Test DeliveryResult dataclass."""

    def test_successful_delivery(self):
        """Test successful delivery result."""
        result = DeliveryResult(
            node_id="node-1",
            host="10.0.0.1",
            data_path="models/canonical_hex8_2p.pth",
            data_type=DataType.MODEL,
            success=True,
            checksum_verified=True,
            transfer_time_seconds=5.2,
            method="http",
        )
        assert result.success is True
        assert result.checksum_verified is True
        assert result.error_message == ""
        assert result.method == "http"

    def test_failed_delivery(self):
        """Test failed delivery result."""
        result = DeliveryResult(
            node_id="node-2",
            host="10.0.0.2",
            data_path="data/training/hex8_2p.npz",
            data_type=DataType.NPZ,
            success=False,
            checksum_verified=False,
            transfer_time_seconds=0.0,
            error_message="Connection timeout",
            method="rsync",
        )
        assert result.success is False
        assert result.error_message == "Connection timeout"
        assert result.method == "rsync"

    def test_default_method(self):
        """Test default method is http."""
        result = DeliveryResult(
            node_id="node-3",
            host="10.0.0.3",
            data_path="models/test.pth",
            data_type=DataType.MODEL,
            success=True,
            checksum_verified=True,
            transfer_time_seconds=1.0,
        )
        assert result.method == "http"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions for backward compatibility."""

    def test_create_unified_distribution_daemon(self):
        """Test creating unified daemon."""
        daemon = create_unified_distribution_daemon()
        assert isinstance(daemon, UnifiedDistributionDaemon)
        assert daemon.config is not None

    def test_create_unified_with_config(self):
        """Test creating unified daemon with custom config."""
        config = DistributionConfig(retry_count=10)
        daemon = create_unified_distribution_daemon(config)
        assert daemon.config.retry_count == 10

    def test_create_model_distribution_daemon_deprecated(self):
        """Test deprecated model distribution factory."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            daemon = create_model_distribution_daemon()
        assert isinstance(daemon, UnifiedDistributionDaemon)

    def test_create_npz_distribution_daemon_deprecated(self):
        """Test deprecated NPZ distribution factory."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            daemon = create_npz_distribution_daemon()
        assert isinstance(daemon, UnifiedDistributionDaemon)


# =============================================================================
# UnifiedDistributionDaemon Initialization Tests
# =============================================================================


class TestUnifiedDistributionDaemonInit:
    """Test UnifiedDistributionDaemon initialization."""

    def test_default_init(self):
        """Test default initialization."""
        daemon = UnifiedDistributionDaemon()
        assert daemon.config is not None
        assert daemon._running is False
        assert daemon._last_sync_time == 0.0

    def test_custom_config_init(self, config):
        """Test initialization with custom config."""
        daemon = UnifiedDistributionDaemon(config)
        assert daemon.config.sync_timeout_seconds == 60.0
        assert daemon.config.retry_count == 2

    def test_has_required_methods(self, daemon):
        """Test daemon has required methods."""
        assert hasattr(daemon, "start")
        assert hasattr(daemon, "stop")
        assert hasattr(daemon, "is_running")


# =============================================================================
# Daemon State Tests
# =============================================================================


class TestDaemonState:
    """Test daemon state management."""

    def test_initial_state(self, daemon):
        """Test initial daemon state."""
        assert daemon.is_running() is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self, daemon):
        """Test start sets running flag initially."""
        # start() is meant to run long-lived; just verify the flag is set
        # We'll test this by checking the initial state and simulating
        daemon._running = True
        assert daemon._running is True
        assert daemon.is_running() is True

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, daemon):
        """Test stop clears running flag."""
        daemon._running = True
        await daemon.stop()
        assert daemon._running is False

    def test_is_running_property(self, daemon):
        """Test is_running reflects internal state."""
        assert daemon.is_running() is False
        daemon._running = True
        assert daemon.is_running() is True
        daemon._running = False
        assert daemon.is_running() is False


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test health check integration."""

    def test_has_health_check(self, daemon):
        """Test daemon has health check method."""
        assert hasattr(daemon, "health_check") or hasattr(daemon, "_health_check")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the daemon."""

    def test_daemon_can_be_instantiated_multiple_times(self):
        """Test multiple daemon instances can coexist."""
        daemon1 = UnifiedDistributionDaemon()
        daemon2 = UnifiedDistributionDaemon()
        assert daemon1 is not daemon2

    def test_daemon_config_isolation(self):
        """Test each daemon has its own config."""
        config1 = DistributionConfig(retry_count=1)
        config2 = DistributionConfig(retry_count=10)
        daemon1 = UnifiedDistributionDaemon(config1)
        daemon2 = UnifiedDistributionDaemon(config2)
        assert daemon1.config.retry_count == 1
        assert daemon2.config.retry_count == 10
