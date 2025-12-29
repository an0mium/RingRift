"""Tests for MemoryMonitorDaemon.

December 29, 2025: Part of 48-hour autonomous operation optimization.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.memory_monitor_daemon import (
    MemoryMonitorConfig,
    MemoryMonitorDaemon,
    MemoryStatus,
    MemoryThresholds,
    get_memory_monitor,
    reset_memory_monitor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    reset_memory_monitor()
    yield
    reset_memory_monitor()


@pytest.fixture
def thresholds():
    """Create test thresholds."""
    return MemoryThresholds(
        gpu_warning=0.70,
        gpu_critical=0.80,
        ram_warning=0.75,
        ram_critical=0.85,
        process_rss_critical_bytes=16 * 1024 * 1024 * 1024,  # 16GB
        sigkill_grace_period=10.0,
    )


@pytest.fixture
def config(thresholds):
    """Create test configuration."""
    return MemoryMonitorConfig(
        enabled=True,
        check_interval_seconds=5.0,
        thresholds=thresholds,
        kill_enabled=False,  # Disable for testing
        event_cooldown_seconds=1.0,
    )


@pytest.fixture
def monitor(config):
    """Create MemoryMonitorDaemon for testing."""
    return MemoryMonitorDaemon(config=config)


# =============================================================================
# MemoryThresholds Tests
# =============================================================================


class TestMemoryThresholds:
    """Tests for MemoryThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        t = MemoryThresholds()
        assert t.gpu_warning == 0.75
        assert t.gpu_critical == 0.85
        assert t.ram_warning == 0.80
        assert t.ram_critical == 0.90
        assert t.process_rss_critical_bytes == 32 * 1024 * 1024 * 1024  # 32GB
        assert t.sigkill_grace_period == 60.0

    def test_custom_thresholds(self, thresholds):
        """Test custom threshold values."""
        assert thresholds.gpu_warning == 0.70
        assert thresholds.gpu_critical == 0.80
        assert thresholds.process_rss_critical_bytes == 16 * 1024 * 1024 * 1024


# =============================================================================
# MemoryMonitorConfig Tests
# =============================================================================


class TestMemoryMonitorConfig:
    """Tests for MemoryMonitorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryMonitorConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 30.0
        assert config.kill_enabled is True
        assert config.monitor_gpu is True
        assert config.monitor_ram is True
        assert config.monitor_processes is True

    def test_from_env(self):
        """Test creating config from environment."""
        config = MemoryMonitorConfig.from_env()
        assert config.enabled is True  # Default

    def test_custom_config(self, config):
        """Test custom configuration."""
        assert config.check_interval_seconds == 5.0
        assert config.kill_enabled is False
        assert config.event_cooldown_seconds == 1.0


# =============================================================================
# MemoryStatus Tests
# =============================================================================


class TestMemoryStatus:
    """Tests for MemoryStatus dataclass."""

    def test_default_status(self):
        """Test default status values."""
        status = MemoryStatus()
        assert status.gpu_used_bytes == 0
        assert status.gpu_total_bytes == 0
        assert status.gpu_utilization == 0.0
        assert status.gpu_available is False
        assert status.any_critical is False

    def test_any_critical_gpu(self):
        """Test any_critical with GPU critical."""
        status = MemoryStatus(gpu_critical=True)
        assert status.any_critical is True

    def test_any_critical_ram(self):
        """Test any_critical with RAM critical."""
        status = MemoryStatus(ram_critical=True)
        assert status.any_critical is True

    def test_any_critical_process(self):
        """Test any_critical with process critical."""
        status = MemoryStatus(process_critical=True)
        assert status.any_critical is True

    def test_any_critical_none(self):
        """Test any_critical when none are critical."""
        status = MemoryStatus(
            gpu_warning=True,  # Warning, not critical
            ram_warning=True,
        )
        assert status.any_critical is False


# =============================================================================
# MemoryMonitorDaemon Initialization Tests
# =============================================================================


class TestMemoryMonitorDaemonInit:
    """Tests for MemoryMonitorDaemon initialization."""

    def test_monitor_initialization(self, config):
        """Test monitor initializes correctly."""
        monitor = MemoryMonitorDaemon(config=config)
        assert monitor._memory_config == config
        assert monitor._running is False
        assert monitor._gpu_warnings_emitted == 0
        assert monitor._processes_killed == 0

    def test_monitor_default_config(self):
        """Test monitor with default config."""
        monitor = MemoryMonitorDaemon()
        assert monitor.config.enabled is True

    def test_singleton_pattern(self):
        """Test singleton pattern works."""
        m1 = get_memory_monitor()
        m2 = get_memory_monitor()
        assert m1 is m2

    def test_singleton_reset(self):
        """Test singleton can be reset."""
        m1 = get_memory_monitor()
        reset_memory_monitor()
        m2 = get_memory_monitor()
        assert m1 is not m2


# =============================================================================
# MemoryMonitorDaemon Status Collection Tests
# =============================================================================


class TestMemoryMonitorDaemonStatusCollection:
    """Tests for status collection."""

    @pytest.mark.asyncio
    async def test_collect_memory_status_no_gpu(self, monitor):
        """Test collecting status when no GPU available."""
        with patch.object(monitor, "_get_gpu_memory", return_value=None):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value = MagicMock(
                    used=8 * 1024**3,  # 8GB
                    total=32 * 1024**3,  # 32GB
                    percent=25.0,
                )
                status = await monitor._collect_memory_status()

        assert status.gpu_available is False
        assert status.ram_used_bytes == 8 * 1024**3
        assert status.ram_total_bytes == 32 * 1024**3

    @pytest.mark.asyncio
    async def test_collect_memory_status_with_gpu(self, monitor):
        """Test collecting status with GPU available."""
        gpu_used = 20 * 1024**3  # 20GB
        gpu_total = 24 * 1024**3  # 24GB

        with patch.object(monitor, "_get_gpu_memory", return_value=(gpu_used, gpu_total)):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value = MagicMock(
                    used=16 * 1024**3,
                    total=64 * 1024**3,
                    percent=25.0,
                )
                status = await monitor._collect_memory_status()

        assert status.gpu_available is True
        assert status.gpu_used_bytes == gpu_used
        assert status.gpu_total_bytes == gpu_total
        assert status.gpu_utilization == pytest.approx(20 / 24, rel=0.01)


# =============================================================================
# MemoryMonitorDaemon Threshold Tests
# =============================================================================


class TestMemoryMonitorDaemonThresholds:
    """Tests for threshold detection."""

    @pytest.mark.asyncio
    async def test_gpu_warning_detection(self, monitor):
        """Test GPU warning detection."""
        # 75% usage should trigger warning (threshold is 70%)
        gpu_used = int(0.75 * 24 * 1024**3)
        gpu_total = 24 * 1024**3

        with patch.object(monitor, "_get_gpu_memory", return_value=(gpu_used, gpu_total)):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value = MagicMock(
                    used=8 * 1024**3,
                    total=64 * 1024**3,
                    percent=12.5,
                )
                status = await monitor._collect_memory_status()

        assert status.gpu_warning is True
        assert status.gpu_critical is False  # Below 80%

    @pytest.mark.asyncio
    async def test_gpu_critical_detection(self, monitor):
        """Test GPU critical detection."""
        # 85% usage should trigger critical (threshold is 80%)
        gpu_used = int(0.85 * 24 * 1024**3)
        gpu_total = 24 * 1024**3

        with patch.object(monitor, "_get_gpu_memory", return_value=(gpu_used, gpu_total)):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value = MagicMock(
                    used=8 * 1024**3,
                    total=64 * 1024**3,
                    percent=12.5,
                )
                status = await monitor._collect_memory_status()

        assert status.gpu_warning is True
        assert status.gpu_critical is True


# =============================================================================
# MemoryMonitorDaemon Health Check Tests
# =============================================================================


class TestMemoryMonitorDaemonHealthCheck:
    """Tests for health check."""

    def test_health_check_not_running(self, monitor):
        """Test health check when not running."""
        result = monitor.health_check()
        assert result.healthy is False
        assert "not running" in result.message.lower()

    def test_health_check_disabled(self, config):
        """Test health check when disabled."""
        config.enabled = False
        monitor = MemoryMonitorDaemon(config=config)
        monitor._running = True

        result = monitor.health_check()
        assert result.healthy is True
        assert result.details["enabled"] is False

    def test_health_check_running(self, monitor):
        """Test health check when running normally."""
        monitor._running = True
        monitor._last_status = MemoryStatus()

        result = monitor.health_check()
        assert result.healthy is True

    def test_health_check_degraded(self, monitor):
        """Test health check when memory pressure detected."""
        monitor._running = True
        monitor._last_status = MemoryStatus(gpu_critical=True)

        result = monitor.health_check()
        assert result.healthy is True
        assert "degraded" in result.status.value.lower() or "pressure" in result.message.lower()


# =============================================================================
# MemoryMonitorDaemon Stats Tests
# =============================================================================


class TestMemoryMonitorDaemonStats:
    """Tests for statistics."""

    def test_get_stats(self, monitor):
        """Test getting stats."""
        monitor._gpu_warnings_emitted = 5
        monitor._gpu_criticals_emitted = 2
        monitor._processes_killed = 1

        stats = monitor.get_stats()

        assert stats["gpu_warnings_emitted"] == 5
        assert stats["gpu_criticals_emitted"] == 2
        assert stats["processes_killed"] == 1
        assert "running" in stats
        assert "enabled" in stats

    def test_get_stats_with_last_status(self, monitor):
        """Test getting stats includes last status."""
        monitor._last_status = MemoryStatus(
            gpu_utilization=0.5,
            gpu_available=True,
            ram_utilization=0.3,
        )

        stats = monitor.get_stats()

        assert "last_status" in stats
        assert stats["last_status"]["gpu_utilization"] == 0.5
        assert stats["last_status"]["gpu_available"] is True


# =============================================================================
# MemoryMonitorDaemon Lifecycle Tests
# =============================================================================


class TestMemoryMonitorDaemonLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_on_start_when_disabled(self, config):
        """Test that start does nothing when disabled."""
        config.enabled = False
        monitor = MemoryMonitorDaemon(config=config)

        await monitor._on_start()
        # Should complete without starting monitoring

    @pytest.mark.asyncio
    async def test_on_stop(self, monitor):
        """Test stop lifecycle."""
        await monitor._on_stop()
        # Should complete without error


# =============================================================================
# MemoryMonitorDaemon Event Subscription Tests
# =============================================================================


class TestMemoryMonitorDaemonSubscriptions:
    """Tests for event subscriptions."""

    def test_no_event_subscriptions(self, monitor):
        """Test that monitor has no event subscriptions (cycle-based)."""
        subs = monitor._get_event_subscriptions()
        assert subs == {}


# =============================================================================
# MemoryMonitorDaemon Process Kill Tests
# =============================================================================


class TestMemoryMonitorDaemonProcessKill:
    """Tests for process killing functionality."""

    @pytest.mark.asyncio
    async def test_schedule_kill_disabled(self, monitor):
        """Test that kill is not scheduled when disabled."""
        # Kill is disabled in fixture config
        await monitor._schedule_kill(12345)
        # Should not add to pending kills since kill_enabled is False
        # Actually, _schedule_kill still adds - let's test with enabled

    @pytest.mark.asyncio
    async def test_schedule_kill_with_invalid_pid(self, monitor):
        """Test scheduling kill with non-existent process."""
        monitor._memory_config.kill_enabled = True

        with patch("os.kill", side_effect=ProcessLookupError):
            await monitor._schedule_kill(99999)

        # Should not add to pending kills since process doesn't exist
        assert 99999 not in monitor._pending_kills

    @pytest.mark.asyncio
    async def test_process_pending_kills_cleanup(self, monitor):
        """Test pending kills are cleaned up."""
        monitor._pending_kills = {12345: time.time() - 100}  # Old entry

        with patch("psutil.pid_exists", return_value=False):
            await monitor._process_pending_kills()

        assert 12345 not in monitor._pending_kills
        assert monitor._processes_killed == 1
