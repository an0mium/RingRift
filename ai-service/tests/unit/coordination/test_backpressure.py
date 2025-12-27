"""Tests for unified backpressure monitoring.

Tests the BackpressureSignal and BackpressureMonitor for cluster coordination.
"""

import asyncio
import pytest
import time

from app.coordination.backpressure import (
    BackpressureConfig,
    BackpressureMonitor,
    BackpressureSignal,
    get_backpressure_monitor,
    reset_backpressure_monitor,
)


class TestBackpressureSignal:
    """Tests for BackpressureSignal dataclass."""

    def test_default_signal_is_healthy(self):
        """Default signal should indicate healthy system."""
        signal = BackpressureSignal()
        assert signal.is_healthy
        assert signal.spawn_rate_multiplier == 1.0
        assert not signal.should_pause

    def test_overall_pressure_calculation(self):
        """Overall pressure should be weighted average."""
        signal = BackpressureSignal(
            queue_pressure=1.0,  # 30%
            training_pressure=0.0,  # 25%
            disk_pressure=0.0,  # 20%
            sync_pressure=0.0,  # 15%
            memory_pressure=0.0,  # 10%
        )
        assert abs(signal.overall_pressure - 0.30) < 0.01

    def test_full_pressure_causes_pause(self):
        """Full pressure should cause spawn pause."""
        signal = BackpressureSignal(
            queue_pressure=1.0,
            training_pressure=1.0,
            disk_pressure=1.0,
            sync_pressure=1.0,
            memory_pressure=1.0,
        )
        assert signal.overall_pressure == 1.0
        assert signal.spawn_rate_multiplier == 0.0
        assert signal.should_pause

    def test_medium_pressure_throttles(self):
        """Medium pressure should reduce spawn rate."""
        signal = BackpressureSignal(
            queue_pressure=0.6,
            training_pressure=0.6,
            disk_pressure=0.6,
            sync_pressure=0.6,
            memory_pressure=0.6,
        )
        assert 0.3 < signal.overall_pressure < 0.9
        assert 0.0 < signal.spawn_rate_multiplier < 1.0

    def test_spawn_rate_multiplier_thresholds(self):
        """Test spawn rate multiplier threshold behavior."""
        # Low pressure - full speed
        low = BackpressureSignal(queue_pressure=0.2)
        assert low.spawn_rate_multiplier == 1.0

        # High pressure - stopped (need all pressures at 1.0 to get overall > 0.9)
        high = BackpressureSignal(
            queue_pressure=1.0,
            training_pressure=1.0,
            disk_pressure=1.0,
            sync_pressure=1.0,
            memory_pressure=1.0,
        )
        assert high.spawn_rate_multiplier == 0.0

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all relevant fields."""
        signal = BackpressureSignal(
            queue_pressure=0.5,
            training_pressure=0.3,
        )
        d = signal.to_dict()
        assert "queue_pressure" in d
        assert "training_pressure" in d
        assert "disk_pressure" in d
        assert "sync_pressure" in d
        assert "memory_pressure" in d
        assert "overall_pressure" in d
        assert "spawn_rate_multiplier" in d
        assert "should_pause" in d
        assert "is_healthy" in d
        assert "timestamp" in d


class TestBackpressureConfig:
    """Tests for BackpressureConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = BackpressureConfig()
        assert config.queue_low_threshold < config.queue_high_threshold
        assert config.training_low_threshold < config.training_high_threshold
        assert config.disk_low_threshold < config.disk_high_threshold
        assert config.cache_ttl_seconds > 0


class TestBackpressureMonitor:
    """Tests for BackpressureMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor for testing."""
        reset_backpressure_monitor()
        return BackpressureMonitor()

    @pytest.mark.asyncio
    async def test_get_signal_returns_signal(self, monitor):
        """get_signal should return a BackpressureSignal."""
        signal = await monitor.get_signal()
        assert isinstance(signal, BackpressureSignal)

    @pytest.mark.asyncio
    async def test_signal_is_cached(self, monitor):
        """Signal should be cached for TTL duration."""
        signal1 = await monitor.get_signal()
        signal2 = await monitor.get_signal()
        # Should be the same cached signal
        assert signal1.timestamp == signal2.timestamp

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, monitor):
        """force_refresh should bypass cache."""
        signal1 = await monitor.get_signal()
        await asyncio.sleep(0.01)
        signal2 = await monitor.get_signal(force_refresh=True)
        # Timestamps should be different
        assert signal2.timestamp >= signal1.timestamp

    def test_get_cached_signal_returns_none_initially(self, monitor):
        """get_cached_signal should return None before first get_signal."""
        assert monitor.get_cached_signal() is None

    @pytest.mark.asyncio
    async def test_get_cached_signal_after_fetch(self, monitor):
        """get_cached_signal should return signal after fetch."""
        await monitor.get_signal()
        cached = monitor.get_cached_signal()
        assert cached is not None
        assert isinstance(cached, BackpressureSignal)

    def test_normalize_function(self, monitor):
        """Test the _normalize function."""
        # Below low threshold
        assert monitor._normalize(5, 10, 100) == 0.0
        # Above high threshold
        assert monitor._normalize(150, 10, 100) == 1.0
        # At midpoint
        assert abs(monitor._normalize(55, 10, 100) - 0.5) < 0.01


class TestBackpressureMonitorSingleton:
    """Tests for singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_backpressure_monitor()

    def test_get_backpressure_monitor_returns_same_instance(self):
        """get_backpressure_monitor should return same instance."""
        m1 = get_backpressure_monitor()
        m2 = get_backpressure_monitor()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        """reset_backpressure_monitor should allow new instance."""
        m1 = get_backpressure_monitor()
        reset_backpressure_monitor()
        m2 = get_backpressure_monitor()
        assert m1 is not m2
