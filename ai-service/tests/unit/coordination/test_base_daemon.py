"""Tests for base_daemon.py module.

December 2025: Added as part of test coverage initiative.
"""

from __future__ import annotations

import asyncio
import os
import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.base_daemon import (
    BaseDaemon,
    DaemonConfig,
)
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockDaemon(BaseDaemon[DaemonConfig]):
    """Concrete mock daemon implementation for testing."""

    def __init__(self, config: DaemonConfig | None = None):
        super().__init__(config)
        self.cycle_count = 0
        self.on_start_called = False
        self.on_stop_called = False
        self.should_fail = False

    async def _run_cycle(self) -> None:
        self.cycle_count += 1
        if self.should_fail:
            raise RuntimeError("Test failure")

    @staticmethod
    def _get_default_config() -> DaemonConfig:
        return DaemonConfig()

    async def _on_start(self) -> None:
        self.on_start_called = True

    async def _on_stop(self) -> None:
        self.on_stop_called = True


# =============================================================================
# DaemonConfig Tests
# =============================================================================


class TestDaemonConfigClass:
    """Tests for DaemonConfig class."""

    def test_default_values(self):
        """DaemonConfig has correct defaults."""
        config = DaemonConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 300

    def test_custom_values(self):
        """DaemonConfig accepts custom values."""
        config = DaemonConfig(enabled=False, check_interval_seconds=60)
        assert config.enabled is False
        assert config.check_interval_seconds == 60

    def test_from_env_enabled(self):
        """from_env reads enabled from environment."""
        with patch.dict(os.environ, {"RINGRIFT_ENABLED": "0"}):
            config = DaemonConfig.from_env("RINGRIFT")
            assert config.enabled is False

        with patch.dict(os.environ, {"RINGRIFT_ENABLED": "1"}):
            config = DaemonConfig.from_env("RINGRIFT")
            assert config.enabled is True

    def test_from_env_interval(self):
        """from_env reads interval from environment."""
        with patch.dict(os.environ, {"RINGRIFT_INTERVAL": "60"}):
            config = DaemonConfig.from_env("RINGRIFT")
            assert config.check_interval_seconds == 60

    def test_from_env_invalid_interval(self):
        """from_env handles invalid interval gracefully."""
        with patch.dict(os.environ, {"RINGRIFT_INTERVAL": "invalid"}):
            config = DaemonConfig.from_env("RINGRIFT")
            assert config.check_interval_seconds == 300  # Default

    def test_from_env_custom_prefix(self):
        """from_env uses custom prefix."""
        with patch.dict(os.environ, {"MYPREFIX_INTERVAL": "120"}):
            config = DaemonConfig.from_env("MYPREFIX")
            assert config.check_interval_seconds == 120


# =============================================================================
# BaseDaemon Initialization Tests
# =============================================================================


class TestBaseDaemonInit:
    """Tests for BaseDaemon initialization."""

    def test_init_with_default_config(self):
        """Daemon initializes with default config."""
        daemon = MockDaemon()
        assert daemon.config.enabled is True
        assert daemon.config.check_interval_seconds == 300

    def test_init_with_custom_config(self):
        """Daemon initializes with custom config."""
        config = DaemonConfig(check_interval_seconds=60)
        daemon = MockDaemon(config)
        assert daemon.config.check_interval_seconds == 60

    def test_init_sets_state(self):
        """Daemon initializes state correctly."""
        daemon = MockDaemon()
        assert daemon._running is False
        assert daemon._task is None
        assert daemon._start_time == 0.0
        assert daemon._events_processed == 0
        assert daemon._errors_count == 0
        assert daemon._cycles_completed == 0

    def test_init_sets_node_id(self):
        """Daemon sets node_id from hostname."""
        daemon = MockDaemon()
        assert daemon.node_id is not None
        assert isinstance(daemon.node_id, str)


# =============================================================================
# BaseDaemon Lifecycle Tests
# =============================================================================


class TestBaseDaemonLifecycle:
    """Tests for BaseDaemon lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """start() sets running flag."""
        daemon = MockDaemon()
        await daemon.start()
        assert daemon.is_running is True
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_sets_start_time(self):
        """start() sets start_time."""
        daemon = MockDaemon()
        await daemon.start()
        assert daemon._start_time > 0
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_calls_on_start(self):
        """start() calls _on_start hook."""
        daemon = MockDaemon()
        await daemon.start()
        assert daemon.on_start_called is True
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """start() creates main loop task."""
        daemon = MockDaemon()
        await daemon.start()
        assert daemon._task is not None
        assert not daemon._task.done()
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """start() is no-op when already running."""
        daemon = MockDaemon()
        await daemon.start()
        initial_task = daemon._task
        await daemon.start()  # Should not create new task
        assert daemon._task is initial_task
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_when_disabled(self):
        """start() is no-op when disabled."""
        config = DaemonConfig(enabled=False)
        daemon = MockDaemon(config)
        await daemon.start()
        assert daemon.is_running is False
        assert daemon._task is None

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self):
        """stop() clears running flag."""
        daemon = MockDaemon()
        await daemon.start()
        await daemon.stop()
        assert daemon.is_running is False

    @pytest.mark.asyncio
    async def test_stop_calls_on_stop(self):
        """stop() calls _on_stop hook."""
        daemon = MockDaemon()
        await daemon.start()
        await daemon.stop()
        assert daemon.on_stop_called is True

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """stop() cancels main loop task."""
        daemon = MockDaemon()
        await daemon.start()
        task = daemon._task
        await daemon.stop()
        assert task.done() or task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """stop() is no-op when not running."""
        daemon = MockDaemon()
        await daemon.stop()  # Should not raise
        assert daemon.is_running is False


class TestBaseDaemonProperties:
    """Tests for BaseDaemon properties."""

    def test_is_running_false_by_default(self):
        """is_running is False initially."""
        daemon = MockDaemon()
        assert daemon.is_running is False

    @pytest.mark.asyncio
    async def test_is_running_true_when_started(self):
        """is_running is True when running."""
        daemon = MockDaemon()
        await daemon.start()
        assert daemon.is_running is True
        await daemon.stop()

    def test_uptime_seconds_zero_initially(self):
        """uptime_seconds is 0 when not started."""
        daemon = MockDaemon()
        assert daemon.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_uptime_seconds_increases(self):
        """uptime_seconds increases when running."""
        daemon = MockDaemon()
        await daemon.start()
        await asyncio.sleep(0.1)
        assert daemon.uptime_seconds > 0
        await daemon.stop()


# =============================================================================
# Main Loop Tests
# =============================================================================


class TestBaseDaemonMainLoop:
    """Tests for BaseDaemon main loop."""

    @pytest.mark.asyncio
    async def test_run_cycle_called(self):
        """Main loop calls _run_cycle."""
        config = DaemonConfig(check_interval_seconds=0)  # No delay
        daemon = MockDaemon(config)
        await daemon.start()
        await asyncio.sleep(0.05)  # Allow cycle to run
        assert daemon.cycle_count > 0
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_cycles_completed_tracked(self):
        """Main loop tracks cycles completed."""
        config = DaemonConfig(check_interval_seconds=0)
        daemon = MockDaemon(config)
        await daemon.start()
        await asyncio.sleep(0.05)
        await daemon.stop()
        assert daemon._cycles_completed > 0

    @pytest.mark.asyncio
    async def test_error_handling_continues(self):
        """Main loop continues after errors."""
        config = DaemonConfig(check_interval_seconds=0)
        daemon = MockDaemon(config)
        daemon.should_fail = True
        await daemon.start()
        await asyncio.sleep(0.05)
        assert daemon._errors_count > 0
        assert daemon.is_running is True  # Still running
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_error_recorded(self):
        """Main loop records error details."""
        config = DaemonConfig(check_interval_seconds=0)
        daemon = MockDaemon(config)
        daemon.should_fail = True
        await daemon.start()
        await asyncio.sleep(0.05)
        await daemon.stop()
        assert "Test failure" in daemon._last_error


# =============================================================================
# Status and Health Tests
# =============================================================================


class TestBaseDaemonStatus:
    """Tests for BaseDaemon status."""

    def test_get_status_structure(self):
        """get_status returns expected structure."""
        daemon = MockDaemon()
        status = daemon.get_status()
        assert "daemon" in status
        assert "running" in status
        assert "uptime_seconds" in status
        assert "node_id" in status
        assert "config" in status
        assert "stats" in status
        assert "coordinator_status" in status

    def test_get_status_config(self):
        """get_status includes config."""
        daemon = MockDaemon()
        status = daemon.get_status()
        assert status["config"]["enabled"] is True
        assert status["config"]["interval"] == 300

    def test_get_status_stats(self):
        """get_status includes stats."""
        daemon = MockDaemon()
        daemon._events_processed = 10
        daemon._errors_count = 2
        daemon._cycles_completed = 5
        status = daemon.get_status()
        assert status["stats"]["events_processed"] == 10
        assert status["stats"]["errors"] == 2
        assert status["stats"]["cycles_completed"] == 5


class TestBaseDaemonHealthCheck:
    """Tests for BaseDaemon health check."""

    def test_health_check_not_running(self):
        """health_check returns unhealthy when not running."""
        daemon = MockDaemon()
        result = daemon.health_check()
        assert isinstance(result, HealthCheckResult)
        assert result.healthy is False
        assert "not running" in result.message

    @pytest.mark.asyncio
    async def test_health_check_running(self):
        """health_check returns healthy when running."""
        daemon = MockDaemon()
        await daemon.start()
        result = daemon.health_check()
        assert result.healthy is True
        assert "healthy" in result.message
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_health_check_high_error_rate(self):
        """health_check returns unhealthy on high error rate."""
        daemon = MockDaemon()
        await daemon.start()
        # Simulate high error rate
        daemon._cycles_completed = 20
        daemon._errors_count = 15  # 75% error rate
        result = daemon.health_check()
        assert result.healthy is False
        assert "error rate" in result.message.lower()
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_health_check_low_error_rate(self):
        """health_check returns healthy on low error rate."""
        daemon = MockDaemon()
        await daemon.start()
        daemon._cycles_completed = 100
        daemon._errors_count = 10  # 10% error rate
        result = daemon.health_check()
        assert result.healthy is True
        await daemon.stop()


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestBaseDaemonUtilities:
    """Tests for BaseDaemon utility methods."""

    def test_record_event_processed(self):
        """record_event_processed increments counter."""
        daemon = MockDaemon()
        assert daemon._events_processed == 0
        daemon.record_event_processed()
        assert daemon._events_processed == 1
        daemon.record_event_processed(5)
        assert daemon._events_processed == 6

    def test_record_error(self):
        """record_error increments counter and stores error."""
        daemon = MockDaemon()
        assert daemon._errors_count == 0
        daemon.record_error("Test error")
        assert daemon._errors_count == 1
        assert daemon._last_error == "Test error"

    def test_record_error_with_exception(self):
        """record_error handles exceptions."""
        daemon = MockDaemon()
        daemon.record_error(ValueError("Bad value"))
        assert daemon._errors_count == 1
        assert "Bad value" in daemon._last_error

    def test_get_daemon_name(self):
        """_get_daemon_name returns class name."""
        daemon = MockDaemon()
        assert daemon._get_daemon_name() == "MockDaemon"


# =============================================================================
# Coordinator Protocol Tests
# =============================================================================


class TestBaseDaemonCoordinatorProtocol:
    """Tests for BaseDaemon coordinator protocol."""

    @pytest.mark.asyncio
    async def test_coordinator_register_on_start(self):
        """start() calls coordinator registration method."""
        daemon = MockDaemon()
        # Track if _coordinator_register was called by patching the instance method
        register_called = False
        original_register = daemon._coordinator_register

        def mock_register():
            nonlocal register_called
            register_called = True
            original_register()

        daemon._coordinator_register = mock_register
        await daemon.start()
        assert register_called is True
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_coordinator_unregister_on_stop(self):
        """stop() unregisters from coordinator protocol."""
        with patch("app.coordination.base_daemon.register_coordinator"):
            with patch("app.coordination.base_daemon.unregister_coordinator") as mock_unreg:
                daemon = MockDaemon()
                await daemon.start()
                await daemon.stop()
                mock_unreg.assert_called_once()

    @pytest.mark.asyncio
    async def test_coordinator_status_updates(self):
        """Coordinator status updates on lifecycle changes."""
        daemon = MockDaemon()
        assert daemon._coordinator_status == CoordinatorStatus.INITIALIZING

        await daemon.start()
        assert daemon._coordinator_status == CoordinatorStatus.RUNNING

        await daemon.stop()
        assert daemon._coordinator_status == CoordinatorStatus.STOPPED


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestBaseDaemonEdgeCases:
    """Tests for BaseDaemon edge cases."""

    @pytest.mark.asyncio
    async def test_multiple_start_stop_cycles(self):
        """Daemon handles multiple start/stop cycles."""
        daemon = MockDaemon()

        for _ in range(3):
            await daemon.start()
            assert daemon.is_running is True
            await daemon.stop()
            assert daemon.is_running is False

    @pytest.mark.asyncio
    async def test_coordinator_register_failure_handled(self):
        """Coordinator registration failure is handled gracefully."""
        with patch(
            "app.coordination.base_daemon.register_coordinator",
            side_effect=Exception("Registration failed"),
        ):
            daemon = MockDaemon()
            await daemon.start()  # Should not raise
            assert daemon.is_running is True
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_coordinator_unregister_failure_handled(self):
        """Coordinator unregistration failure is handled gracefully."""
        with patch("app.coordination.base_daemon.register_coordinator"):
            with patch(
                "app.coordination.base_daemon.unregister_coordinator",
                side_effect=Exception("Unregistration failed"),
            ):
                daemon = MockDaemon()
                await daemon.start()
                await daemon.stop()  # Should not raise
                assert daemon.is_running is False
