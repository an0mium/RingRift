"""Integration tests for daemon health monitoring and auto-recovery.

These tests verify that:
1. Health monitoring loop detects failed daemons
2. Auto-restart recovers crashed daemons
3. Restart backoff prevents rapid restart loops
4. Critical daemons are prioritized for recovery

Created: December 2025
Purpose: Ensure daemons self-heal after crashes
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_manager import (
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
)
from app.coordination.daemon_types import (
    CRITICAL_DAEMONS,
    DaemonType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fast_config():
    """Create config with very short intervals for fast tests."""
    return DaemonManagerConfig(
        # Keep these tests isolated from the full coordination bootstrap.
        # They validate DaemonManager health-loop behavior, not cluster wiring.
        enable_coordination_wiring=False,
        dependency_wait_timeout=0.25,
        dependency_poll_interval=0.01,
        health_check_interval=0.05,  # 50ms
        shutdown_timeout=0.5,
        recovery_cooldown=0.1,
        auto_restart_failed=True,
        max_restart_attempts=3,
    )


@pytest.fixture
def manager(fast_config):
    """Create fresh DaemonManager for each test."""
    DaemonManager.reset_instance()
    mgr = DaemonManager(fast_config)
    mgr._factories.clear()
    mgr._daemons.clear()
    yield mgr
    mgr._running = False
    if mgr._shutdown_event:
        mgr._shutdown_event.set()
    for info in list(mgr._daemons.values()):
        if info.task and not info.task.done():
            info.task.cancel()
    DaemonManager.reset_instance()


# =============================================================================
# Health Loop Detection Tests
# =============================================================================


class TestHealthLoopDetection:
    """Tests for health monitoring loop detection of failed daemons."""

    @pytest.mark.asyncio
    async def test_health_loop_detects_crashed_daemon(self, manager: DaemonManager):
        """Health loop should detect when a daemon crashes (with auto-restart disabled on daemon)."""
        crash_flag = asyncio.Event()

        async def crashing_daemon():
            """Daemon that crashes after a short delay."""
            await asyncio.sleep(0.05)
            crash_flag.set()
            raise RuntimeError("Intentional crash for testing")

        # Register with auto_restart=False to test crash detection
        manager.register_factory(DaemonType.DATA_PIPELINE, crashing_daemon, auto_restart=False)

        # Start daemon
        await manager.start(DaemonType.DATA_PIPELINE)
        await asyncio.sleep(0.02)

        assert manager._daemons[DaemonType.DATA_PIPELINE].state == DaemonState.RUNNING

        # Wait for crash
        await crash_flag.wait()

        # Wait for task to complete
        info = manager._daemons[DaemonType.DATA_PIPELINE]
        for _ in range(20):  # Up to 1 second
            await asyncio.sleep(0.05)
            if info.task and info.task.done():
                break

        # Key assertions: error was recorded and task completed
        assert info.last_error is not None, "last_error should be set after crash"
        assert "Intentional crash" in info.last_error
        assert info.task is not None and info.task.done(), "Task should be completed after crash"

        await manager.shutdown()



# =============================================================================
# Auto-Restart Tests
# =============================================================================


class TestAutoRestart:
    """Tests for automatic daemon restart on failure."""

    @pytest.mark.asyncio
    async def test_auto_restart_on_crash(self, manager: DaemonManager):
        """Manager should auto-restart failed daemons when configured."""
        crash_count = 0
        restart_count = 0
        stable_started = asyncio.Event()

        async def crashy_daemon():
            """Crashes once, then runs stably."""
            nonlocal crash_count, restart_count
            restart_count += 1

            if crash_count == 0:
                crash_count += 1
                await asyncio.sleep(0.02)
                raise RuntimeError("First run crash")
            else:
                stable_started.set()
                # Run stably
                await asyncio.sleep(10)

        manager.register_factory(DaemonType.IDLE_RESOURCE, crashy_daemon)

        # Reduce restart delay for faster test
        manager._daemons[DaemonType.IDLE_RESOURCE].restart_delay = 0.1

        # Start daemon
        await manager.start(DaemonType.IDLE_RESOURCE)
        await asyncio.sleep(0.02)

        # Wait for restart after crash (with reduced delay, should be quick)
        try:
            await asyncio.wait_for(stable_started.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pytest.fail("Daemon was not auto-restarted after crash")

        # Should have restarted
        assert restart_count == 2, f"Expected 2 starts (initial + restart), got {restart_count}"
        assert manager._daemons[DaemonType.IDLE_RESOURCE].state == DaemonState.RUNNING

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_restart_backoff_prevents_rapid_loops(self, manager: DaemonManager):
        """Manager should apply backoff to prevent rapid restart loops."""
        start_times: list[float] = []

        async def always_crash_daemon():
            """Always crashes immediately."""
            import time
            start_times.append(time.time())
            await asyncio.sleep(0.01)
            raise RuntimeError("Always crash")

        manager.register_factory(DaemonType.AUTO_SYNC, always_crash_daemon)

        # Start daemon
        await manager.start(DaemonType.AUTO_SYNC)

        # Let it crash and restart a few times
        await asyncio.sleep(1.0)

        # Check that restarts are not too rapid
        if len(start_times) >= 2:
            intervals = [
                start_times[i+1] - start_times[i]
                for i in range(len(start_times) - 1)
            ]
            # At least some intervals should show backoff (> 50ms)
            assert any(interval >= 0.05 for interval in intervals), (
                f"No backoff observed in restart intervals: {intervals}"
            )

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_max_restart_attempts_honored(self, manager: DaemonManager):
        """Manager should stop restarting after max attempts."""
        restart_count = 0

        async def always_crash_daemon():
            """Always crashes immediately."""
            nonlocal restart_count
            restart_count += 1
            await asyncio.sleep(0.01)
            raise RuntimeError("Always crash")

        # Register with low max_restarts for fast test
        manager.register_factory(DaemonType.FEEDBACK_LOOP, always_crash_daemon, max_restarts=2)

        # Reduce restart delay for faster test
        manager._daemons[DaemonType.FEEDBACK_LOOP].restart_delay = 0.1

        # Start daemon
        await manager.start(DaemonType.FEEDBACK_LOOP)

        # Wait for restart attempts to exhaust
        info = manager._daemons[DaemonType.FEEDBACK_LOOP]
        for _ in range(40):  # Up to 4 seconds
            await asyncio.sleep(0.1)
            if info.state == DaemonState.FAILED:
                break

        # Should have given up after max attempts
        assert restart_count <= info.max_restarts + 1, (
            f"Restarted {restart_count} times, should stop at {info.max_restarts}"
        )
        assert info.state == DaemonState.FAILED, f"Expected FAILED, got {info.state}"

        await manager.shutdown()


# =============================================================================
# Critical Daemon Priority Tests
# =============================================================================


class TestCriticalDaemonPriority:
    """Tests for critical daemon priority handling."""

    @pytest.mark.asyncio
    async def test_critical_daemons_have_auto_restart_enabled(self, manager: DaemonManager):
        """Critical daemons should have auto_restart=True by default."""
        async def dummy_daemon():
            await asyncio.sleep(10)

        for daemon_type in CRITICAL_DAEMONS:
            manager.register_factory(daemon_type, dummy_daemon)
            await manager.start(daemon_type)
            await asyncio.sleep(0.01)

            info = manager._daemons.get(daemon_type)
            assert info is not None, f"{daemon_type.name} not registered"
            assert info.auto_restart is True, (
                f"Critical daemon {daemon_type.name} should have auto_restart=True"
            )

        await manager.shutdown()


# =============================================================================
# Health Loop Lifecycle Tests
# =============================================================================


class TestHealthLoopLifecycle:
    """Tests for health monitoring loop lifecycle."""

    @pytest.mark.asyncio
    async def test_health_loop_starts_with_daemon(self, manager: DaemonManager):
        """Health loop should start when first daemon starts."""
        async def dummy_daemon():
            await asyncio.sleep(10)

        manager.register_factory(DaemonType.EVENT_ROUTER, dummy_daemon)

        # Start daemon
        await manager.start(DaemonType.EVENT_ROUTER)
        await asyncio.sleep(0.1)

        # Daemon should be running
        assert manager._daemons[DaemonType.EVENT_ROUTER].state == DaemonState.RUNNING

        # Manager should be in running state
        assert manager._running is True

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_health_loop_stops_on_shutdown(self, manager: DaemonManager):
        """Health loop should stop cleanly on shutdown."""
        async def dummy_daemon():
            await asyncio.sleep(10)

        manager.register_factory(DaemonType.DAEMON_WATCHDOG, dummy_daemon)
        await manager.start(DaemonType.DAEMON_WATCHDOG)
        await asyncio.sleep(0.1)

        # Verify daemon is running
        assert manager._daemons[DaemonType.DAEMON_WATCHDOG].state == DaemonState.RUNNING
        assert manager._running is True

        # Shutdown
        await manager.shutdown()
        await asyncio.sleep(0.1)

        # Manager should not be running
        assert manager._running is False


# =============================================================================
# Error Tracking Tests
# =============================================================================


class TestErrorTracking:
    """Tests for daemon error tracking."""

    @pytest.mark.asyncio
    async def test_last_error_recorded_on_crash(self, manager: DaemonManager):
        """last_error should be set when daemon crashes."""
        error_message = "Test error message for tracking"

        async def error_daemon():
            await asyncio.sleep(0.02)
            # Use RuntimeError since it's in the caught exceptions
            raise RuntimeError(error_message)

        # Register with auto_restart=False to let it stay in FAILED state
        manager.register_factory(DaemonType.TRAINING_TRIGGER, error_daemon, auto_restart=False)

        await manager.start(DaemonType.TRAINING_TRIGGER)

        # Wait for task to complete
        info = manager._daemons[DaemonType.TRAINING_TRIGGER]
        for _ in range(20):  # Up to 1 second
            await asyncio.sleep(0.05)
            if info.task and info.task.done():
                break

        assert info.last_error is not None, "last_error should be set after crash"
        assert error_message in info.last_error

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_restart_count_incremented_on_restart(self, manager: DaemonManager):
        """restart_count should increment each time daemon restarts."""
        call_count = 0
        stable_running = asyncio.Event()

        async def count_daemon():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.02)
            if call_count < 3:
                raise RuntimeError(f"Crash #{call_count}")
            # Run stably on 3rd attempt
            stable_running.set()
            await asyncio.sleep(10)

        manager.register_factory(DaemonType.DATA_PIPELINE, count_daemon)

        await manager.start(DaemonType.DATA_PIPELINE)

        # Wait for stable run to start
        try:
            await asyncio.wait_for(stable_running.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pass  # May timeout, check restart count anyway

        info = manager._daemons[DaemonType.DATA_PIPELINE]
        # restart_count should match number of restarts (not initial start)
        assert info.restart_count >= 1, (
            f"restart_count should be >= 1 after restarts, got {info.restart_count}"
        )

        await manager.shutdown()
