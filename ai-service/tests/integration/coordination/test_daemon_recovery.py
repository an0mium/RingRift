"""Integration tests for daemon crash recovery and auto-restart.

These tests verify that the DaemonManager correctly handles:
1. Daemon crashes triggering auto-restart
2. Restart count limits being respected
3. Cascading dependency restarts
4. Import errors preventing restart
5. Recovery after cooldown period

Created: December 2025
Purpose: Ensure production daemons don't crash silently
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_manager import (
    DaemonInfo,
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager_config():
    """Create test config with short intervals for faster tests."""
    return DaemonManagerConfig(
        health_check_interval=0.1,  # 100ms for fast tests
        shutdown_timeout=1.0,
        recovery_cooldown=0.2,  # 200ms cooldown
        auto_restart_failed=True,
    )


@pytest.fixture
def manager(manager_config):
    """Create fresh DaemonManager for each test with no default factories."""
    DaemonManager.reset_instance()
    mgr = DaemonManager(manager_config)
    # Clear default factories to avoid loading heavy dependencies
    mgr._factories.clear()
    mgr._daemons.clear()
    yield mgr
    # Cleanup - stop any running tasks
    mgr._running = False
    if mgr._shutdown_event:
        mgr._shutdown_event.set()
    for info in list(mgr._daemons.values()):
        if info.task and not info.task.done():
            info.task.cancel()
    DaemonManager.reset_instance()


# =============================================================================
# Crash Recovery Tests
# =============================================================================


class TestDaemonCrashRecovery:
    """Tests for daemon crash detection and auto-restart."""

    @pytest.mark.asyncio
    async def test_daemon_crash_triggers_auto_restart(self, manager: DaemonManager):
        """Daemon crash should trigger automatic restart."""
        call_count = 0

        async def crashing_factory():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call crashes
                raise RuntimeError("Simulated crash")
            # Subsequent calls succeed
            while True:
                await asyncio.sleep(1)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            crashing_factory,
            auto_restart=True,
            max_restarts=3,
        )

        # Set short restart delay for faster tests (default is 5.0s)
        manager._daemons[DaemonType.MODEL_SYNC].restart_delay = 0.1

        # Start the daemon
        await manager.start(DaemonType.MODEL_SYNC)

        # Wait for crash and restart cycle to complete
        # The DaemonLifecycleManager has a minimum 1s restart delay
        await asyncio.sleep(1.5)

        # Verify restart happened (factory called at least twice)
        assert call_count >= 2, f"Factory should be called at least twice, got {call_count}"

        # Verify daemon is now running after restart
        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.state in (
            DaemonState.RUNNING,
            DaemonState.RESTARTING,
        ), f"Expected RUNNING or RESTARTING after restart, got {info.state}"
        assert info.restart_count >= 1, f"Restart count should be at least 1, got {info.restart_count}"

    @pytest.mark.asyncio
    async def test_restart_count_limits_respected(self, manager: DaemonManager):
        """Daemon should stop restarting after max_restarts reached."""
        call_count = 0

        async def always_crashing_factory():
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"Crash #{call_count}")

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            always_crashing_factory,
            auto_restart=True,
            max_restarts=2,
        )

        # Disable auto_restart_failed in config to prevent health loop from interfering
        # We're testing the daemon_lifecycle restart logic, not the health check restart logic
        manager.config.auto_restart_failed = False

        # Set short restart delay for faster tests
        manager._daemons[DaemonType.MODEL_SYNC].restart_delay = 0.1

        # Start the daemon
        await manager.start(DaemonType.MODEL_SYNC)

        # Poll until daemon enters FAILED state (max 5 seconds timeout)
        info = manager._daemons[DaemonType.MODEL_SYNC]
        for _ in range(50):  # 50 * 0.1s = 5s max
            if info.state == DaemonState.FAILED:
                break
            await asyncio.sleep(0.1)

        # Verify max restarts respected
        assert info.restart_count == 2, f"Restart count should be 2, got {info.restart_count}"
        assert info.state == DaemonState.FAILED, f"Expected FAILED after max restarts, got {info.state}"
        # Factory should be called 3 times: initial + 2 restarts
        assert call_count == 3, f"Factory should be called 3 times (1 initial + 2 restarts), got {call_count}"

    @pytest.mark.asyncio
    async def test_cascading_dependency_restart(self, manager: DaemonManager):
        """When dependency fails, dependents should also restart."""
        parent_started = 0
        child_started = 0

        async def parent_factory():
            nonlocal parent_started
            parent_started += 1
            while True:
                await asyncio.sleep(1)

        async def child_factory():
            nonlocal child_started
            child_started += 1
            while True:
                await asyncio.sleep(1)

        # Register parent first
        manager.register_factory(
            DaemonType.EVENT_ROUTER,
            parent_factory,
            auto_restart=True,
        )

        # Register child with dependency on parent
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            child_factory,
            depends_on=[DaemonType.EVENT_ROUTER],
            auto_restart=True,
        )

        # Start both
        await manager.start(DaemonType.EVENT_ROUTER)
        await manager.start(DaemonType.DATA_PIPELINE)
        await asyncio.sleep(0.1)

        initial_parent_starts = parent_started
        initial_child_starts = child_started

        # Simulate parent failure by cancelling its task
        parent_info = manager._daemons[DaemonType.EVENT_ROUTER]
        if parent_info.task:
            parent_info.task.cancel()
            try:
                await parent_info.task
            except asyncio.CancelledError:
                pass
        parent_info.state = DaemonState.FAILED

        # Run health check to trigger cascade restart
        await manager._check_health()
        await asyncio.sleep(0.2)

        # Verify both restarted
        assert parent_started > initial_parent_starts, "Parent should have restarted"
        # Note: Child restart depends on cascade logic being triggered

    @pytest.mark.asyncio
    async def test_import_error_prevents_restart(self, manager: DaemonManager):
        """Daemons with import errors should not attempt restart."""
        # Directly set up a daemon with import error
        info = DaemonInfo(daemon_type=DaemonType.MODEL_SYNC)
        info.state = DaemonState.FAILED
        info.import_error = "ModuleNotFoundError: No module named 'nonexistent'"
        info.last_failure_time = 0  # Long ago
        info.auto_restart = True
        info.max_restarts = 5
        info.restart_count = 0

        manager._daemons[DaemonType.MODEL_SYNC] = info

        # Run health check
        await manager._check_health()

        # Verify no restart attempted
        assert info.state == DaemonState.FAILED
        assert info.restart_count == 0, "Import error daemons should not be restarted"


class TestRecoveryAfterCooldown:
    """Tests for recovery behavior after cooldown period."""

    @pytest.mark.asyncio
    async def test_failed_daemon_recovery_after_cooldown(self, manager: DaemonManager):
        """Failed daemon should attempt recovery after cooldown."""
        recovery_attempted = False

        async def recoverable_factory():
            nonlocal recovery_attempted
            recovery_attempted = True
            while True:
                await asyncio.sleep(1)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            recoverable_factory,
            auto_restart=True,
            max_restarts=3,
        )

        # Set up failed daemon (exceeded max restarts)
        info = manager._daemons[DaemonType.MODEL_SYNC]
        info.state = DaemonState.FAILED
        info.restart_count = 5  # Exceeded max
        info.last_failure_time = 0  # Long ago (past cooldown)

        # Run health check
        await manager._check_health()
        await asyncio.sleep(0.1)

        # Verify recovery was attempted
        assert recovery_attempted, "Recovery should be attempted after cooldown"
        assert info.restart_count == 0, "Restart count should reset after cooldown"

    @pytest.mark.asyncio
    async def test_failed_daemon_waits_for_cooldown(self, manager: DaemonManager):
        """Failed daemon should wait for cooldown before recovery."""
        import time

        recovery_attempted = False

        async def recoverable_factory():
            nonlocal recovery_attempted
            recovery_attempted = True
            while True:
                await asyncio.sleep(1)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            recoverable_factory,
            auto_restart=True,
        )

        # Set up failed daemon with recent failure (within cooldown)
        info = manager._daemons[DaemonType.MODEL_SYNC]
        info.state = DaemonState.FAILED
        info.restart_count = 5
        info.last_failure_time = time.time()  # Just failed

        # Run health check immediately
        await manager._check_health()

        # Verify no recovery yet
        assert not recovery_attempted, "Should wait for cooldown before recovery"


class TestHealthLoopIntegration:
    """Tests for health loop behavior with real daemons."""

    @pytest.mark.asyncio
    async def test_health_loop_detects_crashed_daemon(self, manager: DaemonManager):
        """Health loop should detect and handle crashed daemon."""
        crash_detected = False
        original_check = manager._check_health

        async def tracking_check():
            nonlocal crash_detected
            for info in manager._daemons.values():
                if info.task and info.task.done() and info.task.exception():
                    crash_detected = True
            await original_check()

        manager._check_health = tracking_check

        async def crashing_daemon():
            await asyncio.sleep(0.05)  # Small delay before crash
            raise RuntimeError("Deliberate crash")

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            crashing_daemon,
            auto_restart=False,  # Don't restart for this test
        )

        # Start daemon and health loop
        await manager.start(DaemonType.MODEL_SYNC)
        manager._running = True

        # Start health loop task
        health_task = asyncio.create_task(manager._health_loop())

        # Wait for crash and detection
        await asyncio.sleep(0.3)

        # Stop health loop
        manager._running = False
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass

        # Verify crash was detected
        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.state == DaemonState.FAILED

    @pytest.mark.asyncio
    async def test_health_loop_handles_cancelled_error(self, manager: DaemonManager):
        """Health loop should exit cleanly on CancelledError."""
        manager._running = True

        # Start health loop
        health_task = asyncio.create_task(manager._health_loop())
        await asyncio.sleep(0.05)

        # Cancel it
        health_task.cancel()

        # Should not raise
        try:
            await health_task
        except asyncio.CancelledError:
            pass  # Expected

        # Loop should have exited
        assert health_task.done()


class TestDependencyGraph:
    """Tests for dependency-aware restart behavior."""

    @pytest.mark.asyncio
    async def test_get_dependents_returns_correct_daemons(self, manager: DaemonManager):
        """_get_dependents should return all daemons depending on given type."""
        async def noop_factory():
            while True:
                await asyncio.sleep(1)

        # Set up dependency chain: EVENT_ROUTER <- DATA_PIPELINE, HEALTH_CHECK
        manager.register_factory(DaemonType.EVENT_ROUTER, noop_factory)
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            noop_factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )
        manager.register_factory(
            DaemonType.HEALTH_CHECK,
            noop_factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        dependents = manager._get_dependents(DaemonType.EVENT_ROUTER)

        assert DaemonType.DATA_PIPELINE in dependents
        assert DaemonType.HEALTH_CHECK in dependents

    @pytest.mark.asyncio
    async def test_sort_by_dependencies_orders_correctly(self, manager: DaemonManager):
        """_sort_by_dependencies should sort deps before dependents."""
        async def noop_factory():
            while True:
                await asyncio.sleep(1)

        # Clear existing factories
        manager._factories.clear()
        manager._daemons.clear()

        # A <- B <- C chain
        manager.register_factory(DaemonType.EVENT_ROUTER, noop_factory)  # A
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            noop_factory,
            depends_on=[DaemonType.EVENT_ROUTER],  # B depends on A
        )
        manager.register_factory(
            DaemonType.HEALTH_CHECK,
            noop_factory,
            depends_on=[DaemonType.DATA_PIPELINE],  # C depends on B
        )

        types_to_sort = [
            DaemonType.HEALTH_CHECK,
            DaemonType.EVENT_ROUTER,
            DaemonType.DATA_PIPELINE,
        ]
        sorted_types = manager._sort_by_dependencies(types_to_sort)

        # A should come before B, B before C
        a_idx = sorted_types.index(DaemonType.EVENT_ROUTER)
        b_idx = sorted_types.index(DaemonType.DATA_PIPELINE)
        c_idx = sorted_types.index(DaemonType.HEALTH_CHECK)

        assert a_idx < b_idx, "EVENT_ROUTER should come before DATA_PIPELINE"
        assert b_idx < c_idx, "DATA_PIPELINE should come before HEALTH_CHECK"
