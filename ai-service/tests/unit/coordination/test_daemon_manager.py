"""Tests for app.coordination.daemon_manager - Unified Daemon Manager.

This module tests the DaemonManager which coordinates lifecycle of all
background services including sync daemons, health checks, and event watchers.

Note: Async daemon lifecycle tests are limited to avoid timeout issues with
long-running background tasks.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_manager import (
    DaemonInfo,
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    get_daemon_manager,
    reset_daemon_manager,
)


# =============================================================================
# DaemonType Tests
# =============================================================================


class TestDaemonType:
    """Tests for DaemonType enum."""

    def test_sync_daemons_defined(self):
        """Sync daemon types should be defined."""
        assert DaemonType.SYNC_COORDINATOR.value == "sync_coordinator"
        assert DaemonType.HIGH_QUALITY_SYNC.value == "high_quality_sync"
        assert DaemonType.ELO_SYNC.value == "elo_sync"
        assert DaemonType.MODEL_SYNC.value == "model_sync"

    def test_monitoring_daemons_defined(self):
        """Monitoring daemon types should be defined."""
        assert DaemonType.HEALTH_CHECK.value == "health_check"
        assert DaemonType.CLUSTER_MONITOR.value == "cluster_monitor"
        assert DaemonType.QUEUE_MONITOR.value == "queue_monitor"

    def test_event_daemons_defined(self):
        """Event processing daemon types should be defined."""
        assert DaemonType.EVENT_ROUTER.value == "event_router"
        assert DaemonType.CROSS_PROCESS_POLLER.value == "cross_process_poller"

    def test_pipeline_daemons_defined(self):
        """Pipeline daemon types should be defined."""
        assert DaemonType.DATA_PIPELINE.value == "data_pipeline"
        assert DaemonType.TRAINING_NODE_WATCHER.value == "training_node_watcher"

    def test_p2p_daemons_defined(self):
        """P2P service daemon types should be defined."""
        assert DaemonType.P2P_BACKEND.value == "p2p_backend"
        assert DaemonType.GOSSIP_SYNC.value == "gossip_sync"
        assert DaemonType.DATA_SERVER.value == "data_server"

    def test_daemon_count(self):
        """Should have expected number of daemon types."""
        assert len(DaemonType) >= 16


# =============================================================================
# DaemonState Tests
# =============================================================================


class TestDaemonState:
    """Tests for DaemonState enum."""

    def test_all_states_defined(self):
        """All states should be defined."""
        assert DaemonState.STOPPED.value == "stopped"
        assert DaemonState.STARTING.value == "starting"
        assert DaemonState.RUNNING.value == "running"
        assert DaemonState.STOPPING.value == "stopping"
        assert DaemonState.FAILED.value == "failed"
        assert DaemonState.RESTARTING.value == "restarting"
        assert DaemonState.IMPORT_FAILED.value == "import_failed"

    def test_state_count(self):
        """Should have exactly 7 states."""
        assert len(DaemonState) == 7


# =============================================================================
# DaemonInfo Tests
# =============================================================================


class TestDaemonInfo:
    """Tests for DaemonInfo dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        info = DaemonInfo(daemon_type=DaemonType.HEALTH_CHECK)

        assert info.daemon_type == DaemonType.HEALTH_CHECK
        assert info.state == DaemonState.STOPPED
        assert info.task is None
        assert info.start_time == 0.0
        assert info.restart_count == 0
        assert info.last_error is None
        assert info.health_check_interval == 60.0
        assert info.auto_restart is True
        assert info.max_restarts == 5
        assert info.restart_delay == 5.0
        assert info.depends_on == []

    def test_uptime_when_stopped(self):
        """uptime_seconds should be 0 when stopped."""
        info = DaemonInfo(daemon_type=DaemonType.HEALTH_CHECK)
        assert info.uptime_seconds == 0.0

    def test_uptime_when_running(self):
        """uptime_seconds should calculate correctly when running."""
        info = DaemonInfo(
            daemon_type=DaemonType.HEALTH_CHECK,
            state=DaemonState.RUNNING,
            start_time=time.time() - 10.0,
        )
        assert 9.5 < info.uptime_seconds < 11.0

    def test_uptime_with_future_start_time(self):
        """uptime_seconds should handle edge cases."""
        info = DaemonInfo(
            daemon_type=DaemonType.HEALTH_CHECK,
            state=DaemonState.STOPPED,  # Not running
            start_time=time.time() - 10.0,
        )
        # Not running, so uptime should be 0
        assert info.uptime_seconds == 0.0

    def test_dependencies_can_be_set(self):
        """Should accept dependency list."""
        info = DaemonInfo(
            daemon_type=DaemonType.DATA_PIPELINE,
            depends_on=[DaemonType.EVENT_ROUTER, DaemonType.SYNC_COORDINATOR],
        )
        assert len(info.depends_on) == 2
        assert DaemonType.EVENT_ROUTER in info.depends_on


# =============================================================================
# DaemonManagerConfig Tests
# =============================================================================


class TestDaemonManagerConfig:
    """Tests for DaemonManagerConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = DaemonManagerConfig()

        assert config.auto_start is False
        assert config.health_check_interval == 30.0
        assert config.shutdown_timeout == 10.0
        assert config.auto_restart_failed is True
        assert config.max_restart_attempts == 5

    def test_custom_config(self):
        """Should accept custom values."""
        config = DaemonManagerConfig(
            auto_start=True,
            health_check_interval=60.0,
            shutdown_timeout=30.0,
        )

        assert config.auto_start is True
        assert config.health_check_interval == 60.0
        assert config.shutdown_timeout == 30.0


# =============================================================================
# DaemonManager Init Tests
# =============================================================================


class TestDaemonManagerInit:
    """Tests for DaemonManager initialization."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_init_with_default_config(self):
        """Should initialize with default config."""
        manager = DaemonManager()

        assert manager.config is not None
        assert manager._running is False
        assert manager._health_task is None

    def test_init_with_custom_config(self):
        """Should accept custom config."""
        config = DaemonManagerConfig(
            health_check_interval=1.0,
            shutdown_timeout=2.0,
        )
        manager = DaemonManager(config)

        assert manager.config.health_check_interval == 1.0
        assert manager.config.shutdown_timeout == 2.0

    def test_default_factories_registered(self):
        """Default factories should be registered."""
        manager = DaemonManager()

        assert DaemonType.SYNC_COORDINATOR in manager._factories
        assert DaemonType.EVENT_ROUTER in manager._factories
        assert DaemonType.HEALTH_CHECK in manager._factories


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_get_instance_creates_singleton(self):
        """get_instance should create singleton."""
        config = DaemonManagerConfig()

        manager1 = DaemonManager.get_instance(config)
        manager2 = DaemonManager.get_instance()

        assert manager1 is manager2

    def test_reset_instance_clears_singleton(self):
        """reset_instance should clear singleton."""
        config = DaemonManagerConfig()

        manager1 = DaemonManager.get_instance(config)
        DaemonManager.reset_instance()
        manager2 = DaemonManager.get_instance(config)

        assert manager1 is not manager2

    def test_get_daemon_manager_function(self):
        """get_daemon_manager should return singleton."""
        config = DaemonManagerConfig()

        manager1 = get_daemon_manager(config)
        manager2 = get_daemon_manager()

        assert manager1 is manager2


# =============================================================================
# Factory Registration Tests
# =============================================================================


class TestFactoryRegistration:
    """Tests for daemon factory registration."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()

    def test_register_simple_factory(self):
        """Should register a simple factory."""
        manager = DaemonManager()

        async def my_factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, my_factory)

        assert DaemonType.MODEL_SYNC in manager._factories
        assert DaemonType.MODEL_SYNC in manager._daemons

    def test_register_factory_with_dependencies(self):
        """Should register factory with dependencies."""
        manager = DaemonManager()

        async def my_factory():
            pass

        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            my_factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        info = manager._daemons[DaemonType.DATA_PIPELINE]
        assert DaemonType.EVENT_ROUTER in info.depends_on

    def test_register_factory_with_config(self):
        """Should register factory with custom config."""
        manager = DaemonManager()

        async def my_factory():
            pass

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            my_factory,
            health_check_interval=120.0,
            auto_restart=False,
            max_restarts=10,
        )

        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.health_check_interval == 120.0
        assert info.auto_restart is False
        assert info.max_restarts == 10


# =============================================================================
# Dependency Sorting Tests
# =============================================================================


class TestDependencySorting:
    """Tests for dependency-based sorting."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()

    def test_sort_by_dependencies(self):
        """Dependencies should be sorted correctly."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        result = manager._sort_by_dependencies([
            DaemonType.DATA_PIPELINE,
            DaemonType.EVENT_ROUTER,
        ])

        # EVENT_ROUTER should come before DATA_PIPELINE
        er_idx = result.index(DaemonType.EVENT_ROUTER)
        dp_idx = result.index(DaemonType.DATA_PIPELINE)
        assert er_idx < dp_idx

    def test_sort_handles_no_deps(self):
        """Should handle daemons with no dependencies."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, factory)
        manager.register_factory(DaemonType.ELO_SYNC, factory)

        result = manager._sort_by_dependencies([
            DaemonType.MODEL_SYNC,
            DaemonType.ELO_SYNC,
        ])

        assert len(result) == 2
        assert DaemonType.MODEL_SYNC in result
        assert DaemonType.ELO_SYNC in result

    def test_sort_circular_dependency(self):
        """Should handle circular dependencies gracefully."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager._daemons[DaemonType.MODEL_SYNC] = DaemonInfo(
            daemon_type=DaemonType.MODEL_SYNC,
            depends_on=[DaemonType.ELO_SYNC],
        )
        manager._daemons[DaemonType.ELO_SYNC] = DaemonInfo(
            daemon_type=DaemonType.ELO_SYNC,
            depends_on=[DaemonType.MODEL_SYNC],
        )
        manager._factories[DaemonType.MODEL_SYNC] = factory
        manager._factories[DaemonType.ELO_SYNC] = factory

        # Should not hang or crash
        result = manager._sort_by_dependencies([
            DaemonType.MODEL_SYNC,
            DaemonType.ELO_SYNC,
        ])

        assert len(result) == 2

    def test_sort_with_chain_dependencies(self):
        """Should handle A -> B -> C dependency chains."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.SYNC_COORDINATOR, factory)
        manager.register_factory(
            DaemonType.EVENT_ROUTER,
            factory,
            depends_on=[DaemonType.SYNC_COORDINATOR],
        )
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        result = manager._sort_by_dependencies([
            DaemonType.DATA_PIPELINE,
            DaemonType.SYNC_COORDINATOR,
            DaemonType.EVENT_ROUTER,
        ])

        sc_idx = result.index(DaemonType.SYNC_COORDINATOR)
        er_idx = result.index(DaemonType.EVENT_ROUTER)
        dp_idx = result.index(DaemonType.DATA_PIPELINE)

        assert sc_idx < er_idx < dp_idx


# =============================================================================
# Status Tests
# =============================================================================


class TestStatus:
    """Tests for status reporting."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()

    def test_get_status_empty(self):
        """Should return status for empty manager."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        status = manager.get_status()

        assert status["running"] is False
        assert status["daemons"] == {}
        assert status["summary"]["total"] == 0
        assert status["summary"]["running"] == 0

    def test_get_status_with_daemons(self):
        """Should return status for registered daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, factory)

        status = manager.get_status()

        assert "model_sync" in status["daemons"]
        assert status["daemons"]["model_sync"]["state"] == "stopped"
        assert status["summary"]["total"] == 1
        assert status["summary"]["stopped"] == 1

    def test_get_status_counts_states(self):
        """Should correctly count daemon states."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        # Add daemons in different states
        manager._daemons[DaemonType.MODEL_SYNC] = DaemonInfo(
            daemon_type=DaemonType.MODEL_SYNC,
            state=DaemonState.RUNNING,
        )
        manager._daemons[DaemonType.ELO_SYNC] = DaemonInfo(
            daemon_type=DaemonType.ELO_SYNC,
            state=DaemonState.FAILED,
        )
        manager._daemons[DaemonType.HEALTH_CHECK] = DaemonInfo(
            daemon_type=DaemonType.HEALTH_CHECK,
            state=DaemonState.STOPPED,
        )

        status = manager.get_status()

        assert status["summary"]["total"] == 3
        assert status["summary"]["running"] == 1
        assert status["summary"]["failed"] == 1
        assert status["summary"]["stopped"] == 1

    def test_is_running_false_for_stopped(self):
        """is_running should return False for stopped daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, factory)
        assert manager.is_running(DaemonType.MODEL_SYNC) is False

    def test_is_running_true_for_running(self):
        """is_running should return True for running daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager._daemons[DaemonType.MODEL_SYNC] = DaemonInfo(
            daemon_type=DaemonType.MODEL_SYNC,
            state=DaemonState.RUNNING,
        )

        assert manager.is_running(DaemonType.MODEL_SYNC) is True

    def test_is_running_unknown_daemon(self):
        """is_running should return False for unknown daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        assert manager.is_running(DaemonType.MODEL_SYNC) is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()

    def test_sync_shutdown_no_loop(self):
        """_sync_shutdown should handle no running loop."""
        manager = DaemonManager()
        # Should not raise
        manager._sync_shutdown()
