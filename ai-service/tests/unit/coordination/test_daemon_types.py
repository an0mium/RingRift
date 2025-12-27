"""Tests for app.coordination.daemon_types module.

Tests the daemon type definitions and data structures:
- DaemonType enum
- DaemonState enum
- DaemonInfo dataclass
- DaemonManagerConfig dataclass
- CRITICAL_DAEMONS set

Created Dec 2025 as part of Phase 3 test coverage improvement.
"""

import asyncio
import time

import pytest

from app.coordination.daemon_types import (
    CRITICAL_DAEMONS,
    DAEMON_RESTART_RESET_AFTER,
    MAX_RESTART_DELAY,
    DaemonInfo,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
)


# =============================================================================
# DaemonType Enum Tests
# =============================================================================


class TestDaemonType:
    """Tests for DaemonType enum."""

    def test_daemon_type_count(self):
        """Verify expected number of daemon types."""
        # Should have 50+ daemon types
        assert len(DaemonType) >= 50, f"Expected 50+ daemon types, got {len(DaemonType)}"

    def test_core_daemon_types_exist(self):
        """Verify core daemon types are defined."""
        core_types = [
            "SYNC_COORDINATOR",
            "EVENT_ROUTER",
            "CLUSTER_MONITOR",
            "AUTO_SYNC",
            "DATA_PIPELINE",
            "P2P_BACKEND",
            "MODEL_DISTRIBUTION",
            "FEEDBACK_LOOP",
            "IDLE_RESOURCE",
            "QUEUE_POPULATOR",
        ]
        for name in core_types:
            assert hasattr(DaemonType, name), f"Missing DaemonType.{name}"

    def test_daemon_types_have_unique_values(self):
        """Verify all daemon types have unique values."""
        values = [dt.value for dt in DaemonType]
        assert len(values) == len(set(values)), "Duplicate daemon type values"

    def test_daemon_type_values_are_strings(self):
        """Verify all daemon type values are strings."""
        for dt in DaemonType:
            assert isinstance(dt.value, str), f"{dt.name}.value should be string"

    def test_daemon_type_december_2025_additions(self):
        """Verify December 2025 daemon types exist."""
        dec_2025_types = [
            "SYSTEM_HEALTH_MONITOR",
            "HEALTH_SERVER",
            "MAINTENANCE",
            "UTILIZATION_OPTIMIZER",
            "CLUSTER_WATCHDOG",
            "CURRICULUM_INTEGRATION",
            "AUTO_EXPORT",
            "TRAINING_TRIGGER",
        ]
        for name in dec_2025_types:
            assert hasattr(DaemonType, name), f"Missing Dec 2025 DaemonType.{name}"


# =============================================================================
# DaemonState Enum Tests
# =============================================================================


class TestDaemonState:
    """Tests for DaemonState enum."""

    def test_all_states_exist(self):
        """Verify all expected daemon states are defined."""
        expected_states = [
            "STOPPED",
            "STARTING",
            "RUNNING",
            "STOPPING",
            "FAILED",
            "RESTARTING",
            "IMPORT_FAILED",
        ]
        for name in expected_states:
            assert hasattr(DaemonState, name), f"Missing DaemonState.{name}"

    def test_state_values_are_strings(self):
        """Verify all state values are strings."""
        for state in DaemonState:
            assert isinstance(state.value, str), f"{state.name}.value should be string"


# =============================================================================
# DaemonInfo Tests
# =============================================================================


class TestDaemonInfo:
    """Tests for DaemonInfo dataclass."""

    def test_create_with_required_fields(self):
        """Test creating DaemonInfo with only required fields."""
        info = DaemonInfo(daemon_type=DaemonType.EVENT_ROUTER)
        assert info.daemon_type == DaemonType.EVENT_ROUTER
        assert info.state == DaemonState.STOPPED
        assert info.task is None
        assert info.restart_count == 0
        assert info.auto_restart is True
        assert info.max_restarts == 5

    def test_create_with_all_fields(self):
        """Test creating DaemonInfo with all fields."""
        info = DaemonInfo(
            daemon_type=DaemonType.AUTO_SYNC,
            state=DaemonState.RUNNING,
            start_time=time.time(),
            restart_count=2,
            last_error="Connection timeout",
            health_check_interval=30.0,
            auto_restart=True,
            max_restarts=10,
            restart_delay=10.0,
            depends_on=[DaemonType.EVENT_ROUTER],
        )
        assert info.state == DaemonState.RUNNING
        assert info.restart_count == 2
        assert info.last_error == "Connection timeout"
        assert len(info.depends_on) == 1

    def test_uptime_seconds_when_running(self):
        """Test uptime_seconds property when daemon is running."""
        info = DaemonInfo(
            daemon_type=DaemonType.CLUSTER_MONITOR,
            state=DaemonState.RUNNING,
            start_time=time.time() - 60.0,  # Started 60 seconds ago
        )
        uptime = info.uptime_seconds
        assert 59.0 <= uptime <= 61.0  # Allow for timing variance

    def test_uptime_seconds_when_stopped(self):
        """Test uptime_seconds returns 0 when daemon is stopped."""
        info = DaemonInfo(
            daemon_type=DaemonType.CLUSTER_MONITOR,
            state=DaemonState.STOPPED,
            start_time=time.time() - 60.0,
        )
        assert info.uptime_seconds == 0.0

    def test_uptime_seconds_when_no_start_time(self):
        """Test uptime_seconds returns 0 when no start time."""
        info = DaemonInfo(
            daemon_type=DaemonType.CLUSTER_MONITOR,
            state=DaemonState.RUNNING,
            start_time=0.0,
        )
        assert info.uptime_seconds == 0.0

    def test_ready_event_default(self):
        """Test ready_event defaults to None."""
        info = DaemonInfo(daemon_type=DaemonType.EVENT_ROUTER)
        assert info.ready_event is None

    def test_import_error_tracking(self):
        """Test import error tracking field."""
        info = DaemonInfo(
            daemon_type=DaemonType.DISTILLATION,
            state=DaemonState.IMPORT_FAILED,
            import_error="ModuleNotFoundError: No module named 'special_lib'",
        )
        assert info.state == DaemonState.IMPORT_FAILED
        assert "ModuleNotFoundError" in info.import_error

    def test_dependencies_list(self):
        """Test daemon dependencies list."""
        info = DaemonInfo(
            daemon_type=DaemonType.MODEL_DISTRIBUTION,
            depends_on=[
                DaemonType.EVENT_ROUTER,
                DaemonType.P2P_BACKEND,
            ],
        )
        assert len(info.depends_on) == 2
        assert DaemonType.EVENT_ROUTER in info.depends_on


# =============================================================================
# DaemonManagerConfig Tests
# =============================================================================


class TestDaemonManagerConfig:
    """Tests for DaemonManagerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DaemonManagerConfig()
        assert config.auto_start is False
        assert config.health_check_interval == 30.0
        assert config.shutdown_timeout == 10.0
        assert config.auto_restart_failed is True
        assert config.max_restart_attempts == 5
        assert config.recovery_cooldown == 10.0
        assert config.critical_daemon_health_interval == 15.0

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = DaemonManagerConfig(
            auto_start=True,
            health_check_interval=60.0,
            shutdown_timeout=30.0,
            max_restart_attempts=10,
        )
        assert config.auto_start is True
        assert config.health_check_interval == 60.0
        assert config.shutdown_timeout == 30.0
        assert config.max_restart_attempts == 10


# =============================================================================
# CRITICAL_DAEMONS Tests
# =============================================================================


class TestCriticalDaemons:
    """Tests for CRITICAL_DAEMONS set."""

    def test_critical_daemons_is_set(self):
        """Verify CRITICAL_DAEMONS is a set."""
        assert isinstance(CRITICAL_DAEMONS, set)

    def test_critical_daemons_not_empty(self):
        """Verify CRITICAL_DAEMONS is not empty."""
        assert len(CRITICAL_DAEMONS) > 0

    def test_critical_daemons_contains_expected(self):
        """Verify expected daemons are in CRITICAL_DAEMONS."""
        # These are the actual critical daemons as defined in daemon_types.py
        expected = [
            DaemonType.QUEUE_POPULATOR,
            DaemonType.EVENT_ROUTER,
            DaemonType.IDLE_RESOURCE,
            DaemonType.AUTO_SYNC,
            DaemonType.FEEDBACK_LOOP,
        ]
        for daemon in expected:
            assert daemon in CRITICAL_DAEMONS, f"{daemon.name} should be critical"

    def test_critical_daemons_are_daemon_types(self):
        """Verify all items in CRITICAL_DAEMONS are DaemonType."""
        for daemon in CRITICAL_DAEMONS:
            assert isinstance(daemon, DaemonType)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_restart_delay_is_positive(self):
        """Verify MAX_RESTART_DELAY is positive."""
        assert MAX_RESTART_DELAY > 0

    def test_daemon_restart_reset_after_is_positive(self):
        """Verify DAEMON_RESTART_RESET_AFTER is positive."""
        assert DAEMON_RESTART_RESET_AFTER > 0

    def test_max_restart_delay_reasonable(self):
        """Verify MAX_RESTART_DELAY is a reasonable value."""
        # Should be between 1 minute and 1 hour
        assert 60 <= MAX_RESTART_DELAY <= 3600

    def test_daemon_restart_reset_after_reasonable(self):
        """Verify DAEMON_RESTART_RESET_AFTER is reasonable."""
        # Should be between 5 minutes and 24 hours
        assert 300 <= DAEMON_RESTART_RESET_AFTER <= 86400


# =============================================================================
# Import Compatibility Tests
# =============================================================================


class TestImportCompatibility:
    """Tests for backward compatibility imports."""

    def test_import_from_daemon_manager(self):
        """Verify types can be imported from daemon_manager."""
        from app.coordination.daemon_manager import (
            CRITICAL_DAEMONS,
            DaemonInfo,
            DaemonManagerConfig,
            DaemonState,
            DaemonType,
        )
        assert DaemonType is not None
        assert DaemonState is not None
        assert DaemonInfo is not None
        assert DaemonManagerConfig is not None
        assert CRITICAL_DAEMONS is not None

    def test_import_from_daemon_types(self):
        """Verify types can be imported from daemon_types."""
        from app.coordination.daemon_types import (
            CRITICAL_DAEMONS,
            DaemonInfo,
            DaemonManagerConfig,
            DaemonState,
            DaemonType,
        )
        assert DaemonType is not None
        assert DaemonState is not None
        assert DaemonInfo is not None
        assert DaemonManagerConfig is not None
        assert CRITICAL_DAEMONS is not None
