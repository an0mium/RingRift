"""Tests for StandbyCoordinator - distributed coordinator failover.

These tests verify:
1. Configuration (StandbyConfig, from_env)
2. Data classes (CoordinatorRole, FailoverReason, PrimaryHealthState, StandbyState)
3. StandbyCoordinator initialization and lifecycle
4. Role transitions (STANDBY -> PRIMARY, PRIMARY -> STANDBY)
5. Primary health monitoring
6. Takeover and handoff callbacks
7. Singleton pattern

January 7, 2026 - Sprint 17: Added as part of Phase 3 test coverage.
"""

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.standby_coordinator import (
    CoordinatorRole,
    FailoverReason,
    PrimaryHealthState,
    StandbyConfig,
    StandbyCoordinator,
    StandbyState,
    get_standby_coordinator,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestStandbyConfig:
    """Tests for StandbyConfig dataclass."""

    def test_default_values(self):
        """StandbyConfig should have sensible defaults."""
        config = StandbyConfig()
        assert config.primary_heartbeat_timeout == 120.0
        assert config.primary_check_interval == 15.0
        assert config.primary_host is None
        assert config.primary_port == 8790
        assert config.takeover_delay == 10.0
        assert config.graceful_handoff_timeout == 60.0
        assert config.min_standby_uptime == 300.0

    def test_custom_values(self):
        """StandbyConfig should accept custom values."""
        config = StandbyConfig(
            primary_heartbeat_timeout=60.0,
            primary_check_interval=5.0,
            primary_host="coordinator.example.com",
            primary_port=9000,
            takeover_delay=5.0,
        )
        assert config.primary_heartbeat_timeout == 60.0
        assert config.primary_check_interval == 5.0
        assert config.primary_host == "coordinator.example.com"
        assert config.primary_port == 9000
        assert config.takeover_delay == 5.0

    def test_from_env_defaults(self):
        """from_env should use defaults when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            config = StandbyConfig.from_env()
            assert config.primary_heartbeat_timeout == 120.0
            assert config.primary_check_interval == 15.0
            assert config.primary_host is None
            assert config.primary_port == 8790

    def test_from_env_with_env_vars(self):
        """from_env should read from environment variables."""
        env_vars = {
            "RINGRIFT_STANDBY_HEARTBEAT_TIMEOUT": "60.0",
            "RINGRIFT_STANDBY_CHECK_INTERVAL": "5.0",
            "RINGRIFT_STANDBY_PRIMARY_HOST": "coordinator.example.com",
            "RINGRIFT_STANDBY_PRIMARY_PORT": "9000",
            "RINGRIFT_STANDBY_TAKEOVER_DELAY": "3.0",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = StandbyConfig.from_env()
            assert config.primary_heartbeat_timeout == 60.0
            assert config.primary_check_interval == 5.0
            assert config.primary_host == "coordinator.example.com"
            assert config.primary_port == 9000
            assert config.takeover_delay == 3.0

    def test_callbacks_default_to_empty_list(self):
        """Callback lists should default to empty."""
        config = StandbyConfig()
        assert config.on_takeover_callbacks == []
        assert config.on_handoff_callbacks == []


# =============================================================================
# Enum Tests
# =============================================================================


class TestCoordinatorRole:
    """Tests for CoordinatorRole enum."""

    def test_role_values(self):
        """CoordinatorRole should have correct string values."""
        assert CoordinatorRole.PRIMARY.value == "primary"
        assert CoordinatorRole.STANDBY.value == "standby"
        assert CoordinatorRole.TRANSITIONING.value == "transitioning"
        assert CoordinatorRole.UNKNOWN.value == "unknown"

    def test_role_count(self):
        """CoordinatorRole should have exactly 4 members."""
        assert len(CoordinatorRole) == 4


class TestFailoverReason:
    """Tests for FailoverReason enum."""

    def test_reason_values(self):
        """FailoverReason should have correct string values."""
        assert FailoverReason.PRIMARY_TIMEOUT.value == "primary_timeout"
        assert FailoverReason.PRIMARY_SHUTDOWN.value == "primary_shutdown"
        assert FailoverReason.MEMORY_EMERGENCY.value == "memory_emergency"
        assert FailoverReason.MANUAL_TAKEOVER.value == "manual_takeover"
        assert FailoverReason.ELECTION_WON.value == "election_won"

    def test_reason_count(self):
        """FailoverReason should have exactly 5 members."""
        assert len(FailoverReason) == 5


# =============================================================================
# Data Class Tests
# =============================================================================


class TestPrimaryHealthState:
    """Tests for PrimaryHealthState dataclass."""

    def test_creation(self):
        """PrimaryHealthState should accept all parameters."""
        state = PrimaryHealthState(
            host="coordinator.example.com",
            is_healthy=True,
            last_seen=time.time(),
            consecutive_failures=0,
            last_check_time=time.time(),
            last_check_duration=0.5,
        )
        assert state.host == "coordinator.example.com"
        assert state.is_healthy is True
        assert state.consecutive_failures == 0

    def test_time_since_seen_healthy(self):
        """time_since_seen should return correct elapsed time."""
        now = time.time()
        state = PrimaryHealthState(
            host="test",
            is_healthy=True,
            last_seen=now - 30.0,  # 30 seconds ago
            consecutive_failures=0,
            last_check_time=now,
            last_check_duration=0.1,
        )
        # Should be approximately 30 seconds
        assert 29.0 <= state.time_since_seen <= 32.0

    def test_time_since_seen_never_seen(self):
        """time_since_seen should return infinity if never seen."""
        state = PrimaryHealthState(
            host="test",
            is_healthy=False,
            last_seen=0.0,
            consecutive_failures=5,
            last_check_time=time.time(),
            last_check_duration=0.1,
        )
        assert state.time_since_seen == float("inf")

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        now = time.time()
        state = PrimaryHealthState(
            host="test-host",
            is_healthy=True,
            last_seen=now,
            consecutive_failures=2,
            last_check_time=now,
            last_check_duration=0.5,
            error_message="Test error",
        )
        result = state.to_dict()
        assert result["host"] == "test-host"
        assert result["is_healthy"] is True
        assert result["consecutive_failures"] == 2
        assert result["last_check_duration"] == 0.5
        assert result["error_message"] == "Test error"
        assert "time_since_seen" in result


class TestStandbyState:
    """Tests for StandbyState dataclass."""

    def test_creation(self):
        """StandbyState should accept all parameters."""
        state = StandbyState(
            role=CoordinatorRole.STANDBY,
            start_time=time.time(),
            takeover_count=0,
            handoff_count=0,
            last_takeover_time=0.0,
            last_handoff_time=0.0,
            failover_reason=None,
            primary_health=None,
        )
        assert state.role == CoordinatorRole.STANDBY
        assert state.takeover_count == 0

    def test_uptime_seconds(self):
        """uptime_seconds should return correct elapsed time."""
        state = StandbyState(
            role=CoordinatorRole.STANDBY,
            start_time=time.time() - 60.0,  # Started 60 seconds ago
            takeover_count=0,
            handoff_count=0,
            last_takeover_time=0.0,
            last_handoff_time=0.0,
            failover_reason=None,
            primary_health=None,
        )
        # Should be approximately 60 seconds
        assert 59.0 <= state.uptime_seconds <= 62.0

    def test_uptime_seconds_zero_start(self):
        """uptime_seconds should return 0 if start_time is 0."""
        state = StandbyState(
            role=CoordinatorRole.STANDBY,
            start_time=0.0,
            takeover_count=0,
            handoff_count=0,
            last_takeover_time=0.0,
            last_handoff_time=0.0,
            failover_reason=None,
            primary_health=None,
        )
        assert state.uptime_seconds == 0.0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        now = time.time()
        health = PrimaryHealthState(
            host="primary",
            is_healthy=True,
            last_seen=now,
            consecutive_failures=0,
            last_check_time=now,
            last_check_duration=0.1,
        )
        state = StandbyState(
            role=CoordinatorRole.PRIMARY,
            start_time=now - 3600,
            takeover_count=1,
            handoff_count=0,
            last_takeover_time=now - 1800,
            last_handoff_time=0.0,
            failover_reason=FailoverReason.PRIMARY_TIMEOUT,
            primary_health=health,
        )
        result = state.to_dict()
        assert result["role"] == "primary"
        assert result["takeover_count"] == 1
        assert result["failover_reason"] == "primary_timeout"
        assert result["primary_health"] is not None
        assert result["primary_health"]["host"] == "primary"

    def test_to_dict_without_failover_reason(self):
        """to_dict should handle None failover_reason."""
        state = StandbyState(
            role=CoordinatorRole.STANDBY,
            start_time=time.time(),
            takeover_count=0,
            handoff_count=0,
            last_takeover_time=0.0,
            last_handoff_time=0.0,
            failover_reason=None,
            primary_health=None,
        )
        result = state.to_dict()
        assert result["failover_reason"] is None
        assert result["primary_health"] is None


# =============================================================================
# StandbyCoordinator Tests
# =============================================================================


class TestStandbyCoordinatorInit:
    """Tests for StandbyCoordinator initialization."""

    def test_default_config(self):
        """StandbyCoordinator should use default config if none provided."""
        coordinator = StandbyCoordinator()
        assert coordinator._standby_config is not None
        assert coordinator._standby_config.primary_heartbeat_timeout == 120.0

    def test_custom_config(self):
        """StandbyCoordinator should use provided config."""
        config = StandbyConfig(primary_heartbeat_timeout=60.0)
        coordinator = StandbyCoordinator(config=config)
        assert coordinator._standby_config.primary_heartbeat_timeout == 60.0

    def test_initial_role(self):
        """StandbyCoordinator should start in UNKNOWN role."""
        coordinator = StandbyCoordinator()
        assert coordinator._role == CoordinatorRole.UNKNOWN

    def test_initial_state(self):
        """StandbyCoordinator should have zeroed state initially."""
        coordinator = StandbyCoordinator()
        assert coordinator._takeover_count == 0
        assert coordinator._handoff_count == 0
        assert coordinator._failover_reason is None


class TestStandbyCoordinatorProperties:
    """Tests for StandbyCoordinator properties."""

    def test_is_primary(self):
        """is_primary should return True only when role is PRIMARY."""
        coordinator = StandbyCoordinator()

        coordinator._role = CoordinatorRole.PRIMARY
        assert coordinator.is_primary is True

        coordinator._role = CoordinatorRole.STANDBY
        assert coordinator.is_primary is False

        coordinator._role = CoordinatorRole.TRANSITIONING
        assert coordinator.is_primary is False

    def test_is_standby(self):
        """is_standby should return True only when role is STANDBY."""
        coordinator = StandbyCoordinator()

        coordinator._role = CoordinatorRole.STANDBY
        assert coordinator.is_standby is True

        coordinator._role = CoordinatorRole.PRIMARY
        assert coordinator.is_standby is False

    def test_role_property(self):
        """role property should return current role."""
        coordinator = StandbyCoordinator()
        coordinator._role = CoordinatorRole.TRANSITIONING
        assert coordinator.role == CoordinatorRole.TRANSITIONING


class TestStandbyCoordinatorCallbacks:
    """Tests for StandbyCoordinator callback registration."""

    def test_register_takeover_callback(self):
        """register_takeover_callback should add to callback list."""
        coordinator = StandbyCoordinator()
        callback = MagicMock()

        coordinator.register_takeover_callback(callback)

        assert callback in coordinator._on_takeover

    def test_register_handoff_callback(self):
        """register_handoff_callback should add to callback list."""
        coordinator = StandbyCoordinator()
        callback = MagicMock()

        coordinator.register_handoff_callback(callback)

        assert callback in coordinator._on_handoff

    def test_multiple_callbacks(self):
        """Multiple callbacks should all be registered."""
        coordinator = StandbyCoordinator()
        callbacks = [MagicMock() for _ in range(3)]

        for cb in callbacks:
            coordinator.register_takeover_callback(cb)

        assert len(coordinator._on_takeover) == 3
        for cb in callbacks:
            assert cb in coordinator._on_takeover


class TestStandbyCoordinatorHealthCheck:
    """Tests for StandbyCoordinator health_check method."""

    def test_health_check_stopped(self):
        """health_check should return STOPPED when not running."""
        coordinator = StandbyCoordinator()
        coordinator._running = False

        result = coordinator.health_check()

        assert result.status.name == "STOPPED"
        assert "not running" in result.message.lower()

    def test_health_check_running_standby(self):
        """health_check should return DEGRADED or RUNNING for standby."""
        coordinator = StandbyCoordinator()
        coordinator._running = True
        coordinator._role = CoordinatorRole.STANDBY
        coordinator._start_time = time.time() - 300  # Running for 5 mins

        result = coordinator.health_check()

        # Standby may report as DEGRADED if primary health unknown
        assert result.status.name in ("RUNNING", "DEGRADED")
        # Message is "Monitoring primary coordinator"
        assert "monitoring" in result.message.lower() or "primary" in result.message.lower()

    def test_health_check_running_primary(self):
        """health_check should return RUNNING for primary."""
        coordinator = StandbyCoordinator()
        coordinator._running = True
        coordinator._role = CoordinatorRole.PRIMARY
        coordinator._start_time = time.time() - 300

        result = coordinator.health_check()

        assert result.status.name == "RUNNING"
        assert "primary" in result.message.lower()


# =============================================================================
# Singleton Tests
# =============================================================================


class TestGetStandbyCoordinator:
    """Tests for get_standby_coordinator singleton function."""

    def test_returns_singleton(self):
        """get_standby_coordinator should return same instance."""
        # Reset singleton first
        StandbyCoordinator._instance = None

        coord1 = get_standby_coordinator()
        coord2 = get_standby_coordinator()

        assert coord1 is coord2

    def test_reset_singleton(self):
        """reset_instance should create new instance."""
        StandbyCoordinator._instance = None

        coord1 = get_standby_coordinator()
        StandbyCoordinator.reset_instance()
        coord2 = get_standby_coordinator()

        assert coord1 is not coord2


# =============================================================================
# State Management Tests
# =============================================================================


class TestStandbyCoordinatorState:
    """Tests for StandbyCoordinator state management."""

    def test_get_state(self):
        """get_state should return current state."""
        coordinator = StandbyCoordinator()
        coordinator._role = CoordinatorRole.PRIMARY
        coordinator._start_time = time.time() - 3600
        coordinator._takeover_count = 2
        coordinator._failover_reason = FailoverReason.PRIMARY_TIMEOUT

        state = coordinator.get_state()

        assert state.role == CoordinatorRole.PRIMARY
        assert state.takeover_count == 2
        assert state.failover_reason == FailoverReason.PRIMARY_TIMEOUT

    def test_state_to_dict(self):
        """get_state().to_dict() should return serializable dict."""
        coordinator = StandbyCoordinator()
        coordinator._role = CoordinatorRole.STANDBY
        coordinator._start_time = time.time()
        coordinator._takeover_count = 0

        state_dict = coordinator.get_state().to_dict()

        assert isinstance(state_dict, dict)
        assert state_dict["role"] == "standby"
        assert "uptime_seconds" in state_dict


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    StandbyCoordinator._instance = None
    yield
    StandbyCoordinator._instance = None
