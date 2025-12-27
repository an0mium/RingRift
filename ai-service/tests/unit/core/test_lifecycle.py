"""Tests for app.core.lifecycle module.

Comprehensive tests for lifecycle management, service ordering, and health integration.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.health import HealthRegistry, HealthStatus, ProbeType
from app.core.lifecycle import (
    LifecycleEvent,
    LifecycleListener,
    LifecycleManager,
    Service,
    ServiceState,
    managed_lifecycle,
)


# =============================================================================
# Test ServiceState Enum
# =============================================================================


class TestServiceState:
    """Tests for ServiceState enum."""

    def test_state_values(self):
        """Verify state enum values."""
        assert ServiceState.CREATED.value == "created"
        assert ServiceState.STARTING.value == "starting"
        assert ServiceState.RUNNING.value == "running"
        assert ServiceState.STOPPING.value == "stopping"
        assert ServiceState.STOPPED.value == "stopped"
        assert ServiceState.FAILED.value == "failed"

    def test_all_states_exist(self):
        """All expected states are defined."""
        states = [s.value for s in ServiceState]
        assert "created" in states
        assert "running" in states
        assert "stopped" in states
        assert "failed" in states


# =============================================================================
# Test LifecycleEvent Dataclass
# =============================================================================


class TestLifecycleEvent:
    """Tests for LifecycleEvent dataclass."""

    def test_event_creation(self):
        """Can create lifecycle event."""
        event = LifecycleEvent(
            service_name="test_service",
            event_type="started",
            old_state=ServiceState.CREATED,
            new_state=ServiceState.RUNNING,
        )
        assert event.service_name == "test_service"
        assert event.event_type == "started"
        assert event.old_state == ServiceState.CREATED
        assert event.new_state == ServiceState.RUNNING

    def test_event_has_timestamp(self):
        """Event has automatic timestamp."""
        before = time.time()
        event = LifecycleEvent(
            service_name="test",
            event_type="started",
            old_state=None,
            new_state=ServiceState.RUNNING,
        )
        after = time.time()
        assert before <= event.timestamp <= after

    def test_event_with_error(self):
        """Event can include error."""
        error = ValueError("Test error")
        event = LifecycleEvent(
            service_name="test",
            event_type="failed",
            old_state=ServiceState.STARTING,
            new_state=ServiceState.FAILED,
            error=error,
        )
        assert event.error is error

    def test_event_with_details(self):
        """Event can include additional details."""
        event = LifecycleEvent(
            service_name="test",
            event_type="started",
            old_state=None,
            new_state=ServiceState.RUNNING,
            details={"port": 8080, "workers": 4},
        )
        assert event.details["port"] == 8080
        assert event.details["workers"] == 4


# =============================================================================
# Test Service ABC
# =============================================================================


class MockService(Service):
    """Mock service for testing."""

    def __init__(self, name: str = "mock_service", deps: list[str] | None = None):
        super().__init__()
        self._name = name
        self._deps = deps or []
        self.start_called = False
        self.stop_called = False
        self.health_status = HealthStatus.healthy()

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def on_start(self) -> None:
        self.start_called = True

    async def on_stop(self) -> None:
        self.stop_called = True

    async def check_health(self) -> HealthStatus:
        return self.health_status


class FailingService(Service):
    """Service that fails on start."""

    @property
    def name(self) -> str:
        return "failing_service"

    async def on_start(self) -> None:
        raise RuntimeError("Start failed")

    async def on_stop(self) -> None:
        pass

    async def check_health(self) -> HealthStatus:
        return HealthStatus.unhealthy("Service failed")


class TestService:
    """Tests for Service base class."""

    def test_initial_state_is_created(self):
        """Service starts in CREATED state."""
        service = MockService()
        assert service.state == ServiceState.CREATED

    def test_is_running_false_initially(self):
        """is_running is False initially."""
        service = MockService()
        assert service.is_running is False

    def test_uptime_none_when_not_running(self):
        """Uptime is None when not running."""
        service = MockService()
        assert service.uptime is None

    def test_error_none_initially(self):
        """Error is None initially."""
        service = MockService()
        assert service.error is None

    def test_get_status_returns_dict(self):
        """get_status returns a dictionary."""
        service = MockService("test")
        status = service.get_status()
        assert status["name"] == "test"
        assert status["state"] == "created"
        assert status["error"] is None

    @pytest.mark.asyncio
    async def test_do_start_changes_state(self):
        """_do_start transitions to RUNNING."""
        service = MockService()
        await service._do_start()
        assert service.state == ServiceState.RUNNING
        assert service.start_called is True

    @pytest.mark.asyncio
    async def test_do_start_sets_started_at(self):
        """_do_start sets _started_at timestamp."""
        service = MockService()
        before = time.time()
        await service._do_start()
        after = time.time()
        assert before <= service._started_at <= after

    @pytest.mark.asyncio
    async def test_do_start_failure_sets_failed_state(self):
        """_do_start failure sets FAILED state."""
        service = FailingService()
        with pytest.raises(RuntimeError):
            await service._do_start()
        assert service.state == ServiceState.FAILED
        assert service.error is not None

    @pytest.mark.asyncio
    async def test_do_stop_changes_state(self):
        """_do_stop transitions to STOPPED."""
        service = MockService()
        await service._do_start()
        await service._do_stop()
        assert service.state == ServiceState.STOPPED
        assert service.stop_called is True

    @pytest.mark.asyncio
    async def test_uptime_calculated_when_running(self):
        """Uptime is calculated when running."""
        service = MockService()
        await service._do_start()
        await asyncio.sleep(0.1)
        assert service.uptime is not None
        assert service.uptime >= 0.1

    def test_dependencies_empty_by_default(self):
        """Dependencies is empty by default."""
        service = MockService()
        assert service.dependencies == []

    def test_dependencies_can_be_specified(self):
        """Dependencies can be specified."""
        service = MockService("child", deps=["parent1", "parent2"])
        assert service.dependencies == ["parent1", "parent2"]

    def test_get_dependency_raises_when_not_resolved(self):
        """_get_dependency raises when dependency not resolved."""
        service = MockService("child", deps=["parent"])
        with pytest.raises(RuntimeError, match="not resolved"):
            service._get_dependency("parent")

    def test_set_dependency_makes_it_available(self):
        """_set_dependency makes dependency available."""
        parent = MockService("parent")
        child = MockService("child", deps=["parent"])
        child._set_dependency("parent", parent)
        assert child._get_dependency("parent") is parent


# =============================================================================
# Test LifecycleManager
# =============================================================================


class TestLifecycleManager:
    """Tests for LifecycleManager."""

    def test_register_service(self):
        """Can register a service."""
        manager = LifecycleManager()
        service = MockService("test")
        manager.register(service)
        assert "test" in manager.services

    def test_register_duplicate_raises(self):
        """Registering duplicate name raises ValueError."""
        manager = LifecycleManager()
        manager.register(MockService("test"))
        with pytest.raises(ValueError, match="already registered"):
            manager.register(MockService("test"))

    def test_unregister_service(self):
        """Can unregister a service."""
        manager = LifecycleManager()
        service = MockService("test")
        manager.register(service)
        manager.unregister("test")
        assert "test" not in manager.services

    def test_get_service(self):
        """Can get service by name."""
        manager = LifecycleManager()
        service = MockService("test")
        manager.register(service)
        assert manager.get("test") is service

    def test_get_returns_none_for_unknown(self):
        """get returns None for unknown service."""
        manager = LifecycleManager()
        assert manager.get("unknown") is None

    def test_is_shutting_down_initially_false(self):
        """is_shutting_down is False initially."""
        manager = LifecycleManager()
        assert manager.is_shutting_down is False


class TestLifecycleManagerStartupOrder:
    """Tests for LifecycleManager dependency ordering."""

    def test_compute_startup_order_no_deps(self):
        """Startup order computed for services without dependencies."""
        manager = LifecycleManager()
        manager.register(MockService("a"))
        manager.register(MockService("b"))
        manager.register(MockService("c"))
        order = manager.compute_startup_order()
        assert set(order) == {"a", "b", "c"}

    def test_compute_startup_order_with_deps(self):
        """Startup order respects dependencies."""
        manager = LifecycleManager()
        manager.register(MockService("database"))
        manager.register(MockService("cache", deps=["database"]))
        manager.register(MockService("api", deps=["database", "cache"]))

        order = manager.compute_startup_order()
        assert order.index("database") < order.index("cache")
        assert order.index("cache") < order.index("api")

    def test_compute_startup_order_missing_dep_raises(self):
        """Missing dependency raises ValueError."""
        manager = LifecycleManager()
        manager.register(MockService("api", deps=["database"]))

        with pytest.raises(ValueError, match="not registered"):
            manager.compute_startup_order()

    def test_compute_startup_order_circular_raises(self):
        """Circular dependency raises ValueError."""
        manager = LifecycleManager()
        manager.register(MockService("a", deps=["b"]))
        manager.register(MockService("b", deps=["a"]))

        with pytest.raises(ValueError, match="Circular dependency"):
            manager.compute_startup_order()


class TestLifecycleManagerStartAll:
    """Tests for LifecycleManager.start_all()."""

    @pytest.mark.asyncio
    async def test_start_all_starts_services(self):
        """start_all starts all services."""
        manager = LifecycleManager()
        s1 = MockService("s1")
        s2 = MockService("s2")
        manager.register(s1)
        manager.register(s2)

        started = await manager.start_all()
        assert set(started) == {"s1", "s2"}
        assert s1.start_called
        assert s2.start_called

    @pytest.mark.asyncio
    async def test_start_all_resolves_dependencies(self):
        """start_all resolves service dependencies."""
        manager = LifecycleManager()
        parent = MockService("parent")
        child = MockService("child", deps=["parent"])
        manager.register(parent)
        manager.register(child)

        await manager.start_all()
        # Child should be able to access parent
        assert child._get_dependency("parent") is parent

    @pytest.mark.asyncio
    async def test_start_all_stops_on_failure(self):
        """start_all stops started services on failure."""
        manager = LifecycleManager()
        good = MockService("good")
        bad = FailingService()
        manager.register(good)
        manager.register(bad)

        with pytest.raises(RuntimeError):
            await manager.start_all()
        # Good service should be stopped after bad service fails
        # (if it was started before the failure)


class TestLifecycleManagerStopAll:
    """Tests for LifecycleManager.stop_all()."""

    @pytest.mark.asyncio
    async def test_stop_all_stops_services(self):
        """stop_all stops all running services."""
        manager = LifecycleManager()
        s1 = MockService("s1")
        s2 = MockService("s2")
        manager.register(s1)
        manager.register(s2)

        await manager.start_all()
        await manager.stop_all()

        assert s1.stop_called
        assert s2.stop_called
        assert s1.state == ServiceState.STOPPED
        assert s2.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_all_reverse_order(self):
        """stop_all stops in reverse startup order."""
        manager = LifecycleManager()
        stop_order = []

        class OrderTracker(MockService):
            async def on_stop(self) -> None:
                stop_order.append(self.name)

        manager.register(OrderTracker("a"))
        manager.register(OrderTracker("b", deps=["a"]))
        manager.register(OrderTracker("c", deps=["b"]))

        await manager.start_all()
        await manager.stop_all()

        # Should stop in reverse: c, b, a
        assert stop_order == ["c", "b", "a"]


class TestLifecycleManagerRestart:
    """Tests for LifecycleManager.restart()."""

    @pytest.mark.asyncio
    async def test_restart_stops_and_starts(self):
        """restart stops and starts a service."""
        manager = LifecycleManager()
        service = MockService("test")
        manager.register(service)

        await manager.start_all()
        service.start_called = False  # Reset
        await manager.restart("test")

        assert service.stop_called
        assert service.start_called

    @pytest.mark.asyncio
    async def test_restart_unknown_raises(self):
        """restart raises for unknown service."""
        manager = LifecycleManager()
        with pytest.raises(KeyError):
            await manager.restart("unknown")


class TestLifecycleManagerHealth:
    """Tests for LifecycleManager health checking."""

    @pytest.mark.asyncio
    async def test_check_health_aggregates(self):
        """check_health aggregates all service health."""
        manager = LifecycleManager()
        healthy = MockService("healthy")
        unhealthy = MockService("unhealthy")
        unhealthy.health_status = HealthStatus.unhealthy("Bad")
        manager.register(healthy)
        manager.register(unhealthy)

        await manager.start_all()
        result = await manager.check_health()

        assert "healthy" in result.components
        assert "unhealthy" in result.components
        assert not result.is_healthy  # One unhealthy


class TestLifecycleListener:
    """Tests for lifecycle event listeners."""

    class MockListener(LifecycleListener):
        def __init__(self):
            self.events = []

        async def on_lifecycle_event(self, event: LifecycleEvent) -> None:
            self.events.append(event)

    @pytest.mark.asyncio
    async def test_listener_receives_start_event(self):
        """Listener receives service start events."""
        manager = LifecycleManager()
        listener = self.MockListener()
        manager.add_listener(listener)
        manager.register(MockService("test"))

        await manager.start_all()

        assert len(listener.events) == 1
        assert listener.events[0].event_type == "started"
        assert listener.events[0].service_name == "test"

    @pytest.mark.asyncio
    async def test_listener_receives_stop_event(self):
        """Listener receives service stop events."""
        manager = LifecycleManager()
        listener = self.MockListener()
        manager.add_listener(listener)
        manager.register(MockService("test"))

        await manager.start_all()
        await manager.stop_all()

        stop_events = [e for e in listener.events if e.event_type == "stopped"]
        assert len(stop_events) == 1

    @pytest.mark.asyncio
    async def test_remove_listener(self):
        """Can remove a listener."""
        manager = LifecycleManager()
        listener = self.MockListener()
        manager.add_listener(listener)
        manager.remove_listener(listener)
        manager.register(MockService("test"))

        await manager.start_all()

        assert len(listener.events) == 0


class TestLifecycleManagerStatus:
    """Tests for LifecycleManager.get_status()."""

    @pytest.mark.asyncio
    async def test_get_status_includes_all_services(self):
        """get_status includes all registered services."""
        manager = LifecycleManager()
        manager.register(MockService("a"))
        manager.register(MockService("b"))

        status = manager.get_status()
        assert "a" in status["services"]
        assert "b" in status["services"]

    @pytest.mark.asyncio
    async def test_get_status_includes_startup_order(self):
        """get_status includes startup order after computation."""
        manager = LifecycleManager()
        manager.register(MockService("a"))
        manager.register(MockService("b", deps=["a"]))

        await manager.start_all()
        status = manager.get_status()

        assert status["startup_order"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_get_status_shows_shutting_down(self):
        """get_status shows shutting_down state."""
        manager = LifecycleManager()
        status = manager.get_status()
        assert status["shutting_down"] is False


# =============================================================================
# Test managed_lifecycle Context Manager
# =============================================================================


class TestManagedLifecycle:
    """Tests for managed_lifecycle context manager."""

    @pytest.mark.asyncio
    async def test_context_starts_and_stops(self):
        """Context manager starts and stops services."""
        manager = LifecycleManager()
        service = MockService("test")
        manager.register(service)

        async with managed_lifecycle(manager, install_signals=False) as m:
            assert service.state == ServiceState.RUNNING
            assert m is manager

        assert service.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_context_stops_on_exception(self):
        """Services are stopped even on exception."""
        manager = LifecycleManager()
        service = MockService("test")
        manager.register(service)

        with pytest.raises(ValueError):
            async with managed_lifecycle(manager, install_signals=False):
                assert service.state == ServiceState.RUNNING
                raise ValueError("Test error")

        assert service.state == ServiceState.STOPPED


# =============================================================================
# Test Parallel Start
# =============================================================================


class TestParallelStart:
    """Tests for parallel service startup."""

    @pytest.mark.asyncio
    async def test_start_all_parallel(self):
        """Independent services can start in parallel."""
        manager = LifecycleManager()

        class SlowService(MockService):
            async def on_start(self) -> None:
                await asyncio.sleep(0.05)
                self.start_called = True

        # Three independent services
        s1 = SlowService("s1")
        s2 = SlowService("s2")
        s3 = SlowService("s3")
        manager.register(s1)
        manager.register(s2)
        manager.register(s3)

        start = time.time()
        await manager.start_all(parallel=True)
        elapsed = time.time() - start

        # If truly parallel, should take ~0.05s not ~0.15s
        assert elapsed < 0.15
        assert all(s.start_called for s in [s1, s2, s3])

    @pytest.mark.asyncio
    async def test_start_parallel_respects_deps(self):
        """Parallel start still respects dependencies."""
        manager = LifecycleManager()
        start_times = {}

        class TimedService(MockService):
            async def on_start(self) -> None:
                start_times[self.name] = time.time()
                await asyncio.sleep(0.02)

        manager.register(TimedService("a"))
        manager.register(TimedService("b", deps=["a"]))

        await manager.start_all(parallel=True)

        assert start_times["a"] < start_times["b"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestLifecycleIntegration:
    """Integration tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_complex_dependency_chain(self):
        """Handles complex dependency chains correctly."""
        manager = LifecycleManager()

        # Diamond dependency: api depends on cache and db, both depend on config
        manager.register(MockService("config"))
        manager.register(MockService("database", deps=["config"]))
        manager.register(MockService("cache", deps=["config"]))
        manager.register(MockService("api", deps=["database", "cache"]))

        order = manager.compute_startup_order()

        # Config must come first
        assert order.index("config") == 0
        # API must come last
        assert order.index("api") == 3

        # Both db and cache must come before api
        assert order.index("database") < order.index("api")
        assert order.index("cache") < order.index("api")

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Full lifecycle: register, start, health check, stop."""
        manager = LifecycleManager()
        service = MockService("test")
        manager.register(service)

        # Start
        await manager.start_all()
        assert service.is_running

        # Health check
        result = await manager.check_health()
        assert result.is_healthy

        # Stop
        await manager.stop_all()
        assert service.state == ServiceState.STOPPED
