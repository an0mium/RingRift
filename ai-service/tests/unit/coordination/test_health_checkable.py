"""Unit tests for HealthCheckable protocol and invoke_health_check helper.

December 29, 2025: Tests for the unified health check protocol.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.coordination.contracts import (
    HealthCheckable,
    HealthCheckResult,
    CoordinatorStatus,
    invoke_health_check,
    is_health_checkable,
)


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = HealthCheckResult(healthy=True)
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert result.message == ""
        assert isinstance(result.timestamp, float)
        assert result.details == {}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="All good",
            details={"uptime": 100},
        )
        d = result.to_dict()
        assert d["healthy"] is True
        assert d["status"] == "running"
        assert d["message"] == "All good"
        assert d["details"]["uptime"] == 100

    def test_factory_healthy(self):
        """Test healthy factory method."""
        result = HealthCheckResult.healthy("OK", uptime=60)
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert result.message == "OK"
        assert result.details["uptime"] == 60

    def test_factory_unhealthy(self):
        """Test unhealthy factory method."""
        result = HealthCheckResult.unhealthy("Connection lost", retries=3)
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert result.message == "Connection lost"
        assert result.details["retries"] == 3

    def test_factory_degraded(self):
        """Test degraded factory method."""
        result = HealthCheckResult.degraded("High latency", latency_ms=500)
        assert result.healthy is True  # Degraded is still operational
        assert result.status == CoordinatorStatus.DEGRADED
        assert result.details["latency_ms"] == 500

    def test_from_metrics_healthy(self):
        """Test from_metrics with healthy metrics."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=1000,
            errors_count=5,
        )
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert result.details["error_rate"] == 0.005

    def test_from_metrics_high_error_rate(self):
        """Test from_metrics with high error rate."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=100,
            errors_count=20,  # 20% error rate
            max_error_rate=0.1,
        )
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "High error rate" in result.message

    def test_from_metrics_inactive(self):
        """Test from_metrics with inactive component."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=100,
            last_activity_ago=400.0,  # 400 seconds
            max_inactivity_seconds=300.0,
        )
        assert result.healthy is True
        assert result.status == CoordinatorStatus.DEGRADED
        assert "Inactive" in result.message

    def test_health_score(self):
        """Test health score computation."""
        # Healthy running
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.RUNNING)
        assert result.health_score() == 1.0

        # Degraded
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.DEGRADED)
        assert result.health_score() == 0.7

        # Error
        result = HealthCheckResult(healthy=False, status=CoordinatorStatus.ERROR)
        assert result.health_score() == 0.0

    def test_with_details(self):
        """Test with_details creates new result with merged details."""
        original = HealthCheckResult(
            healthy=True,
            details={"a": 1, "b": 2},
        )
        new = original.with_details(c=3, b=99)

        # Original unchanged
        assert original.details == {"a": 1, "b": 2}

        # New has merged details
        assert new.details == {"a": 1, "b": 99, "c": 3}


class TestIsHealthCheckable:
    """Tests for is_health_checkable function."""

    def test_with_method(self):
        """Test with object that has health_check method."""
        class HasHealthCheck:
            def health_check(self):
                return HealthCheckResult(healthy=True)

        assert is_health_checkable(HasHealthCheck()) is True

    def test_without_method(self):
        """Test with object without health_check method."""
        class NoHealthCheck:
            pass

        assert is_health_checkable(NoHealthCheck()) is False

    def test_with_non_callable(self):
        """Test with object where health_check is not callable."""
        class NonCallable:
            health_check = "not a method"

        assert is_health_checkable(NonCallable()) is False


class TestInvokeHealthCheck:
    """Tests for invoke_health_check async function."""

    @pytest.mark.asyncio
    async def test_sync_health_check(self):
        """Test invoking sync health_check method."""
        class SyncDaemon:
            def health_check(self):
                return HealthCheckResult(healthy=True, message="sync OK")

        result = await invoke_health_check(SyncDaemon())
        assert result.healthy is True
        assert result.message == "sync OK"

    @pytest.mark.asyncio
    async def test_async_health_check(self):
        """Test invoking async health_check method."""
        class AsyncDaemon:
            async def health_check(self):
                return HealthCheckResult(healthy=True, message="async OK")

        result = await invoke_health_check(AsyncDaemon())
        assert result.healthy is True
        assert result.message == "async OK"

    @pytest.mark.asyncio
    async def test_no_health_check_raises(self):
        """Test that missing health_check raises AttributeError."""
        class NoMethod:
            pass

        with pytest.raises(AttributeError):
            await invoke_health_check(NoMethod())

    @pytest.mark.asyncio
    async def test_no_health_check_default_on_error(self):
        """Test default_on_error returns unhealthy result."""
        class NoMethod:
            pass

        result = await invoke_health_check(NoMethod(), default_on_error=True)
        assert result.healthy is False
        assert "no health_check method" in result.message

    @pytest.mark.asyncio
    async def test_exception_raises(self):
        """Test that exceptions propagate by default."""
        class FailingDaemon:
            def health_check(self):
                raise RuntimeError("Health check failed")

        with pytest.raises(RuntimeError):
            await invoke_health_check(FailingDaemon())

    @pytest.mark.asyncio
    async def test_exception_default_on_error(self):
        """Test default_on_error catches exceptions."""
        class FailingDaemon:
            def health_check(self):
                raise RuntimeError("Health check failed")

        result = await invoke_health_check(FailingDaemon(), default_on_error=True)
        assert result.healthy is False
        assert "Health check failed" in result.message
        assert result.details["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_async_exception(self):
        """Test async health_check that raises."""
        class AsyncFailingDaemon:
            async def health_check(self):
                raise ValueError("Async failure")

        result = await invoke_health_check(AsyncFailingDaemon(), default_on_error=True)
        assert result.healthy is False
        assert "Async failure" in result.message


class TestHealthCheckableProtocol:
    """Tests for HealthCheckable protocol type checking."""

    def test_protocol_structural_typing(self):
        """Test that protocol uses structural typing."""
        class ImplicitImplementation:
            def health_check(self) -> HealthCheckResult:
                return HealthCheckResult(healthy=True)

        # Protocol is runtime_checkable
        assert isinstance(ImplicitImplementation(), HealthCheckable)

    def test_protocol_missing_method(self):
        """Test that missing method fails protocol check."""
        class MissingMethod:
            pass

        assert not isinstance(MissingMethod(), HealthCheckable)
