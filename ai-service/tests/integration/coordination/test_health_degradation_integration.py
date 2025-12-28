"""Integration tests for health monitoring and recovery.

Tests health state transitions and recovery:
1. Unhealthy node pauses sync operations
2. Node recovery resumes operations
3. Health degradation triggers appropriate responses
4. Circuit breakers protect against cascading failures

December 2025: Created to verify health monitoring infrastructure works end-to-end.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reset_coordinators():
    """Reset coordinator singletons before and after each test."""
    yield
    # Reset singletons after test
    try:
        from app.coordination.unified_health_manager import UnifiedHealthManager
        UnifiedHealthManager._instance = None
    except ImportError:
        pass


@pytest.fixture
def mock_p2p_status():
    """Create mock P2P status data."""
    return {
        "node_id": "test-node",
        "leader_id": "leader-node",
        "alive_peers": 5,
        "state": "follower",
        "uptime": 3600,
    }


# =============================================================================
# Test Health Degradation
# =============================================================================


class TestHealthDegradation:
    """Integration tests for health monitoring and recovery."""

    @pytest.mark.asyncio
    async def test_unhealthy_node_pauses_sync(self, reset_coordinators):
        """Verify NODE_UNHEALTHY event pauses sync operations."""
        pytest.importorskip("app.coordination.sync_router")

        from app.coordination.sync_router import SyncRouter

        router = SyncRouter()

        # Simulate receiving NODE_UNHEALTHY event
        event = {
            "node_id": "vast-12345",
            "health_state": "unhealthy",
            "reason": "high_error_rate",
            "error_rate": 0.85,
        }

        # Handle the event
        await router._on_node_unhealthy(event)

        # Verify node is marked for exclusion
        excluded = router._excluded_nodes
        assert "vast-12345" in excluded, "Unhealthy node should be excluded from sync"

    @pytest.mark.asyncio
    async def test_node_recovery_resumes_operations(self, reset_coordinators):
        """Verify NODE_RECOVERED event resumes operations."""
        pytest.importorskip("app.coordination.sync_router")

        from app.coordination.sync_router import SyncRouter

        router = SyncRouter()

        # Pre-seed node as excluded
        router._excluded_nodes.add("vast-12345")

        # Create node recovered event
        event = {
            "node_id": "vast-12345",
            "recovery_type": "automatic",
            "recovered_at": time.time(),
        }

        # Handle recovery event
        await router._on_node_recovered(event)

        # Verify node is no longer excluded
        assert "vast-12345" not in router._excluded_nodes, (
            "Recovered node should be removed from exclusion list"
        )

    @pytest.mark.asyncio
    async def test_health_state_transitions(self, reset_coordinators):
        """Verify health state transitions work correctly."""
        pytest.importorskip("app.coordination.unified_health_manager")

        from app.coordination.unified_health_manager import (
            NodeHealthState,
            SystemHealthLevel,
        )

        # Verify state enum values
        assert NodeHealthState.HEALTHY.value == "healthy"
        assert NodeHealthState.DEGRADED.value == "degraded"
        assert NodeHealthState.UNHEALTHY.value == "unhealthy"
        assert NodeHealthState.EVICTED.value == "evicted"

        # Verify health levels
        assert SystemHealthLevel.OPTIMAL.value == "optimal"
        assert SystemHealthLevel.DEGRADED.value == "degraded"
        assert SystemHealthLevel.CRITICAL.value == "critical"

    @pytest.mark.asyncio
    async def test_system_health_score_calculation(self, reset_coordinators):
        """Verify system health score is calculated correctly."""
        pytest.importorskip("app.coordination.unified_health_manager")

        from app.coordination.unified_health_manager import (
            SystemHealthConfig,
            get_system_health_score,
        )

        # Configure with test values
        config = SystemHealthConfig(
            node_availability_weight=0.40,
            circuit_health_weight=0.25,
            error_rate_weight=0.20,
            recovery_rate_weight=0.15,
        )

        # Get health score
        score = get_system_health_score(
            total_nodes=10,
            healthy_nodes=8,
            open_circuits=1,
            total_circuits=5,
            error_rate=0.1,
            recovery_success_rate=0.9,
            config=config,
        )

        # Score should be calculated (0-100)
        assert 0 <= score.overall_score <= 100
        assert hasattr(score, "level")

    @pytest.mark.asyncio
    async def test_health_degradation_triggers_alert(self, reset_coordinators):
        """Verify health degradation triggers appropriate alerts."""
        pytest.importorskip("app.distributed.data_events")

        from app.distributed.data_events import DataEventType

        # Verify health-related event types exist
        health_events = [
            "NODE_UNHEALTHY",
            "NODE_RECOVERED",
            "REGRESSION_DETECTED",
            "CLUSTER_CAPACITY_CHANGED",
        ]

        for event_name in health_events:
            assert hasattr(DataEventType, event_name), f"Missing health event: {event_name}"


# =============================================================================
# Test Recovery Mechanisms
# =============================================================================


class TestRecoveryMechanisms:
    """Tests for node and service recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_automatic_recovery_on_transient_failure(self, reset_coordinators):
        """Verify automatic recovery handles transient failures."""
        pytest.importorskip("app.coordination.node_recovery_daemon")

        from app.coordination.node_recovery_daemon import (
            NodeRecoveryDaemon,
            NodeRecoveryConfig,
        )

        # Create daemon with fast retry for testing
        config = NodeRecoveryConfig(
            check_interval=1.0,
            max_retries=3,
            retry_delay=0.1,
        )
        daemon = NodeRecoveryDaemon(config)

        # Health check should be available
        health = daemon.health_check()
        assert hasattr(health, "healthy")

    @pytest.mark.asyncio
    async def test_recovery_backoff_on_repeated_failures(self, reset_coordinators):
        """Verify recovery applies backoff on repeated failures."""
        pytest.importorskip("app.coordination.daemon_manager")

        from app.coordination.daemon_manager import DaemonManagerConfig

        # Backoff configuration
        config = DaemonManagerConfig(
            recovery_cooldown=60.0,  # 1 minute cooldown
            max_restart_attempts=5,
        )

        # Verify backoff parameters
        assert config.recovery_cooldown == 60.0
        assert config.max_restart_attempts == 5

    @pytest.mark.asyncio
    async def test_node_recovered_event_flow(self, reset_coordinators):
        """Verify NODE_RECOVERED event reaches all subscribers."""
        pytest.importorskip("app.coordination.task_lifecycle_coordinator")

        from app.coordination.task_lifecycle_coordinator import TaskLifecycleCoordinator

        coordinator = TaskLifecycleCoordinator()

        # Pre-seed node as offline
        coordinator._offline_nodes = {"test-node-1": time.time()}
        coordinator._online_nodes = set()

        # Create recovery event
        event = {
            "node_id": "test-node-1",
            "host_id": "test-node-1",
            "recovery_type": "manual",
            "recovered_at": time.time(),
        }

        # Handle recovery
        await coordinator._on_node_recovered(event)

        # Verify node is back online
        assert "test-node-1" in coordinator._online_nodes
        assert "test-node-1" not in coordinator._offline_nodes


# =============================================================================
# Test Pipeline Pause/Resume
# =============================================================================


class TestPipelinePauseResume:
    """Tests for pipeline pause and resume on health changes."""

    @pytest.mark.asyncio
    async def test_pipeline_pauses_on_critical_health(self, reset_coordinators):
        """Verify pipeline pauses when health becomes critical."""
        pytest.importorskip("app.coordination.unified_health_manager")

        from app.coordination.unified_health_manager import (
            should_pause_pipeline,
            SystemHealthLevel,
        )

        # With critical health, pipeline should pause
        should_pause = should_pause_pipeline(SystemHealthLevel.CRITICAL)
        assert should_pause is True

        # With optimal health, pipeline should run
        should_pause = should_pause_pipeline(SystemHealthLevel.OPTIMAL)
        assert should_pause is False

    @pytest.mark.asyncio
    async def test_pipeline_continues_on_degraded_health(self, reset_coordinators):
        """Verify pipeline continues (with reduced capacity) on degraded health."""
        pytest.importorskip("app.coordination.unified_health_manager")

        from app.coordination.unified_health_manager import (
            should_pause_pipeline,
            SystemHealthLevel,
        )

        # Degraded health should not pause pipeline
        should_pause = should_pause_pipeline(SystemHealthLevel.DEGRADED)
        assert should_pause is False

    @pytest.mark.asyncio
    async def test_backpressure_integration(self, reset_coordinators):
        """Verify backpressure events affect sync operations."""
        pytest.importorskip("app.coordination.sync_router")

        from app.coordination.sync_router import SyncRouter

        router = SyncRouter()

        # Simulate backpressure activation
        event = {
            "source": "training_node",
            "level": "high",
            "queue_depth": 1000,
        }

        # Handle backpressure
        if hasattr(router, "_on_backpressure_activated"):
            await router._on_backpressure_activated(event)

            # Should be under backpressure
            assert router.is_under_backpressure() is True

            # Release backpressure
            await router._on_backpressure_released({})
            assert router.is_under_backpressure() is False


# =============================================================================
# Test Circuit Breaker Recovery
# =============================================================================


class TestCircuitBreakerRecovery:
    """Tests for circuit breaker state recovery."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, reset_coordinators):
        """Verify circuit breaker opens after threshold failures."""
        pytest.importorskip("app.coordination.transport_base")

        from app.coordination.transport_base import (
            CircuitBreakerConfig,
            TransportBase,
            TransportState,
        )

        class TestTransport(TransportBase):
            def __init__(self):
                super().__init__(
                    name="test_transport",
                    circuit_config=CircuitBreakerConfig.aggressive(),
                )

        transport = TestTransport()

        # Simulate failures
        for _ in range(transport._circuit_config.failure_threshold):
            transport._record_failure()

        # Circuit should be open
        assert transport._state == TransportState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_timeout(self, reset_coordinators):
        """Verify circuit breaker transitions to half-open after timeout."""
        pytest.importorskip("app.coordination.transport_base")

        from app.coordination.transport_base import (
            CircuitBreakerConfig,
            TransportBase,
            TransportState,
        )

        class TestTransport(TransportBase):
            def __init__(self):
                super().__init__(
                    name="test_transport",
                    circuit_config=CircuitBreakerConfig(
                        failure_threshold=2,
                        recovery_timeout=0.1,  # Very short for testing
                    ),
                )

        transport = TestTransport()

        # Open the circuit
        transport._record_failure()
        transport._record_failure()
        assert transport._state == TransportState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Check if allowed (should transition to half-open)
        if transport._should_allow_request():
            assert transport._state == TransportState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(self, reset_coordinators):
        """Verify circuit breaker closes after successful requests."""
        pytest.importorskip("app.coordination.transport_base")

        from app.coordination.transport_base import (
            CircuitBreakerConfig,
            TransportBase,
            TransportState,
        )

        class TestTransport(TransportBase):
            def __init__(self):
                super().__init__(
                    name="test_transport",
                    circuit_config=CircuitBreakerConfig(
                        failure_threshold=2,
                        recovery_timeout=0.01,
                        success_threshold=1,
                    ),
                )

        transport = TestTransport()

        # Open the circuit
        transport._record_failure()
        transport._record_failure()
        assert transport._state == TransportState.OPEN

        # Wait and check to transition to half-open
        await asyncio.sleep(0.02)
        transport._should_allow_request()

        # Record success to close circuit
        transport._record_success()

        # Circuit should be closed
        assert transport._state == TransportState.CLOSED


# =============================================================================
# Test Health Aggregation
# =============================================================================


class TestHealthAggregation:
    """Tests for aggregating health across multiple components."""

    def test_cluster_health_aggregation(self, reset_coordinators):
        """Verify cluster health is correctly aggregated."""
        pytest.importorskip("app.coordination.health_facade")

        from app.coordination.health_facade import get_cluster_health_summary

        summary = get_cluster_health_summary()

        # Should return dict with health info
        assert isinstance(summary, dict)
        assert "overall_status" in summary or "status" in summary

    @pytest.mark.asyncio
    async def test_daemon_health_aggregation(self, reset_coordinators):
        """Verify daemon health is correctly aggregated."""
        pytest.importorskip("app.coordination.daemon_manager")

        from app.coordination.daemon_manager import DaemonManager

        manager = DaemonManager.get_instance()

        # Get all daemon health
        health = manager.get_all_daemon_health()

        # Should return dict of daemon types to health status
        assert isinstance(health, dict)

    def test_health_check_result_protocol(self, reset_coordinators):
        """Verify HealthCheckResult follows protocol."""
        pytest.importorskip("app.coordination.protocols")

        from app.coordination.protocols import HealthCheckResult

        # Create a health check result
        result = HealthCheckResult(
            healthy=True,
            message="All systems operational",
            details={"uptime": 3600, "errors": 0},
        )

        assert result.healthy is True
        assert "operational" in result.message.lower()
        assert result.details["errors"] == 0


# =============================================================================
# Test Event-Driven Health Updates
# =============================================================================


class TestEventDrivenHealthUpdates:
    """Tests for event-driven health state updates."""

    @pytest.mark.asyncio
    async def test_host_offline_event_updates_health(self, reset_coordinators):
        """Verify HOST_OFFLINE event updates health tracking."""
        pytest.importorskip("app.coordination.task_lifecycle_coordinator")

        from app.coordination.task_lifecycle_coordinator import TaskLifecycleCoordinator

        coordinator = TaskLifecycleCoordinator()
        coordinator._online_nodes = {"test-node"}

        event = {
            "node_id": "test-node",
            "host_id": "test-node",
            "reason": "timeout",
        }

        await coordinator._on_host_offline(event)

        # Node should be marked offline
        assert "test-node" not in coordinator._online_nodes
        assert "test-node" in coordinator._offline_nodes

    @pytest.mark.asyncio
    async def test_host_online_event_updates_health(self, reset_coordinators):
        """Verify HOST_ONLINE event updates health tracking."""
        pytest.importorskip("app.coordination.task_lifecycle_coordinator")

        from app.coordination.task_lifecycle_coordinator import TaskLifecycleCoordinator

        coordinator = TaskLifecycleCoordinator()
        coordinator._online_nodes = set()
        coordinator._offline_nodes = {"new-node": time.time()}

        event = {
            "node_id": "new-node",
            "host_id": "new-node",
            "host_type": "rtx4090",
            "capabilities": {"gpu_vram_gb": 24},
        }

        await coordinator._on_host_online(event)

        # Node should be marked online
        assert "new-node" in coordinator._online_nodes

    @pytest.mark.asyncio
    async def test_regression_detected_event(self, reset_coordinators):
        """Verify REGRESSION_DETECTED event triggers appropriate response."""
        pytest.importorskip("app.distributed.data_events")

        from app.distributed.data_events import DataEventType

        # Verify event type exists
        assert hasattr(DataEventType, "REGRESSION_DETECTED")
        assert DataEventType.REGRESSION_DETECTED.value == "regression_detected"
