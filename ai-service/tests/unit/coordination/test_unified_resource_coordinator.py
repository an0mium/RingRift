"""Tests for UnifiedResourceCoordinator.

Tests cover:
- Admission control (can_spawn_task)
- Resource allocation recommendations
- Usage reporting
- Backpressure management
- Node health tracking
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.unified_resource_coordinator import (
    BackpressureLevel,
    NodeResourceStatus,
    ResourceAllocation,
    TaskType,
    UnifiedResourceCoordinator,
    get_unified_resource_coordinator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator():
    """Create a fresh UnifiedResourceCoordinator."""
    return UnifiedResourceCoordinator()


# =============================================================================
# NodeResourceStatus Tests
# =============================================================================


class TestNodeResourceStatus:
    """Test NodeResourceStatus dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        status = NodeResourceStatus(node_id="test-node")
        assert status.node_id == "test-node"
        assert status.cpu_percent == 0.0
        assert status.is_healthy is True

    def test_with_values(self):
        """Test initialization with values."""
        status = NodeResourceStatus(
            node_id="test",
            cpu_percent=0.5,
            gpu_percent=0.8,
            active_tasks=3,
        )
        assert status.cpu_percent == 0.5
        assert status.gpu_percent == 0.8
        assert status.active_tasks == 3


# =============================================================================
# ResourceAllocation Tests
# =============================================================================


class TestResourceAllocation:
    """Test ResourceAllocation dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        alloc = ResourceAllocation(config_key="hex8_2p")
        assert alloc.config_key == "hex8_2p"
        assert alloc.gpu_hours == 1.0
        assert alloc.max_concurrent_tasks == 4

    def test_with_recommended_nodes(self):
        """Test allocation with recommended nodes."""
        alloc = ResourceAllocation(
            config_key="hex8_2p",
            recommended_nodes=["node-1", "node-2"],
        )
        assert len(alloc.recommended_nodes) == 2


# =============================================================================
# UnifiedResourceCoordinator Initialization Tests
# =============================================================================


class TestUnifiedResourceCoordinatorInit:
    """Test UnifiedResourceCoordinator initialization."""

    def test_init_default_values(self, coordinator):
        """Test default initialization."""
        assert coordinator._cpu_threshold == 0.85
        assert coordinator._gpu_threshold == 0.90
        assert coordinator._decisions_made == 0

    def test_init_empty_state(self, coordinator):
        """Test initial state is empty."""
        assert len(coordinator._node_status) == 0
        assert coordinator._backpressure_level == BackpressureLevel.NONE


# =============================================================================
# Admission Control Tests
# =============================================================================


class TestCanSpawnTask:
    """Test can_spawn_task admission control."""

    def test_spawn_allowed_no_constraints(self, coordinator):
        """Test spawn allowed when no constraints."""
        allowed, reason = coordinator.can_spawn_task("selfplay", "node-1")

        assert allowed is True
        assert reason == "ok"

    def test_spawn_denied_high_backpressure(self, coordinator):
        """Test spawn denied during high backpressure."""
        coordinator._backpressure_level = BackpressureLevel.HIGH

        allowed, reason = coordinator.can_spawn_task("selfplay", "node-1")

        assert allowed is False
        assert "backpressure" in reason

    def test_spawn_denied_critical_backpressure(self, coordinator):
        """Test spawn denied during critical backpressure."""
        coordinator._backpressure_level = BackpressureLevel.CRITICAL

        allowed, reason = coordinator.can_spawn_task("selfplay", "node-1")

        assert allowed is False

    def test_spawn_denied_unhealthy_node(self, coordinator):
        """Test spawn denied for unhealthy node."""
        coordinator._node_status["node-1"] = NodeResourceStatus(
            node_id="node-1",
            is_healthy=False,
        )

        allowed, reason = coordinator.can_spawn_task("selfplay", "node-1")

        assert allowed is False
        assert "unhealthy" in reason

    def test_spawn_denied_cpu_overloaded(self, coordinator):
        """Test spawn denied when CPU overloaded."""
        coordinator._node_status["node-1"] = NodeResourceStatus(
            node_id="node-1",
            cpu_percent=0.90,  # Above 0.85 threshold
        )

        allowed, reason = coordinator.can_spawn_task("selfplay", "node-1")

        assert allowed is False
        assert "cpu_overloaded" in reason

    def test_spawn_denied_gpu_overloaded(self, coordinator):
        """Test spawn denied when GPU overloaded."""
        coordinator._node_status["node-1"] = NodeResourceStatus(
            node_id="node-1",
            gpu_percent=0.95,  # Above 0.90 threshold
        )

        allowed, reason = coordinator.can_spawn_task("selfplay", "node-1")

        assert allowed is False
        assert "gpu_overloaded" in reason

    def test_spawn_denied_memory_overloaded(self, coordinator):
        """Test spawn denied when memory overloaded."""
        coordinator._node_status["node-1"] = NodeResourceStatus(
            node_id="node-1",
            memory_percent=0.90,  # Above 0.85 threshold
        )

        allowed, reason = coordinator.can_spawn_task("selfplay", "node-1")

        assert allowed is False
        assert "memory_overloaded" in reason

    def test_spawn_with_task_type_enum(self, coordinator):
        """Test spawn with TaskType enum."""
        allowed, reason = coordinator.can_spawn_task(TaskType.TRAINING, "node-1")

        assert allowed is True

    def test_spawn_increments_counters(self, coordinator):
        """Test spawn increments decision counters."""
        coordinator.can_spawn_task("selfplay", "node-1")
        coordinator.can_spawn_task("selfplay", "node-2")

        assert coordinator._decisions_made == 2
        assert coordinator._tasks_allowed == 2


# =============================================================================
# Resource Allocation Tests
# =============================================================================


class TestGetRecommendedAllocation:
    """Test get_recommended_allocation."""

    def test_basic_allocation(self, coordinator):
        """Test basic allocation."""
        alloc = coordinator.get_recommended_allocation("hex8_2p")

        assert alloc.config_key == "hex8_2p"
        assert alloc.gpu_hours > 0
        assert alloc.reason == "normal_allocation"

    def test_allocation_with_healthy_nodes(self, coordinator):
        """Test allocation includes healthy nodes."""
        coordinator._node_status["healthy-1"] = NodeResourceStatus(
            node_id="healthy-1",
            gpu_percent=0.5,
            is_healthy=True,
        )
        coordinator._node_status["healthy-2"] = NodeResourceStatus(
            node_id="healthy-2",
            gpu_percent=0.6,
            is_healthy=True,
        )

        alloc = coordinator.get_recommended_allocation("hex8_2p")

        assert "healthy-1" in alloc.recommended_nodes

    def test_allocation_reduced_during_high_backpressure(self, coordinator):
        """Test allocation reduced during high backpressure."""
        coordinator._backpressure_level = BackpressureLevel.HIGH

        alloc = coordinator.get_recommended_allocation("hex8_2p")

        assert alloc.max_concurrent_tasks == 2
        assert alloc.gpu_hours == 0.5
        assert "backpressure" in alloc.reason

    def test_allocation_minimal_during_critical_backpressure(self, coordinator):
        """Test allocation minimal during critical backpressure."""
        coordinator._backpressure_level = BackpressureLevel.CRITICAL

        alloc = coordinator.get_recommended_allocation("hex8_2p")

        assert alloc.max_concurrent_tasks == 1
        assert alloc.gpu_hours == 0.25


# =============================================================================
# Usage Reporting Tests
# =============================================================================


class TestReportUsage:
    """Test report_usage functionality."""

    def test_report_usage_creates_node(self, coordinator):
        """Test report_usage creates node if not exists."""
        coordinator.report_usage(
            "new-node",
            cpu=0.5,
            gpu=0.6,
            memory=0.4,
        )

        status = coordinator.get_node_status("new-node")
        assert status is not None
        assert status.cpu_percent == 0.5

    def test_report_usage_updates_existing(self, coordinator):
        """Test report_usage updates existing node."""
        coordinator.report_usage("node-1", cpu=0.3)
        coordinator.report_usage("node-1", cpu=0.6)

        status = coordinator.get_node_status("node-1")
        assert status.cpu_percent == 0.6

    def test_report_usage_determines_health(self, coordinator):
        """Test report_usage determines node health."""
        # Healthy node
        coordinator.report_usage("healthy", cpu=0.5, gpu=0.5)
        assert coordinator.get_node_status("healthy").is_healthy

        # Unhealthy node (overloaded)
        coordinator.report_usage("unhealthy", cpu=0.96, gpu=0.99)
        assert not coordinator.get_node_status("unhealthy").is_healthy

    def test_report_usage_updates_backpressure(self, coordinator):
        """Test report_usage updates global backpressure."""
        # Report many overloaded nodes
        for i in range(5):
            coordinator.report_usage(f"node-{i}", gpu=0.96)

        assert coordinator._backpressure_level != BackpressureLevel.NONE


# =============================================================================
# Backpressure Tests
# =============================================================================


class TestBackpressure:
    """Test backpressure management."""

    def test_get_backpressure_level(self, coordinator):
        """Test get_backpressure_level."""
        assert coordinator.get_backpressure_level() == BackpressureLevel.NONE

        coordinator._backpressure_level = BackpressureLevel.HIGH
        assert coordinator.get_backpressure_level() == BackpressureLevel.HIGH

    def test_update_backpressure_critical(self, coordinator):
        """Test backpressure becomes critical when many nodes unhealthy."""
        # Add nodes where less than half are healthy
        coordinator._node_status["healthy-1"] = NodeResourceStatus(
            node_id="healthy-1", is_healthy=True
        )
        for i in range(3):
            coordinator._node_status[f"unhealthy-{i}"] = NodeResourceStatus(
                node_id=f"unhealthy-{i}", is_healthy=False
            )

        coordinator._update_backpressure()

        assert coordinator._backpressure_level == BackpressureLevel.CRITICAL

    def test_update_backpressure_high(self, coordinator):
        """Test backpressure becomes high when GPU > 95%."""
        for i in range(3):
            coordinator._node_status[f"node-{i}"] = NodeResourceStatus(
                node_id=f"node-{i}",
                gpu_percent=0.96,
                is_healthy=True,
            )

        coordinator._update_backpressure()

        assert coordinator._backpressure_level == BackpressureLevel.HIGH

    def test_update_backpressure_none(self, coordinator):
        """Test backpressure is none when load is light."""
        for i in range(3):
            coordinator._node_status[f"node-{i}"] = NodeResourceStatus(
                node_id=f"node-{i}",
                gpu_percent=0.3,
                cpu_percent=0.3,
                is_healthy=True,
            )

        coordinator._update_backpressure()

        assert coordinator._backpressure_level == BackpressureLevel.NONE


# =============================================================================
# Node Health Tests
# =============================================================================


class TestNodeHealth:
    """Test node health management."""

    def test_mark_node_unhealthy(self, coordinator):
        """Test marking node as unhealthy."""
        coordinator._node_status["node-1"] = NodeResourceStatus(
            node_id="node-1", is_healthy=True
        )

        coordinator.mark_node_unhealthy("node-1", "test reason")

        assert not coordinator.get_node_status("node-1").is_healthy

    def test_mark_node_healthy(self, coordinator):
        """Test marking node as healthy."""
        coordinator._node_status["node-1"] = NodeResourceStatus(
            node_id="node-1", is_healthy=False
        )

        coordinator.mark_node_healthy("node-1")

        assert coordinator.get_node_status("node-1").is_healthy

    def test_list_healthy_nodes(self, coordinator):
        """Test listing healthy nodes."""
        coordinator._node_status["healthy-1"] = NodeResourceStatus(
            node_id="healthy-1", gpu_percent=0.3, is_healthy=True
        )
        coordinator._node_status["healthy-2"] = NodeResourceStatus(
            node_id="healthy-2", gpu_percent=0.5, is_healthy=True
        )
        coordinator._node_status["unhealthy-1"] = NodeResourceStatus(
            node_id="unhealthy-1", is_healthy=False
        )

        healthy = coordinator.list_healthy_nodes()

        assert "healthy-1" in healthy
        assert "healthy-2" in healthy
        assert "unhealthy-1" not in healthy

    def test_list_healthy_nodes_sorted_by_capacity(self, coordinator):
        """Test healthy nodes sorted by available capacity."""
        coordinator._node_status["busy"] = NodeResourceStatus(
            node_id="busy", gpu_percent=0.8, is_healthy=True
        )
        coordinator._node_status["idle"] = NodeResourceStatus(
            node_id="idle", gpu_percent=0.2, is_healthy=True
        )

        healthy = coordinator.list_healthy_nodes()

        # Idle node should be first (more capacity)
        assert healthy[0] == "idle"


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics and status."""

    def test_get_cluster_health_summary_empty(self, coordinator):
        """Test cluster health summary with no nodes."""
        summary = coordinator.get_cluster_health_summary()

        assert summary["total_nodes"] == 0
        assert summary["avg_cpu"] == 0.0

    def test_get_cluster_health_summary_with_nodes(self, coordinator):
        """Test cluster health summary with nodes."""
        coordinator._node_status["node-1"] = NodeResourceStatus(
            node_id="node-1",
            cpu_percent=0.4,
            gpu_percent=0.6,
            is_healthy=True,
        )
        coordinator._node_status["node-2"] = NodeResourceStatus(
            node_id="node-2",
            cpu_percent=0.6,
            gpu_percent=0.8,
            is_healthy=True,
        )

        summary = coordinator.get_cluster_health_summary()

        assert summary["total_nodes"] == 2
        assert summary["healthy_nodes"] == 2
        assert summary["avg_cpu"] == 0.5
        assert summary["avg_gpu"] == 0.7

    def test_decision_tracking(self, coordinator):
        """Test decision tracking in summary."""
        coordinator.can_spawn_task("selfplay", "node-1")  # Allowed
        coordinator._backpressure_level = BackpressureLevel.HIGH
        coordinator.can_spawn_task("selfplay", "node-2")  # Denied

        summary = coordinator.get_cluster_health_summary()

        assert summary["decisions_made"] == 2
        assert summary["tasks_allowed"] == 1
        assert summary["tasks_denied"] == 1


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonBehavior:
    """Test singleton behavior."""

    def test_get_unified_resource_coordinator_returns_singleton(self):
        """Test get_unified_resource_coordinator returns same instance."""
        import app.coordination.unified_resource_coordinator as urc
        urc._coordinator = None

        coord1 = get_unified_resource_coordinator()
        coord2 = get_unified_resource_coordinator()

        assert coord1 is coord2


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum values."""

    def test_backpressure_level_values(self):
        """Test BackpressureLevel enum values."""
        assert BackpressureLevel.NONE.value == "none"
        assert BackpressureLevel.HIGH.value == "high"
        assert BackpressureLevel.CRITICAL.value == "critical"

    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.SELFPLAY.value == "selfplay"
        assert TaskType.TRAINING.value == "training"
        assert TaskType.EVALUATION.value == "evaluation"
        assert TaskType.SYNC.value == "sync"
