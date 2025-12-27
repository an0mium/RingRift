"""Tests for app.p2p.models - P2P Data Models.

This module tests the data models used for P2P cluster orchestration:
- Enums: NodeRole, JobType, JobStatus, NodeHealth
- Dataclasses: ResourceMetrics, NodeSummary
"""

from __future__ import annotations

import pytest

from app.p2p.models import (
    JobStatus,
    JobType,
    NodeHealth,
    NodeRole,
    NodeSummary,
    ResourceMetrics,
)


# =============================================================================
# NodeRole Enum Tests
# =============================================================================


class TestNodeRole:
    """Tests for NodeRole enum."""

    def test_role_values(self):
        """Should have correct role values."""
        assert NodeRole.LEADER.value == "leader"
        assert NodeRole.FOLLOWER.value == "follower"
        assert NodeRole.CANDIDATE.value == "candidate"

    def test_role_count(self):
        """Should have exactly 3 roles."""
        assert len(NodeRole) == 3

    def test_is_string_enum(self):
        """NodeRole should be a string enum."""
        assert isinstance(NodeRole.LEADER, str)
        assert NodeRole.LEADER == "leader"


# =============================================================================
# JobType Enum Tests
# =============================================================================


class TestJobType:
    """Tests for JobType enum."""

    def test_selfplay_types(self):
        """Should have selfplay job types."""
        assert JobType.SELFPLAY.value == "selfplay"
        assert JobType.GPU_SELFPLAY.value == "gpu_selfplay"
        assert JobType.HYBRID_SELFPLAY.value == "hybrid_selfplay"
        assert JobType.CPU_SELFPLAY.value == "cpu_selfplay"
        assert JobType.GUMBEL_SELFPLAY.value == "gumbel_selfplay"

    def test_training_type(self):
        """Should have training job type."""
        assert JobType.TRAINING.value == "training"

    def test_distributed_types(self):
        """Should have distributed job types."""
        assert JobType.DISTRIBUTED_CMAES_COORDINATOR.value == "distributed_cmaes_coordinator"
        assert JobType.DISTRIBUTED_CMAES_WORKER.value == "distributed_cmaes_worker"
        assert JobType.DISTRIBUTED_TOURNAMENT_COORDINATOR.value == "distributed_tournament_coordinator"
        assert JobType.DISTRIBUTED_TOURNAMENT_WORKER.value == "distributed_tournament_worker"

    def test_is_string_enum(self):
        """JobType should be a string enum."""
        assert isinstance(JobType.TRAINING, str)


# =============================================================================
# JobStatus Enum Tests
# =============================================================================


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_all_statuses(self):
        """Should have all expected job statuses."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.STARTING.value == "starting"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.STOPPING.value == "stopping"
        assert JobStatus.STOPPED.value == "stopped"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.COMPLETED.value == "completed"

    def test_status_count(self):
        """Should have exactly 7 statuses."""
        assert len(JobStatus) == 7


# =============================================================================
# NodeHealth Enum Tests
# =============================================================================


class TestNodeHealth:
    """Tests for NodeHealth enum."""

    def test_health_values(self):
        """Should have correct health values."""
        assert NodeHealth.HEALTHY.value == "healthy"
        assert NodeHealth.DEGRADED.value == "degraded"
        assert NodeHealth.UNHEALTHY.value == "unhealthy"
        assert NodeHealth.OFFLINE.value == "offline"
        assert NodeHealth.RETIRED.value == "retired"

    def test_health_count(self):
        """Should have exactly 5 health states."""
        assert len(NodeHealth) == 5


# =============================================================================
# ResourceMetrics Tests
# =============================================================================


class TestResourceMetrics:
    """Tests for ResourceMetrics dataclass."""

    def test_default_values(self):
        """Should have zero defaults."""
        metrics = ResourceMetrics()
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_percent == 0.0
        assert metrics.gpu_utilization == 0.0
        assert metrics.load_average_1m == 0.0

    def test_custom_values(self):
        """Should accept custom values."""
        metrics = ResourceMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            gpu_utilization=80.0,
        )
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_percent == 70.0
        assert metrics.gpu_utilization == 80.0

    def test_load_score_cpu_dominated(self):
        """Load score should be max of components."""
        metrics = ResourceMetrics(
            cpu_percent=90.0,
            memory_percent=50.0,
            load_average_1m=2.0,
        )
        assert metrics.load_score == 90.0

    def test_load_score_memory_dominated(self):
        """Load score should pick highest component."""
        metrics = ResourceMetrics(
            cpu_percent=30.0,
            memory_percent=85.0,
            load_average_1m=2.0,
        )
        assert metrics.load_score == 85.0

    def test_load_score_load_average_dominated(self):
        """Load score should normalize load average."""
        metrics = ResourceMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            load_average_1m=10.0,
        )
        assert metrics.load_score == 100.0

    def test_is_overloaded_false(self):
        """Should not be overloaded at normal levels."""
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=60.0)
        assert metrics.is_overloaded is False

    def test_is_overloaded_by_load_score(self):
        """Should be overloaded when load_score > 80."""
        metrics = ResourceMetrics(cpu_percent=85.0, memory_percent=50.0)
        assert metrics.is_overloaded is True

    def test_is_overloaded_by_memory(self):
        """Should be overloaded when memory > 80%."""
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=85.0)
        assert metrics.is_overloaded is True

    def test_to_dict(self):
        """Should convert to dict correctly."""
        metrics = ResourceMetrics(cpu_percent=50.0, memory_percent=60.0, disk_percent=40.0)
        d = metrics.to_dict()
        assert d["cpu_percent"] == 50.0
        assert d["memory_percent"] == 60.0
        assert d["disk_percent"] == 40.0
        assert "load_score" in d
        assert "is_overloaded" in d


# =============================================================================
# NodeSummary Tests
# =============================================================================


class TestNodeSummary:
    """Tests for NodeSummary dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        node = NodeSummary(node_id="test-node", hostname="test-host", ip_address="192.168.1.1")
        assert node.node_id == "test-node"
        assert node.hostname == "test-host"
        assert node.ip_address == "192.168.1.1"
        assert node.port == 8770
        assert node.role == NodeRole.FOLLOWER
        assert node.health == NodeHealth.HEALTHY

    def test_endpoint_property(self):
        """Should generate correct endpoint URL."""
        node = NodeSummary(node_id="test", hostname="host", ip_address="10.0.0.1", port=8080)
        assert node.endpoint == "http://10.0.0.1:8080"

    def test_is_online_healthy(self):
        """Healthy node should be online."""
        node = NodeSummary(node_id="test", hostname="host", ip_address="10.0.0.1", health=NodeHealth.HEALTHY)
        assert node.is_online is True

    def test_is_online_degraded(self):
        """Degraded node should still be online."""
        node = NodeSummary(node_id="test", hostname="host", ip_address="10.0.0.1", health=NodeHealth.DEGRADED)
        assert node.is_online is True

    def test_is_online_offline(self):
        """Offline node should not be online."""
        node = NodeSummary(node_id="test", hostname="host", ip_address="10.0.0.1", health=NodeHealth.OFFLINE)
        assert node.is_online is False

    def test_is_online_retired(self):
        """Retired node should not be online."""
        node = NodeSummary(node_id="test", hostname="host", ip_address="10.0.0.1", health=NodeHealth.RETIRED)
        assert node.is_online is False

    def test_can_accept_jobs_healthy(self):
        """Healthy non-overloaded node should accept jobs."""
        node = NodeSummary(
            node_id="test", hostname="host", ip_address="10.0.0.1",
            health=NodeHealth.HEALTHY,
            resources=ResourceMetrics(cpu_percent=50.0, memory_percent=50.0),
        )
        assert node.can_accept_jobs is True

    def test_can_accept_jobs_degraded(self):
        """Degraded node should not accept jobs."""
        node = NodeSummary(node_id="test", hostname="host", ip_address="10.0.0.1", health=NodeHealth.DEGRADED)
        assert node.can_accept_jobs is False

    def test_can_accept_jobs_overloaded(self):
        """Overloaded node should not accept jobs."""
        node = NodeSummary(
            node_id="test", hostname="host", ip_address="10.0.0.1",
            health=NodeHealth.HEALTHY,
            resources=ResourceMetrics(cpu_percent=90.0, memory_percent=50.0),
        )
        assert node.can_accept_jobs is False

    def test_can_accept_jobs_offline(self):
        """Offline node should not accept jobs."""
        node = NodeSummary(node_id="test", hostname="host", ip_address="10.0.0.1", health=NodeHealth.OFFLINE)
        assert node.can_accept_jobs is False

    def test_to_dict(self):
        """Should convert to dict correctly."""
        node = NodeSummary(
            node_id="test-node", hostname="test-host", ip_address="192.168.1.1",
            gpu_type="H100", gpu_count=2,
        )
        d = node.to_dict()
        assert d["node_id"] == "test-node"
        assert d["hostname"] == "test-host"
        assert d["ip_address"] == "192.168.1.1"
        assert d["gpu_type"] == "H100"
        assert d["gpu_count"] == 2
