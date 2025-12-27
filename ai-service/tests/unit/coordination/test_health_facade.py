"""Tests for app.coordination.health_facade module.

Tests the unified health interface including:
- Convenience functions for node health queries
- Cluster health summary
- Backward-compat deprecated functions
- Re-exports from underlying modules

December 2025: Added as part of consolidation test coverage.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Import Tests
# =============================================================================


class TestHealthFacadeImports:
    """Tests for module imports and re-exports."""

    def test_import_system_health_functions(self):
        """Test importing system-level health functions."""
        from app.coordination.health_facade import (
            get_health_manager,
            get_system_health_score,
            get_system_health_level,
            should_pause_pipeline,
        )
        assert callable(get_health_manager)
        assert callable(get_system_health_score)
        assert callable(get_system_health_level)
        assert callable(should_pause_pipeline)

    def test_import_node_health_functions(self):
        """Test importing node-level health functions."""
        from app.coordination.health_facade import (
            get_health_orchestrator,
            get_node_health,
            get_healthy_nodes,
            get_unhealthy_nodes,
            get_degraded_nodes,
            get_offline_nodes,
        )
        assert callable(get_health_orchestrator)
        assert callable(get_node_health)
        assert callable(get_healthy_nodes)
        assert callable(get_unhealthy_nodes)
        assert callable(get_degraded_nodes)
        assert callable(get_offline_nodes)

    def test_import_health_types(self):
        """Test importing health type classes."""
        from app.coordination.health_facade import (
            UnifiedHealthManager,
            HealthCheckOrchestrator,
            SystemHealthLevel,
            SystemHealthConfig,
            SystemHealthScore,
            NodeHealthState,
            NodeHealthDetails,
        )
        assert UnifiedHealthManager is not None
        assert HealthCheckOrchestrator is not None
        assert SystemHealthLevel is not None
        assert NodeHealthState is not None

    def test_all_exports_listed(self):
        """Test __all__ contains expected exports."""
        from app.coordination.health_facade import __all__

        expected = [
            "get_health_manager",
            "get_health_orchestrator",
            "get_node_health",
            "get_healthy_nodes",
            "get_unhealthy_nodes",
            "get_cluster_health_summary",
        ]
        for exp in expected:
            assert exp in __all__, f"{exp} should be in __all__"


# =============================================================================
# Node Health Convenience Functions
# =============================================================================


class TestGetNodeHealth:
    """Tests for get_node_health function."""

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_node_health_found(self, mock_get_orchestrator):
        """Test getting health for existing node."""
        from app.coordination.health_facade import get_node_health

        mock_orchestrator = MagicMock()
        mock_details = MagicMock()
        mock_orchestrator.get_node_health.return_value = mock_details
        mock_get_orchestrator.return_value = mock_orchestrator

        result = get_node_health("runpod-h100")

        mock_orchestrator.get_node_health.assert_called_once_with("runpod-h100")
        assert result is mock_details

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_node_health_not_found(self, mock_get_orchestrator):
        """Test getting health for non-existent node."""
        from app.coordination.health_facade import get_node_health

        mock_orchestrator = MagicMock()
        mock_orchestrator.get_node_health.return_value = None
        mock_get_orchestrator.return_value = mock_orchestrator

        result = get_node_health("unknown-node")

        assert result is None


class TestGetHealthyNodes:
    """Tests for get_healthy_nodes function."""

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_healthy_nodes(self, mock_get_orchestrator):
        """Test getting list of healthy nodes."""
        from app.coordination.health_facade import get_healthy_nodes, NodeHealthState

        mock_orchestrator = MagicMock()
        mock_orchestrator.node_health = {
            "node1": MagicMock(state=NodeHealthState.HEALTHY),
            "node2": MagicMock(state=NodeHealthState.DEGRADED),
            "node3": MagicMock(state=NodeHealthState.HEALTHY),
        }
        mock_get_orchestrator.return_value = mock_orchestrator

        result = get_healthy_nodes()

        assert set(result) == {"node1", "node3"}

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_healthy_nodes_empty(self, mock_get_orchestrator):
        """Test getting healthy nodes when none healthy."""
        from app.coordination.health_facade import get_healthy_nodes, NodeHealthState

        mock_orchestrator = MagicMock()
        mock_orchestrator.node_health = {
            "node1": MagicMock(state=NodeHealthState.OFFLINE),
        }
        mock_get_orchestrator.return_value = mock_orchestrator

        result = get_healthy_nodes()

        assert result == []


class TestGetUnhealthyNodes:
    """Tests for get_unhealthy_nodes function."""

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_unhealthy_nodes(self, mock_get_orchestrator):
        """Test getting list of unhealthy nodes."""
        from app.coordination.health_facade import get_unhealthy_nodes, NodeHealthState

        mock_orchestrator = MagicMock()
        mock_orchestrator.node_health = {
            "node1": MagicMock(state=NodeHealthState.HEALTHY),
            "node2": MagicMock(state=NodeHealthState.DEGRADED),
            "node3": MagicMock(state=NodeHealthState.UNHEALTHY),
        }
        mock_get_orchestrator.return_value = mock_orchestrator

        result = get_unhealthy_nodes()

        assert set(result) == {"node2", "node3"}


class TestGetDegradedNodes:
    """Tests for get_degraded_nodes function."""

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_degraded_nodes(self, mock_get_orchestrator):
        """Test getting list of degraded nodes."""
        from app.coordination.health_facade import get_degraded_nodes, NodeHealthState

        mock_orchestrator = MagicMock()
        mock_orchestrator.node_health = {
            "node1": MagicMock(state=NodeHealthState.HEALTHY),
            "node2": MagicMock(state=NodeHealthState.DEGRADED),
            "node3": MagicMock(state=NodeHealthState.DEGRADED),
        }
        mock_get_orchestrator.return_value = mock_orchestrator

        result = get_degraded_nodes()

        assert set(result) == {"node2", "node3"}


class TestGetOfflineNodes:
    """Tests for get_offline_nodes function."""

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_offline_nodes(self, mock_get_orchestrator):
        """Test getting list of offline nodes."""
        from app.coordination.health_facade import get_offline_nodes, NodeHealthState

        mock_orchestrator = MagicMock()
        mock_orchestrator.node_health = {
            "node1": MagicMock(state=NodeHealthState.HEALTHY),
            "node2": MagicMock(state=NodeHealthState.OFFLINE),
            "node3": MagicMock(state=NodeHealthState.RETIRED),
        }
        mock_get_orchestrator.return_value = mock_orchestrator

        result = get_offline_nodes()

        assert set(result) == {"node2", "node3"}


class TestMarkNodeRetired:
    """Tests for mark_node_retired function."""

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_mark_node_retired_success(self, mock_get_orchestrator):
        """Test successfully marking node as retired."""
        from app.coordination.health_facade import mark_node_retired

        mock_orchestrator = MagicMock()
        mock_orchestrator.mark_retired.return_value = True
        mock_get_orchestrator.return_value = mock_orchestrator

        result = mark_node_retired("runpod-h100")

        mock_orchestrator.mark_retired.assert_called_once_with("runpod-h100")
        assert result is True

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_mark_node_retired_not_found(self, mock_get_orchestrator):
        """Test marking non-existent node as retired."""
        from app.coordination.health_facade import mark_node_retired

        mock_orchestrator = MagicMock()
        mock_orchestrator.mark_retired.return_value = False
        mock_get_orchestrator.return_value = mock_orchestrator

        result = mark_node_retired("unknown-node")

        assert result is False


# =============================================================================
# Cluster Health Summary
# =============================================================================


class TestGetClusterHealthSummary:
    """Tests for get_cluster_health_summary function."""

    @patch("app.coordination.health_facade.should_pause_pipeline")
    @patch("app.coordination.health_facade.get_health_manager")
    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_cluster_health_summary(
        self, mock_get_orchestrator, mock_get_manager, mock_should_pause
    ):
        """Test getting cluster health summary."""
        from app.coordination.health_facade import (
            get_cluster_health_summary,
            NodeHealthState,
            SystemHealthLevel,
        )

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.node_health = {
            "node1": MagicMock(state=NodeHealthState.HEALTHY),
            "node2": MagicMock(state=NodeHealthState.HEALTHY),
            "node3": MagicMock(state=NodeHealthState.DEGRADED),
            "node4": MagicMock(state=NodeHealthState.OFFLINE),
        }
        mock_get_orchestrator.return_value = mock_orchestrator

        # Mock manager
        mock_manager = MagicMock()
        mock_score = MagicMock()
        mock_score.score = 0.85
        mock_score.level = SystemHealthLevel.HEALTHY
        mock_manager.calculate_system_health_score.return_value = mock_score
        mock_get_manager.return_value = mock_manager

        # Mock should_pause
        mock_should_pause.return_value = (False, [])

        result = get_cluster_health_summary()

        assert result["total_nodes"] == 4
        assert result["node_counts"]["healthy"] == 2
        assert result["node_counts"]["degraded"] == 1
        assert result["node_counts"]["offline"] == 1
        assert result["system_score"] == 0.85
        assert result["system_level"] == "healthy"
        assert result["pipeline_paused"] is False

    @patch("app.coordination.health_facade.should_pause_pipeline")
    @patch("app.coordination.health_facade.get_health_manager")
    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_cluster_health_summary_paused(
        self, mock_get_orchestrator, mock_get_manager, mock_should_pause
    ):
        """Test cluster health summary when pipeline is paused."""
        from app.coordination.health_facade import (
            get_cluster_health_summary,
            NodeHealthState,
            SystemHealthLevel,
        )

        mock_orchestrator = MagicMock()
        mock_orchestrator.node_health = {}
        mock_get_orchestrator.return_value = mock_orchestrator

        mock_manager = MagicMock()
        mock_score = MagicMock()
        mock_score.score = 0.3
        mock_score.level = SystemHealthLevel.CRITICAL
        mock_manager.calculate_system_health_score.return_value = mock_score
        mock_get_manager.return_value = mock_manager

        mock_should_pause.return_value = (True, ["Too few healthy nodes"])

        result = get_cluster_health_summary()

        assert result["pipeline_paused"] is True
        assert "Too few healthy nodes" in result["pause_reasons"]


# =============================================================================
# Backward Compatibility Deprecation
# =============================================================================


class TestDeprecatedFunctions:
    """Tests for deprecated backward-compat functions."""

    @patch("app.coordination.health_facade.get_health_orchestrator")
    def test_get_node_health_monitor_warns(self, mock_get_orchestrator):
        """Test get_node_health_monitor emits deprecation warning."""
        from app.coordination.health_facade import get_node_health_monitor

        mock_orchestrator = MagicMock()
        mock_get_orchestrator.return_value = mock_orchestrator

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_node_health_monitor()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "get_health_orchestrator" in str(w[0].message)

        assert result is mock_orchestrator

    @patch("app.coordination.health_facade.get_health_manager")
    def test_get_system_health_warns(self, mock_get_manager):
        """Test get_system_health emits deprecation warning."""
        from app.coordination.health_facade import get_system_health

        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_system_health()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "get_health_manager" in str(w[0].message)

        assert result is mock_manager


# =============================================================================
# NodeHealthState Enum Tests
# =============================================================================


class TestNodeHealthStateEnum:
    """Tests for NodeHealthState enum values."""

    def test_node_health_state_values(self):
        """Test NodeHealthState has expected values."""
        from app.coordination.health_facade import NodeHealthState

        assert NodeHealthState.HEALTHY.value == "healthy"
        assert NodeHealthState.DEGRADED.value == "degraded"
        assert NodeHealthState.UNHEALTHY.value == "unhealthy"
        assert NodeHealthState.OFFLINE.value == "offline"

    def test_node_health_state_retired(self):
        """Test NodeHealthState has RETIRED value (Dec 2025 addition)."""
        from app.coordination.health_facade import NodeHealthState

        # RETIRED was added in Dec 2025 consolidation
        assert hasattr(NodeHealthState, "RETIRED")
        assert NodeHealthState.RETIRED.value == "retired"


# =============================================================================
# SystemHealthLevel Enum Tests
# =============================================================================


class TestSystemHealthLevelEnum:
    """Tests for SystemHealthLevel enum values."""

    def test_system_health_level_values(self):
        """Test SystemHealthLevel has expected values."""
        from app.coordination.health_facade import SystemHealthLevel

        assert hasattr(SystemHealthLevel, "HEALTHY")
        assert hasattr(SystemHealthLevel, "DEGRADED")
        assert hasattr(SystemHealthLevel, "CRITICAL")
