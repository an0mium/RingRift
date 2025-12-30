"""Tests for app.coordination.cluster.health facade module.

This module verifies:
1. All exports in __all__ are importable
2. Re-exports point to correct source modules
3. Backward-compatible aliases work correctly
4. API compatibility between facade and source modules
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHealthFacadeImports:
    """Test that all exports are importable."""

    def test_import_module(self):
        """Test that the module can be imported."""
        from app.coordination.cluster import health
        assert health is not None

    def test_all_exports_importable(self):
        """Test that all items in __all__ are importable."""
        from app.coordination.cluster import health

        for name in health.__all__:
            assert hasattr(health, name), f"Missing export: {name}"
            obj = getattr(health, name)
            assert obj is not None, f"Export is None: {name}"

    def test_all_exports_count(self):
        """Test that __all__ has expected number of exports."""
        from app.coordination.cluster import health

        # Should have exports from 4 source modules
        assert len(health.__all__) >= 20, "Expected at least 20 exports"


class TestUnifiedHealthManagerReExports:
    """Test re-exports from unified_health_manager."""

    def test_unified_health_manager_class(self):
        """Test UnifiedHealthManager is re-exported."""
        from app.coordination.cluster.health import UnifiedHealthManager
        from app.coordination.unified_health_manager import (
            UnifiedHealthManager as Original,
        )

        assert UnifiedHealthManager is Original

    def test_get_health_manager(self):
        """Test get_health_manager is re-exported."""
        from app.coordination.cluster.health import get_health_manager
        from app.coordination.unified_health_manager import (
            get_health_manager as original_func,
        )

        assert get_health_manager is original_func

    def test_wire_health_events(self):
        """Test wire_health_events is re-exported."""
        from app.coordination.cluster.health import wire_health_events
        from app.coordination.unified_health_manager import (
            wire_health_events as original_func,
        )

        assert wire_health_events is original_func

    def test_error_severity_enum(self):
        """Test ErrorSeverity enum is re-exported."""
        from app.coordination.cluster.health import ErrorSeverity
        from app.coordination.unified_health_manager import (
            ErrorSeverity as Original,
        )

        assert ErrorSeverity is Original

    def test_recovery_status_enum(self):
        """Test RecoveryStatus enum is re-exported."""
        from app.coordination.cluster.health import RecoveryStatus
        from app.coordination.unified_health_manager import (
            RecoveryStatus as Original,
        )

        assert RecoveryStatus is Original


class TestHostHealthPolicyReExports:
    """Test re-exports from host_health_policy."""

    def test_health_status_enum(self):
        """Test HealthStatus enum is re-exported."""
        from app.coordination.cluster.health import HealthStatus
        from app.coordination.host_health_policy import HealthStatus as Original

        assert HealthStatus is Original

    def test_check_host_health(self):
        """Test check_host_health is re-exported."""
        from app.coordination.cluster.health import check_host_health
        from app.coordination.host_health_policy import (
            check_host_health as original_func,
        )

        assert check_host_health is original_func

    def test_is_host_healthy(self):
        """Test is_host_healthy is re-exported."""
        from app.coordination.cluster.health import is_host_healthy
        from app.coordination.host_health_policy import (
            is_host_healthy as original_func,
        )

        assert is_host_healthy is original_func

    def test_get_healthy_hosts(self):
        """Test get_healthy_hosts is re-exported."""
        from app.coordination.cluster.health import get_healthy_hosts
        from app.coordination.host_health_policy import (
            get_healthy_hosts as original_func,
        )

        assert get_healthy_hosts is original_func

    def test_clear_health_cache(self):
        """Test clear_health_cache is re-exported."""
        from app.coordination.cluster.health import clear_health_cache
        from app.coordination.host_health_policy import (
            clear_health_cache as original_func,
        )

        assert clear_health_cache is original_func

    def test_get_health_summary(self):
        """Test get_health_summary is re-exported."""
        from app.coordination.cluster.health import get_health_summary
        from app.coordination.host_health_policy import (
            get_health_summary as original_func,
        )

        assert get_health_summary is original_func

    def test_is_cluster_healthy(self):
        """Test is_cluster_healthy is re-exported."""
        from app.coordination.cluster.health import is_cluster_healthy
        from app.coordination.host_health_policy import (
            is_cluster_healthy as original_func,
        )

        assert is_cluster_healthy is original_func

    def test_check_cluster_health(self):
        """Test check_cluster_health is re-exported."""
        from app.coordination.cluster.health import check_cluster_health
        from app.coordination.host_health_policy import (
            check_cluster_health as original_func,
        )

        assert check_cluster_health is original_func


class TestHealthFacadeReExports:
    """Test re-exports from health_facade (preferred interface)."""

    def test_get_health_orchestrator(self):
        """Test get_health_orchestrator is re-exported."""
        from app.coordination.cluster.health import get_health_orchestrator
        from app.coordination.health_facade import (
            get_health_orchestrator as original_func,
        )

        assert get_health_orchestrator is original_func

    def test_health_check_orchestrator_class(self):
        """Test HealthCheckOrchestrator is re-exported."""
        from app.coordination.cluster.health import HealthCheckOrchestrator
        from app.coordination.health_facade import (
            HealthCheckOrchestrator as Original,
        )

        assert HealthCheckOrchestrator is Original

    def test_node_health_state(self):
        """Test NodeHealthState is re-exported."""
        from app.coordination.cluster.health import NodeHealthState
        from app.coordination.health_facade import NodeHealthState as Original

        assert NodeHealthState is Original

    def test_node_health_details(self):
        """Test NodeHealthDetails is re-exported."""
        from app.coordination.cluster.health import NodeHealthDetails
        from app.coordination.health_facade import NodeHealthDetails as Original

        assert NodeHealthDetails is Original

    def test_get_node_health(self):
        """Test get_node_health is re-exported."""
        from app.coordination.cluster.health import get_node_health
        from app.coordination.health_facade import get_node_health as original_func

        assert get_node_health is original_func

    def test_get_healthy_nodes(self):
        """Test get_healthy_nodes is re-exported."""
        from app.coordination.cluster.health import get_healthy_nodes
        from app.coordination.health_facade import get_healthy_nodes as original_func

        assert get_healthy_nodes is original_func

    def test_get_unhealthy_nodes(self):
        """Test get_unhealthy_nodes is re-exported."""
        from app.coordination.cluster.health import get_unhealthy_nodes
        from app.coordination.health_facade import get_unhealthy_nodes as original_func

        assert get_unhealthy_nodes is original_func

    def test_get_degraded_nodes(self):
        """Test get_degraded_nodes is re-exported."""
        from app.coordination.cluster.health import get_degraded_nodes
        from app.coordination.health_facade import get_degraded_nodes as original_func

        assert get_degraded_nodes is original_func

    def test_get_offline_nodes(self):
        """Test get_offline_nodes is re-exported."""
        from app.coordination.cluster.health import get_offline_nodes
        from app.coordination.health_facade import get_offline_nodes as original_func

        assert get_offline_nodes is original_func

    def test_mark_node_retired(self):
        """Test mark_node_retired is re-exported."""
        from app.coordination.cluster.health import mark_node_retired
        from app.coordination.health_facade import mark_node_retired as original_func

        assert mark_node_retired is original_func

    def test_get_cluster_health_summary(self):
        """Test get_cluster_health_summary is re-exported."""
        from app.coordination.cluster.health import get_cluster_health_summary
        from app.coordination.health_facade import (
            get_cluster_health_summary as original_func,
        )

        assert get_cluster_health_summary is original_func

    def test_get_system_health_score(self):
        """Test get_system_health_score is re-exported."""
        from app.coordination.cluster.health import get_system_health_score
        from app.coordination.health_facade import (
            get_system_health_score as original_func,
        )

        assert get_system_health_score is original_func

    def test_get_system_health_level(self):
        """Test get_system_health_level is re-exported."""
        from app.coordination.cluster.health import get_system_health_level
        from app.coordination.health_facade import (
            get_system_health_level as original_func,
        )

        assert get_system_health_level is original_func

    def test_should_pause_pipeline(self):
        """Test should_pause_pipeline is re-exported."""
        from app.coordination.cluster.health import should_pause_pipeline
        from app.coordination.health_facade import (
            should_pause_pipeline as original_func,
        )

        assert should_pause_pipeline is original_func

    def test_system_health_level_enum(self):
        """Test SystemHealthLevel enum is re-exported."""
        from app.coordination.cluster.health import SystemHealthLevel
        from app.coordination.health_facade import SystemHealthLevel as Original

        assert SystemHealthLevel is Original

    def test_system_health_score_class(self):
        """Test SystemHealthScore is re-exported."""
        from app.coordination.cluster.health import SystemHealthScore
        from app.coordination.health_facade import SystemHealthScore as Original

        assert SystemHealthScore is Original


class TestBackwardCompatibleAliases:
    """Test backward-compatible aliases (deprecated, Q2 2026 removal)."""

    def test_node_health_monitor_alias(self):
        """Test NodeHealthMonitor is alias for HealthCheckOrchestrator."""
        from app.coordination.cluster.health import (
            NodeHealthMonitor,
            HealthCheckOrchestrator,
        )
        from app.coordination.health_check_orchestrator import (
            HealthCheckOrchestrator as Original,
        )

        assert NodeHealthMonitor is HealthCheckOrchestrator
        assert NodeHealthMonitor is Original

    def test_get_node_health_monitor_alias(self):
        """Test get_node_health_monitor is alias for get_health_orchestrator."""
        from app.coordination.cluster.health import (
            get_node_health_monitor,
            get_health_orchestrator,
        )
        from app.coordination.health_check_orchestrator import (
            get_health_orchestrator as original_func,
        )

        assert get_node_health_monitor is get_health_orchestrator
        assert get_node_health_monitor is original_func

    def test_node_status_alias(self):
        """Test NodeStatus is alias for NodeHealthState."""
        from app.coordination.cluster.health import NodeStatus, NodeHealthState
        from app.coordination.node_status import NodeHealthState as Original

        assert NodeStatus is NodeHealthState
        assert NodeStatus is Original

    def test_node_health_alias(self):
        """Test NodeHealth is alias for NodeMonitoringStatus."""
        from app.coordination.cluster.health import NodeHealth
        from app.coordination.node_status import NodeMonitoringStatus as Original

        assert NodeHealth is Original


class TestAPICompatibility:
    """Test API compatibility between facade and source modules."""

    def test_facade_provides_complete_health_interface(self):
        """Test that facade provides all commonly needed health APIs."""
        from app.coordination.cluster import health

        # Core health manager
        assert hasattr(health, "UnifiedHealthManager")
        assert hasattr(health, "get_health_manager")

        # Host health checks
        assert hasattr(health, "check_host_health")
        assert hasattr(health, "is_host_healthy")
        assert hasattr(health, "get_healthy_hosts")

        # Node health monitoring
        assert hasattr(health, "get_health_orchestrator")
        assert hasattr(health, "get_node_health")
        assert hasattr(health, "get_healthy_nodes")

        # System health
        assert hasattr(health, "get_system_health_score")
        assert hasattr(health, "get_system_health_level")
        assert hasattr(health, "should_pause_pipeline")

        # Cluster health
        assert hasattr(health, "is_cluster_healthy")
        assert hasattr(health, "get_cluster_health_summary")

    def test_enums_have_expected_values(self):
        """Test that re-exported enums have expected values."""
        from app.coordination.cluster.health import (
            ErrorSeverity,
            RecoveryStatus,
            HealthStatus,
            SystemHealthLevel,
        )

        # ErrorSeverity should have severity levels (INFO, WARNING, ERROR, CRITICAL)
        assert hasattr(ErrorSeverity, "WARNING")
        assert hasattr(ErrorSeverity, "CRITICAL")

        # RecoveryStatus should have recovery states (PENDING, IN_PROGRESS, COMPLETED, FAILED)
        assert hasattr(RecoveryStatus, "COMPLETED")
        assert hasattr(RecoveryStatus, "PENDING")

        # HealthStatus is a dataclass with health check fields
        assert hasattr(HealthStatus, "latency_ms")
        assert hasattr(HealthStatus, "cpu_count")

        # SystemHealthLevel should have level values
        assert hasattr(SystemHealthLevel, "HEALTHY") or hasattr(
            SystemHealthLevel, "CRITICAL"
        )

    def test_classes_are_instantiable_types(self):
        """Test that re-exported classes are proper types."""
        from app.coordination.cluster.health import (
            UnifiedHealthManager,
            HealthCheckOrchestrator,
        )

        assert isinstance(UnifiedHealthManager, type)
        assert isinstance(HealthCheckOrchestrator, type)

    def test_functions_are_callable(self):
        """Test that re-exported functions are callable."""
        from app.coordination.cluster.health import (
            get_health_manager,
            check_host_health,
            get_health_orchestrator,
            get_system_health_score,
        )

        assert callable(get_health_manager)
        assert callable(check_host_health)
        assert callable(get_health_orchestrator)
        assert callable(get_system_health_score)


class TestModuleDocumentation:
    """Test module documentation accuracy."""

    def test_module_has_docstring(self):
        """Test that module has a docstring."""
        from app.coordination.cluster import health

        assert health.__doc__ is not None
        assert len(health.__doc__) > 50

    def test_docstring_mentions_source_modules(self):
        """Test that docstring documents source modules."""
        from app.coordination.cluster import health

        doc = health.__doc__
        assert "unified_health_manager" in doc
        assert "host_health_policy" in doc

    def test_docstring_has_usage_example(self):
        """Test that docstring has usage example."""
        from app.coordination.cluster import health

        doc = health.__doc__
        assert "from app.coordination.cluster.health import" in doc
