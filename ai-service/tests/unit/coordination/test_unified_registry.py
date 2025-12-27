"""Tests for UnifiedRegistry - unified facade for all registry types.

December 27, 2025: Created as part of test coverage improvement effort.
Tests the facade pattern for accessing model, orchestrator, health, and dynamic registries.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def unified_registry():
    """Create a fresh UnifiedRegistry instance."""
    from app.coordination.unified_registry import UnifiedRegistry, reset_unified_registry
    reset_unified_registry()
    return UnifiedRegistry()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    from app.coordination.unified_registry import reset_unified_registry
    yield
    reset_unified_registry()


# =============================================================================
# RegistryHealth Dataclass Tests
# =============================================================================


class TestRegistryHealth:
    """Tests for RegistryHealth dataclass."""

    def test_defaults(self):
        """Test default values."""
        from app.coordination.unified_registry import RegistryHealth

        health = RegistryHealth(name="test", available=True, healthy=True)
        assert health.name == "test"
        assert health.available is True
        assert health.healthy is True
        assert health.item_count == 0
        assert health.last_access == 0.0
        assert health.errors_count == 0
        assert health.error_message is None

    def test_with_error(self):
        """Test with error state."""
        from app.coordination.unified_registry import RegistryHealth

        health = RegistryHealth(
            name="test",
            available=True,
            healthy=False,
            errors_count=3,
            error_message="Connection failed",
        )
        assert health.healthy is False
        assert health.errors_count == 3
        assert health.error_message == "Connection failed"


# =============================================================================
# ClusterHealth Dataclass Tests
# =============================================================================


class TestClusterHealth:
    """Tests for ClusterHealth dataclass."""

    def test_defaults(self):
        """Test default values."""
        from app.coordination.unified_registry import ClusterHealth

        health = ClusterHealth(healthy=True)
        assert health.healthy is True
        assert health.registries == []
        assert health.total_items == 0
        assert health.unhealthy_count == 0
        assert health.timestamp > 0

    def test_with_registries(self):
        """Test with registry health data."""
        from app.coordination.unified_registry import ClusterHealth, RegistryHealth

        reg1 = RegistryHealth(name="model", available=True, healthy=True, item_count=10)
        reg2 = RegistryHealth(name="health", available=True, healthy=False)

        health = ClusterHealth(
            healthy=False,
            registries=[reg1, reg2],
            total_items=10,
            unhealthy_count=1,
        )
        assert len(health.registries) == 2
        assert health.total_items == 10
        assert health.unhealthy_count == 1


# =============================================================================
# UnifiedRegistry Initialization Tests
# =============================================================================


class TestUnifiedRegistryInit:
    """Tests for UnifiedRegistry initialization."""

    def test_basic_init(self, unified_registry):
        """Test basic initialization."""
        assert unified_registry._model_registry is None
        assert unified_registry._orchestrator_registry is None
        assert unified_registry._health_registry is None
        assert unified_registry._dynamic_registry is None
        assert unified_registry._init_errors == {}

    def test_lazy_loading_not_triggered(self, unified_registry):
        """Test that registries are not loaded on init."""
        # No imports should happen during construction
        assert unified_registry._model_registry is None
        assert unified_registry._orchestrator_registry is None


# =============================================================================
# Model Registry Operation Tests
# =============================================================================


class TestModelRegistryOperations:
    """Tests for model registry operations."""

    def test_get_models_no_registry(self, unified_registry):
        """Test get_models returns empty when registry unavailable."""
        with patch.object(unified_registry, '_get_model_registry', return_value=None):
            result = unified_registry.get_models()
            assert result == []

    def test_get_models_with_mocked_registry(self, unified_registry):
        """Test get_models with mocked registry."""
        mock_registry = MagicMock()
        mock_registry.list_models.return_value = [
            {"id": "model1", "board_type": "hex8"},
            {"id": "model2", "board_type": "hex8"},
        ]

        with patch.object(unified_registry, '_get_model_registry', return_value=mock_registry):
            result = unified_registry.get_models(board_type="hex8")
            assert len(result) == 2
            mock_registry.list_models.assert_called_once_with(
                board_type="hex8",
                num_players=None,
                stage=None,
                limit=100,
            )

    def test_get_models_exception_handling(self, unified_registry):
        """Test get_models handles exceptions gracefully."""
        mock_registry = MagicMock()
        mock_registry.list_models.side_effect = RuntimeError("DB error")

        with patch.object(unified_registry, '_get_model_registry', return_value=mock_registry):
            result = unified_registry.get_models()
            assert result == []

    def test_get_best_model_no_registry(self, unified_registry):
        """Test get_best_model returns None when registry unavailable."""
        with patch.object(unified_registry, '_get_model_registry', return_value=None):
            result = unified_registry.get_best_model("hex8", 2)
            assert result is None

    def test_get_best_model_with_mocked_registry(self, unified_registry):
        """Test get_best_model with mocked registry."""
        mock_model = MagicMock()
        mock_model.to_dict.return_value = {"id": "best_model", "board_type": "hex8"}

        mock_registry = MagicMock()
        mock_registry.get_best_model.return_value = mock_model

        with patch.object(unified_registry, '_get_model_registry', return_value=mock_registry):
            result = unified_registry.get_best_model("hex8", 2, "production")
            assert result == {"id": "best_model", "board_type": "hex8"}

    def test_register_model_no_registry(self, unified_registry):
        """Test register_model returns None when registry unavailable."""
        with patch.object(unified_registry, '_get_model_registry', return_value=None):
            result = unified_registry.register_model("hex8", 2)
            assert result is None

    def test_register_model_success(self, unified_registry):
        """Test successful model registration."""
        mock_registry = MagicMock()
        mock_registry.register_model.return_value = "model-123"

        with patch.object(unified_registry, '_get_model_registry', return_value=mock_registry):
            result = unified_registry.register_model(
                board_type="hex8",
                num_players=2,
                model_path="/path/to/model.pth",
                metrics={"accuracy": 0.95},
            )
            assert result == "model-123"


# =============================================================================
# Orchestrator Registry Operation Tests
# =============================================================================


class TestOrchestratorRegistryOperations:
    """Tests for orchestrator registry operations."""

    def test_get_active_orchestrators_no_registry(self, unified_registry):
        """Test get_active_orchestrators returns empty when registry unavailable."""
        with patch.object(unified_registry, '_get_orchestrator_registry', return_value=None):
            result = unified_registry.get_active_orchestrators()
            assert result == []

    def test_get_active_orchestrators_with_mocked_registry(self, unified_registry):
        """Test get_active_orchestrators with mocked registry."""
        mock_registry = MagicMock()
        mock_registry.get_active_orchestrators.return_value = [
            {"node_id": "node1", "status": "active"},
            {"node_id": "node2", "status": "active"},
        ]

        with patch.object(unified_registry, '_get_orchestrator_registry', return_value=mock_registry):
            result = unified_registry.get_active_orchestrators()
            assert len(result) == 2

    def test_get_healthy_orchestrators_filters_by_heartbeat(self, unified_registry):
        """Test get_healthy_orchestrators filters by recent heartbeat."""
        current_time = time.time()
        mock_registry = MagicMock()
        mock_registry.get_active_orchestrators.return_value = [
            {"node_id": "node1", "last_heartbeat": current_time - 60},  # Recent
            {"node_id": "node2", "last_heartbeat": current_time - 180},  # Old (>120s)
        ]

        with patch.object(unified_registry, '_get_orchestrator_registry', return_value=mock_registry):
            result = unified_registry.get_healthy_orchestrators()
            assert len(result) == 1
            assert result[0]["node_id"] == "node1"

    def test_register_orchestrator_no_registry(self, unified_registry):
        """Test register_orchestrator returns False when registry unavailable."""
        with patch.object(unified_registry, '_get_orchestrator_registry', return_value=None):
            result = unified_registry.register_orchestrator("node1", "Node 1")
            assert result is False

    def test_register_orchestrator_success(self, unified_registry):
        """Test successful orchestrator registration."""
        mock_registry = MagicMock()

        with patch.object(unified_registry, '_get_orchestrator_registry', return_value=mock_registry):
            result = unified_registry.register_orchestrator(
                node_id="node1",
                node_name="Node 1",
                orchestrator_type="coordinator",
                capabilities={"gpu": True},
            )
            assert result is True
            mock_registry.register.assert_called_once()


# =============================================================================
# Health Registry Operation Tests
# =============================================================================


class TestHealthRegistryOperations:
    """Tests for health registry operations."""

    def test_get_node_health_no_registry(self, unified_registry):
        """Test get_node_health returns None when registry unavailable."""
        with patch.object(unified_registry, '_get_health_registry', return_value=None):
            result = unified_registry.get_node_health("node1")
            assert result is None

    def test_get_node_health_with_mocked_registry(self, unified_registry):
        """Test get_node_health with mocked registry."""
        mock_registry = MagicMock()
        mock_registry.get_node_health.return_value = {"status": "healthy", "cpu": 50}

        with patch.object(unified_registry, '_get_health_registry', return_value=mock_registry):
            result = unified_registry.get_node_health("node1")
            assert result == {"status": "healthy", "cpu": 50}

    def test_get_all_node_health_no_registry(self, unified_registry):
        """Test get_all_node_health returns empty when registry unavailable."""
        with patch.object(unified_registry, '_get_health_registry', return_value=None):
            result = unified_registry.get_all_node_health()
            assert result == []

    def test_update_node_health_no_registry(self, unified_registry):
        """Test update_node_health returns False when registry unavailable."""
        with patch.object(unified_registry, '_get_health_registry', return_value=None):
            result = unified_registry.update_node_health("node1", "healthy")
            assert result is False

    def test_update_node_health_success(self, unified_registry):
        """Test successful health update."""
        mock_registry = MagicMock()

        with patch.object(unified_registry, '_get_health_registry', return_value=mock_registry):
            result = unified_registry.update_node_health(
                node_id="node1",
                status="degraded",
                metrics={"cpu": 90},
            )
            assert result is True
            mock_registry.update_health.assert_called_once_with("node1", "degraded", {"cpu": 90})


# =============================================================================
# Dynamic Host Registry Operation Tests
# =============================================================================


class TestDynamicHostRegistryOperations:
    """Tests for dynamic host registry operations."""

    def test_get_available_hosts_no_registry(self, unified_registry):
        """Test get_available_hosts returns empty when registry unavailable."""
        with patch.object(unified_registry, '_get_dynamic_registry', return_value=None):
            result = unified_registry.get_available_hosts()
            assert result == []

    def test_get_available_hosts_with_mocked_registry(self, unified_registry):
        """Test get_available_hosts with mocked registry."""
        mock_registry = MagicMock()
        mock_registry.get_available_hosts.return_value = [
            {"host": "10.0.0.1", "port": 8770},
            {"host": "10.0.0.2", "port": 8770},
        ]

        with patch.object(unified_registry, '_get_dynamic_registry', return_value=mock_registry):
            result = unified_registry.get_available_hosts()
            assert len(result) == 2

    def test_register_host_no_registry(self, unified_registry):
        """Test register_host returns False when registry unavailable."""
        with patch.object(unified_registry, '_get_dynamic_registry', return_value=None):
            result = unified_registry.register_host("10.0.0.1", 8770)
            assert result is False

    def test_register_host_success(self, unified_registry):
        """Test successful host registration."""
        mock_registry = MagicMock()

        with patch.object(unified_registry, '_get_dynamic_registry', return_value=mock_registry):
            result = unified_registry.register_host(
                host="10.0.0.1",
                port=8770,
                capabilities={"gpu": "RTX 4090"},
            )
            assert result is True
            mock_registry.register_host.assert_called_once()


# =============================================================================
# Unified Operations Tests
# =============================================================================


class TestUnifiedOperations:
    """Tests for unified cluster operations."""

    def test_get_cluster_health_all_unavailable(self, unified_registry):
        """Test get_cluster_health when all registries are unavailable."""
        with patch.object(unified_registry, '_get_model_registry', return_value=None), \
             patch.object(unified_registry, '_get_orchestrator_registry', return_value=None), \
             patch.object(unified_registry, '_get_health_registry', return_value=None), \
             patch.object(unified_registry, '_get_dynamic_registry', return_value=None):

            health = unified_registry.get_cluster_health()

            assert health.healthy is True  # No unhealthy registries (unavailable != unhealthy)
            assert len(health.registries) == 4
            assert all(not r.available for r in health.registries)

    def test_get_cluster_health_partial_availability(self, unified_registry):
        """Test get_cluster_health with some registries available."""
        mock_model_reg = MagicMock()
        mock_model_reg.get_stats.return_value = {"total_models": 5}

        with patch.object(unified_registry, '_get_model_registry', return_value=mock_model_reg), \
             patch.object(unified_registry, '_get_orchestrator_registry', return_value=None), \
             patch.object(unified_registry, '_get_health_registry', return_value=None), \
             patch.object(unified_registry, '_get_dynamic_registry', return_value=None):

            health = unified_registry.get_cluster_health()

            model_health = next(r for r in health.registries if r.name == "model_registry")
            assert model_health.available is True
            assert model_health.healthy is True
            assert model_health.item_count == 5

    def test_is_cluster_healthy(self, unified_registry):
        """Test is_cluster_healthy convenience method."""
        with patch.object(unified_registry, 'get_cluster_health') as mock_health:
            from app.coordination.unified_registry import ClusterHealth
            mock_health.return_value = ClusterHealth(healthy=True)

            assert unified_registry.is_cluster_healthy() is True

            mock_health.return_value = ClusterHealth(healthy=False)
            assert unified_registry.is_cluster_healthy() is False

    def test_get_status_returns_dict(self, unified_registry):
        """Test get_status returns properly structured dict."""
        with patch.object(unified_registry, '_get_model_registry', return_value=None), \
             patch.object(unified_registry, '_get_orchestrator_registry', return_value=None), \
             patch.object(unified_registry, '_get_health_registry', return_value=None), \
             patch.object(unified_registry, '_get_dynamic_registry', return_value=None):

            status = unified_registry.get_status()

            assert "healthy" in status
            assert "total_items" in status
            assert "unhealthy_count" in status
            assert "timestamp" in status
            assert "registries" in status
            assert "init_errors" in status

            # Check registry keys
            assert "model_registry" in status["registries"]
            assert "orchestrator_registry" in status["registries"]
            assert "health_registry" in status["registries"]
            assert "dynamic_registry" in status["registries"]


# =============================================================================
# Singleton Management Tests
# =============================================================================


class TestSingletonManagement:
    """Tests for singleton pattern."""

    def test_get_unified_registry_returns_singleton(self):
        """Test that get_unified_registry returns the same instance."""
        from app.coordination.unified_registry import get_unified_registry, reset_unified_registry

        reset_unified_registry()
        reg1 = get_unified_registry()
        reg2 = get_unified_registry()

        assert reg1 is reg2

    def test_reset_unified_registry(self):
        """Test that reset creates new instance."""
        from app.coordination.unified_registry import get_unified_registry, reset_unified_registry

        reg1 = get_unified_registry()
        reset_unified_registry()
        reg2 = get_unified_registry()

        assert reg1 is not reg2


# =============================================================================
# Subscribe to Changes Tests
# =============================================================================


class TestSubscribeToChanges:
    """Tests for change subscription."""

    def test_subscribe_with_no_registries(self, unified_registry):
        """Test subscribe_to_changes when no registries available."""
        callback = MagicMock()

        with patch.object(unified_registry, '_get_model_registry', return_value=None), \
             patch.object(unified_registry, '_get_orchestrator_registry', return_value=None), \
             patch.object(unified_registry, '_get_health_registry', return_value=None):

            # Should not raise
            unified_registry.subscribe_to_changes(callback)

    def test_subscribe_with_available_registry(self, unified_registry):
        """Test subscribe_to_changes wires callback to registry."""
        mock_model_reg = MagicMock()
        mock_model_reg.on_change = MagicMock()
        callback = MagicMock()

        with patch.object(unified_registry, '_get_model_registry', return_value=mock_model_reg), \
             patch.object(unified_registry, '_get_orchestrator_registry', return_value=None), \
             patch.object(unified_registry, '_get_health_registry', return_value=None):

            unified_registry.subscribe_to_changes(callback)

            mock_model_reg.on_change.assert_called_once()
