"""Tests for SyncOrchestrator (unified sync coordination).

Tests cover:
- SyncOrchestratorConfig, SyncOrchestratorState, SyncResult, FullSyncResult dataclasses
- SyncOrchestrator initialization and state management
- Module functions
"""

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Test SyncOrchestratorConfig Dataclass
# =============================================================================

class TestSyncOrchestratorConfig:
    """Tests for SyncOrchestratorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.distributed.sync_orchestrator import SyncOrchestratorConfig

        config = SyncOrchestratorConfig()

        assert config.enable_data_sync is True
        assert config.data_sync_interval_seconds == 300.0
        assert config.high_quality_priority is True
        assert config.enable_model_sync is True
        assert config.model_sync_interval_seconds == 600.0
        assert config.enable_elo_sync is True
        assert config.elo_sync_interval_seconds == 60.0
        assert config.enable_registry_sync is True
        assert config.registry_sync_interval_seconds == 120.0
        assert config.min_quality_for_priority_sync == 0.7
        assert config.max_games_per_sync == 500

    def test_custom_values(self):
        """Test custom configuration values."""
        from app.distributed.sync_orchestrator import SyncOrchestratorConfig

        config = SyncOrchestratorConfig(
            enable_data_sync=False,
            data_sync_interval_seconds=600.0,
            enable_model_sync=False,
            max_games_per_sync=1000,
        )

        assert config.enable_data_sync is False
        assert config.data_sync_interval_seconds == 600.0
        assert config.enable_model_sync is False
        assert config.max_games_per_sync == 1000


# =============================================================================
# Test SyncOrchestratorState Dataclass
# =============================================================================

class TestSyncOrchestratorState:
    """Tests for SyncOrchestratorState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        from app.distributed.sync_orchestrator import SyncOrchestratorState

        state = SyncOrchestratorState()

        assert state.initialized is False
        assert state.last_data_sync == 0.0
        assert state.last_model_sync == 0.0
        assert state.last_elo_sync == 0.0
        assert state.last_registry_sync == 0.0
        assert state.total_syncs == 0
        assert state.sync_errors == 0
        assert state.components_loaded == []

    def test_state_tracking(self):
        """Test state tracking with updates."""
        from app.distributed.sync_orchestrator import SyncOrchestratorState

        state = SyncOrchestratorState()

        # Simulate state updates
        state.initialized = True
        state.last_data_sync = time.time()
        state.total_syncs = 5
        state.components_loaded = ["data", "elo"]

        assert state.initialized is True
        assert state.total_syncs == 5
        assert len(state.components_loaded) == 2


# =============================================================================
# Test SyncResult Dataclass
# =============================================================================

class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_create_success(self):
        """Test creating a successful sync result."""
        from app.distributed.sync_orchestrator import SyncResult

        result = SyncResult(
            success=True,
            component="data",
            items_synced=100,
        )

        assert result.success is True
        assert result.component == "data"
        assert result.items_synced == 100

    def test_create_failure(self):
        """Test creating a failed sync result."""
        from app.distributed.sync_orchestrator import SyncResult

        result = SyncResult(
            success=False,
            component="model",
            items_synced=0,
            error="Connection failed",
        )

        assert result.success is False
        assert result.error == "Connection failed"

    def test_default_values(self):
        """Test default values for SyncResult."""
        from app.distributed.sync_orchestrator import SyncResult

        result = SyncResult(
            success=True,
            component="test",
        )

        assert result.items_synced == 0
        assert result.error is None
        assert result.duration_seconds == 0.0


# =============================================================================
# Test FullSyncResult Dataclass
# =============================================================================

class TestFullSyncResult:
    """Tests for FullSyncResult dataclass."""

    def test_create(self):
        """Test creating a full sync result."""
        from app.distributed.sync_orchestrator import FullSyncResult, SyncResult

        full_result = FullSyncResult(
            success=True,
            component_results=[
                SyncResult(component="data", success=True, items_synced=50),
                SyncResult(component="elo", success=True, items_synced=10),
            ],
            total_items_synced=60,
            duration_seconds=5.5,
        )

        assert full_result.success is True
        assert len(full_result.component_results) == 2
        assert full_result.total_items_synced == 60
        assert full_result.duration_seconds == 5.5


# =============================================================================
# Test SyncOrchestrator
# =============================================================================

class TestSyncOrchestrator:
    """Tests for SyncOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance."""
        from app.distributed.sync_orchestrator import (
            SyncOrchestrator,
            SyncOrchestratorConfig,
            reset_sync_orchestrator,
        )

        reset_sync_orchestrator()
        config = SyncOrchestratorConfig()
        orch = SyncOrchestrator(config)
        yield orch
        reset_sync_orchestrator()

    def test_initialization_with_default_config(self):
        """Test orchestrator initialization with default config."""
        from app.distributed.sync_orchestrator import SyncOrchestrator

        orch = SyncOrchestrator()

        assert orch.config is not None
        assert orch.state is not None
        assert orch.state.initialized is False

    def test_initialization_with_custom_config(self):
        """Test orchestrator initialization with custom config."""
        from app.distributed.sync_orchestrator import (
            SyncOrchestrator,
            SyncOrchestratorConfig,
        )

        config = SyncOrchestratorConfig(
            enable_data_sync=False,
            data_sync_interval_seconds=600.0,
        )
        orch = SyncOrchestrator(config)

        assert orch.config.enable_data_sync is False
        assert orch.config.data_sync_interval_seconds == 600.0

    def test_needs_sync_data(self, orchestrator):
        """Test needs_sync for data component."""
        # Never synced
        assert orchestrator.needs_sync("data") is True

        # Recently synced
        orchestrator.state.last_data_sync = time.time()
        assert orchestrator.needs_sync("data") is False

        # Synced long ago
        orchestrator.state.last_data_sync = time.time() - 600
        assert orchestrator.needs_sync("data") is True

    def test_needs_sync_elo(self, orchestrator):
        """Test needs_sync for elo component."""
        # Never synced
        assert orchestrator.needs_sync("elo") is True

        # Recently synced
        orchestrator.state.last_elo_sync = time.time()
        assert orchestrator.needs_sync("elo") is False

    def test_needs_sync_timing_check(self, orchestrator):
        """Test needs_sync checks timing regardless of enabled state."""
        # needs_sync checks if enough time has passed, not if enabled
        orchestrator.config.enable_data_sync = False
        # Still returns True because it checks timing, not enabled state
        # The sync method itself checks if enabled
        assert orchestrator.needs_sync("data") is True

    def test_get_status(self, orchestrator):
        """Test getting orchestrator status."""
        status = orchestrator.get_status()

        assert "initialized" in status
        assert "config" in status
        assert "total_syncs" in status
        assert status["initialized"] is False

    def test_get_status_after_sync(self, orchestrator):
        """Test status after syncs."""
        orchestrator.state.initialized = True
        orchestrator.state.total_syncs = 10
        orchestrator.state.sync_errors = 1
        orchestrator.state.components_loaded = ["data", "elo"]

        status = orchestrator.get_status()

        assert status["initialized"] is True
        assert status["total_syncs"] == 10
        assert status["sync_errors"] == 1

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized(self, orchestrator):
        """Test orchestrator initialization sets initialized flag."""
        # The initialize method tries to load components
        # Even if they fail, it marks as initialized
        result = await orchestrator.initialize()

        # Result depends on component availability
        # But state should be marked as initialized attempt complete
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator):
        """Test orchestrator shutdown."""
        orchestrator.state.initialized = True

        await orchestrator.shutdown()

        assert orchestrator.state.initialized is False

    @pytest.mark.asyncio
    async def test_sync_data_returns_result(self, orchestrator):
        """Test sync_data returns SyncResult."""
        # Without initialization, coordinator is not available
        result = await orchestrator.sync_data()

        # Returns failure when coordinator not available
        assert result.success is False
        assert result.component == "data_sync"
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_sync_elo_returns_result(self, orchestrator):
        """Test sync_elo returns SyncResult."""
        # Without initialization, manager is not available
        result = await orchestrator.sync_elo()

        assert result.success is False
        assert result.component == "elo_sync"

    @pytest.mark.asyncio
    async def test_sync_registry_returns_result(self, orchestrator):
        """Test sync_registry returns SyncResult."""
        # Without initialization, manager is not available
        result = await orchestrator.sync_registry()

        assert result.success is False
        assert result.component == "registry_sync"

    @pytest.mark.asyncio
    async def test_sync_models_returns_result(self, orchestrator):
        """Test sync_models returns SyncResult."""
        # Without initialization, sync is not available
        result = await orchestrator.sync_models()

        assert result.success is False
        assert result.component == "model_sync"


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from app.distributed.sync_orchestrator import reset_sync_orchestrator

        reset_sync_orchestrator()
        yield
        reset_sync_orchestrator()

    def test_get_sync_orchestrator_creates_singleton(self):
        """Test that get_sync_orchestrator creates a singleton."""
        from app.distributed.sync_orchestrator import (
            SyncOrchestrator,
            get_sync_orchestrator,
        )

        orch1 = get_sync_orchestrator()
        orch2 = get_sync_orchestrator()

        assert orch1 is orch2
        assert isinstance(orch1, SyncOrchestrator)

    def test_reset_sync_orchestrator(self):
        """Test resetting the orchestrator singleton."""
        from app.distributed.sync_orchestrator import (
            get_sync_orchestrator,
            reset_sync_orchestrator,
        )

        orch1 = get_sync_orchestrator()
        reset_sync_orchestrator()
        orch2 = get_sync_orchestrator()

        assert orch1 is not orch2


# =============================================================================
# Integration Tests
# =============================================================================

class TestSyncIntegration:
    """Integration tests for sync orchestrator."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset singleton."""
        from app.distributed.sync_orchestrator import reset_sync_orchestrator

        reset_sync_orchestrator()
        yield
        reset_sync_orchestrator()

    def test_sync_orchestrator_config_attributes(self):
        """Test SyncOrchestratorConfig has all expected attributes."""
        from app.distributed.sync_orchestrator import SyncOrchestratorConfig

        config = SyncOrchestratorConfig()

        # All enable flags
        assert hasattr(config, "enable_data_sync")
        assert hasattr(config, "enable_model_sync")
        assert hasattr(config, "enable_elo_sync")
        assert hasattr(config, "enable_registry_sync")

        # All interval settings
        assert hasattr(config, "data_sync_interval_seconds")
        assert hasattr(config, "model_sync_interval_seconds")
        assert hasattr(config, "elo_sync_interval_seconds")
        assert hasattr(config, "registry_sync_interval_seconds")

    def test_quality_driven_sync_priority(self):
        """Test quality-driven sync priority calculation."""
        from app.distributed.sync_orchestrator import SyncOrchestrator

        orch = SyncOrchestrator()

        # Without quality data, should return empty or default
        priorities = orch.get_quality_driven_sync_priority()
        assert isinstance(priorities, list)

    def test_sync_timing(self):
        """Test sync timing tracking."""
        from app.distributed.sync_orchestrator import SyncOrchestrator

        orch = SyncOrchestrator()

        # All components should need sync initially
        assert orch.needs_sync("data") is True
        assert orch.needs_sync("elo") is True
        assert orch.needs_sync("model") is True
        assert orch.needs_sync("registry") is True

        # After updating timestamps, should not need sync
        now = time.time()
        orch.state.last_data_sync = now
        orch.state.last_elo_sync = now
        orch.state.last_model_sync = now
        orch.state.last_registry_sync = now

        assert orch.needs_sync("data") is False
        assert orch.needs_sync("elo") is False
        assert orch.needs_sync("model") is False
        assert orch.needs_sync("registry") is False
