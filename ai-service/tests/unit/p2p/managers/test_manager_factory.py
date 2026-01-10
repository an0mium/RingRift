"""Tests for ManagerFactory - P2P manager dependency injection.

Tests cover:
- ManagerConfig dataclass defaults and customization
- ManagerFactory lazy loading and caching
- Circular dependency detection
- Module-level singleton functions
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.p2p.managers.manager_factory import (
    ManagerConfig,
    ManagerFactory,
    get_manager_factory,
    init_manager_factory,
    reset_manager_factory,
)


class TestManagerConfig:
    """Tests for ManagerConfig dataclass."""

    def test_default_values(self):
        """Test ManagerConfig has sensible defaults."""
        config = ManagerConfig()

        assert config.db_path == Path("data/p2p.db")
        assert config.models_dir == Path("models")
        assert config.data_dir == Path("data")
        assert config.node_id == ""
        assert config.is_coordinator is False
        assert config.verbose is False
        assert config.dry_run is False
        assert config.port == 8770
        assert config.bind_address == "0.0.0.0"
        assert config.job_timeout == 3600
        assert config.sync_timeout == 300
        assert config.enable_selfplay is True
        assert config.enable_training is True
        assert config.enable_sync is True
        assert config.orchestrator is None

    def test_custom_values(self):
        """Test ManagerConfig with custom values."""
        config = ManagerConfig(
            db_path=Path("/custom/db.sqlite"),
            node_id="test-node",
            is_coordinator=True,
            verbose=True,
            port=9999,
            enable_selfplay=False,
        )

        assert config.db_path == Path("/custom/db.sqlite")
        assert config.node_id == "test-node"
        assert config.is_coordinator is True
        assert config.verbose is True
        assert config.port == 9999
        assert config.enable_selfplay is False

    def test_orchestrator_reference(self):
        """Test orchestrator can be passed for backward compat."""
        mock_orchestrator = MagicMock()
        config = ManagerConfig(orchestrator=mock_orchestrator)

        assert config.orchestrator is mock_orchestrator


class TestManagerFactory:
    """Tests for ManagerFactory class."""

    def test_init(self):
        """Test factory initialization."""
        config = ManagerConfig(node_id="test")
        factory = ManagerFactory(config)

        assert factory.config is config
        assert factory._managers == {}
        assert factory._creating == set()

    def test_config_property(self):
        """Test config property returns the configuration."""
        config = ManagerConfig(node_id="my-node", verbose=True)
        factory = ManagerFactory(config)

        assert factory.config.node_id == "my-node"
        assert factory.config.verbose is True

    def test_reset_clears_managers(self):
        """Test reset() clears all cached managers."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Manually add some managers
        factory._managers["test"] = "value"

        factory.reset()

        assert factory._managers == {}
        assert factory._creating == set()

    def test_check_cycle_detects_circular_deps(self):
        """Test _check_cycle raises on circular dependency."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Simulate being in the middle of creating manager_a
        factory._creating.add("manager_a")
        factory._creating.add("manager_b")

        # Trying to create manager_a again should raise
        with pytest.raises(RuntimeError, match="Circular dependency detected"):
            factory._check_cycle("manager_a")

    def test_check_cycle_allows_fresh_creation(self):
        """Test _check_cycle allows first-time creation."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Should not raise for fresh creation
        factory._check_cycle("new_manager")

        # Should be tracked
        assert "new_manager" in factory._creating

    def test_done_creating_removes_from_set(self):
        """Test _done_creating removes manager from creating set."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        factory._creating.add("manager_a")
        factory._done_creating("manager_a")

        assert "manager_a" not in factory._creating

    def test_done_creating_safe_for_missing(self):
        """Test _done_creating is safe when manager not in set."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Should not raise
        factory._done_creating("nonexistent")

    @patch("scripts.p2p.managers.StateManager")
    def test_state_manager_lazy_creation(self, mock_state_manager_class):
        """Test state_manager is created lazily."""
        mock_instance = MagicMock()
        mock_state_manager_class.return_value = mock_instance

        config = ManagerConfig(
            db_path=Path("/test/db.sqlite"),
            verbose=True,
        )
        factory = ManagerFactory(config)

        # Not created yet
        assert "state_manager" not in factory._managers

        # Access triggers creation
        manager = factory.state_manager

        assert manager is mock_instance
        assert "state_manager" in factory._managers
        mock_state_manager_class.assert_called_once_with(
            db_path=Path("/test/db.sqlite"),
            verbose=True,
        )

    @patch("scripts.p2p.managers.StateManager")
    def test_state_manager_cached(self, mock_state_manager_class):
        """Test state_manager is cached after first access."""
        mock_instance = MagicMock()
        mock_state_manager_class.return_value = mock_instance

        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Access twice
        manager1 = factory.state_manager
        manager2 = factory.state_manager

        # Same instance returned
        assert manager1 is manager2
        # Only created once
        mock_state_manager_class.assert_called_once()

    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_job_manager_depends_on_state_manager(
        self, mock_state_class, mock_job_class
    ):
        """Test job_manager gets state_manager as dependency."""
        mock_state = MagicMock()
        mock_job = MagicMock()
        mock_state_class.return_value = mock_state
        mock_job_class.return_value = mock_job

        config = ManagerConfig()
        factory = ManagerFactory(config)

        job_manager = factory.job_manager

        assert job_manager is mock_job
        # JobManager should be created with state_manager
        mock_job_class.assert_called_once()
        call_kwargs = mock_job_class.call_args.kwargs
        assert call_kwargs["state_manager"] is mock_state

    @patch("scripts.p2p.managers.TrainingCoordinator")
    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_training_coordinator_depends_on_state_and_job(
        self, mock_state_class, mock_job_class, mock_training_class
    ):
        """Test training_coordinator gets state and job managers."""
        mock_state = MagicMock()
        mock_job = MagicMock()
        mock_training = MagicMock()
        mock_state_class.return_value = mock_state
        mock_job_class.return_value = mock_job
        mock_training_class.return_value = mock_training

        config = ManagerConfig()
        factory = ManagerFactory(config)

        training = factory.training_coordinator

        assert training is mock_training
        call_kwargs = mock_training_class.call_args.kwargs
        assert call_kwargs["state_manager"] is mock_state
        assert call_kwargs["job_manager"] is mock_job

    @patch("scripts.p2p.managers.SelfplayScheduler")
    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_selfplay_scheduler_wires_callbacks(
        self, mock_state_class, mock_job_class, mock_scheduler_class
    ):
        """Test selfplay_scheduler gets callback functions wired."""
        mock_state = MagicMock()
        mock_job = MagicMock()
        mock_scheduler = MagicMock()
        mock_state_class.return_value = mock_state
        mock_job_class.return_value = mock_job
        mock_scheduler_class.return_value = mock_scheduler

        config = ManagerConfig(verbose=True)
        factory = ManagerFactory(config)

        scheduler = factory.selfplay_scheduler

        assert scheduler is mock_scheduler
        call_kwargs = mock_scheduler_class.call_args.kwargs
        assert callable(call_kwargs["get_active_configs_for_node_fn"])
        assert callable(call_kwargs["load_curriculum_weights_fn"])
        assert call_kwargs["verbose"] is True

    @patch("scripts.p2p.managers.SyncPlanner")
    @patch("scripts.p2p.managers.StateManager")
    def test_sync_planner_depends_on_state(self, mock_state_class, mock_sync_class):
        """Test sync_planner gets state_manager."""
        mock_state = MagicMock()
        mock_sync = MagicMock()
        mock_state_class.return_value = mock_state
        mock_sync_class.return_value = mock_sync

        config = ManagerConfig()
        factory = ManagerFactory(config)

        planner = factory.sync_planner

        assert planner is mock_sync
        call_kwargs = mock_sync_class.call_args.kwargs
        assert call_kwargs["state_manager"] is mock_state

    @patch("scripts.p2p.managers.NodeSelector")
    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_node_selector_depends_on_state_and_job(
        self, mock_state_class, mock_job_class, mock_selector_class
    ):
        """Test node_selector gets state and job managers."""
        mock_state = MagicMock()
        mock_job = MagicMock()
        mock_selector = MagicMock()
        mock_state_class.return_value = mock_state
        mock_job_class.return_value = mock_job
        mock_selector_class.return_value = mock_selector

        config = ManagerConfig()
        factory = ManagerFactory(config)

        selector = factory.node_selector

        assert selector is mock_selector
        call_kwargs = mock_selector_class.call_args.kwargs
        assert call_kwargs["state_manager"] is mock_state
        assert call_kwargs["job_manager"] is mock_job

    @patch("scripts.p2p.managers.NodeSelector")
    @patch("scripts.p2p.managers.SyncPlanner")
    @patch("scripts.p2p.managers.SelfplayScheduler")
    @patch("scripts.p2p.managers.TrainingCoordinator")
    @patch("scripts.p2p.managers.JobManager")
    @patch("scripts.p2p.managers.StateManager")
    def test_get_all_managers_forces_init(
        self,
        mock_state_class,
        mock_job_class,
        mock_training_class,
        mock_scheduler_class,
        mock_sync_class,
        mock_selector_class,
    ):
        """Test get_all_managers initializes all managers."""
        mock_state_class.return_value = MagicMock(name="state")
        mock_job_class.return_value = MagicMock(name="job")
        mock_training_class.return_value = MagicMock(name="training")
        mock_scheduler_class.return_value = MagicMock(name="scheduler")
        mock_sync_class.return_value = MagicMock(name="sync")
        mock_selector_class.return_value = MagicMock(name="selector")

        config = ManagerConfig()
        factory = ManagerFactory(config)

        managers = factory.get_all_managers()

        assert len(managers) == 6
        assert "state_manager" in managers
        assert "job_manager" in managers
        assert "training_coordinator" in managers
        assert "selfplay_scheduler" in managers
        assert "sync_planner" in managers
        assert "node_selector" in managers

    @patch("scripts.p2p.managers.StateManager")
    def test_get_initialized_managers_only_returns_accessed(self, mock_state_class):
        """Test get_initialized_managers only returns accessed managers."""
        mock_state = MagicMock()
        mock_state_class.return_value = mock_state

        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Only access state_manager
        _ = factory.state_manager

        initialized = factory.get_initialized_managers()

        assert "state_manager" in initialized
        # Others should not be present
        assert "job_manager" not in initialized
        assert "training_coordinator" not in initialized


class TestModuleLevelFunctions:
    """Tests for module-level singleton functions."""

    def setup_method(self):
        """Reset factory before each test."""
        reset_manager_factory()

    def teardown_method(self):
        """Reset factory after each test."""
        reset_manager_factory()

    def test_get_manager_factory_returns_none_initially(self):
        """Test get_manager_factory returns None before init."""
        factory = get_manager_factory()
        assert factory is None

    def test_init_manager_factory_creates_factory(self):
        """Test init_manager_factory creates and returns factory."""
        config = ManagerConfig(node_id="test-node")
        factory = init_manager_factory(config)

        assert factory is not None
        assert factory.config.node_id == "test-node"

    def test_init_manager_factory_sets_global(self):
        """Test init_manager_factory sets global factory."""
        config = ManagerConfig(node_id="test-node")
        init_manager_factory(config)

        factory = get_manager_factory()
        assert factory is not None
        assert factory.config.node_id == "test-node"

    def test_reset_manager_factory_clears_global(self):
        """Test reset_manager_factory clears global factory."""
        config = ManagerConfig()
        init_manager_factory(config)

        reset_manager_factory()

        assert get_manager_factory() is None

    def test_reset_manager_factory_calls_reset(self):
        """Test reset_manager_factory calls factory.reset()."""
        config = ManagerConfig()
        factory = init_manager_factory(config)

        # Add something to managers
        factory._managers["test"] = "value"

        reset_manager_factory()

        # Global is cleared
        assert get_manager_factory() is None

    def test_reset_manager_factory_safe_when_none(self):
        """Test reset_manager_factory is safe when no factory exists."""
        # Should not raise
        reset_manager_factory()
        reset_manager_factory()  # Safe to call multiple times

    def test_multiple_init_replaces_factory(self):
        """Test multiple init calls replace the factory."""
        config1 = ManagerConfig(node_id="node-1")
        config2 = ManagerConfig(node_id="node-2")

        factory1 = init_manager_factory(config1)
        factory2 = init_manager_factory(config2)

        # Second call replaces first
        assert get_manager_factory() is factory2
        assert get_manager_factory().config.node_id == "node-2"


class TestCircularDependencyDetection:
    """Tests for circular dependency detection edge cases."""

    def test_nested_dependency_chain_works(self):
        """Test normal dependency chains don't trigger cycle detection."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Simulate a chain: A depends on B depends on C
        factory._check_cycle("C")
        factory._done_creating("C")

        factory._check_cycle("B")
        factory._done_creating("B")

        factory._check_cycle("A")
        factory._done_creating("A")

        # All completed successfully
        assert factory._creating == set()

    def test_parallel_managers_work(self):
        """Test creating independent managers doesn't trigger cycle."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        # Start creating A
        factory._check_cycle("A")

        # B is independent, should work
        factory._check_cycle("B")

        # Both in creating set
        assert "A" in factory._creating
        assert "B" in factory._creating

        # Complete both
        factory._done_creating("A")
        factory._done_creating("B")
        assert factory._creating == set()

    def test_cycle_error_includes_path(self):
        """Test cycle error message includes dependency path."""
        config = ManagerConfig()
        factory = ManagerFactory(config)

        factory._creating.add("state_manager")
        factory._creating.add("job_manager")

        try:
            factory._check_cycle("state_manager")
            pytest.fail("Should have raised RuntimeError")
        except RuntimeError as e:
            error_msg = str(e)
            assert "state_manager" in error_msg
            assert "job_manager" in error_msg
            assert "Circular dependency detected" in error_msg
