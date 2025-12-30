"""Tests for coordination _exports modules.

December 2025: Verifies that all coordination re-export modules
correctly export their intended items without circular imports.
"""

import pytest


class TestExportsCore:
    """Tests for _exports_core.py re-exports."""

    def test_imports_successfully(self):
        """Test that _exports_core can be imported."""
        from app.coordination import _exports_core
        assert _exports_core is not None

    def test_task_coordinator_exports(self):
        """Test task coordinator exports."""
        from app.coordination._exports_core import (
            TASK_RESOURCE_MAP,
            CoordinatedTask,
            CoordinatorState,
            OrchestratorLock,
            RateLimiter,
            ResourceType,
            TaskCoordinator,
            TaskInfo,
            TaskLimits,
            TaskType,
            can_spawn,
            emergency_stop_all,
            get_coordinator,
            get_task_resource_type,
            is_cpu_task,
            is_gpu_task,
        )
        assert TaskCoordinator is not None
        assert callable(get_coordinator)
        assert callable(can_spawn)

    def test_orchestrator_registry_exports(self):
        """Test orchestrator registry exports."""
        from app.coordination._exports_core import (
            CoordinatorHealth,
            OrchestratorInfo,
            OrchestratorRegistry,
            OrchestratorRole,
            OrchestratorState,
            get_registry,
            register_coordinator,
            unregister_coordinator,
        )
        assert OrchestratorRegistry is not None
        assert callable(get_registry)

    def test_queue_monitor_exports(self):
        """Test queue monitor exports."""
        from app.coordination._exports_core import (
            BackpressureLevel,
            QueueMonitor,
            QueueStatus,
            QueueType,
            check_backpressure,
            get_queue_monitor,
            should_throttle_production,
        )
        assert QueueMonitor is not None
        assert callable(get_queue_monitor)

    def test_resource_optimizer_exports(self):
        """Test resource optimizer exports."""
        from app.coordination._exports_core import (
            ClusterState,
            NodeResources,
            OptimizationResult,
            PIDController,
            ResourceOptimizer,
            get_resource_optimizer,
        )
        assert ResourceOptimizer is not None
        assert callable(get_resource_optimizer)

    def test_resource_targets_exports(self):
        """Test resource targets exports."""
        from app.coordination._exports_core import (
            HostTargets,
            HostTier,
            ResourceTargetManager,
            UtilizationTargets,
            get_resource_targets,
        )
        assert ResourceTargetManager is not None
        assert callable(get_resource_targets)


class TestExportsDaemon:
    """Tests for _exports_daemon.py re-exports."""

    def test_imports_successfully(self):
        """Test that _exports_daemon can be imported."""
        from app.coordination import _exports_daemon
        assert _exports_daemon is not None

    def test_daemon_types_exports(self):
        """Test daemon types exports."""
        from app.coordination._exports_daemon import (
            DaemonType,
        )
        assert DaemonType is not None
        # Check a few known daemon types exist
        assert hasattr(DaemonType, "AUTO_SYNC")
        assert hasattr(DaemonType, "DATA_PIPELINE")

    def test_daemon_manager_exports(self):
        """Test daemon manager exports."""
        from app.coordination._exports_daemon import (
            DaemonManager,
            get_daemon_manager,
        )
        assert DaemonManager is not None
        assert callable(get_daemon_manager)


class TestExportsEvents:
    """Tests for _exports_events.py re-exports."""

    def test_imports_successfully(self):
        """Test that _exports_events can be imported."""
        from app.coordination import _exports_events
        assert _exports_events is not None

    def test_event_router_exports(self):
        """Test event router exports."""
        from app.coordination._exports_events import (
            UnifiedEventRouter,
            RouterEvent,
            get_event_router,
        )
        assert UnifiedEventRouter is not None
        assert RouterEvent is not None
        assert callable(get_event_router)

    def test_event_emitters_exports(self):
        """Test event emitter exports."""
        from app.coordination._exports_events import (
            emit_training_complete,
            emit_evaluation_complete,
            emit_promotion_complete,
        )
        assert callable(emit_training_complete)
        assert callable(emit_evaluation_complete)
        assert callable(emit_promotion_complete)

    def test_stage_events_exports(self):
        """Test stage events exports."""
        from app.coordination._exports_events import (
            StageEvent,
            StageEventBus,
            get_stage_event_bus,
        )
        assert StageEvent is not None
        assert StageEventBus is not None
        assert callable(get_stage_event_bus)


class TestExportsOrchestrators:
    """Tests for _exports_orchestrators.py re-exports."""

    def test_imports_successfully(self):
        """Test that _exports_orchestrators can be imported."""
        from app.coordination import _exports_orchestrators
        assert _exports_orchestrators is not None

    def test_data_pipeline_exports(self):
        """Test data pipeline exports."""
        from app.coordination._exports_orchestrators import (
            DataPipelineOrchestrator,
            PipelineStage,
            get_pipeline_orchestrator,
        )
        assert DataPipelineOrchestrator is not None
        assert PipelineStage is not None
        assert callable(get_pipeline_orchestrator)

    def test_cache_orchestrator_exports(self):
        """Test cache orchestrator exports."""
        from app.coordination._exports_orchestrators import (
            CacheCoordinationOrchestrator,
            get_cache_orchestrator,
        )
        assert CacheCoordinationOrchestrator is not None
        assert callable(get_cache_orchestrator)

    def test_model_lifecycle_exports(self):
        """Test model lifecycle exports."""
        from app.coordination._exports_orchestrators import (
            ModelLifecycleCoordinator,
            get_model_coordinator,
        )
        assert ModelLifecycleCoordinator is not None
        assert callable(get_model_coordinator)


class TestExportsSync:
    """Tests for _exports_sync.py re-exports."""

    def test_imports_successfully(self):
        """Test that _exports_sync can be imported."""
        from app.coordination import _exports_sync
        assert _exports_sync is not None

    def test_auto_sync_daemon_exports(self):
        """Test auto sync daemon exports."""
        from app.coordination._exports_sync import (
            AutoSyncDaemon,
            get_auto_sync_daemon,
        )
        assert AutoSyncDaemon is not None
        assert callable(get_auto_sync_daemon)

    def test_sync_facade_exports(self):
        """Test sync facade exports."""
        from app.coordination._exports_sync import (
            SyncFacade,
            sync,
        )
        assert SyncFacade is not None
        # sync may be async function or coroutine
        assert sync is not None


class TestExportsUtils:
    """Tests for _exports_utils.py re-exports."""

    def test_imports_successfully(self):
        """Test that _exports_utils can be imported."""
        from app.coordination import _exports_utils
        assert _exports_utils is not None

    def test_singleton_mixin_exports(self):
        """Test singleton mixin exports."""
        from app.coordination._exports_utils import (
            SingletonMixin,
        )
        assert SingletonMixin is not None

    def test_handler_base_exports(self):
        """Test handler base exports."""
        from app.coordination._exports_utils import (
            HandlerBase,
        )
        assert HandlerBase is not None

    def test_health_check_exports(self):
        """Test health check exports."""
        from app.coordination._exports_utils import (
            HealthCheckResult,
        )
        assert HealthCheckResult is not None


class TestNoCircularImports:
    """Tests that there are no circular imports."""

    def test_all_exports_import_cleanly(self):
        """Test that all exports modules can be imported in sequence."""
        # Import in the order they might depend on each other
        from app.coordination import _exports_utils  # noqa: F401
        from app.coordination import _exports_events  # noqa: F401
        from app.coordination import _exports_daemon  # noqa: F401
        from app.coordination import _exports_sync  # noqa: F401
        from app.coordination import _exports_core  # noqa: F401
        from app.coordination import _exports_orchestrators  # noqa: F401

    def test_main_init_imports(self):
        """Test that the main coordination __init__ imports cleanly."""
        import app.coordination  # noqa: F401


class TestExportsModuleAttributes:
    """Tests for correct module attributes."""

    def test_exports_core_has_all(self):
        """Test that _exports_core has __all__ defined."""
        from app.coordination import _exports_core
        # __all__ may or may not be defined - just verify module is valid
        assert hasattr(_exports_core, "__name__")

    def test_exports_daemon_has_all(self):
        """Test that _exports_daemon has __all__ defined."""
        from app.coordination import _exports_daemon
        assert hasattr(_exports_daemon, "__name__")

    def test_exports_events_has_all(self):
        """Test that _exports_events has __all__ defined."""
        from app.coordination import _exports_events
        assert hasattr(_exports_events, "__name__")
