"""Coordination Bootstrap - Unified initialization for all coordinators (December 2025).

This module provides a single entry point for initializing the entire
coordination layer. It handles the correct initialization order, event
wiring, and registry registration for all coordinators.

Usage:
    from app.coordination.coordination_bootstrap import (
        bootstrap_coordination,
        shutdown_coordination,
        get_bootstrap_status,
    )

    # Initialize all coordination components
    bootstrap_coordination()

    # Or initialize specific components
    bootstrap_coordination(
        enable_metrics=True,
        enable_optimization=True,
        enable_leadership=False,  # Disable if single-node
    )

    # Check initialization status
    status = get_bootstrap_status()
    print(f"Coordinators initialized: {status['initialized_count']}")

    # Graceful shutdown
    shutdown_coordination()

Benefits:
- Single entry point for coordination initialization
- Correct initialization order (dependencies first)
- Consistent error handling across all coordinators
- Unified status reporting
- Graceful shutdown support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Initialization State
# =============================================================================

@dataclass
class CoordinatorStatus:
    """Status of a single coordinator."""

    name: str
    initialized: bool = False
    subscribed: bool = False
    error: Optional[str] = None
    initialized_at: Optional[datetime] = None


@dataclass
class BootstrapState:
    """Global bootstrap state."""

    initialized: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    coordinators: Dict[str, CoordinatorStatus] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    shutdown_requested: bool = False


_state = BootstrapState()


# =============================================================================
# Initialization Functions
# =============================================================================

def _init_resource_coordinator() -> CoordinatorStatus:
    """Initialize ResourceMonitoringCoordinator."""
    status = CoordinatorStatus(name="resource_coordinator")
    try:
        from app.coordination.resource_monitoring_coordinator import wire_resource_events

        coordinator = wire_resource_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] ResourceMonitoringCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] ResourceMonitoringCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize ResourceMonitoringCoordinator: {e}")

    return status


def _init_metrics_orchestrator() -> CoordinatorStatus:
    """Initialize MetricsAnalysisOrchestrator."""
    status = CoordinatorStatus(name="metrics_orchestrator")
    try:
        from app.coordination.metrics_analysis_orchestrator import wire_metrics_events

        orchestrator = wire_metrics_events()
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] MetricsAnalysisOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] MetricsAnalysisOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize MetricsAnalysisOrchestrator: {e}")

    return status


def _init_optimization_coordinator() -> CoordinatorStatus:
    """Initialize OptimizationCoordinator."""
    status = CoordinatorStatus(name="optimization_coordinator")
    try:
        from app.coordination.optimization_coordinator import wire_optimization_events

        coordinator = wire_optimization_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] OptimizationCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] OptimizationCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize OptimizationCoordinator: {e}")

    return status


def _init_cache_orchestrator() -> CoordinatorStatus:
    """Initialize CacheCoordinationOrchestrator."""
    status = CoordinatorStatus(name="cache_orchestrator")
    try:
        from app.coordination.cache_coordination_orchestrator import wire_cache_events

        orchestrator = wire_cache_events()
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] CacheCoordinationOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] CacheCoordinationOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize CacheCoordinationOrchestrator: {e}")

    return status


def _init_model_coordinator() -> CoordinatorStatus:
    """Initialize ModelLifecycleCoordinator."""
    status = CoordinatorStatus(name="model_coordinator")
    try:
        from app.coordination.model_lifecycle_coordinator import wire_model_events

        coordinator = wire_model_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] ModelLifecycleCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] ModelLifecycleCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize ModelLifecycleCoordinator: {e}")

    return status


def _init_error_coordinator() -> CoordinatorStatus:
    """Initialize ErrorRecoveryCoordinator."""
    status = CoordinatorStatus(name="error_coordinator")
    try:
        from app.coordination.error_recovery_coordinator import wire_error_events

        coordinator = wire_error_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] ErrorRecoveryCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] ErrorRecoveryCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize ErrorRecoveryCoordinator: {e}")

    return status


def _init_leadership_coordinator() -> CoordinatorStatus:
    """Initialize LeadershipCoordinator."""
    status = CoordinatorStatus(name="leadership_coordinator")
    try:
        from app.coordination.leadership_coordinator import wire_leadership_events

        coordinator = wire_leadership_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] LeadershipCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] LeadershipCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize LeadershipCoordinator: {e}")

    return status


def _init_selfplay_orchestrator() -> CoordinatorStatus:
    """Initialize SelfplayOrchestrator."""
    status = CoordinatorStatus(name="selfplay_orchestrator")
    try:
        from app.coordination.selfplay_orchestrator import wire_selfplay_events

        orchestrator = wire_selfplay_events()
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] SelfplayOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] SelfplayOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize SelfplayOrchestrator: {e}")

    return status


def _init_pipeline_orchestrator(auto_trigger: bool = False) -> CoordinatorStatus:
    """Initialize DataPipelineOrchestrator."""
    status = CoordinatorStatus(name="pipeline_orchestrator")
    try:
        from app.coordination.data_pipeline_orchestrator import wire_pipeline_events

        orchestrator = wire_pipeline_events(auto_trigger=auto_trigger)
        status.initialized = True
        status.subscribed = orchestrator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] DataPipelineOrchestrator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] DataPipelineOrchestrator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize DataPipelineOrchestrator: {e}")

    return status


def _init_task_coordinator() -> CoordinatorStatus:
    """Initialize TaskLifecycleCoordinator."""
    status = CoordinatorStatus(name="task_coordinator")
    try:
        from app.coordination.task_lifecycle_coordinator import wire_task_events

        coordinator = wire_task_events()
        status.initialized = True
        status.subscribed = coordinator._subscribed
        status.initialized_at = datetime.now()
        logger.info("[Bootstrap] TaskLifecycleCoordinator initialized")

    except ImportError as e:
        status.error = f"Import error: {e}"
        logger.warning(f"[Bootstrap] TaskLifecycleCoordinator not available: {e}")
    except Exception as e:
        status.error = str(e)
        logger.error(f"[Bootstrap] Failed to initialize TaskLifecycleCoordinator: {e}")

    return status


def _register_coordinators() -> bool:
    """Register all coordinators with OrchestratorRegistry."""
    try:
        from app.coordination.orchestrator_registry import auto_register_known_coordinators

        count = auto_register_known_coordinators()
        logger.info(f"[Bootstrap] Registered {count} coordinators with registry")
        return True

    except ImportError:
        logger.warning("[Bootstrap] OrchestratorRegistry not available")
        return False
    except Exception as e:
        logger.error(f"[Bootstrap] Failed to register coordinators: {e}")
        return False


# =============================================================================
# Main Bootstrap Function
# =============================================================================

def bootstrap_coordination(
    enable_resources: bool = True,
    enable_metrics: bool = True,
    enable_optimization: bool = True,
    enable_cache: bool = True,
    enable_model: bool = True,
    enable_error: bool = True,
    enable_leadership: bool = True,
    enable_selfplay: bool = True,
    enable_pipeline: bool = True,
    enable_task: bool = True,
    pipeline_auto_trigger: bool = False,
    register_with_registry: bool = True,
) -> Dict[str, Any]:
    """Initialize all coordination components.

    Initializes coordinators in the correct dependency order per coordinator_dependencies.py:
    1. Task lifecycle (foundational - no dependencies)
    2. Resource monitoring (foundational - no dependencies)
    3. Cache coordination (foundational - no dependencies)
    4. Error recovery (infrastructure support)
    5. Model lifecycle (depends on cache)
    6. Selfplay orchestrator (depends on task_lifecycle, resources)
    7. Pipeline orchestrator (depends on selfplay, cache)
    8. Metrics analysis (depends on pipeline)
    9. Optimization (depends on metrics)
    10. Leadership (depends on all others)

    Args:
        enable_resources: Initialize ResourceMonitoringCoordinator
        enable_metrics: Initialize MetricsAnalysisOrchestrator
        enable_optimization: Initialize OptimizationCoordinator
        enable_cache: Initialize CacheCoordinationOrchestrator
        enable_model: Initialize ModelLifecycleCoordinator
        enable_error: Initialize ErrorRecoveryCoordinator
        enable_leadership: Initialize LeadershipCoordinator
        enable_selfplay: Initialize SelfplayOrchestrator
        enable_pipeline: Initialize DataPipelineOrchestrator
        enable_task: Initialize TaskLifecycleCoordinator
        pipeline_auto_trigger: Auto-trigger pipeline on events
        register_with_registry: Register coordinators with OrchestratorRegistry

    Returns:
        Status dict with initialization results
    """
    global _state

    if _state.initialized:
        logger.warning("[Bootstrap] Coordination already initialized, skipping")
        return get_bootstrap_status()

    _state.started_at = datetime.now()
    _state.errors = []

    logger.info("[Bootstrap] Starting coordination bootstrap...")

    # Initialize in dependency order per coordinator_dependencies.py
    # Foundational coordinators first (no dependencies), then dependents
    init_order = [
        # Foundational layer (no dependencies)
        ("task_coordinator", enable_task, _init_task_coordinator),
        ("resource_coordinator", enable_resources, _init_resource_coordinator),
        ("cache_orchestrator", enable_cache, _init_cache_orchestrator),
        # Infrastructure support layer
        ("error_coordinator", enable_error, _init_error_coordinator),
        ("model_coordinator", enable_model, _init_model_coordinator),
        # Selfplay layer (depends on task_lifecycle, resources)
        ("selfplay_orchestrator", enable_selfplay, _init_selfplay_orchestrator),
        # Pipeline layer (depends on selfplay, cache)
        ("pipeline_orchestrator", enable_pipeline, lambda: _init_pipeline_orchestrator(pipeline_auto_trigger)),
        # Metrics layer (depends on pipeline)
        ("metrics_orchestrator", enable_metrics, _init_metrics_orchestrator),
        # Optimization layer (depends on metrics)
        ("optimization_coordinator", enable_optimization, _init_optimization_coordinator),
        # Leadership layer (coordinates all others)
        ("leadership_coordinator", enable_leadership, _init_leadership_coordinator),
    ]

    for name, enabled, init_func in init_order:
        if not enabled:
            logger.debug(f"[Bootstrap] Skipping {name} (disabled)")
            continue

        status = init_func()
        _state.coordinators[name] = status

        if status.error:
            _state.errors.append(f"{name}: {status.error}")

    # Register with OrchestratorRegistry
    if register_with_registry:
        _register_coordinators()

    _state.initialized = True
    _state.completed_at = datetime.now()

    # Log summary
    initialized_count = sum(1 for s in _state.coordinators.values() if s.initialized)
    total_count = len(_state.coordinators)
    error_count = len(_state.errors)

    if error_count > 0:
        logger.warning(
            f"[Bootstrap] Coordination bootstrap completed with errors: "
            f"{initialized_count}/{total_count} coordinators, {error_count} errors"
        )
    else:
        logger.info(
            f"[Bootstrap] Coordination bootstrap completed: "
            f"{initialized_count}/{total_count} coordinators initialized"
        )

    return get_bootstrap_status()


def shutdown_coordination() -> Dict[str, Any]:
    """Gracefully shutdown all coordination components.

    Returns:
        Status dict with shutdown results
    """
    global _state

    if not _state.initialized:
        logger.warning("[Bootstrap] Coordination not initialized, nothing to shutdown")
        return {"shutdown": False, "reason": "not initialized"}

    _state.shutdown_requested = True
    logger.info("[Bootstrap] Starting coordination shutdown...")

    # Shutdown coordinators in reverse of initialization order
    shutdown_order = [
        "leadership_coordinator",
        "optimization_coordinator",
        "metrics_orchestrator",
        "pipeline_orchestrator",
        "selfplay_orchestrator",
        "model_coordinator",
        "error_coordinator",
        "cache_orchestrator",
        "resource_coordinator",
        "task_coordinator",
    ]

    shutdown_results: Dict[str, bool] = {}

    for name in shutdown_order:
        if name not in _state.coordinators:
            continue

        status = _state.coordinators[name]
        if not status.initialized:
            continue

        try:
            # Try to get coordinator and call shutdown if available
            if name == "resource_coordinator":
                from app.coordination.resource_monitoring_coordinator import get_resource_coordinator
                coordinator = get_resource_coordinator()
            elif name == "metrics_orchestrator":
                from app.coordination.metrics_analysis_orchestrator import get_metrics_orchestrator
                coordinator = get_metrics_orchestrator()
            elif name == "optimization_coordinator":
                from app.coordination.optimization_coordinator import get_optimization_coordinator
                coordinator = get_optimization_coordinator()
            elif name == "cache_orchestrator":
                from app.coordination.cache_coordination_orchestrator import get_cache_orchestrator
                coordinator = get_cache_orchestrator()
            elif name == "model_coordinator":
                from app.coordination.model_lifecycle_coordinator import get_model_coordinator
                coordinator = get_model_coordinator()
            elif name == "error_coordinator":
                from app.coordination.error_recovery_coordinator import get_error_coordinator
                coordinator = get_error_coordinator()
            elif name == "leadership_coordinator":
                from app.coordination.leadership_coordinator import get_leadership_coordinator
                coordinator = get_leadership_coordinator()
            elif name == "selfplay_orchestrator":
                from app.coordination.selfplay_orchestrator import get_selfplay_orchestrator
                coordinator = get_selfplay_orchestrator()
            elif name == "pipeline_orchestrator":
                from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
                coordinator = get_pipeline_orchestrator()
            elif name == "task_coordinator":
                from app.coordination.task_lifecycle_coordinator import get_task_lifecycle_coordinator
                coordinator = get_task_lifecycle_coordinator()
            else:
                coordinator = None

            # Call shutdown method if available
            if coordinator and hasattr(coordinator, "shutdown"):
                coordinator.shutdown()
                logger.debug(f"[Bootstrap] Shutdown {name}")

            shutdown_results[name] = True

        except Exception as e:
            logger.error(f"[Bootstrap] Error shutting down {name}: {e}")
            shutdown_results[name] = False

    logger.info("[Bootstrap] Coordination shutdown completed")

    return {
        "shutdown": True,
        "results": shutdown_results,
        "successful": sum(1 for v in shutdown_results.values() if v),
        "failed": sum(1 for v in shutdown_results.values() if not v),
    }


def get_bootstrap_status() -> Dict[str, Any]:
    """Get current bootstrap status.

    Returns:
        Status dict with initialization details
    """
    global _state

    coordinators_summary = {
        name: {
            "initialized": status.initialized,
            "subscribed": status.subscribed,
            "error": status.error,
        }
        for name, status in _state.coordinators.items()
    }

    initialized_count = sum(1 for s in _state.coordinators.values() if s.initialized)
    subscribed_count = sum(1 for s in _state.coordinators.values() if s.subscribed)

    return {
        "initialized": _state.initialized,
        "started_at": _state.started_at.isoformat() if _state.started_at else None,
        "completed_at": _state.completed_at.isoformat() if _state.completed_at else None,
        "initialized_count": initialized_count,
        "subscribed_count": subscribed_count,
        "total_count": len(_state.coordinators),
        "coordinators": coordinators_summary,
        "errors": _state.errors,
        "shutdown_requested": _state.shutdown_requested,
    }


def is_coordination_ready() -> bool:
    """Check if coordination layer is ready for use.

    Returns:
        True if at least the core coordinators are initialized
    """
    global _state

    if not _state.initialized:
        return False

    # Check that core coordinators are ready
    core_coordinators = [
        "resource_coordinator",
        "metrics_orchestrator",
        "cache_orchestrator",
    ]

    for name in core_coordinators:
        if name not in _state.coordinators:
            return False
        if not _state.coordinators[name].initialized:
            return False

    return True


def reset_bootstrap_state() -> None:
    """Reset bootstrap state for testing purposes.

    WARNING: Only use in tests or development.
    """
    global _state
    _state = BootstrapState()
    logger.warning("[Bootstrap] Bootstrap state reset")


__all__ = [
    "bootstrap_coordination",
    "shutdown_coordination",
    "get_bootstrap_status",
    "is_coordination_ready",
    "reset_bootstrap_state",
    "CoordinatorStatus",
    "BootstrapState",
]
