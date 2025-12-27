"""Deprecated coordination modules - ARCHIVED.

All deprecated modules have been moved to:
  archive/deprecated_coordination/

See archive/deprecated_coordination/README.md for:
- Migration guides
- Replacement module documentation
- Verification commands

Archived December 27, 2025 (3,339 LOC total):
- _deprecated_auto_evaluation_daemon.py -> evaluation_daemon + auto_promotion_daemon
- _deprecated_cross_process_events.py -> event_router
- _deprecated_event_emitters.py -> event_router.emit()
- _deprecated_health_check_orchestrator.py -> unified_health_manager
- _deprecated_host_health_policy.py -> unified_health_manager
- _deprecated_queue_populator_daemon.py -> queue_populator
- _deprecated_sync_coordinator.py -> AutoSyncDaemon + sync_router
- _deprecated_system_health_monitor.py -> unified_health_manager
"""

import warnings


def __getattr__(name: str):
    """Raise clear error for archived modules."""
    archived_modules = {
        "cross_process_events": "event_router",
        "event_emitters": "event_router",
        "health_check_orchestrator": "unified_health_manager",
        "host_health_policy": "unified_health_manager",
        "system_health_monitor": "unified_health_manager",
        "auto_evaluation_daemon": "daemon_manager with EVALUATION_DAEMON",
        "sync_coordinator": "auto_sync_daemon + sync_router",
        "queue_populator_daemon": "queue_populator",
    }

    if name in archived_modules:
        replacement = archived_modules[name]
        raise ImportError(
            f"app.coordination.deprecated.{name} has been archived. "
            f"Use app.coordination.{replacement} instead. "
            f"See archive/deprecated_coordination/README.md for migration."
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
