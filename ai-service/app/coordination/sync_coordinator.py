"""Deprecated sync coordinator shim.

.. deprecated:: February 2026
    This module has been archived to ``deprecated/_deprecated_sync_coordinator.py``.
    For new code, use:
    - ``AutoSyncDaemon`` from ``auto_sync_daemon.py`` for automated sync
    - ``SyncFacade`` from ``sync_facade.py`` for manual sync
    - ``SyncCoordinator`` from ``app.distributed.sync_coordinator`` for execution

    This shim re-exports all symbols for backward compatibility and will be
    removed in Q3 2026.
"""

import warnings

warnings.warn(
    "app.coordination.sync_coordinator is deprecated and archived. "
    "Use AutoSyncDaemon or SyncFacade instead. See DEPRECATION_ROADMAP.md.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the archived module
from app.coordination.deprecated._deprecated_sync_coordinator import (  # noqa: E402, F401
    CRITICAL_STALE_THRESHOLD_SECONDS,
    STALE_DATA_THRESHOLD_SECONDS,
    ClusterDataStatus,
    HostDataState,
    HostType,
    SyncAction,
    SyncCoordinator,
    SyncPriority,
    SyncRecommendation,
    SyncScheduler,
    execute_priority_sync,
    get_cluster_data_status,
    get_next_sync_target,
    get_sync_coordinator,
    get_sync_recommendations,
    get_sync_scheduler,
    record_games_generated,
    record_sync_complete,
    record_sync_start,
    register_host,
    reset_sync_coordinator,
    reset_sync_scheduler,
    update_host_state,
    wire_sync_events,
)

__all__ = [
    "CRITICAL_STALE_THRESHOLD_SECONDS",
    "STALE_DATA_THRESHOLD_SECONDS",
    "ClusterDataStatus",
    "HostDataState",
    "HostType",
    "SyncAction",
    "SyncCoordinator",
    "SyncPriority",
    "SyncRecommendation",
    "SyncScheduler",
    "execute_priority_sync",
    "get_cluster_data_status",
    "get_next_sync_target",
    "get_sync_coordinator",
    "get_sync_recommendations",
    "get_sync_scheduler",
    "record_games_generated",
    "record_sync_complete",
    "record_sync_start",
    "register_host",
    "reset_sync_coordinator",
    "reset_sync_scheduler",
    "update_host_state",
    "wire_sync_events",
]
