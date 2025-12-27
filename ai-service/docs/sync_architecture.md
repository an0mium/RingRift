# Sync Architecture

This document clarifies the sync-related modules and their responsibilities.

## Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      SyncFacade                             │
│   app/coordination/sync_facade.py                            │
│   - Recommended entry point for programmatic sync            │
│   - Routes to AutoSyncDaemon / SyncRouter / DistributedSync  │
└──────────────────────────┬──────────────────────────────────┘
                           │ routes
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│    AutoSyncDaemon     │       │ DistributedSyncCoord. │
│ app/coordination/     │       │ app/distributed/      │
│ auto_sync_daemon.py   │       │ sync_coordinator.py   │
│ ───────────────────── │       │ ────────────────────  │
│ CONTINUOUS sync       │       │ EXECUTION layer       │
│ - P2P + routing       │       │ - Performs syncs      │
│ - Strategy aware      │       │ - aria2/SSH/P2P       │
└───────────────────────┘       └───────────────────────┘

Legacy (avoid for new code):
  - SyncOrchestrator (app/distributed/sync_orchestrator.py)
  - SyncScheduler (app/coordination/sync_coordinator.py)
```

## Which module should I use?

| Use Case                          | Module                       | Import                                                             |
| --------------------------------- | ---------------------------- | ------------------------------------------------------------------ |
| **Most cases**: Programmatic sync | `SyncFacade`                 | `from app.coordination.sync_facade import sync`                    |
| Continuous/daemon sync            | `AutoSyncDaemon`             | `from app.coordination.auto_sync_daemon import AutoSyncDaemon`     |
| Low-level sync execution          | `DistributedSyncCoordinator` | `from app.coordination import DistributedSyncCoordinator`          |
| Legacy scheduling (deprecated)    | `SyncScheduler`              | `from app.coordination.sync_coordinator import get_sync_scheduler` |

## Detailed Responsibilities

### SyncFacade (Recommended Entry Point)

Unified programmatic entry point for sync operations:

```python
from app.coordination.sync_facade import sync

# Sync all data types
await sync("all")

# Sync specific data with routing hints
await sync("games", board_type="hex8", priority="high")
```

Features:

- Routes to AutoSyncDaemon / SyncRouter / DistributedSyncCoordinator
- Centralized logging + metrics for sync operations
- Backward-compatible with legacy backends (deprecated)

### AutoSyncDaemon (Recommended for Continuous Sync)

Daemonized sync loop for cluster-wide continuous sync:

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon

daemon = AutoSyncDaemon()
await daemon.start()
```

Features:

- Strategy-aware sync (HYBRID/EPHEMERAL/BROADCAST/AUTO)
- Prioritizes ephemeral nodes and avoids shared storage
- Integrates with sync routing + bandwidth controls

### SyncOrchestrator (Legacy, Pending Deprecation)

The older unified facade remains for backward compatibility:

```python
from app.distributed.sync_orchestrator import get_sync_orchestrator

orchestrator = get_sync_orchestrator()
await orchestrator.sync_all()
```

Prefer `SyncFacade` for new code.

### SyncScheduler (Deprecated Scheduling Layer)

Decides **when** and **what** to sync:

```python
from app.coordination.sync_coordinator import (
    get_sync_scheduler,
    get_cluster_data_status,
    schedule_priority_sync,
    get_sync_recommendations,
)

status = get_cluster_data_status()
recommendations = get_sync_recommendations()
await schedule_priority_sync()
```

Features (legacy):

- Data freshness tracking across all hosts
- Priority-based sync scheduling
- Bandwidth-aware transfer balancing
- Cluster-wide data state visibility

### DistributedSyncCoordinator (Execution Layer)

Performs **actual sync operations**:

```python
from app.coordination import DistributedSyncCoordinator

coordinator = DistributedSyncCoordinator.get_instance()
await coordinator.sync_training_data()
await coordinator.sync_models(model_ids=["model_v42"])
stats = await coordinator.full_cluster_sync()
```

Features:

- Multiple transport backends (aria2, SSH/rsync, P2P HTTP, Gossip)
- Automatic transport selection based on capabilities
- NFS optimization (skip sync when storage is shared)
- Circuit breaker integration
- Sync watchdog (deadline per sync, consecutive failure tracking)
- Data server health checks with best-effort restart and `get_sync_health()` metrics

## Exports from app.coordination

Recommended:

```python
from app.coordination.sync_facade import sync
```

Legacy exports (still available, but deprecated for new code):

```python
from app.coordination import (
    SyncScheduler,              # Scheduling layer (deprecated)
    DistributedSyncCoordinator, # Execution layer
)
```

## Migration Notes

The naming evolved over time:

- `SyncCoordinator` in `app/distributed/` is the original execution layer
- `SyncCoordinator` in `app/coordination/` was renamed to `SyncScheduler` in Dec 2025 (deprecated)
- `SyncOrchestrator` is pending deprecation; prefer `SyncFacade` + `AutoSyncDaemon`
