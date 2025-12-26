# Deprecated Coordination Modules

This directory contains coordination modules that have been superseded by consolidated implementations or are no longer needed.

## sync_coordination_core.py

**Archived**: December 26, 2025

**Reason**: Zero external imports detected. Functionality superseded by active sync modules.

**Superseded By**:

The sync coordination functionality is now handled by these active modules:

- `app/coordination/sync_coordinator.py` - Actual sync scheduling and coordination (17+ imports)
- `app/coordination/auto_sync_daemon.py` - Automated P2P data sync
- `app/coordination/sync_router.py` - Intelligent sync routing decisions
- `app/coordination/sync_bandwidth.py` - Bandwidth-coordinated transfers
- `app/distributed/sync_coordinator.py` - Distributed sync coordination

**Original Purpose**:

Central coordinator for sync operations with the following responsibilities:

- Listen for SYNC_REQUEST events and execute sync operations
- Track sync state across the cluster
- Manage sync priorities and queuing
- Emit sync completion/failure events
- Integrate with SyncRouter and bandwidth management

**Migration**:

No migration needed - this module had zero external imports and was not being used.

If you need sync coordination functionality, use:

```python
# For sync scheduling and coordination
from app.coordination.sync_coordinator import get_sync_coordinator
coordinator = get_sync_coordinator()

# For automated P2P sync
from app.coordination.auto_sync_daemon import AutoSyncDaemon
daemon = AutoSyncDaemon()

# For sync routing decisions
from app.coordination.sync_router import get_sync_router
router = get_sync_router()
```

**Verification**:

Grep analysis confirmed zero imports:

```bash
grep -r "from app.coordination.sync_coordination_core import" --include="*.py" .
# Result: No matches found (only self-references in the file itself)
```

---

## unified_event_coordinator.py

**Archived**: December 2025

**Reason**: Functionality consolidated into `app/coordination/event_router.py`

**Migration**:

- All imports from `unified_event_coordinator` can be replaced with imports from `event_router`
- The following aliases are provided in `event_router.py` for backwards compatibility:
  - `UnifiedEventCoordinator` -> alias for `UnifiedEventRouter`
  - `get_event_coordinator()` -> alias for `get_router()`
  - `start_coordinator()` / `stop_coordinator()`
  - `get_coordinator_stats()` -> returns `CoordinatorStats`
  - All `emit_*` helper functions

**Original Purpose**:
The unified_event_coordinator bridged three event systems:

1. DataEventBus (data_events.py) - In-memory async events
2. StageEventBus (stage_events.py) - Pipeline stage events
3. CrossProcessEventQueue (cross_process_events.py) - SQLite-backed IPC

This functionality is now provided by `UnifiedEventRouter` in `event_router.py`, which:

- Provides the same bridging between event systems
- Has a cleaner API with unified `publish()` and `subscribe()` methods
- Includes event history and metrics
- Supports cross-process polling
