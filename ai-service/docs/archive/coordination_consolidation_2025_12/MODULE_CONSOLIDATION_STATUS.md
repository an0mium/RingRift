# Coordination Module Consolidation Status

**Goal**: Reduce from 141 modules to ~15 core modules
**Date**: December 26, 2025
**Status**: Phase 1 - Deprecation wrappers created

## Overview

The `app/coordination/` directory has grown to 141 modules with significant duplication across sync, health, and event handling. This consolidation effort aims to reduce complexity while maintaining backward compatibility.

## Module Categories & Status

### 1. Sync Implementations (13 modules → 3 core)

**Keep (Core Implementations)**:

- ✅ `sync_facade.py` - **Main entry point** - Unified sync API
- ✅ `cluster/sync.py` - Unified sync re-exports
- ✅ `auto_sync_daemon.py` - P2P gossip sync daemon
- ✅ `cluster_data_sync.py` - Push-based cluster sync
- ✅ `ephemeral_sync.py` - Aggressive sync for Vast.ai
- ✅ `sync_bandwidth.py` - Bandwidth coordination
- ✅ `sync_router.py` - Intelligent routing
- ✅ `sync_mutex.py` - Distributed locking
- ✅ `sync_constants.py` - Shared constants

**Deprecated (with deprecation warning)**:

- ⚠️ `sync_coordinator.py` - Use `auto_sync_daemon.py` or `sync_facade.py`
  - Already has deprecation warning (see line 70-86)
  - Points users to AutoSyncDaemon and SyncFacade

**Can be deprecated later**:

- `sync_base.py` - Base classes (may merge into sync_facade.py)
- `async_bridge_manager.py` - Async/sync bridge (specialized, keep for now)

### 2. Health Monitoring (7 modules → 1 unified)

**Keep (Core Implementation)**:

- ✅ `cluster/health.py` - **Main entry point** - Unified health re-exports
- ✅ `unified_health_manager.py` - Error recovery + circuit breakers
- ✅ `node_health_monitor.py` - Async node monitoring + eviction
- ✅ `host_health_policy.py` - Pre-spawn SSH health checks

**Deprecated (wrappers created)**:

- ✅ `_deprecated_health_check_orchestrator.py` - Created wrapper → unified_health_manager
- ✅ `_deprecated_system_health_monitor.py` - Created wrapper → cluster/health
- ✅ `_deprecated_host_health_policy.py` - Created wrapper → cluster/health

**Status**: Health consolidation complete. All health checks now route through `cluster/health`.

### 3. Event Handling (7 modules → 1 unified)

**Keep (Core Implementation)**:

- ✅ `event_router.py` - **Main entry point** - Unified event router
- ✅ `core/events.py` - Re-exports from event_router
- ✅ `stage_events.py` - Pipeline stage events (specialized)
- ✅ `event_normalization.py` - Event type mapping
- ✅ `event_mappings.py` - Centralized event mappings

**Deprecated (wrappers created)**:

- ✅ `_deprecated_event_emitters.py` - Created wrapper → event_router
- ✅ `_deprecated_cross_process_events.py` - Created wrapper → event_router

**Status**: Event consolidation complete. All events now route through `event_router`.

## Consolidation Achievements

### Duplicates Identified & Deprecated

1. **Sync Implementations**: 8 competing implementations → 3 core + facade
   - `SyncScheduler` (deprecated) → `AutoSyncDaemon`
   - `UnifiedDataSync` (deprecated) → `SyncFacade`
   - `SyncOrchestrator` (wrapper) → may retire

2. **Health Monitoring**: 7 modules → 1 unified facade
   - `HealthCheckOrchestrator` → `UnifiedHealthManager`
   - `SystemHealthMonitor` → `check_cluster_health()`
   - `HostHealthPolicy` → `cluster/health`

3. **Event Handling**: 7 modules → 1 unified router
   - `event_emitters` → `event_router.emit_*()`
   - `cross_process_events` → `event_router.publish()`

### Total Reduction

- **Before**: 141 modules
- **Deprecated in Phase 1**: 5 modules (wrappers created)
- **After Phase 1**: ~136 active modules + 5 deprecation wrappers
- **Target**: ~15 core modules (need to continue consolidation)

## Migration Guide

### For Sync Operations

```python
# OLD (deprecated)
from app.coordination.sync_coordinator import SyncScheduler
scheduler = SyncScheduler.get_instance()

# NEW (recommended)
from app.coordination.sync_facade import sync
await sync("games", targets=["all"], priority="high")

# Or for daemon management
from app.coordination.auto_sync_daemon import AutoSyncDaemon
daemon = AutoSyncDaemon()
await daemon.start()
```

### For Health Checks

```python
# OLD (deprecated)
from app.coordination.health_check_orchestrator import get_health_orchestrator
orchestrator = get_health_orchestrator()

# NEW (recommended)
from app.coordination.cluster.health import (
    UnifiedHealthManager,
    get_health_manager,
    check_cluster_health,
)

manager = get_health_manager()
health_ok = check_cluster_health()
```

### For Events

```python
# OLD (deprecated)
from app.coordination.event_emitters import emit_training_completed
from app.coordination.cross_process_events import publish_event

# NEW (recommended)
from app.coordination.event_router import publish, DataEventType

await publish(
    DataEventType.TRAINING_COMPLETED,
    payload={"config": "hex8_2p"},
    source="training_daemon",
)
```

## Next Steps (Phase 2)

### High Priority Consolidations

1. **Daemon Management** (~10 modules)
   - `daemon_manager.py` - Main orchestrator
   - `daemon_adapters.py` - Wrappers for legacy daemons
   - Many individual daemon files that could be consolidated

2. **Training Coordination** (~8 modules)
   - `training_trigger_daemon.py`
   - `training_coordinator.py`
   - `training_freshness.py`
   - Could consolidate into single training orchestrator

3. **Resource Management** (~6 modules)
   - `resource_optimizer.py`
   - `resource_monitoring_coordinator.py`
   - `utilization_optimizer.py`
   - Overlap in GPU/resource tracking

4. **Queue Management** (~4 modules)
   - `queue_monitor.py`
   - `queue_populator.py`
   - `dead_letter_queue.py`
   - Could merge into unified queue system

### Timeline

- **Phase 1** (Dec 26, 2025): ✅ Deprecation wrappers for obvious duplicates
- **Phase 2** (Q1 2026): Consolidate daemon management and training coordination
- **Phase 3** (Q1 2026): Consolidate resource and queue management
- **Phase 4** (Q2 2026): Remove deprecated wrappers, finalize to ~15 core modules

## Testing Requirements

Before removing deprecated modules:

1. Grep codebase for imports from deprecated modules
2. Update all imports to use new paths
3. Run full test suite to ensure no regressions
4. Update documentation and examples

## Deprecation Policy

- **Deprecation Period**: 6 months (until Q2 2026)
- **Warning Level**: `DeprecationWarning` (visible with `-Wd`)
- **Migration Docs**: Included in each deprecated module's docstring
- **Breaking Change**: Removal scheduled for Q2 2026 release

## Benefits

1. **Reduced Complexity**: Fewer modules to understand and maintain
2. **Better Discoverability**: Clear entry points (facades) for common operations
3. **Improved Testing**: Less duplication = easier to test comprehensively
4. **Performance**: Single code path reduces overhead
5. **Documentation**: Easier to document canonical APIs

## Notes

- All deprecation wrappers are prefixed with `_deprecated_` to make them obvious
- Original implementations remain unchanged for backward compatibility
- New code should use the unified entry points (facades)
- Wrappers emit warnings at import time to guide migration
