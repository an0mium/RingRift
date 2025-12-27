# Coordination Module Deprecation Guide

**Last Updated:** December 2025
**Removal Target:** Q2 2026

This guide documents all deprecated modules in `app/coordination/` and their replacements.

## Quick Reference

| Deprecated Module                             | Replacement                                         | Status              |
| --------------------------------------------- | --------------------------------------------------- | ------------------- |
| `auto_evaluation_daemon.py`                   | `evaluation_daemon.py` + `auto_promotion_daemon.py` | Emits warning       |
| `replication_monitor.py`                      | `unified_replication_daemon.py`                     | Emits warning       |
| `replication_repair_daemon.py`                | `unified_replication_daemon.py`                     | Emits warning       |
| `cross_process_events.py`                     | `event_router.py`                                   | Archived            |
| `sync_coordinator.py` (class SyncCoordinator) | `SyncScheduler` (same file)                         | Alias exists        |
| `bandwidth_manager.py`                        | `resources/bandwidth.py`                            | Emits warning       |
| `system_health_monitor.py` (scoring)          | `unified_health_manager.py`                         | Partial deprecation |

## Detailed Migration

### 1. Auto-Evaluation Daemon

**Old:**

```python
from app.coordination.auto_evaluation_daemon import AutoEvaluationDaemon
daemon = AutoEvaluationDaemon()
await daemon.start()
```

**New:**

```python
from app.coordination.evaluation_daemon import EvaluationDaemon
from app.coordination.auto_promotion_daemon import AutoPromotionDaemon

# Use separate daemons for better control
eval_daemon = EvaluationDaemon()
promo_daemon = AutoPromotionDaemon()
await eval_daemon.start()
await promo_daemon.start()
```

### 2. Replication Daemons

**Old:**

```python
from app.coordination.replication_monitor import ReplicationMonitor
from app.coordination.replication_repair_daemon import ReplicationRepairDaemon

monitor = ReplicationMonitor()
repair = ReplicationRepairDaemon()
```

**New:**

```python
from app.coordination.unified_replication_daemon import (
    UnifiedReplicationDaemon,
    create_replication_monitor,  # Backward-compat factory
    create_replication_repair_daemon,  # Backward-compat factory
)

# Single daemon handles both monitoring and repair
daemon = UnifiedReplicationDaemon()
await daemon.start()

# Or use factories for drop-in replacement
monitor = create_replication_monitor()
repair = create_replication_repair_daemon()
```

### 3. Cross-Process Events

**Old:**

```python
from app.coordination.cross_process_events import (
    CrossProcessEventQueue,
    publish,
    poll_events,
)
```

**New:**

```python
from app.coordination.event_router import (
    CrossProcessEventQueue,
    cp_publish as publish,
    cp_poll_events as poll_events,
    get_cross_process_queue,
)

# Or use the unified event system
from app.coordination.event_router import EventRouter, emit
router = EventRouter.get_instance()
await emit("EVENT_TYPE", {"data": "value"})
```

### 4. Sync Coordinator Alias

**Old:**

```python
from app.coordination.sync_coordinator import SyncCoordinator
coordinator = SyncCoordinator()
```

**New:**

```python
from app.coordination.sync_coordinator import SyncScheduler
scheduler = SyncScheduler()  # Same class, new canonical name

# Or use the helper functions
from app.coordination import get_sync_scheduler
scheduler = get_sync_scheduler()
```

### 5. Health Scoring

**Old:**

```python
from app.coordination.system_health_monitor import (
    calculate_system_health_score,
    get_system_health_level,
)
```

**New:**

```python
from app.coordination.unified_health_manager import (
    get_system_health_score,
    get_system_health_level,
    should_pause_pipeline,
    SystemHealthLevel,
    SystemHealthScore,
)
```

### 6. Bandwidth Manager

**Old:**

```python
from app.coordination.bandwidth_manager import BandwidthManager
```

**New:**

```python
from app.coordination.resources.bandwidth import BandwidthManager
# Or via package
from app.coordination.resources import BandwidthManager
```

## Suppressing Deprecation Warnings

During migration, you can suppress warnings:

```python
import warnings

# Suppress all coordination deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"app\.coordination\..*"
)

# Or suppress specific module
warnings.filterwarnings(
    "ignore",
    message=r".*auto_evaluation_daemon.*"
)
```

## Timeline

| Phase     | Date         | Action                      |
| --------- | ------------ | --------------------------- |
| Warning   | Dec 2025     | Deprecation warnings active |
| Migration | Jan-Mar 2026 | Update all internal imports |
| Removal   | Q2 2026      | Deprecated modules archived |

## Archived Modules

Already moved to `app/coordination/deprecated/`:

- `_deprecated_cross_process_events.py` → `event_router.py`
- `_deprecated_event_emitters.py` → `event_router.emit()`
- `_deprecated_health_check_orchestrator.py` → `cluster.health`
- `_deprecated_host_health_policy.py` → `cluster.health`
- `_deprecated_system_health_monitor.py` → `cluster.health`

## Package Structure

The coordination module is being reorganized into focused packages:

```
app/coordination/
├── core/                    # Event system, tasks, pipeline
├── cluster/                 # Health, sync, transport, P2P
├── training/                # Training orchestration, scheduling
├── resources/               # Bandwidth, thresholds, optimization
└── deprecated/              # Archived modules with shims
```

See `app/coordination/deprecated/README.md` for the full package migration guide.

## Checking Your Code

Find deprecated imports in your code:

```bash
# Find auto_evaluation_daemon usage
grep -r "from app.coordination.auto_evaluation_daemon" .

# Find replication_monitor usage
grep -r "from app.coordination.replication_monitor" .

# Find all deprecated imports
grep -rE "from app\.coordination\.(auto_evaluation_daemon|replication_monitor|replication_repair_daemon|cross_process_events)" .
```

## Questions?

- See `app/coordination/deprecated/README.md` for package structure
- See `app/coordination/COORDINATOR_GUIDE.md` for usage patterns
- File issues at https://github.com/anthropics/ringrift/issues
