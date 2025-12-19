# RingRift AI-Service Consolidation Status

**Date:** December 19, 2025
**Status:** Phase 1 Complete (consolidation + cleanup)

---

## Completed Consolidation Work

### 1. Monitoring and Cluster Script Cleanup

- Removed legacy cluster monitor/control/sync scripts in favor of:
  - `scripts/p2p_orchestrator.py` (primary cluster orchestration)
  - `python -m scripts.monitor status|health|alert` (unified monitoring)
  - `scripts/unified_cluster_monitor.py` (deep health checks)
- Removal list includes:
  - `scripts/cluster_*` (monitor/control/manager/worker/sync shell + python)
  - `scripts/monitor_10h_enhanced.sh`

### 2. Data Validator Consolidation

- Added deprecation notice to `app/training/data_validation.py`
- Added backward-compatible re-exports in `app/training/unified_data_validator.py`
- Users can now import from either module during migration

### 3. Event Router Integration

- Added router imports to `app/coordination/event_emitters.py`
- Added `_emit_via_router()` and `_emit_via_router_sync()` helpers
- Updated `_emit_stage_event()` to optionally use unified router
- Set `USE_UNIFIED_ROUTER = True` for automatic routing

### 4. Daemon Manager Integration

- Added `DaemonManager` import to `app/main.py`
- Integrated daemon startup/shutdown in FastAPI lifespan
- Enabled via `RINGRIFT_START_DAEMONS=1` environment variable

### 5. Configuration Migration

- Added threshold constant re-exports to `app/config/__init__.py`:
  - `TRAINING_TRIGGER_GAMES`, `TRAINING_MIN_INTERVAL_SECONDS`
  - `ELO_DROP_ROLLBACK`, `WIN_RATE_DROP_ROLLBACK`
  - `ELO_IMPROVEMENT_PROMOTE`, `MIN_GAMES_PROMOTE`
  - `INITIAL_ELO_RATING`, `ELO_K_FACTOR`, `MIN_GAMES_FOR_ELO`
- Added queue/scaling/duration defaults to `app/config/coordination_defaults.py`

### 6. Training Pipeline Rationalization

- Added cross-reference documentation to:
  - `app/training/unified_orchestrator.py` (execution-level)
  - `app/training/orchestrated_training.py` (service-level)

### 7. Event Wiring Helpers

- Added event wiring hooks for coordinators:
  - `queue_populator`, `sync_scheduler`, `task_coordinator`
  - `training_coordinator`, `transfer_verification`, `ephemeral_data_guard`
  - `multi_provider_orchestrator`

### 8. Coordination Exports

- Standardized `__all__` exports across coordination modules for canonical imports.

---

## Open Gaps / Risks

- Unified module tests referenced in prior notes are not present in repo.
- `tests/test_model_registry.py` and `tests/test_promotion_controller.py` were removed;
  replacement coverage is still needed.

---

## Recommended Next Steps (Priority Order)

### Priority 1: Critical Consolidation

#### 1.1 Sync Coordinator Naming Clarification

**Files:**

- `app/coordination/sync_coordinator.py` (SCHEDULING)
- `app/distributed/sync_coordinator.py` (EXECUTION)

**Action:** Finalize `SyncScheduler` as canonical name for scheduling layer.

#### 1.2 Model Registry Consolidation

**Files to consolidate:**

- `app/training/model_registry.py` -> deprecate
- `app/training/unified_model_store.py` -> canonical
- `app/training/training_registry.py` -> rename to `TrainingJobRegistry`

**Action:** Make `unified_model_store.UnifiedModelStore` the canonical model lifecycle tracker.

#### 1.3 Complete Data Validation Unification

**Action:** Integrate remaining validators into `unified_data_validator.py`:

- `territory_dataset_validation.py`
- `db/parity_validator.py`

### Priority 2: High-Value Consolidation

#### 2.1 Centralize Hardcoded Constants

Create new config files:

- `app/config/timeout_defaults.py`
- `app/config/threshold_defaults.py`

Replace hardcoded values across:

- `orchestrator_registry.py` (HEARTBEAT_TIMEOUT_SECONDS)
- `distributed_lock.py` (DEFAULT_LOCK_TIMEOUT)
- `cluster_transport.py` (DEFAULT_CONNECT_TIMEOUT)
- `sync_mutex.py` (LOCK_TIMEOUT_SECONDS)

#### 2.2 Standardize Package Imports

Add missing exports to `app/training/__init__.py`:

- `TrainingEnvConfig`, `make_env` from `env.py`
- `seed_all` from `seed_utils.py`
- `infer_victory_reason` from `tournament.py`

#### 2.3 Orchestrator Registry Cleanup

- Verify all orchestrators extend `CoordinatorBase`
- Standardize `.get_status()` method signature
- Add `get_all_orchestrators()` helper function

### Priority 3: Medium-Value Cleanup

#### 3.1 Clean Deprecation Warnings

Update lazy loading to explicit deprecation warnings:

- `app/distributed/__init__.py` (cluster_coordinator symbols)
- `app/distributed/ingestion_wal.py`

#### 3.2 AI Module Organization

Expand `app/ai/__init__.py` to centralize AI class exports.

### Priority 4: Integration Verification

#### 4.1 Verify Unified Module Usage

Check that new unified modules are used instead of old:

- `unified_orchestrator.py` vs `train_loop.py`
- `checkpoint_unified.py` vs `checkpointing.py`
- `distributed_unified.py` vs `distributed.py`

#### 4.2 Event System Unification

Verify all event emission uses unified router:

- Check `app/distributed/event_helpers.py`
- Update old emitters to use `coordination.event_router`

---

## File Changes Summary

### Added Files

| File                                  | Purpose                       |
| ------------------------------------- | ----------------------------- |
| `app/core/thread_spawner.py`          | Thread supervision helper     |
| `app/ai/ebmo_ai.py`                   | EBMO inference AI             |
| `app/ai/ebmo_network.py`              | EBMO network architecture     |
| `app/training/ebmo_dataset.py`        | EBMO dataset utilities        |
| `app/training/ebmo_trainer.py`        | EBMO training loop            |
| `app/training/model_state_machine.py` | Model lifecycle state machine |
| `app/training/train_gmo.py`           | GMO training script           |
| `tests/test_gmo_ai.py`                | GMO tests                     |
| `docs/GMO_ALGORITHM.md`               | GMO algorithm reference       |

### Removed Files

| File                                 | Notes                                                        |
| ------------------------------------ | ------------------------------------------------------------ |
| `scripts/cluster_*`                  | Legacy cluster scripts removed (see `scripts/DEPRECATED.md`) |
| `scripts/monitor_10h_enhanced.sh`    | Legacy monitoring script removed                             |
| `tests/test_model_registry.py`       | Removed pending consolidation                                |
| `tests/test_promotion_controller.py` | Removed pending consolidation                                |

---

## Import Migration Guide

### Old -> New Import Patterns

```python
# Monitoring (DEPRECATED)
# OLD: from scripts.cluster_monitor import ClusterMonitor
# NEW:
from scripts.unified_cluster_monitor import UnifiedClusterMonitor

# Unified monitor CLI
# python -m scripts.monitor status|health|alert

# Data Validation (DEPRECATED)
# OLD: from app.training.data_validation import DataValidator
# NEW:
from app.training.unified_data_validator import (
    UnifiedDataValidator,
    get_validator,
    # Legacy re-exports also available:
    DataValidator,
    validate_npz_file,
)

# Model Store
from app.training import (
    UnifiedModelStore,
    get_model_store,
    ModelInfo,
)

# Model Lifecycle
from app.coordination import (
    ModelLifecycleCoordinator,
    get_model_coordinator,
    wire_model_events,
)

# Configuration Thresholds
from app.config import (
    TRAINING_TRIGGER_GAMES,
    ELO_DROP_ROLLBACK,
    INITIAL_ELO_RATING,
)

# Event Routing (Unified)
from app.coordination import (
    get_event_router,
    router_publish_event,
    publish_event_sync,
    subscribe_event,
)

# Daemon Management
from app.coordination.daemon_manager import (
    DaemonManager,
    get_daemon_manager,
    DaemonType,
)
```
