# ADR-006: ClusterManifest Decomposition Plan

**Status**: Proposed
**Date**: December 2025
**Author**: RingRift Team

## Context

`app/distributed/cluster_manifest.py` has grown to 2,564 lines with 49 methods,
making it difficult to maintain and test. The class manages:

1. Game location tracking (where games exist in the cluster)
2. Model location tracking (where models are stored)
3. NPZ file tracking (training data locations)
4. Checkpoint tracking (training checkpoint locations)
5. Database tracking (database file locations)
6. Node capacity management (disk space tracking)
7. Sync target selection (where to replicate data)
8. Disk cleanup policies (what to delete when full)

## Decision

Decompose `ClusterManifest` into focused registry classes, each with a single
responsibility. The main `ClusterManifest` becomes a facade that delegates to
specialized registries.

### Proposed Structure

```
app/distributed/
├── cluster_manifest.py          # Facade (reduced to ~500 lines)
├── registries/
│   ├── __init__.py
│   ├── base.py                  # BaseRegistry with common DB operations
│   ├── game_registry.py         # GameLocationRegistry (~400 lines)
│   ├── model_registry.py        # ModelLocationRegistry (~300 lines)
│   ├── npz_registry.py          # NPZLocationRegistry (~300 lines)
│   ├── checkpoint_registry.py   # CheckpointLocationRegistry (~400 lines)
│   └── database_registry.py     # DatabaseLocationRegistry (~300 lines)
├── capacity/
│   ├── __init__.py
│   ├── node_inventory.py        # NodeInventoryManager (~400 lines)
│   └── disk_cleanup.py          # DiskCleanupManager (~400 lines)
└── sync/
    ├── __init__.py
    └── replication.py           # ReplicationTargetSelector (~300 lines)
```

### Method Distribution

| Current Methods (49 total)                                                   | Target Module          |
| ---------------------------------------------------------------------------- | ---------------------- |
| register_game, find_game, get_game_replication_count, etc. (10)              | game_registry.py       |
| register_model, find_model, find_models_for_config (4)                       | model_registry.py      |
| register_npz, find_npz_for_config (3)                                        | npz_registry.py        |
| register_checkpoint, find_checkpoint, mark_checkpoint_as_best (5)            | checkpoint_registry.py |
| register_database, update_database_game_count, find_databases_for_config (5) | database_registry.py   |
| update_node_capacity, get_node_inventory, get_all_db_paths (8)               | node_inventory.py      |
| get_replication_targets, can_receive_data, get_sync_policy (4)               | replication.py         |
| check_disk_cleanup_needed, get_cleanup_candidates, run_disk_cleanup (7)      | disk_cleanup.py        |
| \_init_db, \_connection, close (3)                                           | base.py                |

## Consequences

### Positive

- **Testability**: Each registry can be unit tested in isolation
- **Maintainability**: Smaller files (~300-400 lines each)
- **Single Responsibility**: Each module has one clear purpose
- **Parallel Development**: Teams can work on different registries

### Negative

- **Migration Risk**: ClusterManifest is used throughout the codebase
- **Backwards Compatibility**: Must maintain existing API during transition
- **Coordination**: SQLite connection sharing needs careful design

## Implementation Plan

### Phase 1: Extract BaseRegistry (Low Risk)

1. Create `registries/base.py` with connection management
2. Move `_init_db`, `_connection`, `close` methods
3. Test database operations still work

### Phase 2: Extract GameLocationRegistry (Medium Risk)

1. Create `registries/game_registry.py`
2. Move game-related methods
3. Update ClusterManifest to delegate to registry
4. Maintain backwards-compatible API

### Phase 3: Continue with remaining registries

- Model → NPZ → Checkpoint → Database order
- Each extraction follows Phase 2 pattern

### Phase 4: Extract Capacity Management

1. Create `capacity/node_inventory.py`
2. Create `capacity/disk_cleanup.py`
3. Update delegation in ClusterManifest

### Phase 5: Extract Sync Logic

1. Create `sync/replication.py`
2. Final cleanup and removal of deprecated code

## Backwards Compatibility

During transition, `ClusterManifest` will maintain its existing API:

```python
# This continues to work:
manifest = get_cluster_manifest()
manifest.register_game("game-123", "node-a", "/data/games.db")
locations = manifest.find_game("game-123")

# Internally delegates to:
manifest._game_registry.register_game(...)
manifest._game_registry.find_game(...)
```

## Related ADRs

- ADR-001: Event-Driven Architecture (events emitted by registries)
- ADR-002: Daemon Lifecycle Management (uses ClusterManifest)

## Files Affected

- `app/distributed/cluster_manifest.py` (refactor)
- `app/coordination/auto_sync_daemon.py` (uses ClusterManifest)
- `app/coordination/sync_router.py` (uses ClusterManifest)
- `app/distributed/data_catalog.py` (uses ClusterManifest)
- Tests in `tests/unit/distributed/`
