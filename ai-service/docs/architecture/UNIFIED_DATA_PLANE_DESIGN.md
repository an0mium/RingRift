# Unified Data Plane Daemon Architecture

**Status**: Design Draft
**Created**: December 28, 2025
**Author**: RingRift Engineering

---

## Executive Summary

This document proposes consolidating 5+ separate data synchronization modules into a single **Unified Data Plane Daemon** that coordinates all data movement across the RingRift cluster.

### Problem Statement

The current codebase has fragmented data sync infrastructure:

| Module                         | LOC    | Purpose                      | Transport  |
| ------------------------------ | ------ | ---------------------------- | ---------- |
| `AutoSyncDaemon`               | ~1,200 | P2P gossip sync              | HTTP/rsync |
| `SyncFacade`                   | ~860   | Backend routing              | Delegates  |
| `S3NodeSyncDaemon`             | ~1,130 | S3 backup                    | AWS S3     |
| `dynamic_data_distribution.py` | ~724   | OWC distribution             | HTTP/rsync |
| `SyncRouter`                   | ~600   | Intelligent routing          | N/A        |
| **Total**                      | ~4,514 | Overlapping responsibilities | Mixed      |

**Key Problems**:

1. **No single source of truth** for "where is my data?"
2. **Event chains break** - ORPHAN_GAMES_DETECTED doesn't reliably reach export
3. **Transport selection fragmented** - each module picks its own transport
4. **No unified backpressure** - can overwhelm nodes during sync storms
5. **Duplicate health checks** - each daemon tracks its own health

### Proposed Solution

A single **UnifiedDataPlaneDaemon** that:

- Absorbs AutoSyncDaemon, S3NodeSyncDaemon, and dynamic_data_distribution logic
- Provides unified data catalog tracking what exists where
- Implements event-driven sync with proper chain completion
- Uses intelligent transport selection with fallback chains
- Manages cluster-wide bandwidth allocation

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Unified Data Plane Daemon                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ DataCatalog │  │ SyncPlanner │  │ TransportMgr│  │ EventBridge│ │
│  │  (registry) │  │  (routing)  │  │  (transfer) │  │  (events)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Transport Layer                            │   │
│  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────────┐  │   │
│  │  │ P2P   │  │ HTTP  │  │ rsync │  │  S3   │  │ base64/ssh│  │   │
│  │  │gossip │  │ fetch │  │ push  │  │backup │  │  fallback │  │   │
│  │  └───────┘  └───────┘  └───────┘  └───────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. DataCatalog - Central Registry

The DataCatalog maintains a real-time view of what data exists where:

```python
@dataclass
class DataEntry:
    """Entry in the data catalog."""
    path: str                    # Relative path (e.g., "games/canonical_hex8_2p.db")
    data_type: DataType          # GAMES, MODELS, NPZ, CHECKPOINT
    config_key: str              # e.g., "hex8_2p"
    size_bytes: int
    checksum: str                # SHA256
    mtime: float
    locations: set[str]          # Node IDs where this data exists
    primary_location: str        # Authoritative source (usually generator or coordinator)


class DataCatalog:
    """Central registry of all cluster data."""

    def __init__(self):
        self._entries: dict[str, DataEntry] = {}
        self._by_node: dict[str, set[str]] = {}  # node_id -> set of paths
        self._by_type: dict[DataType, set[str]] = {}
        self._by_config: dict[str, set[str]] = {}

    def register(self, entry: DataEntry) -> None:
        """Register or update a data entry."""

    def get_missing_on_node(self, node_id: str, data_type: DataType = None) -> list[DataEntry]:
        """Get data entries that should exist on node but don't."""

    def get_replication_factor(self, path: str) -> int:
        """Get number of nodes that have this data."""

    def mark_synced(self, path: str, node_id: str) -> None:
        """Mark data as synced to a node."""
```

**Benefits**:

- Single source of truth for data locations
- Efficient queries for sync planning
- Replication factor tracking for data safety

### 2. SyncPlanner - Intelligent Routing

The SyncPlanner decides WHAT to sync WHERE and WHEN:

```python
@dataclass
class SyncPlan:
    """A planned sync operation."""
    source_node: str
    target_nodes: list[str]
    entries: list[DataEntry]
    priority: SyncPriority        # CRITICAL, HIGH, NORMAL, LOW, BACKGROUND
    reason: str                   # Why this sync was triggered
    transport_preference: list[Transport]  # Preferred transports in order
    deadline: float | None        # Optional deadline (for training deps)


class SyncPlanner:
    """Plans sync operations based on cluster state."""

    def __init__(self, catalog: DataCatalog, bandwidth_manager: BandwidthManager):
        self._catalog = catalog
        self._bandwidth = bandwidth_manager
        self._pending_plans: PriorityQueue[SyncPlan] = PriorityQueue()

    def plan_for_event(self, event_type: str, payload: dict) -> list[SyncPlan]:
        """Generate sync plans in response to an event."""

    def plan_training_deps(self, node_id: str, config_key: str) -> SyncPlan:
        """Plan sync to satisfy training dependencies."""

    def plan_replication(self, min_factor: int = 3) -> list[SyncPlan]:
        """Plan syncs to meet replication requirements."""

    def plan_orphan_recovery(self, source_node: str, config_key: str) -> SyncPlan:
        """Plan urgent sync for orphan game recovery."""
```

**Routing Rules**:

| Scenario          | Source Selection    | Target Selection    | Transport        |
| ----------------- | ------------------- | ------------------- | ---------------- |
| Selfplay complete | Generator node      | Training nodes + S3 | P2P gossip       |
| Training starting | Closest with data   | Training node       | rsync            |
| Orphan recovery   | Ephemeral node      | Coordinator + S3    | rsync (priority) |
| Model promotion   | Trainer             | All GPU nodes       | P2P broadcast    |
| S3 backup         | Any with fresh data | S3 bucket           | AWS S3 API       |

### 3. TransportManager - Unified Transfer Layer

The TransportManager executes sync plans using the best available transport:

```python
class Transport(Enum):
    """Available transport mechanisms."""
    P2P_GOSSIP = "p2p"           # P2P HTTP gossip (fast for small files)
    HTTP_FETCH = "http"          # Direct HTTP download
    RSYNC = "rsync"              # rsync over SSH (reliable for large files)
    S3 = "s3"                    # AWS S3 API
    BASE64_SSH = "base64"        # Base64 over SSH (last resort)


@dataclass
class TransportResult:
    """Result of a transfer operation."""
    success: bool
    transport_used: Transport
    bytes_transferred: int
    duration_seconds: float
    error: str | None = None
    retries: int = 0


class TransportManager:
    """Manages data transfers across cluster."""

    # Transport selection order by scenario
    TRANSPORT_CHAINS = {
        "small_file": [Transport.P2P_GOSSIP, Transport.HTTP_FETCH, Transport.BASE64_SSH],
        "large_file": [Transport.RSYNC, Transport.HTTP_FETCH, Transport.BASE64_SSH],
        "s3_backup": [Transport.S3, Transport.RSYNC],
        "ephemeral_urgent": [Transport.RSYNC, Transport.BASE64_SSH],  # Skip P2P for speed
    }

    async def transfer(
        self,
        plan: SyncPlan,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TransportResult:
        """Execute a sync plan with automatic fallback."""

    async def _try_transport(
        self,
        transport: Transport,
        source: str,
        target: str,
        path: str,
    ) -> TransportResult:
        """Try a single transport, return result."""
```

**Transport Selection Logic**:

```python
def select_transport_chain(self, entry: DataEntry, target_node: str) -> list[Transport]:
    """Select transport chain based on file characteristics and target."""

    # Large files (>100MB) prefer rsync
    if entry.size_bytes > 100 * 1024 * 1024:
        return self.TRANSPORT_CHAINS["large_file"]

    # Ephemeral nodes need urgent sync (skip P2P overhead)
    if self._is_ephemeral_node(target_node):
        return self.TRANSPORT_CHAINS["ephemeral_urgent"]

    # S3 backup path
    if target_node == "s3":
        return self.TRANSPORT_CHAINS["s3_backup"]

    # Default: small file chain
    return self.TRANSPORT_CHAINS["small_file"]
```

### 4. EventBridge - Event-Driven Coordination

The EventBridge subscribes to all data-related events and triggers appropriate sync actions:

```python
class EventBridge:
    """Bridges events to sync operations."""

    # Event -> Action mappings
    EVENT_HANDLERS = {
        "SELFPLAY_COMPLETE": "_handle_selfplay_complete",
        "TRAINING_COMPLETED": "_handle_training_completed",
        "MODEL_PROMOTED": "_handle_model_promoted",
        "ORPHAN_GAMES_DETECTED": "_handle_orphan_detected",
        "DATA_NEEDED": "_handle_data_needed",  # NEW: explicit data request
        "NODE_TERMINATING": "_handle_node_terminating",
        "TRAINING_STARTED": "_handle_training_started",
    }

    async def _handle_selfplay_complete(self, event: dict) -> None:
        """Handle selfplay completion - sync games to training nodes."""
        config_key = event.get("config_key")
        games_count = event.get("games_count", 0)
        source_node = event.get("source_node")

        # Update catalog
        self._catalog.register(DataEntry(...))

        # Plan sync to training nodes
        plan = self._planner.plan_for_event("SELFPLAY_COMPLETE", event)

        # Execute with proper event emission on completion
        result = await self._transport.transfer(plan)

        if result.success:
            # Emit completion for pipeline coordination
            await self._emit_event("DATA_SYNC_COMPLETED", {
                "config_key": config_key,
                "games_synced": games_count,
                "source_node": source_node,
            })

            # Chain to NEW_GAMES_AVAILABLE for export trigger
            await self._emit_event("NEW_GAMES_AVAILABLE", {
                "config_key": config_key,
                "game_count": games_count,
            })
```

**Event Chain Completion**:

The key improvement is ensuring event chains complete properly:

```
SELFPLAY_COMPLETE
    │
    └──→ UnifiedDataPlaneDaemon._handle_selfplay_complete()
              │
              ├──→ Update DataCatalog
              ├──→ Plan & execute sync to training nodes
              ├──→ Plan & execute S3 backup
              │
              └──→ Emit DATA_SYNC_COMPLETED
                       │
                       └──→ Emit NEW_GAMES_AVAILABLE
                                 │
                                 └──→ DataPipelineOrchestrator triggers export
```

---

## Implementation Plan

### Phase 1: DataCatalog Foundation (Day 1-2)

1. Create `app/coordination/data_catalog.py`:
   - DataEntry, DataType dataclasses
   - DataCatalog class with SQLite persistence
   - Node manifest collection (reuse from S3NodeSyncDaemon)

2. Migrate manifest logic from:
   - `S3NodeSyncDaemon._build_local_manifest()`
   - `SyncPlanner.collect_manifest()` in p2p managers

### Phase 2: TransportManager Consolidation (Day 2-3)

1. Create `app/coordination/transport_manager.py`:
   - Consolidate transport implementations from:
     - `cluster_transport.py` (multi-transport)
     - `sync_bandwidth.py` (bandwidth limits)
     - `resilient_transfer.py` (retry logic)
   - Add transport chain fallback logic

2. Add health tracking per transport:
   - Circuit breakers per transport type
   - Success rate tracking

### Phase 3: SyncPlanner Intelligence (Day 3-4)

1. Create `app/coordination/sync_planner_v2.py`:
   - Absorb routing logic from `SyncRouter`
   - Add deadline-aware planning for training deps
   - Implement replication factor enforcement

2. Integrate with DataCatalog for efficient queries

### Phase 4: EventBridge & Daemon (Day 4-5)

1. Create `app/coordination/unified_data_plane_daemon.py`:
   - Main daemon class integrating all components
   - EventBridge with complete event handling
   - Health check aggregation

2. Event chain completion:
   - Ensure all event chains emit completion events
   - Add DATA_NEEDED event type for explicit requests

### Phase 5: Migration & Deprecation (Day 5-6)

1. Update callers to use new daemon:
   - `daemon_manager.py` factory functions
   - `master_loop.py` startup order
   - P2P orchestrator integration

2. Add deprecation warnings to old modules:
   - `AutoSyncDaemon` → `UnifiedDataPlaneDaemon`
   - `S3NodeSyncDaemon` → `UnifiedDataPlaneDaemon`
   - `dynamic_data_distribution.py` → `UnifiedDataPlaneDaemon`

---

## API Reference

### Main Entry Point

```python
from app.coordination.unified_data_plane_daemon import (
    UnifiedDataPlaneDaemon,
    get_data_plane,
    request_sync,
)

# Get daemon singleton
daemon = get_data_plane()

# Request sync (event-driven is preferred, but explicit requests supported)
await request_sync(
    data_type=DataType.GAMES,
    config_key="hex8_2p",
    target_nodes=["training-node-1"],
    priority=SyncPriority.HIGH,
)

# Query catalog
missing = daemon.catalog.get_missing_on_node("training-node-1", DataType.NPZ)
```

### Configuration

```python
@dataclass
class DataPlaneConfig:
    """Configuration for Unified Data Plane Daemon."""

    # Catalog settings
    catalog_db_path: Path = Path("data/coordination/data_catalog.db")
    manifest_refresh_interval: float = 60.0  # seconds

    # Sync settings
    min_replication_factor: int = 3
    max_concurrent_syncs: int = 5
    default_bandwidth_limit_mbps: int = 100

    # S3 settings (when S3 backup enabled)
    s3_enabled: bool = True
    s3_bucket: str = "ringrift-models-20251214"
    s3_sync_interval: float = 3600.0  # 1 hour

    # Transport settings
    large_file_threshold_mb: int = 100
    transport_timeout_seconds: float = 600.0
    max_retries: int = 3

    # Event settings
    emit_completion_events: bool = True
    chain_events: bool = True  # Emit downstream events on completion
```

### Health Check

```python
def health_check(self) -> HealthCheckResult:
    """Aggregate health from all components."""
    return HealthCheckResult(
        healthy=all_healthy,
        status=status,
        message=message,
        details={
            "catalog": self._catalog.health_check().details,
            "planner": self._planner.health_check().details,
            "transport": self._transport.health_check().details,
            "event_bridge": self._event_bridge.health_check().details,
            "pending_syncs": len(self._pending_syncs),
            "total_syncs": self._total_syncs,
            "total_bytes": self._total_bytes,
        },
    )
```

---

## Migration Guide

### For AutoSyncDaemon Users

```python
# Before
from app.coordination.auto_sync_daemon import get_auto_sync_daemon
daemon = get_auto_sync_daemon()
await daemon.sync_now()

# After
from app.coordination.unified_data_plane_daemon import get_data_plane
daemon = get_data_plane()
await daemon.sync_all()  # Or let event-driven sync handle it
```

### For S3NodeSyncDaemon Users

```python
# Before
from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon
daemon = S3NodeSyncDaemon()
await daemon.start()

# After
from app.coordination.unified_data_plane_daemon import get_data_plane
daemon = get_data_plane()
# S3 sync is automatic, or trigger explicitly:
await daemon.backup_to_s3()
```

### For dynamic_data_distribution Users

```python
# Before
python scripts/dynamic_data_distribution.py --daemon

# After
# Integrated into master_loop.py automatically
# Or run standalone:
python -m app.coordination.unified_data_plane_daemon
```

---

## Success Metrics

| Metric                           | Current    | Target     | Measurement                                       |
| -------------------------------- | ---------- | ---------- | ------------------------------------------------- |
| Event chain completion rate      | ~70%       | 99%+       | Track DATA_SYNC_COMPLETED after SELFPLAY_COMPLETE |
| Sync latency (selfplay→training) | 60-300s    | <30s       | Time from SELFPLAY_COMPLETE to data available     |
| Transport success rate           | ~85%       | 95%+       | First-attempt success rate                        |
| Code duplication                 | ~4,500 LOC | ~2,500 LOC | Combined LOC after consolidation                  |
| Number of sync modules           | 5+         | 1          | Unified daemon + transport adapters               |

---

## Risks and Mitigations

| Risk                                    | Impact | Mitigation                                                   |
| --------------------------------------- | ------ | ------------------------------------------------------------ |
| Breaking existing sync during migration | High   | Feature flag rollout, backward-compat factories              |
| S3 costs increase                       | Medium | Rate limiting, compression, deduplication                    |
| Single point of failure                 | High   | Health checks, graceful degradation, fallback to old daemons |
| Performance regression                  | Medium | Benchmark before/after, parallel transport execution         |

---

## Future Enhancements

1. **Delta sync**: Only transfer changed portions of large files
2. **Content-addressable storage**: Dedup identical data across configs
3. **Predictive pre-staging**: Anticipate training needs, pre-sync data
4. **Cross-region sync**: Support multi-region clusters
5. **Encryption at rest**: Encrypt S3 backups

---

## Appendix: Current Module Inventory

### Modules to Absorb

| Module                    | Path                                      | LOC   | Key Logic to Preserve                 |
| ------------------------- | ----------------------------------------- | ----- | ------------------------------------- |
| AutoSyncDaemon            | `app/coordination/auto_sync_daemon.py`    | 1,200 | P2P gossip, 5 strategies              |
| SyncFacade                | `app/coordination/sync_facade.py`         | 860   | Backend routing, priority sync        |
| S3NodeSyncDaemon          | `app/coordination/s3_node_sync_daemon.py` | 1,130 | S3 upload/download, manifest          |
| dynamic_data_distribution | `scripts/dynamic_data_distribution.py`    | 724   | OWC HTTP distribution, rsync fallback |
| SyncRouter                | `app/coordination/sync_router.py`         | 600   | Node selection, exclusion rules       |

### Modules to Keep (Lower Level)

| Module                | Path                | Reason                              |
| --------------------- | ------------------- | ----------------------------------- |
| sync_bandwidth.py     | `app/coordination/` | Bandwidth limiting primitives       |
| cluster_transport.py  | `app/coordination/` | Low-level transport implementations |
| resilient_transfer.py | `app/coordination/` | Retry logic, circuit breakers       |

### New Modules to Create

| Module                       | Path                | Purpose                   |
| ---------------------------- | ------------------- | ------------------------- |
| data_catalog.py              | `app/coordination/` | Central data registry     |
| sync_planner_v2.py           | `app/coordination/` | Intelligent sync planning |
| transport_manager.py         | `app/coordination/` | Unified transport layer   |
| unified_data_plane_daemon.py | `app/coordination/` | Main daemon               |

---

## Approval

- [ ] Architecture review
- [ ] Security review (S3 credentials handling)
- [ ] Performance review
- [ ] Implementation approved
