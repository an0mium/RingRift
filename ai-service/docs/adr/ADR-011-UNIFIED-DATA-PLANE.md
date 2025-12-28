# ADR-011: Unified Data Plane Daemon

**Status**: Proposed
**Date**: 2025-12-28
**Context**: RingRift AI Training Infrastructure

## Summary

This ADR proposes consolidating the fragmented data distribution infrastructure into a single **Unified Data Plane Daemon** that coordinates all data movement across the cluster.

## Problem Statement

The current data distribution infrastructure has multiple overlapping systems:

### Current Components (Fragmented)

| Component                      | Purpose                 | Refresh Cycle | Issues                 |
| ------------------------------ | ----------------------- | ------------- | ---------------------- |
| `AutoSyncDaemon`               | P2P data sync           | 60s           | Push-only, no pull     |
| `S3NodeSyncDaemon`             | S3 backup               | 1 hour        | Separate manifest      |
| `dynamic_data_distribution.py` | OWC → nodes             | 5 min         | HTTP-only, no manifest |
| `TrainingDataManifest`         | Training data discovery | Per-request   | Local+S3+OWC           |
| `ClusterManifest`              | Game/model locations    | Per-request   | SQLite-based           |
| `UnifiedInventory`             | Node discovery          | 60s           | Vast CLI polling       |

### Identified Gaps (10 Total)

1. **Three manifest systems** with separate refresh cycles → inconsistent views
2. **No unified priority queue** across data/training/eval
3. **Loose coupling** between BackpressureMonitor and SyncRouter
4. **Parallel tracking** (delivery ledger + events) without sync
5. **Data validation timing** happens during distribution, not collection
6. **NPZ validation at wrong stage** - should validate at export, not distribution
7. **Orphan games not tracked** when nodes become unreachable
8. **P2P/DaemonManager decoupling** - sync plans don't match job placement
9. **Data staleness gate** can be bypassed
10. **S3 consolidated manifest not queried** for training data

## Decision

Create a **Unified Data Plane Daemon** that:

1. **Single Manifest**: Merges all three manifest systems into one
2. **Event-Driven**: Responds to `DATA_NEEDED` events from training nodes
3. **Intent-Aware**: Replicates data based on training job placement
4. **Priority-Based**: Unified priority queue for all data operations
5. **Validated**: NPZ validation at export time, not distribution

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                 Unified Data Plane Daemon                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Manifest   │  │   Priority   │  │   Transport      │   │
│  │   Manager    │  │   Queue      │  │   Coordinator    │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Event Router Integration                 │   │
│  │  - DATA_NEEDED (subscribe)                           │   │
│  │  - DATA_AVAILABLE (emit)                             │   │
│  │  - TRAINING_STARTED (subscribe, triggers preload)    │   │
│  │  - SYNC_COMPLETED (emit)                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         │                 │                    │
         ▼                 ▼                    ▼
    ┌─────────┐      ┌──────────┐        ┌───────────┐
    │  Nodes  │      │  OWC     │        │   S3      │
    │ (P2P)   │      │ (mac-    │        │  (cold)   │
    │         │      │ studio)  │        │           │
    └─────────┘      └──────────┘        └───────────┘
```

### ManifestManager (Single Source of Truth)

```python
@dataclass
class DataItem:
    """Unified representation of any data item."""
    item_id: str           # game_id, model_id, npz_id
    item_type: DataType    # GAME, MODEL, NPZ, CHECKPOINT
    config_key: str        # "hex8_2p", "square8_4p"
    size_bytes: int
    sha256: str
    created_at: datetime
    quality_score: float | None

    # Location tracking
    locations: set[str]    # node_ids where this item exists
    primary_location: str  # preferred source for transfers

    # Metadata
    sample_count: int | None   # for NPZ
    game_count: int | None     # for DBs
    elo: float | None          # for models

class ManifestManager:
    """Unified manifest combining all three existing systems."""

    def __init__(self):
        self._items: dict[str, DataItem] = {}
        self._by_config: dict[str, set[str]] = {}
        self._by_location: dict[str, set[str]] = {}
        self._refresh_lock = asyncio.Lock()

    async def refresh(self) -> None:
        """Unified refresh from all sources."""
        async with self._refresh_lock:
            # 1. P2P node discovery (from UnifiedInventory)
            nodes = await self._discover_nodes()

            # 2. Local disk scan (from TrainingDataManifest)
            for node in nodes:
                items = await self._scan_node(node)
                self._merge_items(items)

            # 3. S3 manifest (from S3ConsolidationDaemon)
            s3_items = await self._scan_s3()
            self._merge_items(s3_items)

            # 4. OWC drive (from dynamic_data_distribution)
            owc_items = await self._scan_owc()
            self._merge_items(owc_items)

    def get_best_source(self, item_id: str, target_node: str) -> str:
        """Find best location to fetch item from for target node."""
        item = self._items.get(item_id)
        if not item:
            raise KeyError(f"Item {item_id} not found")

        # Priority: same rack > same provider > lowest latency > any
        return self._rank_locations(item.locations, target_node)[0]

    def get_data_for_config(
        self,
        config_key: str,
        data_type: DataType,
        min_quality: float = 0.0,
    ) -> list[DataItem]:
        """Get all data items for a config, sorted by quality."""
        items = [
            self._items[id]
            for id in self._by_config.get(config_key, set())
            if self._items[id].item_type == data_type
            and (self._items[id].quality_score or 0) >= min_quality
        ]
        return sorted(items, key=lambda x: x.quality_score or 0, reverse=True)
```

### PriorityQueue (Unified Data Operations)

```python
class DataPriority(Enum):
    CRITICAL = 0      # Training blocked, needs data NOW
    HIGH = 1          # Training imminent (<5 min)
    NORMAL = 2        # Regular sync
    LOW = 3           # Background replication
    COLD_STORAGE = 4  # S3 backup

@dataclass(order=True)
class DataOperation:
    """Unified data operation request."""
    priority: DataPriority
    created_at: datetime  # For FIFO within priority

    # Not used for ordering
    operation_id: str = field(compare=False)
    operation_type: Literal["sync", "replicate", "backup", "preload"] = field(compare=False)
    source_node: str = field(compare=False)
    target_node: str = field(compare=False)
    item_ids: list[str] = field(compare=False)
    config_key: str = field(compare=False)
    requester: str = field(compare=False)  # Event that triggered this

class DataPriorityQueue:
    """Unified priority queue for all data operations."""

    def __init__(self, max_concurrent: int = 10):
        self._queue: asyncio.PriorityQueue[DataOperation] = asyncio.PriorityQueue()
        self._active: dict[str, DataOperation] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def submit(self, op: DataOperation) -> str:
        """Submit operation, returns operation_id."""
        await self._queue.put(op)
        return op.operation_id

    async def process_forever(self, executor: TransportCoordinator) -> None:
        """Main processing loop."""
        while True:
            async with self._semaphore:
                op = await self._queue.get()
                self._active[op.operation_id] = op
                try:
                    await executor.execute(op)
                finally:
                    del self._active[op.operation_id]
```

### Event Integration

```python
class UnifiedDataPlaneDaemon(HandlerBase):
    """Unified daemon for all data movement."""

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        return {
            # Training triggers
            "TRAINING_STARTED": self._on_training_started,
            "TRAINING_THRESHOLD_REACHED": self._on_training_threshold,

            # Data needs
            "DATA_NEEDED": self._on_data_needed,
            "DATA_STALE": self._on_data_stale,

            # Sync coordination
            "SELFPLAY_COMPLETE": self._on_selfplay_complete,
            "NPZ_EXPORT_COMPLETE": self._on_npz_export,

            # Model distribution
            "MODEL_PROMOTED": self._on_model_promoted,

            # Cluster events
            "HOST_ONLINE": self._on_host_online,
            "HOST_OFFLINE": self._on_host_offline,

            # Backpressure
            "BACKPRESSURE_ACTIVATED": self._on_backpressure_activated,
            "BACKPRESSURE_RELEASED": self._on_backpressure_released,
        }

    async def _on_training_started(self, event: dict) -> None:
        """Pre-fetch data for training job."""
        config_key = event["config_key"]
        target_node = event["node_id"]

        # Find best NPZ for this config
        npz_items = self.manifest.get_data_for_config(
            config_key, DataType.NPZ, min_quality=0.5
        )

        if not npz_items:
            self._emit("TRAINING_BLOCKED_BY_DATA", {
                "config_key": config_key,
                "reason": "no_npz_available",
            })
            return

        # Queue preload operation
        await self.queue.submit(DataOperation(
            priority=DataPriority.CRITICAL,
            created_at=datetime.now(),
            operation_id=f"preload-{config_key}-{target_node}",
            operation_type="preload",
            source_node=npz_items[0].primary_location,
            target_node=target_node,
            item_ids=[npz_items[0].item_id],
            config_key=config_key,
            requester="TRAINING_STARTED",
        ))

    async def _on_data_needed(self, event: dict) -> None:
        """Handle explicit data request from a node."""
        config_key = event["config_key"]
        target_node = event["node_id"]
        data_type = DataType[event.get("data_type", "NPZ")]
        min_samples = event.get("min_samples", 500)

        # Find data meeting requirements
        items = self.manifest.get_data_for_config(config_key, data_type)

        # Filter by sample count if NPZ
        if data_type == DataType.NPZ:
            items = [i for i in items if (i.sample_count or 0) >= min_samples]

        if not items:
            # No data available - emit need for selfplay
            self._emit("SELFPLAY_NEEDED", {
                "config_key": config_key,
                "reason": "training_data_insufficient",
                "required_samples": min_samples,
            })
            return

        # Queue sync operation
        for item in items[:3]:  # Top 3 by quality
            await self.queue.submit(DataOperation(
                priority=DataPriority.HIGH,
                created_at=datetime.now(),
                operation_id=f"sync-{item.item_id}-{target_node}",
                operation_type="sync",
                source_node=self.manifest.get_best_source(item.item_id, target_node),
                target_node=target_node,
                item_ids=[item.item_id],
                config_key=config_key,
                requester="DATA_NEEDED",
            ))
```

### Transport Coordinator (Multi-Method)

```python
class TransportCoordinator:
    """Coordinates multiple transport methods with failover."""

    TRANSPORT_PRIORITY = [
        "p2p_http",      # Fastest for small files
        "bittorrent",    # Best for large files to multiple nodes
        "rsync",         # Reliable for large single transfers
        "s3",            # Fallback via S3
    ]

    async def execute(self, op: DataOperation) -> TransportResult:
        """Execute operation with automatic failover."""
        errors = []

        for transport in self.TRANSPORT_PRIORITY:
            if not self._transport_available(transport, op):
                continue

            try:
                result = await self._execute_with_transport(transport, op)
                if result.success:
                    self._emit("DATA_TRANSFER_COMPLETE", {
                        "operation_id": op.operation_id,
                        "transport": transport,
                        "duration_ms": result.duration_ms,
                    })
                    return result
                errors.append((transport, result.error))
            except Exception as e:
                errors.append((transport, str(e)))

        # All transports failed
        self._emit("DATA_TRANSFER_FAILED", {
            "operation_id": op.operation_id,
            "errors": errors,
        })
        raise DataTransferError(f"All transports failed: {errors}")
```

## Migration Plan

### Phase 1: ManifestManager (Week 1)

1. Create `app/coordination/manifest_manager.py`
2. Integrate existing manifest sources
3. Add manifest refresh loop (30s default)
4. Add tests for manifest consistency

### Phase 2: PriorityQueue (Week 1-2)

1. Create `app/coordination/data_priority_queue.py`
2. Migrate `UnifiedQueuePopulator` priority logic
3. Add operation tracking and metrics
4. Add tests for priority ordering

### Phase 3: Event Integration (Week 2)

1. Create `app/coordination/unified_data_plane.py`
2. Wire event subscriptions
3. Migrate `DATA_NEEDED` emitters
4. Add `TRAINING_STARTED` preload logic

### Phase 4: Transport Coordinator (Week 2-3)

1. Create `app/coordination/transport_coordinator.py`
2. Consolidate existing transport code
3. Add failover logic
4. Add bandwidth coordination

### Phase 5: Deprecation (Week 3-4)

1. Mark deprecated: `TrainingDataManifest`, `ClusterManifest`, `UnifiedInventory`
2. Update callers to use `ManifestManager`
3. Add deprecation warnings
4. Archive deprecated modules

## Metrics

| Metric                      | Before                    | After   |
| --------------------------- | ------------------------- | ------- |
| Manifest refresh cycles     | 3 (60s, per-request, 60s) | 1 (30s) |
| Data transfer failures      | ~5%/day                   | <1%/day |
| Time to first training data | 5-10 min                  | <2 min  |
| Orphan data detection       | None                      | <5 min  |
| S3 cold storage utilization | 10%                       | 80%     |

## Decision Drivers

1. **Consistency**: Single manifest ensures all components see the same data view
2. **Priority**: Training-critical data gets precedence over background sync
3. **Reliability**: Multi-transport failover reduces transfer failures
4. **Efficiency**: Intent-aware replication reduces unnecessary transfers
5. **Observability**: Unified metrics for all data operations

## Alternatives Considered

### Alternative 1: Keep Separate Systems

- Pros: No migration work
- Cons: Inconsistency continues, priority conflicts remain
- Decision: Rejected - gaps causing production issues

### Alternative 2: Centralized S3-Only

- Pros: Simpler architecture
- Cons: High latency, egress costs, single point of failure
- Decision: Rejected - P2P is faster and more reliable

### Alternative 3: Pure Event-Driven (No Manifest)

- Pros: Truly decoupled
- Cons: No global view, difficult to answer "where is data X?"
- Decision: Rejected - need centralized view for debugging and recovery

## References

- `app/coordination/auto_sync_daemon.py` - Current P2P sync
- `app/coordination/s3_node_sync_daemon.py` - Current S3 sync
- `scripts/dynamic_data_distribution.py` - Current OWC distribution
- `app/coordination/training_data_manifest.py` - Current training manifest
- `app/coordination/cluster_manifest.py` - Current cluster manifest
- `app/coordination/unified_inventory.py` - Current node inventory
