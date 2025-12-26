# Sync Architecture Documentation

This document clarifies the responsibilities and relationships between the 13 sync-related modules in the RingRift AI service.

## Overview

The sync system ensures data (games, models, NPZ files) flows reliably across the distributed cluster. It uses a layered architecture with clear separation of concerns.

## Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAEMON LAYER (What runs)                     │
├─────────────────────────────────────────────────────────────────┤
│  auto_sync_daemon.py    │ Primary: P2P gossip-based sync       │
│  ephemeral_sync.py      │ Aggressive sync for Vast.ai nodes    │
│  cluster_data_sync.py   │ Cluster-wide data distribution       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  COORDINATION LAYER (How to sync)               │
├─────────────────────────────────────────────────────────────────┤
│  sync_coordinator.py    │ DEPRECATED - use auto_sync_daemon    │
│  sync_router.py         │ Intelligent routing by node caps     │
│  sync_bandwidth.py      │ Adaptive bandwidth per host          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BASE LAYER (Primitives)                      │
├─────────────────────────────────────────────────────────────────┤
│  sync_base.py           │ SyncEntry, SyncStatus dataclasses    │
│  sync_constants.py      │ Timeouts, thresholds, categories     │
│  sync_mutex.py          │ Distributed locking for sync ops     │
│  sync_coordination_core.py │ Core sync logic (push/pull)       │
└─────────────────────────────────────────────────────────────────┘
```

## Module Details

### Daemon Layer (Start These)

#### `auto_sync_daemon.py` - PRIMARY SYNC DAEMON

**Purpose**: Automated P2P data synchronization using push-from-generator + gossip replication.

**Key Features**:

- Push games immediately after generation
- Gossip protocol for cluster-wide consistency
- Respects `excluded_hosts` from config
- Bandwidth-aware transfers

**When to use**: Always running on coordinator and training nodes.

**Events emitted**: `SYNC_STARTED`, `SYNC_COMPLETED`, `SYNC_FAILED`

---

#### `ephemeral_sync.py` - AGGRESSIVE SYNC FOR EPHEMERAL NODES

**Purpose**: Ultra-aggressive sync for termination-prone Vast.ai nodes.

**Key Features**:

- 5-second poll interval (vs 60s for persistent nodes)
- Write-through mode: waits for push confirmation
- Termination signal handlers (SIGTERM/SIGINT)
- Emits `HOST_OFFLINE` with pending games count

**When to use**: On Vast.ai and other spot instances.

**Config**: `ephemeral_sync.poll_interval_seconds = 5`

---

#### `cluster_data_sync.py` - CLUSTER-WIDE DISTRIBUTION

**Purpose**: Distribute data to all nodes in the cluster.

**Key Features**:

- TRAINING_NODE_WATCHER daemon for priority sync
- Detects active training and syncs fresh data
- Coordinates with ClusterManifest for data locations

**When to use**: For bulk distribution after major exports.

---

### Coordination Layer (Called by Daemons)

#### `sync_router.py` - INTELLIGENT ROUTING

**Purpose**: Routes sync operations based on node capabilities and exclusion rules.

**Key Features**:

- Checks node health before routing
- Respects `sync_routing.excluded_hosts` config
- Routes to nodes with sufficient disk space
- Handles `allowed_external_storage` for mac-studio

**API**:

```python
from app.coordination.sync_router import SyncRouter
router = SyncRouter()
targets = router.get_sync_targets(data_type="games", size_mb=100)
```

---

#### `sync_bandwidth.py` - BANDWIDTH MANAGEMENT

**Purpose**: Adaptive bandwidth limiting per host.

**Key Features**:

- Host-specific bandwidth limits (Runpod: 100MB/s, Vast: 50MB/s)
- Concurrent transfer limiting
- Graceful degradation under load

**Config** (from `distributed_hosts.yaml`):

```yaml
auto_sync:
  bandwidth_limit_mbps: 100
  host_bandwidth_overrides:
    vast-*: 50
    runpod-*: 100
```

---

#### `sync_coordinator.py` - DEPRECATED

**Status**: Superseded by `auto_sync_daemon.py` (December 2025)

**Migration**: Use `AutoSyncDaemon` instead. Removal planned Q2 2026.

---

### Base Layer (Primitives)

#### `sync_base.py` - DATA STRUCTURES

```python
@dataclass
class SyncEntry:
    source_path: str
    target_host: str
    data_type: SyncCategory  # GAMES, MODELS, NPZ, DATABASES
    status: SyncStatus       # PENDING, IN_PROGRESS, COMPLETED, FAILED
    priority: int            # 0-100, higher = more urgent
```

---

#### `sync_constants.py` - CONFIGURATION CONSTANTS

```python
class SyncCategory(Enum):
    GAMES = "games"
    MODELS = "models"
    NPZ = "npz"
    DATABASES = "databases"

SYNC_TIMEOUT_SECONDS = 300
SYNC_RETRY_LIMIT = 3
SYNC_BATCH_SIZE = 100
```

---

#### `sync_mutex.py` - DISTRIBUTED LOCKING

**Purpose**: Prevents concurrent syncs to the same target.

**Usage**:

```python
async with sync_mutex.acquire("host1:games"):
    await sync_to_host("host1", games_data)
```

---

#### `sync_coordination_core.py` - CORE SYNC LOGIC

**Purpose**: Low-level push/pull operations.

**API**:

```python
await push_data(source, target, data_type)
await pull_data(source, target, data_type)
```

---

## Data Flow

### Normal Operation (Persistent Nodes)

```
Selfplay Complete → auto_sync_daemon (60s interval)
                         │
                         ▼
                   sync_router (find targets)
                         │
                         ▼
                   sync_bandwidth (rate limit)
                         │
                         ▼
                   sync_coordination_core (rsync/aria2)
                         │
                         ▼
                   ClusterManifest (register location)
```

### Ephemeral Nodes (Vast.ai)

```
Selfplay Complete → ephemeral_sync (5s interval, write-through)
                         │
                         ▼
                   sync_router (priority targets only)
                         │
                         ▼
                   sync_bandwidth (aggressive)
                         │
                         ▼
                   Confirmation wait (write-through mode)
                         │
                         ▼
                   Success/Failure → retry or emit HOST_OFFLINE
```

## Configuration

### `distributed_hosts.yaml`

```yaml
sync_routing:
  max_disk_usage_percent: 70
  replication_target: 2
  excluded_hosts:
    - name: mac-studio
      receive_games: false
      reason: coordinator/dev machine

auto_sync:
  enabled: true
  interval_seconds: 300
  gossip_interval_seconds: 60
  max_concurrent_syncs: 4
  min_games_to_sync: 10
```

## Troubleshooting

### Sync Not Working

1. Check `auto_sync_daemon` is running: `DaemonType.AUTO_SYNC`
2. Verify target not in `excluded_hosts`
3. Check disk space on target: `sync_routing.max_disk_usage_percent`

### Slow Sync

1. Check bandwidth limits: `host_bandwidth_overrides`
2. Verify network connectivity: P2P status
3. Check concurrent sync count: `max_concurrent_syncs`

### Data Loss on Ephemeral

1. Ensure `ephemeral_sync` daemon is running
2. Enable write-through mode for critical data
3. Check termination handlers installed

## See Also

- `docs/EVENT_CATALOG.md` - Sync-related events
- `app/distributed/cluster_manifest.py` - Data location registry
- `config/distributed_hosts.yaml` - Cluster configuration
