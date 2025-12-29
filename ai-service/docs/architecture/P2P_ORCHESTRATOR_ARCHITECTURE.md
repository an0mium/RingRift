# P2P Orchestrator Architecture

> **December 2025 Refactoring Summary**: The P2P orchestrator underwent major decomposition, extracting ~1,990 LOC into modular managers and background loops. This document describes the current architecture.

## Overview

The P2P orchestrator is a distributed coordination layer that enables autonomous cluster-wide training, selfplay, and data synchronization across 30+ nodes. It runs on each node and uses leader election to coordinate work distribution.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         P2P Orchestrator                            │
│                    (scripts/p2p_orchestrator.py)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Mixins    │  │  Managers   │  │    Loops    │  │  Handlers   │ │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤ │
│  │ Leadership  │  │ JobManager  │  │ JobReaper   │  │ /spawn/*    │ │
│  │ Membership  │  │ StateManager│  │ IdleDetect  │  │ /status     │ │
│  │ Gossip      │  │ SyncPlanner │  │ EloSync     │  │ /health     │ │
│  │ Consensus   │  │ NodeSelector│  │ AutoScale   │  │ /sync       │ │
│  │ PeerManager │  │ SelfplaySch │  │ SelfHealing │  │ /gossip     │ │
│  │             │  │ TrainingCrd │  │ DataSync    │  │             │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────────┤
│  │ Event System (DataEventType → Cross-Process Queue → Handlers)   │
│  └──────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────────┤
│  │ State Persistence (SQLite WAL mode, peers, jobs, leader state)  │
│  └──────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────┘
```

## Manager Classes

All managers are located in `scripts/p2p/managers/` and follow a dependency injection pattern for testability.

### Manager Overview

| Manager               | LOC   | Purpose             | Key Methods                                                          |
| --------------------- | ----- | ------------------- | -------------------------------------------------------------------- |
| `StateManager`        | 1,016 | SQLite persistence  | `load_state()`, `save_state()`, `get_cluster_epoch()`                |
| `NodeSelector`        | 741   | Node ranking        | `get_best_gpu_node()`, `get_training_nodes_ranked()`                 |
| `SyncPlanner`         | 1,508 | Data sync planning  | `collect_manifest()`, `create_sync_plan()`, `execute_sync()`         |
| `JobManager`          | 1,859 | Job lifecycle       | `run_gpu_selfplay_job()`, `spawn_training()`, `cleanup_stale_jobs()` |
| `SelfplayScheduler`   | 1,510 | Selfplay allocation | `pick_weighted_config()`, `get_target_jobs_for_node()`               |
| `TrainingCoordinator` | 1,363 | Training workflow   | `dispatch_training_job()`, `handle_completion()`                     |

### Dependency Injection Pattern

All managers receive callbacks to access orchestrator state rather than importing it directly:

```python
class MyManager:
    def __init__(
        self,
        get_peers: Callable[[], dict[str, NodeInfo]],
        get_self_info: Callable[[], NodeInfo],
        peers_lock: threading.Lock,
        config: ManagerConfig | None = None,
    ):
        self._get_peers = get_peers
        self._get_self_info = get_self_info
        self._peers_lock = peers_lock
```

**Benefits:**

- **Testability**: Unit tests use mock callbacks
- **Decoupling**: No circular imports with orchestrator
- **Flexibility**: Different data sources for testing vs production

### Manager Responsibilities

#### StateManager

- SQLite persistence for cluster state (WAL mode for durability)
- Database tables: `peers`, `jobs`, `state`, `metrics_history`, `peer_cache`, `config`
- Thread-safe operations with explicit locking
- Cluster epoch tracking for split-brain resolution

#### NodeSelector

- GPU power scoring: H100 > GH200 > A100 > L40S > RTX 4090 > RTX 3090
- Node filtering by health, availability, capabilities
- Training node selection based on VRAM and memory

#### SyncPlanner

- Local manifest collection (game DBs, models, NPZ files)
- Cluster manifest aggregation from all peers
- Sync plan generation based on file hashes
- Event emission: `DATA_SYNC_STARTED`, `DATA_SYNC_COMPLETED`

#### JobManager

- Engine mode routing (search modes → hybrid, simple modes → GPU)
- Job count tracking per node
- Stale job cleanup (1hr threshold)

#### SelfplayScheduler

- Priority calculation combining: static priority, Elo boost, curriculum weights
- Diversity tracking across configs
- Backpressure integration

#### TrainingCoordinator

- Training readiness checks (data thresholds)
- Post-training workflow: gauntlet → promotion → distribution
- Cooldown management to prevent duplicate training

---

## Background Loops

Loops run via `LoopManager` and provide continuous background processing. Located in `scripts/p2p/loops/`.

### Loop Overview

| Loop                       | File                          | Interval | Purpose                          |
| -------------------------- | ----------------------------- | -------- | -------------------------------- |
| `JobReaperLoop`            | `job_loops.py`                | 5 min    | Clean stale/stuck jobs           |
| `IdleDetectionLoop`        | `job_loops.py`                | 30 sec   | Detect idle GPUs, spawn selfplay |
| `WorkerPullLoop`           | `job_loops.py`                | 30 sec   | Workers poll leader for work     |
| `SelfHealingLoop`          | `resilience_loops.py`         | 5 min    | Recover stuck processes          |
| `PredictiveMonitoringLoop` | `resilience_loops.py`         | 5 min    | Trend analysis, early alerts     |
| `EloSyncLoop`              | `elo_sync_loop.py`            | 5 min    | Elo rating synchronization       |
| `QueuePopulatorLoop`       | `queue_populator_loop.py`     | 1 min    | Work queue maintenance           |
| `AutoScalingLoop`          | `coordination_loops.py`       | 5 min    | Scale cluster resources          |
| `ManifestCollectionLoop`   | `manifest_collection_loop.py` | 1 min    | Collect data manifests           |

### LoopManager

The `LoopManager` class (`scripts/p2p/loops/base.py`) coordinates all background loops:

```python
from scripts.p2p.loops.base import LoopManager

# Initialize
loop_manager = LoopManager()

# Register loops
loop_manager.register(job_reaper_loop)
loop_manager.register(idle_detection_loop)

# Start all loops
await loop_manager.start_all()

# Stop all loops gracefully
await loop_manager.stop_all()
```

### Loop Health Checks

All loops implement `health_check()` returning standardized status:

```python
{
    "healthy": True,
    "message": "Loop running normally",
    "details": {
        "cycles_completed": 142,
        "last_cycle_time": "2025-12-28T10:30:00Z",
        "errors_count": 0,
        "average_cycle_duration_ms": 1523,
    }
}
```

---

## Mixin Classes

Mixins provide modular functionality that the orchestrator composes. Located in `scripts/p2p/`.

### Mixin Overview

| Mixin                 | LOC    | Purpose                                 |
| --------------------- | ------ | --------------------------------------- |
| `P2PMixinBase`        | 995    | Shared helpers (DB, logging, state)     |
| `LeaderElectionMixin` | 800+   | Bully election, lease management        |
| `MembershipMixin`     | 330    | HTTP polling membership (optional SWIM) |
| `GossipProtocolMixin` | 1,000+ | Gossip propagation, metrics             |
| `ConsensusMixin`      | 709    | Vote collection (optional Raft)         |
| `PeerManagerMixin`    | 450+   | Peer lifecycle, retirement              |

### P2PMixinBase Helpers

```python
from scripts.p2p.p2p_mixin_base import P2PMixinBase

class MyMixin(P2PMixinBase):
    MIXIN_TYPE = "my_mixin"

    def my_method(self):
        # Database helpers
        self._execute_db_query("SELECT * FROM peers")

        # State management
        self._ensure_state_attr("_cache", {})

        # Event emission
        self._safe_emit_event("MY_EVENT", {"data": "value"})

        # Logging (prefixed with MIXIN_TYPE)
        self._log_info("Operation completed")
```

### EventSubscriptionMixin

Standardized event subscription for managers:

```python
from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

class MyManager(EventSubscriptionMixin):
    _subscription_log_prefix = "MyManager"

    def _get_event_subscriptions(self) -> dict:
        return {
            "HOST_OFFLINE": self._on_host_offline,
            "NODE_RECOVERED": self._on_node_recovered,
        }

    async def _on_host_offline(self, event) -> None:
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id")
        self._log_info(f"Host offline: {node_id}")

# Subscribe during initialization
manager = MyManager()
manager.subscribe_to_events()
```

---

## Event Flow

The P2P orchestrator integrates with the unified event system for cross-component coordination.

### Key Events Emitted

| Event                 | Emitter             | Purpose                 |
| --------------------- | ------------------- | ----------------------- |
| `HOST_OFFLINE`        | PeerManagerMixin    | Node went offline       |
| `HOST_ONLINE`         | PeerManagerMixin    | Node came back online   |
| `LEADER_ELECTED`      | LeaderElectionMixin | Leadership change       |
| `DATA_SYNC_STARTED`   | SyncPlanner         | Sync operation began    |
| `DATA_SYNC_COMPLETED` | SyncPlanner         | Sync operation finished |
| `TRAINING_STARTED`    | TrainingCoordinator | Training job dispatched |
| `TRAINING_COMPLETED`  | TrainingCoordinator | Training finished       |
| `SELFPLAY_COMPLETE`   | JobManager          | Selfplay batch finished |

### Event Subscription Example

```python
# In TrainingCoordinator
def _get_event_subscriptions(self) -> dict:
    return {
        "DATA_SYNC_COMPLETED": self._on_sync_complete,
        "SELFPLAY_COMPLETE": self._on_selfplay_complete,
    }

async def _on_sync_complete(self, event):
    # Check if training should start
    if self.check_training_readiness():
        await self.dispatch_training_job()
```

---

## HTTP Endpoints

The orchestrator exposes HTTP endpoints on port 8770 for inter-node communication.

| Endpoint          | Method | Purpose                                |
| ----------------- | ------ | -------------------------------------- |
| `/status`         | GET    | Full node status (leader, peers, jobs) |
| `/health`         | GET    | Quick health check                     |
| `/gossip`         | POST   | Receive gossip from peers              |
| `/spawn/selfplay` | POST   | Spawn selfplay job                     |
| `/spawn/training` | POST   | Spawn training job                     |
| `/sync/manifest`  | GET    | Get local data manifest                |
| `/sync/pull`      | POST   | Pull data from this node               |

### Status Response Structure

```json
{
  "node_id": "nebius-h100-3",
  "role": "LEADER",
  "leader_id": "nebius-h100-3",
  "alive_peers": 28,
  "total_peers": 36,
  "uptime_seconds": 86400,
  "jobs": {
    "running": 12,
    "pending": 3,
    "completed_24h": 847
  },
  "managers": {
    "job_manager": { "healthy": true },
    "sync_planner": { "healthy": true },
    "selfplay_scheduler": { "healthy": true }
  },
  "loops": {
    "job_reaper": { "healthy": true, "cycles": 288 },
    "idle_detection": { "healthy": true, "cycles": 2880 }
  }
}
```

---

## Leader Election

The P2P cluster uses a Bully election algorithm with lease-based leadership.

### Election Flow

```
1. Startup
   └── Load persisted state (including vote grants)

2. Election Trigger
   ├── Leader lease expired
   ├── Leader went offline
   └── Explicit stepdown request

3. Election Phase
   ├── Collect votes from higher-ranked nodes
   ├── If no higher node responds → become leader
   └── If higher node responds → defer to it

4. Leader Duties
   ├── Renew lease every 30 seconds
   ├── Dispatch training/selfplay jobs
   ├── Coordinate data sync
   └── Monitor cluster health
```

### Voter Configuration

P2P voters are stable, non-NAT-blocked nodes (configured in `distributed_hosts.yaml`):

```yaml
p2p_voters:
  - nebius-backbone-1
  - nebius-h100-3
  - hetzner-cpu1
  - hetzner-cpu2
  - vultr-a100-20gb
  - runpod-a100-1
  - runpod-a100-2
  # With 7 voters, quorum = 4
```

---

## State Persistence

### SQLite Schema

The orchestrator uses SQLite with WAL mode for state persistence:

```sql
-- Core tables
CREATE TABLE peers (
    node_id TEXT PRIMARY KEY,
    last_seen REAL,
    status TEXT,
    metadata JSON
);

CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT,
    node_id TEXT,
    status TEXT,
    created_at REAL,
    updated_at REAL,
    config JSON
);

CREATE TABLE state (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Metrics and cache
CREATE TABLE metrics_history (...);
CREATE TABLE peer_cache (...);
CREATE TABLE config (...);
```

### State Recovery

On startup, the orchestrator:

1. Loads persisted state (peers, jobs, leader info)
2. Validates leader lease (may trigger election)
3. Reconnects to known peers
4. Resumes background loops

---

## Health Check Integration

All components implement `health_check()` for unified monitoring:

```python
# Aggregated by DaemonManager
def health_check(self) -> HealthCheckResult:
    manager_health = self._validate_manager_health()
    loop_health = self._validate_loop_health()

    is_healthy = all([
        manager_health["all_healthy"],
        loop_health["all_healthy"],
        self._is_leader_healthy(),
    ])

    return HealthCheckResult(
        healthy=is_healthy,
        message="P2P orchestrator status",
        details={
            "node_id": self.node_id,
            "role": self.role.name,
            "managers": manager_health,
            "loops": loop_health,
        }
    )
```

---

## Configuration

### Environment Variables

| Variable                            | Default | Description                                 |
| ----------------------------------- | ------- | ------------------------------------------- |
| `RINGRIFT_P2P_PORT`                 | 8770    | HTTP server port                            |
| `RINGRIFT_P2P_LEADER_LEASE_SECONDS` | 60      | Leader lease duration                       |
| `RINGRIFT_HEARTBEAT_INTERVAL`       | 15      | Peer heartbeat interval                     |
| `RINGRIFT_PEER_TIMEOUT`             | 60      | Peer timeout for retirement                 |
| `RINGRIFT_EXTRACTED_LOOPS`          | true    | Use LoopManager for background loops        |
| `RINGRIFT_P2P_STARTUP_GRACE_PERIOD` | 120     | Grace period before killing stale processes |

### Feature Flags

| Flag                       | Default | Description                     |
| -------------------------- | ------- | ------------------------------- |
| `RINGRIFT_SWIM_ENABLED`    | false   | Enable SWIM membership protocol |
| `RINGRIFT_RAFT_ENABLED`    | false   | Enable Raft consensus           |
| `RINGRIFT_MEMBERSHIP_MODE` | http    | Membership: http, swim, hybrid  |
| `RINGRIFT_CONSENSUS_MODE`  | bully   | Consensus: bully, raft, hybrid  |

---

## December 2025 Refactoring Summary

### Code Removed from p2p_orchestrator.py

| Category                       | Methods Removed | LOC Saved  |
| ------------------------------ | --------------- | ---------- |
| StateManager delegation        | 7 methods       | ~200       |
| JobManager delegation          | 7 methods       | ~400       |
| SyncPlanner delegation         | 4 methods       | ~60        |
| SelfplayScheduler delegation   | 7 methods       | ~430       |
| TrainingCoordinator delegation | 5 methods       | ~450       |
| NodeSelector delegation        | 6 methods       | ~50        |
| Background loops extraction    | 5 loops         | ~400       |
| **Total**                      | **41 methods**  | **~1,990** |

### New Module Structure

```
scripts/p2p/
├── p2p_orchestrator.py        # Main orchestrator (now slimmer)
├── p2p_mixin_base.py          # Base classes and helpers
├── managers/
│   ├── state_manager.py
│   ├── node_selector.py
│   ├── sync_planner.py
│   ├── job_manager.py
│   ├── selfplay_scheduler.py
│   └── training_coordinator.py
├── loops/
│   ├── base.py                # LoopManager
│   ├── job_loops.py           # JobReaper, IdleDetection
│   ├── resilience_loops.py    # SelfHealing, Predictive
│   ├── elo_sync_loop.py
│   └── ...
├── handlers/
│   ├── handlers_base.py
│   ├── spawn_handlers.py
│   └── status_handlers.py
└── mixins/
    ├── leader_election.py
    ├── membership_mixin.py
    ├── gossip_protocol.py
    └── consensus_mixin.py
```

---

## See Also

- `scripts/p2p/managers/README.md` - Manager module documentation
- `docs/SYNC_STRATEGY_GUIDE.md` - Data sync strategies
- `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - P2P troubleshooting
- `docs/CLUSTER_INTEGRATION_GUIDE.md` - Cluster architecture
