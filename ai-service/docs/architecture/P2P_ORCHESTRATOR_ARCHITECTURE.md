# P2P Orchestrator Architecture

Comprehensive architecture documentation for the RingRift P2P orchestration layer.

**Created**: December 28, 2025
**Last Updated**: December 28, 2025

---

## Overview

The P2P orchestrator is the distributed coordination brain of RingRift's AI training infrastructure. It manages:

- **Cluster membership**: Node discovery, health tracking, and peer-to-peer gossip
- **Leader election**: Bully-based election with optional SWIM/Raft upgrades
- **Job orchestration**: Selfplay, training, and tournament job dispatch
- **Data synchronization**: Cross-node game and model data sync
- **Resource management**: GPU utilization and idle detection

**Location**: `scripts/p2p_orchestrator.py` (~25,000 LOC after Dec 2025 cleanup)

---

## Architecture Diagram

```
                         ┌─────────────────────────────────────────────────┐
                         │              P2P Orchestrator                   │
                         │                 (Leader)                        │
                         └─────────────────────────────────────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
              ▼                             ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │   P2P Follower  │◀────────▶│   P2P Follower  │◀────────▶│   P2P Follower  │
    │   (Worker)      │  gossip  │   (Worker)      │  gossip  │   (Worker)      │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
           │                            │                            │
           ▼                            ▼                            ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │  GPU Selfplay   │          │   Training      │          │  GPU Selfplay   │
    │     Jobs        │          │     Job         │          │     Jobs        │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
```

---

## Manager Delegation Pattern

The P2P orchestrator uses a modular manager architecture for separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          P2P Orchestrator                                   │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ StateManager│  │NodeSelector │  │ JobManager  │  │SelfplaySched│        │
│  │             │  │             │  │             │  │    uler     │        │
│  │ - SQLite    │  │ - GPU/CPU   │  │ - Spawn     │  │ - Priority  │        │
│  │ - State     │  │   ranking   │  │ - Track     │  │ - Diversity │        │
│  │ - Epochs    │  │ - Selection │  │ - Cleanup   │  │ - Curriculum│        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────────┐ │
│  │ SyncPlanner │  │TrainingCoord│  │            LoopManager              │ │
│  │             │  │   inator    │  │                                     │ │
│  │ - Manifest  │  │ - Dispatch  │  │ JobReaper, IdleDetection, EloSync, │ │
│  │ - Planning  │  │ - Gauntlet  │  │ QueuePopulator, SelfHealing, ...   │ │
│  │ - Sync exec │  │ - Promotion │  │                                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Manager Summary

| Manager               | Responsibility                       | LOC  | Key Methods                                               |
| --------------------- | ------------------------------------ | ---- | --------------------------------------------------------- |
| `StateManager`        | SQLite persistence, cluster epochs   | ~450 | `load_state()`, `save_state()`                            |
| `NodeSelector`        | Node ranking and selection           | ~350 | `get_best_gpu_node()`, `get_training_nodes_ranked()`      |
| `JobManager`          | Job spawning and lifecycle           | ~550 | `run_gpu_selfplay_job()`, `spawn_training()`              |
| `SelfplayScheduler`   | Priority-based config selection      | ~600 | `pick_weighted_config()`, `get_target_jobs_for_node()`    |
| `SyncPlanner`         | Data sync planning and execution     | ~450 | `collect_manifest()`, `create_sync_plan()`                |
| `TrainingCoordinator` | Training job dispatch and completion | ~500 | `dispatch_training_job()`, `handle_training_completion()` |
| `LoopManager`         | Background loop coordination         | ~300 | `start_all()`, `stop_all()`, `health_check()`             |

**Full manager documentation**: See `scripts/p2p/managers/README.md`

---

## Leader Election

### Bully Algorithm (Production)

The default leader election uses a modified Bully algorithm:

1. **Node ordering**: Nodes are ranked by (IP, port) tuple
2. **Election trigger**: When leader is unresponsive for >60 seconds
3. **Election process**:
   - Node sends ELECTION message to all higher-ranked nodes
   - If no response in 10 seconds, node declares itself leader
   - If higher node responds, yield to that node's election
4. **Quorum**: Requires 3+ nodes from voter set for valid election

```python
# Voter configuration in distributed_hosts.yaml
p2p_voters:
  - nebius-backbone-1
  - nebius-h100-3
  - hetzner-cpu1
  - hetzner-cpu2
  - vultr-a100-20gb
```

### Optional SWIM/Raft (Experimental)

**SWIM Protocol** (membership):

- Gossip-based membership with 5s failure detection
- Requires `swim-p2p>=1.2.0`
- Enable: `export RINGRIFT_SWIM_ENABLED=true`

**Raft Protocol** (consensus):

- Replicated work queue with sub-second failover
- Requires `pysyncobj>=0.3.14`
- Enable: `export RINGRIFT_RAFT_ENABLED=true`

---

## Background Loops

The LoopManager coordinates all background processing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LoopManager                                      │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  JobReaperLoop  │  │IdleDetectionLoop│  │   EloSyncLoop   │             │
│  │   (5 min)       │  │   (30 sec)      │  │   (5 min)       │             │
│  │                 │  │                 │  │                 │             │
│  │ Clean stale     │  │ Detect idle     │  │ Sync Elo        │             │
│  │ and stuck jobs  │  │ GPUs, trigger   │  │ ratings across  │             │
│  │                 │  │ selfplay        │  │ cluster         │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │QueuePopulatorLp │  │ SelfHealingLoop │  │ WorkerPullLoop  │             │
│  │   (1 min)       │  │   (5 min)       │  │   (30 sec)      │             │
│  │                 │  │                 │  │                 │             │
│  │ Maintain work   │  │ Recover stuck   │  │ Workers poll    │             │
│  │ queue until     │  │ jobs, clean     │  │ leader for new  │             │
│  │ Elo targets met │  │ stale processes │  │ work            │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ManifestCollLoop │  │TrainingSyncLoop │  │PredictiveMonLoop│             │
│  │   (1 min)       │  │   (5 min)       │  │   (5 min)       │             │
│  │                 │  │                 │  │                 │             │
│  │ Collect data    │  │ Sync training   │  │ Track trends,   │             │
│  │ manifests from  │  │ data to         │  │ emit alerts     │             │
│  │ peers           │  │ training nodes  │  │ before threshold│             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Creating Custom Loops

```python
from scripts.p2p.loops import BaseLoop

class MyCustomLoop(BaseLoop):
    def __init__(self, get_data_fn, process_fn):
        super().__init__(
            name="my_custom_loop",
            interval=60.0,  # Run every 60 seconds
            depends_on=["elo_sync"],  # Start after elo_sync
        )
        self.get_data = get_data_fn
        self.process = process_fn

    async def _run_once(self) -> None:
        """Execute one iteration of the loop."""
        data = self.get_data()
        if data:
            await self.process(data)

    async def _on_start(self) -> None:
        self._log_info("Starting custom loop")

    async def _on_error(self, error: Exception) -> None:
        self._log_error(f"Error: {error}")
```

---

## Event System Integration

The P2P orchestrator emits events for coordination with the daemon layer:

### Events Emitted

| Event                 | Trigger                      | Subscribers                     |
| --------------------- | ---------------------------- | ------------------------------- |
| `HOST_OFFLINE`        | Peer retired (offline >300s) | UnifiedHealthManager            |
| `HOST_ONLINE`         | Retired peer recovers        | UnifiedHealthManager            |
| `LEADER_ELECTED`      | This node becomes leader     | LeadershipCoordinator           |
| `DATA_SYNC_STARTED`   | Sync operation begins        | DataPipelineOrchestrator        |
| `DATA_SYNC_COMPLETED` | Sync operation ends          | DataPipelineOrchestrator        |
| `TRAINING_STARTED`    | Training job dispatched      | SyncRouter, IdleShutdown        |
| `TRAINING_COMPLETED`  | Training job finishes        | FeedbackLoop, ModelDistribution |

### Event Emission Pattern

```python
def _emit_p2p_host_offline(self, node_id: str) -> None:
    """Emit HOST_OFFLINE event when a peer is retired."""
    try:
        from app.coordination.event_emitters import emit_host_offline
        emit_host_offline(node_id=node_id, source="p2p_orchestrator")
    except ImportError:
        logger.debug("Event emitters not available")
    except Exception as e:
        logger.warning(f"Failed to emit HOST_OFFLINE: {e}")
```

---

## HTTP API Endpoints

The orchestrator exposes a REST API on port 8770:

### Core Endpoints

| Endpoint  | Method | Description                         |
| --------- | ------ | ----------------------------------- |
| `/status` | GET    | Cluster status, leader info, health |
| `/health` | GET    | Simple health check (200/503)       |
| `/peers`  | GET    | List all known peers                |
| `/jobs`   | GET    | List active jobs                    |

### Job Control (Leader Only)

| Endpoint               | Method | Description           |
| ---------------------- | ------ | --------------------- |
| `/dispatch_selfplay`   | POST   | Dispatch selfplay job |
| `/dispatch_training`   | POST   | Dispatch training job |
| `/cancel_job/{job_id}` | POST   | Cancel a running job  |

### Data Sync (Leader Only)

| Endpoint     | Method | Description             |
| ------------ | ------ | ----------------------- |
| `/sync_data` | POST   | Trigger data sync       |
| `/manifest`  | GET    | Get local data manifest |

### Example Usage

```bash
# Check cluster status
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive peers: {d.get(\"alive_peers\")}")
print(f"Role: {d.get(\"role\")}")
'

# Dispatch selfplay (if leader)
curl -X POST http://localhost:8770/dispatch_selfplay \
  -H "Content-Type: application/json" \
  -d '{"board_type": "hex8", "num_players": 2, "num_games": 100}'
```

---

## State Persistence

### SQLite State Database

Location: `data/coordination/p2p_state.db`

**Tables**:

| Table        | Purpose                                 |
| ------------ | --------------------------------------- |
| `peers`      | Node information (IP, port, last_seen)  |
| `jobs`       | Running job records                     |
| `state`      | Key-value state (leader, role)          |
| `config`     | Cluster epoch and settings              |
| `peer_cache` | Persistent peer storage with reputation |

### Cluster Epochs

Epochs prevent split-brain scenarios:

1. Each leader election increments the epoch
2. Stale leaders (lower epoch) yield to new leaders
3. Epoch is persisted and synchronized via gossip

```python
# StateManager epoch handling
def increment_epoch(self) -> int:
    """Increment and persist cluster epoch."""
    with self._db_lock:
        self._current_epoch += 1
        self._execute("UPDATE config SET epoch = ?", (self._current_epoch,))
        return self._current_epoch
```

---

## Health Monitoring

### Health Check Aggregation

The orchestrator aggregates health from all managers:

```python
def health_check(self) -> HealthCheckResult:
    """Aggregate health from all managers."""
    manager_health = self._validate_manager_health()

    if not manager_health["all_healthy"]:
        return HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.DEGRADED,
            message=f"Unhealthy managers: {manager_health['unhealthy']}",
            details=manager_health,
        )

    return HealthCheckResult(
        healthy=True,
        status=CoordinatorStatus.RUNNING,
        message="P2P orchestrator operational",
        details={
            "node_id": self.node_id,
            "role": self.role.name,
            "leader_id": self.leader_id,
            "active_peers": len(self.peers),
            "uptime_seconds": time.time() - self._start_time,
        },
    )
```

### Manager Health Metrics

| Manager             | Key Health Metrics                   |
| ------------------- | ------------------------------------ |
| StateManager        | DB connection, pending writes, epoch |
| NodeSelector        | Cache freshness, selection latency   |
| TrainingCoordinator | Active jobs, cooldown status         |
| JobManager          | Active jobs, spawn rate, errors      |
| SelfplayScheduler   | Diversity metrics, curriculum state  |
| SyncPlanner         | Sync in progress, manifest freshness |
| LoopManager         | Loops running, failing loops         |

---

## Gossip Protocol

### Peer Discovery

Nodes discover each other via:

1. **YAML configuration**: Initial peer list from `distributed_hosts.yaml`
2. **Gossip protocol**: Peers share their peer lists
3. **Health probing**: Periodic `/status` checks (every 15 seconds)

### Gossip Message Types

| Message     | Direction                | Content                     |
| ----------- | ------------------------ | --------------------------- |
| `HELLO`     | Node → Peers             | Node identity, capabilities |
| `PEERS`     | Leader → Nodes           | Full peer list              |
| `HEARTBEAT` | Node → Leader            | Health status, job count    |
| `ELECTION`  | Candidate → Higher nodes | Election initiation         |
| `LEADER`    | Winner → All             | New leader announcement     |

### Failure Detection

- **Heartbeat interval**: 15 seconds (reduced from 30s in Dec 2025)
- **Peer timeout**: 60 seconds (reduced from 90s)
- **Retirement threshold**: 300 seconds (5 minutes)

---

## Configuration

### Environment Variables

| Variable                            | Default | Description                                 |
| ----------------------------------- | ------- | ------------------------------------------- |
| `RINGRIFT_P2P_PORT`                 | 8770    | P2P HTTP API port                           |
| `RINGRIFT_P2P_HEARTBEAT_INTERVAL`   | 15      | Heartbeat interval (seconds)                |
| `RINGRIFT_P2P_PEER_TIMEOUT`         | 60      | Peer timeout (seconds)                      |
| `RINGRIFT_P2P_STARTUP_GRACE_PERIOD` | 120     | Grace period before killing stuck processes |
| `RINGRIFT_SWIM_ENABLED`             | false   | Enable SWIM protocol                        |
| `RINGRIFT_RAFT_ENABLED`             | false   | Enable Raft protocol                        |

### distributed_hosts.yaml

```yaml
# P2P voter nodes (must be stable, non-NAT-blocked)
p2p_voters:
  - nebius-backbone-1
  - nebius-h100-3
  - hetzner-cpu1
  - hetzner-cpu2
  - vultr-a100-20gb

# Node definitions
nodes:
  nebius-backbone-1:
    tailscale_ip: '100.x.x.x'
    ssh_host: '89.169.112.47'
    status: ready
    role: backbone
    gpu: L40S
    is_coordinator: false
```

---

## December 2025 Refactoring Summary

### LOC Removed from p2p_orchestrator.py

| Phase              | LOC Removed | Description                |
| ------------------ | ----------- | -------------------------- |
| Manager Delegation | ~1,990      | 7 managers fully delegated |
| Loop Extraction    | ~400        | 5 loops to LoopManager     |
| Dead Code Removal  | ~186        | Legacy manifest, wrappers  |
| **Total**          | **~2,576**  | From ~28,000 to ~25,000    |

### Key Improvements

1. **Modular managers**: Each concern in separate, testable module
2. **Background loops**: Centralized via LoopManager with dependency ordering
3. **Health integration**: All managers report to DaemonManager
4. **Event emission**: P2P lifecycle events for coordination layer
5. **Faster failure detection**: 15s heartbeat, 60s timeout

---

## Related Documentation

- `scripts/p2p/managers/README.md` - Manager delegation details
- `docs/architecture/COORDINATION_SYSTEM.md` - Coordination layer overview
- `docs/architecture/DAEMON_LIFECYCLE.md` - Daemon management
- `docs/runbooks/P2P_LEADER_FAILOVER.md` - Leader election troubleshooting
- `docs/runbooks/P2P_ORCHESTRATOR_OPERATIONS.md` - Operational procedures
