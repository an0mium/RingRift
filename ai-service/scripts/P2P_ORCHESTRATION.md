# P2P Orchestration System Documentation

The P2P orchestrator is a distributed coordination system for RingRift AI training.
It manages selfplay, training, model distribution, and data synchronization across a cluster of GPU nodes.

## Quick Start

### Starting the P2P Cluster

```bash
# Start P2P orchestrator on a node
cd ai-service
python scripts/p2p_orchestrator.py --node-id my-node

# Or with specific configuration
python scripts/p2p_orchestrator.py \
  --node-id runpod-h100 \
  --port 8770 \
  --voter  # Mark as voting node
```

### Checking Cluster Status

```bash
# Quick status check
curl -s http://localhost:8770/status | python3 -c '
import sys,json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive: {d.get(\"alive_peers\")} nodes")
'

# Comprehensive cluster status
python scripts/check_p2p_status.py

# All nodes status (parallel SSH)
python scripts/check_p2p_status_all_nodes.py
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         P2P Orchestrator (p2p_orchestrator.py)              │
│                              ~27,000 lines, port 8770                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Leader Election│  │  Gossip Protocol│  │   Work Queue                │  │
│  │  Bully Algorithm│  │  State Sync     │  │   Job Assignment            │  │
│  │  Voter Quorum   │  │  Peer Discovery │  │   Auto-scaling              │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                    │                          │                  │
│  ┌────────┴────────────────────┴──────────────────────────┴──────────────┐  │
│  │                         HTTP REST API                                  │  │
│  │  /status, /health, /work/*, /election/*, /gossip/*, /admin/*         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           Handlers (scripts/p2p/handlers/)            │  │
│  │  work_queue.py | election.py | gossip.py | gauntlet.py | relay.py    │  │
│  │  tournament.py | cmaes.py | elo_sync.py | admin.py | swim.py | raft.py│  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           Managers (scripts/p2p/managers/)            │  │
│  │  StateManager      | NodeSelector     | SyncPlanner                   │  │
│  │  JobManager        | SelfplayScheduler| TrainingCoordinator           │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           Background Loops (scripts/p2p/loops/)       │  │
│  │  base.py (LoopRunner) | elo_sync_loop.py | queue_populator_loop.py   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
scripts/
├── p2p_orchestrator.py           # Main orchestrator (27K lines)
├── p2p/                          # Package modules
│   ├── __init__.py               # Package exports
│   ├── constants.py              # Configuration constants
│   ├── types.py                  # Enums: NodeRole, JobType
│   ├── models.py                 # Dataclasses: NodeInfo, ClusterJob
│   ├── client.py                 # P2P client for external use
│   ├── network.py                # HTTP client, circuit breaker
│   ├── network_utils.py          # URL building, Tailscale detection
│   ├── peer_manager.py           # Peer cache & reputation
│   ├── leader_election.py        # Voter quorum, consistency
│   ├── gossip_protocol.py        # State synchronization
│   ├── gossip_metrics.py         # Gossip health monitoring
│   ├── metrics_manager.py        # Metrics recording
│   ├── resource_detector.py      # System resource detection
│   ├── p2p_event_bridge.py       # Event system bridge
│   ├── cluster_config.py         # Cluster configuration
│   ├── consensus_mixin.py        # Raft consensus (optional)
│   ├── membership_mixin.py       # SWIM membership (optional)
│   ├── handlers/                 # HTTP handlers
│   │   ├── work_queue.py         # Job queue handlers
│   │   ├── election.py           # Election handlers
│   │   ├── gossip.py             # Gossip handlers
│   │   ├── gauntlet.py           # Evaluation handlers
│   │   ├── relay.py              # NAT relay handlers
│   │   ├── tournament.py         # Tournament handlers
│   │   ├── cmaes.py              # CMA-ES handlers
│   │   ├── elo_sync.py           # Elo sync handlers
│   │   ├── admin.py              # Admin handlers
│   │   ├── swim.py               # SWIM protocol handlers
│   │   └── raft.py               # Raft protocol handlers
│   ├── managers/                 # Manager classes
│   │   ├── state_manager.py      # SQLite persistence
│   │   ├── node_selector.py      # Node ranking/selection
│   │   ├── sync_planner.py       # Sync planning
│   │   ├── job_manager.py        # Job lifecycle
│   │   ├── selfplay_scheduler.py # Priority scheduling
│   │   └── training_coordinator.py # Training dispatch
│   └── loops/                    # Background loops
│       ├── base.py               # LoopRunner base class
│       ├── elo_sync_loop.py      # Elo synchronization
│       └── queue_populator_loop.py # Queue population
├── check_p2p_status.py           # Status checker
├── check_p2p_comprehensive.py    # Comprehensive check
├── check_p2p_status_all_nodes.py # Parallel SSH status
├── deploy_p2p_cluster.py         # Cluster deployment
├── batch_deploy_p2p.py           # Batch deployment
├── start_p2p_cluster.py          # Cluster startup
└── update_all_nodes.py           # Parallel code update
```

## Configuration

### Environment Variables

| Variable                            | Default    | Description                  |
| ----------------------------------- | ---------- | ---------------------------- |
| `P2P_NODE_ID`                       | hostname   | Unique node identifier       |
| `P2P_PORT`                          | 8770       | HTTP API port                |
| `P2P_DATA_DIR`                      | `data/p2p` | State persistence directory  |
| `RINGRIFT_P2P_HEARTBEAT_INTERVAL`   | 15         | Heartbeat interval (seconds) |
| `RINGRIFT_P2P_PEER_TIMEOUT`         | 60         | Peer timeout (seconds)       |
| `RINGRIFT_P2P_STARTUP_GRACE_PERIOD` | 120        | Startup grace period         |

### Cluster Configuration

Nodes are configured in `config/distributed_hosts.yaml`:

```yaml
# Voter nodes (stable, for quorum)
voters:
  - nebius-backbone-1
  - runpod-h100
  - runpod-a100-1
  - runpod-a100-2
  - vultr-a100-20gb

# All cluster nodes
nodes:
  runpod-h100:
    host: '102.210.171.65'
    port: 30178
    provider: runpod
    gpu: H100

  vast-5090-1:
    host: 'example.vast.ai'
    port: 22
    provider: vast
    gpu: RTX_5090
```

## API Endpoints

### Health & Status

| Endpoint  | Method | Description                       |
| --------- | ------ | --------------------------------- |
| `/health` | GET    | Health check (for load balancers) |
| `/status` | GET    | Full cluster status               |
| `/peers`  | GET    | List all known peers              |

### Work Queue

| Endpoint                   | Method | Description                |
| -------------------------- | ------ | -------------------------- |
| `/work/add`                | POST   | Add work item to queue     |
| `/work/claim`              | POST   | Claim work item            |
| `/work/complete`           | POST   | Mark work as completed     |
| `/work/fail`               | POST   | Mark work as failed        |
| `/work/status`             | GET    | Get work queue status      |
| `/work/for-node/<node_id>` | GET    | Get work for specific node |

### Election

| Endpoint           | Method | Description             |
| ------------------ | ------ | ----------------------- |
| `/election/status` | GET    | Current election status |
| `/election/start`  | POST   | Initiate election       |
| `/lease/request`   | POST   | Request leader lease    |
| `/lease/status`    | GET    | Get lease status        |

### Gossip

| Endpoint        | Method | Description           |
| --------------- | ------ | --------------------- |
| `/gossip`       | POST   | Exchange gossip state |
| `/gossip/peers` | GET    | Get gossip peer list  |

### Admin

| Endpoint          | Method   | Description        |
| ----------------- | -------- | ------------------ |
| `/admin/shutdown` | POST     | Graceful shutdown  |
| `/admin/reset`    | POST     | Reset node state   |
| `/admin/config`   | GET/POST | View/update config |

## CLI Commands

### Status Checking

```bash
# Check local P2P status
python scripts/check_p2p_status.py

# Comprehensive cluster check
python scripts/check_p2p_comprehensive.py

# Check all nodes via SSH
python scripts/check_p2p_status_all_nodes.py

# Get cluster status as JSON
curl -s http://localhost:8770/status | jq .
```

### Deployment

```bash
# Deploy P2P to cluster
python scripts/deploy_p2p_cluster.py

# Batch deploy with specific options
python scripts/batch_deploy_p2p.py --restart

# Update all nodes to latest code
python scripts/update_all_nodes.py --restart-p2p
```

### Job Management

```bash
# Add selfplay job
curl -X POST http://localhost:8770/work/add \
  -H "Content-Type: application/json" \
  -d '{"work_type": "selfplay", "board_type": "hex8", "num_players": 2, "num_games": 1000}'

# Check work queue
curl -s http://localhost:8770/work/status | jq .

# Claim work (node auto-claims)
curl -X POST http://localhost:8770/work/claim \
  -H "Content-Type: application/json" \
  -d '{"node_id": "my-node"}'
```

## Leader Election

The P2P cluster uses a Bully-based election algorithm with voter quorum:

1. **Voters**: Stable nodes designated for quorum (min 3 for consensus)
2. **Leases**: Leader acquires lease from majority of voters
3. **Health Checks**: Leader is monitored; election triggered on failure
4. **Partition Handling**: Local election when main quorum unreachable

### Election Flow

```
1. Node detects leader failure
2. Node initiates election (higher priority wins)
3. Winner requests lease from voters
4. Voters grant lease if:
   - No current lease OR
   - Current lease expired OR
   - New leader has higher priority
5. Winner becomes leader with lease
```

## Work Queue

The distributed work queue manages all cluster jobs:

### Work Types

| Type         | Description               |
| ------------ | ------------------------- |
| `selfplay`   | Self-play game generation |
| `training`   | Model training            |
| `evaluation` | Model evaluation/gauntlet |
| `tournament` | Elo tournament            |
| `sync`       | Data synchronization      |

### Job States

```
PENDING → ASSIGNED → RUNNING → COMPLETED
                  ↘           ↗
                   → FAILED →
```

### Priority Scheduling

Jobs are scheduled based on:

1. **Config Priority**: Curriculum weights for each board type
2. **Elo Velocity**: Configs with faster improvement get more resources
3. **Data Freshness**: Configs with stale data get priority
4. **GPU Matching**: Jobs matched to appropriate GPU types

## Data Synchronization

### Sync Strategy

1. **Push from Generator**: Nodes push generated data immediately
2. **Gossip Replication**: Data locations gossip to all nodes
3. **Training Freshness**: Nodes pull fresh data before training

### Sync Types

| Type     | Direction | Description                      |
| -------- | --------- | -------------------------------- |
| `games`  | Push      | Game databases to training nodes |
| `models` | Pull      | Models to selfplay nodes         |
| `npz`    | Both      | NPZ training data                |

## Troubleshooting

### Common Issues

**P2P not starting**

```bash
# Check if port is in use
lsof -i :8770

# Kill existing process
pkill -f p2p_orchestrator
```

**No leader elected**

```bash
# Check voter status
curl -s http://localhost:8770/election/status | jq .

# Force election
curl -X POST http://localhost:8770/election/start
```

**Nodes not discovering each other**

```bash
# Check gossip status
curl -s http://localhost:8770/gossip/peers | jq .

# Verify network connectivity
python scripts/check_p2p_comprehensive.py
```

**Work not being distributed**

```bash
# Check work queue
curl -s http://localhost:8770/work/status | jq .

# Check node availability
curl -s http://localhost:8770/status | jq .alive_peers
```

### Log Locations

| Location                    | Description              |
| --------------------------- | ------------------------ |
| `logs/p2p_orchestrator.log` | Main P2P logs            |
| `data/p2p/state.db`         | SQLite state persistence |
| `data/p2p/metrics.json`     | Metrics history          |

### Debug Mode

```bash
# Enable debug logging
P2P_LOG_LEVEL=DEBUG python scripts/p2p_orchestrator.py

# Trace gossip messages
RINGRIFT_TRACE_GOSSIP=1 python scripts/p2p_orchestrator.py
```

## Integration with Coordination Layer

The P2P orchestrator integrates with the coordination layer via events:

### Events Emitted

| Event               | When                       |
| ------------------- | -------------------------- |
| `LEADER_ELECTED`    | This node becomes leader   |
| `HOST_ONLINE`       | Peer recovers from offline |
| `HOST_OFFLINE`      | Peer goes offline          |
| `SELFPLAY_COMPLETE` | Selfplay batch completed   |
| `TRAINING_COMPLETE` | Training job finished      |

### Event Wiring

```python
# In app/coordination/leadership_coordinator.py
from app.distributed.data_events import DataEventType

router.subscribe(DataEventType.LEADER_ELECTED.value, self._on_leader_elected)
router.subscribe(DataEventType.HOST_ONLINE.value, self._on_host_online)
router.subscribe(DataEventType.HOST_OFFLINE.value, self._on_host_offline)
```

## Optional Protocols

### SWIM Membership (Experimental)

Gossip-based membership with 5s failure detection:

```bash
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=swim  # http | swim | hybrid
```

### Raft Consensus (Experimental)

Replicated work queue with sub-second failover:

```bash
export RINGRIFT_RAFT_ENABLED=true
export RINGRIFT_CONSENSUS_MODE=raft  # bully | raft | hybrid
```

## Performance Tuning

### Recommended Settings

| Setting              | Value | Description               |
| -------------------- | ----- | ------------------------- |
| `HEARTBEAT_INTERVAL` | 15s   | Peer heartbeat frequency  |
| `PEER_TIMEOUT`       | 60s   | Peer failure threshold    |
| `GOSSIP_INTERVAL`    | 30s   | State sync frequency      |
| `WORK_POLL_INTERVAL` | 5s    | Work queue poll frequency |

### Scaling

- **Small cluster (<10 nodes)**: Default settings work well
- **Medium cluster (10-50 nodes)**: Increase gossip interval to 60s
- **Large cluster (50+ nodes)**: Consider SWIM for O(1) membership

## Related Documentation

- `scripts/p2p/README.md` - Detailed module documentation
- `scripts/p2p/handlers/README.md` - Handler documentation
- `scripts/p2p/managers/README.md` - Manager documentation
- `scripts/p2p/loops/README.md` - Background loop documentation
- `app/p2p/README.md` - P2P adapter (SWIM/Raft) documentation
- `config/distributed_hosts.template.yaml` - Configuration template
