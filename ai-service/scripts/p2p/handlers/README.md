# P2P Handler Architecture

This directory contains HTTP handler mixins for the P2P orchestrator. Each mixin provides
a set of related endpoints that are composed into `P2POrchestrator` via multiple inheritance.

## Architecture Overview

```
P2POrchestrator (scripts/p2p_orchestrator.py)
    |
    +-- Core Protocol Mixins (scripts/p2p/*.py)
    |   +-- MembershipMixin      # SWIM membership integration
    |   +-- ConsensusMixin       # Raft consensus integration
    |   +-- GossipProtocolMixin  # Gossip state propagation
    |   +-- LeaderElectionMixin  # Bully algorithm election
    |
    +-- Handler Mixins (scripts/p2p/handlers/*.py)
        +-- SwimHandlersMixin       # SWIM status endpoints
        +-- RaftHandlersMixin       # Raft consensus endpoints
        +-- GossipHandlersMixin     # Gossip exchange endpoints
        +-- ElectionHandlersMixin   # Leader election endpoints
        +-- WorkQueueHandlersMixin  # Distributed work queue
        +-- GauntletHandlersMixin   # Model evaluation
        +-- TournamentHandlersMixin # Tournament execution
        +-- AdminHandlersMixin      # Git/health/admin
        +-- RelayHandlersMixin      # NAT relay for blocked nodes
        +-- CMAESHandlersMixin      # Hyperparameter optimization
        +-- EloSyncHandlersMixin    # Elo rating synchronization
        +-- SSHTournamentHandlersMixin  # SSH-based tournaments
```

## Handler Categories

### Core Cluster (Required)

| Mixin                   | Purpose                                           |
| ----------------------- | ------------------------------------------------- |
| `ElectionHandlersMixin` | Bully algorithm leader election with voter quorum |
| `GossipHandlersMixin`   | Decentralized state sharing via gossip protocol   |
| `RelayHandlersMixin`    | Communication relay for NAT-blocked nodes         |

### Protocol Integration (December 2025)

| Mixin               | Purpose                                   |
| ------------------- | ----------------------------------------- |
| `SwimHandlersMixin` | SWIM membership protocol status endpoints |
| `RaftHandlersMixin` | Raft consensus and distributed locking    |

### Work Distribution

| Mixin                        | Purpose                                      |
| ---------------------------- | -------------------------------------------- |
| `WorkQueueHandlersMixin`     | Centralized work queue (claim/complete/fail) |
| `GauntletHandlersMixin`      | Model vs baseline evaluation                 |
| `TournamentHandlersMixin`    | Round-robin tournament execution             |
| `SSHTournamentHandlersMixin` | SSH-based remote tournament execution        |

### Optimization and Administration

| Mixin                  | Purpose                                        |
| ---------------------- | ---------------------------------------------- |
| `CMAESHandlersMixin`   | Distributed CMA-ES hyperparameter optimization |
| `EloSyncHandlersMixin` | Elo rating synchronization across cluster      |
| `AdminHandlersMixin`   | Git status/update, health checks               |

## SWIM Gossip Protocol Handlers

SWIM (Scalable Weakly-consistent Infection-style Membership) provides fast failure
detection via gossip-based protocol with O(1) bandwidth per node.

### Endpoints

| Method | Path            | Description                                               |
| ------ | --------------- | --------------------------------------------------------- |
| GET    | `/swim/status`  | SWIM configuration and summary statistics                 |
| GET    | `/swim/members` | List of SWIM members with states (alive/suspected/failed) |

### Response: `/swim/status`

```json
{
  "node_id": "my-node",
  "swim_enabled": true,
  "swim_available": true,
  "swim_started": true,
  "membership_mode": "hybrid",
  "config": {
    "bind_port": 7947,
    "failure_timeout": 5.0,
    "suspicion_timeout": 3.0,
    "ping_interval": 1.0
  },
  "summary": { "members": 10, "alive": 8, "suspected": 1, "failed": 1 }
}
```

## Raft Consensus Handlers

Raft provides strongly consistent replicated state machines for work queue and job
assignments with automatic leader failover.

### Endpoints

| Method | Path                | Description                                       |
| ------ | ------------------- | ------------------------------------------------- |
| GET    | `/raft/status`      | Raft consensus status and cluster health          |
| GET    | `/raft/work`        | Work queue statistics (pending/claimed/completed) |
| GET    | `/raft/jobs`        | Job assignment statistics with per-node breakdown |
| POST   | `/raft/lock/{name}` | Acquire distributed lock (requires auth)          |
| DELETE | `/raft/lock/{name}` | Release distributed lock (requires auth)          |

### Response: `/raft/status`

```json
{
  "node_id": "my-node",
  "raft_enabled": true,
  "pysyncobj_available": true,
  "raft_initialized": true,
  "consensus_mode": "hybrid",
  "work_queue": { "is_ready": true, "is_leader": false, "leader_address": "192.168.1.10:4321" },
  "job_assignments": { "is_ready": true, "is_leader": false },
  "cluster_health": "healthy"
}
```

## P2P Orchestrator Integration

Handlers are integrated into `P2POrchestrator` via mixin inheritance:

```python
from scripts.p2p.handlers import (
    SwimHandlersMixin,
    RaftHandlersMixin,
    GossipHandlersMixin,
    ElectionHandlersMixin,
    WorkQueueHandlersMixin,
)

class P2POrchestrator(
    SwimHandlersMixin,
    RaftHandlersMixin,
    GossipHandlersMixin,
    ElectionHandlersMixin,
    WorkQueueHandlersMixin,
):
    def _setup_routes(self, app: web.Application):
        # SWIM routes
        app.router.add_get('/swim/status', self.handle_swim_status)
        app.router.add_get('/swim/members', self.handle_swim_members)

        # Raft routes
        app.router.add_get('/raft/status', self.handle_raft_status)
        app.router.add_get('/raft/work', self.handle_raft_work_queue)
        app.router.add_post('/raft/lock/{name}', self.handle_raft_lock)
        app.router.add_delete('/raft/lock/{name}', self.handle_raft_unlock)
```

## Event Bridge Integration

Handlers emit events to the coordination layer via `p2p_event_bridge.py`:

| Handler                  | Event Function                  | Event Type                                |
| ------------------------ | ------------------------------- | ----------------------------------------- |
| `WorkQueueHandlersMixin` | `emit_p2p_work_completed()`     | `SELFPLAY_COMPLETE`, `TRAINING_COMPLETED` |
| `GauntletHandlersMixin`  | `emit_p2p_gauntlet_completed()` | `EVALUATION_COMPLETED`                    |
| `GossipHandlersMixin`    | `emit_p2p_node_online()`        | `HOST_ONLINE`                             |
| `ElectionHandlersMixin`  | `emit_p2p_leader_changed()`     | `LEADER_CHANGED`                          |
| `EloSyncHandlersMixin`   | `emit_p2p_elo_updated()`        | `ELO_UPDATED`                             |

## Adding a New Handler

1. **Create the mixin file** in `scripts/p2p/handlers/`:

```python
"""My Feature HTTP Handlers Mixin.

Provides HTTP endpoints for my feature.

Usage:
    class P2POrchestrator(MyFeatureHandlersMixin, ...):
        pass

Endpoints:
    GET /myfeature/status - Get feature status
    POST /myfeature/action - Perform action
"""

from __future__ import annotations
import logging
from aiohttp import web

logger = logging.getLogger(__name__)

class MyFeatureHandlersMixin:
    """Mixin providing my feature HTTP handlers.

    Requires the implementing class to have:
    - node_id: str
    - auth_token: str | None
    - _is_request_authorized(request) method
    """

    node_id: str
    auth_token: str | None

    async def handle_myfeature_status(self, request: web.Request) -> web.Response:
        """GET /myfeature/status - Get feature status."""
        return web.json_response({"node_id": self.node_id, "status": "ok"})

    async def handle_myfeature_action(self, request: web.Request) -> web.Response:
        """POST /myfeature/action - Perform action (requires auth)."""
        if self.auth_token and not self._is_request_authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        # ... implementation
        return web.json_response({"status": "done"})
```

2. **Export from `__init__.py`**:

```python
from .myfeature import MyFeatureHandlersMixin

__all__ = [
    # ... existing exports
    "MyFeatureHandlersMixin",
]
```

3. **Add to P2POrchestrator inheritance** in `scripts/p2p_orchestrator.py`

4. **Register routes** in `_setup_routes()`:

```python
app.router.add_get('/myfeature/status', self.handle_myfeature_status)
app.router.add_post('/myfeature/action', self.handle_myfeature_action)
```

## Handler Requirements

Each mixin documents required attributes/methods from `P2POrchestrator`:

- `node_id: str` - This node's identifier
- `auth_token: str | None` - Optional auth token for protected endpoints
- `_is_request_authorized(request)` - Auth check method
- Other mixin-specific requirements (see handler docstrings)

## See Also

- `scripts/p2p/membership_mixin.py` - SWIM protocol integration
- `scripts/p2p/consensus_mixin.py` - Raft consensus integration
- `app/p2p/swim_adapter.py` - SWIM protocol adapter
- `app/p2p/raft_state.py` - Raft replicated state machines
- `scripts/p2p/p2p_event_bridge.py` - Event emission helpers
