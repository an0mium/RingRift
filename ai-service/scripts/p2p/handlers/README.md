# P2P Handler Mixins

This directory contains modular HTTP handler mixins for the P2P orchestrator. Each mixin provides endpoints for a specific domain.

## Handler Overview

| Handler                      | Purpose                     | Key Endpoints                               | Dependencies                   |
| ---------------------------- | --------------------------- | ------------------------------------------- | ------------------------------ |
| `AdminHandlersMixin`         | Cluster admin               | `/admin/shutdown`, `/admin/health`          | None                           |
| `CMAESHandlersMixin`         | Hyperparameter optimization | `/cmaes/start`, `/cmaes/status`             | None                           |
| `ElectionHandlersMixin`      | Leader election             | `/election/vote`, `/election/heartbeat`     | `peers_lock`, `voter_node_ids` |
| `EloSyncHandlersMixin`       | Elo synchronization         | `/elo/sync`, `/elo/rankings`                | `elo_sync_manager`             |
| `GauntletHandlersMixin`      | Model evaluation            | `/gauntlet/execute`, `/gauntlet/quick-eval` | `ringrift_path`                |
| `GossipHandlersMixin`        | Peer discovery              | `/gossip`, `/gossip/anti-entropy`           | `_gossip_peer_states`          |
| `RaftHandlersMixin`          | Raft consensus              | `/raft/append`, `/raft/vote`                | Raft state machine             |
| `RelayHandlersMixin`         | NAT traversal               | `/relay/forward`, `/relay/register`         | None                           |
| `SSHTournamentHandlersMixin` | SSH tournaments             | `/ssh_tournament/run`                       | SSH config                     |
| `SwimHandlersMixin`          | SWIM membership             | `/swim/status`, `/swim/members`             | `_swim_manager`                |
| `TournamentHandlersMixin`    | Model tournaments           | `/tournament/start`, `/tournament/bracket`  | None                           |
| `WorkQueueHandlersMixin`     | Distributed work            | `/work/claim`, `/work/complete`             | Work queue singleton           |

## Event Bridge Integration

Handlers emit events to the coordination layer via `p2p_event_bridge.py`:

```python
from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed

# In work_queue handler after work completes
await emit_p2p_work_completed(
    work_id=work_id,
    work_type="selfplay",
    config_key="hex8_2p",
    result=result,
    node_id=self.node_id,
)
```

### Events Emitted by Handler

| Handler                  | Event Function                  | Event Type                                |
| ------------------------ | ------------------------------- | ----------------------------------------- |
| `WorkQueueHandlersMixin` | `emit_p2p_work_completed()`     | `SELFPLAY_COMPLETE`, `TRAINING_COMPLETED` |
| `GauntletHandlersMixin`  | `emit_p2p_gauntlet_completed()` | `EVALUATION_COMPLETED`                    |
| `GossipHandlersMixin`    | `emit_p2p_node_online()`        | `HOST_ONLINE`                             |
| `ElectionHandlersMixin`  | `emit_p2p_leader_changed()`     | `LEADER_CHANGED`                          |
| `EloSyncHandlersMixin`   | `emit_p2p_elo_updated()`        | `ELO_UPDATED`                             |

## Adding a New Handler

1. Create `your_handler.py`:

```python
class YourHandlersMixin:
    """Mixin for your feature.

    Requires implementing class to have:
    - node_id: str
    - auth_token: str | None
    """

    async def handle_your_endpoint(self, request):
        # Implementation
        pass
```

2. Add to `__init__.py`:

```python
from .your_handler import YourHandlersMixin
__all__.append("YourHandlersMixin")
```

3. Add to P2POrchestrator inheritance chain in `p2p_orchestrator.py`

4. Register routes in orchestrator's `_setup_routes()` method

## See Also

- `docs/P2P_HANDLERS.md` - Full handler documentation
- `docs/EVENT_CATALOG.md` - Event types reference
- `p2p_event_bridge.py` - Event emission helpers
