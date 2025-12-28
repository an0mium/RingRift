# ADR-012: P2P Orchestrator Handler Extraction

**Status**: Proposed
**Date**: 2025-12-28
**Context**: RingRift AI Training Infrastructure

## Summary

This ADR proposes extracting the ~20 event handlers from `p2p_orchestrator.py` (28,276 lines) into a dedicated `scripts/p2p/handlers/event_handlers.py` module.

## Problem Statement

The P2P orchestrator has grown to 28,276 lines, making it difficult to:

- Navigate the codebase
- Test individual handlers
- Understand the event flow
- Make isolated changes

### Current Handler Locations (Lines 3910-4300)

| Handler                       | Purpose                     | Lines |
| ----------------------------- | --------------------------- | ----- |
| `handle_daemon_status`        | Track daemon lifecycle      | ~15   |
| `handle_quality_degraded`     | Trigger selfplay adjustment | ~15   |
| `handle_elo_velocity_changed` | Adjust training rate        | ~15   |
| `handle_evaluation_completed` | Post-eval actions           | ~15   |
| `handle_plateau_detected`     | Trigger exploration boost   | ~15   |
| `handle_exploration_boost`    | Apply exploration config    | ~25   |
| `handle_promotion_failed`     | Track failed promotions     | ~30   |
| `handle_handler_failed`       | Log handler errors          | ~15   |
| `handle_training_started`     | Pause idle detection        | ~15   |
| `handle_training_completed`   | Trigger eval pipeline       | ~20   |
| `handle_task_spawned`         | Track job start             | ~10   |
| `handle_task_completed`       | Update job status           | ~10   |
| `handle_task_failed`          | Handle job failure          | ~10   |
| `handle_data_sync_started`    | Track sync start            | ~10   |
| `handle_data_sync_completed`  | Trigger post-sync actions   | ~15   |
| `handle_node_unhealthy`       | Reschedule jobs             | ~15   |
| `handle_node_recovered`       | Resume jobs                 | ~15   |
| `handle_cluster_healthy`      | Clear alerts                | ~10   |
| `handle_cluster_unhealthy`    | Pause non-critical work     | ~10   |

**Total: ~20 handlers, ~290 lines**

## Decision

Extract event handlers to `scripts/p2p/handlers/event_handlers.py`:

```python
# scripts/p2p/handlers/event_handlers.py

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

class EventHandlerMixin:
    """Mixin providing event handlers for P2POrchestrator."""

    def _register_event_handlers(self: "P2POrchestrator") -> None:
        """Register all event handlers."""
        handlers = [
            ("DAEMON_STATUS_CHANGED", self._handle_daemon_status),
            ("QUALITY_DEGRADED", self._handle_quality_degraded),
            ("ELO_VELOCITY_CHANGED", self._handle_elo_velocity_changed),
            ("EVALUATION_COMPLETED", self._handle_evaluation_completed),
            ("PLATEAU_DETECTED", self._handle_plateau_detected),
            ("EXPLORATION_BOOST", self._handle_exploration_boost),
            ("PROMOTION_FAILED", self._handle_promotion_failed),
            ("HANDLER_FAILED", self._handle_handler_failed),
            ("TRAINING_STARTED", self._handle_training_started),
            ("TRAINING_COMPLETED", self._handle_training_completed),
            ("TASK_SPAWNED", self._handle_task_spawned),
            ("TASK_COMPLETED", self._handle_task_completed),
            ("TASK_FAILED", self._handle_task_failed),
            ("DATA_SYNC_STARTED", self._handle_data_sync_started),
            ("DATA_SYNC_COMPLETED", self._handle_data_sync_completed),
            ("NODE_UNHEALTHY", self._handle_node_unhealthy),
            ("NODE_RECOVERED", self._handle_node_recovered),
            ("CLUSTER_HEALTHY", self._handle_cluster_healthy),
            ("CLUSTER_UNHEALTHY", self._handle_cluster_unhealthy),
        ]
        for event_type, handler in handlers:
            self._event_bus.subscribe(event_type, handler)

    def _handle_daemon_status(self: "P2POrchestrator", event: dict) -> None:
        """Handle daemon status changes."""
        daemon_type = event.get("daemon_type")
        status = event.get("status")
        self.logger.debug(f"Daemon {daemon_type} -> {status}")

    # ... rest of handlers ...
```

### Updated P2POrchestrator

```python
# scripts/p2p_orchestrator.py

from scripts.p2p.handlers.event_handlers import EventHandlerMixin

class P2POrchestrator(
    EventHandlerMixin,  # NEW
    P2PMixinBase,
    LeaderElectionMixin,
    PeerManagerMixin,
    GossipProtocolMixin,
    ConsensusMixin,
    MembershipMixin,
):
    def __init__(self):
        # ...
        self._register_event_handlers()  # NEW
```

## Migration Plan

### Phase 1: Extract (No Behavior Change)

1. Create `scripts/p2p/handlers/event_handlers.py`
2. Copy all `handle_*` functions
3. Convert to `_handle_*` methods on mixin
4. Update P2POrchestrator to inherit mixin
5. Run tests to verify no regressions

### Phase 2: Add Tests

1. Create `tests/unit/p2p/test_event_handlers.py`
2. Test each handler in isolation
3. Mock orchestrator dependencies

### Phase 3: Documentation

1. Document event → handler mapping
2. Add handler contract documentation
3. Update CLAUDE.md with handler reference

## Benefits

1. **Testability**: Handlers can be unit tested without full orchestrator
2. **Maintainability**: 28K → ~27.7K lines in main file
3. **Discoverability**: All event handlers in one place
4. **Reusability**: Handlers can be shared with other orchestrator implementations

## Risks

1. **Circular imports**: Mixin needs TYPE_CHECKING to reference P2POrchestrator
2. **State access**: Handlers access orchestrator state via `self`
3. **Testing complexity**: Need to mock orchestrator in handler tests

## References

- `scripts/p2p_orchestrator.py:3910-4300` - Current handler location
- `scripts/p2p/p2p_mixin_base.py` - Existing mixin pattern
- `scripts/p2p/handlers/` - Existing handler directory structure
