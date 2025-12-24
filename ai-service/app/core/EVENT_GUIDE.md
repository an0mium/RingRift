# Event System Guide

**Created**: December 2025
**Purpose**: Document the unified event architecture

## Overview

The RingRift AI service uses a three-tier event system with centralized mappings and a bridge coordinator. This architecture enables both in-process and cross-process event communication.

## Event Buses

### 1. Core EventBus (`app/core/event_bus.py`)

Primary in-memory pub/sub event bus with async support:

```python
from app.core.event_bus import get_event_bus, Event, subscribe

bus = get_event_bus()

# Subscribe with decorator
@bus.subscribe("training.completed")
async def on_training_completed(event: Event):
    print(f"Training done: {event.metadata}")

# Subscribe with pattern
@bus.subscribe(EventFilter(topic_pattern="training.*"))
def on_any_training_event(event):
    pass

# Publish
await bus.publish(Event(
    topic="training.completed",
    metadata={"loss": 0.01},
))
```

**Features**:

- Topic-based pub/sub
- Async and sync handlers
- Pattern matching (`training.*`)
- Event history and replay
- Priority-based handler ordering
- Weak references for auto-cleanup

### 2. StageEventBus (`app/coordination/stage_events.py`)

Pipeline stage completion events:

```python
from app.coordination.stage_events import get_event_bus, StageEvent

bus = get_event_bus()

# Emit stage completion
bus.emit(StageEvent(
    stage="training_complete",
    success=True,
    metrics={"accuracy": 0.85},
))
```

**Purpose**: Track pipeline stage transitions (selfplay → sync → training → evaluation → promotion)

### 3. CrossProcessEventQueue (`app/coordination/cross_process_events.py`)

SQLite-backed inter-process communication:

```python
from app.coordination.cross_process_events import CrossProcessEventQueue

queue = CrossProcessEventQueue(db_path="data/events.db")

# Send event to other processes
queue.send("TRAINING_COMPLETED", {"config": "hex8_2p"})

# Receive events
events = queue.receive(event_types=["TRAINING_COMPLETED"])
```

**Purpose**: Cross-process/cross-machine event propagation

## Event Mappings

Centralized in `app/coordination/event_mappings.py`:

```python
from app.coordination.event_mappings import (
    STAGE_TO_DATA_EVENT_MAP,      # Stage → Data event
    DATA_TO_CROSS_PROCESS_MAP,    # Data → CrossProcess
    STAGE_TO_CROSS_PROCESS_MAP,   # Stage → CrossProcess (direct)
    get_cross_process_event_type, # Helper function
)

# Convert stage event to data event
data_type = get_data_event_type("training_complete")  # → "training_completed"

# Convert data event to cross-process
cp_type = get_cross_process_event_type("training_completed")  # → "TRAINING_COMPLETED"
```

### Mapping Categories

| Source Event          | Target Event           | Example              |
| --------------------- | ---------------------- | -------------------- |
| `selfplay_complete`   | `new_games`            | Stage → Data         |
| `training_completed`  | `TRAINING_COMPLETED`   | Data → CrossProcess  |
| `evaluation_complete` | `EVALUATION_COMPLETED` | Stage → CrossProcess |

## UnifiedEventCoordinator

Bridges events between all three buses:

```python
from app.coordination.unified_event_coordinator import get_event_coordinator

coordinator = get_event_coordinator()
await coordinator.start()

# Events automatically flow between buses:
# StageEventBus → DataEventBus → CrossProcessEventQueue
```

**Automatic Forwarding**:

- Stage events → Data events (in-memory)
- Data events → CrossProcess events (SQLite)
- CrossProcess events → Data events (incoming)

## Event Types

### Training Events

| Event                 | Description          | Payload                         |
| --------------------- | -------------------- | ------------------------------- |
| `training.started`    | Training job started | config_key, model_version       |
| `training.completed`  | Training finished    | config_key, metrics, model_path |
| `training.failed`     | Training error       | config_key, error               |
| `training.checkpoint` | Checkpoint saved     | step, path                      |

### Evaluation Events

| Event                   | Description         | Payload                    |
| ----------------------- | ------------------- | -------------------------- |
| `evaluation.completed`  | Eval finished       | config_key, elo, win_rate  |
| `evaluation.regression` | Regression detected | config_key, elo_drop       |
| `elo.updated`           | Elo changed         | model_id, old_elo, new_elo |

### Data Events

| Event                  | Description         | Payload               |
| ---------------------- | ------------------- | --------------------- |
| `data.new_games`       | New games available | count, source         |
| `data.sync_completed`  | Sync finished       | node_id, games_synced |
| `data.quality_updated` | Quality changed     | score, distribution   |

### Cluster Events

| Event                    | Description  | Payload               |
| ------------------------ | ------------ | --------------------- |
| `cluster.node_joined`    | Node online  | node_id, capabilities |
| `cluster.node_left`      | Node offline | node_id, reason       |
| `cluster.leader_elected` | New leader   | leader_id             |

## Usage Patterns

### Subscribe to Multiple Topics

```python
from app.core.event_bus import get_event_bus, EventFilter

bus = get_event_bus()

# Pattern matching
@bus.subscribe(EventFilter(topic_pattern="training\\..*"))
async def handle_training_events(event):
    if event.topic == "training.completed":
        await notify_promotion_controller(event)
```

### Cross-Process Communication

```python
from app.coordination.cross_process_events import CrossProcessEventQueue

# Process A: Send
queue = CrossProcessEventQueue("data/events.db")
queue.send("MODEL_PROMOTED", {"model_id": "v3", "elo": 1850})

# Process B: Receive
events = queue.receive(event_types=["MODEL_PROMOTED"])
for event in events:
    sync_model_to_cluster(event.payload["model_id"])
```

### Event History

```python
bus = get_event_bus()

# Get recent training events
history = bus.get_history(topic="training.completed", limit=10)

# Replay events to a new handler
await bus.replay(
    handler=new_handler,
    topic="training.completed",
    since=time.time() - 3600,  # Last hour
)
```

## Event Naming Conventions

| Bus           | Convention        | Example              |
| ------------- | ----------------- | -------------------- |
| EventBus      | `domain.action`   | `training.completed` |
| StageEventBus | `snake_case`      | `training_complete`  |
| CrossProcess  | `UPPERCASE_SNAKE` | `TRAINING_COMPLETED` |

## Best Practices

1. **Use centralized mappings** - Don't hardcode event type translations
2. **Prefer async handlers** - Use `async def` for event handlers
3. **Include correlation IDs** - For tracing related events
4. **Use weak references** - For handlers in transient objects
5. **Check event history** - Before subscribing, replay relevant history

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    UnifiedEventCoordinator                       │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐ │
│  │  StageEvent  │ ──► │  DataEvent   │ ──► │  CrossProcess    │ │
│  │     Bus      │     │     Bus      │     │     Queue        │ │
│  │  (pipeline)  │     │  (in-memory) │     │  (SQLite IPC)    │ │
│  └──────────────┘     └──────────────┘     └──────────────────┘ │
│                                                                  │
│                    event_mappings.py                             │
│                    (centralized translations)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Other Processes/Nodes     │
              │  (via SQLite or P2P gossip)   │
              └───────────────────────────────┘
```

## Related Files

| File                                            | Purpose                      |
| ----------------------------------------------- | ---------------------------- |
| `app/core/event_bus.py`                         | Core EventBus implementation |
| `app/coordination/stage_events.py`              | StageEventBus                |
| `app/coordination/cross_process_events.py`      | CrossProcessEventQueue       |
| `app/coordination/unified_event_coordinator.py` | Bridge coordinator           |
| `app/coordination/event_mappings.py`            | Centralized mappings         |
| `app/coordination/event_emitters.py`            | Helper emitters              |
| `app/coordination/event_router.py`              | Event routing logic          |
