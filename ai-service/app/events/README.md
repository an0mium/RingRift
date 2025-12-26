# Event Types Module

This module defines the event type taxonomy for the RingRift training system.

## Files

| File       | Description                                    |
| ---------- | ---------------------------------------------- |
| `types.py` | `RingRiftEventType` enum with 140+ event types |

## Event Categories

Events are organized into categories:

- **Training**: `TRAINING_STARTED`, `TRAINING_COMPLETED`, `TRAINING_FAILED`
- **Selfplay**: `SELFPLAY_BATCH_COMPLETE`, `GAMES_SAVED`
- **Evaluation**: `EVALUATION_COMPLETED`, `GAUNTLET_COMPLETE`
- **Promotion**: `MODEL_PROMOTED`, `PROMOTION_FAILED`
- **Regression**: `REGRESSION_DETECTED`, `REGRESSION_CRITICAL`
- **Quality**: `QUALITY_SCORE_UPDATED`, `TRAINING_BLOCKED_BY_QUALITY`
- **Sync**: `DATA_SYNC_COMPLETE`, `MODEL_SYNC_COMPLETE`
- **Cluster**: `HOST_ONLINE`, `HOST_OFFLINE`, `NODE_RECOVERED`

## Usage

```python
from app.events.types import RingRiftEventType, EventCategory

# Check event category
event_type = RingRiftEventType.TRAINING_COMPLETED
category = EVENT_CATEGORIES.get(event_type)  # EventCategory.TRAINING
```

## Relationship to DataEventType

`RingRiftEventType` in this module is a superset of `DataEventType` in
`app/distributed/data_events.py`. The latter is used for cross-process
communication while this module provides the complete taxonomy.

## See Also

- `app/distributed/data_events.py` - Cross-process event emission
- `app/coordination/event_router.py` - Unified event routing
