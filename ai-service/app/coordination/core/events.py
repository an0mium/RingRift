"""Unified event system (December 2025).

Consolidates event-related functionality from event_router.py and stage_events.py.

Usage:
    from app.coordination.core.events import (
        UnifiedEventRouter,
        get_router,
        publish,
        subscribe,
    )
"""

from __future__ import annotations

# Re-export from event_router
from app.coordination.event_router import (
    UnifiedEventRouter,
    get_router,
    publish,
    publish_sync,
    subscribe,
    unsubscribe,
    DataEventType,
    DataEvent,
    StageEvent,
    get_event_bus,
    emit_training_completed,
    emit_training_started,
    emit_selfplay_batch_completed,
    emit_model_promoted,
)

# Re-export from stage_events
from app.coordination.stage_events import (
    StageEvent as StageEventModel,
    StageEventBus,
    StageCompletionResult,
    get_event_bus as get_stage_event_bus,
)

__all__ = [
    # From event_router
    "UnifiedEventRouter",
    "get_router",
    "publish",
    "publish_sync",
    "subscribe",
    "unsubscribe",
    "DataEventType",
    "DataEvent",
    "StageEvent",
    "get_event_bus",
    "emit_training_completed",
    "emit_training_started",
    "emit_selfplay_batch_completed",
    "emit_model_promoted",
    # From stage_events
    "StageEventModel",
    "StageEventBus",
    "StageCompletionResult",
    "get_stage_event_bus",
]
