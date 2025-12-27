from __future__ import annotations

from app.coordination.event_router import DataEventType
from app.integration.unified_loop_extensions import UnifiedLoopExtensions


class DummyEventBus:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[DataEventType, object]] = []

    def subscribe(self, event_type: DataEventType, handler: object) -> None:
        self.subscriptions.append((event_type, handler))


class DummyLoop:
    def __init__(self) -> None:
        self.event_bus = DummyEventBus()


def test_event_subscriptions_created() -> None:
    loop = DummyLoop()
    extensions = UnifiedLoopExtensions(loop)

    subscribed_types = {event_type for event_type, _ in loop.event_bus.subscriptions}

    assert DataEventType.TRAINING_COMPLETED in subscribed_types
    assert DataEventType.EVALUATION_COMPLETED in subscribed_types
    assert DataEventType.MODEL_PROMOTED in subscribed_types
    assert extensions.state.events_wired is True
    assert extensions.state.events_error is None
