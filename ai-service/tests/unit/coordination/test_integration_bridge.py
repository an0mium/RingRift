"""Tests for integration bridge wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from app.coordination import integration_bridge as bridge
from app.coordination.event_router import RouterEvent


class FakeLifecycleManager:
    """Captures callbacks registered by the integration bridge."""

    def __init__(self) -> None:
        self.callbacks: dict[str, Any] = {}

    def register_callback(self, event: str, callback: Any) -> None:
        self.callbacks[event] = callback


@dataclass
class FakeTraining:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return {"ok": True}


class FakeP2PManager:
    def __init__(self) -> None:
        self.callbacks: dict[str, Any] = {}
        self.sync_calls: list[tuple[str, Path]] = []
        self.triggered: int = 0
        self.training = FakeTraining()

    def register_callback(self, event: str, callback: Any) -> None:
        self.callbacks[event] = callback

    async def sync_model_to_cluster(self, model_id: str, model_path: Path) -> dict[str, Any]:
        self.sync_calls.append((model_id, model_path))
        return {"ok": True}

    async def trigger_training(self) -> dict[str, Any]:
        self.triggered += 1
        return {"ok": True}


def test_model_lifecycle_callbacks_publish_kwargs(monkeypatch) -> None:
    publish = MagicMock()
    monkeypatch.setattr(bridge, "publish_sync", publish)
    monkeypatch.setattr(bridge, "subscribe", lambda *_args, **_kwargs: None)

    manager = FakeLifecycleManager()
    bridge.wire_model_lifecycle_events(manager)

    assert "model_promoted" in manager.callbacks
    manager.callbacks["model_promoted"](model_id="m1", version=2, stage="production")

    publish.assert_called_with(
        bridge.EVENT_MODEL_PROMOTED,
        {"model_id": "m1", "version": 2, "stage": "production"},
        source="model_lifecycle",
    )


def test_p2p_model_promotion_sync(monkeypatch) -> None:
    subscribed: dict[str, Any] = {}

    def capture_subscribe(event_type, callback):
        subscribed[event_type] = callback

    monkeypatch.setattr(bridge, "publish_sync", MagicMock())
    monkeypatch.setattr(bridge, "subscribe", capture_subscribe)

    manager = FakeP2PManager()
    bridge.wire_p2p_integration_events(manager)

    callback = subscribed[bridge.EVENT_MODEL_PROMOTED]
    event = RouterEvent(
        event_type=bridge.EVENT_MODEL_PROMOTED,
        payload={
            "model_id": "ringrift_v1",
            "model_path": "/tmp/ringrift_v1.pth",
            "promotion_type": "production",
        },
        source="test",
    )
    callback(event)

    assert manager.sync_calls == [("ringrift_v1", Path("/tmp/ringrift_v1.pth"))]


def test_p2p_training_trigger_uses_payload(monkeypatch) -> None:
    subscribed: dict[str, Any] = {}

    def capture_subscribe(event_type, callback):
        subscribed[event_type] = callback

    monkeypatch.setattr(bridge, "publish_sync", MagicMock())
    monkeypatch.setattr(bridge, "subscribe", capture_subscribe)

    manager = FakeP2PManager()
    bridge.wire_p2p_integration_events(manager)

    callback = subscribed[bridge.EVENT_TRAINING_TRIGGERED]
    event = RouterEvent(
        event_type=bridge.EVENT_TRAINING_TRIGGERED,
        payload={"board_type": "square8", "num_players": 2},
        source="test",
    )
    callback(event)

    assert manager.training.calls == [{"board_type": "square8", "num_players": 2}]
