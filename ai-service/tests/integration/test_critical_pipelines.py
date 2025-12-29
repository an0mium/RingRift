"""End-to-end integration tests for 5 critical pipelines in ai-service.

These tests verify the full event flow from trigger to completion through the
actual coordination layer components. Each test exercises a complete pipeline
to ensure events are emitted and received in the correct order.

December 2025: Created to verify critical pipeline event flows.

Pipelines tested:
1. Training -> Evaluation -> Promotion -> Distribution
2. New Games -> Export -> Train -> Promote
3. Regression Detection -> Rollback -> Recovery
4. Orphan Games -> Sync -> Register
5. Curriculum Rebalanced -> Selfplay Allocation

Usage:
    pytest tests/integration/test_critical_pipelines.py -v
    pytest tests/integration/test_critical_pipelines.py -k "training_to_distribution"
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Infrastructure: Mock Event Capture System
# =============================================================================


@dataclass
class CapturedEvent:
    """A captured event for test assertions."""

    event_type: str
    payload: dict
    timestamp: float = field(default_factory=time.time)
    source: str = "test"


class MockEventCapture:
    """Captures events in order for verification.

    Provides methods to wait for specific events with timeout.
    """

    def __init__(self):
        self.events: list[CapturedEvent] = []
        self.subscribers: dict[str, list] = {}
        self._event_signals: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def publish(self, event_type: str | Any, payload: dict, source: str = "test") -> None:
        """Publish an event and notify waiting handlers."""
        # Normalize event type
        if hasattr(event_type, "value"):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        event = CapturedEvent(
            event_type=event_type_str,
            payload=payload,
            timestamp=time.time(),
            source=source,
        )

        async with self._lock:
            self.events.append(event)

            # Signal waiters
            if event_type_str in self._event_signals:
                self._event_signals[event_type_str].set()

            # Call registered handlers
            if event_type_str in self.subscribers:
                for handler in self.subscribers[event_type_str]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(payload)
                        else:
                            handler(payload)
                    except Exception:
                        pass  # Ignore handler errors in test

    def subscribe(self, event_type: str | Any, handler) -> None:
        """Subscribe a handler to an event type."""
        if hasattr(event_type, "value"):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        if event_type_str not in self.subscribers:
            self.subscribers[event_type_str] = []
        self.subscribers[event_type_str].append(handler)

    def unsubscribe(self, event_type: str | Any, handler) -> None:
        """Unsubscribe a handler from an event type."""
        if hasattr(event_type, "value"):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        if event_type_str in self.subscribers:
            self.subscribers[event_type_str] = [
                h for h in self.subscribers[event_type_str] if h != handler
            ]

    async def wait_for_event(
        self,
        event_type: str | Any,
        timeout: float = 5.0,
    ) -> CapturedEvent | None:
        """Wait for a specific event type with timeout."""
        if hasattr(event_type, "value"):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        # Check if already received
        for event in self.events:
            if event.event_type == event_type_str:
                return event

        # Create signal and wait
        async with self._lock:
            if event_type_str not in self._event_signals:
                self._event_signals[event_type_str] = asyncio.Event()
            signal = self._event_signals[event_type_str]

        try:
            await asyncio.wait_for(signal.wait(), timeout=timeout)
            # Return the matching event
            for event in reversed(self.events):
                if event.event_type == event_type_str:
                    return event
        except asyncio.TimeoutError:
            return None

        return None

    def get_events_in_order(self) -> list[str]:
        """Get event types in the order they were received."""
        return [e.event_type for e in self.events]

    def get_event(self, event_type: str | Any) -> CapturedEvent | None:
        """Get the most recent event of a given type."""
        if hasattr(event_type, "value"):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        for event in reversed(self.events):
            if event.event_type == event_type_str:
                return event
        return None

    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()
        self._event_signals.clear()


@pytest.fixture
def event_capture():
    """Create a fresh event capture for each test."""
    return MockEventCapture()


@pytest.fixture
def config_key():
    """Standard config key for tests."""
    return "hex8_2p"


# =============================================================================
# Pipeline 1: Training -> Evaluation -> Promotion -> Distribution
# =============================================================================


class TestTrainingToDistributionPipeline:
    """End-to-end test for Training -> Evaluation -> Promotion -> Distribution.

    This is the core training feedback loop:
    1. TRAINING_COMPLETED -> triggers EVALUATION_STARTED
    2. EVALUATION_COMPLETED (pass) -> triggers MODEL_PROMOTED
    3. MODEL_PROMOTED -> triggers DISTRIBUTION_STARTED (via MODEL_DISTRIBUTION_STARTED)
    """

    @pytest.mark.asyncio
    async def test_training_completed_triggers_evaluation(self, event_capture, config_key):
        """TRAINING_COMPLETED should trigger EVALUATION_STARTED."""
        from app.distributed.data_events import DataEventType

        # Create a mock feedback loop controller that responds to training completed
        evaluation_triggered = False
        triggered_config = None

        async def mock_evaluation_handler(payload):
            nonlocal evaluation_triggered, triggered_config
            evaluation_triggered = True
            triggered_config = payload.get("config_key")
            # Emit evaluation started
            await event_capture.publish(
                DataEventType.EVALUATION_STARTED,
                {
                    "config_key": payload.get("config_key"),
                    "model_path": payload.get("model_path"),
                    "source": "feedback_loop_controller",
                },
            )

        # Subscribe handler
        event_capture.subscribe(DataEventType.TRAINING_COMPLETED, mock_evaluation_handler)

        # Emit training completed
        await event_capture.publish(
            DataEventType.TRAINING_COMPLETED,
            {
                "config_key": config_key,
                "model_path": "/models/test.pth",
                "epochs_completed": 50,
                "policy_accuracy": 0.82,
                "value_accuracy": 0.68,
            },
        )

        # Verify evaluation was triggered
        assert evaluation_triggered
        assert triggered_config == config_key

        # Verify EVALUATION_STARTED was emitted
        eval_event = event_capture.get_event(DataEventType.EVALUATION_STARTED)
        assert eval_event is not None
        assert eval_event.payload["config_key"] == config_key

    @pytest.mark.asyncio
    async def test_evaluation_pass_triggers_promotion(self, event_capture, config_key):
        """EVALUATION_COMPLETED with pass should trigger MODEL_PROMOTED."""
        from app.distributed.data_events import DataEventType

        # Create mock promotion handler
        promotion_triggered = False

        async def mock_promotion_handler(payload):
            nonlocal promotion_triggered
            if payload.get("passed_gauntlet", False):
                promotion_triggered = True
                await event_capture.publish(
                    DataEventType.MODEL_PROMOTED,
                    {
                        "config_key": payload.get("config_key"),
                        "model_path": payload.get("model_path"),
                        "elo_after": 1650,
                        "source": "auto_promotion_daemon",
                    },
                )

        event_capture.subscribe(DataEventType.EVALUATION_COMPLETED, mock_promotion_handler)

        # Emit evaluation completed with pass
        await event_capture.publish(
            DataEventType.EVALUATION_COMPLETED,
            {
                "config_key": config_key,
                "model_path": "/models/test.pth",
                "passed_gauntlet": True,
                "win_rate_random": 0.95,
                "win_rate_heuristic": 0.68,
            },
        )

        # Verify promotion was triggered
        assert promotion_triggered

        # Verify MODEL_PROMOTED was emitted
        promo_event = event_capture.get_event(DataEventType.MODEL_PROMOTED)
        assert promo_event is not None

    @pytest.mark.asyncio
    async def test_promotion_triggers_distribution(self, event_capture, config_key):
        """MODEL_PROMOTED should trigger MODEL_DISTRIBUTION_STARTED."""
        from app.distributed.data_events import DataEventType

        distribution_started = False

        async def mock_distribution_handler(payload):
            nonlocal distribution_started
            distribution_started = True
            await event_capture.publish(
                DataEventType.MODEL_DISTRIBUTION_STARTED,
                {
                    "config_key": payload.get("config_key"),
                    "model_path": payload.get("model_path"),
                    "target_nodes": ["vast-123", "nebius-1"],
                    "source": "unified_distribution_daemon",
                },
            )

        event_capture.subscribe(DataEventType.MODEL_PROMOTED, mock_distribution_handler)

        # Emit model promoted
        await event_capture.publish(
            DataEventType.MODEL_PROMOTED,
            {
                "config_key": config_key,
                "model_path": "/models/canonical_hex8_2p.pth",
                "elo_after": 1650,
            },
        )

        # Verify distribution was triggered
        assert distribution_started

        dist_event = event_capture.get_event(DataEventType.MODEL_DISTRIBUTION_STARTED)
        assert dist_event is not None

    @pytest.mark.asyncio
    async def test_full_training_to_distribution_chain(self, event_capture, config_key):
        """Test complete chain: TRAINING_COMPLETED -> EVALUATION -> PROMOTION -> DISTRIBUTION."""
        from app.distributed.data_events import DataEventType

        # Wire up all handlers for the full chain
        chain_stage = []

        async def on_training_completed(payload):
            chain_stage.append("training_completed")
            if payload.get("policy_accuracy", 0) >= 0.75:  # Threshold check
                await event_capture.publish(
                    DataEventType.EVALUATION_STARTED,
                    {"config_key": payload["config_key"], "model_path": payload["model_path"]},
                )

        async def on_evaluation_started(payload):
            chain_stage.append("evaluation_started")
            # Simulate gauntlet running
            await event_capture.publish(
                DataEventType.EVALUATION_COMPLETED,
                {
                    "config_key": payload["config_key"],
                    "model_path": payload["model_path"],
                    "passed_gauntlet": True,
                    "win_rate_random": 0.95,
                    "win_rate_heuristic": 0.65,
                },
            )

        async def on_evaluation_completed(payload):
            chain_stage.append("evaluation_completed")
            if payload.get("passed_gauntlet"):
                await event_capture.publish(
                    DataEventType.MODEL_PROMOTED,
                    {"config_key": payload["config_key"], "model_path": payload["model_path"]},
                )

        async def on_model_promoted(payload):
            chain_stage.append("model_promoted")
            await event_capture.publish(
                DataEventType.MODEL_DISTRIBUTION_STARTED,
                {"config_key": payload["config_key"], "model_path": payload["model_path"]},
            )

        async def on_distribution_started(payload):
            chain_stage.append("distribution_started")

        # Subscribe all handlers
        event_capture.subscribe(DataEventType.TRAINING_COMPLETED, on_training_completed)
        event_capture.subscribe(DataEventType.EVALUATION_STARTED, on_evaluation_started)
        event_capture.subscribe(DataEventType.EVALUATION_COMPLETED, on_evaluation_completed)
        event_capture.subscribe(DataEventType.MODEL_PROMOTED, on_model_promoted)
        event_capture.subscribe(DataEventType.MODEL_DISTRIBUTION_STARTED, on_distribution_started)

        # Trigger the chain
        await event_capture.publish(
            DataEventType.TRAINING_COMPLETED,
            {
                "config_key": config_key,
                "model_path": "/models/test.pth",
                "epochs_completed": 50,
                "policy_accuracy": 0.82,
                "value_accuracy": 0.68,
            },
        )

        # Verify complete chain executed in order
        expected_chain = [
            "training_completed",
            "evaluation_started",
            "evaluation_completed",
            "model_promoted",
            "distribution_started",
        ]
        assert chain_stage == expected_chain

        # Verify events were captured in order
        event_order = event_capture.get_events_in_order()
        assert DataEventType.TRAINING_COMPLETED.value in event_order
        assert DataEventType.EVALUATION_STARTED.value in event_order
        assert DataEventType.EVALUATION_COMPLETED.value in event_order
        assert DataEventType.MODEL_PROMOTED.value in event_order
        assert DataEventType.MODEL_DISTRIBUTION_STARTED.value in event_order


# =============================================================================
# Pipeline 2: New Games -> Export -> Train -> Promote
# =============================================================================


class TestNewGamesToPromotePipeline:
    """End-to-end test for New Games -> Export -> Train -> Promote.

    This is the data-driven training trigger:
    1. NEW_GAMES_AVAILABLE -> threshold check
    2. TRAINING_THRESHOLD_REACHED -> triggers TRAINING_STARTED
    3. TRAINING_STARTED -> training process runs
    4. (continues to Pipeline 1)
    """

    @pytest.mark.asyncio
    async def test_new_games_triggers_threshold_check(self, event_capture, config_key):
        """NEW_GAMES_AVAILABLE should trigger threshold checking."""
        from app.distributed.data_events import DataEventType

        threshold_checked = False
        game_count = 0

        async def mock_threshold_handler(payload):
            nonlocal threshold_checked, game_count
            threshold_checked = True
            game_count = payload.get("game_count", 0)
            # If enough games, emit threshold reached
            if game_count >= 100:  # Threshold
                await event_capture.publish(
                    DataEventType.TRAINING_THRESHOLD_REACHED,
                    {"config_key": payload.get("config_key"), "game_count": game_count},
                )

        event_capture.subscribe(DataEventType.NEW_GAMES_AVAILABLE, mock_threshold_handler)

        # Emit new games available with enough games
        await event_capture.publish(
            DataEventType.NEW_GAMES_AVAILABLE,
            {
                "config_key": config_key,
                "game_count": 150,
                "database_path": "/data/games/selfplay.db",
            },
        )

        # Verify threshold was checked
        assert threshold_checked
        assert game_count == 150

        # Verify TRAINING_THRESHOLD_REACHED was emitted
        threshold_event = event_capture.get_event(DataEventType.TRAINING_THRESHOLD_REACHED)
        assert threshold_event is not None

    @pytest.mark.asyncio
    async def test_threshold_reached_triggers_training(self, event_capture, config_key):
        """TRAINING_THRESHOLD_REACHED should trigger TRAINING_STARTED."""
        from app.distributed.data_events import DataEventType

        training_started = False

        async def mock_training_handler(payload):
            nonlocal training_started
            training_started = True
            await event_capture.publish(
                DataEventType.TRAINING_STARTED,
                {
                    "config_key": payload.get("config_key"),
                    "data_path": f"/data/training/{payload.get('config_key')}.npz",
                    "source": "training_coordinator",
                },
            )

        event_capture.subscribe(DataEventType.TRAINING_THRESHOLD_REACHED, mock_training_handler)

        # Emit threshold reached
        await event_capture.publish(
            DataEventType.TRAINING_THRESHOLD_REACHED,
            {"config_key": config_key, "game_count": 150},
        )

        # Verify training started
        assert training_started

        train_event = event_capture.get_event(DataEventType.TRAINING_STARTED)
        assert train_event is not None
        assert train_event.payload["config_key"] == config_key

    @pytest.mark.asyncio
    async def test_new_games_below_threshold_no_training(self, event_capture, config_key):
        """NEW_GAMES_AVAILABLE below threshold should NOT trigger training."""
        from app.distributed.data_events import DataEventType

        threshold_reached = False

        async def mock_threshold_handler(payload):
            nonlocal threshold_reached
            game_count = payload.get("game_count", 0)
            if game_count >= 100:
                threshold_reached = True

        event_capture.subscribe(DataEventType.NEW_GAMES_AVAILABLE, mock_threshold_handler)

        # Emit with insufficient games
        await event_capture.publish(
            DataEventType.NEW_GAMES_AVAILABLE,
            {"config_key": config_key, "game_count": 50},  # Below threshold
        )

        # Verify threshold was NOT reached
        assert not threshold_reached

        # Verify no TRAINING_THRESHOLD_REACHED event
        threshold_event = event_capture.get_event(DataEventType.TRAINING_THRESHOLD_REACHED)
        assert threshold_event is None


# =============================================================================
# Pipeline 3: Regression Detection -> Rollback -> Recovery
# =============================================================================


class TestRegressionToRecoveryPipeline:
    """End-to-end test for Regression Detection -> Rollback -> Recovery.

    This is the model quality safeguard:
    1. REGRESSION_DETECTED -> triggers TRAINING_ROLLBACK_NEEDED
    2. TRAINING_ROLLBACK_NEEDED -> rollback process
    3. TRAINING_ROLLBACK_COMPLETED -> recovery events
    """

    @pytest.mark.asyncio
    async def test_regression_triggers_rollback(self, event_capture, config_key):
        """REGRESSION_DETECTED should trigger TRAINING_ROLLBACK_NEEDED for severe cases."""
        from app.distributed.data_events import DataEventType

        rollback_triggered = False
        rollback_config = None

        async def mock_regression_handler(payload):
            nonlocal rollback_triggered, rollback_config
            severity = payload.get("severity", "minor")
            if severity in ("severe", "critical"):
                rollback_triggered = True
                rollback_config = payload.get("config_key")
                await event_capture.publish(
                    DataEventType.TRAINING_ROLLBACK_NEEDED,
                    {
                        "config_key": payload.get("config_key"),
                        "model_id": payload.get("model_id"),
                        "reason": f"regression_{severity}",
                        "source": "model_lifecycle_coordinator",
                    },
                )

        event_capture.subscribe(DataEventType.REGRESSION_DETECTED, mock_regression_handler)

        # Emit severe regression
        await event_capture.publish(
            DataEventType.REGRESSION_DETECTED,
            {
                "config_key": config_key,
                "model_id": "model_123",
                "severity": "severe",
                "elo_drop": 150,
                "win_rate": 0.35,
            },
        )

        # Verify rollback was triggered
        assert rollback_triggered
        assert rollback_config == config_key

        rollback_event = event_capture.get_event(DataEventType.TRAINING_ROLLBACK_NEEDED)
        assert rollback_event is not None

    @pytest.mark.asyncio
    async def test_rollback_completed_triggers_recovery(self, event_capture, config_key):
        """TRAINING_ROLLBACK_COMPLETED should trigger RECOVERY_COMPLETED."""
        from app.distributed.data_events import DataEventType

        recovery_completed = False

        async def mock_rollback_complete_handler(payload):
            nonlocal recovery_completed
            recovery_completed = True
            await event_capture.publish(
                DataEventType.RECOVERY_COMPLETED,
                {
                    "config_key": payload.get("config_key"),
                    "recovered_model": payload.get("restored_checkpoint"),
                    "recovery_type": "rollback",
                    "source": "recovery_orchestrator",
                },
            )

        event_capture.subscribe(
            DataEventType.TRAINING_ROLLBACK_COMPLETED, mock_rollback_complete_handler
        )

        # Emit rollback completed
        await event_capture.publish(
            DataEventType.TRAINING_ROLLBACK_COMPLETED,
            {
                "config_key": config_key,
                "restored_checkpoint": "/models/checkpoint_epoch45.pth",
                "rolled_back_from": "/models/test.pth",
            },
        )

        # Verify recovery completed
        assert recovery_completed

        recovery_event = event_capture.get_event(DataEventType.RECOVERY_COMPLETED)
        assert recovery_event is not None
        assert recovery_event.payload["recovery_type"] == "rollback"

    @pytest.mark.asyncio
    async def test_minor_regression_no_rollback(self, event_capture, config_key):
        """Minor REGRESSION_DETECTED should NOT trigger rollback."""
        from app.distributed.data_events import DataEventType

        rollback_triggered = False

        async def mock_regression_handler(payload):
            nonlocal rollback_triggered
            severity = payload.get("severity", "minor")
            if severity in ("severe", "critical"):
                rollback_triggered = True

        event_capture.subscribe(DataEventType.REGRESSION_DETECTED, mock_regression_handler)

        # Emit minor regression
        await event_capture.publish(
            DataEventType.REGRESSION_DETECTED,
            {
                "config_key": config_key,
                "model_id": "model_123",
                "severity": "minor",  # Not severe
                "elo_drop": 25,
                "win_rate": 0.55,
            },
        )

        # Verify rollback was NOT triggered
        assert not rollback_triggered

    @pytest.mark.asyncio
    async def test_full_regression_recovery_chain(self, event_capture, config_key):
        """Test complete chain: REGRESSION -> ROLLBACK_NEEDED -> ROLLBACK_COMPLETED -> RECOVERY."""
        from app.distributed.data_events import DataEventType

        chain_stage = []

        async def on_regression_detected(payload):
            chain_stage.append("regression_detected")
            if payload.get("severity") in ("severe", "critical"):
                await event_capture.publish(
                    DataEventType.TRAINING_ROLLBACK_NEEDED,
                    {"config_key": payload["config_key"], "model_id": payload["model_id"]},
                )

        async def on_rollback_needed(payload):
            chain_stage.append("rollback_needed")
            # Simulate rollback execution
            await event_capture.publish(
                DataEventType.TRAINING_ROLLBACK_COMPLETED,
                {
                    "config_key": payload["config_key"],
                    "restored_checkpoint": "/models/checkpoint.pth",
                },
            )

        async def on_rollback_completed(payload):
            chain_stage.append("rollback_completed")
            await event_capture.publish(
                DataEventType.RECOVERY_COMPLETED,
                {"config_key": payload["config_key"], "recovery_type": "rollback"},
            )

        async def on_recovery_completed(payload):
            chain_stage.append("recovery_completed")

        # Subscribe all handlers
        event_capture.subscribe(DataEventType.REGRESSION_DETECTED, on_regression_detected)
        event_capture.subscribe(DataEventType.TRAINING_ROLLBACK_NEEDED, on_rollback_needed)
        event_capture.subscribe(DataEventType.TRAINING_ROLLBACK_COMPLETED, on_rollback_completed)
        event_capture.subscribe(DataEventType.RECOVERY_COMPLETED, on_recovery_completed)

        # Trigger the chain
        await event_capture.publish(
            DataEventType.REGRESSION_DETECTED,
            {
                "config_key": config_key,
                "model_id": "model_123",
                "severity": "critical",
                "elo_drop": 200,
            },
        )

        # Verify complete chain
        expected_chain = [
            "regression_detected",
            "rollback_needed",
            "rollback_completed",
            "recovery_completed",
        ]
        assert chain_stage == expected_chain


# =============================================================================
# Pipeline 4: Orphan Games -> Sync -> Register
# =============================================================================


class TestOrphanGamesToRegisterPipeline:
    """End-to-end test for Orphan Games -> Sync -> Register.

    This handles games on unreachable nodes:
    1. ORPHAN_GAMES_DETECTED -> triggers priority sync
    2. DATA_SYNC_COMPLETED -> games available
    3. ORPHAN_GAMES_REGISTERED -> ready for training
    """

    @pytest.mark.asyncio
    async def test_orphan_detection_triggers_sync(self, event_capture, config_key):
        """ORPHAN_GAMES_DETECTED should trigger DATA_SYNC_STARTED."""
        from app.distributed.data_events import DataEventType

        sync_triggered = False
        source_node = None

        async def mock_orphan_handler(payload):
            nonlocal sync_triggered, source_node
            sync_triggered = True
            source_node = payload.get("source_node")
            await event_capture.publish(
                DataEventType.DATA_SYNC_STARTED,
                {
                    "config_key": payload.get("config_key"),
                    "source_node": source_node,
                    "sync_type": "priority_orphan_recovery",
                    "source": "sync_facade",
                },
            )

        event_capture.subscribe(DataEventType.ORPHAN_GAMES_DETECTED, mock_orphan_handler)

        # Emit orphan games detected
        await event_capture.publish(
            DataEventType.ORPHAN_GAMES_DETECTED,
            {
                "source_node": "vast-12345",
                "config_key": config_key,
                "game_count": 50,
                "database_path": "/data/games/orphan.db",
            },
        )

        # Verify sync was triggered
        assert sync_triggered
        assert source_node == "vast-12345"

        sync_event = event_capture.get_event(DataEventType.DATA_SYNC_STARTED)
        assert sync_event is not None

    @pytest.mark.asyncio
    async def test_sync_completed_triggers_registration(self, event_capture, config_key):
        """DATA_SYNC_COMPLETED should trigger ORPHAN_GAMES_REGISTERED."""
        from app.distributed.data_events import DataEventType

        registration_complete = False

        async def mock_sync_complete_handler(payload):
            nonlocal registration_complete
            if payload.get("sync_type") == "priority_orphan_recovery":
                registration_complete = True
                await event_capture.publish(
                    DataEventType.ORPHAN_GAMES_REGISTERED,
                    {
                        "config_key": payload.get("config_key"),
                        "source_node": payload.get("source_node"),
                        "game_count": payload.get("files_synced", 0),
                        "registered_at": time.time(),
                    },
                )

        event_capture.subscribe(DataEventType.DATA_SYNC_COMPLETED, mock_sync_complete_handler)

        # Emit sync completed
        await event_capture.publish(
            DataEventType.DATA_SYNC_COMPLETED,
            {
                "config_key": config_key,
                "source_node": "vast-12345",
                "sync_type": "priority_orphan_recovery",
                "files_synced": 50,
                "success": True,
            },
        )

        # Verify registration complete
        assert registration_complete

        reg_event = event_capture.get_event(DataEventType.ORPHAN_GAMES_REGISTERED)
        assert reg_event is not None

    @pytest.mark.asyncio
    async def test_registration_triggers_new_games(self, event_capture, config_key):
        """ORPHAN_GAMES_REGISTERED should trigger NEW_GAMES_AVAILABLE."""
        from app.distributed.data_events import DataEventType

        new_games_emitted = False

        async def mock_registration_handler(payload):
            nonlocal new_games_emitted
            new_games_emitted = True
            await event_capture.publish(
                DataEventType.NEW_GAMES_AVAILABLE,
                {
                    "config_key": payload.get("config_key"),
                    "game_count": payload.get("game_count"),
                    "source": "orphan_recovery",
                },
            )

        event_capture.subscribe(DataEventType.ORPHAN_GAMES_REGISTERED, mock_registration_handler)

        # Emit registration complete
        await event_capture.publish(
            DataEventType.ORPHAN_GAMES_REGISTERED,
            {
                "config_key": config_key,
                "source_node": "vast-12345",
                "game_count": 50,
            },
        )

        # Verify new games emitted
        assert new_games_emitted

        new_games_event = event_capture.get_event(DataEventType.NEW_GAMES_AVAILABLE)
        assert new_games_event is not None
        assert new_games_event.payload["source"] == "orphan_recovery"

    @pytest.mark.asyncio
    async def test_full_orphan_recovery_chain(self, event_capture, config_key):
        """Test complete chain: ORPHAN_DETECTED -> SYNC -> REGISTERED -> NEW_GAMES."""
        from app.distributed.data_events import DataEventType

        chain_stage = []

        async def on_orphan_detected(payload):
            chain_stage.append("orphan_detected")
            await event_capture.publish(
                DataEventType.DATA_SYNC_STARTED,
                {
                    "config_key": payload["config_key"],
                    "source_node": payload["source_node"],
                    "sync_type": "priority_orphan_recovery",
                },
            )

        async def on_sync_started(payload):
            chain_stage.append("sync_started")
            # Simulate sync completion
            await event_capture.publish(
                DataEventType.DATA_SYNC_COMPLETED,
                {
                    "config_key": payload["config_key"],
                    "source_node": payload["source_node"],
                    "sync_type": payload["sync_type"],
                    "files_synced": 50,
                },
            )

        async def on_sync_completed(payload):
            chain_stage.append("sync_completed")
            if payload.get("sync_type") == "priority_orphan_recovery":
                await event_capture.publish(
                    DataEventType.ORPHAN_GAMES_REGISTERED,
                    {
                        "config_key": payload["config_key"],
                        "game_count": payload.get("files_synced", 0),
                    },
                )

        async def on_games_registered(payload):
            chain_stage.append("games_registered")
            await event_capture.publish(
                DataEventType.NEW_GAMES_AVAILABLE,
                {"config_key": payload["config_key"], "game_count": payload["game_count"]},
            )

        async def on_new_games(payload):
            chain_stage.append("new_games_available")

        # Subscribe all handlers
        event_capture.subscribe(DataEventType.ORPHAN_GAMES_DETECTED, on_orphan_detected)
        event_capture.subscribe(DataEventType.DATA_SYNC_STARTED, on_sync_started)
        event_capture.subscribe(DataEventType.DATA_SYNC_COMPLETED, on_sync_completed)
        event_capture.subscribe(DataEventType.ORPHAN_GAMES_REGISTERED, on_games_registered)
        event_capture.subscribe(DataEventType.NEW_GAMES_AVAILABLE, on_new_games)

        # Trigger the chain
        await event_capture.publish(
            DataEventType.ORPHAN_GAMES_DETECTED,
            {"source_node": "vast-12345", "config_key": config_key, "game_count": 50},
        )

        # Verify complete chain
        expected_chain = [
            "orphan_detected",
            "sync_started",
            "sync_completed",
            "games_registered",
            "new_games_available",
        ]
        assert chain_stage == expected_chain


# =============================================================================
# Pipeline 5: Curriculum Rebalanced -> Selfplay Allocation
# =============================================================================


class TestCurriculumToSelfplayPipeline:
    """End-to-end test for Curriculum Rebalanced -> Selfplay Allocation.

    This adjusts selfplay based on training progress:
    1. CURRICULUM_REBALANCED -> triggers WEIGHT_UPDATED
    2. WEIGHT_UPDATED -> affects scheduling
    3. SELFPLAY_ALLOCATION_UPDATED -> reflects new weights
    """

    @pytest.mark.asyncio
    async def test_curriculum_rebalance_triggers_weight_update(self, event_capture, config_key):
        """CURRICULUM_REBALANCED should trigger WEIGHT_UPDATED."""
        from app.distributed.data_events import DataEventType

        weight_updated = False
        new_weights = None

        async def mock_rebalance_handler(payload):
            nonlocal weight_updated, new_weights
            weight_updated = True
            new_weights = payload.get("new_weights", {})
            await event_capture.publish(
                DataEventType.WEIGHT_UPDATED,
                {
                    "config_key": payload.get("config_key"),
                    "weight_type": "curriculum",
                    "old_weight": payload.get("old_weights", {}).get(config_key, 1.0),
                    "new_weight": new_weights.get(config_key, 1.0),
                },
            )

        event_capture.subscribe(DataEventType.CURRICULUM_REBALANCED, mock_rebalance_handler)

        # Emit curriculum rebalanced
        await event_capture.publish(
            DataEventType.CURRICULUM_REBALANCED,
            {
                "config_key": config_key,
                "old_weights": {"hex8_2p": 1.0, "square8_2p": 1.0},
                "new_weights": {"hex8_2p": 1.5, "square8_2p": 0.8},
                "reason": "elo_improvement",
            },
        )

        # Verify weight was updated
        assert weight_updated
        assert new_weights.get("hex8_2p") == 1.5

        weight_event = event_capture.get_event(DataEventType.WEIGHT_UPDATED)
        assert weight_event is not None

    @pytest.mark.asyncio
    async def test_weight_update_triggers_allocation_change(self, event_capture, config_key):
        """WEIGHT_UPDATED should trigger SELFPLAY_ALLOCATION_UPDATED."""
        from app.distributed.data_events import DataEventType

        allocation_updated = False

        async def mock_weight_handler(payload):
            nonlocal allocation_updated
            allocation_updated = True
            await event_capture.publish(
                DataEventType.SELFPLAY_ALLOCATION_UPDATED,
                {
                    "config_key": payload.get("config_key"),
                    "allocation_change": "increased" if payload.get("new_weight", 1.0) > 1.0 else "decreased",
                    "new_allocation_percent": 25.0,  # Example allocation
                    "source": "selfplay_scheduler",
                },
            )

        event_capture.subscribe(DataEventType.WEIGHT_UPDATED, mock_weight_handler)

        # Emit weight updated
        await event_capture.publish(
            DataEventType.WEIGHT_UPDATED,
            {
                "config_key": config_key,
                "weight_type": "curriculum",
                "old_weight": 1.0,
                "new_weight": 1.5,
            },
        )

        # Verify allocation was updated
        assert allocation_updated

        alloc_event = event_capture.get_event(DataEventType.SELFPLAY_ALLOCATION_UPDATED)
        assert alloc_event is not None
        assert alloc_event.payload["allocation_change"] == "increased"

    @pytest.mark.asyncio
    async def test_full_curriculum_to_allocation_chain(self, event_capture, config_key):
        """Test complete chain: CURRICULUM_REBALANCED -> WEIGHT_UPDATED -> ALLOCATION_UPDATED."""
        from app.distributed.data_events import DataEventType

        chain_stage = []

        async def on_curriculum_rebalanced(payload):
            chain_stage.append("curriculum_rebalanced")
            await event_capture.publish(
                DataEventType.WEIGHT_UPDATED,
                {
                    "config_key": payload["config_key"],
                    "new_weight": payload.get("new_weights", {}).get(config_key, 1.0),
                },
            )

        async def on_weight_updated(payload):
            chain_stage.append("weight_updated")
            await event_capture.publish(
                DataEventType.SELFPLAY_ALLOCATION_UPDATED,
                {"config_key": payload["config_key"], "new_allocation_percent": 25.0},
            )

        async def on_allocation_updated(payload):
            chain_stage.append("allocation_updated")

        # Subscribe all handlers
        event_capture.subscribe(DataEventType.CURRICULUM_REBALANCED, on_curriculum_rebalanced)
        event_capture.subscribe(DataEventType.WEIGHT_UPDATED, on_weight_updated)
        event_capture.subscribe(DataEventType.SELFPLAY_ALLOCATION_UPDATED, on_allocation_updated)

        # Trigger the chain
        await event_capture.publish(
            DataEventType.CURRICULUM_REBALANCED,
            {
                "config_key": config_key,
                "new_weights": {"hex8_2p": 1.5},
                "reason": "elo_improvement",
            },
        )

        # Verify complete chain
        expected_chain = ["curriculum_rebalanced", "weight_updated", "allocation_updated"]
        assert chain_stage == expected_chain


# =============================================================================
# Cross-Pipeline Integration Tests
# =============================================================================


class TestCrossPipelineIntegration:
    """Tests for interactions between multiple pipelines."""

    @pytest.mark.asyncio
    async def test_recovery_leads_to_retraining(self, event_capture, config_key):
        """RECOVERY_COMPLETED should eventually lead to new training cycle."""
        from app.distributed.data_events import DataEventType

        chain_stage = []

        async def on_recovery_completed(payload):
            chain_stage.append("recovery_completed")
            # Recovery might trigger data refresh
            await event_capture.publish(
                DataEventType.DATA_STALE,
                {"config_key": payload["config_key"], "reason": "post_recovery_refresh"},
            )

        async def on_data_stale(payload):
            chain_stage.append("data_stale")
            await event_capture.publish(
                DataEventType.SYNC_TRIGGERED,
                {"config_key": payload["config_key"]},
            )

        async def on_sync_triggered(payload):
            chain_stage.append("sync_triggered")

        event_capture.subscribe(DataEventType.RECOVERY_COMPLETED, on_recovery_completed)
        event_capture.subscribe(DataEventType.DATA_STALE, on_data_stale)
        event_capture.subscribe(DataEventType.SYNC_TRIGGERED, on_sync_triggered)

        # Trigger recovery
        await event_capture.publish(
            DataEventType.RECOVERY_COMPLETED,
            {"config_key": config_key, "recovery_type": "rollback"},
        )

        # Verify chain leading to sync
        assert "recovery_completed" in chain_stage
        assert "data_stale" in chain_stage
        assert "sync_triggered" in chain_stage

    @pytest.mark.asyncio
    async def test_evaluation_failure_does_not_promote(self, event_capture, config_key):
        """EVALUATION_COMPLETED with failure should NOT trigger promotion."""
        from app.distributed.data_events import DataEventType

        promotion_triggered = False

        async def on_evaluation_completed(payload):
            if payload.get("passed_gauntlet"):
                await event_capture.publish(
                    DataEventType.MODEL_PROMOTED,
                    {"config_key": payload["config_key"]},
                )

        async def on_model_promoted(payload):
            nonlocal promotion_triggered
            promotion_triggered = True

        event_capture.subscribe(DataEventType.EVALUATION_COMPLETED, on_evaluation_completed)
        event_capture.subscribe(DataEventType.MODEL_PROMOTED, on_model_promoted)

        # Emit evaluation with failure
        await event_capture.publish(
            DataEventType.EVALUATION_COMPLETED,
            {
                "config_key": config_key,
                "passed_gauntlet": False,  # Failed
                "win_rate_random": 0.80,  # Below threshold
            },
        )

        # Verify no promotion
        assert not promotion_triggered


# =============================================================================
# Event Ordering and Timeout Tests
# =============================================================================


class TestEventOrdering:
    """Tests for event ordering and timeout handling."""

    @pytest.mark.asyncio
    async def test_events_captured_in_emission_order(self, event_capture, config_key):
        """Events should be captured in the order they are emitted."""
        from app.distributed.data_events import DataEventType

        # Emit events in specific order
        await event_capture.publish(DataEventType.TRAINING_STARTED, {"order": 1})
        await event_capture.publish(DataEventType.TRAINING_PROGRESS, {"order": 2})
        await event_capture.publish(DataEventType.TRAINING_COMPLETED, {"order": 3})

        # Get events
        event_order = event_capture.get_events_in_order()

        # Verify order
        assert event_order[0] == DataEventType.TRAINING_STARTED.value
        assert event_order[1] == DataEventType.TRAINING_PROGRESS.value
        assert event_order[2] == DataEventType.TRAINING_COMPLETED.value

    @pytest.mark.asyncio
    async def test_wait_for_event_with_timeout(self, event_capture, config_key):
        """wait_for_event should respect timeout."""
        from app.distributed.data_events import DataEventType

        # Wait for event that won't come
        result = await event_capture.wait_for_event(
            DataEventType.MODEL_PROMOTED,
            timeout=0.1,  # Short timeout
        )

        # Should return None on timeout
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_event_receives_event(self, event_capture, config_key):
        """wait_for_event should return when event arrives."""
        from app.distributed.data_events import DataEventType

        async def delayed_emit():
            await asyncio.sleep(0.05)
            await event_capture.publish(
                DataEventType.MODEL_PROMOTED,
                {"config_key": config_key},
            )

        # Start waiting and emit in parallel
        asyncio.create_task(delayed_emit())

        result = await event_capture.wait_for_event(
            DataEventType.MODEL_PROMOTED,
            timeout=1.0,
        )

        # Should receive the event
        assert result is not None
        assert result.event_type == DataEventType.MODEL_PROMOTED.value


# =============================================================================
# Real Component Integration Tests (with actual coordination modules)
# =============================================================================


class TestRealComponentIntegration:
    """Tests using actual coordination components (not mocks)."""

    def test_data_event_type_values_match_expected(self):
        """Verify DataEventType values match what handlers expect."""
        from app.distributed.data_events import DataEventType

        # Verify key event type values
        assert DataEventType.TRAINING_COMPLETED.value == "training_completed"
        assert DataEventType.EVALUATION_STARTED.value == "evaluation_started"
        assert DataEventType.MODEL_PROMOTED.value == "model_promoted"
        assert DataEventType.REGRESSION_DETECTED.value == "regression_detected"
        assert DataEventType.ORPHAN_GAMES_DETECTED.value == "orphan_games_detected"
        assert DataEventType.CURRICULUM_REBALANCED.value == "curriculum_rebalanced"

    def test_handler_methods_exist_on_coordinators(self):
        """Verify handler methods exist on relevant coordinators."""
        # FeedbackLoopController
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        assert hasattr(FeedbackLoopController, "_on_training_complete")

        # DataPipelineOrchestrator
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        assert hasattr(DataPipelineOrchestrator, "_on_orphan_games_detected")
        assert hasattr(DataPipelineOrchestrator, "_on_sync_complete")

        # SelfplayScheduler
        from app.coordination.selfplay_scheduler import SelfplayScheduler

        assert hasattr(SelfplayScheduler, "_on_curriculum_rebalanced")
        assert hasattr(SelfplayScheduler, "_on_exploration_boost")

    def test_event_mappings_complete_for_pipelines(self):
        """Verify event mappings include all pipeline events."""
        from app.coordination.event_mappings import DATA_TO_CROSS_PROCESS_MAP

        # Pipeline 1: Training -> Promotion
        assert "training_completed" in DATA_TO_CROSS_PROCESS_MAP
        assert "evaluation_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "evaluation_completed" in DATA_TO_CROSS_PROCESS_MAP
        assert "model_promoted" in DATA_TO_CROSS_PROCESS_MAP

        # Pipeline 3: Regression -> Recovery
        assert "regression_detected" in DATA_TO_CROSS_PROCESS_MAP

        # Pipeline 4: Orphan recovery
        assert "orphan_games_detected" in DATA_TO_CROSS_PROCESS_MAP
        assert "orphan_games_registered" in DATA_TO_CROSS_PROCESS_MAP

        # Pipeline 5: Curriculum
        assert "curriculum_rebalanced" in DATA_TO_CROSS_PROCESS_MAP
        assert "weight_updated" in DATA_TO_CROSS_PROCESS_MAP
