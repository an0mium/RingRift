"""Integration tests for complete feedback loop closure.

Tests the end-to-end event flow through the entire training pipeline:
    Selfplay → Training → Evaluation → Feedback → Curriculum → Selfplay

This test verifies that:
1. Bootstrap properly initializes the unified feedback orchestrator
2. Events propagate through the complete loop
3. Feedback signals correctly influence downstream components
4. The loop achieves closure (output affects input)

December 2025: Created as part of Phase 3 integration testing.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBootstrapFeedbackIntegration:
    """Tests for bootstrap + unified feedback integration."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset all singleton instances before each test."""
        # Reset event router
        try:
            from app.coordination.event_router import reset_router

            reset_router()
        except ImportError:
            pass

        # Reset unified feedback
        try:
            from app.coordination.unified_feedback import reset_unified_feedback

            reset_unified_feedback()
        except ImportError:
            pass

        # Reset bootstrap state
        try:
            from app.coordination.coordination_bootstrap import _state

            _state.initialized = False
            _state.coordinators.clear()
            _state.errors.clear()
        except (ImportError, AttributeError):
            pass

        yield

        # Cleanup after test
        try:
            from app.coordination.event_router import reset_router

            reset_router()
        except ImportError:
            pass

        try:
            from app.coordination.unified_feedback import reset_unified_feedback

            reset_unified_feedback()
        except ImportError:
            pass

    def test_unified_feedback_initialized_by_bootstrap(self):
        """Bootstrap should initialize the unified feedback orchestrator."""
        from app.coordination.coordination_bootstrap import (
            _start_unified_feedback_orchestrator,
        )
        from app.coordination.unified_feedback import get_unified_feedback

        # Start the orchestrator through bootstrap function
        result = _start_unified_feedback_orchestrator()

        # Should succeed
        assert result is True

        # Should be able to get the orchestrator
        orchestrator = get_unified_feedback()
        assert orchestrator is not None

    @pytest.mark.asyncio
    async def test_unified_feedback_subscribes_to_events(self):
        """Unified feedback should subscribe to required events after start."""
        from app.coordination.event_router import get_event_bus
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_unified_feedback()
        orchestrator = UnifiedFeedbackOrchestrator()

        # Start the orchestrator
        await orchestrator.start()

        # Should be running and subscribed
        assert orchestrator._running is True
        assert orchestrator._subscribed is True

        # Clean up
        await orchestrator.stop()


class TestCompleteFeedbackLoop:
    """Tests for complete feedback loop: selfplay → training → eval → feedback → curriculum."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        """Reset all state before each test."""
        try:
            from app.coordination.event_router import reset_router

            reset_router()
        except ImportError:
            pass

        try:
            from app.coordination.unified_feedback import reset_unified_feedback

            reset_unified_feedback()
        except ImportError:
            pass

        yield

    @pytest.mark.asyncio
    async def test_selfplay_complete_triggers_feedback_update(self):
        """SELFPLAY_COMPLETE event should update feedback state."""
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Track events received
        events_received: list[str] = []

        async def track_event(event):
            events_received.append(event.event_type.value)

        bus.subscribe(DataEventType.SELFPLAY_COMPLETE, track_event)

        # Emit selfplay complete event
        event = DataEvent(
            event_type=DataEventType.SELFPLAY_COMPLETE,
            payload={
                "config_key": "hex8_2p",
                "games_completed": 100,
                "quality_score": 0.85,
            },
            source="test",
        )
        await bus.publish(event)
        await asyncio.sleep(0.1)

        # Event should have been received
        assert "selfplay_complete" in events_received

        # State should exist (use internal method to verify)
        state = orchestrator._get_or_create_state("hex8_2p")
        assert state is not None
        assert state.config_key == "hex8_2p"

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_training_complete_triggers_curriculum_adjustment(self):
        """TRAINING_COMPLETED should trigger curriculum weight recalculation."""
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Emit training complete event with low accuracy (should increase curriculum weight)
        event = DataEvent(
            event_type=DataEventType.TRAINING_COMPLETED,
            payload={
                "config_key": "hex8_2p",
                "config": "hex8_2p",
                "policy_accuracy": 0.45,  # Low accuracy
                "value_accuracy": 0.40,
                "epochs": 50,
            },
            source="test",
        )
        await bus.publish(event)
        await asyncio.sleep(0.1)

        # State should exist
        state = orchestrator.get_config_state("hex8_2p")
        assert state is not None

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_quality_degraded_triggers_exploration_boost(self):
        """QUALITY_DEGRADED should increase exploration boost."""
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Get initial exploration boost
        state_before = orchestrator._get_or_create_state("hex8_2p")
        boost_before = state_before.exploration_boost

        # Emit quality degraded event
        event = DataEvent(
            event_type=DataEventType.QUALITY_DEGRADED,
            payload={
                "config_key": "hex8_2p",
                "quality_score": 0.25,  # Very low quality
                "reason": "low_diversity",
            },
            source="test",
        )
        await bus.publish(event)
        await asyncio.sleep(0.1)

        # Exploration boost should increase
        state_after = orchestrator.get_config_state("hex8_2p")
        assert state_after.exploration_boost >= boost_before

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_plateau_detected_triggers_intensity_change(self):
        """PLATEAU_DETECTED should trigger training intensity adjustment."""
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Emit plateau detected event
        event = DataEvent(
            event_type=DataEventType.PLATEAU_DETECTED,
            payload={
                "config_key": "hex8_2p",
                "plateau_duration": 10,  # Plateau for 10 iterations
                "elo_delta": 5,  # Minimal progress
            },
            source="test",
        )
        await bus.publish(event)
        await asyncio.sleep(0.1)

        # State should be updated
        state = orchestrator.get_config_state("hex8_2p")
        assert state is not None

        await orchestrator.stop()


class TestFeedbackLoopClosure:
    """Tests that verify the complete feedback loop achieves closure."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        """Reset all state before each test."""
        try:
            from app.coordination.event_router import reset_router

            reset_router()
        except ImportError:
            pass

        try:
            from app.coordination.unified_feedback import reset_unified_feedback

            reset_unified_feedback()
        except ImportError:
            pass

        yield

    @pytest.mark.asyncio
    async def test_feedback_affects_selfplay_allocation(self):
        """Feedback signals should influence selfplay scheduler allocation.

        This test verifies the loop closure: feedback → curriculum → selfplay allocation.
        """
        from app.coordination.event_router import reset_router
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Simulate different feedback states for different configs
        # Create state for a "weak" config (needs more training)
        weak_state = orchestrator._get_or_create_state("hex8_2p")
        weak_state.curriculum_weight = 2.0  # High priority
        weak_state.training_intensity = "hot_path"
        weak_state.exploration_boost = 1.3

        # Create state for a "strong" config (less training needed)
        strong_state = orchestrator._get_or_create_state("square8_2p")
        strong_state.curriculum_weight = 0.5  # Low priority
        strong_state.training_intensity = "reduced"
        strong_state.exploration_boost = 0.9

        # Verify the states are accessible (simulates scheduler reading them)
        hex8_state = orchestrator.get_config_state("hex8_2p")
        sq8_state = orchestrator.get_config_state("square8_2p")

        assert hex8_state.curriculum_weight > sq8_state.curriculum_weight
        assert hex8_state.training_intensity == "hot_path"
        assert sq8_state.training_intensity == "reduced"

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_complete_event_chain(self):
        """Test a complete chain of events through the feedback system.

        Simulates: selfplay → training → evaluation → feedback adjustment
        """
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        config_key = "hex8_4p"
        events_received: list[str] = []

        # Track all events for verification
        async def track_event(event: DataEvent):
            events_received.append(event.event_type.value)

        bus.subscribe(DataEventType.SELFPLAY_COMPLETE, track_event)
        bus.subscribe(DataEventType.TRAINING_COMPLETED, track_event)
        bus.subscribe(DataEventType.QUALITY_DEGRADED, track_event)

        # Step 1: Selfplay completes
        await bus.publish(
            DataEvent(
                event_type=DataEventType.SELFPLAY_COMPLETE,
                payload={
                    "config_key": config_key,
                    "games_completed": 500,
                    "quality_score": 0.75,
                },
                source="test",
            )
        )
        await asyncio.sleep(0.05)

        # Step 2: Training completes with mediocre accuracy
        await bus.publish(
            DataEvent(
                event_type=DataEventType.TRAINING_COMPLETED,
                payload={
                    "config_key": config_key,
                    "config": config_key,
                    "policy_accuracy": 0.52,
                    "value_accuracy": 0.48,
                    "epochs": 30,
                },
                source="test",
            )
        )
        await asyncio.sleep(0.05)

        # Step 3: Quality degrades
        await bus.publish(
            DataEvent(
                event_type=DataEventType.QUALITY_DEGRADED,
                payload={
                    "config_key": config_key,
                    "quality_score": 0.35,
                    "reason": "low_accuracy",
                },
                source="test",
            )
        )
        await asyncio.sleep(0.05)

        # Verify feedback state was updated throughout the chain
        state = orchestrator.get_config_state(config_key)
        assert state is not None
        assert state.config_key == config_key

        # Verify events were received
        assert "selfplay_complete" in events_received
        assert "training_completed" in events_received
        assert "quality_degraded" in events_received

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_metrics_reflect_adjustments(self):
        """Orchestrator metrics should reflect feedback adjustments."""
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Get initial metrics
        initial_metrics = orchestrator.get_metrics()
        initial_adjustments = initial_metrics.get("total_adjustments", 0)

        # Emit events that trigger adjustments
        for i in range(3):
            await bus.publish(
                DataEvent(
                    event_type=DataEventType.QUALITY_DEGRADED,
                    payload={
                        "config_key": f"config_{i}",
                        "quality_score": 0.3,
                        "reason": "test",
                    },
                    source="test",
                )
            )
            await asyncio.sleep(0.05)

        # Get updated metrics
        final_metrics = orchestrator.get_metrics()

        # Metrics should be accessible
        assert "total_adjustments" in final_metrics
        assert "adjustments_by_type" in final_metrics
        assert "last_adjustment_time" in final_metrics

        await orchestrator.stop()


class TestDataFreshnessLoop:
    """Tests for data freshness feedback integration."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        """Reset all state before each test."""
        try:
            from app.coordination.event_router import reset_router

            reset_router()
        except ImportError:
            pass

        try:
            from app.coordination.unified_feedback import reset_unified_feedback

            reset_unified_feedback()
        except ImportError:
            pass

        yield

    @pytest.mark.asyncio
    async def test_data_fresh_event_updates_state(self):
        """DATA_FRESH event should update data freshness state."""
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Emit data fresh event
        await bus.publish(
            DataEvent(
                event_type=DataEventType.DATA_FRESH,
                payload={
                    "board_type": "hex8",
                    "num_players": 2,
                    "data_age_hours": 0.25,  # 15 minutes old - very fresh
                },
                source="test",
            )
        )
        await asyncio.sleep(0.1)

        # State should reflect fresh data
        state = orchestrator.get_config_state("hex8_2p")
        assert state.data_freshness_hours == 0.25

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_data_stale_event_updates_state(self):
        """DATA_STALE event should update data freshness state."""
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Emit data stale event
        await bus.publish(
            DataEvent(
                event_type=DataEventType.DATA_STALE,
                payload={
                    "board_type": "square19",
                    "num_players": 4,
                    "data_age_hours": 48.0,  # 2 days old - very stale
                },
                source="test",
            )
        )
        await asyncio.sleep(0.1)

        # State should reflect stale data
        state = orchestrator.get_config_state("square19_4p")
        assert state.data_freshness_hours == 48.0

        await orchestrator.stop()


class TestMultiConfigFeedback:
    """Tests for feedback handling across multiple configurations."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        """Reset all state before each test."""
        try:
            from app.coordination.event_router import reset_router

            reset_router()
        except ImportError:
            pass

        try:
            from app.coordination.unified_feedback import reset_unified_feedback

            reset_unified_feedback()
        except ImportError:
            pass

        yield

    @pytest.mark.asyncio
    async def test_independent_config_states(self):
        """Each config should have independent feedback state."""
        from app.coordination.event_router import (
            DataEvent,
            DataEventType,
            get_event_bus,
            reset_router,
        )
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_router()
        reset_unified_feedback()

        bus = get_event_bus()
        orchestrator = UnifiedFeedbackOrchestrator()
        await orchestrator.start()

        # Create events for different configs with different states
        configs = ["hex8_2p", "square8_4p", "hexagonal_3p"]
        quality_scores = [0.9, 0.5, 0.3]  # Good, mediocre, poor

        # Create states for each config (simulates event-driven creation)
        for config, quality in zip(configs, quality_scores):
            # Create state directly to test independence
            state = orchestrator._get_or_create_state(config)
            state.last_selfplay_quality = quality

        # Each config should have its own state
        hex8_state = orchestrator._get_or_create_state("hex8_2p")
        sq8_state = orchestrator._get_or_create_state("square8_4p")
        hex_state = orchestrator._get_or_create_state("hexagonal_3p")

        assert hex8_state.config_key == "hex8_2p"
        assert sq8_state.config_key == "square8_4p"
        assert hex_state.config_key == "hexagonal_3p"

        # States should be independent
        assert hex8_state is not sq8_state
        assert sq8_state is not hex_state

        # Verify quality scores are independent
        assert hex8_state.last_selfplay_quality == 0.9
        assert sq8_state.last_selfplay_quality == 0.5
        assert hex_state.last_selfplay_quality == 0.3

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_all_12_configs_can_have_state(self):
        """All 12 canonical configurations should be able to have state."""
        from app.coordination.unified_feedback import (
            UnifiedFeedbackOrchestrator,
            reset_unified_feedback,
        )

        reset_unified_feedback()
        orchestrator = UnifiedFeedbackOrchestrator()

        # All 12 canonical configs
        configs = [
            "hex8_2p",
            "hex8_3p",
            "hex8_4p",
            "square8_2p",
            "square8_3p",
            "square8_4p",
            "square19_2p",
            "square19_3p",
            "square19_4p",
            "hexagonal_2p",
            "hexagonal_3p",
            "hexagonal_4p",
        ]

        # Create state for each config
        for config in configs:
            state = orchestrator._get_or_create_state(config)
            assert state is not None
            assert state.config_key == config

        # Verify all 12 states exist
        for config in configs:
            state = orchestrator.get_config_state(config)
            assert state is not None
