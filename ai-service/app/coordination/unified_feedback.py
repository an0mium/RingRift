"""Unified Feedback Orchestrator - Single source of truth for all feedback signals.

This module consolidates 4 previously separate feedback systems:
1. FeedbackLoopController - intensity, exploration signals
2. GauntletFeedbackController - evaluation-driven adjustments
3. CurriculumIntegration - curriculum weight bridges
4. TrainingFreshness - data staleness signals

The UnifiedFeedbackOrchestrator provides:
- Single event bus for all feedback signals
- Centralized state management
- Observable metrics for dashboards
- Configurable adjustment strategies

Usage:
    from app.coordination.unified_feedback import (
        UnifiedFeedbackOrchestrator,
        get_feedback_orchestrator,
    )

    # Get singleton
    orchestrator = get_feedback_orchestrator()

    # Start orchestration
    await orchestrator.start()

    # Query current state
    state = orchestrator.get_state("hex8_2p")
    print(f"Intensity: {state.intensity}")
    print(f"Exploration: {state.exploration_multiplier}")

    # Manual signal (usually automatic via events)
    orchestrator.signal_evaluation_complete("hex8_2p", win_rate=0.75, elo=1650)

December 2025: Created for Phase 2 feedback consolidation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.coordination.feedback_signals import (
    FeedbackSignal,
    FeedbackState,
    SignalSource,
    SignalType,
    emit_exploration_signal,
    emit_intensity_signal,
    emit_promotion_signal,
    emit_quality_signal,
    emit_regression_signal,
    get_all_feedback_states,
    get_feedback_state,
    subscribe_to_signal,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class UnifiedFeedbackConfig:
    """Configuration for the unified feedback orchestrator."""

    # Intensity thresholds
    hot_path_accuracy_threshold: float = 0.65  # Below this → hot_path
    cool_down_accuracy_threshold: float = 0.85  # Above this → cool_down

    # Exploration thresholds
    strong_model_win_rate: float = 0.80  # Reduce exploration above this
    weak_model_win_rate: float = 0.50  # Increase exploration below this
    exploration_reduction_factor: float = 0.8
    exploration_boost_factor: float = 1.3
    max_exploration_multiplier: float = 2.0
    min_exploration_multiplier: float = 0.5

    # Quality thresholds
    quality_export_threshold: float = 0.70  # Trigger export above this
    quality_warning_threshold: float = 0.50  # Emit warning below this

    # Freshness thresholds (hours)
    fresh_data_hours: float = 1.0
    stale_data_hours: float = 4.0
    critical_data_hours: float = 24.0

    # Regression thresholds
    regression_elo_drop: float = 50.0
    regression_win_rate_drop: float = 0.10

    # Cooldowns (seconds)
    intensity_cooldown: float = 300.0
    exploration_cooldown: float = 300.0
    curriculum_cooldown: float = 600.0

    # Event integration
    subscribe_to_events: bool = True


# =============================================================================
# Adjustment Strategies
# =============================================================================


class IntensityStrategy:
    """Strategy for adjusting training intensity based on signals."""

    def __init__(self, config: UnifiedFeedbackConfig):
        self.config = config
        self._last_adjustment: dict[str, float] = {}

    def compute_intensity(
        self,
        config_key: str,
        policy_accuracy: float | None = None,
        win_rate: float | None = None,
        elo_velocity: float | None = None,
    ) -> str | None:
        """Compute recommended intensity based on metrics.

        Returns:
            "hot_path", "normal", "cool_down", or None if no change needed
        """
        # Check cooldown
        now = time.time()
        last = self._last_adjustment.get(config_key, 0)
        if now - last < self.config.intensity_cooldown:
            return None

        # Compute intensity
        if policy_accuracy is not None:
            if policy_accuracy < self.config.hot_path_accuracy_threshold:
                self._last_adjustment[config_key] = now
                return "hot_path"
            elif policy_accuracy > self.config.cool_down_accuracy_threshold:
                self._last_adjustment[config_key] = now
                return "cool_down"

        if elo_velocity is not None and elo_velocity < 0:
            # Negative velocity = regressing, need more training
            self._last_adjustment[config_key] = now
            return "hot_path"

        return None


class ExplorationStrategy:
    """Strategy for adjusting exploration based on signals."""

    def __init__(self, config: UnifiedFeedbackConfig):
        self.config = config
        self._last_adjustment: dict[str, float] = {}

    def compute_exploration(
        self,
        config_key: str,
        win_rate_vs_heuristic: float | None = None,
        promotion_failed: bool = False,
    ) -> float | None:
        """Compute recommended exploration multiplier.

        Returns:
            Multiplier (0.5-2.0) or None if no change needed
        """
        # Check cooldown
        now = time.time()
        last = self._last_adjustment.get(config_key, 0)
        if now - last < self.config.exploration_cooldown:
            return None

        current_state = get_feedback_state(config_key)
        current = current_state.exploration_multiplier

        # Promotion failure → boost exploration
        if promotion_failed:
            new_value = min(
                current * self.config.exploration_boost_factor,
                self.config.max_exploration_multiplier,
            )
            self._last_adjustment[config_key] = now
            return new_value

        # Strong model → reduce exploration
        if win_rate_vs_heuristic is not None:
            if win_rate_vs_heuristic > self.config.strong_model_win_rate:
                new_value = max(
                    current * self.config.exploration_reduction_factor,
                    self.config.min_exploration_multiplier,
                )
                self._last_adjustment[config_key] = now
                return new_value
            elif win_rate_vs_heuristic < self.config.weak_model_win_rate:
                new_value = min(
                    current * self.config.exploration_boost_factor,
                    self.config.max_exploration_multiplier,
                )
                self._last_adjustment[config_key] = now
                return new_value

        return None


# =============================================================================
# Unified Feedback Orchestrator
# =============================================================================


class UnifiedFeedbackOrchestrator:
    """Central orchestrator for all training feedback signals.

    Consolidates:
    - FeedbackLoopController (intensity, exploration)
    - GauntletFeedbackController (evaluation adjustments)
    - CurriculumIntegration (weight bridges)
    - TrainingFreshness (data staleness)
    """

    def __init__(self, config: UnifiedFeedbackConfig | None = None):
        self.config = config or UnifiedFeedbackConfig()
        self._running = False
        self._lock = threading.Lock()

        # Strategies
        self._intensity_strategy = IntensityStrategy(self.config)
        self._exploration_strategy = ExplorationStrategy(self.config)

        # Event subscriptions
        self._unsubscribers: list = []

        # Metrics
        self._signals_processed = 0
        self._last_activity = time.time()

    async def start(self) -> None:
        """Start the unified feedback orchestrator."""
        if self._running:
            logger.warning("[UnifiedFeedback] Already running")
            return

        self._running = True
        logger.info("[UnifiedFeedback] Starting unified feedback orchestrator")

        # Subscribe to data events if enabled
        if self.config.subscribe_to_events:
            self._subscribe_to_events()

        # Wire legacy systems to emit through unified signals
        self._wire_legacy_systems()

        logger.info("[UnifiedFeedback] Started successfully")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False

        # Unsubscribe from events
        for unsub in self._unsubscribers:
            try:
                unsub()
            except Exception:
                pass
        self._unsubscribers.clear()

        logger.info("[UnifiedFeedback] Stopped")

    def _subscribe_to_events(self) -> None:
        """Subscribe to data events from the event bus."""
        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus is None:
                logger.debug("[UnifiedFeedback] No event bus available")
                return

            # Subscribe to key events
            events_to_subscribe = [
                ("TRAINING_COMPLETE", self._on_training_complete),
                ("EVALUATION_COMPLETED", self._on_evaluation_complete),
                ("MODEL_PROMOTED", self._on_model_promoted),
                ("PROMOTION_FAILED", self._on_promotion_failed),
                ("SELFPLAY_COMPLETE", self._on_selfplay_complete),
            ]

            for event_name, handler in events_to_subscribe:
                try:
                    if hasattr(DataEventType, event_name):
                        bus.subscribe(event_name, handler)
                        logger.debug(f"[UnifiedFeedback] Subscribed to {event_name}")
                except Exception as e:
                    logger.debug(f"[UnifiedFeedback] Could not subscribe to {event_name}: {e}")

        except ImportError:
            logger.debug("[UnifiedFeedback] Data events not available")

    def _wire_legacy_systems(self) -> None:
        """Wire legacy feedback systems to emit through unified signals."""
        # Wire FeedbackLoopController signals
        try:
            unsub = subscribe_to_signal(
                SignalType.INTENSITY,
                self._on_intensity_signal,
            )
            self._unsubscribers.append(unsub)
        except Exception as e:
            logger.debug(f"[UnifiedFeedback] Could not wire intensity signals: {e}")

        # Wire exploration signals
        try:
            unsub = subscribe_to_signal(
                SignalType.EXPLORATION,
                self._on_exploration_signal,
            )
            self._unsubscribers.append(unsub)
        except Exception as e:
            logger.debug(f"[UnifiedFeedback] Could not wire exploration signals: {e}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_training_complete(self, event: Any) -> None:
        """Handle training complete event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            policy_accuracy = payload.get("policy_accuracy", 0.0)

            if not config_key:
                return

            self._signals_processed += 1
            self._last_activity = time.time()

            # Compute new intensity
            new_intensity = self._intensity_strategy.compute_intensity(
                config_key,
                policy_accuracy=policy_accuracy,
            )

            if new_intensity:
                emit_intensity_signal(
                    config_key=config_key,
                    intensity=new_intensity,
                    reason=f"policy_accuracy={policy_accuracy:.2%}",
                    source=SignalSource.TRAINING,
                )
                logger.info(
                    f"[UnifiedFeedback] {config_key}: intensity → {new_intensity} "
                    f"(accuracy={policy_accuracy:.2%})"
                )

        except Exception as e:
            logger.error(f"[UnifiedFeedback] Error handling training complete: {e}")

    def _on_evaluation_complete(self, event: Any) -> None:
        """Handle evaluation complete event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            win_rate = payload.get("win_rate_vs_heuristic", 0.0)
            elo = payload.get("elo", 0.0)

            if not config_key:
                return

            self._signals_processed += 1
            self._last_activity = time.time()

            # Compute new exploration
            new_exploration = self._exploration_strategy.compute_exploration(
                config_key,
                win_rate_vs_heuristic=win_rate,
            )

            if new_exploration:
                emit_exploration_signal(
                    config_key=config_key,
                    multiplier=new_exploration,
                    reason=f"win_rate={win_rate:.2%}",
                    source=SignalSource.EVALUATION,
                )
                logger.info(
                    f"[UnifiedFeedback] {config_key}: exploration → {new_exploration:.2f} "
                    f"(win_rate={win_rate:.2%})"
                )

            # Check for regression
            state = get_feedback_state(config_key)
            if hasattr(state, "last_elo") and state.last_elo > 0:
                elo_drop = state.last_elo - elo
                if elo_drop > self.config.regression_elo_drop:
                    emit_regression_signal(
                        config_key=config_key,
                        detected=True,
                        elo_drop=elo_drop,
                    )
                    logger.warning(
                        f"[UnifiedFeedback] {config_key}: REGRESSION detected "
                        f"(elo_drop={elo_drop:.1f})"
                    )

        except Exception as e:
            logger.error(f"[UnifiedFeedback] Error handling evaluation complete: {e}")

    def _on_model_promoted(self, event: Any) -> None:
        """Handle model promoted event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            model_path = payload.get("model_path", "")

            if not config_key:
                return

            self._signals_processed += 1
            self._last_activity = time.time()

            # Reset exploration on successful promotion
            emit_exploration_signal(
                config_key=config_key,
                multiplier=1.0,
                reason="promotion_success",
                source=SignalSource.PROMOTION,
            )

            emit_promotion_signal(
                config_key=config_key,
                outcome="promoted",
                model_path=model_path,
            )

            logger.info(f"[UnifiedFeedback] {config_key}: promotion SUCCESS, exploration reset")

        except Exception as e:
            logger.error(f"[UnifiedFeedback] Error handling model promoted: {e}")

    def _on_promotion_failed(self, event: Any) -> None:
        """Handle promotion failed event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")

            if not config_key:
                return

            self._signals_processed += 1
            self._last_activity = time.time()

            # Boost exploration on failed promotion
            new_exploration = self._exploration_strategy.compute_exploration(
                config_key,
                promotion_failed=True,
            )

            if new_exploration:
                emit_exploration_signal(
                    config_key=config_key,
                    multiplier=new_exploration,
                    reason="promotion_failed",
                    source=SignalSource.PROMOTION,
                )

            emit_promotion_signal(
                config_key=config_key,
                outcome="rejected",
            )

            logger.info(
                f"[UnifiedFeedback] {config_key}: promotion FAILED, "
                f"exploration → {new_exploration:.2f}"
            )

        except Exception as e:
            logger.error(f"[UnifiedFeedback] Error handling promotion failed: {e}")

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay complete event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            quality_score = payload.get("quality_score", 0.0)

            if not config_key:
                return

            self._signals_processed += 1
            self._last_activity = time.time()

            # Emit quality signal
            if quality_score > 0:
                emit_quality_signal(
                    config_key=config_key,
                    score=quality_score,
                    reason="selfplay_batch",
                    source=SignalSource.SELFPLAY,
                )

        except Exception as e:
            logger.error(f"[UnifiedFeedback] Error handling selfplay complete: {e}")

    def _on_intensity_signal(self, signal: FeedbackSignal) -> None:
        """Handle intensity signal from legacy system."""
        self._signals_processed += 1
        self._last_activity = time.time()
        # Signal is already processed by feedback_signals module
        logger.debug(f"[UnifiedFeedback] Intensity signal: {signal}")

    def _on_exploration_signal(self, signal: FeedbackSignal) -> None:
        """Handle exploration signal from legacy system."""
        self._signals_processed += 1
        self._last_activity = time.time()
        # Signal is already processed by feedback_signals module
        logger.debug(f"[UnifiedFeedback] Exploration signal: {signal}")

    # =========================================================================
    # Public API
    # =========================================================================

    def signal_training_complete(
        self,
        config_key: str,
        policy_accuracy: float,
        loss: float = 0.0,
    ) -> None:
        """Manually signal training completion.

        Args:
            config_key: Config that completed training
            policy_accuracy: Final policy accuracy
            loss: Final training loss
        """
        self._on_training_complete({
            "config_key": config_key,
            "policy_accuracy": policy_accuracy,
            "loss": loss,
        })

    def signal_evaluation_complete(
        self,
        config_key: str,
        win_rate: float,
        elo: float = 0.0,
    ) -> None:
        """Manually signal evaluation completion.

        Args:
            config_key: Config that was evaluated
            win_rate: Win rate vs heuristic
            elo: Current Elo rating
        """
        self._on_evaluation_complete({
            "config_key": config_key,
            "win_rate_vs_heuristic": win_rate,
            "elo": elo,
        })

    def get_state(self, config_key: str) -> FeedbackState:
        """Get current feedback state for a config.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Current feedback state
        """
        return get_feedback_state(config_key)

    def get_all_states(self) -> dict[str, FeedbackState]:
        """Get current feedback state for all configs."""
        return get_all_feedback_states()

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics."""
        return {
            "running": self._running,
            "signals_processed": self._signals_processed,
            "last_activity": self._last_activity,
            "states_count": len(get_all_feedback_states()),
        }


# =============================================================================
# Singleton
# =============================================================================

_orchestrator: UnifiedFeedbackOrchestrator | None = None
_orchestrator_lock = threading.Lock()


def get_feedback_orchestrator() -> UnifiedFeedbackOrchestrator:
    """Get the singleton UnifiedFeedbackOrchestrator instance."""
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = UnifiedFeedbackOrchestrator()
        return _orchestrator


async def start_unified_feedback() -> UnifiedFeedbackOrchestrator:
    """Start the unified feedback orchestrator.

    Returns:
        The running orchestrator instance
    """
    orchestrator = get_feedback_orchestrator()
    await orchestrator.start()
    return orchestrator
