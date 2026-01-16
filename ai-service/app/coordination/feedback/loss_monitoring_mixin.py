"""Loss monitoring handling for FeedbackLoopController.

Sprint 17.9 (Jan 16, 2026): Extracted from feedback_loop_controller.py (~130 LOC)

This mixin provides loss monitoring and regression detection logic that responds to:
- Training loss anomalies (spikes, unexpected values)
- Training loss trends (improving, stalled, degrading)
- Adaptive severe count thresholds based on training epoch

The loss monitoring affects quality checks and exploration boosts to ensure
training data quality when loss patterns indicate problems.

Usage:
    class FeedbackLoopController(LossMonitoringMixin, ...):
        pass
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from app.config.thresholds import (
    TREND_DURATION_MODERATE,
    TREND_DURATION_SEVERE,
)
from app.coordination.event_handler_utils import extract_config_key

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)


class LossMonitoringMixin:
    """Mixin for loss monitoring handling in FeedbackLoopController.

    Requires the host class to implement:
    - _get_or_create_state(config_key: str) -> FeedbackState
    - _trigger_quality_check(config_key: str, reason: str)
    - _boost_exploration_for_anomaly(config_key: str, anomaly_count: int)
    - _boost_exploration_for_stall(config_key: str, stall_epochs: int)
    - _reduce_exploration_after_improvement(config_key: str)

    Provides:
    - _on_training_loss_anomaly(event) - Handle loss spike events
    - _on_training_loss_trend(event) - Handle loss trend events (improving/stalled/degrading)
    """

    def _get_or_create_state(self, config_key: str) -> "FeedbackState":
        """Get or create state for a config. Must be implemented by host class."""
        raise NotImplementedError("Host class must implement _get_or_create_state")

    def _trigger_quality_check(self, config_key: str, reason: str) -> None:
        """Trigger a quality check. Must be implemented by host class."""
        raise NotImplementedError("Host class must implement _trigger_quality_check")

    def _boost_exploration_for_anomaly(self, config_key: str, anomaly_count: int) -> None:
        """Boost exploration for anomaly. Must be implemented by host class (ExplorationBoostMixin)."""
        raise NotImplementedError("Host class must implement _boost_exploration_for_anomaly")

    def _boost_exploration_for_stall(self, config_key: str, stall_epochs: int) -> None:
        """Boost exploration for stall. Must be implemented by host class (ExplorationBoostMixin)."""
        raise NotImplementedError("Host class must implement _boost_exploration_for_stall")

    def _reduce_exploration_after_improvement(self, config_key: str) -> None:
        """Reduce exploration after improvement. Must be implemented by host class (ExplorationBoostMixin)."""
        raise NotImplementedError("Host class must implement _reduce_exploration_after_improvement")

    def _on_training_loss_anomaly(self, event: Any) -> None:
        """Handle training loss anomaly events (Phase 8).

        Closes the critical feedback loop: training loss spike detected ->
        trigger quality check and/or exploration boost.

        Uses adaptive severe count thresholds (January 2026):
        - Early training (epochs 0-4): 5 consecutive anomalies before escalation
        - Mid training (epochs 5-14): 3 consecutive anomalies
        - Late training (epochs 15+): 2 consecutive anomalies (catch early)

        Actions:
        1. Log the anomaly for monitoring
        2. Trigger quality check on training data
        3. Optionally boost exploration if anomaly is severe
        4. Track consecutive anomalies for escalation
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            loss_value = payload.get("loss", 0.0)
            expected_loss = payload.get("expected_loss", 0.0)
            deviation = payload.get("deviation", 0.0)
            epoch = payload.get("epoch", 0)
            severity = payload.get("severity", "unknown")

            if not config_key:
                return

            # Get adaptive severe count threshold based on epoch
            from app.config.thresholds import get_severe_anomaly_count
            severe_count_threshold = get_severe_anomaly_count(epoch)

            logger.warning(
                f"[LossMonitoring] Training loss anomaly for {config_key}: "
                f"loss={loss_value:.4f} (expected={expected_loss:.4f}, "
                f"deviation={deviation:.2f}sigma), epoch={epoch}, severity={severity}, "
                f"severe_threshold={severe_count_threshold}"
            )

            # Track anomaly count for escalation
            state = self._get_or_create_state(config_key)
            if not hasattr(state, 'loss_anomaly_count'):
                state.loss_anomaly_count = 0
            state.loss_anomaly_count += 1
            state.last_loss_anomaly_time = time.time()

            # Trigger quality check on training data
            self._trigger_quality_check(config_key, reason="training_loss_anomaly")

            # If severe or consecutive anomalies exceed adaptive threshold, boost exploration
            # Early training is more permissive (5 anomalies), late training is strict (2 anomalies)
            if severity == "severe" or state.loss_anomaly_count >= severe_count_threshold:
                self._boost_exploration_for_anomaly(config_key, state.loss_anomaly_count)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[LossMonitoring] Error handling loss anomaly: {e}")

    def _on_training_loss_trend(self, event: Any) -> None:
        """Handle training loss trend events (Phase 8).

        Responds to ongoing trends in training loss (improving/stalled/degrading).

        Actions:
        - Stalled: Increase exploration diversity, consider curriculum rebalance
        - Degrading: Trigger quality check, potential pause
        - Improving: Reset anomaly counters, reduce exploration boost
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            trend = payload.get("trend", "unknown")  # improving, stalled, degrading
            current_loss = payload.get("current_loss", 0.0)
            trend_duration_epochs = payload.get("trend_duration_epochs", 0)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)

            logger.info(
                f"[LossMonitoring] Training loss trend for {config_key}: "
                f"trend={trend}, loss={current_loss:.4f}, duration={trend_duration_epochs} epochs"
            )

            if trend == "stalled":
                # Training has stagnated - boost exploration to generate diverse data
                if trend_duration_epochs >= TREND_DURATION_SEVERE:
                    self._boost_exploration_for_stall(config_key, trend_duration_epochs)

            elif trend == "plateau":
                # Dec 29, 2025: Plateau detected - boost exploration to escape
                # The PLATEAU_DETECTED event handler provides more detailed response
                self._boost_exploration_for_stall(config_key, trend_duration_epochs or 10)
                logger.info(
                    f"[LossMonitoring] Plateau trend for {config_key}, "
                    f"boosting exploration"
                )

            elif trend == "degrading":
                # Loss is getting worse - check data quality, consider rollback
                self._trigger_quality_check(config_key, reason="training_loss_degrading")
                if trend_duration_epochs >= TREND_DURATION_MODERATE:
                    logger.warning(
                        f"[LossMonitoring] Sustained loss degradation for {config_key}, "
                        f"consider training pause or rollback"
                    )

            elif trend == "improving":
                # Good news - reset anomaly tracking
                if hasattr(state, 'loss_anomaly_count'):
                    state.loss_anomaly_count = 0
                # Optionally reduce exploration boost since training is on track
                self._reduce_exploration_after_improvement(config_key)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[LossMonitoring] Error handling loss trend: {e}")


__all__ = ["LossMonitoringMixin"]
