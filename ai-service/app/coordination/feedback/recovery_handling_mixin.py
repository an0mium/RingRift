"""Recovery handling mixin for FeedbackLoopController.

Extracted from feedback_loop_controller.py (Sprint 18, Feb 2026).
Handles training rollback and timeout recovery events.

~170 LOC extracted.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_handler_utils import extract_config_key

logger = logging.getLogger(__name__)


class RecoveryHandlingMixin:
    """Mixin providing rollback and timeout recovery for FeedbackLoopController."""

    def _on_training_rollback_needed(self, event) -> None:
        """Handle TRAINING_ROLLBACK_NEEDED - training needs checkpoint rollback.

        Coordinates rollback to previous checkpoint and boosts exploration
        to escape the failure mode.

        Added: December 2025
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config = extract_config_key(payload)
        reason = payload.get("reason", "")
        epoch = payload.get("epoch", 0)

        logger.error(
            f"[FeedbackLoopController] Training rollback needed: {config} "
            f"(epoch {epoch}, reason: {reason})"
        )

        if config:
            state = self._get_or_create_state(config)
            state.consecutive_failures += 1

            # Emit exploration boost to escape failure mode
            try:
                from app.coordination.event_router import emit_exploration_boost
                from app.coordination.feedback_loop_controller import _safe_create_task

                _safe_create_task(
                    emit_exploration_boost(
                        config_key=config,
                        boost_factor=2.0,
                        reason=reason,
                        anomaly_count=state.consecutive_failures,
                        source="feedback_loop_controller",
                    ),
                    "exploration_boost_emit"
                )
            except ImportError:
                pass

    def _on_training_rollback_completed(self, event) -> None:
        """Handle TRAINING_ROLLBACK_COMPLETED - checkpoint rollback completed.

        Updates feedback state after a rollback has been completed:
        1. Resets consecutive failure count to allow fresh start
        2. Reduces exploration boost gradually (rollback fixed the issue)
        3. Emits TRAINING_INTENSITY_CHANGED to resume training with adjusted params

        Added: December 28, 2025 - Fixes orphan event (no prior subscriber)
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = extract_config_key(payload)
        model_id = payload.get("model_id", "")
        rollback_from = payload.get("rollback_from", "")
        rollback_to = payload.get("rollback_to", "")

        logger.info(
            f"[FeedbackLoopController] Training rollback completed: {config_key} "
            f"(from {rollback_from} to {rollback_to})"
        )

        if config_key:
            state = self._get_or_create_state(config_key)

            # Reset failure count - rollback gives us a fresh start
            old_failures = state.consecutive_failures
            state.consecutive_failures = max(0, state.consecutive_failures - 1)

            # Reduce exploration boost (rollback should have fixed the issue)
            if state.exploration_boost > 1.0:
                state.exploration_boost = max(1.0, state.exploration_boost * 0.7)

            # Update training intensity to resume
            old_intensity = state.training_intensity
            if old_intensity == "paused":
                state.training_intensity = "reduced"  # Cautious restart
            elif old_intensity == "reduced":
                state.training_intensity = "normal"  # Gradual increase

            logger.info(
                f"[FeedbackLoopController] Post-rollback: {config_key} "
                f"failures={old_failures}->{state.consecutive_failures}, "
                f"intensity={old_intensity}->{state.training_intensity}"
            )

            # Emit training intensity change to inform training triggers
            safe_emit_event(
                "TRAINING_INTENSITY_CHANGED",
                {
                    "config_key": config_key,
                    "old_intensity": old_intensity,
                    "new_intensity": state.training_intensity,
                    "reason": "post_rollback_recovery",
                },
                context="feedback_loop_controller",
            )

    def _on_training_timeout_reached(self, event: Any) -> None:
        """Handle TRAINING_TIMEOUT_REACHED - training job exceeded timeout threshold.

        Jan 3, 2026: Closes critical gap where TRAINING_TIMEOUT_REACHED was emitted
        at training_trigger_daemon.py:3314 but had no subscriber/handler.

        Triggers recovery actions:
        1. Boost exploration to help break potential training plateau
        2. Increase selfplay games multiplier to generate fresh data
        3. Track timeout for metrics/observability
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            timeout_hours = payload.get("timeout_hours", 0)
            training_job_id = payload.get("job_id", "")

            if not config_key:
                logger.warning("[FeedbackLoopController] TRAINING_TIMEOUT_REACHED missing config_key")
                return

            logger.warning(
                f"[FeedbackLoopController] Training timeout for {config_key} after {timeout_hours}h, "
                f"triggering recovery actions (job_id={training_job_id})"
            )

            state = self._get_or_create_state(config_key)

            # 1. Boost exploration to help break potential training plateau
            old_exploration = getattr(state, 'current_exploration_boost', 1.0)
            state.current_exploration_boost = min(old_exploration * 1.5, 2.5)
            state.exploration_boost_expires_at = time.time() + 3600  # 1 hour boost

            # 2. Increase selfplay games multiplier to generate fresh data
            old_games_mult = getattr(state, 'games_multiplier', 1.0)
            state.games_multiplier = min(old_games_mult * 1.5, 2.5)

            # 3. Track consecutive timeouts for escalation
            if not hasattr(state, 'consecutive_timeouts'):
                state.consecutive_timeouts = 0
            state.consecutive_timeouts += 1
            state.last_timeout_time = time.time()

            logger.info(
                f"[FeedbackLoopController] Timeout recovery applied to {config_key}: "
                f"exploration={old_exploration:.2f}->{state.current_exploration_boost:.2f}, "
                f"games_mult={old_games_mult:.2f}->{state.games_multiplier:.2f}, "
                f"consecutive_timeouts={state.consecutive_timeouts}"
            )

            # 4. Emit exploration boost event to propagate to SelfplayScheduler
            try:
                from app.coordination.event_router import emit_exploration_boost
                from app.coordination.feedback_loop_controller import _safe_create_task

                _safe_create_task(
                    emit_exploration_boost(
                        config_key=config_key,
                        boost_factor=state.current_exploration_boost,
                        reason="timeout_recovery",
                        source="feedback_loop_controller",
                    ),
                    "exploration_boost_timeout_recovery"
                )
            except ImportError:
                pass

        except (AttributeError, TypeError, KeyError, ValueError) as e:
            logger.error(f"[FeedbackLoopController] Error handling training timeout: {e}")
