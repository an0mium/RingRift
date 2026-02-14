"""Plateau handling mixin for FeedbackLoopController.

Extracted from feedback_loop_controller.py (Sprint 18, Feb 2026).
Handles PLATEAU_DETECTED events, curriculum advancement on velocity plateaus,
and hyperparameter search for persistent plateaus.

~330 LOC extracted.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from app.config.thresholds import ELO_PLATEAU_PER_HOUR
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_handler_utils import extract_config_key

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)


class PlateauHandlingMixin:
    """Mixin providing plateau detection and recovery for FeedbackLoopController."""

    def _on_plateau_detected(self, event: Any) -> None:
        """Handle training plateau by boosting exploration aggressively.

        Dec 29, 2025: Implements exploration boost based on plateau type.
        Jan 2026 Sprint 10: AGGRESSIVE plateau breaking for faster Elo gains.
        - Overfitting: 2.0x exploration boost + temperature increase + quality boost
        - Data limitation: 1.8x exploration boost + quality boost + request more games

        Closes feedback loop: PLATEAU_DETECTED -> exploration boost -> SelfplayScheduler
        Expected improvement: +5-10 Elo per config from faster plateau recovery.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event

            config_key = extract_config_key(payload)
            plateau_type = payload.get("plateau_type", "data_limitation")
            # Jan 2026 Sprint 10: More aggressive default boost
            exploration_boost = payload.get("exploration_boost", 1.8)
            train_val_gap = payload.get("train_val_gap", 0.0)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)

            # Track plateau count for escalation
            if not hasattr(state, "plateau_count"):
                state.plateau_count = 0
            state.plateau_count += 1
            state.last_plateau_time = time.time()

            # Jan 2026 Sprint 10: Scale boost with plateau count for persistent plateaus
            plateau_multiplier = 1.0 + (state.plateau_count * 0.2)  # 1.2x, 1.4x, 1.6x...
            plateau_multiplier = min(plateau_multiplier, 2.0)  # Cap at 2x

            # Apply exploration boost (scaled by plateau count)
            final_exploration_boost = exploration_boost * plateau_multiplier
            state.exploration_boost = final_exploration_boost
            state.exploration_boost_expires_at = time.time() + 3600  # 1 hour

            if plateau_type == "overfitting":
                # High train/val gap indicates overfitting - aggressive diversity boost
                state.selfplay_temperature_boost = 1.3 + (state.plateau_count * 0.1)
                state.selfplay_temperature_boost = min(state.selfplay_temperature_boost, 1.8)
                logger.info(
                    f"[FeedbackLoopController] Plateau (overfitting) for {config_key}: "
                    f"exploration_boost={final_exploration_boost:.2f}, "
                    f"temp_boost={state.selfplay_temperature_boost:.2f}, "
                    f"train_val_gap={train_val_gap:.4f}, plateau_count={state.plateau_count}"
                )
            else:
                # Data-limited plateau - request more high-quality games
                state.games_multiplier = 1.5 + (state.plateau_count * 0.2)
                state.games_multiplier = min(state.games_multiplier, 2.5)
                logger.info(
                    f"[FeedbackLoopController] Plateau (data limited) for {config_key}: "
                    f"exploration_boost={final_exploration_boost:.2f}, "
                    f"games_multiplier={state.games_multiplier:.2f}, "
                    f"plateau_count={state.plateau_count}"
                )

            # Emit EXPLORATION_BOOST event for SelfplayScheduler
            try:
                from app.coordination.event_router import emit_exploration_boost

                from app.coordination.feedback_loop_controller import _safe_create_task

                _safe_create_task(
                    emit_exploration_boost(
                        config_key=config_key,
                        boost_factor=final_exploration_boost,
                        reason="plateau",
                        anomaly_count=state.plateau_count,  # Signal plateau severity
                        source="FeedbackLoopController",
                    ),
                    context=f"emit_exploration_boost:plateau:{config_key}",
                )
                logger.debug(
                    f"[FeedbackLoopController] Emitted EXPLORATION_BOOST event for {config_key}"
                )
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.warning(f"[FeedbackLoopController] Failed to emit EXPLORATION_BOOST: {e}")

            # Jan 2026 Sprint 10: Emit TRAINING_BLOCKED_BY_QUALITY to trigger quality boost
            # This increases Gumbel budget in SelfplayScheduler for higher quality games
            safe_emit_event(
                "TRAINING_BLOCKED_BY_QUALITY",
                {
                    "config_key": config_key,
                    "quality_score": 0.5 - (state.plateau_count * 0.1),  # Lower score with more plateaus
                    "threshold": 0.7,
                    "reason": f"plateau_{plateau_type}",
                },
                context="FeedbackLoopController",
                log_after=f"Triggered quality boost for {config_key} (plateau)",
            )

            # If repeated plateaus, consider triggering hyperparameter search
            if state.plateau_count >= 3:
                logger.warning(
                    f"[FeedbackLoopController] Repeated plateaus ({state.plateau_count}) "
                    f"for {config_key}, triggering aggressive hyperparameter search"
                )
                # Jan 2026 Sprint 10: Emit hyperparameter update request
                self._trigger_hyperparameter_search(config_key, state)

            # Jan 2026 Sprint 10: Start curriculum advancement earlier (after 1 plateau)
            # to provide harder training data and break out of the plateau faster
            if state.plateau_count >= 1:
                self._advance_curriculum_on_velocity_plateau(config_key, state)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.warning(f"[FeedbackLoopController] Error handling plateau: {e}")

    def _advance_curriculum_on_velocity_plateau(
        self, config_key: str, state: "FeedbackState"
    ) -> None:
        """Advance curriculum tier when velocity indicates persistent plateau.

        Dec 29, 2025: Implements velocity-based curriculum advancement to break
        out of training plateaus. When Elo velocity is low and we've had
        repeated plateaus, we advance to a harder curriculum tier.

        Curriculum tiers:
        - 0: Beginner (basic positions, weaker opponents)
        - 1: Intermediate (moderate complexity)
        - 2: Advanced (complex positions, stronger opponents)
        - 3: Expert (most challenging positions)
        """
        try:
            # Check velocity - only advance if we're truly plateauing
            velocity = state.elo_velocity
            is_low_velocity = velocity < ELO_PLATEAU_PER_HOUR and len(state.elo_history or []) >= 3

            if not is_low_velocity:
                logger.debug(
                    f"[FeedbackLoopController] Velocity {velocity:.1f} Elo/hr not low enough "
                    f"for curriculum advancement ({config_key})"
                )
                return

            # Check cooldown - don't advance too frequently (min 2 hours between advances)
            cooldown_seconds = 7200  # 2 hours
            time_since_advance = time.time() - state.curriculum_last_advanced
            if time_since_advance < cooldown_seconds:
                logger.debug(
                    f"[FeedbackLoopController] Curriculum advancement on cooldown "
                    f"({time_since_advance:.0f}s < {cooldown_seconds}s) for {config_key}"
                )
                return

            # Check max tier - don't exceed expert level
            max_tier = 3
            if state.curriculum_tier >= max_tier:
                logger.debug(
                    f"[FeedbackLoopController] Already at max curriculum tier "
                    f"({state.curriculum_tier}) for {config_key}"
                )
                return

            # Advance the curriculum tier
            old_tier = state.curriculum_tier
            new_tier = old_tier + 1
            state.curriculum_tier = new_tier
            state.curriculum_last_advanced = time.time()

            # Reset plateau count after advancement
            state.plateau_count = 0

            tier_names = ["Beginner", "Intermediate", "Advanced", "Expert"]
            logger.info(
                f"[FeedbackLoopController] Curriculum advancement for {config_key}: "
                f"{tier_names[old_tier]} -> {tier_names[new_tier]} "
                f"(velocity={velocity:.1f} Elo/hr, plateaus={state.plateau_count})"
            )

            # Emit CURRICULUM_ADVANCED event for downstream consumers
            from app.coordination.event_router import safe_emit_event as router_safe_emit

            router_safe_emit(
                "CURRICULUM_ADVANCED",
                {
                    "config_key": config_key,
                    "old_tier": old_tier,
                    "new_tier": new_tier,
                    "trigger": "velocity_plateau",
                    "velocity": velocity,
                    "plateau_count": state.plateau_count,
                    "source": "FeedbackLoopController",
                },
                log_after=f"[FeedbackLoopController] Emitted CURRICULUM_ADVANCED for {config_key}",
                log_level=logging.DEBUG,
                context="FeedbackLoopController._check_velocity_plateau",
            )

            # Also notify CurriculumFeedback to adjust weights
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                feedback = get_curriculum_feedback()
                if feedback and hasattr(feedback, "set_difficulty_tier"):
                    feedback.set_difficulty_tier(config_key, new_tier)
                    logger.debug(
                        f"[FeedbackLoopController] Updated CurriculumFeedback tier for {config_key}"
                    )
            except ImportError:
                logger.debug("[FeedbackLoopController] curriculum_feedback not available")
            except (AttributeError, TypeError) as cf_err:
                logger.debug(f"[FeedbackLoopController] CurriculumFeedback error: {cf_err}")

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.warning(
                f"[FeedbackLoopController] Error advancing curriculum for {config_key}: {e}"
            )

    def _trigger_hyperparameter_search(self, config_key: str, state: "FeedbackState") -> None:
        """Trigger hyperparameter search for persistently plateaued configs.

        Jan 2026 Sprint 10: When a config has 3+ consecutive plateaus, trigger
        aggressive hyperparameter adjustments to break out of local minima.

        Adjustments:
        - Increase learning rate by 50% temporarily (shake out of local minimum)
        - Reduce batch size by 25% (more gradient updates)
        - Enable cosine annealing if not already active
        - Emit HYPERPARAMETER_UPDATED event for downstream consumers
        """
        try:
            # Calculate hyperparameter adjustments based on plateau severity
            plateau_count = getattr(state, "plateau_count", 1)
            lr_boost = 1.0 + (plateau_count * 0.15)  # 1.15x, 1.30x, 1.45x per plateau
            lr_boost = min(lr_boost, 2.0)  # Cap at 2x

            batch_reduction = max(0.5, 1.0 - (plateau_count * 0.08))  # 0.92, 0.84, 0.76...

            logger.info(
                f"[FeedbackLoopController] Hyperparameter search for {config_key}: "
                f"lr_boost={lr_boost:.2f}x, batch_reduction={batch_reduction:.2f}x, "
                f"plateau_count={plateau_count}"
            )

            # Emit HYPERPARAMETER_UPDATED event for training to pick up
            safe_emit_event(
                "HYPERPARAMETER_UPDATED",
                {
                    "config_key": config_key,
                    "learning_rate_multiplier": lr_boost,
                    "batch_size_multiplier": batch_reduction,
                    "enable_cosine_annealing": True,
                    "reason": f"plateau_count_{plateau_count}",
                    "source": "FeedbackLoopController",
                },
                context="FeedbackLoopController",
                log_after=f"Emitted HYPERPARAMETER_UPDATED for {config_key}",
            )

            # Also update state to track that we triggered hyperparam search
            state.last_hyperparam_search = time.time()
            state.hyperparam_search_count = getattr(state, "hyperparam_search_count", 0) + 1

        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(
                f"[FeedbackLoopController] Error triggering hyperparameter search: {e}"
            )
