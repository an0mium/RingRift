"""Selfplay feedback mixin for FeedbackLoopController.

Sprint 17.9 (Jan 16, 2026): Extracted from feedback_loop_controller.py (~230 LOC)

This mixin provides selfplay-related feedback logic that:
- Handles SELFPLAY_COMPLETE events with quality assessment
- Tracks selfplay rate changes for curriculum coordination
- Handles CPU_PIPELINE_JOB_COMPLETED for Vast.ai CPU selfplay
- Tracks database creation for freshness monitoring

The selfplay feedback loop closes the training data cycle:
SELFPLAY_COMPLETE -> quality assessment -> training intensity -> curriculum weight

Usage:
    class FeedbackLoopController(SelfplayFeedbackMixin, ...):
        pass
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from app.coordination.event_handler_utils import extract_config_key

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)

# Rate change thresholds for curriculum coordination
RATE_CHANGE_SIGNIFICANT_PERCENT = 20  # Trigger curriculum update if rate changes by 20%+
CURRICULUM_WEIGHT_ADJUSTMENT_UP = 0.05  # When rate decreases (struggling)
CURRICULUM_WEIGHT_ADJUSTMENT_DOWN = -0.03  # When rate increases (improving)


class SelfplayFeedbackMixin:
    """Mixin for selfplay feedback handling in FeedbackLoopController.

    Requires the host class to implement (typically via HandlerBase or other mixins):
    - _get_or_create_state(config_key: str) -> FeedbackState
    - _assess_selfplay_quality_async(db_path, games_count) -> quality score (from QualityFeedbackMixin)
    - _update_training_intensity(config_key, quality_score) (from QualityFeedbackMixin)
    - _update_curriculum_weight_from_selfplay(config_key, quality_score) (from QualityFeedbackMixin)
    - _signal_training_ready(config_key, quality_score) (from QualityFeedbackMixin)
    - _emit_quality_degraded(config_key, quality_score, threshold, previous) (from QualityFeedbackMixin)
    - _rate_history: dict[str, list] - Rate change history per config

    Provides:
    - _on_selfplay_complete(event) - Handle SELFPLAY_COMPLETE events
    - _on_selfplay_rate_changed(event) - Handle SELFPLAY_RATE_CHANGED events
    - _on_cpu_pipeline_job_completed(event) - Handle CPU_PIPELINE_JOB_COMPLETED events
    - _on_database_created(event) - Handle DATABASE_CREATED events

    Note: This mixin depends on QualityFeedbackMixin being in the MRO before it.
    """

    # Ensure _rate_history exists
    _rate_history: dict = {}

    async def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion.

        Actions:
        1. Assess data quality
        2. Update training intensity based on quality
        3. Signal training readiness if quality is sufficient
        4. Track engine mode for bandit feedback (Dec 29 2025)

        Sprint 17.9: Converted to async to avoid blocking event loop during
        SQLite quality assessment.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            games_count = payload.get("games_count", 0)
            db_path = payload.get("db_path", "")
            engine_mode = payload.get("engine_mode", "gumbel-mcts")  # Dec 29 2025

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_selfplay_time = time.time()

            # Dec 29 2025: Track engine mode for bandit feedback
            state.last_selfplay_engine = engine_mode
            state.last_selfplay_games += games_count  # Accumulate across batches

            # Assess data quality (Sprint 17.9: now async to avoid blocking event loop)
            previous_quality = state.last_selfplay_quality
            quality_score = await self._assess_selfplay_quality_async(db_path, games_count)
            state.last_selfplay_quality = quality_score

            logger.info(
                f"[SelfplayFeedback] Selfplay complete for {config_key}: "
                f"{games_count} games, quality={quality_score:.2f}"
            )

            # Phase 5 (Dec 2025): Emit QUALITY_DEGRADED event when quality drops below threshold
            # December 2025: Use centralized threshold from config
            try:
                from app.config.thresholds import MEDIUM_QUALITY_THRESHOLD
                quality_threshold = MEDIUM_QUALITY_THRESHOLD
            except ImportError:
                quality_threshold = 0.6  # Fallback default
            if quality_score < quality_threshold:
                self._emit_quality_degraded(config_key, quality_score, quality_threshold, previous_quality)

            # Update training intensity based on quality
            self._update_training_intensity(config_key, quality_score)

            # Gap 4 fix (Dec 2025): Update curriculum weight based on selfplay quality
            self._update_curriculum_weight_from_selfplay(config_key, quality_score)

            # Signal training readiness if quality is good
            if quality_score >= quality_threshold:
                self._signal_training_ready(config_key, quality_score)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[SelfplayFeedback] Error handling selfplay complete: {e}")

    def _on_selfplay_rate_changed(self, event: Any) -> None:
        """Handle selfplay rate change events (Phase 23.1).

        Tracks rate changes for monitoring and logs significant adjustments.
        This enables visibility into how Elo momentum affects selfplay rates.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            old_rate = payload.get("old_rate", 1.0)
            new_rate = payload.get("new_rate", 1.0)
            change_percent = payload.get("change_percent", 0.0)
            momentum_state = payload.get("momentum_state", "unknown")
            timestamp = payload.get("timestamp", time.time())

            if not config_key:
                return

            # Track in rate history
            if config_key not in self._rate_history:
                self._rate_history[config_key] = []

            self._rate_history[config_key].append({
                "rate": new_rate,
                "old_rate": old_rate,
                "change_percent": change_percent,
                "momentum_state": momentum_state,
                "timestamp": timestamp,
            })

            # Keep bounded history (last 100 changes per config)
            if len(self._rate_history[config_key]) > 100:
                self._rate_history[config_key] = self._rate_history[config_key][-100:]

            logger.info(
                f"[SelfplayFeedback] Selfplay rate for {config_key}: "
                f"{old_rate:.2f}x → {new_rate:.2f}x ({change_percent:+.0f}%), "
                f"momentum={momentum_state}"
            )

            # Gap 2 fix (Dec 2025): Sync curriculum weight when rate changes significantly
            # When Elo momentum drives big rate changes, adjust curriculum priority accordingly
            if abs(change_percent) >= RATE_CHANGE_SIGNIFICANT_PERCENT:
                try:
                    from app.training.curriculum_feedback import get_curriculum_feedback

                    feedback = get_curriculum_feedback()
                    state = self._get_or_create_state(config_key)

                    # Increasing rate = model improving = can reduce curriculum weight slightly
                    # Decreasing rate = model struggling = increase curriculum weight for more focus
                    if change_percent > 0:  # Rate increased
                        adjustment = CURRICULUM_WEIGHT_ADJUSTMENT_DOWN
                    else:  # Rate decreased
                        adjustment = CURRICULUM_WEIGHT_ADJUSTMENT_UP

                    current_weight = feedback._current_weights.get(config_key, 1.0)
                    new_weight = max(
                        feedback.weight_min,
                        min(feedback.weight_max, current_weight + adjustment)
                    )

                    if abs(new_weight - current_weight) > 0.01:
                        feedback._current_weights[config_key] = new_weight
                        state.current_curriculum_weight = new_weight

                        logger.info(
                            f"[SelfplayFeedback] Curriculum weight adjusted for {config_key}: "
                            f"{current_weight:.2f} → {new_weight:.2f} (rate change {change_percent:+.0f}%)"
                        )
                except ImportError:
                    logger.debug("[SelfplayFeedback] curriculum_feedback not available")
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"[SelfplayFeedback] Failed to adjust curriculum: {e}")

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug(f"[SelfplayFeedback] Error handling rate change: {e}")

    async def _on_cpu_pipeline_job_completed(self, event: Any) -> None:
        """Handle CPU_PIPELINE_JOB_COMPLETED from Vast.ai CPU selfplay jobs.

        December 2025: Closes integration gap - CPU selfplay completions now trigger
        downstream pipeline actions (training readiness, quality assessment, etc.).

        This event is emitted by VastCpuPipelineDaemon when CPU-based selfplay jobs
        complete on Vast.ai nodes. We treat these like GPU selfplay completions.

        Sprint 17.9: Converted to async to avoid blocking event loop during
        SQLite quality assessment.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            games_count = payload.get("games_count", 0) or payload.get("games_generated", 0)
            db_path = payload.get("db_path", "")
            node_id = payload.get("node_id", "")
            job_id = payload.get("job_id", "")

            if not config_key:
                logger.debug("[SelfplayFeedback] CPU pipeline job missing config_key")
                return

            state = self._get_or_create_state(config_key)
            state.last_selfplay_time = time.time()

            logger.info(
                f"[SelfplayFeedback] CPU pipeline job complete for {config_key}: "
                f"{games_count} games from node={node_id}, job={job_id}"
            )

            # Assess data quality (Sprint 17.9: now async to avoid blocking event loop)
            quality_score = await self._assess_selfplay_quality_async(db_path, games_count)
            state.last_selfplay_quality = quality_score

            # Update training intensity based on quality
            self._update_training_intensity(config_key, quality_score)

            # Signal training readiness if quality is good
            try:
                from app.config.thresholds import MEDIUM_QUALITY_THRESHOLD
                quality_threshold = MEDIUM_QUALITY_THRESHOLD
            except ImportError:
                quality_threshold = 0.6

            if quality_score >= quality_threshold:
                self._signal_training_ready(config_key, quality_score)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[SelfplayFeedback] Error handling CPU pipeline complete: {e}")

    def _on_database_created(self, event: Any) -> None:
        """Handle DATABASE_CREATED event (December 2025 - Phase 4A.3).

        Provides early awareness of new databases for feedback loop coordination.
        This handler primarily logs and tracks database creation for monitoring.
        The main processing happens in DataPipelineOrchestrator.

        Actions:
        1. Log database creation for visibility
        2. Track creation timestamps for freshness monitoring
        3. Update state for potential training triggers
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            db_path = payload.get("db_path", "")
            config_key = extract_config_key(payload)
            node_id = payload.get("node_id", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)

            # Track database creation time for freshness awareness
            if not hasattr(state, 'last_database_created'):
                state.last_database_created = 0.0
            state.last_database_created = time.time()

            logger.info(
                f"[SelfplayFeedback] New database created for {config_key}: "
                f"{db_path} on {node_id}"
            )

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug(f"[SelfplayFeedback] Error handling database_created: {e}")


__all__ = ["SelfplayFeedbackMixin"]
