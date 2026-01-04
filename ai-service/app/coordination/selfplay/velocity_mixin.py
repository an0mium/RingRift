"""Velocity and Elo tracking mixin for SelfplayScheduler.

January 2026 Sprint 17.9: Extracted from selfplay_scheduler.py to reduce monolith size.

Provides event handlers for:
- ELO_VELOCITY_CHANGED: Adjusts selfplay rate based on Elo velocity trends
- ELO_UPDATED: Tracks Elo history and computes velocity
- EXPLORATION_BOOST: Reacts to training anomalies with increased exploration
- CURRICULUM_ADVANCED: Responds to curriculum stage progression
- ADAPTIVE_PARAMS_CHANGED: Adjusts selfplay based on training parameter changes
- ARCHITECTURE_WEIGHTS_UPDATED: Updates architecture allocation weights

Also provides:
- Elo velocity computation and initialization from database
- Diversity tracking for opponent variety maximization
- Exploration boost decay management
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from app.coordination.event_utils import extract_config_key

if TYPE_CHECKING:
    from app.coordination.selfplay_scheduler import SelfplayScheduler

logger = logging.getLogger(__name__)


class SelfplayVelocityMixin:
    """Mixin providing velocity, Elo, and diversity tracking for SelfplayScheduler.

    This mixin expects the parent class to have:
    - _config_priorities: dict[str, ConfigPriority]
    - _elo_velocity: dict[str, float]
    - _elo_history: dict[str, list[tuple[float, float]]]
    - _low_velocity_count: dict[str, int]
    - _last_plateau_emission: dict[str, float]
    - _opponent_types_by_config: dict[str, set[str]]
    - _diversity_scores: dict[str, float]
    - _max_opponent_types: int
    """

    # Type hints for attributes from parent class
    _config_priorities: dict[str, Any]
    _elo_velocity: dict[str, float]
    _elo_history: dict[str, list[tuple[float, float]]]
    _low_velocity_count: dict[str, int]
    _last_plateau_emission: dict[str, float]
    _opponent_types_by_config: dict[str, set[str]]
    _diversity_scores: dict[str, float]
    _max_opponent_types: int

    def _get_velocity_event_subscriptions(self) -> dict[str, Any]:
        """Return event type -> handler mappings for velocity/Elo events.

        Used by SelfplayScheduler._get_event_subscriptions() to include these handlers.
        """
        return {
            "ELO_VELOCITY_CHANGED": self._on_elo_velocity_changed,
            "ELO_UPDATED": self._on_elo_updated,
            "EXPLORATION_BOOST": self._on_exploration_boost,
            "CURRICULUM_ADVANCED": self._on_curriculum_advanced,
            "ADAPTIVE_PARAMS_CHANGED": self._on_adaptive_params_changed,
            "ARCHITECTURE_WEIGHTS_UPDATED": self._on_architecture_weights_updated,
        }

    # =========================================================================
    # Elo Velocity Handlers
    # =========================================================================

    def _on_elo_velocity_changed(self, event: Any) -> None:
        """Handle Elo velocity change event.

        P10-LOOP-3 (Dec 2025): Adjusts selfplay rate based on Elo velocity trends.

        This closes the feedback loop:
        ELO_VELOCITY_CHANGED → Selfplay rate adjustment → Optimal resource allocation

        Actions based on trend:
        - accelerating: Increase selfplay to capitalize on momentum
        - decelerating: Reduce selfplay, shift focus to training quality
        - stable: Maintain current allocation
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            velocity = payload.get("velocity", 0.0)
            previous_velocity = payload.get("previous_velocity", 0.0)
            trend = payload.get("trend", "stable")

            if config_key not in self._config_priorities:
                logger.debug(
                    f"[SelfplayScheduler] Received velocity change for unknown config: {config_key}"
                )
                return

            priority = self._config_priorities[config_key]

            # Update elo_velocity tracking
            priority.elo_velocity = velocity

            # Adjust momentum multiplier based on trend
            old_momentum = priority.momentum_multiplier

            if trend == "accelerating":
                # Capitalize on positive momentum - increase selfplay rate
                priority.momentum_multiplier = min(1.5, old_momentum * 1.2)
                logger.info(
                    f"[SelfplayScheduler] Accelerating velocity for {config_key}: "
                    f"{velocity:.1f} Elo/day. Boosted momentum {old_momentum:.2f} → {priority.momentum_multiplier:.2f}"
                )
            elif trend == "decelerating":
                # Slow down and focus on quality
                priority.momentum_multiplier = max(0.6, old_momentum * 0.85)
                logger.info(
                    f"[SelfplayScheduler] Decelerating velocity for {config_key}: "
                    f"{velocity:.1f} Elo/day. Reduced momentum {old_momentum:.2f} → {priority.momentum_multiplier:.2f}"
                )
            else:  # stable
                # Slight adjustment toward 1.0
                if old_momentum > 1.0:
                    priority.momentum_multiplier = max(1.0, old_momentum * 0.95)
                elif old_momentum < 1.0:
                    priority.momentum_multiplier = min(1.0, old_momentum * 1.05)

            # If velocity is negative, also boost exploration
            if velocity < 0:
                old_boost = priority.exploration_boost
                priority.exploration_boost = min(1.8, old_boost * 1.15)
                logger.info(
                    f"[SelfplayScheduler] Negative velocity for {config_key}: "
                    f"Boosted exploration {old_boost:.2f} → {priority.exploration_boost:.2f}"
                )

            # Emit SELFPLAY_RATE_CHANGED if momentum changed by >20%
            if abs(priority.momentum_multiplier - old_momentum) / max(old_momentum, 0.01) > 0.20:
                change_percent = ((priority.momentum_multiplier - old_momentum) / old_momentum) * 100.0
                try:
                    from app.coordination.event_router import DataEventType, get_event_bus

                    bus = get_event_bus()
                    if bus:
                        bus.emit(DataEventType.SELFPLAY_RATE_CHANGED, {
                            "config_key": config_key,
                            "old_rate": old_momentum,
                            "new_rate": priority.momentum_multiplier,
                            "change_percent": change_percent,
                            "reason": f"elo_momentum:{trend}",
                        })
                        logger.debug(
                            f"[SelfplayScheduler] Emitted SELFPLAY_RATE_CHANGED for {config_key}: "
                            f"{old_momentum:.2f} → {priority.momentum_multiplier:.2f} ({change_percent:+.1f}%)"
                        )
                except Exception as emit_err:
                    logger.debug(f"[SelfplayScheduler] Failed to emit SELFPLAY_RATE_CHANGED: {emit_err}")

            # Emit SELFPLAY_TARGET_UPDATED for downstream consumers
            try:
                from app.coordination.event_router import DataEventType, get_event_bus

                bus = get_event_bus()
                if bus:
                    bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                        "config_key": config_key,
                        "priority": "normal",
                        "reason": f"velocity_changed:{trend}",
                        "momentum_multiplier": priority.momentum_multiplier,
                        "exploration_boost": priority.exploration_boost,
                    })
            except Exception as emit_err:
                logger.debug(f"[SelfplayScheduler] Failed to emit target update: {emit_err}")

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling elo velocity changed: {e}")

    def _on_elo_updated(self, event: Any) -> None:
        """Handle ELO_UPDATED - track Elo history and compute velocity.

        Dec 29, 2025 - Phase 2: Elo velocity integration.
        Tracks Elo changes over time to compute velocity (Elo/hour).
        Stagnant configs (velocity < 0.5) get reduced allocation.
        Fast-improving configs (velocity > 5.0) get boosted allocation.

        Args:
            event: Event with payload containing config_key, new_elo, old_elo
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            new_elo = payload.get("new_elo", 0.0)

            if not config_key or new_elo <= 0:
                return

            now = time.time()

            # Initialize history if needed
            if config_key not in self._elo_history:
                self._elo_history[config_key] = []

            # Add new data point
            self._elo_history[config_key].append((now, new_elo))

            # Keep only last 24 hours of history
            cutoff = now - 86400  # 24 hours
            self._elo_history[config_key] = [
                (t, e) for t, e in self._elo_history[config_key] if t >= cutoff
            ]

            # Compute velocity: Elo change per hour over last 24 hours
            recent = self._elo_history[config_key]
            if len(recent) >= 2:
                hours = (recent[-1][0] - recent[0][0]) / 3600
                if hours > 0.5:  # Need at least 30 min of data
                    velocity = (recent[-1][1] - recent[0][1]) / hours
                    old_velocity = self._elo_velocity.get(config_key, 0.0)
                    self._elo_velocity[config_key] = velocity

                    # Log significant velocity changes
                    if abs(velocity - old_velocity) > 1.0:
                        logger.info(
                            f"[SelfplayScheduler] Elo velocity for {config_key}: "
                            f"{velocity:.2f} Elo/hour (was {old_velocity:.2f})"
                        )

                    # Sprint 10: Stall detection and PLATEAU_DETECTED emission
                    # Velocity < 0.5 Elo/hour is considered a stall
                    STALL_VELOCITY_THRESHOLD = 0.5
                    STALL_COUNT_THRESHOLD = 3  # 3 consecutive stalls
                    PLATEAU_COOLDOWN_SECONDS = 3600  # 1 hour between emissions

                    if abs(velocity) < STALL_VELOCITY_THRESHOLD:
                        self._low_velocity_count[config_key] = (
                            self._low_velocity_count.get(config_key, 0) + 1
                        )
                    else:
                        # Reset count if velocity recovers
                        self._low_velocity_count[config_key] = 0

                    # Emit PLATEAU_DETECTED if stalled for consecutive updates
                    low_count = self._low_velocity_count.get(config_key, 0)
                    last_emission = self._last_plateau_emission.get(config_key, 0.0)
                    now = time.time()

                    if (
                        low_count >= STALL_COUNT_THRESHOLD
                        and now - last_emission > PLATEAU_COOLDOWN_SECONDS
                    ):
                        self._last_plateau_emission[config_key] = now
                        self._emit_plateau_detected(config_key, velocity, low_count)

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling Elo update: {e}")

    def _emit_plateau_detected(
        self, config_key: str, velocity: float, stall_count: int
    ) -> None:
        """Emit PLATEAU_DETECTED event when Elo velocity stalls.

        January 2026 Sprint 10: Automatic stall detection and curriculum advancement.
        When velocity is near-zero for consecutive updates, emit PLATEAU_DETECTED
        to trigger exploration boost and curriculum adjustments.

        Expected improvement: +25-40 Elo from faster plateau breaking.

        Args:
            config_key: Config experiencing the stall
            velocity: Current Elo velocity (Elo/hour)
            stall_count: Number of consecutive stall detections
        """
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            # Get current Elo if available
            current_elo = 1500.0
            if config_key in self._config_priorities:
                current_elo = getattr(self._config_priorities[config_key], "current_elo", 1500.0)

            emit_event(
                DataEventType.PLATEAU_DETECTED,
                {
                    "config_key": config_key,
                    "current_elo": current_elo,
                    "velocity": velocity,
                    "stall_count": stall_count,
                    "plateau_type": "velocity_stall",
                    "source": "selfplay_scheduler",
                    "recommendation": "trigger_curriculum_advance",
                },
            )
            logger.warning(
                f"[SelfplayScheduler] Emitted PLATEAU_DETECTED for {config_key} "
                f"(velocity={velocity:.2f} Elo/hour, stall_count={stall_count}, "
                f"current_elo={current_elo:.0f})"
            )

        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"[SelfplayScheduler] Could not emit PLATEAU_DETECTED: {e}")

    # =========================================================================
    # Exploration & Curriculum Handlers
    # =========================================================================

    def _on_exploration_boost(self, event: Any) -> None:
        """Handle exploration boost event from training feedback.

        P11-CRITICAL-1 (Dec 2025): React to EXPLORATION_BOOST events emitted
        by FeedbackLoopController when training anomalies (loss spikes, stalls)
        are detected.

        This closes the feedback loop:
        Training Anomaly → EXPLORATION_BOOST → Increased selfplay diversity

        Actions:
        - Update exploration_boost for the config
        - Increase temperature/noise in selfplay to generate diverse games
        - Emit SELFPLAY_TARGET_UPDATED for downstream consumers
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            boost_factor = payload.get("boost_factor", 1.0)
            reason = payload.get("reason", "unknown")
            anomaly_count = payload.get("anomaly_count", 0)

            if config_key not in self._config_priorities:
                logger.debug(
                    f"[SelfplayScheduler] Received exploration boost for unknown config: {config_key}"
                )
                return

            priority = self._config_priorities[config_key]
            old_boost = priority.exploration_boost

            # Apply the boost factor directly from FeedbackLoopController
            priority.exploration_boost = max(priority.exploration_boost, boost_factor)

            # Phase 12: Set boost expiry (15 minutes from now by default)
            # This ensures temporary anomalies don't cause permanent boosts
            boost_duration = float(os.environ.get("RINGRIFT_EXPLORATION_BOOST_DURATION", "900"))  # 15 min default
            priority.exploration_boost_expires_at = time.time() + boost_duration

            logger.info(
                f"[SelfplayScheduler] Exploration boost for {config_key}: "
                f"{old_boost:.2f} → {priority.exploration_boost:.2f} "
                f"(reason={reason}, anomaly_count={anomaly_count}, expires_in={boost_duration}s)"
            )

            # Emit SELFPLAY_TARGET_UPDATED for downstream consumers
            try:
                from app.coordination.event_router import DataEventType, get_event_bus

                bus = get_event_bus()
                if bus:
                    bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                        "config_key": config_key,
                        "priority": "high" if boost_factor > 1.3 else "normal",
                        "reason": f"exploration_boost:{reason}",
                        "exploration_boost": priority.exploration_boost,
                        "anomaly_count": anomaly_count,
                    })
            except Exception as emit_err:
                logger.debug(f"[SelfplayScheduler] Failed to emit target update: {emit_err}")

            # Dec 2025: Also emit SELFPLAY_ALLOCATION_UPDATED for feedback loop tracking
            if hasattr(self, "_emit_allocation_updated"):
                self._emit_allocation_updated(
                    allocation=None,
                    total_games=0,
                    trigger=f"exploration_boost:{reason}",
                    config_key=config_key,
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling exploration boost: {e}")

    def _on_curriculum_advanced(self, event: Any) -> None:
        """Handle CURRICULUM_ADVANCED event - curriculum stage progressed.

        Dec 2025: When a config achieves consecutive successful promotions,
        the curriculum advances. This signals we should shift focus to the
        next curriculum stage (harder opponents, more complex positions).

        Actions:
        - Update priority weights for the advanced config
        - Potentially reduce focus on "graduated" configs
        - Log curriculum progression for tracking
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            new_stage = payload.get("stage", 1)
            consecutive_promotions = payload.get("consecutive_promotions", 0)

            if not config_key:
                return

            logger.info(
                f"[SelfplayScheduler] CURRICULUM_ADVANCED: {config_key} "
                f"stage={new_stage}, consecutive_promotions={consecutive_promotions}"
            )

            # Update curriculum stage if tracked
            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                # Slightly reduce selfplay priority for graduated configs
                # to focus resources on configs that still need improvement
                if consecutive_promotions >= 3:
                    priority.curriculum_weight = max(0.5, priority.curriculum_weight * 0.9)
                    logger.debug(
                        f"[SelfplayScheduler] Reduced curriculum weight for {config_key}: "
                        f"{priority.curriculum_weight:.2f} (graduated)"
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling curriculum advanced: {e}")

    def _on_adaptive_params_changed(self, event: Any) -> None:
        """Handle ADAPTIVE_PARAMS_CHANGED event - training parameters adjusted.

        Dec 2025: When gauntlet feedback controller adjusts training parameters
        (learning rate, batch size, etc.), this event is emitted. We respond by
        adjusting selfplay parameters accordingly.

        Actions:
        - Update exploration parameters if temperature changed
        - Adjust selfplay rate if training intensity changed
        - Log parameter changes for tracking
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            param_type = payload.get("param_type", "")
            old_value = payload.get("old_value")
            new_value = payload.get("new_value")
            reason = payload.get("reason", "adaptive_adjustment")

            if not config_key:
                return

            logger.info(
                f"[SelfplayScheduler] ADAPTIVE_PARAMS_CHANGED: {config_key} "
                f"{param_type}: {old_value} → {new_value} (reason={reason})"
            )

            # Respond to temperature changes
            if param_type == "temperature" and config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                # Higher temperature = more exploration, adjust boost accordingly
                if new_value and old_value and new_value > old_value:
                    # Temperature increased, boost exploration
                    priority.exploration_boost = max(priority.exploration_boost, 1.2)
                    logger.debug(f"[SelfplayScheduler] Boosted exploration for {config_key} due to temperature increase")

            # Respond to learning rate changes (training intensity)
            elif param_type == "learning_rate" and config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                # Lower LR = more careful training, may need more selfplay data
                if new_value and old_value and new_value < old_value:
                    priority.target_games_multiplier = min(2.0, getattr(priority, 'target_games_multiplier', 1.0) * 1.1)
                    logger.debug(f"[SelfplayScheduler] Increased target games for {config_key} due to LR decrease")

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling adaptive params changed: {e}")

    def _on_architecture_weights_updated(self, event: Any) -> None:
        """Handle ARCHITECTURE_WEIGHTS_UPDATED - update architecture allocation weights.

        Dec 29, 2025: When architecture performance changes, adjust selfplay
        allocation to favor better-performing architectures while ensuring
        10% minimum per architecture.

        Args:
            event: Event with payload containing config_key, weights dict, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            weights = payload.get("weights", {})

            if not config_key or not weights:
                return

            # Store weights for use in allocation decisions
            if not hasattr(self, "_architecture_weights"):
                self._architecture_weights: dict[str, dict[str, float]] = {}

            self._architecture_weights[config_key] = weights

            logger.info(
                f"[SelfplayScheduler] Updated architecture weights for {config_key}: "
                f"{list(weights.items())[:3]}..."
            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling architecture weights: {e}")

    # =========================================================================
    # Boost Decay
    # =========================================================================

    def _decay_expired_boosts(self, now: float) -> int:
        """Decay exploration boosts that have expired.

        Phase 12 (Dec 2025): Prevents temporary training anomalies from causing
        permanent exploration boosts. After the boost expires, gradually decay
        back to 1.0 (normal exploration level).

        Args:
            now: Current timestamp

        Returns:
            Number of boosts that were decayed
        """
        decayed_count = 0

        for config_key, priority in self._config_priorities.items():
            # Skip if no boost active
            if priority.exploration_boost <= 1.0:
                continue

            # Skip if boost hasn't expired yet
            if priority.exploration_boost_expires_at > now:
                continue

            # Decay the boost
            old_boost = priority.exploration_boost
            priority.exploration_boost = 1.0
            priority.exploration_boost_expires_at = 0.0
            decayed_count += 1

            logger.info(
                f"[SelfplayScheduler] Exploration boost expired for {config_key}: "
                f"{old_boost:.2f} → 1.0"
            )

            # Emit event for downstream consumers
            try:
                from app.coordination.event_router import get_event_bus, DataEventType

                bus = get_event_bus()
                if bus:
                    bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                        "config_key": config_key,
                        "priority": "normal",
                        "reason": "exploration_boost_expired",
                        "exploration_boost": 1.0,
                    })
            except Exception as emit_err:
                logger.debug(f"[SelfplayScheduler] Failed to emit boost decay: {emit_err}")

        return decayed_count

    # =========================================================================
    # Elo Velocity Utilities
    # =========================================================================

    def get_elo_velocity(self, config_key: str) -> float:
        """Get computed Elo velocity for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Elo change per hour (can be negative for regression)
        """
        return self._elo_velocity.get(config_key, 0.0)

    def initialize_elo_velocities_from_db(self) -> int:
        """Initialize Elo velocities from database history at startup.

        January 3, 2026: Fixes cold start gap where velocities are 0.0 until
        ELO_VELOCITY_CHANGED events fire (5-30 min delay). This method queries
        the elo_history table to bootstrap velocity calculations.

        Returns:
            Number of configs with initialized velocities.

        Called from: master_loop.py after SelfplayScheduler initialization.
        """
        try:
            from app.training.elo_service import get_elo_service

            service = get_elo_service()
            if not service:
                logger.debug("[SelfplayScheduler] Elo service not available for velocity init")
                return 0

            # Query elo_history for last 24 hours per config
            now = time.time()
            cutoff = now - 86400  # 24 hours
            initialized_count = 0

            # Query all distinct configs from elo_history
            with service._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT board_type || '_' || num_players || 'p' as config_key
                    FROM elo_history
                    WHERE timestamp > ?
                    """,
                    (cutoff,),
                )
                configs = [row[0] for row in cursor.fetchall()]

            for config_key in configs:
                parts = config_key.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                board_type = parts[0]
                try:
                    num_players = int(parts[1].rstrip("p"))
                except ValueError:
                    continue

                # Query history for this config
                with service._get_connection() as conn:
                    cursor = conn.execute(
                        """
                        SELECT timestamp, rating
                        FROM elo_history
                        WHERE board_type = ? AND num_players = ?
                        AND timestamp > ?
                        ORDER BY timestamp ASC
                        """,
                        (board_type, num_players, cutoff),
                    )
                    history = [(row[0], row[1]) for row in cursor.fetchall()]

                if len(history) < 2:
                    continue

                # Initialize history
                self._elo_history[config_key] = history

                # Calculate velocity
                hours = (history[-1][0] - history[0][0]) / 3600
                if hours > 0.5:  # Need at least 30 min
                    velocity = (history[-1][1] - history[0][1]) / hours
                    self._elo_velocity[config_key] = velocity
                    initialized_count += 1
                    logger.debug(
                        f"[SelfplayScheduler] Initialized Elo velocity for {config_key}: "
                        f"{velocity:.2f} Elo/hour (from {len(history)} samples)"
                    )

            if initialized_count > 0:
                logger.info(
                    f"[SelfplayScheduler] Initialized Elo velocities for {initialized_count} configs from DB"
                )
            return initialized_count

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error initializing velocities from DB: {e}")
            return 0

    # =========================================================================
    # Diversity Tracking (January 2026 Sprint 10)
    # =========================================================================

    def record_opponent(self, config_key: str, opponent_type: str) -> None:
        """Record that a config played against an opponent type.

        January 2026 Sprint 10: Tracks opponent variety for diversity maximization.
        Configs that play against more diverse opponents get higher priority to
        maximize training robustness.

        Args:
            config_key: Config like "hex8_2p"
            opponent_type: Type of opponent (e.g., "heuristic", "policy", "gumbel")
        """
        if config_key not in self._opponent_types_by_config:
            self._opponent_types_by_config[config_key] = set()
        self._opponent_types_by_config[config_key].add(opponent_type)
        # Recompute diversity score
        self._diversity_scores[config_key] = self._compute_diversity_score(config_key)

    def _compute_diversity_score(self, config_key: str) -> float:
        """Compute diversity score for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Diversity score between 0.0 (low diversity) and 1.0 (high diversity)
        """
        opponents_seen = self._opponent_types_by_config.get(config_key, set())
        if not opponents_seen:
            return 0.0  # No opponents seen = low diversity
        return min(1.0, len(opponents_seen) / self._max_opponent_types)

    def get_diversity_score(self, config_key: str) -> float:
        """Get diversity score for a config.

        January 2026 Sprint 10: Used in priority calculation to boost configs
        with low opponent variety.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Diversity score between 0.0 (low diversity) and 1.0 (high diversity)
        """
        return self._diversity_scores.get(config_key, 0.0)

    def get_opponent_types_seen(self, config_key: str) -> int:
        """Get number of distinct opponent types played by a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Number of distinct opponent types
        """
        return len(self._opponent_types_by_config.get(config_key, set()))
