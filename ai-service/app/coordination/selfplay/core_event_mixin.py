"""Core event handler mixin for SelfplayScheduler.

Sprint 17.9+ (Feb 2026): Extracted from selfplay_scheduler.py to reduce file size.

Provides the core event handler implementations that were still in the main file:
- _on_selfplay_complete: Handle selfplay completion
- _on_training_complete: Handle training completion
- _on_promotion_complete: Handle model promotion
- _on_selfplay_target_updated: Handle target updates with budget coupling
- _on_curriculum_rebalanced: Handle curriculum rebalancing (regression + quality triggers)
- _on_selfplay_rate_changed: Handle momentum multiplier changes
- _on_memory_pressure: Handle memory pressure events
- _on_resource_constraint: Handle resource constraint events
- _on_new_games_available: Handle real-time game count updates

Also provides the legacy subscribe_to_events() method for backward compatibility.

The _get_event_subscriptions() method remains on the main class since it
references handlers from multiple mixins and needs the full class hierarchy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from app.coordination.event_handler_utils import extract_config_key

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CoreEventHandlerMixin:
    """Mixin providing core event handlers for SelfplayScheduler.

    Expects the host class to have:
    - _config_priorities: dict of ConfigPriority
    - _subscribed: bool
    - record_opponent(config_key, opponent_type): method from VelocityMixin
    """

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion event."""
        try:
            config_key = extract_config_key(event.payload)
            if config_key in self._config_priorities:
                # Reset staleness for this config
                self._config_priorities[config_key].staleness_hours = 0.0

                # January 2026 Sprint 10: Record opponent type for diversity tracking
                # Extract opponent type from event (e.g., "heuristic", "policy", "gumbel", "nnue")
                opponent_type = event.payload.get("opponent_type") or event.payload.get("engine_mode")
                if opponent_type:
                    self.record_opponent(config_key, str(opponent_type))
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay complete: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion event."""
        try:
            config_key = extract_config_key(event.payload)
            if config_key in self._config_priorities:
                # Clear training pending flag
                self._config_priorities[config_key].training_pending = False
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling training complete: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle promotion completion event."""
        try:
            config_key = extract_config_key(event.payload)
            success = event.payload.get("success", False)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                if success:
                    # Boost for continued improvement
                    priority.exploration_boost = 1.0
                else:
                    # Increase exploration on failure
                    priority.exploration_boost = min(2.0, priority.exploration_boost * 1.3)
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling promotion complete: {e}")

    def _on_selfplay_target_updated(self, event: Any) -> None:
        """Handle selfplay target update request.

        Phase 5 (Dec 2025): Responds to requests for more/fewer selfplay games.
        Typically emitted when training needs more data urgently.

        Dec 28 2025: Extended to handle search_budget and velocity feedback
        from FeedbackLoopController for reaching 2000+ Elo.
        """
        try:
            from app.coordination.budget_calculator import get_board_adjusted_budget

            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            target_games = payload.get("target_games", 0)
            priority_val = payload.get("priority", "normal")
            search_budget = payload.get("search_budget", 0)
            exploration_boost = payload.get("exploration_boost", 1.0)
            velocity = payload.get("velocity", 0.0)
            reason = payload.get("reason", "")

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Dec 28 2025: Apply search budget from velocity feedback
                if search_budget > 0 and reason == "velocity_feedback":
                    old_budget = getattr(priority, "search_budget", 400)
                    # Feb 2026: Apply large board budget caps scaled by player count
                    board_type = config_key.split("_")[0]
                    num_players = int(config_key.split("_")[1].rstrip("p"))
                    game_count = getattr(priority, "game_count", 0)
                    search_budget = get_board_adjusted_budget(board_type, search_budget, game_count, num_players)
                    priority.search_budget = search_budget
                    logger.info(
                        f"[SelfplayScheduler] Updating {config_key} search budget: "
                        f"{old_budget}->{search_budget} (velocity={velocity:.1f} Elo/hr)"
                    )

                # Apply exploration boost
                if exploration_boost != 1.0:
                    priority.exploration_boost = exploration_boost

                # Boost priority based on urgency
                if priority_val.upper() == "HIGH":
                    priority.training_pending = True
                    priority.exploration_boost = max(1.2, priority.exploration_boost)
                    logger.info(
                        f"[SelfplayScheduler] Boosting {config_key} priority "
                        f"(target: {target_games} games, priority: {priority_val})"
                    )
                elif priority_val.lower() == "low":
                    # Reduce priority for this config
                    priority.exploration_boost = min(0.8, priority.exploration_boost)
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay target: {e}")

    def _on_curriculum_rebalanced(self, event: Any) -> None:
        """Handle curriculum rebalancing event.

        Phase 4A.1 (December 2025): Updates priority weights when curriculum
        feedback adjusts config priorities based on training progress.

        December 30, 2025: Enhanced to handle regression-triggered curriculum
        updates with factor-based allocation reduction.

        December 2025 Guard: Skip events originated by SelfplayScheduler itself
        to prevent echo loops in the event system.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event

            # Loop guard: Skip events we emitted (prevents echo loops)
            source = payload.get("source", "")
            if source == "selfplay_scheduler":
                logger.debug("[SelfplayScheduler] Skipping self-originated CURRICULUM_REBALANCED")
                return

            trigger = payload.get("trigger", "")
            config_key = extract_config_key(payload)
            new_weight = payload.get("weight", 1.0)
            reason = payload.get("reason", "")

            # December 30, 2025: Handle regression-triggered curriculum emergency updates
            if trigger == "regression_detected":
                changed_configs = payload.get("changed_configs", [])
                factor = payload.get("factor", 0.5)
                elo_loss = payload.get("elo_loss", 0.0)

                for cfg in changed_configs:
                    if cfg in self._config_priorities:
                        old_weight = self._config_priorities[cfg].curriculum_weight
                        new_wt = old_weight * factor
                        self._config_priorities[cfg].curriculum_weight = new_wt
                        # Also boost exploration to encourage diversity
                        self._config_priorities[cfg].exploration_boost = min(
                            2.0, self._config_priorities[cfg].exploration_boost * 1.5
                        )
                        logger.warning(
                            f"[SelfplayScheduler] Regression-triggered curriculum reduction: "
                            f"{cfg} weight {old_weight:.2f} -> {new_wt:.2f} (factor={factor}, "
                            f"elo_loss={elo_loss:.0f}), exploration boosted"
                        )
                return  # Handled regression case

            # Session 17.11: Handle quality critical drop - boost allocation for affected config
            # This enables immediate response to quality degradation (+8-12 Elo improvement)
            if trigger == "quality_critical_drop":
                changed_configs = payload.get("changed_configs", [])
                factor = payload.get("factor", 1.5)
                quality_drop = payload.get("quality_drop", 0.0)

                for cfg in changed_configs:
                    if cfg in self._config_priorities:
                        old_weight = self._config_priorities[cfg].curriculum_weight
                        # Boost allocation to prioritize data generation
                        new_wt = min(2.0, old_weight * factor)
                        self._config_priorities[cfg].curriculum_weight = new_wt
                        # Also increase exploration boost to improve diversity
                        self._config_priorities[cfg].exploration_boost = min(
                            2.0, self._config_priorities[cfg].exploration_boost * 1.3
                        )
                        logger.warning(
                            f"[SelfplayScheduler] Quality-drop allocation boost: "
                            f"{cfg} weight {old_weight:.2f} -> {new_wt:.2f} (factor={factor}, "
                            f"quality_drop={quality_drop:.2f}), exploration boosted"
                        )
                return  # Handled quality_critical_drop case

            if config_key in self._config_priorities:
                old_weight = self._config_priorities[config_key].curriculum_weight
                self._config_priorities[config_key].curriculum_weight = new_weight

                # Only log significant changes
                if abs(new_weight - old_weight) > 0.1:
                    logger.info(
                        f"[SelfplayScheduler] Curriculum rebalanced: {config_key} "
                        f"weight {old_weight:.2f} -> {new_weight:.2f}"
                        + (f" ({reason})" if reason else "")
                    )

            # Also handle batch updates (multiple configs at once)
            weights = payload.get("weights", {})
            if weights:
                for cfg, weight in weights.items():
                    if cfg in self._config_priorities:
                        old_w = self._config_priorities[cfg].curriculum_weight
                        self._config_priorities[cfg].curriculum_weight = weight
                        if abs(weight - old_w) > 0.1:
                            logger.info(
                                f"[SelfplayScheduler] Curriculum batch update: "
                                f"{cfg} weight {old_w:.2f} -> {weight:.2f}"
                            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling curriculum rebalanced: {e}")

    def _on_selfplay_rate_changed(self, event: Any) -> None:
        """Handle selfplay rate change event from FeedbackAccelerator.

        P0.1 (Dec 2025): Closes the Elo momentum -> Selfplay rate feedback loop.
        FeedbackAccelerator emits SELFPLAY_RATE_CHANGED when it detects:
        - ACCELERATING: 1.5x multiplier (capitalize on positive momentum)
        - IMPROVING: 1.25x multiplier (boost for continued improvement)
        - STABLE: 1.0x multiplier (normal rate)
        - PLATEAU: 1.1x multiplier (slight boost to break plateau)
        - REGRESSING: 0.75x multiplier (reduce noise, focus on quality)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            new_rate = payload.get("new_rate", 1.0) or payload.get("rate_multiplier", 1.0)
            momentum_state = payload.get("momentum_state", "unknown")

            if config_key in self._config_priorities:
                old_rate = self._config_priorities[config_key].momentum_multiplier
                self._config_priorities[config_key].momentum_multiplier = float(new_rate)

                # Log significant changes
                if abs(new_rate - old_rate) > 0.1:
                    logger.info(
                        f"[SelfplayScheduler] Selfplay rate changed: {config_key} "
                        f"multiplier {old_rate:.2f} -> {new_rate:.2f} "
                        f"(momentum: {momentum_state})"
                    )
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received rate change for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay rate changed: {e}")

    # =========================================================================
    # Memory Pressure Event Handlers (January 13, 2026)
    # =========================================================================

    def _on_memory_pressure(self, event: Any) -> None:
        """Handle MEMORY_PRESSURE event - pause selfplay when memory critical.

        January 13, 2026: Added to prevent OOM and allow cleanup daemons time to
        free disk space. When memory pressure is CRITICAL or EMERGENCY, we pause
        selfplay allocation to reduce memory load.

        Event payload:
            tier: str - "CAUTION", "WARNING", "CRITICAL", or "EMERGENCY"
            source: str - "coordinator", "gpu_vram", "system_ram", etc.
            node_id: str - Which node is under pressure (optional)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            tier = payload.get("tier", "").upper()
            source = payload.get("source", "unknown")
            node_id = payload.get("node_id", "")

            if tier in ("CRITICAL", "EMERGENCY"):
                if not self._memory_constrained:
                    logger.warning(
                        f"[SelfplayScheduler] Memory pressure {tier} ({source}), "
                        f"pausing selfplay allocation"
                    )
                self._memory_constrained = True
                self._memory_constraint_source = f"{source}:{node_id}" if node_id else source
            elif tier in ("CAUTION", "WARNING"):
                # Resume on lower pressure tiers
                if self._memory_constrained:
                    logger.info(
                        f"[SelfplayScheduler] Memory pressure reduced to {tier}, "
                        f"resuming selfplay allocation"
                    )
                self._memory_constrained = False
                self._memory_constraint_source = ""

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling memory pressure: {e}")

    def _on_resource_constraint(self, event: Any) -> None:
        """Handle RESOURCE_CONSTRAINT event - general resource limits.

        January 13, 2026: Handles general resource constraints (disk space, etc.)
        that may require pausing selfplay to allow recovery.

        Event payload:
            resource_type: str - "disk", "memory", "gpu_vram", etc.
            level: str - "warning", "critical", "emergency"
            node_id: str - Which node is constrained (optional)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            level = payload.get("level", "").lower()
            resource_type = payload.get("resource_type", "unknown")
            node_id = payload.get("node_id", "")

            if level in ("critical", "emergency"):
                if not self._memory_constrained:
                    logger.warning(
                        f"[SelfplayScheduler] Resource constraint {level} "
                        f"({resource_type}), pausing selfplay allocation"
                    )
                self._memory_constrained = True
                self._memory_constraint_source = f"{resource_type}:{node_id}" if node_id else resource_type
            elif level in ("normal", "ok", "released"):
                if self._memory_constrained:
                    logger.info(
                        f"[SelfplayScheduler] Resource constraint released ({resource_type}), "
                        f"resuming selfplay allocation"
                    )
                self._memory_constrained = False
                self._memory_constraint_source = ""

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling resource constraint: {e}")

    def subscribe_to_events(self) -> None:
        """Subscribe to relevant pipeline events.

        December 30, 2025: This method is retained for backward compatibility.
        New code should use start() which automatically subscribes via
        _get_event_subscriptions(). This method delegates to the HandlerBase
        subscription infrastructure.
        """
        if self._subscribed:
            return

        try:
            # Dec 2025 fix: Use get_router() instead of get_event_bus() because:
            # 1. get_router() always returns a valid UnifiedEventRouter singleton
            # 2. get_event_bus() can return None if data_events module unavailable
            # 3. UnifiedEventRouter handles event type normalization automatically
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            # Dec 2025: Use per-subscription error handling to ensure one failure
            # doesn't prevent other subscriptions from being registered
            subscribed_count = 0
            failed_count = 0

            def _safe_subscribe(event_type, handler, name: str) -> bool:
                """Subscribe with individual error handling."""
                nonlocal subscribed_count, failed_count
                try:
                    # UnifiedEventRouter.subscribe() handles both enum and string types
                    router.subscribe(event_type, handler)
                    subscribed_count += 1
                    return True
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"[SelfplayScheduler] Failed to subscribe to {name}: {e}")
                    return False

            # Core subscriptions (always attempt)
            _safe_subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete, "SELFPLAY_COMPLETE")
            _safe_subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete, "TRAINING_COMPLETED")
            _safe_subscribe(DataEventType.MODEL_PROMOTED, self._on_promotion_complete, "MODEL_PROMOTED")
            # Phase 5: Subscribe to feedback events
            _safe_subscribe(DataEventType.SELFPLAY_TARGET_UPDATED, self._on_selfplay_target_updated, "SELFPLAY_TARGET_UPDATED")
            _safe_subscribe(DataEventType.QUALITY_DEGRADED, self._on_quality_degraded, "QUALITY_DEGRADED")
            # Phase 4A.1: Subscribe to curriculum rebalancing (December 2025)
            _safe_subscribe(DataEventType.CURRICULUM_REBALANCED, self._on_curriculum_rebalanced, "CURRICULUM_REBALANCED")
            # P0.1 (Dec 2025): Subscribe to SELFPLAY_RATE_CHANGED from FeedbackAccelerator
            _safe_subscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed, "SELFPLAY_RATE_CHANGED")
            # P1.1 (Dec 2025): Subscribe to TRAINING_BLOCKED_BY_QUALITY for selfplay acceleration
            _safe_subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY, self._on_training_blocked_by_quality, "TRAINING_BLOCKED_BY_QUALITY")

            # Optional subscriptions (graceful degradation if event type doesn't exist)
            if hasattr(DataEventType, 'OPPONENT_MASTERED'):
                _safe_subscribe(DataEventType.OPPONENT_MASTERED, self._on_opponent_mastered, "OPPONENT_MASTERED")
            if hasattr(DataEventType, 'TRAINING_EARLY_STOPPED'):
                _safe_subscribe(DataEventType.TRAINING_EARLY_STOPPED, self._on_training_early_stopped, "TRAINING_EARLY_STOPPED")
            if hasattr(DataEventType, 'ELO_VELOCITY_CHANGED'):
                _safe_subscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed, "ELO_VELOCITY_CHANGED")
            if hasattr(DataEventType, 'EXPLORATION_BOOST'):
                _safe_subscribe(DataEventType.EXPLORATION_BOOST, self._on_exploration_boost, "EXPLORATION_BOOST")
            # Jan 7, 2026: Subscribe to EXPLORATION_ADJUSTED for quality-driven exploration
            if hasattr(DataEventType, 'EXPLORATION_ADJUSTED'):
                _safe_subscribe(DataEventType.EXPLORATION_ADJUSTED, self._on_exploration_adjusted, "EXPLORATION_ADJUSTED")
            # Dec 2025: Subscribe to CURRICULUM_ADVANCED for curriculum progression
            if hasattr(DataEventType, 'CURRICULUM_ADVANCED'):
                _safe_subscribe(DataEventType.CURRICULUM_ADVANCED, self._on_curriculum_advanced, "CURRICULUM_ADVANCED")
            # Dec 2025: Subscribe to ADAPTIVE_PARAMS_CHANGED for parameter adjustments
            if hasattr(DataEventType, 'ADAPTIVE_PARAMS_CHANGED'):
                _safe_subscribe(DataEventType.ADAPTIVE_PARAMS_CHANGED, self._on_adaptive_params_changed, "ADAPTIVE_PARAMS_CHANGED")
            if hasattr(DataEventType, 'LOW_QUALITY_DATA_WARNING'):
                _safe_subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, self._on_low_quality_warning, "LOW_QUALITY_DATA_WARNING")

            # P2P cluster health events (December 2025)
            if hasattr(DataEventType, 'NODE_UNHEALTHY'):
                _safe_subscribe(DataEventType.NODE_UNHEALTHY, self._on_node_unhealthy, "NODE_UNHEALTHY")
            if hasattr(DataEventType, 'NODE_RECOVERED'):
                _safe_subscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered, "NODE_RECOVERED")
            # Dec 28, 2025: NODE_ACTIVATED from cluster activator/watchdog also means node is available
            if hasattr(DataEventType, 'NODE_ACTIVATED'):
                _safe_subscribe(DataEventType.NODE_ACTIVATED, self._on_node_recovered, "NODE_ACTIVATED")
            if hasattr(DataEventType, 'P2P_NODE_DEAD'):
                _safe_subscribe(DataEventType.P2P_NODE_DEAD, self._on_node_unhealthy, "P2P_NODE_DEAD")
            if hasattr(DataEventType, 'P2P_CLUSTER_UNHEALTHY'):
                _safe_subscribe(DataEventType.P2P_CLUSTER_UNHEALTHY, self._on_cluster_unhealthy, "P2P_CLUSTER_UNHEALTHY")
            if hasattr(DataEventType, 'P2P_CLUSTER_HEALTHY'):
                _safe_subscribe(DataEventType.P2P_CLUSTER_HEALTHY, self._on_cluster_healthy, "P2P_CLUSTER_HEALTHY")
            if hasattr(DataEventType, 'HOST_OFFLINE'):
                _safe_subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline, "HOST_OFFLINE")
            # Dec 2025: NODE_TERMINATED from idle shutdown - reuse host_offline handler
            if hasattr(DataEventType, 'NODE_TERMINATED'):
                _safe_subscribe(DataEventType.NODE_TERMINATED, self._on_host_offline, "NODE_TERMINATED")

            # Jan 3, 2026: Subscribe to voter demotion/promotion for allocation adjustment
            # When a voter is demoted, it signals potential node health issues
            if hasattr(DataEventType, 'VOTER_DEMOTED'):
                _safe_subscribe(DataEventType.VOTER_DEMOTED, self._on_voter_demoted, "VOTER_DEMOTED")
            if hasattr(DataEventType, 'VOTER_PROMOTED'):
                _safe_subscribe(DataEventType.VOTER_PROMOTED, self._on_voter_promoted, "VOTER_PROMOTED")

            # Jan 3, 2026 Session 10: Subscribe to CIRCUIT_RESET for proactive recovery monitoring
            # When a circuit breaker is reset after proactive health probe, boost node priority
            if hasattr(DataEventType, 'CIRCUIT_RESET'):
                _safe_subscribe(DataEventType.CIRCUIT_RESET, self._on_circuit_reset, "CIRCUIT_RESET")

            # Dec 2025: Subscribe to regression events for curriculum rebalancing
            if hasattr(DataEventType, 'REGRESSION_DETECTED'):
                _safe_subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected, "REGRESSION_DETECTED")

            # Dec 29, 2025: Subscribe to backpressure events for reactive scheduling
            if hasattr(DataEventType, 'BACKPRESSURE_ACTIVATED'):
                _safe_subscribe(DataEventType.BACKPRESSURE_ACTIVATED, self._on_backpressure_activated, "BACKPRESSURE_ACTIVATED")
            if hasattr(DataEventType, 'BACKPRESSURE_RELEASED'):
                _safe_subscribe(DataEventType.BACKPRESSURE_RELEASED, self._on_backpressure_released, "BACKPRESSURE_RELEASED")
            # Jan 2026 Sprint 10: Subscribe to evaluation-specific backpressure
            # When evaluation queue is backlogged, slow down selfplay to reduce queue pressure
            if hasattr(DataEventType, 'EVALUATION_BACKPRESSURE'):
                _safe_subscribe(DataEventType.EVALUATION_BACKPRESSURE, self._on_evaluation_backpressure, "EVALUATION_BACKPRESSURE")
            if hasattr(DataEventType, 'EVALUATION_BACKPRESSURE_RELEASED'):
                _safe_subscribe(DataEventType.EVALUATION_BACKPRESSURE_RELEASED, self._on_backpressure_released, "EVALUATION_BACKPRESSURE_RELEASED")
            # Dec 29, 2025: Subscribe to NODE_OVERLOADED for per-node backpressure
            if hasattr(DataEventType, 'NODE_OVERLOADED'):
                _safe_subscribe(DataEventType.NODE_OVERLOADED, self._on_node_overloaded, "NODE_OVERLOADED")

            # Dec 29, 2025 - Phase 2: Subscribe to ELO_UPDATED for velocity tracking
            if hasattr(DataEventType, 'ELO_UPDATED'):
                _safe_subscribe(DataEventType.ELO_UPDATED, self._on_elo_updated, "ELO_UPDATED")

            # Dec 29, 2025: Subscribe to progress stall events for 48h autonomous operation
            if hasattr(DataEventType, 'PROGRESS_STALL_DETECTED'):
                _safe_subscribe(DataEventType.PROGRESS_STALL_DETECTED, self._on_progress_stall, "PROGRESS_STALL_DETECTED")
            if hasattr(DataEventType, 'PROGRESS_RECOVERED'):
                _safe_subscribe(DataEventType.PROGRESS_RECOVERED, self._on_progress_recovered, "PROGRESS_RECOVERED")

            # Dec 29, 2025: Subscribe to architecture weight updates for allocation adjustment
            if hasattr(DataEventType, 'ARCHITECTURE_WEIGHTS_UPDATED'):
                _safe_subscribe(DataEventType.ARCHITECTURE_WEIGHTS_UPDATED, self._on_architecture_weights_updated, "ARCHITECTURE_WEIGHTS_UPDATED")

            # Dec 30, 2025: Subscribe to quality feedback for immediate cache invalidation
            # This enables faster response to quality degradation (+12-18 Elo improvement)
            if hasattr(DataEventType, 'QUALITY_FEEDBACK_ADJUSTED'):
                _safe_subscribe(DataEventType.QUALITY_FEEDBACK_ADJUSTED, self._on_quality_feedback_adjusted, "QUALITY_FEEDBACK_ADJUSTED")

            # Jan 2026: Subscribe to work queue backpressure events
            if hasattr(DataEventType, 'WORK_QUEUE_BACKPRESSURE'):
                _safe_subscribe(DataEventType.WORK_QUEUE_BACKPRESSURE, self._on_work_queue_backpressure, "WORK_QUEUE_BACKPRESSURE")
            if hasattr(DataEventType, 'WORK_QUEUE_BACKPRESSURE_RELEASED'):
                _safe_subscribe(DataEventType.WORK_QUEUE_BACKPRESSURE_RELEASED, self._on_work_queue_backpressure_released, "WORK_QUEUE_BACKPRESSURE_RELEASED")

            # Jan 2026: Subscribe to multi-harness evaluation events
            if hasattr(DataEventType, 'MULTI_HARNESS_EVALUATION_COMPLETED'):
                _safe_subscribe(DataEventType.MULTI_HARNESS_EVALUATION_COMPLETED, self._on_multi_harness_evaluation_completed, "MULTI_HARNESS_EVALUATION_COMPLETED")
            if hasattr(DataEventType, 'CROSS_CONFIG_TOURNAMENT_COMPLETED'):
                _safe_subscribe(DataEventType.CROSS_CONFIG_TOURNAMENT_COMPLETED, self._on_cross_config_tournament_completed, "CROSS_CONFIG_TOURNAMENT_COMPLETED")

            # Jan 2026: New game count events
            if hasattr(DataEventType, 'NEW_GAMES_AVAILABLE'):
                _safe_subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games_available, "NEW_GAMES_AVAILABLE")

            # Jan 2026: Memory pressure events
            if hasattr(DataEventType, 'MEMORY_PRESSURE'):
                _safe_subscribe(DataEventType.MEMORY_PRESSURE, self._on_memory_pressure, "MEMORY_PRESSURE")
            if hasattr(DataEventType, 'RESOURCE_CONSTRAINT'):
                _safe_subscribe(DataEventType.RESOURCE_CONSTRAINT, self._on_resource_constraint, "RESOURCE_CONSTRAINT")

            self._subscribed = subscribed_count > 0
            if self._subscribed:
                logger.info(
                    f"[SelfplayScheduler] Subscribed to {subscribed_count} pipeline events "
                    f"(failed: {failed_count}, includes P2P health events)"
                )
            else:
                logger.warning("[SelfplayScheduler] No subscriptions succeeded, reactive scheduling disabled")

        except ImportError as e:
            logger.warning(f"[SelfplayScheduler] Event router unavailable: {e}")
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Failed to subscribe to events: {e}")
