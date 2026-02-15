"""EventHandlersMixin: Event handler methods extracted from SelfplayScheduler.

Extracted from selfplay_scheduler.py for better modularity.
Contains all event subscription and handler methods, plus event emission helpers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

# Import constants from canonical source
try:
    from app.p2p.constants import (
        EXPLORATION_BOOST_DEFAULT_DURATION,
        PLATEAU_CLEAR_WIN_RATE,
        PLATEAU_PENALTY_DEFAULT_DURATION,
        TRAINING_BOOST_DURATION,
        PROMOTION_PENALTY_DURATION_CRITICAL,
        PROMOTION_PENALTY_DURATION_MULTIPLE,
        PROMOTION_PENALTY_DURATION_SINGLE,
        PROMOTION_PENALTY_FACTOR_CRITICAL,
        PROMOTION_PENALTY_FACTOR_MULTIPLE,
        PROMOTION_PENALTY_FACTOR_SINGLE,
    )
except ImportError:
    # Fallback for testing/standalone use - match canonical values in app/p2p/constants.py
    EXPLORATION_BOOST_DEFAULT_DURATION = 900  # 15 minutes
    PLATEAU_CLEAR_WIN_RATE = 0.50
    PLATEAU_PENALTY_DEFAULT_DURATION = 1800  # 30 minutes
    TRAINING_BOOST_DURATION = 1800  # 30 minutes
    # Promotion penalty constants - match app/p2p/constants.py
    PROMOTION_PENALTY_DURATION_CRITICAL = 7200  # 2 hours
    PROMOTION_PENALTY_DURATION_MULTIPLE = 3600  # 1 hour
    PROMOTION_PENALTY_DURATION_SINGLE = 1800  # 30 min
    PROMOTION_PENALTY_FACTOR_CRITICAL = 0.3
    PROMOTION_PENALTY_FACTOR_MULTIPLE = 0.5
    PROMOTION_PENALTY_FACTOR_SINGLE = 0.7

# Session 17.22: Architecture selection feedback loop
# Import architecture tracker functions for per-(config, architecture) performance tracking
# and intelligent architecture selection based on Elo performance
try:
    from app.training.architecture_tracker import (
        record_evaluation as _record_architecture_eval,
    )
except ImportError:
    _record_architecture_eval = None  # type: ignore

logger = logging.getLogger(__name__)


class EventHandlersMixin:
    """Mixin containing all event handler methods for SelfplayScheduler.

    Extracted from SelfplayScheduler for modularity. Methods use `self` and
    expect the host class to provide:
    - self._rate_multipliers: dict[str, float]
    - self._curriculum_weights: dict[str, float]
    - self._configs_in_training_pipeline: set[str]
    - self._plateaued_configs: dict[str, float]
    - self._evaluation_backpressure_active: bool
    - self._evaluation_backpressure_start: float
    - self._quality_blocked_configs: dict[str, tuple[float, float]]
    - self._quality_scores: dict[str, tuple[float, int, float]]
    - self._architecture_weights_cache: dict[str, tuple[dict, float]]
    - self._force_rebalance: bool
    - self._orchestrator: Any
    - self.verbose: bool
    - self._log_info(), self._log_debug(), self._log_warning()
    - self._extract_event_payload()
    - self.set_exploration_boost()
    - self.on_training_complete()
    """

    # =========================================================================
    # Event Subscriptions (December 2025)
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions for EventSubscriptionMixin.

        Dec 28, 2025: Migrated to use EventSubscriptionMixin pattern.
        Dec 29, 2025 (Phase 2B): Added pipeline coordination events:
        - NPZ_EXPORT_COMPLETE: Temporarily boost priority for freshly exported config
        - TRAINING_STARTED: Mark config as in-training, reduce selfplay allocation
        - EVALUATION_COMPLETED: Update curriculum weights based on gauntlet results

        Returns:
            Dict mapping event names to handler methods
        """
        return {
            "SELFPLAY_RATE_CHANGED": self._on_selfplay_rate_changed,
            "EXPLORATION_BOOST": self._on_exploration_boost,
            "TRAINING_COMPLETED": self._on_training_completed,
            "ELO_VELOCITY_CHANGED": self._on_elo_velocity_changed,
            # December 2025 - Phase 2B: Pipeline coordination events
            "NPZ_EXPORT_COMPLETE": self._on_npz_export_complete,
            "TRAINING_STARTED": self._on_training_started,
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
            # December 2025 - Phase 4D: Plateau detection for resource balancing
            "PLATEAU_DETECTED": self._on_plateau_detected,
            # December 2025 - Phase 5: Evaluation backpressure handling
            "EVALUATION_BACKPRESSURE": self._on_evaluation_backpressure,
            "EVALUATION_BACKPRESSURE_RELEASED": self._on_evaluation_backpressure_released,
            # January 2026 - Sprint 10: Quality-blocked training feedback
            "TRAINING_BLOCKED_BY_QUALITY": self._on_training_blocked_by_quality,
            # January 2026 - Sprint 10: Hyperparameter updates affect selfplay quality
            "HYPERPARAMETER_UPDATED": self._on_hyperparameter_updated,
            # January 2026 - Sprint 13: Peer reconnection triggers work rebalancing
            "PEER_RECONNECTED": self._on_peer_reconnected,
            # January 2026 - Session 17.22: Immediate quality score application
            "QUALITY_SCORE_UPDATED": self._on_quality_score_updated,
            # January 2026 - Session 17.26: Architecture weights cache refresh
            "ARCHITECTURE_WEIGHTS_UPDATED": self._on_architecture_weights_updated,
            # January 2026 - Session 17.27: Auto-dispatch on critical starvation
            "DATA_STARVATION_CRITICAL": self._on_data_starvation_critical,
        }

    async def _on_selfplay_rate_changed(self, event) -> None:
        """Handle SELFPLAY_RATE_CHANGED events from feedback loop.

        Adjusts rate multipliers for configs based on Elo velocity and performance.

        Args:
            event: Event with payload containing config_key, new_rate, reason
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        new_rate = payload.get("new_rate", 1.0)
        reason = payload.get("reason", "unknown")

        if not config_key:
            return

        old_rate = self._rate_multipliers.get(config_key, 1.0)
        self._rate_multipliers[config_key] = new_rate

        if abs(new_rate - old_rate) > 0.01:
            self._log_info(
                f"Rate changed for {config_key}: {old_rate:.2f} -> {new_rate:.2f} ({reason})"
            )

    def get_rate_multiplier(self, config_key: str) -> float:
        """Get current rate multiplier for a config.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Rate multiplier (1.0 = normal, >1 = boost, <1 = throttle)
        """
        return self._rate_multipliers.get(config_key, 1.0)

    async def _on_exploration_boost(self, event) -> None:
        """Handle EXPLORATION_BOOST events from FeedbackLoopController.

        Dec 2025: React to training anomalies (loss spikes, stalls) by increasing
        exploration to generate more diverse training data.

        Args:
            event: Event with payload containing config_key, boost_factor, reason
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        boost_factor = payload.get("boost_factor", 1.3)
        duration = payload.get("duration_seconds", EXPLORATION_BOOST_DEFAULT_DURATION)
        reason = payload.get("reason", "training_anomaly")

        if not config_key:
            return

        # Delegate to existing set_exploration_boost method
        self.set_exploration_boost(config_key, boost_factor, duration)

        self._log_info(
            f"Exploration boost from event: {config_key} "
            f"{boost_factor:.2f}x for {duration}s ({reason})"
        )

    async def _on_training_completed(self, event) -> None:
        """Handle TRAINING_COMPLETED events to boost selfplay after training.

        Dec 2025: When training completes, boost selfplay for that config to
        generate more data for the next training cycle.

        Args:
            event: Event with payload containing config_key
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")

        if not config_key:
            return

        # Delegate to existing on_training_complete method
        self.on_training_complete(config_key)

    async def _on_elo_velocity_changed(self, event) -> None:
        """Handle ELO_VELOCITY_CHANGED events for selfplay rate adjustment.

        Dec 2025: Adjusts selfplay rate based on Elo velocity trends.
        - accelerating: Increase selfplay to capitalize on momentum
        - decelerating: Reduce selfplay, shift focus to training quality
        - stable: Maintain current allocation

        Args:
            event: Event with payload containing config_key, velocity, trend
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        velocity = payload.get("velocity", 0.0)
        trend = payload.get("trend", "stable")

        if not config_key:
            return

        old_rate = self._rate_multipliers.get(config_key, 1.0)

        # Adjust rate based on velocity trend
        if trend == "accelerating":
            # Capitalize on positive momentum - increase selfplay rate
            new_rate = min(1.5, old_rate * 1.2)
        elif trend == "decelerating":
            # Slow down and focus on quality
            new_rate = max(0.6, old_rate * 0.85)
        else:  # stable
            # Slight adjustment toward 1.0
            if old_rate > 1.0:
                new_rate = max(1.0, old_rate * 0.95)
            elif old_rate < 1.0:
                new_rate = min(1.0, old_rate * 1.05)
            else:
                new_rate = 1.0

        self._rate_multipliers[config_key] = new_rate

        if abs(new_rate - old_rate) > 0.01:
            self._log_info(
                f"Elo velocity {trend} for {config_key}: "
                f"velocity={velocity:.1f}, rate {old_rate:.2f} -> {new_rate:.2f}"
            )
            # P0.2 Dec 2025: Emit rate change event for significant changes
            self._emit_selfplay_rate_changed(
                config_key, old_rate, new_rate, f"elo_velocity_{trend}"
            )

    # =========================================================================
    # Phase 2B Event Handlers (December 2025)
    # =========================================================================

    async def _on_npz_export_complete(self, event) -> None:
        """Handle NPZ_EXPORT_COMPLETE events - boost priority for freshly exported config.

        Dec 2025 Phase 2B: When NPZ export completes, temporarily boost the config's
        priority to generate more training data quickly while the exported data
        is being used for training.

        This ensures selfplay doesn't pile on extra games for a config that's about
        to enter training, but still keeps generating some data for the next cycle.

        Args:
            event: Event with payload containing config_key, samples, output_path
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "") or payload.get("config", "")
        samples = payload.get("samples", 0)

        if not config_key:
            return

        # Temporarily reduce selfplay rate while training is consuming this export
        # This prevents wasting compute on a config that's actively training
        old_rate = self._rate_multipliers.get(config_key, 1.0)

        # Reduce to 70% of normal rate during training phase
        new_rate = max(0.5, old_rate * 0.7)
        self._rate_multipliers[config_key] = new_rate

        self._log_info(
            f"NPZ export complete for {config_key}: {samples} samples, "
            f"reducing selfplay rate {old_rate:.2f} -> {new_rate:.2f}"
        )

        # Track that this config is in "export->training" transition
        self._configs_in_training_pipeline.add(config_key)

    async def _on_training_started(self, event) -> None:
        """Handle TRAINING_STARTED events - mark config as in-training.

        Dec 2025 Phase 2B: When training starts for a config, mark it as
        "in training pipeline" and reduce selfplay allocation slightly.
        This prioritizes sync and evaluation over generating excess selfplay data.

        Args:
            event: Event with payload containing config_key, epochs, batch_size
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "") or payload.get("config", "")

        if not config_key:
            return

        # Mark config as in training
        self._configs_in_training_pipeline.add(config_key)

        # Reduce selfplay rate while training is active
        old_rate = self._rate_multipliers.get(config_key, 1.0)
        new_rate = max(0.4, old_rate * 0.6)  # 60% of current rate
        self._rate_multipliers[config_key] = new_rate

        self._log_info(
            f"Training started for {config_key}, "
            f"reducing selfplay rate {old_rate:.2f} -> {new_rate:.2f}"
        )

    async def _on_evaluation_completed(self, event) -> None:
        """Handle EVALUATION_COMPLETED events - update curriculum weights.

        Dec 2025 Phase 2B: When gauntlet evaluation completes, update curriculum
        weights based on performance results. This enables real-time curriculum
        adjustments based on model strength.

        Session 17.22: Added architecture-specific Elo tracking. Records evaluation
        results to ArchitectureTracker for per-(config, architecture) performance.

        Curriculum weight updates:
        - High win rate (>75%): Reduce weight, model is strong
        - Low win rate (<50%): Increase weight, model needs more training data
        - Mid-range: Maintain current weight

        Args:
            event: Event with payload containing config_key, win_rate, elo, architecture
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "") or payload.get("config", "")
        win_rate = payload.get("win_rate", 0.5)
        elo = payload.get("elo", 1500.0)
        architecture = payload.get("architecture", "")
        games_played = payload.get("games_played", 0)

        if not config_key:
            return

        # Remove from training pipeline (evaluation is final step)
        self._configs_in_training_pipeline.discard(config_key)

        # Dec 2025 Phase 4D: Clear plateau penalty on successful evaluation
        # If win rate is acceptable, the config is making progress
        if win_rate >= PLATEAU_CLEAR_WIN_RATE and config_key in self._plateaued_configs:
            del self._plateaued_configs[config_key]
            self._log_info(
                f"Plateau cleared for {config_key} (win_rate={win_rate:.1%}), "
                f"restoring normal priority"
            )

        # Restore normal selfplay rate after training cycle completes
        old_rate = self._rate_multipliers.get(config_key, 1.0)
        if old_rate < 0.8:
            # Restore to normal rate
            self._rate_multipliers[config_key] = 1.0
            self._log_info(
                f"Evaluation complete for {config_key}, "
                f"restoring selfplay rate {old_rate:.2f} -> 1.00"
            )

        # Update curriculum weight based on performance
        current_weight = self._curriculum_weights.get(config_key, 1.0)

        if win_rate > 0.75:
            # Strong model - reduce curriculum weight
            new_weight = max(0.3, current_weight * 0.8)
            reason = "strong_model"
        elif win_rate < 0.50:
            # Struggling model - increase curriculum weight
            new_weight = min(2.5, current_weight * 1.3)
            reason = "struggling_model"
        else:
            # Mid-range - slight adjustment toward 1.0
            if current_weight > 1.0:
                new_weight = max(1.0, current_weight * 0.95)
            elif current_weight < 1.0:
                new_weight = min(1.0, current_weight * 1.05)
            else:
                new_weight = 1.0
            reason = "stable_model"

        if abs(new_weight - current_weight) > 0.05:
            self._curriculum_weights[config_key] = new_weight
            self._log_info(
                f"Curriculum weight updated for {config_key} "
                f"(win_rate={win_rate:.1%}, elo={elo:.0f}): "
                f"{current_weight:.2f} -> {new_weight:.2f} ({reason})"
            )

        # Session 17.22: Record evaluation to ArchitectureTracker for per-(config, arch) Elo
        # This enables intelligent architecture allocation based on performance
        if architecture and _record_architecture_eval is not None:
            try:
                # Parse config_key to get board_type and num_players
                # Format: "hex8_2p" -> board_type="hex8", num_players=2
                parts = config_key.rsplit("_", 1)
                if len(parts) == 2 and parts[1].endswith("p"):
                    board_type = parts[0]
                    num_players = int(parts[1].rstrip("p"))
                    training_hours = payload.get("training_hours", 0.0)

                    _record_architecture_eval(
                        architecture=architecture,
                        board_type=board_type,
                        num_players=num_players,
                        elo=elo,
                        training_hours=training_hours,
                        games_evaluated=games_played,
                    )
                    self._log_debug(
                        f"Recorded evaluation for {architecture}:{config_key} "
                        f"(elo={elo:.0f}, games={games_played})"
                    )
            except (ValueError, TypeError) as e:
                self._log_debug(f"Could not parse config_key {config_key}: {e}")

    async def _on_plateau_detected(self, event) -> None:
        """Handle PLATEAU_DETECTED events - reduce priority for plateaued configs.

        Dec 2025 Phase 4D: When a config is detected as plateaued (no Elo improvement
        despite training), reduce its selfplay allocation to avoid wasting resources.
        The config will still receive some games for exploration, but healthy configs
        get priority.

        Plateau penalty expires after a duration (default 30 minutes) to allow
        recovery after hyperparameter adjustments or curriculum changes.

        Args:
            event: Event with payload containing config_key, reason, duration_seconds
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "") or payload.get("config", "")
        duration_seconds = payload.get("duration_seconds", PLATEAU_PENALTY_DEFAULT_DURATION)
        reason = payload.get("reason", "elo_stagnation")

        if not config_key:
            return

        # Calculate expiration timestamp
        expiry_time = time.time() + duration_seconds
        old_expiry = self._plateaued_configs.get(config_key)

        # Update or add plateau tracking
        self._plateaued_configs[config_key] = expiry_time

        if old_expiry is None:
            self._log_info(
                f"Plateau detected for {config_key} ({reason}), "
                f"reducing selfplay priority for {duration_seconds}s"
            )
        else:
            self._log_debug(
                f"Plateau extended for {config_key}, new expiry in {duration_seconds}s"
            )

    def _is_config_plateaued(self, config_key: str) -> bool:
        """Check if a config is currently in plateau state.

        Dec 2025 Phase 4D: Returns True if the config has an active plateau penalty.
        Automatically cleans up expired entries.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            True if config is plateaued and penalty is active, False otherwise.
        """
        expiry_time = self._plateaued_configs.get(config_key)
        if expiry_time is None:
            return False

        current_time = time.time()
        if current_time >= expiry_time:
            # Plateau expired - clean up and return False
            del self._plateaued_configs[config_key]
            self._log_debug(f"Plateau expired for {config_key}, restoring priority")
            return False

        return True

    async def _on_evaluation_backpressure(self, event) -> None:
        """Handle EVALUATION_BACKPRESSURE events - pause selfplay to let evaluations catch up.

        Dec 2025 Phase 5: When evaluation queue exceeds threshold (typically 70 pending),
        the evaluation daemon emits this event. Selfplay should pause to prevent
        cascading backpressure: selfplay -> training -> evaluation bottleneck.

        Without this handler, selfplay continues producing games even when downstream
        pipeline is saturated, leading to training loop deadlock.

        Args:
            event: Event with payload containing queue_depth, threshold
        """
        payload = self._extract_event_payload(event)
        queue_depth = payload.get("queue_depth", 0)
        threshold = payload.get("threshold", 70)

        self._evaluation_backpressure_active = True
        self._evaluation_backpressure_start = time.time()

        self._log_info(
            f"Evaluation backpressure ACTIVATED: queue_depth={queue_depth} > threshold={threshold}, "
            "pausing selfplay allocation until evaluation queue drains"
        )

    async def _on_evaluation_backpressure_released(self, event) -> None:
        """Handle EVALUATION_BACKPRESSURE_RELEASED events - resume selfplay.

        Dec 2025 Phase 5: When evaluation queue drops below release threshold
        (typically 35), resume selfplay allocation.

        Args:
            event: Event with payload containing queue_depth, release_threshold
        """
        payload = self._extract_event_payload(event)
        queue_depth = payload.get("queue_depth", 0)
        release_threshold = payload.get("release_threshold", 35)

        if self._evaluation_backpressure_active:
            duration = time.time() - self._evaluation_backpressure_start
            self._log_info(
                f"Evaluation backpressure RELEASED: queue_depth={queue_depth} < release_threshold={release_threshold}, "
                f"resuming selfplay allocation after {duration:.1f}s pause"
            )
        self._evaluation_backpressure_active = False
        self._evaluation_backpressure_start = 0.0

    async def _on_training_blocked_by_quality(self, event) -> None:
        """Handle TRAINING_BLOCKED_BY_QUALITY events - boost high-quality selfplay.

        Jan 2026 Sprint 10: When training is blocked due to low data quality,
        increase the proportion of high-quality selfplay modes (Gumbel MCTS with
        higher budget) and reduce total game volume to focus on quality over quantity.

        Jan 5, 2026 - Session 17.29: Added auto-dispatch for significant quality deficits.
        When quality_deficit >= 0.15 (15% below threshold), immediately dispatch
        high-quality selfplay jobs to address the issue. Expected +5-8 Elo improvement
        from faster quality response.

        This closes the quality feedback loop:
        - Quality gate blocks training -> Signal to selfplay scheduler
        - Scheduler boosts quality mode percentage (e.g., 50% -> 80% Gumbel MCTS)
        - Scheduler auto-dispatches high-quality selfplay if deficit is significant
        - Higher quality data unblocks training -> Better Elo gains

        Expected improvement: +5-8 Elo per config from immediate reallocation.

        Args:
            event: Event with payload containing config_key, quality_score, threshold
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.0)
        threshold = payload.get("threshold", 0.7)
        reason = payload.get("reason", "low_data_quality")

        if not config_key:
            return

        # Calculate quality boost factor based on how far below threshold we are
        # Quality score 0.5 with threshold 0.7 -> boost 1.5x
        # Quality score 0.3 with threshold 0.7 -> boost 2.0x
        quality_deficit = max(0, threshold - quality_score)
        quality_boost = 1.0 + (quality_deficit * 2.5)  # 1.0 to ~2.5x boost
        quality_boost = min(quality_boost, 2.5)  # Cap at 2.5x

        # Set quality boost for 30 minutes (time to generate and train on better data)
        duration = 1800  # 30 minutes
        expiry = time.time() + duration
        self._quality_blocked_configs[config_key] = (quality_boost, expiry)

        self._log_info(
            f"Quality boost for {config_key}: {quality_boost:.2f}x for {duration}s "
            f"(quality={quality_score:.2f}, threshold={threshold:.2f}, reason={reason})"
        )

        # Also set exploration boost to diversify the data
        self.set_exploration_boost(config_key, min(1.3, quality_boost), duration)

        # Jan 5, 2026 - Session 17.29: Auto-dispatch for significant quality deficits
        # Only trigger auto-dispatch if quality deficit is significant (>= 15%)
        if quality_deficit >= 0.15:
            await self._auto_dispatch_quality_selfplay(
                config_key=config_key,
                quality_score=quality_score,
                quality_deficit=quality_deficit,
            )

    def get_quality_boost(self, config_key: str) -> float:
        """Get current quality boost factor for a config.

        Jan 2026 Sprint 10: Quality boost increases the proportion of high-quality
        selfplay modes when a config is blocked by the quality gate.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Quality boost factor (1.0 = normal, >1.0 = boosted high-quality mode %).
        """
        boost_info = self._quality_blocked_configs.get(config_key)
        if not boost_info:
            return 1.0

        boost_factor, expiry = boost_info
        if time.time() > expiry:
            # Boost expired, clean up
            del self._quality_blocked_configs[config_key]
            return 1.0

        return boost_factor

    async def _on_quality_score_updated(self, event) -> None:
        """Handle QUALITY_SCORE_UPDATED events - apply quality score immediately.

        Session 17.22: Reduces latency from 0-60s (polling) to <1s (event-driven).
        When quality_monitor_daemon.py assesses data quality, this handler
        immediately stores the score so allocation decisions use fresh data.

        This enables:
        - Immediate allocation shift to higher-quality modes when quality drops
        - Faster response to quality improvements (unblock training sooner)
        - More responsive feedback loop (+8-15 Elo improvement expected)

        Args:
            event: Event with payload containing config_key, quality_score,
                   games_assessed, confidence
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.5)
        games_assessed = payload.get("games_assessed", 0)

        if not config_key:
            return

        # Store the quality score with timestamp
        self._quality_scores[config_key] = (
            quality_score,
            games_assessed,
            time.time(),
        )

        # If quality is low, proactively boost high-quality mode ratio
        # This happens BEFORE training is blocked, preventing quality-gate delays
        if quality_score < 0.5 and games_assessed >= 50:
            # Low quality detected - boost quality-focused selfplay modes
            quality_deficit = 0.5 - quality_score
            boost_factor = 1.0 + (quality_deficit * 3.0)  # 1.0 to 2.5x boost
            boost_factor = min(boost_factor, 2.0)  # Cap lower than blocking boost

            duration = 600  # 10 minutes - reassess on next quality event
            self._quality_blocked_configs[config_key] = (boost_factor, time.time() + duration)

            self._log_info(
                f"Quality preemptive boost for {config_key}: {boost_factor:.2f}x "
                f"(quality={quality_score:.2f}, games={games_assessed})"
            )
        elif quality_score >= 0.7:
            # Good quality - clear any existing boost to restore normal allocation
            if config_key in self._quality_blocked_configs:
                del self._quality_blocked_configs[config_key]
                self._log_info(
                    f"Quality restored for {config_key} (score={quality_score:.2f})"
                )

    async def _on_architecture_weights_updated(self, event) -> None:
        """Handle ARCHITECTURE_WEIGHTS_UPDATED events - refresh weight cache immediately.

        Session 17.26: Enables event-driven cache refresh for architecture weights.
        When ArchitectureFeedbackController computes new weights (every 30 min or
        on evaluation completion), this handler immediately updates the cache.

        Benefits:
        - Reduces DB queries from every job to cache-hit most of the time
        - Fresher weights propagation (immediate vs 30 min TTL expiry)
        - Consistent weights across cluster (all nodes receive same event)
        - +8-15 Elo improvement expected from faster weight application

        Args:
            event: Event with payload containing config_key, weights dict, timestamp
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        weights = payload.get("weights", {})

        if not config_key or not weights:
            return

        # Update cache immediately
        self._architecture_weights_cache[config_key] = (weights, time.time())

        self._log_info(
            f"[P2P] Refreshed architecture weights for {config_key}: "
            f"{list(weights.items())[:3]}..."
        )

    def get_quality_score(self, config_key: str) -> tuple[float, int]:
        """Get current quality score and games assessed for a config.

        Session 17.22: Returns the most recent quality score from events.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Tuple of (quality_score, games_assessed). Defaults to (0.5, 0) if unknown.
        """
        quality_info = self._quality_scores.get(config_key)
        if not quality_info:
            return 0.5, 0

        score, games, timestamp = quality_info
        # Quality scores older than 1 hour are considered stale - decay toward 0.5
        age_hours = (time.time() - timestamp) / 3600.0
        if age_hours > 1.0:
            # Decay toward 0.5 (neutral) with half-life of 1 hour
            decay_factor = 0.5 ** age_hours
            score = 0.5 + (score - 0.5) * decay_factor

        return score, games

    async def _on_hyperparameter_updated(self, event) -> None:
        """Handle HYPERPARAMETER_UPDATED events - adjust selfplay strategy.

        Jan 2026 Sprint 10: When training hyperparameters change (e.g., from
        GauntletFeedbackController), adjust selfplay quality and exploration
        to match the new training configuration.

        Key hyperparameters we respond to:
        - exploration_boost: Adjust temperature/exploration in selfplay
        - quality_threshold: Adjust Gumbel budget and quality mode ratio
        - learning_rate_multiplier: Higher LR = need more diverse data
        - batch_reduction: Smaller batches = can handle higher quality data

        Args:
            event: Event with payload containing param_name, new_value, config
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config", payload.get("config_key", ""))
        param_name = payload.get("param_name", "")
        new_value = payload.get("new_value", 0.0)
        reason = payload.get("reason", "hyperparameter_update")

        if not config_key or not param_name:
            return

        self._log_info(
            f"Hyperparameter update for {config_key}: {param_name}={new_value} ({reason})"
        )

        # Handle specific hyperparameter changes
        if param_name == "exploration_boost":
            # Direct exploration boost from gauntlet feedback
            boost_duration = 1800  # 30 minutes
            self.set_exploration_boost(config_key, float(new_value), boost_duration)

        elif param_name == "quality_threshold":
            # Quality threshold raised = need higher quality selfplay
            threshold = float(new_value)
            if threshold >= 0.8:
                # High quality threshold - boost quality mode ratio
                quality_boost = 1.5
                expiry = time.time() + 1800
                self._quality_blocked_configs[config_key] = (quality_boost, expiry)
                self._log_info(
                    f"Quality boost {quality_boost:.2f}x for {config_key} "
                    f"(high threshold: {threshold:.2f})"
                )

        elif param_name == "learning_rate_multiplier":
            # Higher LR needs more diverse data to prevent overfitting
            lr_mult = float(new_value)
            if lr_mult > 1.2:
                # High LR - boost exploration slightly
                exploration = min(1.3, lr_mult * 0.9)
                self.set_exploration_boost(config_key, exploration, 1800)

        elif param_name == "batch_reduction":
            # Smaller batches can handle higher quality data
            batch_factor = float(new_value)
            if batch_factor < 0.8:
                # Significantly smaller batches - we can afford higher quality
                quality_boost = 1.0 + (1.0 - batch_factor)  # e.g., 0.7 batch -> 1.3 boost
                expiry = time.time() + 1800
                self._quality_blocked_configs[config_key] = (quality_boost, expiry)

    async def _on_peer_reconnected(self, event: dict) -> None:
        """Handle PEER_RECONNECTED events - rebalance work when node rejoins.

        Jan 2026 Sprint 13: When a peer reconnects (via Tailscale discovery),
        trigger work rebalancing to take advantage of the newly available node.
        This ensures recovered nodes immediately receive selfplay allocations
        rather than waiting for the next scheduling cycle.
        """
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", payload.get("peer_id", "unknown"))

        logger.info(
            f"[SelfplayScheduler] Peer reconnected: {node_id}, "
            "triggering work rebalancing"
        )

        # Mark that we need to rebalance allocations
        # The next scheduling cycle will pick this up
        self._force_rebalance = True

        # Emit event for other components to react
        try:
            from app.coordination.event_router import publish_sync
            publish_sync("SELFPLAY_REBALANCE_TRIGGERED", {
                "reason": "peer_reconnected",
                "node_id": node_id,
                "source": "selfplay_scheduler",
            })
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Could not emit rebalance event: {e}")

    async def _on_data_starvation_critical(self, event: dict) -> None:
        """Handle DATA_STARVATION_CRITICAL events - auto-dispatch priority selfplay.

        Jan 5, 2026 - Session 17.27: When ULTRA starvation is detected (<20 games),
        automatically dispatch priority selfplay jobs to address the starvation.
        This enables the cluster to automatically respond to critically low game
        counts without manual intervention.

        Args:
            event: Event with payload containing config_key, game_count, tier
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        game_count = payload.get("game_count", 0)
        tier = payload.get("tier", "")

        if not config_key:
            return

        # Auto-dispatch for ULTRA and EMERGENCY tiers (most critical)
        # Jan 5, 2026 - Session 17.29: Extended to EMERGENCY tier to help
        # configs like hexagonal_3p (86 games) that need immediate attention
        if tier not in ("ULTRA", "EMERGENCY"):
            logger.debug(
                f"[SelfplayScheduler] Starvation alert for {config_key} "
                f"({tier} tier, {game_count} games) - below EMERGENCY, skipping auto-dispatch"
            )
            return

        # Check cooldown to avoid duplicate dispatches
        cooldown_key = f"starvation_dispatch_{config_key}"
        last_dispatch = getattr(self, "_starvation_dispatch_times", {}).get(cooldown_key, 0)
        if time.time() - last_dispatch < 600:  # 10 minute cooldown
            logger.debug(
                f"[SelfplayScheduler] Skipping auto-dispatch for {config_key} "
                "(dispatched within last 10 minutes)"
            )
            return

        # Parse config_key to get board_type and num_players
        try:
            from app.coordination.event_utils import parse_config_key
            parsed = parse_config_key(config_key)
            board_type = parsed.board_type
            num_players = parsed.num_players
        except Exception as e:
            logger.warning(
                f"[SelfplayScheduler] Failed to parse config_key {config_key}: {e}"
            )
            return

        # Dispatch priority selfplay based on tier severity
        # ULTRA (< 20 games): 200/100 games, EMERGENCY (< 100 games): 75 games
        if tier == "ULTRA":
            games_to_dispatch = 200 if game_count < 10 else 100
        else:  # EMERGENCY
            games_to_dispatch = 75
        logger.info(
            f"[SelfplayScheduler] Auto-dispatching {games_to_dispatch} selfplay games "
            f"for {config_key} ({tier} starvation, only {game_count} games)"
        )

        try:
            # Use P2P dispatch endpoint if available
            if hasattr(self, "_orchestrator") and self._orchestrator:
                # Dispatch via orchestrator's dispatch_selfplay method
                result = await self._orchestrator.dispatch_selfplay(
                    board_type=board_type,
                    num_players=num_players,
                    num_games=games_to_dispatch,
                    priority="critical",
                )
                if result.get("success"):
                    # Track cooldown
                    if not hasattr(self, "_starvation_dispatch_times"):
                        self._starvation_dispatch_times: dict[str, float] = {}
                    self._starvation_dispatch_times[cooldown_key] = time.time()

                    logger.info(
                        f"[SelfplayScheduler] Successfully dispatched priority selfplay "
                        f"for {config_key}: {result.get('work_items_created', 0)} work items"
                    )
                else:
                    logger.warning(
                        f"[SelfplayScheduler] Failed to dispatch selfplay for {config_key}: "
                        f"{result.get('error', 'unknown error')}"
                    )
            else:
                # Fallback: just boost priority and let normal scheduling handle it
                logger.info(
                    f"[SelfplayScheduler] No orchestrator available, "
                    f"boosting priority for {config_key} instead"
                )
                self._force_rebalance = True
                # Add temporary priority boost
                self._rate_multipliers[config_key] = self._rate_multipliers.get(config_key, 1.0) * 2.0

        except Exception as e:
            logger.warning(
                f"[SelfplayScheduler] Failed to auto-dispatch selfplay for {config_key}: {e}"
            )

    async def _auto_dispatch_quality_selfplay(
        self,
        config_key: str,
        quality_score: float,
        quality_deficit: float,
    ) -> None:
        """Auto-dispatch high-quality selfplay when quality is significantly low.

        Jan 5, 2026 - Session 17.29: When quality deficit >= 15%, immediately dispatch
        high-quality selfplay jobs to address the quality issue. Uses a 5-minute
        cooldown to avoid duplicate dispatches.

        This complements the quality boost by actively generating new high-quality
        data rather than just adjusting priorities.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            quality_score: Current quality score (0.0-1.0)
            quality_deficit: How far below threshold (threshold - quality_score)
        """
        # Check cooldown to avoid duplicate dispatches (5 minute cooldown)
        cooldown_key = f"quality_dispatch_{config_key}"
        if not hasattr(self, "_quality_dispatch_times"):
            self._quality_dispatch_times: dict[str, float] = {}
        last_dispatch = self._quality_dispatch_times.get(cooldown_key, 0)
        if time.time() - last_dispatch < 300:  # 5 minute cooldown
            self._log_debug(
                f"Skipping quality auto-dispatch for {config_key} "
                "(dispatched within last 5 minutes)"
            )
            return

        # Parse config_key to get board_type and num_players
        try:
            from app.coordination.event_utils import parse_config_key
            parsed = parse_config_key(config_key)
            board_type = parsed.board_type
            num_players = parsed.num_players
        except Exception as e:
            self._log_warning(f"Failed to parse config_key {config_key}: {e}")
            return

        # Dispatch high-quality selfplay (fewer games, higher quality)
        # Scale games based on deficit severity: 15% deficit -> 50 games, 30%+ -> 100 games
        games_to_dispatch = 50 if quality_deficit < 0.25 else 100
        self._log_info(
            f"Auto-dispatching {games_to_dispatch} high-quality selfplay games "
            f"for {config_key} (quality={quality_score:.2f}, deficit={quality_deficit:.2f})"
        )

        try:
            if hasattr(self, "_orchestrator") and self._orchestrator:
                # Dispatch via orchestrator's dispatch_selfplay method
                # High priority ensures Gumbel MCTS mode is used for quality
                result = await self._orchestrator.dispatch_selfplay(
                    board_type=board_type,
                    num_players=num_players,
                    num_games=games_to_dispatch,
                    priority="high",  # High priority uses Gumbel MCTS for better quality
                )
                if result.get("success"):
                    # Track cooldown
                    self._quality_dispatch_times[cooldown_key] = time.time()

                    self._log_info(
                        f"Successfully dispatched quality selfplay for {config_key}: "
                        f"{result.get('work_items_created', 0)} work items"
                    )

                    # Emit event for observability
                    try:
                        from app.distributed.data_events import DataEventType
                        from app.coordination.event_router import publish_sync
                        publish_sync(DataEventType.QUALITY_SELFPLAY_DISPATCHED.value, {
                            "config_key": config_key,
                            "quality_score": quality_score,
                            "quality_deficit": quality_deficit,
                            "games_dispatched": games_to_dispatch,
                            "work_items_created": result.get("work_items_created", 0),
                            "source": "selfplay_scheduler",
                        })
                    except (ImportError, AttributeError, TypeError, RuntimeError):
                        pass  # Non-critical, don't fail on event emission
                else:
                    self._log_warning(
                        f"Failed to dispatch quality selfplay for {config_key}: "
                        f"{result.get('error', 'unknown error')}"
                    )
            else:
                # Fallback: force rebalance to prioritize this config
                self._log_info(
                    f"No orchestrator available, forcing rebalance for {config_key}"
                )
                self._force_rebalance = True
                self._rate_multipliers[config_key] = self._rate_multipliers.get(config_key, 1.0) * 1.5

        except Exception as e:
            self._log_warning(f"Failed to auto-dispatch quality selfplay for {config_key}: {e}")

    # =========================================================================
    # Event Emission Helpers (December 2025)
    # =========================================================================

    def _emit_selfplay_target_updated(
        self,
        config_key: str,
        priority: str,
        reason: str,
        *,
        target_jobs: int | None = None,
        effective_priority: int | None = None,
        exploration_boost: float | None = None,
    ) -> bool:
        """Emit SELFPLAY_TARGET_UPDATED event for feedback loop integration.

        P0.2 (December 2025): Enables pipeline coordination to respond to
        scheduling decisions. Events trigger:
        - DaemonManager workload scaling
        - FeedbackLoopController priority adjustments
        - Training pipeline data freshness checks

        Dec 2025 (P0-1 fix): Returns bool for caller to check success/failure.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            priority: Priority level ("urgent", "high", "normal")
            reason: Descriptive reason for the update
            target_jobs: Optional target job count
            effective_priority: Optional effective priority value
            exploration_boost: Optional exploration boost multiplier

        Returns:
            True if event was emitted successfully, False otherwise.
        """
        try:
            from app.coordination.event_router import publish_sync

            payload: dict[str, Any] = {
                "config_key": config_key,
                "priority": priority,
                "reason": reason,
                "source": "p2p_selfplay_scheduler",
            }
            if target_jobs is not None:
                payload["target_jobs"] = target_jobs
            if effective_priority is not None:
                payload["effective_priority"] = effective_priority
            if exploration_boost is not None:
                payload["exploration_boost"] = exploration_boost

            publish_sync("SELFPLAY_TARGET_UPDATED", payload)
            if self.verbose:
                logger.info(
                    f"[SelfplayScheduler] Emitted SELFPLAY_TARGET_UPDATED: "
                    f"{config_key} priority={priority} reason={reason}"
                )
            return True
        except ImportError:
            logger.debug("[SelfplayScheduler] Event router not available for target updates")
            return False
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit target update: {e}")
            return False

    def _emit_selfplay_rate_changed(
        self,
        config_key: str,
        old_rate: float,
        new_rate: float,
        reason: str,
    ) -> bool:
        """Emit SELFPLAY_RATE_CHANGED event when rate multiplier changes >20%.

        P0.2 (December 2025): Enables IdleResourceDaemon and other consumers
        to react to significant rate changes.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            old_rate: Previous rate multiplier
            new_rate: New rate multiplier
            reason: Descriptive reason for the change

        Returns:
            True if event was emitted successfully, False otherwise.
        """
        # Only emit for significant changes (>20%)
        if old_rate > 0 and abs(new_rate - old_rate) / old_rate < 0.2:
            return False

        try:
            from app.coordination.event_router import publish_sync

            payload = {
                "config_key": config_key,
                "old_rate": old_rate,
                "new_rate": new_rate,
                "reason": reason,
                "source": "p2p_selfplay_scheduler",
            }
            publish_sync("SELFPLAY_RATE_CHANGED", payload)
            if self.verbose:
                logger.info(
                    f"[SelfplayScheduler] Emitted SELFPLAY_RATE_CHANGED: "
                    f"{config_key} {old_rate:.2f} -> {new_rate:.2f} ({reason})"
                )
            return True
        except ImportError:
            logger.debug("[SelfplayScheduler] Event router not available for rate changes")
            return False
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit rate change: {e}")
            return False

    def _emit_selfplay_allocation_updated(
        self,
        config_key: str,
        allocation_weights: dict[str, float],
        exploration_boost: float,
        reason: str,
    ) -> bool:
        """Emit SELFPLAY_ALLOCATION_UPDATED event when allocation weights change.

        P0.2 (December 2025): Enables IdleResourceDaemon and SelfplayScheduler
        to react to curriculum weight changes and exploration boosts.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            allocation_weights: Current allocation weights by config
            exploration_boost: Current exploration boost factor
            reason: Descriptive reason for the change

        Returns:
            True if event was emitted successfully, False otherwise.
        """
        try:
            from app.coordination.event_router import publish_sync

            payload = {
                "config_key": config_key,
                "allocation_weights": allocation_weights,
                "exploration_boost": exploration_boost,
                "reason": reason,
                "source": "p2p_selfplay_scheduler",
            }
            publish_sync("SELFPLAY_ALLOCATION_UPDATED", payload)
            if self.verbose:
                logger.info(
                    f"[SelfplayScheduler] Emitted SELFPLAY_ALLOCATION_UPDATED: "
                    f"{config_key} boost={exploration_boost:.2f} ({reason})"
                )
            return True
        except ImportError:
            logger.debug("[SelfplayScheduler] Event router not available for allocation updates")
            return False
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit allocation update: {e}")
            return False
