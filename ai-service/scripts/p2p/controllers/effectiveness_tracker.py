"""Effectiveness Tracking - Recovery action effectiveness measurement and feedback.

Jan 2026: Created as part of P2P Self-Healing Architecture.

This module tracks whether recovery actions actually improve cluster health,
closing the feedback loop:
    Action Taken -> Cooldown Period -> Metrics Measured -> Weight Adjusted

Key features:
- Records pre/post metrics around each action
- Computes effectiveness scores (0.0-1.0)
- Adjusts action weights based on measured effectiveness
- Integrates with StabilityController for action prioritization
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Environment variable to disable effectiveness tracking
EFFECTIVENESS_TRACKING_ENABLED = os.environ.get(
    "RINGRIFT_EFFECTIVENESS_TRACKING_ENABLED", "true"
).lower() in ("true", "1", "yes")


@dataclass
class ActionRecord:
    """Record of a recovery action and its outcome.

    Stores the state before and after the action to compute effectiveness.
    """
    action: str
    nodes: list[str]
    timestamp: float
    pre_metrics: dict[str, Any]  # alive_count, stability_score before action
    post_metrics: dict[str, Any] | None = None  # filled after cooldown
    effectiveness: float | None = None  # 0.0-1.0, computed from metric delta
    symptom_context: dict[str, Any] = field(default_factory=dict)  # Original symptom info

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action": self.action,
            "nodes": self.nodes[:5],  # Limit for display
            "timestamp": self.timestamp,
            "pre_metrics": self.pre_metrics,
            "post_metrics": self.post_metrics,
            "effectiveness": self.effectiveness,
            "symptom_context": self.symptom_context,
        }


class EffectivenessTracker:
    """
    Tracks recovery action effectiveness for feedback loop closure.

    This is the "learning" component that adjusts action weights based on
    measured outcomes. Actions that improve health get higher weights;
    actions that don't help get lower weights.

    The formula for effectiveness:
        score = 0.5 + alive_delta * 0.3 + stability_delta * 0.2

    Where:
    - alive_delta = (post.alive_count - pre.alive_count) / pre.alive_count
    - stability_delta = post.stability_score - pre.stability_score

    Integration points:
    - record_action(): Called by StabilityController after executing recovery
    - check_effectiveness(): Called periodically to evaluate pending actions
    - get_action_weight(): Used by StabilityController to prioritize actions
    """

    COOLDOWN_SECONDS = 120  # Wait 2 min before measuring effectiveness
    WEIGHT_BOOST = 1.1     # Multiplier for effective actions (>60% score)
    WEIGHT_PENALTY = 0.9   # Multiplier for ineffective actions (<40% score)
    MIN_WEIGHT = 0.1       # Minimum action weight
    MAX_WEIGHT = 2.0       # Maximum action weight
    MAX_HISTORY = 100      # Maximum completed records to keep

    def __init__(
        self,
        get_current_metrics: Callable[[], dict[str, Any]] | None = None,
    ):
        """Initialize the effectiveness tracker.

        Args:
            get_current_metrics: Callable that returns current cluster metrics
                                 (alive_count, stability_score, etc.)
        """
        self._get_metrics = get_current_metrics
        self._pending: list[ActionRecord] = []
        self._completed: list[ActionRecord] = []
        self._action_weights: dict[str, float] = {}

        logger.info(
            f"EffectivenessTracker initialized: enabled={EFFECTIVENESS_TRACKING_ENABLED}, "
            f"cooldown={self.COOLDOWN_SECONDS}s"
        )

    def set_metrics_callback(self, callback: Callable[[], dict[str, Any]]) -> None:
        """Set the metrics callback (for late binding during initialization)."""
        self._get_metrics = callback

    def record_action(
        self,
        action: str,
        nodes: list[str],
        symptom_context: dict[str, Any] | None = None,
    ) -> None:
        """Record that a recovery action was taken.

        Should be called immediately after executing a recovery action.
        The pre-metrics are captured at this point.

        Args:
            action: The action type (e.g., "increase_timeout")
            nodes: List of affected node IDs
            symptom_context: Optional context about the triggering symptom
        """
        if not EFFECTIVENESS_TRACKING_ENABLED:
            return

        if not self._get_metrics:
            logger.warning("EffectivenessTracker: No metrics callback configured")
            return

        try:
            pre_metrics = self._get_metrics()
        except Exception as e:
            logger.warning(f"Failed to capture pre-metrics: {e}")
            pre_metrics = {}

        record = ActionRecord(
            action=action,
            nodes=nodes,
            timestamp=time.time(),
            pre_metrics=pre_metrics,
            symptom_context=symptom_context or {},
        )
        self._pending.append(record)
        logger.debug(
            f"Recorded action: {action} for {len(nodes)} nodes "
            f"(pre: alive={pre_metrics.get('alive_count', '?')})"
        )

    def check_effectiveness(self) -> list[ActionRecord]:
        """Check pending actions that have passed the cooldown period.

        For each action that has waited long enough, capture post-metrics,
        compute effectiveness, and update action weights.

        Returns:
            List of newly completed ActionRecords
        """
        if not EFFECTIVENESS_TRACKING_ENABLED:
            return []

        now = time.time()
        newly_completed = []
        still_pending = []

        for record in self._pending:
            if now - record.timestamp >= self.COOLDOWN_SECONDS:
                # Capture post-metrics
                try:
                    record.post_metrics = self._get_metrics() if self._get_metrics else {}
                except Exception as e:
                    logger.warning(f"Failed to capture post-metrics: {e}")
                    record.post_metrics = {}

                # Compute effectiveness
                record.effectiveness = self._compute_effectiveness(record)

                # Update action weight
                self._update_weight(record.action, record.effectiveness)

                # Log result
                logger.info(
                    f"Action effectiveness: {record.action} = {record.effectiveness:.2f} "
                    f"(alive: {record.pre_metrics.get('alive_count', '?')} -> "
                    f"{record.post_metrics.get('alive_count', '?')})"
                )

                self._completed.append(record)
                newly_completed.append(record)
            else:
                still_pending.append(record)

        self._pending = still_pending

        # Prune old completed records
        if len(self._completed) > self.MAX_HISTORY:
            self._completed = self._completed[-self.MAX_HISTORY:]

        return newly_completed

    def _compute_effectiveness(self, record: ActionRecord) -> float:
        """Compute effectiveness score (0.0-1.0) from metric changes.

        The formula considers:
        - Change in alive node count (normalized)
        - Change in stability score

        Args:
            record: The action record with pre and post metrics

        Returns:
            Effectiveness score between 0.0 and 1.0
        """
        pre = record.pre_metrics
        post = record.post_metrics or {}

        # Get alive counts
        pre_alive = pre.get("alive_count", 0)
        post_alive = post.get("alive_count", 0)

        # Calculate alive delta (normalized by pre-count)
        if pre_alive > 0:
            alive_delta = (post_alive - pre_alive) / pre_alive
        else:
            alive_delta = 1.0 if post_alive > 0 else 0.0

        # Get stability scores
        pre_stability = pre.get("stability_score", 0)
        post_stability = post.get("stability_score", 0)
        stability_delta = post_stability - pre_stability

        # Combined score (baseline 0.5, can go up or down)
        # alive_delta weighted at 0.3 (±30% contribution)
        # stability_delta weighted at 0.002 (scores are typically 0-100 range)
        score = 0.5 + alive_delta * 0.3 + stability_delta * 0.002

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def _update_weight(self, action: str, effectiveness: float) -> None:
        """Adjust action weight based on measured effectiveness.

        Effective actions (>60% score) get a 10% boost.
        Ineffective actions (<40% score) get a 10% penalty.
        Neutral actions (40-60%) are unchanged.

        Args:
            action: The action type
            effectiveness: The computed effectiveness score
        """
        current = self._action_weights.get(action, 1.0)

        if effectiveness > 0.6:
            new_weight = current * self.WEIGHT_BOOST
        elif effectiveness < 0.4:
            new_weight = current * self.WEIGHT_PENALTY
        else:
            new_weight = current

        # Clamp to valid range
        self._action_weights[action] = max(
            self.MIN_WEIGHT,
            min(self.MAX_WEIGHT, new_weight)
        )

        if new_weight != current:
            logger.debug(
                f"Action weight updated: {action} = {current:.2f} -> "
                f"{self._action_weights[action]:.2f} (effectiveness={effectiveness:.2f})"
            )

    def get_action_weight(self, action: str) -> float:
        """Get current weight for an action.

        Higher weights indicate more effective actions that should be preferred.

        Args:
            action: The action type

        Returns:
            Weight multiplier (1.0 = neutral, >1 = preferred, <1 = deprioritized)
        """
        return self._action_weights.get(action, 1.0)

    def get_all_weights(self) -> dict[str, float]:
        """Return all current action weights."""
        return dict(self._action_weights)

    def get_pending_count(self) -> int:
        """Get number of actions pending effectiveness check."""
        return len(self._pending)

    def get_stats(self) -> dict[str, Any]:
        """Return effectiveness stats for monitoring endpoint."""
        return {
            "pending_actions": len(self._pending),
            "completed_actions": len(self._completed),
            "action_weights": dict(self._action_weights),
            "recent_effectiveness": [
                {
                    "action": r.action,
                    "effectiveness": r.effectiveness,
                    "timestamp": r.timestamp,
                    "nodes_affected": len(r.nodes),
                }
                for r in self._completed[-10:]
            ],
        }

    def get_status(self) -> dict[str, Any]:
        """Return status for HTTP endpoint."""
        # Compute average effectiveness per action
        action_effectiveness: dict[str, list[float]] = {}
        for record in self._completed:
            if record.effectiveness is not None:
                if record.action not in action_effectiveness:
                    action_effectiveness[record.action] = []
                action_effectiveness[record.action].append(record.effectiveness)

        avg_effectiveness = {
            action: sum(scores) / len(scores)
            for action, scores in action_effectiveness.items()
            if scores
        }

        return {
            "enabled": EFFECTIVENESS_TRACKING_ENABLED,
            "pending_actions": len(self._pending),
            "completed_actions": len(self._completed),
            "action_weights": dict(self._action_weights),
            "avg_effectiveness": avg_effectiveness,
            "cooldown_seconds": self.COOLDOWN_SECONDS,
            "recent": [r.to_dict() for r in self._completed[-5:]],
        }
