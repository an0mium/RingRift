"""P2P Stability Controller - Automated self-healing for cluster stability.

Jan 2026: Created as part of P2P Self-Healing Architecture.

This controller closes the feedback loop between diagnostic observation and
corrective action:
    Diagnostics -> Symptom Detection -> Root Cause Mapping -> Recovery Action

Key features:
- Subscribes to Phase 0 diagnostic trackers
- Detects stability symptoms (flapping, exhaustion, degradation)
- Maps symptoms to root causes automatically
- Triggers appropriate recovery actions
- Tracks action history for effectiveness analysis
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from scripts.p2p.loops.base import BaseLoop

if TYPE_CHECKING:
    from scripts.p2p.diagnostics.peer_state_tracker import PeerStateTracker
    from scripts.p2p.diagnostics.connection_failure_tracker import ConnectionFailureTracker
    from scripts.p2p.diagnostics.probe_tracker import ProbeEffectivenessTracker

logger = logging.getLogger(__name__)


# Environment variable to disable controller
STABILITY_CONTROLLER_ENABLED = os.environ.get(
    "RINGRIFT_STABILITY_CONTROLLER_ENABLED", "true"
).lower() in ("true", "1", "yes")


class SymptomType(Enum):
    """Detected stability symptoms."""
    PEER_FLAPPING = "peer_flapping"
    CONNECTION_EXHAUSTION = "connection_exhaustion"
    HIGH_FALSE_POSITIVE = "high_false_positive"
    PROVIDER_DEGRADATION = "provider_degradation"
    CIRCUIT_CASCADE = "circuit_cascade"
    TIMEOUT_MISMATCH = "timeout_mismatch"
    LOW_ALIVE_RATIO = "low_alive_ratio"


class RecoveryAction(Enum):
    """Available recovery actions."""
    INCREASE_TIMEOUT = "increase_timeout"
    DECREASE_TIMEOUT = "decrease_timeout"
    INCREASE_COOLDOWN = "increase_cooldown"
    DECREASE_COOLDOWN = "decrease_cooldown"
    RESET_CIRCUIT = "reset_circuit"
    SCALE_POOL_UP = "scale_pool_up"
    REINJECT_PEER = "reinject_peer"
    EMIT_ALERT = "emit_alert"


@dataclass
class SymptomDetection:
    """A detected stability symptom with root cause analysis."""
    symptom: SymptomType
    confidence: float  # 0.0 - 1.0
    affected_nodes: list[str]
    root_cause: str
    suggested_actions: list[RecoveryAction]
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symptom": self.symptom.value,
            "confidence": self.confidence,
            "affected_nodes": self.affected_nodes[:5],  # Limit for display
            "root_cause": self.root_cause,
            "suggested_actions": [a.value for a in self.suggested_actions],
            "detected_at": self.detected_at,
        }


class StabilityController(BaseLoop):
    """
    Automated P2P stability controller.

    Subscribes to diagnostic trackers, detects symptoms, maps to root causes,
    and triggers appropriate recovery actions.

    This is the "reasoning layer" that closes the feedback loop between
    observation and action.
    """

    # Thresholds for symptom detection
    FLAPPING_CONFIDENCE_MULTIPLIER = 0.33  # 3+ flapping = full confidence
    FALSE_POSITIVE_THRESHOLD = 0.15  # >15% FP rate triggers action
    EXHAUSTION_THRESHOLD = 5  # >5 pool_exhausted failures
    PROVIDER_FAILURE_THRESHOLD = 10  # >10 failures on same transport
    LOW_ALIVE_THRESHOLD = 0.5  # <50% alive ratio
    ACTION_CONFIDENCE_THRESHOLD = 0.5  # Only act if confidence >= 50%

    def __init__(
        self,
        peer_state_tracker: PeerStateTracker | None = None,
        connection_failure_tracker: ConnectionFailureTracker | None = None,
        probe_tracker: ProbeEffectivenessTracker | None = None,
        action_callbacks: dict[RecoveryAction, Callable] | None = None,
        check_interval: float = 30.0,
    ):
        """Initialize the stability controller.

        Args:
            peer_state_tracker: Tracker for peer state transitions and flapping
            connection_failure_tracker: Tracker for connection failures by type
            probe_tracker: Tracker for probe effectiveness and false positives
            action_callbacks: Dict mapping RecoveryAction to async callback functions
            check_interval: How often to analyze diagnostics (seconds)
        """
        super().__init__(
            name="stability_controller",
            interval=check_interval,
            enabled=STABILITY_CONTROLLER_ENABLED,
        )
        self._peer_tracker = peer_state_tracker
        self._conn_tracker = connection_failure_tracker
        self._probe_tracker = probe_tracker
        self._action_callbacks = action_callbacks or {}
        self._action_history: list[tuple[float, RecoveryAction, list[str]]] = []
        self._symptom_history: list[SymptomDetection] = []

        # Rate limiting: don't repeat same action type within cooldown
        self._last_action_time: dict[RecoveryAction, float] = {}
        self._action_cooldown = 120.0  # 2 minutes between same action type

        # Track first run for verification logging
        self._first_run_logged = False

        logger.info(
            f"StabilityController initialized: enabled={STABILITY_CONTROLLER_ENABLED}, "
            f"interval={check_interval}s, callbacks={list(action_callbacks.keys()) if action_callbacks else []}"
        )

    async def _run_once(self) -> None:
        """Main analysis loop - runs every check_interval seconds."""
        if not STABILITY_CONTROLLER_ENABLED:
            return

        # Log first successful run for verification
        if not self._first_run_logged:
            logger.info(
                "[StabilityController] First symptom check starting - self-healing ACTIVE"
            )
            self._first_run_logged = True

        symptoms = self._detect_symptoms()

        for symptom in symptoms:
            self._symptom_history.append(symptom)
            logger.info(
                f"Symptom detected: {symptom.symptom.value} "
                f"(confidence={symptom.confidence:.2f}, cause={symptom.root_cause})"
            )
            await self._execute_recovery(symptom)

        self._prune_history()

    def _detect_symptoms(self) -> list[SymptomDetection]:
        """Analyze diagnostics and detect symptoms.

        This is the core "reasoning" logic that maps observations to symptoms.
        """
        symptoms = []

        # 1. Check for peer flapping
        if self._peer_tracker:
            symptoms.extend(self._detect_flapping())

        # 2. Check false positive rate
        if self._probe_tracker:
            symptoms.extend(self._detect_false_positives())

        # 3. Check connection exhaustion
        if self._conn_tracker:
            symptoms.extend(self._detect_connection_issues())

        return symptoms

    def _detect_flapping(self) -> list[SymptomDetection]:
        """Detect peer flapping symptoms."""
        symptoms = []
        try:
            peer_diag = self._peer_tracker.get_diagnostics()
            flapping = peer_diag.get("flapping_peers", [])

            if flapping:
                death_reasons = peer_diag.get("death_reasons", {})
                probe_timeouts = death_reasons.get("probe_timeout", 0)
                gossip_suspects = death_reasons.get("gossip_suspect", 0)

                # Root cause analysis: what's causing the flapping?
                if probe_timeouts > gossip_suspects:
                    root_cause = "Probe timeouts causing flapping - timeouts may be too aggressive"
                    actions = [RecoveryAction.INCREASE_TIMEOUT]
                else:
                    root_cause = "Gossip suspects causing flapping - network instability or cooldown too short"
                    actions = [RecoveryAction.INCREASE_COOLDOWN]

                symptoms.append(SymptomDetection(
                    symptom=SymptomType.PEER_FLAPPING,
                    confidence=min(len(flapping) * self.FLAPPING_CONFIDENCE_MULTIPLIER, 1.0),
                    affected_nodes=flapping,
                    root_cause=root_cause,
                    suggested_actions=actions,
                ))

            # Check alive ratio
            alive = peer_diag.get("alive_count", 0)
            dead = peer_diag.get("dead_count", 0)
            total = alive + dead
            if total > 0:
                alive_ratio = alive / total
                if alive_ratio < self.LOW_ALIVE_THRESHOLD:
                    symptoms.append(SymptomDetection(
                        symptom=SymptomType.LOW_ALIVE_RATIO,
                        confidence=1.0 - alive_ratio,
                        affected_nodes=[],
                        root_cause=f"Low alive ratio ({alive_ratio:.1%}) - cluster health degraded",
                        suggested_actions=[RecoveryAction.RESET_CIRCUIT, RecoveryAction.EMIT_ALERT],
                    ))

        except Exception as e:
            logger.warning(f"Error detecting flapping: {e}")

        return symptoms

    def _detect_false_positives(self) -> list[SymptomDetection]:
        """Detect high false positive rate symptoms."""
        symptoms = []
        try:
            probe_diag = self._probe_tracker.get_diagnostics()
            fp_rate = probe_diag.get("false_positive_rate", 0)

            if fp_rate > self.FALSE_POSITIVE_THRESHOLD:
                worst_nodes = [n for n, _ in probe_diag.get("worst_success_rates", [])]
                symptoms.append(SymptomDetection(
                    symptom=SymptomType.HIGH_FALSE_POSITIVE,
                    confidence=min(fp_rate * 2, 1.0),
                    affected_nodes=worst_nodes,
                    root_cause=f"High false positive rate ({fp_rate:.1%}) - probes timing out on healthy nodes",
                    suggested_actions=[RecoveryAction.INCREASE_TIMEOUT],
                ))

        except Exception as e:
            logger.warning(f"Error detecting false positives: {e}")

        return symptoms

    def _detect_connection_issues(self) -> list[SymptomDetection]:
        """Detect connection-related symptoms."""
        symptoms = []
        try:
            conn_diag = self._conn_tracker.get_diagnostics()

            # Check pool exhaustion
            by_type = conn_diag.get("by_type", {})
            exhausted = by_type.get("pool_exhausted", 0)
            if exhausted > self.EXHAUSTION_THRESHOLD:
                worst_nodes = [n for n, _ in conn_diag.get("worst_nodes", [])]
                symptoms.append(SymptomDetection(
                    symptom=SymptomType.CONNECTION_EXHAUSTION,
                    confidence=min(exhausted / 10, 1.0),
                    affected_nodes=worst_nodes,
                    root_cause=f"Connection pool exhaustion ({exhausted} failures) - pool too small for cluster",
                    suggested_actions=[RecoveryAction.SCALE_POOL_UP],
                ))

            # Check provider/transport degradation
            by_transport = conn_diag.get("by_transport", {})
            for transport, count in by_transport.items():
                if count > self.PROVIDER_FAILURE_THRESHOLD:
                    symptoms.append(SymptomDetection(
                        symptom=SymptomType.PROVIDER_DEGRADATION,
                        confidence=min(count / 20, 1.0),
                        affected_nodes=[],
                        root_cause=f"Transport '{transport}' degraded ({count} failures in 10min)",
                        suggested_actions=[RecoveryAction.INCREASE_TIMEOUT, RecoveryAction.EMIT_ALERT],
                    ))

        except Exception as e:
            logger.warning(f"Error detecting connection issues: {e}")

        return symptoms

    async def _execute_recovery(self, symptom: SymptomDetection) -> None:
        """Execute recovery actions for a detected symptom."""
        if symptom.confidence < self.ACTION_CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Skipping action for {symptom.symptom.value} "
                f"(confidence {symptom.confidence:.2f} < threshold {self.ACTION_CONFIDENCE_THRESHOLD})"
            )
            return

        now = time.time()
        for action in symptom.suggested_actions:
            # Rate limiting
            last_time = self._last_action_time.get(action, 0)
            if now - last_time < self._action_cooldown:
                logger.debug(
                    f"Rate limiting {action.value} "
                    f"(last action {now - last_time:.0f}s ago, cooldown {self._action_cooldown}s)"
                )
                continue

            callback = self._action_callbacks.get(action)
            if callback:
                try:
                    await callback(symptom.affected_nodes, symptom)
                    self._action_history.append((now, action, symptom.affected_nodes))
                    self._last_action_time[action] = now
                    logger.info(
                        f"Executed recovery action: {action.value} "
                        f"for {len(symptom.affected_nodes)} nodes"
                    )
                except Exception as e:
                    logger.error(f"Recovery action {action.value} failed: {e}")
            else:
                logger.debug(f"No callback registered for action: {action.value}")

    def _prune_history(self) -> None:
        """Remove old history entries (keep 1 hour)."""
        cutoff = time.time() - 3600
        self._symptom_history = [s for s in self._symptom_history if s.detected_at > cutoff]
        self._action_history = [(t, a, n) for t, a, n in self._action_history if t > cutoff]

    def get_status(self) -> dict[str, Any]:
        """Return controller status for HTTP endpoint."""
        return {
            "enabled": STABILITY_CONTROLLER_ENABLED,
            "running": self._running,
            "symptoms_last_hour": len(self._symptom_history),
            "actions_last_hour": len(self._action_history),
            "recent_symptoms": [s.to_dict() for s in self._symptom_history[-5:]],
            "recent_actions": [
                {"timestamp": t, "action": a.value, "nodes": n[:3]}
                for t, a, n in self._action_history[-5:]
            ],
            "action_cooldowns": {
                a.value: max(0, self._action_cooldown - (time.time() - t))
                for a, t in self._last_action_time.items()
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Return health check result for DaemonManager integration."""
        now = time.time()
        recent_symptoms = [s for s in self._symptom_history if now - s.detected_at < 300]
        recent_actions = [(t, a, n) for t, a, n in self._action_history if now - t < 300]

        # Healthy if we're running and not overwhelmed with symptoms
        is_healthy = (
            STABILITY_CONTROLLER_ENABLED and
            self._running and
            len(recent_symptoms) < 10  # <10 symptoms in 5 min is OK
        )

        return {
            "healthy": is_healthy,
            "enabled": STABILITY_CONTROLLER_ENABLED,
            "running": self._running,
            "symptoms_5min": len(recent_symptoms),
            "actions_5min": len(recent_actions),
        }
