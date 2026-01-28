"""SelfplayHealthMonitor - P2P health event handlers for selfplay scheduling.

January 2026 Sprint 17.4: Extracted from selfplay_scheduler.py as part of
the Phase 3 decomposition effort. This mixin provides handlers for P2P
cluster health events that affect selfplay allocation decisions.

Responsibilities:
- Track unhealthy/demoted nodes and exclude from allocation
- Handle backpressure signals to reduce selfplay rate
- Respond to cluster health changes
- Track node overload and apply backoff
- Handle progress stall/recovery events

Usage:
    class SelfplayScheduler(SelfplayHealthMonitorMixin, HandlerBase):
        def __init__(self):
            super().__init__()
            self._init_health_monitor_state()

Event Subscriptions (via _get_health_event_subscriptions):
- NODE_UNHEALTHY, P2P_NODE_DEAD
- NODE_RECOVERED
- HOST_OFFLINE
- VOTER_DEMOTED, VOTER_PROMOTED
- CIRCUIT_RESET
- P2P_CLUSTER_UNHEALTHY, P2P_CLUSTER_HEALTHY
- P2P_RESTARTED
- BACKPRESSURE_ACTIVATED, BACKPRESSURE_RELEASED
- EVALUATION_BACKPRESSURE, EVALUATION_BACKPRESSURE_RELEASED
- NODE_OVERLOADED
- PROGRESS_STALL_DETECTED, PROGRESS_RECOVERED
"""

from __future__ import annotations

__all__ = [
    "SelfplayHealthMonitorMixin",
    "HealthMonitorState",
]

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from app.coordination.node_allocator import NodeCapability

logger = logging.getLogger(__name__)


@dataclass
class HealthMonitorState:
    """State tracked by the health monitor mixin.

    January 2026: Extracted to facilitate testing and state inspection.
    """
    # Node health tracking
    unhealthy_nodes: set[str] = field(default_factory=set)
    demoted_voters: set[str] = field(default_factory=set)
    overloaded_nodes: dict[str, float] = field(default_factory=dict)  # node_id -> backoff_expiry

    # Cluster health
    cluster_health_factor: float = 1.0

    # Backpressure tracking
    backpressure_active: bool = False
    eval_backpressure_active: bool = False

    # Circuit reset stats
    circuit_reset_count: int = 0

    # Session 17.35: HOST_OFFLINE auto-recovery tracking
    host_offline_timestamps: dict[str, float] = field(default_factory=dict)  # host -> offline_time


class SelfplayHealthMonitorMixin:
    """Mixin providing P2P health event handlers for selfplay scheduling.

    This mixin manages node health state and handles events that affect
    which nodes can receive selfplay job allocations.

    Required attributes from parent class:
    - _node_capabilities: dict[str, NodeCapability]
    - _config_priorities: dict[str, ConfigPriority]
    - _exploration_boost: float
    - _subscribed: bool
    - subscribe_to_events(): Callable

    State managed by this mixin (initialized via _init_health_monitor_state):
    - _unhealthy_nodes: set[str]
    - _demoted_voters: set[str]
    - _overloaded_nodes: dict[str, float]
    - _cluster_health_factor: float
    - _backpressure_active: bool
    - _eval_backpressure_active: bool
    - _circuit_reset_count: int
    """

    # Type hints for attributes from parent class
    _node_capabilities: dict[str, "NodeCapability"]
    _subscribed: bool
    _exploration_boost: float

    def _init_health_monitor_state(self) -> None:
        """Initialize health monitor state.

        Call this from the parent class __init__ method.
        """
        self._unhealthy_nodes: set[str] = set()
        self._demoted_voters: set[str] = set()
        self._overloaded_nodes: dict[str, float] = {}
        self._cluster_health_factor: float = 1.0
        self._backpressure_active: bool = False
        self._eval_backpressure_active: bool = False
        self._circuit_reset_count: int = 0
        # Session 17.35: Track HOST_OFFLINE timestamps for auto-recovery
        self._host_offline_timestamps: dict[str, float] = {}
        self._host_offline_recovery_seconds: float = 1800.0  # 30 minutes

        # Jan 2026 P1: Stall escalation tracking for progressive intervention
        # Level 0: No stall, Level 1: 6h (1.5x), Level 2: 48h (2.0x), Level 3: 96h+ (3.0x)
        self._stall_escalation_level: dict[str, int] = {}
        self._stall_start_times: dict[str, float] = {}

    def _get_health_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for health monitoring.

        Call this from _get_event_subscriptions() and merge with other subscriptions.

        Returns:
            Dict mapping event type strings to handler methods.
        """
        return {
            # Node health events
            "NODE_UNHEALTHY": self._on_node_unhealthy,
            "P2P_NODE_DEAD": self._on_node_unhealthy,
            "NODE_RECOVERED": self._on_node_recovered,
            "HOST_OFFLINE": self._on_host_offline,

            # Voter events
            "VOTER_DEMOTED": self._on_voter_demoted,
            "VOTER_PROMOTED": self._on_voter_promoted,

            # Circuit breaker events
            "CIRCUIT_RESET": self._on_circuit_reset,

            # Cluster health events
            "P2P_CLUSTER_UNHEALTHY": self._on_cluster_unhealthy,
            "P2P_CLUSTER_HEALTHY": self._on_cluster_healthy,
            "P2P_RESTARTED": self._on_p2p_restarted,

            # Backpressure events
            "BACKPRESSURE_ACTIVATED": self._on_backpressure_activated,
            "BACKPRESSURE_RELEASED": self._on_backpressure_released,
            "EVALUATION_BACKPRESSURE": self._on_evaluation_backpressure,
            "EVALUATION_BACKPRESSURE_RELEASED": self._on_backpressure_released,
            # Jan 2026: Work queue specific backpressure events
            "WORK_QUEUE_BACKPRESSURE": self._on_work_queue_backpressure,
            "WORK_QUEUE_BACKPRESSURE_RELEASED": self._on_work_queue_backpressure_released,

            # Resource events
            "NODE_OVERLOADED": self._on_node_overloaded,

            # Progress events
            "PROGRESS_STALL_DETECTED": self._on_progress_stall,
            "PROGRESS_RECOVERED": self._on_progress_recovered,
        }

    def get_health_monitor_state(self) -> HealthMonitorState:
        """Get current health monitor state for inspection/testing.

        Returns:
            HealthMonitorState dataclass with current state values.
        """
        return HealthMonitorState(
            unhealthy_nodes=set(getattr(self, "_unhealthy_nodes", set())),
            demoted_voters=set(getattr(self, "_demoted_voters", set())),
            overloaded_nodes=dict(getattr(self, "_overloaded_nodes", {})),
            cluster_health_factor=getattr(self, "_cluster_health_factor", 1.0),
            backpressure_active=getattr(self, "_backpressure_active", False),
            eval_backpressure_active=getattr(self, "_eval_backpressure_active", False),
            circuit_reset_count=getattr(self, "_circuit_reset_count", 0),
            host_offline_timestamps=dict(getattr(self, "_host_offline_timestamps", {})),
        )

    # =========================================================================
    # Node Health Event Handlers
    # =========================================================================

    def _on_node_unhealthy(self, event: Any) -> None:
        """Handle NODE_UNHEALTHY or P2P_NODE_DEAD - mark node as unavailable.

        December 2025: Prevents allocating selfplay to failing/unhealthy nodes.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            node_id = payload.get("node_id", "")
            reason = payload.get("reason", "unknown")

            if node_id:
                if not hasattr(self, "_unhealthy_nodes"):
                    self._unhealthy_nodes: set[str] = set()

                self._unhealthy_nodes.add(node_id)

                # Also mark node as unavailable in capabilities if tracked
                if node_id in self._node_capabilities:
                    self._node_capabilities[node_id].current_load = 1.0  # Mark as fully loaded

                logger.warning(
                    f"[SelfplayScheduler] Node {node_id} marked unhealthy: {reason}. "
                    f"Will not allocate selfplay to this node."
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling node unhealthy: {e}")

    def _on_node_recovered(self, event: Any) -> None:
        """Handle NODE_RECOVERED - re-enable node for allocation.

        December 2025: Restores node availability after recovery.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            node_id = payload.get("node_id", "")

            if node_id:
                unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())
                if node_id in unhealthy_nodes:
                    self._unhealthy_nodes.discard(node_id)

                    logger.info(
                        f"[SelfplayScheduler] Node {node_id} recovered. "
                        f"Re-enabled for selfplay allocation."
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling node recovered: {e}")

    def _on_host_offline(self, event: Any) -> None:
        """Handle HOST_OFFLINE - mark P2P peer as unavailable after retirement.

        December 2025: P2P orchestrator emits this when a peer is retired after
        ~300s of being offline. This is a stronger signal than NODE_UNHEALTHY
        as the node has been definitively removed from the cluster.

        Session 17.35: Track timestamp for auto-recovery after 30 minutes.
        Nodes may come back online without explicit HOST_ONLINE events.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "") or payload.get("node_id", "")
            reason = payload.get("reason", "retired")

            if host:
                if not hasattr(self, "_unhealthy_nodes"):
                    self._unhealthy_nodes: set[str] = set()
                if not hasattr(self, "_host_offline_timestamps"):
                    self._host_offline_timestamps: dict[str, float] = {}

                self._unhealthy_nodes.add(host)
                # Track when the host went offline for auto-recovery
                self._host_offline_timestamps[host] = time.time()

                # Mark node as fully loaded to prevent allocation
                if host in self._node_capabilities:
                    self._node_capabilities[host].current_load = 1.0

                logger.warning(
                    f"[SelfplayScheduler] Host {host} offline (reason: {reason}). "
                    f"Removed from selfplay allocation pool. "
                    f"Will auto-recover in 30 minutes if not back sooner."
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling host offline: {e}")

    # =========================================================================
    # Voter Event Handlers
    # =========================================================================

    def _on_voter_demoted(self, event: Any) -> None:
        """Handle VOTER_DEMOTED - mark voter node as having potential issues.

        Jan 3, 2026: When a voter is demoted due to health issues, it signals
        the node may be unstable. We reduce its allocation weight to minimize
        impact on selfplay throughput.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            node_id = payload.get("node_id", "") or payload.get("host", "")
            reason = payload.get("reason", "health_check_failed")

            if node_id:
                # Track demoted voters separately (subset of unhealthy nodes)
                if not hasattr(self, "_demoted_voters"):
                    self._demoted_voters: set[str] = set()

                self._demoted_voters.add(node_id)

                # Reduce allocation weight for this node (not full exclusion)
                # Voters are important nodes, so we reduce but don't fully disable
                if node_id in self._node_capabilities:
                    # Reduce to 25% capacity instead of full exclusion
                    self._node_capabilities[node_id].current_load = 0.75

                logger.warning(
                    f"[SelfplayScheduler] Voter {node_id} demoted (reason: {reason}). "
                    f"Reduced allocation weight to 25%."
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling voter demoted: {e}")

    def _on_voter_promoted(self, event: Any) -> None:
        """Handle VOTER_PROMOTED - restore voter node to full allocation.

        Jan 3, 2026: When a voter is re-promoted after recovery, restore
        its allocation capacity to full.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            node_id = payload.get("node_id", "") or payload.get("host", "")

            if node_id:
                demoted_voters = getattr(self, "_demoted_voters", set())
                if node_id in demoted_voters:
                    self._demoted_voters.discard(node_id)

                    # Restore full allocation capacity
                    if node_id in self._node_capabilities:
                        self._node_capabilities[node_id].current_load = 0.0

                    logger.info(
                        f"[SelfplayScheduler] Voter {node_id} re-promoted. "
                        f"Restored to full allocation capacity."
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling voter promoted: {e}")

    # =========================================================================
    # Circuit Breaker Event Handlers
    # =========================================================================

    def _on_circuit_reset(self, event: Any) -> None:
        """Handle CIRCUIT_RESET - node recovered via proactive health probe.

        Jan 3, 2026 Session 10: When a circuit breaker is proactively reset
        after a successful health probe (vs waiting for CB timeout), the node
        is confirmed healthy. We restore full allocation and optionally give
        a brief priority boost to help the recovered node catch up.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            node_id = payload.get("node_id", "") or payload.get("peer_id", "")
            source = payload.get("source", "unknown")
            recovery_time_ms = payload.get("recovery_time_ms", 0)

            if node_id:
                # Track circuit reset stats
                if not hasattr(self, "_circuit_reset_count"):
                    self._circuit_reset_count: int = 0
                self._circuit_reset_count += 1

                # Remove from demoted voters if present
                demoted_voters = getattr(self, "_demoted_voters", set())
                was_demoted = node_id in demoted_voters
                if was_demoted:
                    self._demoted_voters.discard(node_id)

                # Remove from unhealthy nodes if present
                unhealthy = getattr(self, "_unhealthy_nodes", set())
                was_unhealthy = node_id in unhealthy
                if was_unhealthy:
                    self._unhealthy_nodes.discard(node_id)

                # Restore full allocation capacity
                if node_id in self._node_capabilities:
                    self._node_capabilities[node_id].current_load = 0.0

                logger.info(
                    f"[SelfplayScheduler] CIRCUIT_RESET: Node {node_id} recovered "
                    f"via proactive probe (source={source}, time={recovery_time_ms}ms). "
                    f"Restored to full capacity. "
                    f"Total circuit resets: {self._circuit_reset_count}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling circuit reset: {e}")

    # =========================================================================
    # Cluster Health Event Handlers
    # =========================================================================

    def _on_cluster_unhealthy(self, event: Any) -> None:
        """Handle P2P_CLUSTER_UNHEALTHY - reduce allocation rate.

        December 2025: When cluster health degrades, reduce overall selfplay
        allocation to avoid overwhelming healthy nodes.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            healthy_nodes = payload.get("healthy_nodes", 0)
            total_nodes = payload.get("total_nodes", 0)

            if not hasattr(self, "_cluster_health_factor"):
                self._cluster_health_factor = 1.0

            if total_nodes > 0:
                self._cluster_health_factor = max(0.3, healthy_nodes / total_nodes)
            else:
                self._cluster_health_factor = 0.5

            logger.warning(
                f"[SelfplayScheduler] Cluster unhealthy: {healthy_nodes}/{total_nodes}. "
                f"Reducing allocation to {self._cluster_health_factor:.0%}."
            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling cluster unhealthy: {e}")

    def _on_cluster_healthy(self, event: Any) -> None:
        """Handle P2P_CLUSTER_HEALTHY - restore normal allocation.

        December 2025: Restores full allocation when cluster recovers.
        """
        try:
            logger.info("[SelfplayScheduler] Cluster healthy. Restoring normal allocation.")

            if hasattr(self, "_cluster_health_factor"):
                self._cluster_health_factor = 1.0

            # Clear unhealthy node tracking
            if hasattr(self, "_unhealthy_nodes"):
                self._unhealthy_nodes.clear()

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling cluster healthy: {e}")

    def _on_p2p_restarted(self, event: Any) -> None:
        """Handle P2P_RESTARTED - ensure subscriptions are valid.

        Dec 30, 2025: After a P2P restart, event subscriptions may be lost if
        the event bus was reinitialized. This handler verifies and re-establishes
        subscriptions to maintain pipeline resilience.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            restart_count = payload.get("restart_count", 0)
            trigger = payload.get("trigger", "unknown")

            logger.info(
                f"[SelfplayScheduler] P2P restarted (count={restart_count}, trigger={trigger}). "
                f"Verifying subscriptions..."
            )

            # Ensure subscriptions are still active
            if hasattr(self, "ensure_subscribed"):
                self.ensure_subscribed()  # type: ignore

            # Trigger a priority update to refresh state from recovered P2P
            if hasattr(self, "_priority_update_pending"):
                self._priority_update_pending = True  # type: ignore

            logger.info("[SelfplayScheduler] Subscription verification complete after P2P restart")

        except Exception as e:
            logger.error(f"[SelfplayScheduler] Error handling P2P restart: {e}")

    # =========================================================================
    # Backpressure Event Handlers
    # =========================================================================

    def _on_backpressure_activated(self, event: Any) -> None:
        """Handle BACKPRESSURE_ACTIVATED - reduce selfplay rate.

        Dec 29, 2025: When work queue hits backpressure limits, reduce selfplay
        rate to prevent generating games that can't be processed.

        Args:
            event: Event with payload containing pending_count, trigger, limits
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            pending_count = payload.get("pending_count", 0)
            trigger = payload.get("trigger", "unknown")
            soft_limit = payload.get("soft_limit", 500)

            # Track backpressure state
            if not hasattr(self, "_backpressure_active"):
                self._backpressure_active = False
            self._backpressure_active = True

            # Calculate reduction factor based on how far over soft limit we are
            overage = pending_count - soft_limit
            reduction_factor = max(0.5, 1.0 - (overage / soft_limit) * 0.5)

            # Apply reduction to exploration boost (reduces all allocations)
            if hasattr(self, "_exploration_boost"):
                old_boost = self._exploration_boost
                self._exploration_boost = max(0.5, self._exploration_boost * reduction_factor)
                logger.warning(
                    f"[SelfplayScheduler] Backpressure activated ({trigger}): "
                    f"queue={pending_count}, exploration_boost {old_boost:.2f} -> {self._exploration_boost:.2f}"
                )
            else:
                logger.warning(
                    f"[SelfplayScheduler] Backpressure activated ({trigger}): queue={pending_count}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling backpressure activated: {e}")

    def _on_evaluation_backpressure(self, event: Any) -> None:
        """Handle EVALUATION_BACKPRESSURE - reduce selfplay rate when evaluation queue is backlogged.

        January 2026 Sprint 10: When evaluation queue exceeds threshold, reduce selfplay
        rate to prevent generating more games that will just add to evaluation backlog.
        This complements BACKPRESSURE_ACTIVATED (work queue) with evaluation-specific handling.

        Args:
            event: Event with payload containing queue_depth, threshold
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            queue_depth = payload.get("queue_depth", 0)
            threshold = payload.get("threshold", 70)

            # Track evaluation backpressure state separately
            if not hasattr(self, "_eval_backpressure_active"):
                self._eval_backpressure_active = False
            self._eval_backpressure_active = True

            # Calculate reduction factor: more aggressive for evaluation backpressure
            # since evaluation is a harder bottleneck than work queue
            overage = queue_depth - threshold
            reduction_factor = max(0.3, 1.0 - (overage / threshold) * 0.7)

            # Apply reduction to exploration boost
            if hasattr(self, "_exploration_boost"):
                old_boost = self._exploration_boost
                self._exploration_boost = max(0.3, self._exploration_boost * reduction_factor)
                logger.warning(
                    f"[SelfplayScheduler] Evaluation backpressure: "
                    f"queue_depth={queue_depth}, exploration_boost {old_boost:.2f} -> {self._exploration_boost:.2f}"
                )
            else:
                logger.warning(
                    f"[SelfplayScheduler] Evaluation backpressure: queue_depth={queue_depth}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling evaluation backpressure: {e}")

    def _on_backpressure_released(self, event: Any) -> None:
        """Handle BACKPRESSURE_RELEASED and EVALUATION_BACKPRESSURE_RELEASED.

        Dec 29, 2025: When work queue drains below recovery threshold,
        restore normal selfplay allocation rates.

        Jan 2026 Sprint 10: Also handles EVALUATION_BACKPRESSURE_RELEASED to restore
        selfplay rate when evaluation queue drains.

        Args:
            event: Event with payload containing pending_count or queue_depth
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            # Support both work queue and evaluation queue payloads
            pending_count = payload.get("pending_count", payload.get("queue_depth", 0))

            # Track backpressure state (both types)
            if not hasattr(self, "_backpressure_active"):
                self._backpressure_active = False
            if not hasattr(self, "_eval_backpressure_active"):
                self._eval_backpressure_active = False

            was_active = self._backpressure_active or self._eval_backpressure_active
            self._backpressure_active = False
            self._eval_backpressure_active = False

            if was_active:
                # Restore exploration boost gradually
                if hasattr(self, "_exploration_boost") and self._exploration_boost < 1.0:
                    old_boost = self._exploration_boost
                    self._exploration_boost = min(1.0, self._exploration_boost * 1.5)
                    logger.info(
                        f"[SelfplayScheduler] Backpressure released: "
                        f"queue={pending_count}, exploration_boost {old_boost:.2f} -> {self._exploration_boost:.2f}"
                    )
                else:
                    logger.info(
                        f"[SelfplayScheduler] Backpressure released: queue={pending_count}"
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling backpressure released: {e}")

    def _on_work_queue_backpressure(self, event: Any) -> None:
        """Handle WORK_QUEUE_BACKPRESSURE - reduce selfplay allocation.

        January 2026: When work queue backpressure crosses MEDIUM threshold,
        reduce selfplay allocation to prevent queue overflow during export latency.

        Args:
            event: Event with payload containing level, reduction_factor, queue_depth, utilization
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            level = payload.get("level", "unknown")
            reduction_factor = payload.get("reduction_factor", 0.25)
            queue_depth = payload.get("queue_depth", 0)
            utilization = payload.get("utilization", 0.0)

            # Track work queue backpressure state
            if not hasattr(self, "_work_queue_backpressure_active"):
                self._work_queue_backpressure_active = False
            if not hasattr(self, "_work_queue_reduction_factor"):
                self._work_queue_reduction_factor = 1.0

            self._work_queue_backpressure_active = True
            self._work_queue_reduction_factor = reduction_factor

            # Apply reduction to exploration boost
            if hasattr(self, "_exploration_boost"):
                old_boost = self._exploration_boost
                self._exploration_boost = max(0.25, self._exploration_boost * reduction_factor)
                logger.warning(
                    f"[SelfplayScheduler] Work queue backpressure: level={level}, "
                    f"queue_depth={queue_depth}, utilization={utilization:.1%}, "
                    f"exploration_boost {old_boost:.2f} -> {self._exploration_boost:.2f}"
                )
            else:
                logger.warning(
                    f"[SelfplayScheduler] Work queue backpressure: level={level}, "
                    f"queue_depth={queue_depth}, utilization={utilization:.1%}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling work queue backpressure: {e}")

    def _on_work_queue_backpressure_released(self, event: Any) -> None:
        """Handle WORK_QUEUE_BACKPRESSURE_RELEASED - restore selfplay allocation.

        January 2026: When work queue backpressure drops below MEDIUM threshold,
        restore normal selfplay allocation.

        Args:
            event: Event with payload containing level, previous_level, queue_depth, utilization
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            level = payload.get("level", "unknown")
            previous_level = payload.get("previous_level", "unknown")
            queue_depth = payload.get("queue_depth", 0)

            # Track state
            if not hasattr(self, "_work_queue_backpressure_active"):
                self._work_queue_backpressure_active = False

            was_active = self._work_queue_backpressure_active
            self._work_queue_backpressure_active = False
            self._work_queue_reduction_factor = 1.0

            if was_active:
                # Restore exploration boost gradually
                if hasattr(self, "_exploration_boost") and self._exploration_boost < 1.0:
                    old_boost = self._exploration_boost
                    self._exploration_boost = min(1.0, self._exploration_boost * 1.5)
                    logger.info(
                        f"[SelfplayScheduler] Work queue backpressure released: "
                        f"{previous_level} -> {level}, queue_depth={queue_depth}, "
                        f"exploration_boost {old_boost:.2f} -> {self._exploration_boost:.2f}"
                    )
                else:
                    logger.info(
                        f"[SelfplayScheduler] Work queue backpressure released: "
                        f"{previous_level} -> {level}, queue_depth={queue_depth}"
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling work queue backpressure released: {e}")

    # =========================================================================
    # Resource Event Handlers
    # =========================================================================

    def _on_node_overloaded(self, event: Any) -> None:
        """Handle NODE_OVERLOADED - add backoff period for overloaded node.

        Dec 29, 2025: When a node reports high CPU/GPU/memory utilization,
        temporarily reduce job dispatch to that node.

        Args:
            event: Event with payload containing host, cpu_percent, gpu_percent,
                   memory_percent, resource_type
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "")
            cpu_pct = payload.get("cpu_percent", 0)
            gpu_pct = payload.get("gpu_percent", 0)
            memory_pct = payload.get("memory_percent", 0)
            resource_type = payload.get("resource_type", "unknown")

            if not host:
                return

            # Initialize overloaded nodes tracking if needed
            if not hasattr(self, "_overloaded_nodes"):
                self._overloaded_nodes: dict[str, float] = {}

            # Add node to overloaded set with backoff timestamp (60 seconds default)
            backoff_duration = 60.0
            if resource_type == "consecutive_failures":
                backoff_duration = 120.0  # Longer backoff for failures
            elif resource_type == "memory":
                backoff_duration = 90.0  # Memory issues take longer to resolve

            self._overloaded_nodes[host] = time.time() + backoff_duration

            logger.warning(
                f"[SelfplayScheduler] Node overloaded ({resource_type}): {host} - "
                f"CPU={cpu_pct:.0f}%, GPU={gpu_pct:.0f}%, MEM={memory_pct:.0f}%, "
                f"backoff={backoff_duration:.0f}s"
            )

            # Clean up expired backoffs
            current_time = time.time()
            expired = [n for n, t in self._overloaded_nodes.items() if t < current_time]
            for n in expired:
                del self._overloaded_nodes[n]

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling node overloaded: {e}")

    # =========================================================================
    # Progress Event Handlers
    # =========================================================================

    def _get_stall_escalation_level(self, stall_duration_hours: float) -> int:
        """Determine escalation level based on stall duration.

        Jan 2026 P1: Tiered escalation for progressive intervention.

        | Level | Duration | Multiplier | Additional Actions |
        |-------|----------|------------|-------------------|
        | 0     | <6h      | 1.0x       | None              |
        | 1     | 6-48h    | 1.5x       | Priority boost    |
        | 2     | 48-96h   | 2.0x       | + Diversity injection |
        | 3     | 96h+     | 3.0x       | + Architecture exploration |

        Args:
            stall_duration_hours: How long the config has been stalled.

        Returns:
            Escalation level 0-3.
        """
        if stall_duration_hours < 6:
            return 0
        elif stall_duration_hours < 48:
            return 1
        elif stall_duration_hours < 96:
            return 2
        else:
            return 3

    def _get_escalation_multiplier(self, level: int) -> float:
        """Get priority multiplier for escalation level."""
        multipliers = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0}
        return multipliers.get(level, 1.0)

    def _on_progress_stall(self, event: Any) -> None:
        """Handle PROGRESS_STALL_DETECTED - tiered escalation for stalled configs.

        Jan 2026 P1: Implements progressive intervention based on stall duration:
        - Level 1 (6h): 1.5x priority boost
        - Level 2 (48h): 2.0x boost + emit OPPONENT_DIVERSITY_NEEDED
        - Level 3 (96h+): 3.0x boost + emit ARCHITECTURE_EXPLORATION_NEEDED

        Args:
            event: Event with payload containing config_key, stall_duration_hours, etc.
        """
        try:
            from app.coordination.event_handler_utils import extract_config_key

            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            stall_duration = payload.get("stall_duration_hours", 0.0)

            if not config_key:
                return

            # Access parent class _config_priorities if available
            config_priorities = getattr(self, "_config_priorities", {})
            if config_key not in config_priorities:
                return

            priority = config_priorities[config_key]

            # Determine escalation level based on stall duration
            new_level = self._get_stall_escalation_level(stall_duration)
            old_level = self._stall_escalation_level.get(config_key, 0)

            # Track stall start time if not already tracking
            if config_key not in self._stall_start_times:
                self._stall_start_times[config_key] = time.time()

            # Update escalation level
            self._stall_escalation_level[config_key] = new_level

            # Apply escalation multiplier to exploration boost
            multiplier = self._get_escalation_multiplier(new_level)
            priority.exploration_boost = max(priority.exploration_boost, multiplier)

            # Increase staleness to prioritize this config
            priority.staleness_hours = max(priority.staleness_hours, stall_duration * 2.0)

            # Log escalation
            level_changed = new_level > old_level
            if level_changed:
                logger.warning(
                    f"[SelfplayScheduler] Stall ESCALATED for {config_key}: "
                    f"level {old_level} -> {new_level} (stalled {stall_duration:.1f}h, {multiplier}x boost)"
                )
            else:
                logger.info(
                    f"[SelfplayScheduler] Progress stall for {config_key}: "
                    f"level {new_level} (stalled {stall_duration:.1f}h, {multiplier}x boost)"
                )

            # Emit additional events at higher escalation levels
            if new_level >= 2 and (level_changed or old_level < 2):
                self._emit_opponent_diversity_needed(config_key, stall_duration)

            if new_level >= 3 and (level_changed or old_level < 3):
                self._emit_architecture_exploration_needed(config_key, stall_duration)

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling progress stall: {e}")

    def _emit_opponent_diversity_needed(self, config_key: str, stall_duration: float) -> None:
        """Emit OPPONENT_DIVERSITY_NEEDED to trigger curriculum diversity injection.

        Jan 2026 P1: At escalation level 2 (48h+ stall), inject opponent diversity
        to help break through local optimum.
        """
        try:
            from app.coordination.safe_event_emit import safe_emit_event

            safe_emit_event(
                "OPPONENT_DIVERSITY_NEEDED",
                {
                    "config_key": config_key,
                    "stall_duration_hours": stall_duration,
                    "escalation_level": 2,
                    "suggested_mix": {
                        "best": 0.30,
                        "previous": 0.40,
                        "heuristic": 0.20,
                        "random": 0.10,
                    },
                },
            )
            logger.info(
                f"[SelfplayScheduler] Emitted OPPONENT_DIVERSITY_NEEDED for {config_key}"
            )
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error emitting diversity event: {e}")

    def _emit_architecture_exploration_needed(self, config_key: str, stall_duration: float) -> None:
        """Emit ARCHITECTURE_EXPLORATION_NEEDED for severe stalls.

        Jan 2026 P1: At escalation level 3 (96h+ stall), signal that architecture
        changes may be needed to break through the plateau.
        """
        try:
            from app.coordination.safe_event_emit import safe_emit_event

            safe_emit_event(
                "ARCHITECTURE_EXPLORATION_NEEDED",
                {
                    "config_key": config_key,
                    "stall_duration_hours": stall_duration,
                    "escalation_level": 3,
                    "suggested_actions": [
                        "Try different model architecture",
                        "Increase model capacity",
                        "Adjust learning rate schedule",
                        "Consider transfer learning from similar config",
                    ],
                },
            )
            logger.warning(
                f"[SelfplayScheduler] Emitted ARCHITECTURE_EXPLORATION_NEEDED for {config_key} "
                f"(stalled {stall_duration:.1f}h - severe)"
            )
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error emitting architecture event: {e}")

    def _on_progress_recovered(self, event: Any) -> None:
        """Handle PROGRESS_RECOVERED - reset boost and escalation for recovered config.

        Jan 2026 P1: When a config recovers from a stall, reset the exploration
        boost and clear escalation tracking to allow other configs to get resources.

        Args:
            event: Event with payload containing config_key, velocity, etc.
        """
        try:
            from app.coordination.event_handler_utils import extract_config_key

            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            new_velocity = payload.get("current_velocity", 0.0)

            if not config_key:
                return

            # Access parent class _config_priorities if available
            config_priorities = getattr(self, "_config_priorities", {})
            if config_key not in config_priorities:
                return

            priority = config_priorities[config_key]

            # Get previous escalation level for logging
            old_level = self._stall_escalation_level.get(config_key, 0)

            # Reset exploration boost to normal
            priority.exploration_boost = 1.0

            # Clear escalation tracking
            self._stall_escalation_level.pop(config_key, None)
            self._stall_start_times.pop(config_key, None)

            if old_level > 0:
                logger.warning(
                    f"[SelfplayScheduler] Progress RECOVERED for {config_key}: "
                    f"was at escalation level {old_level}, now reset "
                    f"(velocity: {new_velocity:.2f} Elo/hour)"
                )
            else:
                logger.info(
                    f"[SelfplayScheduler] Progress recovered for {config_key} "
                    f"(velocity: {new_velocity:.2f} Elo/hour). Reset exploration boost."
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling progress recovered: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_node_under_backoff(self, node_id: str) -> bool:
        """Check if a node is currently under overload backoff.

        Args:
            node_id: The node ID to check.

        Returns:
            True if the node is under backoff, False otherwise.
        """
        overloaded = getattr(self, "_overloaded_nodes", {})
        if node_id not in overloaded:
            return False

        # Check if backoff has expired
        if overloaded[node_id] < time.time():
            del self._overloaded_nodes[node_id]
            return False

        return True

    def get_overloaded_nodes(self) -> list[str]:
        """Get list of currently overloaded nodes.

        Returns:
            List of node IDs that are currently under backoff.
        """
        overloaded = getattr(self, "_overloaded_nodes", {})
        current_time = time.time()

        # Filter to active backoffs and clean up expired
        active = []
        expired = []
        for node_id, expiry in overloaded.items():
            if expiry > current_time:
                active.append(node_id)
            else:
                expired.append(node_id)

        # Clean up expired
        for node_id in expired:
            del self._overloaded_nodes[node_id]

        return active

    def is_node_healthy(self, node_id: str) -> bool:
        """Check if a node is healthy for selfplay allocation.

        Args:
            node_id: The node ID to check.

        Returns:
            True if the node is healthy, False otherwise.
        """
        unhealthy = getattr(self, "_unhealthy_nodes", set())
        if node_id in unhealthy:
            return False

        if self.is_node_under_backoff(node_id):
            return False

        return True

    def get_cluster_health_factor(self) -> float:
        """Get current cluster health factor for allocation scaling.

        Returns:
            Float between 0.3 and 1.0 representing cluster health.
        """
        return getattr(self, "_cluster_health_factor", 1.0)

    def is_backpressure_active(self) -> bool:
        """Check if any backpressure is currently active.

        Returns:
            True if work queue or evaluation backpressure is active.
        """
        return (
            getattr(self, "_backpressure_active", False) or
            getattr(self, "_eval_backpressure_active", False)
        )

    def cleanup_stale_host_offline_exclusions(self) -> list[str]:
        """Auto-recover nodes that have been in HOST_OFFLINE state for too long.

        Session 17.35: Nodes may come back online without explicit HOST_ONLINE
        events (e.g., if the P2P orchestrator was restarted). This method
        removes stale exclusions after 30 minutes to allow re-probing.

        Should be called periodically (e.g., every 5 minutes from _run_cycle).

        Returns:
            List of host IDs that were auto-recovered.
        """
        offline_timestamps = getattr(self, "_host_offline_timestamps", {})
        if not offline_timestamps:
            return []

        recovery_seconds = getattr(self, "_host_offline_recovery_seconds", 1800.0)
        current_time = time.time()
        recovered = []

        for host, offline_time in list(offline_timestamps.items()):
            age_seconds = current_time - offline_time
            if age_seconds >= recovery_seconds:
                # Auto-recover this node
                recovered.append(host)
                del self._host_offline_timestamps[host]

                # Remove from unhealthy nodes
                unhealthy = getattr(self, "_unhealthy_nodes", set())
                if host in unhealthy:
                    self._unhealthy_nodes.discard(host)

                # Restore node capability
                if host in self._node_capabilities:
                    self._node_capabilities[host].current_load = 0.0

                logger.info(
                    f"[SelfplayScheduler] Auto-recovered HOST_OFFLINE node {host} "
                    f"after {age_seconds/60:.1f} minutes. Will re-probe for availability."
                )

        if recovered:
            logger.info(
                f"[SelfplayScheduler] Auto-recovered {len(recovered)} stale HOST_OFFLINE exclusions: "
                f"{recovered[:5]}{'...' if len(recovered) > 5 else ''}"
            )

        return recovered

    def get_host_offline_count(self) -> int:
        """Get count of currently excluded HOST_OFFLINE nodes.

        Returns:
            Number of nodes in HOST_OFFLINE exclusion state.
        """
        return len(getattr(self, "_host_offline_timestamps", {}))
