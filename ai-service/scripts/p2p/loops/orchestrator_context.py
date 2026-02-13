"""Orchestrator Context for P2P Background Loops.

January 2026: Phase 2 P2P Orchestrator Deep Decomposition.

This module provides a unified context object that loops use to access
orchestrator functionality. Instead of passing 5-10 individual callbacks
to each loop, we pass a single context object.

Benefits:
- Reduces loop constructor complexity
- Makes dependencies explicit
- Easier to add new capabilities without changing loop signatures
- Better testability via mock context objects

Usage:
    # In orchestrator
    context = OrchestratorContext.from_orchestrator(self)

    # In loop
    idle_loop = IdleDetectionLoop(context=context)

    # Loop can access what it needs
    if context.is_leader():
        await context.auto_start_selfplay(peer, idle_duration)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from scripts.p2p.managers import (
        JobManager,
        SelfplayScheduler,
        StateManager,
    )

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorContext:
    """Minimal context passed to P2P loops.

    This dataclass bundles all the callbacks and accessors that loops
    typically need from the orchestrator. Loops can use this single
    context instead of receiving 5-10 individual parameters.

    Thread Safety:
    - Read-only accessors (get_peers, is_leader) are safe for concurrent use
    - Write operations should acquire appropriate locks (peers_lock, jobs_lock)

    Attributes:
        node_id: This node's identifier
        get_peers: Callback returning dict of node_id -> NodeInfo
        get_role: Callback returning node role string
        is_leader: Callback returning True if this node is cluster leader
        peers_lock: RLock for thread-safe peers dict access
        jobs_lock: RLock for thread-safe jobs dict access

    Optional Components:
        selfplay_scheduler: Reference to selfplay scheduling component
        job_manager: Reference to job management component
        state_manager: Reference to state persistence component

    Callbacks:
        emit_event: Callback to emit events (event_type, payload)
        auto_start_selfplay: Callback for idle detection (peer, duration)
        handle_zombie_detected: Callback for zombie processes (peer, duration)
        url_for_peer: Callback to build URLs for peer endpoints
        auth_headers: Callback to get authentication headers
    """

    # Core identity
    node_id: str

    # State accessors (read-only)
    get_peers: Callable[[], dict[str, Any]]
    get_role: Callable[[], str]
    is_leader: Callable[[], bool]
    get_leader_id: Callable[[], str | None] | None = None

    # Thread safety
    peers_lock: threading.RLock | None = None
    jobs_lock: threading.RLock | None = None

    # Manager references
    selfplay_scheduler: Any | None = None
    job_manager: Any | None = None
    state_manager: Any | None = None

    # Event emission
    emit_event: Callable[[str, dict[str, Any]], None] | None = None

    # HTTP callbacks
    url_for_peer: Callable[[Any, str], str] | None = None
    auth_headers: Callable[[], dict[str, str]] | None = None

    # Idle detection callbacks
    auto_start_selfplay: Callable[
        [Any, float], Coroutine[Any, Any, None]
    ] | None = None
    handle_zombie_detected: Callable[
        [Any, float], Coroutine[Any, Any, None]
    ] | None = None

    # Job management callbacks
    get_active_jobs: Callable[[], dict[str, Any]] | None = None
    cancel_job: Callable[[str], Coroutine[Any, Any, bool]] | None = None
    get_job_heartbeats: Callable[[], dict[str, float]] | None = None

    # Work queue callbacks
    get_work_queue: Callable[[], Any] | None = None
    get_work_queue_depth: Callable[[], int] | None = None

    # Node metrics callbacks
    get_node_metrics: Callable[[], dict[str, dict[str, Any]]] | None = None

    # Scaling callbacks
    get_pending_jobs_for_node: Callable[[str], int] | None = None
    spawn_preemptive_job: Callable[
        [dict[str, Any]], Coroutine[Any, Any, bool]
    ] | None = None

    # Sync manager reference
    sync_in_progress: Callable[[], bool] | None = None
    elo_sync_manager: Any | None = None

    @classmethod
    def from_orchestrator(cls, orchestrator: Any) -> "OrchestratorContext":
        """Create context from a P2P orchestrator instance.

        This factory method extracts all relevant callbacks and references
        from the orchestrator, handling missing attributes gracefully.

        Args:
            orchestrator: P2POrchestrator instance

        Returns:
            OrchestratorContext with all available callbacks wired
        """
        return cls(
            # Core identity
            node_id=orchestrator.node_id,

            # State accessors
            get_peers=lambda: orchestrator.peers,
            get_role=lambda: orchestrator.role,
            is_leader=orchestrator._is_leader,
            get_leader_id=lambda: getattr(orchestrator, "leader_id", None),

            # Thread safety
            peers_lock=getattr(orchestrator, "peers_lock", None),
            jobs_lock=getattr(orchestrator, "jobs_lock", None),

            # Manager references
            selfplay_scheduler=getattr(orchestrator, "selfplay_scheduler", None),
            job_manager=getattr(orchestrator, "job_manager", None),
            state_manager=getattr(orchestrator, "state_manager", None),

            # Event emission
            emit_event=getattr(orchestrator, "_emit_event", None),

            # HTTP callbacks
            url_for_peer=getattr(orchestrator, "_url_for_peer", None),
            auth_headers=getattr(orchestrator, "_auth_headers", None),

            # Idle detection callbacks
            # Feb 2026: _auto_start_selfplay was removed from orchestrator Jan 28.
            # Wire to selfplay_scheduler.auto_start_selfplay which has the same signature.
            auto_start_selfplay=getattr(
                getattr(orchestrator, "selfplay_scheduler", None),
                "auto_start_selfplay",
                None,
            ),
            handle_zombie_detected=getattr(orchestrator, "_handle_zombie_detected", None),

            # Job management callbacks
            get_active_jobs=getattr(orchestrator, "_get_all_active_jobs_for_reaper", None),
            cancel_job=getattr(orchestrator, "_cancel_job_for_reaper", None),
            get_job_heartbeats=getattr(orchestrator, "_get_job_heartbeats_for_reaper", None),

            # Work queue callbacks
            get_work_queue=getattr(orchestrator, "_get_work_queue", None),
            get_work_queue_depth=getattr(orchestrator, "_get_work_queue_depth", None),

            # Scaling callbacks
            get_pending_jobs_for_node=getattr(orchestrator, "_get_pending_jobs_for_node", None),
            spawn_preemptive_job=getattr(orchestrator, "_spawn_preemptive_selfplay_job", None),

            # Sync manager
            sync_in_progress=lambda: getattr(orchestrator, "sync_in_progress", False),
            elo_sync_manager=getattr(orchestrator, "elo_sync_manager", None),
        )

    def health_check(self) -> dict[str, Any]:
        """Return health status of the context.

        Returns:
            Dict with status and available callbacks
        """
        available = []
        missing = []

        # Check key callbacks
        callbacks = [
            ("get_peers", self.get_peers),
            ("is_leader", self.is_leader),
            ("selfplay_scheduler", self.selfplay_scheduler),
            ("job_manager", self.job_manager),
            ("emit_event", self.emit_event),
            ("auto_start_selfplay", self.auto_start_selfplay),
        ]

        for name, callback in callbacks:
            if callback is not None:
                available.append(name)
            else:
                missing.append(name)

        return {
            "status": "healthy" if len(missing) <= 2 else "degraded",
            "node_id": self.node_id,
            "available_callbacks": available,
            "missing_callbacks": missing,
            "has_locks": self.peers_lock is not None and self.jobs_lock is not None,
        }


@dataclass
class IdleDetectionContext:
    """Specialized context for idle detection loops.

    A subset of OrchestratorContext focused on what IdleDetectionLoop needs.
    This provides a cleaner interface for loops that don't need full context.
    """

    get_role: Callable[[], str]
    get_peers: Callable[[], dict[str, Any]]
    get_work_queue: Callable[[], Any] | None = None
    auto_start_selfplay: Callable[
        [Any, float], Coroutine[Any, Any, None]
    ] | None = None
    handle_zombie_detected: Callable[
        [Any, float], Coroutine[Any, Any, None]
    ] | None = None

    @classmethod
    def from_context(cls, ctx: OrchestratorContext) -> "IdleDetectionContext":
        """Create from full orchestrator context."""
        return cls(
            get_role=ctx.get_role,
            get_peers=ctx.get_peers,
            get_work_queue=ctx.get_work_queue,
            auto_start_selfplay=ctx.auto_start_selfplay,
            handle_zombie_detected=ctx.handle_zombie_detected,
        )


@dataclass
class JobReaperContext:
    """Specialized context for job reaper loops."""

    get_active_jobs: Callable[[], dict[str, Any]]
    cancel_job: Callable[[str], Coroutine[Any, Any, bool]]
    get_job_heartbeats: Callable[[], dict[str, float]] | None = None

    @classmethod
    def from_context(cls, ctx: OrchestratorContext) -> "JobReaperContext":
        """Create from full orchestrator context."""
        if ctx.get_active_jobs is None or ctx.cancel_job is None:
            raise ValueError("JobReaperContext requires get_active_jobs and cancel_job")
        return cls(
            get_active_jobs=ctx.get_active_jobs,
            cancel_job=ctx.cancel_job,
            get_job_heartbeats=ctx.get_job_heartbeats,
        )


@dataclass
class PredictiveScalingContext:
    """Specialized context for predictive scaling loops."""

    get_role: Callable[[], str]
    get_peers: Callable[[], dict[str, Any]]
    get_queue_depth: Callable[[], int]
    get_pending_jobs_for_node: Callable[[str], int]
    spawn_preemptive_job: Callable[[dict[str, Any]], Coroutine[Any, Any, bool]]

    @classmethod
    def from_context(cls, ctx: OrchestratorContext) -> "PredictiveScalingContext":
        """Create from full orchestrator context."""
        if ctx.get_work_queue_depth is None:
            raise ValueError("PredictiveScalingContext requires get_work_queue_depth")
        if ctx.get_pending_jobs_for_node is None:
            raise ValueError("PredictiveScalingContext requires get_pending_jobs_for_node")
        if ctx.spawn_preemptive_job is None:
            raise ValueError("PredictiveScalingContext requires spawn_preemptive_job")
        return cls(
            get_role=ctx.get_role,
            get_peers=ctx.get_peers,
            get_queue_depth=ctx.get_work_queue_depth,
            get_pending_jobs_for_node=ctx.get_pending_jobs_for_node,
            spawn_preemptive_job=ctx.spawn_preemptive_job,
        )
