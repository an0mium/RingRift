"""Job Coordination Manager for P2P Orchestrator.

January 2026: Phase 15 of P2P Orchestrator Decomposition.
Consolidates job management loop, cluster job coordination, and auto-scaling.

This module provides:
- Job management loop orchestration
- Local job management (decentralized)
- GPU auto-scaling
- Resource cleanup
- Work queue rebalancing
- Cluster balance checking

Extracted from p2p_orchestrator.py to reduce complexity and improve testability.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = 8770
JOB_CHECK_INTERVAL = 30  # Seconds between job management cycles


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class JobCoordinationConfig:
    """Configuration for job coordination behavior."""
    job_management_interval: float = 30.0
    auto_scale_interval: float = 120.0
    rebalance_interval: float = 60.0
    gpu_utilization_target: float = 0.75
    gpu_utilization_min: float = 0.60
    gpu_utilization_max: float = 0.80
    disk_cleanup_threshold: float = 80.0
    disk_warning_threshold: float = 75.0
    memory_warning_threshold: float = 70.0
    memory_critical_threshold: float = 85.0
    # Job limits
    max_selfplay_jobs_per_node: int = 4
    max_training_jobs_per_node: int = 1


@dataclass
class JobCoordinationStats:
    """Statistics about job coordination operations."""
    coordination_cycles: int = 0
    jobs_started: int = 0
    jobs_stopped: int = 0
    cluster_rebalances: int = 0
    gpu_scaling_adjustments: int = 0
    work_queue_dispatches: int = 0
    stuck_jobs_detected: int = 0
    resource_cleanups: int = 0
    local_job_cycles: int = 0
    cluster_job_cycles: int = 0


# ============================================================================
# Singleton Pattern
# ============================================================================

_job_coordination_manager: JobCoordinationManager | None = None


def get_job_coordination_manager() -> JobCoordinationManager | None:
    """Get the global JobCoordinationManager singleton."""
    return _job_coordination_manager


def set_job_coordination_manager(manager: JobCoordinationManager) -> None:
    """Set the global JobCoordinationManager singleton."""
    global _job_coordination_manager
    _job_coordination_manager = manager


def reset_job_coordination_manager() -> None:
    """Reset the global JobCoordinationManager singleton (for testing)."""
    global _job_coordination_manager
    _job_coordination_manager = None


def create_job_coordination_manager(
    config: JobCoordinationConfig | None = None,
    orchestrator: P2POrchestrator | None = None,
) -> JobCoordinationManager:
    """Factory function to create and register a JobCoordinationManager.

    Args:
        config: Optional configuration. Uses defaults if not provided.
        orchestrator: The P2P orchestrator instance.

    Returns:
        The created JobCoordinationManager instance.
    """
    manager = JobCoordinationManager(
        config=config or JobCoordinationConfig(),
        orchestrator=orchestrator,
    )
    set_job_coordination_manager(manager)
    return manager


# ============================================================================
# JobCoordinationManager
# ============================================================================

class JobCoordinationManager:
    """Manages job coordination for P2P cluster.

    This manager handles:
    - Job management loop orchestration
    - Local job management (decentralized)
    - GPU auto-scaling
    - Resource cleanup
    - Work queue rebalancing
    - Cluster balance checking

    Jan 27, 2026: Phase 15 decomposition from p2p_orchestrator.py.
    """

    def __init__(
        self,
        config: JobCoordinationConfig | None = None,
        orchestrator: P2POrchestrator | None = None,
    ):
        """Initialize the JobCoordinationManager.

        Args:
            config: Job coordination configuration.
            orchestrator: The P2P orchestrator instance.
        """
        self.config = config or JobCoordinationConfig()
        self._orchestrator = orchestrator
        self._stats = JobCoordinationStats()

        # State tracking with proper dicts (not dynamic attributes)
        self._last_local_job_manage: float = 0.0
        self._last_local_gpu_scale: float = 0.0
        self._last_local_resource_check: float = 0.0
        self._last_gpu_auto_scale: float = 0.0
        self._last_work_queue_rebalance: float = 0.0
        self._last_cluster_balance_check: float = 0.0

        # Per-node state tracking
        self._gpu_idle_since: dict[str, float] = {}
        self._wq_idle_since: dict[str, float] = {}
        self._diverse_config_counter: dict[str, int] = {}
        self._local_gpu_idle_since: float = 0.0

    # =========================================================================
    # Properties (delegate to orchestrator)
    # =========================================================================

    @property
    def _peers(self) -> dict[str, Any]:
        """Get peers dict from orchestrator."""
        return getattr(self._orchestrator, "peers", {})

    @property
    def _peers_lock(self) -> Any:
        """Get peers lock from orchestrator."""
        return getattr(self._orchestrator, "peers_lock", None)

    @property
    def _node_id(self) -> str:
        """Get this node's ID."""
        return getattr(self._orchestrator, "node_id", "unknown")

    @property
    def _self_info(self) -> Any:
        """Get this node's info."""
        return getattr(self._orchestrator, "self_info", None)

    @property
    def _leader_id(self) -> str | None:
        """Get current leader ID."""
        return getattr(self._orchestrator, "leader_id", None)

    @property
    def _role(self) -> Any:
        """Get this node's role."""
        return getattr(self._orchestrator, "role", None)

    @property
    def _running(self) -> bool:
        """Check if orchestrator is running."""
        return getattr(self._orchestrator, "running", False)

    @property
    def _job_lifecycle_manager(self) -> Any:
        """Get job lifecycle manager."""
        return getattr(self._orchestrator, "job_lifecycle_manager", None)

    @property
    def _selfplay_scheduler(self) -> Any:
        """Get selfplay scheduler."""
        return getattr(self._orchestrator, "selfplay_scheduler", None)

    @property
    def _work_queue(self) -> Any:
        """Get work queue."""
        return getattr(self._orchestrator, "work_queue", None)

    # =========================================================================
    # Helper Methods (delegate to orchestrator)
    # =========================================================================

    def _auth_headers(self) -> dict[str, str]:
        """Get auth headers from orchestrator."""
        if self._orchestrator and hasattr(self._orchestrator, "_auth_headers"):
            return self._orchestrator._auth_headers()
        return {}

    def _url_for_peer(self, peer: Any, path: str) -> str:
        """Get URL for peer."""
        if self._orchestrator and hasattr(self._orchestrator, "_url_for_peer"):
            return self._orchestrator._url_for_peer(peer, path)
        # Fallback
        scheme = getattr(peer, "scheme", "http") or "http"
        host = getattr(peer, "host", "localhost")
        port = getattr(peer, "port", DEFAULT_PORT)
        return f"{scheme}://{host}:{port}{path}"

    async def _update_self_info(self) -> None:
        """Update self info."""
        if self._orchestrator and hasattr(self._orchestrator, "_update_self_info"):
            self._orchestrator._update_self_info()

    def _get_cached_peer_snapshot(self) -> dict[str, Any]:
        """Get cached peer snapshot for lock-free access."""
        if self._orchestrator and hasattr(self._orchestrator, "_get_cached_peer_snapshot"):
            return self._orchestrator._get_cached_peer_snapshot()
        if self._peers_lock:
            with self._peers_lock:
                return dict(self._peers)
        return dict(self._peers)

    def _check_yaml_gpu_config(self) -> bool:
        """Check if this node has GPU configured in YAML."""
        if self._orchestrator and hasattr(self._orchestrator, "_check_yaml_gpu_config"):
            return self._orchestrator._check_yaml_gpu_config()
        return False

    # =========================================================================
    # Local Job Management (Decentralized)
    # =========================================================================

    async def manage_local_jobs_decentralized(self) -> None:
        """DECENTRALIZED: Each node manages its own job capacity.

        This runs on ALL nodes independently of leader status.
        Nodes determine their own job targets based on:
        - Local GPU/CPU capacity
        - Current job counts
        - Selfplay scheduler priorities

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        # Delegate to orchestrator's implementation for now
        # This method has complex GPU detection and job spawning logic
        if self._orchestrator and hasattr(self._orchestrator, "_manage_local_jobs_decentralized"):
            await self._orchestrator._manage_local_jobs_decentralized()
            self._stats.local_job_cycles += 1

    async def local_gpu_auto_scale(self) -> None:
        """DECENTRALIZED: Optimize GPU utilization on this node.

        Monitors GPU utilization and adjusts job count to maintain
        target utilization range (60-80%).

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_local_gpu_auto_scale"):
            await self._orchestrator._local_gpu_auto_scale()
            self._stats.gpu_scaling_adjustments += 1

    async def local_resource_cleanup(self) -> None:
        """DECENTRALIZED: Handle disk/memory pressure on this node.

        Cleans up resources when thresholds are exceeded.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_local_resource_cleanup"):
            await self._orchestrator._local_resource_cleanup()
            self._stats.resource_cleanups += 1

    # =========================================================================
    # Leader-Only Operations
    # =========================================================================

    async def manage_cluster_jobs(self) -> None:
        """LEADER ONLY: Manage jobs across the cluster.

        Coordinates job distribution, handles stuck jobs, and
        balances load across nodes.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_manage_cluster_jobs"):
            await self._orchestrator._manage_cluster_jobs()
            self._stats.cluster_job_cycles += 1

    async def auto_scale_gpu_utilization(self) -> None:
        """LEADER ONLY: Scale GPU utilization across cluster.

        Identifies under/over-utilized nodes and adjusts job counts
        to optimize cluster-wide GPU utilization.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_auto_scale_gpu_utilization"):
            await self._orchestrator._auto_scale_gpu_utilization()

    async def auto_rebalance_from_work_queue(self) -> None:
        """LEADER ONLY: Dispatch queued work to idle nodes.

        Claims work from the distributed work queue and dispatches
        to available nodes.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_auto_rebalance_from_work_queue"):
            await self._orchestrator._auto_rebalance_from_work_queue()
            self._stats.work_queue_dispatches += 1

    async def dispatch_queued_work(self, work_item: Any, target_node: Any) -> bool:
        """Dispatch a work item to a specific node.

        Args:
            work_item: The work item to dispatch.
            target_node: The node to dispatch to.

        Returns:
            True if dispatch was successful.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_dispatch_queued_work"):
            return await self._orchestrator._dispatch_queued_work(work_item, target_node)
        return False

    async def schedule_diverse_selfplay_on_node(
        self,
        node: Any,
        num_jobs: int = 1,
    ) -> int:
        """Schedule diverse selfplay jobs on a node.

        Args:
            node: The target node.
            num_jobs: Number of jobs to schedule.

        Returns:
            Number of jobs successfully scheduled.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_schedule_diverse_selfplay_on_node"):
            return await self._orchestrator._schedule_diverse_selfplay_on_node(node, num_jobs)
        return 0

    async def check_cluster_balance(self) -> None:
        """LEADER ONLY: Check and rebalance cluster load.

        Identifies overloaded/underutilized nodes and migrates
        jobs to balance the cluster.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_check_cluster_balance"):
            await self._orchestrator._check_cluster_balance()
            self._stats.cluster_rebalances += 1

    # =========================================================================
    # Coordination Cycle
    # =========================================================================

    async def run_coordination_cycle(self) -> None:
        """Run one job coordination cycle.

        This is the main entry point called from the orchestrator's
        job management loop. It handles both decentralized and
        leader-only operations.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 15).
        """
        self._stats.coordination_cycles += 1

        # Decentralized operations (all nodes)
        await self.local_resource_cleanup()
        await self.manage_local_jobs_decentralized()
        await self.local_gpu_auto_scale()

        # Leader-only operations
        from scripts.p2p.models import NodeRole
        if self._role == NodeRole.LEADER:
            await self.manage_cluster_jobs()
            await self.check_cluster_balance()
            await self.auto_rebalance_from_work_queue()
            await self.auto_scale_gpu_utilization()

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            dict with healthy status, message, and details.
        """
        is_healthy = self._running
        stats = asdict(self._stats)

        message = "Job coordination healthy" if is_healthy else "Job coordination stopped"

        return {
            "healthy": is_healthy,
            "message": message,
            "details": stats,
        }

    def get_stats(self) -> JobCoordinationStats:
        """Get current statistics."""
        return self._stats
