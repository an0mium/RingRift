"""Job Management Loops for P2P Orchestrator.

December 2025: Background loops for job lifecycle management.

Loops:
- JobReaperLoop: Cleans up stale/stuck jobs
- IdleDetectionLoop: Detects idle nodes for potential shutdown

Usage:
    from scripts.p2p.loops import JobReaperLoop, IdleDetectionLoop

    reaper = JobReaperLoop(
        get_active_jobs=lambda: orchestrator.active_jobs,
        cancel_job=orchestrator.cancel_job,
    )
    await reaper.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)


@dataclass
class JobReaperConfig:
    """Configuration for job reaper loop."""

    stale_job_threshold_seconds: float = 3600.0  # 1 hour
    stuck_job_threshold_seconds: float = 7200.0  # 2 hours
    max_jobs_to_reap_per_cycle: int = 10
    check_interval_seconds: float = 300.0  # 5 minutes


class JobReaperLoop(BaseLoop):
    """Background loop that cleans up stale and stuck jobs.

    Identifies jobs that:
    - Have been running too long (stuck)
    - Were claimed but never started (stale)
    - Have no heartbeat for too long (abandoned)

    And either cancels them or marks them for retry.
    """

    def __init__(
        self,
        get_active_jobs: Callable[[], dict[str, Any]],
        cancel_job: Callable[[str], Coroutine[Any, Any, bool]],
        get_job_heartbeats: Callable[[], dict[str, float]] | None = None,
        config: JobReaperConfig | None = None,
    ):
        """Initialize job reaper loop.

        Args:
            get_active_jobs: Callback returning dict of job_id -> job_info
            cancel_job: Async callback to cancel a job by ID
            get_job_heartbeats: Optional callback returning job_id -> last_heartbeat_time
            config: Reaper configuration
        """
        self.config = config or JobReaperConfig()
        super().__init__(
            name="job_reaper",
            interval=self.config.check_interval_seconds,
        )
        self._get_active_jobs = get_active_jobs
        self._cancel_job = cancel_job
        self._get_job_heartbeats = get_job_heartbeats
        self._reap_stats = {
            "stale_jobs_reaped": 0,
            "stuck_jobs_reaped": 0,
            "abandoned_jobs_reaped": 0,
        }

    async def _run_once(self) -> None:
        """Check for and clean up problematic jobs."""
        active_jobs = self._get_active_jobs()
        if not active_jobs:
            return

        now = time.time()
        heartbeats = self._get_job_heartbeats() if self._get_job_heartbeats else {}
        jobs_to_reap: list[tuple[str, str]] = []  # (job_id, reason)

        for job_id, job_info in active_jobs.items():
            # Check for stale jobs (claimed but not started)
            started_at = job_info.get("started_at", 0)
            claimed_at = job_info.get("claimed_at", 0)
            status = job_info.get("status", "")

            if status == "claimed" and not started_at:
                age = now - claimed_at if claimed_at else now
                if age > self.config.stale_job_threshold_seconds:
                    jobs_to_reap.append((job_id, "stale"))
                    continue

            # Check for stuck jobs (running too long)
            if started_at and status in ("running", "started"):
                runtime = now - started_at
                if runtime > self.config.stuck_job_threshold_seconds:
                    jobs_to_reap.append((job_id, "stuck"))
                    continue

            # Check for abandoned jobs (no heartbeat)
            if job_id in heartbeats:
                last_heartbeat = heartbeats[job_id]
                silence = now - last_heartbeat
                if silence > self.config.stale_job_threshold_seconds:
                    jobs_to_reap.append((job_id, "abandoned"))

        # Reap jobs up to limit
        reaped_count = 0
        for job_id, reason in jobs_to_reap[:self.config.max_jobs_to_reap_per_cycle]:
            try:
                success = await self._cancel_job(job_id)
                if success:
                    reaped_count += 1
                    self._reap_stats[f"{reason}_jobs_reaped"] += 1
                    logger.info(f"[JobReaper] Reaped {reason} job: {job_id}")
            except Exception as e:
                logger.warning(f"[JobReaper] Failed to reap job {job_id}: {e}")

        if reaped_count > 0:
            logger.info(f"[JobReaper] Reaped {reaped_count} jobs this cycle")

    def get_reap_stats(self) -> dict[str, Any]:
        """Get reaping statistics."""
        return {
            **self._reap_stats,
            **self.stats.to_dict(),
        }


@dataclass
class IdleDetectionConfig:
    """Configuration for idle detection loop."""

    gpu_idle_threshold_percent: float = 10.0  # GPU utilization below this = idle
    idle_duration_threshold_seconds: float = 60.0  # 1 minute (reduced from 15 min for faster dispatch)
    check_interval_seconds: float = 30.0  # Check every 30 seconds
    min_nodes_to_keep: int = 2  # Never flag last N nodes as idle


class IdleDetectionLoop(BaseLoop):
    """Background loop that detects idle nodes and triggers selfplay on them.

    Monitors GPU utilization and starts selfplay on nodes that have been
    idle for too long. Only runs on the cluster leader.
    """

    def __init__(
        self,
        get_role: Callable[[], str] | None = None,
        get_peers: Callable[[], dict[str, Any]] | None = None,
        get_work_queue: Callable[[], Any] | None = None,
        on_idle_detected: Callable[[Any, float], Coroutine[Any, Any, None]] | None = None,
        config: IdleDetectionConfig | None = None,
        # Legacy parameters for backward compatibility
        get_node_metrics: Callable[[], dict[str, dict[str, Any]]] | None = None,
    ):
        """Initialize idle detection loop.

        Args:
            get_role: Callback returning node role ("leader", "follower", etc.)
            get_peers: Callback returning dict of node_id -> peer info
            get_work_queue: Callback returning work queue (to check for pending work)
            on_idle_detected: Optional async callback (peer, idle_duration) - auto-start selfplay
            config: Detection configuration
            get_node_metrics: Legacy param - if provided, used instead of get_peers
        """
        self.config = config or IdleDetectionConfig()
        super().__init__(
            name="idle_detection",
            interval=self.config.check_interval_seconds,
        )
        self._get_role = get_role
        self._get_peers = get_peers
        self._get_work_queue = get_work_queue
        self._on_idle_detected = on_idle_detected
        # Legacy support
        self._get_node_metrics = get_node_metrics
        self._idle_since: dict[str, float] = {}  # node_id -> timestamp when became idle
        self._detected_count = 0
        self._skipped_not_leader = 0

    async def _run_once(self) -> None:
        """Check for idle nodes and trigger selfplay."""
        # Only run on leader
        if self._get_role:
            role = self._get_role()
            if role != "leader":
                self._skipped_not_leader += 1
                return

        # Get peer metrics
        if self._get_node_metrics:
            # Legacy path
            peers = self._get_node_metrics()
        elif self._get_peers:
            peers = self._get_peers()
        else:
            return

        if not peers:
            return

        now = time.time()
        gpu_peers = {}

        # Filter to GPU peers and extract metrics
        for node_id, peer_info in peers.items():
            # Handle both dict and object forms
            if hasattr(peer_info, "has_gpu"):
                has_gpu = peer_info.has_gpu
                gpu_util = getattr(peer_info, "gpu_percent", 0) or 0
                selfplay_jobs = getattr(peer_info, "selfplay_jobs", 0) or 0
            else:
                has_gpu = peer_info.get("has_gpu", False)
                gpu_util = peer_info.get("gpu_percent", 0) or peer_info.get("gpu_utilization", 0) or 0
                selfplay_jobs = peer_info.get("selfplay_jobs", 0) or 0

            if has_gpu:
                gpu_peers[node_id] = {
                    "peer": peer_info,
                    "gpu_utilization": gpu_util,
                    "selfplay_jobs": selfplay_jobs,
                }

        active_nodes = len(gpu_peers)
        if active_nodes == 0:
            return

        for node_id, metrics in gpu_peers.items():
            gpu_util = metrics["gpu_utilization"]
            selfplay_jobs = metrics["selfplay_jobs"]

            # Node is idle if GPU < threshold AND no selfplay jobs
            is_idle = gpu_util < self.config.gpu_idle_threshold_percent and selfplay_jobs == 0

            if is_idle:
                if node_id not in self._idle_since:
                    self._idle_since[node_id] = now
                    logger.debug(f"[IdleDetection] Node {node_id} became idle (GPU: {gpu_util}%, jobs: {selfplay_jobs})")

                # Check if idle long enough
                idle_duration = now - self._idle_since[node_id]
                if idle_duration >= self.config.idle_duration_threshold_seconds:
                    # Don't flag if we're at minimum nodes
                    non_idle_count = active_nodes - len([
                        n for n in self._idle_since
                        if now - self._idle_since[n] >= self.config.idle_duration_threshold_seconds
                    ])
                    if non_idle_count >= self.config.min_nodes_to_keep:
                        peer = metrics["peer"]
                        if self._on_idle_detected:
                            try:
                                await self._on_idle_detected(peer, idle_duration)
                                self._detected_count += 1
                                logger.info(
                                    f"[IdleDetection] Triggered selfplay on {node_id} (idle for {idle_duration:.0f}s)"
                                )
                                # Remove from idle tracking after triggering
                                del self._idle_since[node_id]
                            except Exception as e:
                                logger.warning(f"[IdleDetection] Callback failed for {node_id}: {e}")
                        else:
                            logger.info(
                                f"[IdleDetection] Node {node_id} idle for {idle_duration:.0f}s (no callback configured)"
                            )
            else:
                # Node is active, remove from idle tracking
                if node_id in self._idle_since:
                    del self._idle_since[node_id]
                    logger.debug(f"[IdleDetection] Node {node_id} became active (GPU: {gpu_util}%)")

    def get_idle_nodes(self) -> dict[str, float]:
        """Get currently tracked idle nodes and their idle duration."""
        now = time.time()
        return {
            node_id: now - idle_since
            for node_id, idle_since in self._idle_since.items()
        }

    def get_detection_stats(self) -> dict[str, Any]:
        """Get detection statistics."""
        return {
            "currently_idle": len(self._idle_since),
            "total_detections": self._detected_count,
            "idle_nodes": list(self._idle_since.keys()),
            **self.stats.to_dict(),
        }


__all__ = [
    "IdleDetectionConfig",
    "IdleDetectionLoop",
    "JobReaperConfig",
    "JobReaperLoop",
]
