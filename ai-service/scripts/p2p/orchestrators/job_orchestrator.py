"""Job Orchestrator - Handles job spawning and process management.

January 2026: Created as part of P2POrchestrator decomposition.

Responsibilities:
- Job spawning with rate limiting
- Task isolation and subprocess management
- Spawn gating (resource checks)
- GPU job tracking
- Job result recording
- Process lifecycle management
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)


class JobOrchestrator(BaseOrchestrator):
    """Orchestrator for job spawning and process management.

    This orchestrator handles all aspects of job management in the P2P cluster:
    - Rate-limited job spawning
    - Process spawn gating based on resources
    - GPU job tracking and result recording
    - Safe task creation with error isolation

    The actual job execution is delegated to JobManager, but this orchestrator
    provides spawn control, resource gating, and health monitoring.

    Usage:
        # In P2POrchestrator.__init__:
        self.jobs = JobOrchestrator(self)

        # Check if spawn allowed:
        can_spawn, reason = self.jobs.can_spawn_process("selfplay")

        # Record job result:
        self.jobs.record_gpu_job_result(success=True)
    """

    # Rate limiting constants
    SPAWN_RATE_LIMIT_PER_MINUTE = 30

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the job orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance.
        """
        super().__init__(p2p)

        # Rate limiting state
        self._spawn_timestamps: list[float] = []

        # Job statistics
        self._jobs_spawned: int = 0
        self._jobs_completed: int = 0
        self._jobs_failed: int = 0
        self._gpu_jobs_active: int = 0

    @property
    def name(self) -> str:
        """Return the name of this orchestrator."""
        return "jobs"

    def health_check(self) -> HealthCheckResult:
        """Check the health of job orchestrator.

        Returns:
            HealthCheckResult with job status details.
        """
        try:
            issues = []

            # Check job manager availability
            job_manager = getattr(self._p2p, "job_manager", None)
            if job_manager is None:
                issues.append("JobManager not available")

            # Check job queue health
            if job_manager is not None and hasattr(job_manager, "get_queue_size"):
                queue_size = job_manager.get_queue_size()
                if queue_size > 100:
                    issues.append(f"Large job queue: {queue_size}")

            # Check failure rate
            total_completed = self._jobs_completed + self._jobs_failed
            if total_completed > 10:
                failure_rate = self._jobs_failed / total_completed
                if failure_rate > 0.5:
                    issues.append(f"High job failure rate: {failure_rate:.0%}")

            # Check rate limiting
            can_spawn, reason = self.check_spawn_rate_limit()
            if not can_spawn:
                issues.append(f"Rate limited: {reason}")

            healthy = len(issues) == 0
            message = "Jobs healthy" if healthy else "; ".join(issues)

            return HealthCheckResult(
                healthy=healthy,
                message=message,
                details={
                    "jobs_spawned": self._jobs_spawned,
                    "jobs_completed": self._jobs_completed,
                    "jobs_failed": self._jobs_failed,
                    "gpu_jobs_active": self._gpu_jobs_active,
                    "rate_limit_status": reason,
                    "issues": issues,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def check_spawn_rate_limit(self) -> tuple[bool, str]:
        """Check if we're within the spawn rate limit.

        Jan 29, 2026: Implementation moved from P2POrchestrator.

        Returns:
            (can_spawn, reason) - True if within rate limit
        """
        now = time.time()
        # Clean old timestamps (older than 60 seconds)
        self._spawn_timestamps = [t for t in self._spawn_timestamps if now - t < 60]

        current = len(self._spawn_timestamps)
        limit = self.SPAWN_RATE_LIMIT_PER_MINUTE

        if current >= limit:
            return False, f"Rate limit: {current}/{limit} spawns in last minute"

        return True, f"Rate OK: {current}/{limit}"

    def record_spawn(self) -> None:
        """Record a process spawn for rate limiting."""
        self._spawn_timestamps.append(time.time())
        self._jobs_spawned += 1

    # =========================================================================
    # Spawn Gating
    # =========================================================================

    def can_spawn_process(self, reason: str = "job") -> tuple[bool, str]:
        """Combined safeguard check before spawning any process.

        Jan 29, 2026: Wrapper for P2POrchestrator._can_spawn_process().

        Checks:
        1. Load average (system resources)
        2. Rate limit (spawn frequency)
        3. Agent mode (if applicable)

        Args:
            reason: Description of why we want to spawn

        Returns:
            (can_spawn, reason) - True if spawning is allowed
        """
        if hasattr(self._p2p, "_can_spawn_process"):
            return self._p2p._can_spawn_process(reason)

        # Fallback: just check rate limit
        return self.check_spawn_rate_limit()

    # =========================================================================
    # Task Management
    # =========================================================================

    def create_safe_task(
        self,
        coro: Coroutine,
        name: str,
        factory: Callable[[], Coroutine] | None = None,
    ) -> asyncio.Task:
        """Create a task wrapped with exception isolation and restart support.

        Jan 29, 2026: Wrapper for P2POrchestrator._create_safe_task().

        Args:
            coro: The coroutine to run
            name: Task name for logging
            factory: Optional callable that returns a new coroutine for restarts

        Returns:
            asyncio.Task wrapped with safe error handling
        """
        if hasattr(self._p2p, "_create_safe_task"):
            return self._p2p._create_safe_task(coro, name, factory)

        # Fallback: create task directly
        return asyncio.create_task(coro, name=name)

    # =========================================================================
    # GPU Job Tracking
    # =========================================================================

    def record_gpu_job_result(self, success: bool) -> None:
        """Record a GPU job result.

        Jan 29, 2026: Wrapper for P2POrchestrator._record_gpu_job_result().

        Args:
            success: Whether the job completed successfully
        """
        if success:
            self._jobs_completed += 1
        else:
            self._jobs_failed += 1

        if hasattr(self._p2p, "_record_gpu_job_result"):
            self._p2p._record_gpu_job_result(success)

    def update_gpu_job_count(self, delta: int) -> None:
        """Update the active GPU job count.

        Jan 29, 2026: Wrapper for P2POrchestrator._update_gpu_job_count().

        Args:
            delta: Change in job count (+1 for start, -1 for end)
        """
        self._gpu_jobs_active += delta

        if hasattr(self._p2p, "_update_gpu_job_count"):
            self._p2p._update_gpu_job_count(delta)

    def get_gpu_job_count(self) -> int:
        """Get the current active GPU job count.

        Returns:
            Number of active GPU jobs
        """
        return self._gpu_jobs_active

    # =========================================================================
    # Job Preferences
    # =========================================================================

    def get_node_job_preference(self, node_id: str) -> str:
        """Get the job type preference for a specific node.

        Jan 29, 2026: Wrapper for P2POrchestrator._get_node_job_preference().

        Args:
            node_id: The node ID to check

        Returns:
            Preferred job type (e.g., "selfplay", "training", "any")
        """
        if hasattr(self._p2p, "_get_node_job_preference"):
            return self._p2p._get_node_job_preference(node_id)
        return "any"

    # =========================================================================
    # Job Status
    # =========================================================================

    def get_job_status(self) -> dict[str, Any]:
        """Get current job status.

        Returns:
            Dict with job statistics and state.
        """
        job_manager = getattr(self._p2p, "job_manager", None)

        status = {
            "jobs_spawned": self._jobs_spawned,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "gpu_jobs_active": self._gpu_jobs_active,
            "job_manager_available": job_manager is not None,
        }

        if job_manager is not None:
            if hasattr(job_manager, "get_active_jobs"):
                status["active_jobs"] = len(job_manager.get_active_jobs())
            if hasattr(job_manager, "get_queue_size"):
                status["queue_size"] = job_manager.get_queue_size()

        return status
