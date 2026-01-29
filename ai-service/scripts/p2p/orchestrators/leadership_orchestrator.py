"""Leadership Orchestrator - Handles leader election and consensus.

January 2026: Created as part of P2POrchestrator decomposition.

Responsibilities:
- Leader election state machine (CANDIDATE → LEADER → STEPPING_DOWN → FOLLOWER)
- Fence token management (generation, validation, refresh)
- Lease management (epoch, validation)
- Leadership consistency validation
- Leadership desync recovery
- Incumbent grace period (prevent flapping)
- Broadcast leadership claims via gossip
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from aiohttp import web
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)


class LeadershipOrchestrator(BaseOrchestrator):
    """Orchestrator for leader election and consensus management.

    This orchestrator handles all aspects of leadership in the P2P cluster:
    - Determining and setting the current leader
    - Managing fence tokens to prevent split-brain
    - Handling leader leases and their renewal
    - Broadcasting leadership claims to peers
    - Recovering from leadership desync situations

    The orchestrator delegates to existing managers:
    - state_manager: For persistent state
    - quorum_manager: For voter management
    - leader_probe_loop: For health monitoring

    Usage:
        # In P2POrchestrator.__init__:
        self.leadership = LeadershipOrchestrator(self)

        # Check leadership:
        if self.leadership.is_leader:
            ...

        # Get fence token:
        token = self.leadership.get_fence_token()
    """

    # Grace period constants
    INCUMBENT_GRACE_PERIOD_SECONDS = 60.0  # Prevent leader flapping
    RECENT_LEADER_WINDOW_SECONDS = 300.0  # Consider "recently leader" for 5 min

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the leadership orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance.
        """
        super().__init__(p2p)

        # Leadership state (mirrors p2p state for now, will migrate)
        self._last_leader_change_time: float = 0.0
        self._last_election_time: float = 0.0
        self._election_in_progress: bool = False
        self._provisional_leader: bool = False
        self._leader_claim_broadcast_count: int = 0

    @property
    def name(self) -> str:
        """Return the name of this orchestrator."""
        return "leadership"

    def health_check(self) -> HealthCheckResult:
        """Check the health of leadership orchestrator.

        Returns:
            HealthCheckResult with leadership status details.
        """
        try:
            # Get leadership state from parent orchestrator
            if hasattr(self._p2p, "is_leader") and callable(self._p2p.is_leader):
                is_leader = self._p2p.is_leader()
            else:
                is_leader = getattr(self._p2p, "is_leader", False)
            leader_id = getattr(self._p2p, "leader_id", None)
            role = getattr(self._p2p, "role", None)

            # Check for potential issues
            issues = []

            # Check lease validity
            lease_valid = self._is_leader_lease_valid()
            if is_leader and not lease_valid:
                issues.append("Leader with invalid lease")

            # Check for leadership consistency
            if hasattr(self._p2p, "_get_leadership_consistency_metrics"):
                metrics = self._p2p._get_leadership_consistency_metrics()
                if not metrics.get("is_consistent", True):
                    issues.append(f"Leadership inconsistency: {metrics.get('reason', 'unknown')}")

            healthy = len(issues) == 0
            message = "Leadership healthy" if healthy else "; ".join(issues)

            return HealthCheckResult(
                healthy=healthy,
                message=message,
                details={
                    "is_leader": is_leader,
                    "leader_id": leader_id,
                    "role": str(role) if role else None,
                    "lease_valid": lease_valid,
                    "election_in_progress": self._election_in_progress,
                    "provisional_leader": self._provisional_leader,
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
    # Leadership State - Delegated to P2POrchestrator for now
    # These will be migrated incrementally
    # =========================================================================

    def is_leader(self) -> bool:
        """Check if this node is the current cluster leader.

        Returns:
            True if this node is the leader with a valid lease.
        """
        if hasattr(self._p2p, "is_leader") and callable(self._p2p.is_leader):
            return self._p2p.is_leader()
        return getattr(self._p2p, "is_leader", False)

    def get_leader_id(self) -> str | None:
        """Get the current leader's node ID.

        Returns:
            The leader's node ID, or None if no leader.
        """
        return getattr(self._p2p, "leader_id", None)

    def get_fence_token(self) -> str:
        """Get the current fence token for split-brain prevention.

        Returns:
            The current fence token string.
        """
        if hasattr(self._p2p, "get_fence_token"):
            return self._p2p.get_fence_token()
        return ""

    def get_lease_epoch(self) -> int:
        """Get the current leader lease epoch.

        Returns:
            The current lease epoch number.
        """
        if hasattr(self._p2p, "get_lease_epoch"):
            return self._p2p.get_lease_epoch()
        return 0

    def validate_fence_token(self, token: str) -> tuple[bool, str]:
        """Validate a fence token.

        Args:
            token: The fence token to validate.

        Returns:
            Tuple of (is_valid, reason).
        """
        if hasattr(self._p2p, "validate_fence_token"):
            return self._p2p.validate_fence_token(token)
        return (False, "Fence token validation not available")

    def _is_leader_lease_valid(self) -> bool:
        """Check if the current leader lease is valid.

        Returns:
            True if the lease is valid.
        """
        if hasattr(self._p2p, "_is_leader_lease_valid"):
            return self._p2p._is_leader_lease_valid()
        # Fallback: check lease expiry directly
        lease_expires = getattr(self._p2p, "leader_lease_expires", 0)
        return time.time() < lease_expires

    # =========================================================================
    # Leadership Consistency
    # =========================================================================

    def get_consistency_metrics(self) -> dict[str, Any]:
        """Get leadership consistency metrics.

        Returns:
            Dictionary with consistency status and details.
        """
        if hasattr(self._p2p, "_get_leadership_consistency_metrics"):
            return self._p2p._get_leadership_consistency_metrics()
        return {"is_consistent": True, "reason": "No metrics available"}

    def recover_leadership_desync(self) -> bool:
        """Attempt to recover from leadership desync.

        Returns:
            True if recovery was successful.
        """
        if hasattr(self._p2p, "_recover_leadership_desync"):
            return self._p2p._recover_leadership_desync()
        return False

    def reconcile_leadership_state(self) -> bool:
        """Reconcile leadership state across subsystems.

        Returns:
            True if reconciliation was needed and successful.
        """
        if hasattr(self._p2p, "_reconcile_leadership_state"):
            return self._p2p._reconcile_leadership_state()
        return False

    # =========================================================================
    # Leadership Broadcasting
    # =========================================================================

    def broadcast_leadership_claim(self) -> None:
        """Broadcast a leadership claim to all peers."""
        if hasattr(self._p2p, "_broadcast_leadership_claim"):
            self._p2p._broadcast_leadership_claim()
            self._leader_claim_broadcast_count += 1

    async def async_broadcast_leader_claim(self) -> None:
        """Asynchronously broadcast a leadership claim to all peers."""
        if hasattr(self._p2p, "_async_broadcast_leader_claim"):
            await self._p2p._async_broadcast_leader_claim()
            self._leader_claim_broadcast_count += 1

    # =========================================================================
    # Grace Period Management
    # =========================================================================

    def was_recently_leader(self) -> bool:
        """Check if this node was recently the leader.

        Returns:
            True if the node was leader within RECENT_LEADER_WINDOW_SECONDS.
        """
        if hasattr(self._p2p, "_was_recently_leader"):
            return self._p2p._was_recently_leader()
        return False

    def in_incumbent_grace_period(self) -> bool:
        """Check if we're in the incumbent grace period.

        Returns:
            True if within grace period (leader stickiness).
        """
        if hasattr(self._p2p, "_in_incumbent_grace_period"):
            return self._p2p._in_incumbent_grace_period()
        return False

    # =========================================================================
    # Election Management
    # =========================================================================

    async def start_election(self, reason: str = "manual") -> None:
        """Start a leader election.

        Args:
            reason: Human-readable reason for the election.
        """
        if self._election_in_progress:
            self._log_warning(f"Election already in progress, skipping: {reason}")
            return

        self._election_in_progress = True
        self._last_election_time = time.time()

        try:
            if hasattr(self._p2p, "_start_election"):
                await self._p2p._start_election()
        finally:
            self._election_in_progress = False

    async def become_leader(self) -> None:
        """Transition this node to leader state."""
        if hasattr(self._p2p, "_become_leader"):
            await self._p2p._become_leader()

    async def request_election_from_voters(self, reason: str = "non_voter_request") -> bool:
        """Request an election from voter nodes.

        Args:
            reason: Reason for the election request.

        Returns:
            True if election was triggered.
        """
        if hasattr(self._p2p, "_request_election_from_voters"):
            return await self._p2p._request_election_from_voters(reason)
        return False

    # =========================================================================
    # Leader Queries
    # =========================================================================

    def get_leader_peer(self) -> Any | None:
        """Get the NodeInfo for the current leader.

        Returns:
            NodeInfo of the leader, or None if no leader.
        """
        if hasattr(self._p2p, "_get_leader_peer"):
            return self._p2p._get_leader_peer()
        return None

    async def determine_leased_leader_from_voters(self) -> str | None:
        """Query voters to determine the leased leader.

        Returns:
            The leader ID according to voter consensus, or None.
        """
        if hasattr(self._p2p, "_determine_leased_leader_from_voters"):
            return await self._p2p._determine_leased_leader_from_voters()
        return None

    def count_peers_reporting_leader(self, leader_id: str) -> int:
        """Count how many peers report a specific node as leader.

        Args:
            leader_id: The leader ID to check.

        Returns:
            Number of peers reporting this node as leader.
        """
        if hasattr(self._p2p, "_count_peers_reporting_leader"):
            return self._p2p._count_peers_reporting_leader(leader_id)
        return 0

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_leader_hint(self) -> dict[str, Any]:
        """Get leader hint information for peer responses.

        Returns:
            Dictionary with leader_id and other hints.
        """
        if hasattr(self._p2p, "_get_leader_hint"):
            return self._p2p._get_leader_hint()
        return {"leader_id": self.get_leader_id()}

    def get_cluster_leader_consensus(self) -> dict[str, Any]:
        """Get cluster-wide leader consensus information.

        Returns:
            Dictionary with consensus status and voting details.
        """
        if hasattr(self._p2p, "_get_cluster_leader_consensus"):
            return self._p2p._get_cluster_leader_consensus()
        return {"leader_id": self.get_leader_id(), "consensus": "unknown"}
