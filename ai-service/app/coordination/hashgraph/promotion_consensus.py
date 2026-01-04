"""Hashgraph-inspired consensus for model promotion decisions.

Uses virtual voting to achieve Byzantine Fault Tolerant (BFT) model promotion
with audit trail and rollback capability.

Key benefits:
1. Prevents single-node promotion of bad models
2. Requires supermajority approval (2/3+)
3. Creates cryptographic certificates for audit
4. Enables rollback with consensus proof

Usage:
    manager = PromotionConsensusManager(dag)

    # Propose promotion
    proposal = await manager.propose_promotion(
        model_hash="abc123",
        config_key="hex8_2p",
        evaluation_evidence={"win_rate": 0.85, "elo": 1450},
    )

    # Vote on proposal
    await manager.vote_on_proposal(proposal.proposal_id, approve=True)

    # Check consensus
    result = await manager.get_promotion_consensus(proposal.proposal_id)
    if result.approved:
        promote_model(result.certificate)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.hashgraph.event import HashgraphEvent, EventType
from app.coordination.hashgraph.dag import HashgraphDAG
from app.coordination.hashgraph.consensus import ConsensusEngine

logger = logging.getLogger(__name__)


class PromotionEventType(str, Enum):
    """Types of promotion-related events."""

    PROPOSAL = "promotion_proposal"
    VOTE = "promotion_vote"
    CONSENSUS = "promotion_consensus"
    CERTIFICATE = "promotion_certificate"


class VoteType(str, Enum):
    """Types of votes on promotion proposals."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class EvaluationEvidence:
    """Evidence supporting a promotion proposal.

    Attributes:
        win_rate: Win rate from evaluation
        elo: Elo rating
        games_played: Number of evaluation games
        vs_baselines: Results vs specific baselines
        timestamp: When evaluation was performed
    """

    win_rate: float
    elo: float
    games_played: int
    vs_baselines: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Validate evidence."""
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(f"Win rate must be 0.0-1.0, got {self.win_rate}")
        if self.games_played < 0:
            raise ValueError(f"Games played must be non-negative")
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PromotionProposal:
    """A proposal to promote a model.

    Attributes:
        proposal_id: Unique proposal identifier
        model_hash: Hash of model to promote
        config_key: Board/player configuration
        proposer: Node that proposed promotion
        evidence: Evaluation evidence supporting proposal
        timestamp: When proposal was created
        event_hash: Hash of the proposal event
    """

    proposal_id: str
    model_hash: str
    config_key: str
    proposer: str
    evidence: EvaluationEvidence
    timestamp: float
    event_hash: str = ""


@dataclass
class PromotionVote:
    """A vote on a promotion proposal.

    Attributes:
        proposal_id: Proposal being voted on
        voter: Node casting the vote
        vote: Approve, reject, or abstain
        reason: Optional reason for vote
        timestamp: When vote was cast
        event_hash: Hash of the vote event
    """

    proposal_id: str
    voter: str
    vote: VoteType
    reason: str = ""
    timestamp: float = 0.0
    event_hash: str = ""


@dataclass
class PromotionCertificate:
    """Cryptographic certificate for approved promotion.

    Attributes:
        proposal_id: Original proposal
        model_hash: Model being promoted
        config_key: Target configuration
        approvers: Nodes that approved
        approval_count: Number of approvals
        total_votes: Total votes cast
        certificate_hash: Hash for verification
        timestamp: When certificate was issued
    """

    proposal_id: str
    model_hash: str
    config_key: str
    approvers: list[str]
    approval_count: int
    total_votes: int
    certificate_hash: str
    timestamp: float


@dataclass
class PromotionConsensusResult:
    """Result of promotion consensus.

    Attributes:
        proposal_id: Proposal that was evaluated
        model_hash: Model in question
        config_key: Target configuration
        approved: Whether promotion was approved
        approval_count: Number of approvals
        rejection_count: Number of rejections
        abstention_count: Number of abstentions
        total_voters: Total voters that participated
        required_votes: Votes needed for approval
        has_consensus: Whether consensus was reached
        votes: Individual votes
        certificate: Certificate if approved
        consensus_timestamp: When consensus was determined
    """

    proposal_id: str
    model_hash: str
    config_key: str
    approved: bool = False
    approval_count: int = 0
    rejection_count: int = 0
    abstention_count: int = 0
    total_voters: int = 0
    required_votes: int = 0
    has_consensus: bool = False
    votes: list[PromotionVote] = field(default_factory=list)
    certificate: PromotionCertificate | None = None
    consensus_timestamp: float = 0.0


@dataclass
class PromotionConsensusConfig:
    """Configuration for promotion consensus.

    Attributes:
        min_voters: Minimum voters for consensus (default: 3)
        supermajority_fraction: Fraction required for approval (default: 2/3)
        max_wait_seconds: Maximum wait for votes (default: 300)
        min_win_rate: Minimum win rate to propose (default: 0.6)
        min_elo: Minimum Elo to propose (default: 1200)
        min_games: Minimum evaluation games (default: 50)
        require_baseline_wins: Require positive win rate vs baselines
    """

    min_voters: int = 3
    supermajority_fraction: float = 2 / 3
    max_wait_seconds: float = 300.0
    min_win_rate: float = 0.6
    min_elo: float = 1200.0
    min_games: int = 50
    require_baseline_wins: bool = True


class PromotionConsensusManager:
    """Hashgraph-inspired consensus for model promotion.

    Provides Byzantine Fault Tolerant model promotion through
    virtual voting and supermajority consensus.

    Attributes:
        dag: The hashgraph DAG for event storage
        engine: The consensus engine for virtual voting
        config: Configuration for consensus parameters
        node_id: Local node identifier
    """

    def __init__(
        self,
        dag: HashgraphDAG,
        node_id: str = "",
        config: PromotionConsensusConfig | None = None,
    ):
        """Initialize promotion consensus manager.

        Args:
            dag: Hashgraph DAG for event storage
            node_id: Local node identifier
            config: Optional configuration
        """
        self.dag = dag
        self.engine = ConsensusEngine(dag)
        self.config = config or PromotionConsensusConfig()
        self.node_id = node_id or dag.node_id

        # Track proposals and votes
        self._proposals: dict[str, PromotionProposal] = {}
        self._votes: dict[str, list[PromotionVote]] = {}
        self._consensus_reached: dict[str, PromotionConsensusResult] = {}
        self._consensus_waiters: dict[str, list[asyncio.Event]] = {}
        self._lock = asyncio.Lock()

    async def propose_promotion(
        self,
        model_hash: str,
        config_key: str,
        evidence: EvaluationEvidence | dict[str, Any],
        proposer: str | None = None,
    ) -> PromotionProposal:
        """Propose a model for promotion.

        Creates a hashgraph event with the proposal and adds it to the DAG.

        Args:
            model_hash: Hash of model to promote
            config_key: Target configuration (e.g., "hex8_2p")
            evidence: Evaluation evidence or dict
            proposer: Node proposing (defaults to node_id)

        Returns:
            The created proposal

        Raises:
            ValueError: If evidence doesn't meet minimums
        """
        proposer = proposer or self.node_id

        # Convert dict to EvaluationEvidence if needed
        if isinstance(evidence, dict):
            evidence = EvaluationEvidence(
                win_rate=evidence.get("win_rate", 0.0),
                elo=evidence.get("elo", 0.0),
                games_played=evidence.get("games_played", 0),
                vs_baselines=evidence.get("vs_baselines", {}),
                timestamp=evidence.get("timestamp", time.time()),
            )

        # Validate evidence meets minimums
        self._validate_evidence(evidence)

        timestamp = time.time()

        # Generate proposal ID
        proposal_id = self._compute_proposal_id(
            model_hash, config_key, proposer, timestamp
        )

        # Get latest event for self_parent
        latest = self.dag.get_latest_event(proposer)
        self_parent = latest.event_hash if latest else None

        # Compute round
        round_number = self.dag.compute_round_for_parents(self_parent, None)

        # Create proposal event
        event = HashgraphEvent.create(
            creator=proposer,
            payload={
                "type": PromotionEventType.PROPOSAL.value,
                "proposal_id": proposal_id,
                "model_hash": model_hash,
                "config_key": config_key,
                "evidence": {
                    "win_rate": evidence.win_rate,
                    "elo": evidence.elo,
                    "games_played": evidence.games_played,
                    "vs_baselines": evidence.vs_baselines,
                    "timestamp": evidence.timestamp,
                },
            },
            self_parent=self_parent,
            round_number=round_number,
            timestamp=timestamp,
        )

        # Add to DAG
        self.dag.add_event(event)

        # Create proposal
        proposal = PromotionProposal(
            proposal_id=proposal_id,
            model_hash=model_hash,
            config_key=config_key,
            proposer=proposer,
            evidence=evidence,
            timestamp=timestamp,
            event_hash=event.event_hash,
        )

        async with self._lock:
            self._proposals[proposal_id] = proposal
            self._votes[proposal_id] = []

        logger.info(
            f"[{self.node_id}] Proposed promotion: {model_hash[:8]} "
            f"for {config_key} (proposal: {proposal_id[:8]})"
        )

        return proposal

    def _validate_evidence(self, evidence: EvaluationEvidence) -> None:
        """Validate evidence meets minimum thresholds.

        Args:
            evidence: Evidence to validate

        Raises:
            ValueError: If evidence doesn't meet minimums
        """
        if evidence.win_rate < self.config.min_win_rate:
            raise ValueError(
                f"Win rate {evidence.win_rate:.2f} below minimum "
                f"{self.config.min_win_rate}"
            )
        if evidence.elo < self.config.min_elo:
            raise ValueError(
                f"Elo {evidence.elo:.0f} below minimum {self.config.min_elo}"
            )
        if evidence.games_played < self.config.min_games:
            raise ValueError(
                f"Games {evidence.games_played} below minimum {self.config.min_games}"
            )
        if self.config.require_baseline_wins and evidence.vs_baselines:
            for baseline, win_rate in evidence.vs_baselines.items():
                if win_rate < 0.5:
                    raise ValueError(
                        f"Win rate vs {baseline} ({win_rate:.2f}) below 0.5"
                    )

    async def vote_on_proposal(
        self,
        proposal_id: str,
        approve: bool,
        reason: str = "",
        voter: str | None = None,
    ) -> PromotionVote:
        """Cast a vote on a promotion proposal.

        Args:
            proposal_id: Proposal to vote on
            approve: True to approve, False to reject
            reason: Optional reason for vote
            voter: Node casting vote (defaults to node_id)

        Returns:
            The created vote

        Raises:
            ValueError: If proposal doesn't exist
        """
        voter = voter or self.node_id
        vote_type = VoteType.APPROVE if approve else VoteType.REJECT
        timestamp = time.time()

        async with self._lock:
            if proposal_id not in self._proposals:
                raise ValueError(f"Unknown proposal: {proposal_id}")

            proposal = self._proposals[proposal_id]

            # Check for duplicate vote
            existing_votes = self._votes.get(proposal_id, [])
            for existing in existing_votes:
                if existing.voter == voter:
                    logger.warning(
                        f"[{self.node_id}] {voter} already voted on {proposal_id[:8]}"
                    )
                    return existing

        # Get latest event for self_parent
        latest = self.dag.get_latest_event(voter)
        self_parent = latest.event_hash if latest else None

        # Compute round
        round_number = self.dag.compute_round_for_parents(self_parent, None)

        # Create vote event
        event = HashgraphEvent.create(
            creator=voter,
            payload={
                "type": PromotionEventType.VOTE.value,
                "proposal_id": proposal_id,
                "vote": vote_type.value,
                "reason": reason,
            },
            self_parent=self_parent,
            round_number=round_number,
            timestamp=timestamp,
        )

        # Add to DAG
        self.dag.add_event(event)

        # Create vote
        vote = PromotionVote(
            proposal_id=proposal_id,
            voter=voter,
            vote=vote_type,
            reason=reason,
            timestamp=timestamp,
            event_hash=event.event_hash,
        )

        async with self._lock:
            if proposal_id not in self._votes:
                self._votes[proposal_id] = []
            self._votes[proposal_id].append(vote)

            # Check if we have enough votes for consensus
            await self._check_consensus(proposal_id)

        logger.info(
            f"[{self.node_id}] Vote cast: {voter} {vote_type.value}s "
            f"proposal {proposal_id[:8]}"
        )

        return vote

    async def abstain_on_proposal(
        self,
        proposal_id: str,
        reason: str = "",
        voter: str | None = None,
    ) -> PromotionVote:
        """Abstain from voting on a proposal.

        Args:
            proposal_id: Proposal to abstain on
            reason: Optional reason for abstention
            voter: Node abstaining (defaults to node_id)

        Returns:
            The created abstention vote
        """
        voter = voter or self.node_id
        timestamp = time.time()

        async with self._lock:
            if proposal_id not in self._proposals:
                raise ValueError(f"Unknown proposal: {proposal_id}")

        # Get latest event for self_parent
        latest = self.dag.get_latest_event(voter)
        self_parent = latest.event_hash if latest else None

        # Compute round
        round_number = self.dag.compute_round_for_parents(self_parent, None)

        # Create vote event
        event = HashgraphEvent.create(
            creator=voter,
            payload={
                "type": PromotionEventType.VOTE.value,
                "proposal_id": proposal_id,
                "vote": VoteType.ABSTAIN.value,
                "reason": reason,
            },
            self_parent=self_parent,
            round_number=round_number,
            timestamp=timestamp,
        )

        # Add to DAG
        self.dag.add_event(event)

        # Create vote
        vote = PromotionVote(
            proposal_id=proposal_id,
            voter=voter,
            vote=VoteType.ABSTAIN,
            reason=reason,
            timestamp=timestamp,
            event_hash=event.event_hash,
        )

        async with self._lock:
            if proposal_id not in self._votes:
                self._votes[proposal_id] = []
            self._votes[proposal_id].append(vote)

        return vote

    async def get_promotion_consensus(
        self,
        proposal_id: str,
        timeout: float | None = None,
    ) -> PromotionConsensusResult:
        """Get consensus result for a promotion proposal.

        Waits for sufficient votes and computes consensus.

        Args:
            proposal_id: Proposal to get consensus for
            timeout: Override timeout (uses config default)

        Returns:
            Consensus result

        Raises:
            ValueError: If proposal doesn't exist
        """
        timeout = timeout or self.config.max_wait_seconds

        async with self._lock:
            if proposal_id not in self._proposals:
                raise ValueError(f"Unknown proposal: {proposal_id}")

            # Check if already have consensus
            if proposal_id in self._consensus_reached:
                return self._consensus_reached[proposal_id]

            # Check if we have enough votes now
            votes = self._votes.get(proposal_id, [])
            if len(votes) >= self.config.min_voters:
                return await self._compute_consensus(proposal_id)

            # Set up waiter
            if proposal_id not in self._consensus_waiters:
                self._consensus_waiters[proposal_id] = []
            waiter = asyncio.Event()
            self._consensus_waiters[proposal_id].append(waiter)

        # Wait for consensus
        try:
            await asyncio.wait_for(waiter.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(
                f"[{self.node_id}] Consensus timeout for {proposal_id[:8]}"
            )

        # Return whatever we have
        async with self._lock:
            if proposal_id in self._consensus_reached:
                return self._consensus_reached[proposal_id]

            # Compute partial result
            return await self._compute_consensus(proposal_id, require_consensus=False)

    async def _check_consensus(self, proposal_id: str) -> None:
        """Check if consensus can be reached and notify waiters.

        Must be called with _lock held.
        """
        if proposal_id not in self._votes:
            return

        votes = self._votes[proposal_id]
        if len(votes) < self.config.min_voters:
            return

        # Compute consensus
        result = await self._compute_consensus(proposal_id)

        if result.has_consensus:
            # Notify waiters
            if proposal_id in self._consensus_waiters:
                for waiter in self._consensus_waiters[proposal_id]:
                    waiter.set()
                del self._consensus_waiters[proposal_id]

    async def _compute_consensus(
        self,
        proposal_id: str,
        require_consensus: bool = True,
    ) -> PromotionConsensusResult:
        """Compute consensus from votes.

        Args:
            proposal_id: Proposal to compute consensus for
            require_consensus: Whether to require supermajority

        Returns:
            Consensus result
        """
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")

        proposal = self._proposals[proposal_id]
        votes = self._votes.get(proposal_id, [])

        # Count votes
        approval_count = sum(1 for v in votes if v.vote == VoteType.APPROVE)
        rejection_count = sum(1 for v in votes if v.vote == VoteType.REJECT)
        abstention_count = sum(1 for v in votes if v.vote == VoteType.ABSTAIN)

        total_voters = len(votes)
        required_votes = max(
            self.config.min_voters,
            int(total_voters * self.config.supermajority_fraction) + 1,
        )

        # Determine consensus
        approved = approval_count >= required_votes
        has_consensus = (
            approval_count >= required_votes or rejection_count >= required_votes
        )

        if require_consensus and not has_consensus:
            return PromotionConsensusResult(
                proposal_id=proposal_id,
                model_hash=proposal.model_hash,
                config_key=proposal.config_key,
                approval_count=approval_count,
                rejection_count=rejection_count,
                abstention_count=abstention_count,
                total_voters=total_voters,
                required_votes=required_votes,
                has_consensus=False,
                votes=votes,
            )

        # Create certificate if approved
        certificate = None
        if approved:
            approvers = [v.voter for v in votes if v.vote == VoteType.APPROVE]
            certificate = self._create_certificate(
                proposal, approvers, approval_count, total_voters
            )

        result = PromotionConsensusResult(
            proposal_id=proposal_id,
            model_hash=proposal.model_hash,
            config_key=proposal.config_key,
            approved=approved,
            approval_count=approval_count,
            rejection_count=rejection_count,
            abstention_count=abstention_count,
            total_voters=total_voters,
            required_votes=required_votes,
            has_consensus=has_consensus,
            votes=votes,
            certificate=certificate,
            consensus_timestamp=time.time(),
        )

        # Cache result
        if has_consensus:
            self._consensus_reached[proposal_id] = result

        return result

    def _create_certificate(
        self,
        proposal: PromotionProposal,
        approvers: list[str],
        approval_count: int,
        total_votes: int,
    ) -> PromotionCertificate:
        """Create cryptographic certificate for approved promotion.

        Args:
            proposal: The approved proposal
            approvers: Nodes that approved
            approval_count: Number of approvals
            total_votes: Total votes cast

        Returns:
            Promotion certificate
        """
        timestamp = time.time()

        # Compute certificate hash
        cert_data = {
            "proposal_id": proposal.proposal_id,
            "model_hash": proposal.model_hash,
            "config_key": proposal.config_key,
            "approvers": sorted(approvers),
            "approval_count": approval_count,
            "total_votes": total_votes,
            "timestamp": timestamp,
        }
        canonical = json.dumps(cert_data, sort_keys=True, separators=(",", ":"))
        certificate_hash = hashlib.sha256(canonical.encode()).hexdigest()

        return PromotionCertificate(
            proposal_id=proposal.proposal_id,
            model_hash=proposal.model_hash,
            config_key=proposal.config_key,
            approvers=approvers,
            approval_count=approval_count,
            total_votes=total_votes,
            certificate_hash=certificate_hash,
            timestamp=timestamp,
        )

    def _compute_proposal_id(
        self,
        model_hash: str,
        config_key: str,
        proposer: str,
        timestamp: float,
    ) -> str:
        """Compute unique proposal ID."""
        data = {
            "model_hash": model_hash,
            "config_key": config_key,
            "proposer": proposer,
            "timestamp": timestamp,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get_proposal(self, proposal_id: str) -> PromotionProposal | None:
        """Get a proposal by ID."""
        return self._proposals.get(proposal_id)

    def get_proposal_votes(self, proposal_id: str) -> list[PromotionVote]:
        """Get votes for a proposal."""
        return list(self._votes.get(proposal_id, []))

    def get_pending_proposals(self) -> list[PromotionProposal]:
        """Get all proposals awaiting consensus."""
        return [
            p
            for p in self._proposals.values()
            if p.proposal_id not in self._consensus_reached
        ]

    def get_approved_promotions(self) -> list[PromotionConsensusResult]:
        """Get all approved promotion results."""
        return [
            r for r in self._consensus_reached.values() if r.approved
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_proposals": len(self._proposals),
            "pending_proposals": len(self.get_pending_proposals()),
            "consensus_reached": len(self._consensus_reached),
            "approved_count": len(self.get_approved_promotions()),
            "node_id": self.node_id,
        }

    def clear(self) -> None:
        """Clear all proposals and votes."""
        self._proposals.clear()
        self._votes.clear()
        self._consensus_reached.clear()
        for waiters in self._consensus_waiters.values():
            for waiter in waiters:
                waiter.set()
        self._consensus_waiters.clear()


# Singleton instance
_promotion_consensus_manager: PromotionConsensusManager | None = None


def get_promotion_consensus_manager(
    dag: HashgraphDAG | None = None,
    node_id: str = "",
    config: PromotionConsensusConfig | None = None,
) -> PromotionConsensusManager:
    """Get or create the singleton promotion consensus manager.

    Args:
        dag: Hashgraph DAG (required on first call)
        node_id: Local node identifier
        config: Configuration

    Returns:
        The singleton manager instance

    Raises:
        ValueError: If dag is None on first call
    """
    global _promotion_consensus_manager

    if _promotion_consensus_manager is None:
        if dag is None:
            raise ValueError("DAG required for initial manager creation")
        _promotion_consensus_manager = PromotionConsensusManager(
            dag=dag,
            node_id=node_id,
            config=config,
        )

    return _promotion_consensus_manager


def reset_promotion_consensus_manager() -> None:
    """Reset the singleton manager (for testing)."""
    global _promotion_consensus_manager
    _promotion_consensus_manager = None


__all__ = [
    "PromotionEventType",
    "VoteType",
    "EvaluationEvidence",
    "PromotionProposal",
    "PromotionVote",
    "PromotionCertificate",
    "PromotionConsensusResult",
    "PromotionConsensusConfig",
    "PromotionConsensusManager",
    "get_promotion_consensus_manager",
    "reset_promotion_consensus_manager",
]
