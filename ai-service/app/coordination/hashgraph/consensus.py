"""Virtual voting consensus algorithm for hashgraph.

Implements the core hashgraph consensus mechanism:
- Virtual voting: Compute votes from DAG ancestry without message exchange
- Strongly seeing: X strongly sees Y if X can see 2/3+ of nodes seeing Y
- Consensus order: Events reach consensus when famous witnesses are decided

Key Insight: Nodes don't send vote messages. Instead, they compute what
other nodes would vote based on the DAG structure (gossip-about-gossip).

Byzantine Fault Tolerance:
- Tolerates up to n/3 Byzantine (malicious) nodes
- Uses 2/3 supermajority for all decisions
- Equivocation (forking) is detectable via DAG structure

Usage:
    from app.coordination.hashgraph import ConsensusEngine, HashgraphDAG

    dag = HashgraphDAG(node_id="node-1")
    engine = ConsensusEngine(dag, supermajority_fraction=2/3)

    # Check if event X strongly sees event Y
    result = engine.strongly_sees(x_hash, y_hash)
    if result.strongly_sees:
        print(f"X strongly sees Y via {len(result.seeing_witnesses)} witnesses")

    # Get consensus on events
    result = engine.get_consensus_for_round(round_num)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.hashgraph.dag import HashgraphDAG

logger = logging.getLogger(__name__)


class VoteType(Enum):
    """Types of virtual votes."""

    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


@dataclass
class VirtualVote:
    """A computed virtual vote.

    Represents a vote that a node would cast based on DAG ancestry.
    These are computed, not transmitted.

    Attributes:
        voter: Node ID casting the vote
        target: Event hash being voted on
        vote: Vote value (YES/NO/ABSTAIN)
        round: Round in which this vote is cast
        via_event: Event through which the vote is computed
    """

    voter: str
    target: str
    vote: VoteType
    round: int
    via_event: str = ""


@dataclass
class StronglySeeingResult:
    """Result of a strongly-seeing query.

    X strongly sees Y if X can see events from 2/3+ of nodes that
    themselves see Y. This is the key mechanism for virtual voting.

    Attributes:
        strongly_sees: Whether X strongly sees Y
        x_event: The seeing event hash
        y_event: The seen event hash
        seeing_creators: Set of creators whose events X sees that also see Y
        total_creators: Total number of creators in the network
        required_creators: Number required for supermajority
    """

    strongly_sees: bool
    x_event: str
    y_event: str
    seeing_creators: set[str] = field(default_factory=set)
    total_creators: int = 0
    required_creators: int = 0


@dataclass
class ConsensusResult:
    """Result of consensus for a round.

    Contains the events that have achieved consensus ordering.

    Attributes:
        round: Round number
        consensus_events: Events in consensus order
        pending_events: Events not yet in consensus
        famous_witnesses: Witness events decided as famous
    """

    round: int
    consensus_events: list[str] = field(default_factory=list)
    pending_events: list[str] = field(default_factory=list)
    famous_witnesses: list[str] = field(default_factory=list)


class ConsensusEngine:
    """Virtual voting consensus engine.

    Computes consensus ordering using hashgraph's virtual voting algorithm.
    Requires no message exchange - votes are derived from DAG structure.

    Attributes:
        dag: The underlying hashgraph DAG
        supermajority_fraction: Fraction required for consensus (default 2/3)
    """

    def __init__(
        self,
        dag: HashgraphDAG,
        supermajority_fraction: float = 2 / 3,
    ):
        """Initialize consensus engine.

        Args:
            dag: HashgraphDAG to compute consensus on
            supermajority_fraction: Fraction of nodes for supermajority
        """
        self.dag = dag
        self.supermajority_fraction = supermajority_fraction

        # Cache for strongly-seeing results
        self._strongly_seeing_cache: dict[tuple[str, str], StronglySeeingResult] = {}

    def get_supermajority_threshold(self) -> int:
        """Compute number of nodes needed for supermajority.

        Returns:
            Minimum number of nodes for 2/3+ majority
        """
        n = len(self.dag.get_known_creators())
        if n == 0:
            return 1
        # Ceiling of 2n/3
        return (2 * n + 2) // 3

    def can_see(self, x_hash: str, y_hash: str) -> bool:
        """Check if event X can see event Y.

        X can see Y if Y is an ancestor of X (Y happened before X).

        Args:
            x_hash: Seeing event
            y_hash: Seen event

        Returns:
            True if X can see Y
        """
        if x_hash == y_hash:
            return True
        return self.dag.is_ancestor(y_hash, x_hash)

    def strongly_sees(self, x_hash: str, y_hash: str) -> StronglySeeingResult:
        """Check if event X strongly sees event Y.

        X strongly sees Y if X can see events from 2/3+ of creators that
        themselves can see Y.

        This is the key mechanism enabling virtual voting without
        message exchange.

        Args:
            x_hash: The seeing event
            y_hash: The seen event

        Returns:
            StronglySeeingResult with details
        """
        # Check cache
        cache_key = (x_hash, y_hash)
        if cache_key in self._strongly_seeing_cache:
            return self._strongly_seeing_cache[cache_key]

        result = self._compute_strongly_sees(x_hash, y_hash)
        self._strongly_seeing_cache[cache_key] = result
        return result

    def _compute_strongly_sees(self, x_hash: str, y_hash: str) -> StronglySeeingResult:
        """Compute strongly-seeing relationship.

        Algorithm:
        1. For each creator in the network
        2. Find their most recent event that X can see
        3. Check if that event can see Y
        4. If 2/3+ creators have such events, X strongly sees Y

        Args:
            x_hash: The seeing event
            y_hash: The seen event

        Returns:
            StronglySeeingResult
        """
        creators = self.dag.get_known_creators()
        total_creators = len(creators)
        required = self.get_supermajority_threshold()

        if total_creators == 0:
            return StronglySeeingResult(
                strongly_sees=False,
                x_event=x_hash,
                y_event=y_hash,
                total_creators=0,
                required_creators=1,
            )

        seeing_creators: set[str] = set()

        for creator in creators:
            # Get all events from this creator that X can see
            creator_events = self.dag.get_creator_chain(creator)
            for event_hash in reversed(creator_events):  # Most recent first
                if self.can_see(x_hash, event_hash):
                    # Check if this creator's event can see Y
                    if self.can_see(event_hash, y_hash):
                        seeing_creators.add(creator)
                    break  # Only need most recent visible event

        strongly_sees = len(seeing_creators) >= required

        return StronglySeeingResult(
            strongly_sees=strongly_sees,
            x_event=x_hash,
            y_event=y_hash,
            seeing_creators=seeing_creators,
            total_creators=total_creators,
            required_creators=required,
        )

    def compute_virtual_votes(
        self,
        voter_round: int,
        target_hash: str,
    ) -> list[VirtualVote]:
        """Compute virtual votes for an event.

        Determines what each node would vote regarding the target event,
        based on DAG ancestry from the given round.

        Args:
            voter_round: Round from which to compute votes
            target_hash: Event being voted on

        Returns:
            List of virtual votes
        """
        votes: list[VirtualVote] = []
        witnesses = self.dag.get_round_witnesses(voter_round)

        for witness in witnesses:
            # A witness votes YES if it can strongly see the target
            # This is the core virtual voting mechanism
            result = self.strongly_sees(witness.event_hash, target_hash)

            if result.strongly_sees:
                vote_type = VoteType.YES
            else:
                # Check if it can see the target at all
                if self.can_see(witness.event_hash, target_hash):
                    vote_type = VoteType.YES
                else:
                    vote_type = VoteType.NO

            votes.append(
                VirtualVote(
                    voter=witness.creator,
                    target=target_hash,
                    vote=vote_type,
                    round=voter_round,
                    via_event=witness.event_hash,
                )
            )

        return votes

    def count_votes(
        self,
        votes: list[VirtualVote],
    ) -> tuple[int, int, int]:
        """Count vote tallies.

        Args:
            votes: List of virtual votes

        Returns:
            Tuple of (yes_count, no_count, abstain_count)
        """
        yes_count = sum(1 for v in votes if v.vote == VoteType.YES)
        no_count = sum(1 for v in votes if v.vote == VoteType.NO)
        abstain_count = sum(1 for v in votes if v.vote == VoteType.ABSTAIN)
        return yes_count, no_count, abstain_count

    def has_supermajority_yes(self, votes: list[VirtualVote]) -> bool:
        """Check if votes have supermajority YES.

        Args:
            votes: List of virtual votes

        Returns:
            True if 2/3+ voted YES
        """
        if not votes:
            return False
        yes_count, _, _ = self.count_votes(votes)
        required = self.get_supermajority_threshold()
        return yes_count >= required

    def has_supermajority_no(self, votes: list[VirtualVote]) -> bool:
        """Check if votes have supermajority NO.

        Args:
            votes: List of virtual votes

        Returns:
            True if 2/3+ voted NO
        """
        if not votes:
            return False
        _, no_count, _ = self.count_votes(votes)
        required = self.get_supermajority_threshold()
        return no_count >= required

    def get_consensus_timestamp(self, event_hash: str) -> float | None:
        """Compute consensus timestamp for an event.

        The consensus timestamp is the median of timestamps when
        famous witnesses from later rounds first see the event.

        Args:
            event_hash: Event to get timestamp for

        Returns:
            Consensus timestamp or None if not yet decided
        """
        event = self.dag.get_event(event_hash)
        if not event:
            return None

        # Collect timestamps from witnesses that see this event
        seeing_timestamps: list[float] = []
        latest_round = self.dag.get_latest_round()

        for round_num in range(event.round_number + 1, latest_round + 1):
            witnesses = self.dag.get_round_witnesses(round_num)
            for witness in witnesses:
                if self.can_see(witness.event_hash, event_hash):
                    seeing_timestamps.append(witness.timestamp)

        if not seeing_timestamps:
            return None

        # Median timestamp
        seeing_timestamps.sort()
        mid = len(seeing_timestamps) // 2
        if len(seeing_timestamps) % 2 == 0:
            return (seeing_timestamps[mid - 1] + seeing_timestamps[mid]) / 2
        return seeing_timestamps[mid]

    def get_consensus_for_round(self, round_number: int) -> ConsensusResult:
        """Get consensus state for a round.

        Determines which events have achieved consensus ordering
        based on famous witness decisions.

        Args:
            round_number: Round to check

        Returns:
            ConsensusResult with ordered and pending events
        """
        round_events = self.dag.get_round_events(round_number)
        witnesses = self.dag.get_round_witnesses(round_number)

        # For consensus, we need later rounds to decide witness fame
        consensus_events: list[str] = []
        pending_events: list[str] = []
        famous_witnesses: list[str] = []

        # Check each witness for fame (simplified)
        for witness in witnesses:
            node = self.dag.get_node(witness.event_hash)
            if node and node.is_famous:
                famous_witnesses.append(witness.event_hash)

        # Events with decided ordering go to consensus
        for event in round_events:
            # An event is in consensus if it's received by famous witnesses
            in_consensus = False
            for famous_hash in famous_witnesses:
                if self.can_see(famous_hash, event.event_hash):
                    in_consensus = True
                    break

            if in_consensus:
                consensus_events.append(event.event_hash)
            else:
                pending_events.append(event.event_hash)

        return ConsensusResult(
            round=round_number,
            consensus_events=consensus_events,
            pending_events=pending_events,
            famous_witnesses=famous_witnesses,
        )

    def is_consensus_reached(self, event_hash: str) -> bool:
        """Check if consensus has been reached for an event.

        Consensus is reached when the event's round has famous
        witnesses decided and the event is received by them.

        Args:
            event_hash: Event to check

        Returns:
            True if consensus reached
        """
        event = self.dag.get_event(event_hash)
        if not event:
            return False

        result = self.get_consensus_for_round(event.round_number)
        return event_hash in result.consensus_events

    def clear_cache(self) -> None:
        """Clear the strongly-seeing cache."""
        self._strongly_seeing_cache.clear()


__all__ = [
    "ConsensusEngine",
    "ConsensusResult",
    "VirtualVote",
    "VoteType",
    "StronglySeeingResult",
]
