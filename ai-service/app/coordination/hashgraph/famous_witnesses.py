"""Famous witness selection and fame determination.

In hashgraph consensus, "famous witnesses" are witnesses that 2/3+ of
later round witnesses can "strongly see". Once a witness is decided
as famous, events it receives have consensus ordering.

Key Concepts:
- Witness: First event from a creator in a round
- Famous: A witness strongly seen by 2/3+ of next round witnesses
- Round Received: Round when an event is first received by famous witnesses

The fame determination uses virtual voting:
1. Round r+1 witnesses vote on round r witness fame
2. If 2/3+ vote YES or NO, fame is decided
3. If undecided, iterate to round r+2, r+3, etc.

Usage:
    from app.coordination.hashgraph import (
        HashgraphDAG,
        ConsensusEngine,
        WitnessSelector,
    )

    dag = HashgraphDAG(node_id="node-1")
    engine = ConsensusEngine(dag)
    selector = WitnessSelector(dag, engine)

    # Determine fame for round witnesses
    round_info = selector.decide_round_fame(round_number=5)
    for witness in round_info.famous_witnesses:
        print(f"Famous: {witness.event_hash[:8]}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.hashgraph.consensus import ConsensusEngine
    from app.coordination.hashgraph.dag import HashgraphDAG

logger = logging.getLogger(__name__)


class FameStatus(Enum):
    """Fame determination status for a witness."""

    UNDECIDED = "undecided"
    FAMOUS = "famous"
    NOT_FAMOUS = "not_famous"


@dataclass
class WitnessInfo:
    """Information about a witness event.

    Attributes:
        event_hash: Hash of the witness event
        creator: Node ID that created the witness
        round: Round number of the witness
        fame_status: Current fame determination status
        fame_decided_round: Round in which fame was decided (if decided)
        vote_count_yes: Number of YES votes from later witnesses
        vote_count_no: Number of NO votes from later witnesses
    """

    event_hash: str
    creator: str
    round: int
    fame_status: FameStatus = FameStatus.UNDECIDED
    fame_decided_round: int | None = None
    vote_count_yes: int = 0
    vote_count_no: int = 0


@dataclass
class RoundInfo:
    """Information about witnesses in a round.

    Attributes:
        round: Round number
        witnesses: All witnesses in this round
        famous_witnesses: Witnesses decided as famous
        not_famous_witnesses: Witnesses decided as not famous
        undecided_witnesses: Witnesses with undecided fame
        is_fully_decided: Whether all witnesses have decided fame
    """

    round: int
    witnesses: list[WitnessInfo] = field(default_factory=list)
    famous_witnesses: list[WitnessInfo] = field(default_factory=list)
    not_famous_witnesses: list[WitnessInfo] = field(default_factory=list)
    undecided_witnesses: list[WitnessInfo] = field(default_factory=list)

    @property
    def is_fully_decided(self) -> bool:
        """Check if all witnesses have decided fame."""
        return len(self.undecided_witnesses) == 0 and len(self.witnesses) > 0


class WitnessSelector:
    """Selects famous witnesses using virtual voting.

    Determines which witnesses are famous by computing virtual votes
    from later round witnesses. A witness is famous if 2/3+ of later
    witnesses strongly see it.

    Attributes:
        dag: The hashgraph DAG
        consensus: The consensus engine for strongly-seeing queries
        coin_round_interval: Rounds between coin flips for undecided fame
    """

    def __init__(
        self,
        dag: HashgraphDAG,
        consensus: ConsensusEngine,
        coin_round_interval: int = 10,
    ):
        """Initialize witness selector.

        Args:
            dag: HashgraphDAG for event access
            consensus: ConsensusEngine for strongly-seeing queries
            coin_round_interval: How often to use coin flip for undecided
        """
        self.dag = dag
        self.consensus = consensus
        self.coin_round_interval = coin_round_interval

        # Track witness info
        self._witness_info: dict[str, WitnessInfo] = {}
        self._round_info: dict[int, RoundInfo] = {}

    def get_witness_info(self, event_hash: str) -> WitnessInfo | None:
        """Get witness info for an event.

        Args:
            event_hash: Witness event hash

        Returns:
            WitnessInfo or None if not a witness
        """
        return self._witness_info.get(event_hash)

    def get_round_info(self, round_number: int) -> RoundInfo:
        """Get or create round info.

        Args:
            round_number: Round to get info for

        Returns:
            RoundInfo for the round
        """
        if round_number not in self._round_info:
            self._round_info[round_number] = RoundInfo(round=round_number)
            self._populate_round_witnesses(round_number)
        return self._round_info[round_number]

    def _populate_round_witnesses(self, round_number: int) -> None:
        """Populate witness info for a round.

        Args:
            round_number: Round to populate
        """
        round_info = self._round_info[round_number]
        witnesses = self.dag.get_round_witnesses(round_number)

        for witness in witnesses:
            if witness.event_hash not in self._witness_info:
                info = WitnessInfo(
                    event_hash=witness.event_hash,
                    creator=witness.creator,
                    round=round_number,
                )
                self._witness_info[witness.event_hash] = info

            info = self._witness_info[witness.event_hash]
            round_info.witnesses.append(info)
            round_info.undecided_witnesses.append(info)

    def decide_witness_fame(
        self,
        witness_hash: str,
        max_rounds_ahead: int = 20,
    ) -> WitnessInfo:
        """Determine fame for a single witness.

        Uses virtual voting from later round witnesses to decide
        if this witness is famous.

        Algorithm:
        1. Get witnesses from round r+1
        2. Each r+1 witness votes YES if it can strongly see the target
        3. If 2/3+ vote YES → famous, 2/3+ NO → not famous
        4. If undecided, continue to round r+2, etc.

        Args:
            witness_hash: Hash of witness to decide
            max_rounds_ahead: Maximum rounds to look ahead for decision

        Returns:
            Updated WitnessInfo with fame decision
        """
        info = self._witness_info.get(witness_hash)
        if not info:
            # Create info for unknown witness
            event = self.dag.get_event(witness_hash)
            if not event:
                raise ValueError(f"Unknown event: {witness_hash}")
            info = WitnessInfo(
                event_hash=witness_hash,
                creator=event.creator,
                round=event.round_number,
            )
            self._witness_info[witness_hash] = info

        if info.fame_status != FameStatus.UNDECIDED:
            return info

        witness_round = info.round
        latest_round = self.dag.get_latest_round()

        # Check rounds r+1, r+2, ... for fame decision
        for voting_round in range(witness_round + 1, min(latest_round + 1, witness_round + max_rounds_ahead + 1)):
            decided = self._check_fame_at_round(info, voting_round)
            if decided:
                break

        return info

    def _check_fame_at_round(self, info: WitnessInfo, voting_round: int) -> bool:
        """Check if fame can be decided at a specific round.

        Args:
            info: WitnessInfo to update
            voting_round: Round to check votes from

        Returns:
            True if fame was decided
        """
        threshold = self.consensus.get_supermajority_threshold()
        voting_witnesses = self.dag.get_round_witnesses(voting_round)

        yes_votes = 0
        no_votes = 0

        for voter in voting_witnesses:
            # A witness votes YES if it strongly sees the target witness
            result = self.consensus.strongly_sees(voter.event_hash, info.event_hash)
            if result.strongly_sees:
                yes_votes += 1
            else:
                no_votes += 1

        info.vote_count_yes = yes_votes
        info.vote_count_no = no_votes

        # Check for supermajority
        if yes_votes >= threshold:
            info.fame_status = FameStatus.FAMOUS
            info.fame_decided_round = voting_round
            self._update_round_info(info)
            return True
        elif no_votes >= threshold:
            info.fame_status = FameStatus.NOT_FAMOUS
            info.fame_decided_round = voting_round
            self._update_round_info(info)
            return True

        # Check for coin round (random tie-breaker after many undecided rounds)
        rounds_diff = voting_round - info.round
        if rounds_diff > 0 and rounds_diff % self.coin_round_interval == 0:
            # Use middle bit of voting round's witnesses' signatures as coin
            # For simplicity, we use first witness hash's middle character
            if voting_witnesses:
                coin_bit = ord(voting_witnesses[0].event_hash[16]) % 2
                if coin_bit == 1:
                    info.fame_status = FameStatus.FAMOUS
                else:
                    info.fame_status = FameStatus.NOT_FAMOUS
                info.fame_decided_round = voting_round
                self._update_round_info(info)
                return True

        return False

    def _update_round_info(self, info: WitnessInfo) -> None:
        """Update round info after fame decision.

        Args:
            info: WitnessInfo that was decided
        """
        round_info = self._round_info.get(info.round)
        if not round_info:
            return

        # Remove from undecided
        round_info.undecided_witnesses = [
            w for w in round_info.undecided_witnesses
            if w.event_hash != info.event_hash
        ]

        # Add to appropriate list
        if info.fame_status == FameStatus.FAMOUS:
            round_info.famous_witnesses.append(info)
        elif info.fame_status == FameStatus.NOT_FAMOUS:
            round_info.not_famous_witnesses.append(info)

    def decide_round_fame(
        self,
        round_number: int,
        max_rounds_ahead: int = 20,
    ) -> RoundInfo:
        """Decide fame for all witnesses in a round.

        Args:
            round_number: Round to decide
            max_rounds_ahead: Maximum rounds to look ahead

        Returns:
            RoundInfo with fame decisions
        """
        round_info = self.get_round_info(round_number)

        # Decide fame for each undecided witness
        for witness_info in list(round_info.undecided_witnesses):
            self.decide_witness_fame(
                witness_info.event_hash,
                max_rounds_ahead=max_rounds_ahead,
            )

        return round_info

    def get_famous_witnesses(self, round_number: int) -> list[WitnessInfo]:
        """Get famous witnesses for a round.

        Args:
            round_number: Round to query

        Returns:
            List of famous witness info
        """
        round_info = self.get_round_info(round_number)
        return round_info.famous_witnesses.copy()

    def get_received_round(self, event_hash: str) -> int | None:
        """Get the round an event was received (consensus ordering).

        An event's received round is the round of the first famous
        witness that can see it.

        Args:
            event_hash: Event to check

        Returns:
            Received round or None if not yet decided
        """
        event = self.dag.get_event(event_hash)
        if not event:
            return None

        # Check each round from event's round onward
        latest = self.dag.get_latest_round()
        for check_round in range(event.round_number, latest + 1):
            round_info = self.get_round_info(check_round)
            if not round_info.is_fully_decided:
                continue

            for famous in round_info.famous_witnesses:
                if self.consensus.can_see(famous.event_hash, event_hash):
                    return check_round

        return None

    def get_consensus_order_in_round(self, round_number: int) -> list[str]:
        """Get events in consensus order for a round.

        Orders events by:
        1. Received round (when famous witnesses first see them)
        2. Consensus timestamp (median of famous witness times)
        3. Event hash (for determinism)

        Args:
            round_number: Round to order

        Returns:
            List of event hashes in consensus order
        """
        round_info = self.get_round_info(round_number)
        if not round_info.is_fully_decided:
            return []

        events = self.dag.get_round_events(round_number)
        famous_witnesses = round_info.famous_witnesses

        # Filter to events received by famous witnesses
        received_events: list[tuple[str, float, str]] = []
        for event in events:
            for famous in famous_witnesses:
                if self.consensus.can_see(famous.event_hash, event.event_hash):
                    timestamp = self.consensus.get_consensus_timestamp(event.event_hash)
                    if timestamp:
                        received_events.append((event.event_hash, timestamp, event.event_hash))
                    break

        # Sort by timestamp, then hash
        received_events.sort(key=lambda x: (x[1], x[2]))
        return [e[0] for e in received_events]

    def clear(self) -> None:
        """Clear all witness tracking state."""
        self._witness_info.clear()
        self._round_info.clear()


__all__ = [
    "WitnessSelector",
    "WitnessInfo",
    "FameStatus",
    "RoundInfo",
]
