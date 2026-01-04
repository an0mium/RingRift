"""Hashgraph Event dataclass with parent hash references.

Implements the core data structure for "gossip about gossip" - each event
contains hashes of its parent events, creating a Directed Acyclic Graph (DAG)
that enables Byzantine fault detection and deterministic consensus.

Key Properties:
- self_parent: Hash of this creator's previous event (forms creator's chain)
- other_parent: Hash of event received from another node (cross-links chains)
- event_hash: SHA256 of canonical JSON representation (immutable identifier)

The parent references enable:
1. Causal ordering (happened-before relationships)
2. Equivocation detection (forked chains = Byzantine behavior)
3. Virtual voting without explicit vote messages

Usage:
    # Create first event (no parents)
    genesis = HashgraphEvent.create(
        creator="node-1",
        payload={"type": "genesis"},
    )

    # Create event with parents
    event = HashgraphEvent.create(
        creator="node-1",
        payload={"type": "evaluation", "elo": 1450},
        self_parent=genesis.event_hash,
        other_parent=received_event_hash,
    )

    # Verify integrity
    assert event.verify_hash()
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """Types of hashgraph events for RingRift consensus."""

    # Genesis events (no parents)
    GENESIS = "genesis"

    # Evaluation events
    EVALUATION_RESULT = "evaluation_result"
    EVALUATION_CONSENSUS = "evaluation_consensus"

    # Model promotion events
    PROMOTION_PROPOSAL = "promotion_proposal"
    PROMOTION_VOTE = "promotion_vote"
    PROMOTION_CERTIFIED = "promotion_certified"

    # Elo synchronization
    ELO_UPDATE = "elo_update"
    ELO_CONSENSUS = "elo_consensus"

    # Generic sync events
    STATE_SYNC = "state_sync"
    HEARTBEAT = "heartbeat"


def canonical_json(obj: dict[str, Any]) -> str:
    """Convert dict to canonical JSON for deterministic hashing.

    Uses sorted keys and no extra whitespace to ensure the same dict
    always produces the same JSON string across all nodes.

    Args:
        obj: Dictionary to serialize

    Returns:
        Canonical JSON string
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_event_hash(
    creator: str,
    payload: dict[str, Any],
    timestamp: float,
    self_parent: str | None,
    other_parent: str | None,
    round_number: int,
) -> str:
    """Compute SHA256 hash of event contents.

    Hash is computed over canonical JSON of all event fields except
    the hash itself. This is the immutable identifier for the event.

    Args:
        creator: Node ID that created the event
        payload: Event data
        timestamp: Local timestamp when event was created
        self_parent: Hash of creator's previous event
        other_parent: Hash of event received from another node
        round_number: Hashgraph round number

    Returns:
        Hex-encoded SHA256 hash
    """
    content = {
        "creator": creator,
        "payload": payload,
        "timestamp": timestamp,
        "self_parent": self_parent,
        "other_parent": other_parent,
        "round_number": round_number,
    }
    json_bytes = canonical_json(content).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


@dataclass(frozen=True)
class HashgraphEvent:
    """Immutable hashgraph event with parent references.

    Each event in the hashgraph contains references to two parent events:
    - self_parent: The creator's own previous event (forms a chain)
    - other_parent: An event received from another node (creates DAG structure)

    The frozen=True makes events immutable, ensuring hash integrity.

    Attributes:
        creator: Node ID that created this event
        payload: Event-specific data (evaluation result, vote, etc.)
        timestamp: Local timestamp when event was created (seconds since epoch)
        self_parent: Hash of creator's previous event (None for genesis)
        other_parent: Hash of event received from another node (None for genesis)
        round_number: Hashgraph round (derived from parents' rounds)
        event_hash: SHA256 hash of event contents (computed on creation)
        signature: Optional cryptographic signature (for Byzantine tolerance)
    """

    creator: str
    payload: dict[str, Any]
    timestamp: float
    self_parent: str | None
    other_parent: str | None
    round_number: int
    event_hash: str
    signature: str | None = None

    @classmethod
    def create(
        cls,
        creator: str,
        payload: dict[str, Any],
        self_parent: str | None = None,
        other_parent: str | None = None,
        round_number: int = 0,
        timestamp: float | None = None,
        signature: str | None = None,
    ) -> HashgraphEvent:
        """Create a new hashgraph event.

        Factory method that automatically computes the event hash.

        Args:
            creator: Node ID creating this event
            payload: Event data
            self_parent: Hash of creator's previous event
            other_parent: Hash of received event from another node
            round_number: Hashgraph round (typically max(parent_rounds) + 1)
            timestamp: Event timestamp (defaults to current time)
            signature: Optional cryptographic signature

        Returns:
            New immutable HashgraphEvent
        """
        ts = timestamp if timestamp is not None else time.time()
        event_hash = compute_event_hash(
            creator=creator,
            payload=payload,
            timestamp=ts,
            self_parent=self_parent,
            other_parent=other_parent,
            round_number=round_number,
        )
        return cls(
            creator=creator,
            payload=payload,
            timestamp=ts,
            self_parent=self_parent,
            other_parent=other_parent,
            round_number=round_number,
            event_hash=event_hash,
            signature=signature,
        )

    @classmethod
    def create_genesis(cls, creator: str, payload: dict[str, Any] | None = None) -> HashgraphEvent:
        """Create a genesis event (no parents).

        Genesis events are the starting point for each node's event chain.
        They have no parents and are in round 0.

        Args:
            creator: Node ID creating the genesis event
            payload: Optional genesis payload

        Returns:
            Genesis HashgraphEvent
        """
        return cls.create(
            creator=creator,
            payload=payload or {"type": EventType.GENESIS.value},
            self_parent=None,
            other_parent=None,
            round_number=0,
        )

    def verify_hash(self) -> bool:
        """Verify that the event hash is correct.

        Recomputes the hash from event contents and compares.
        Used to detect tampering or transmission errors.

        Returns:
            True if hash is valid
        """
        computed = compute_event_hash(
            creator=self.creator,
            payload=self.payload,
            timestamp=self.timestamp,
            self_parent=self.self_parent,
            other_parent=self.other_parent,
            round_number=self.round_number,
        )
        return computed == self.event_hash

    def is_genesis(self) -> bool:
        """Check if this is a genesis event (no parents)."""
        return self.self_parent is None and self.other_parent is None

    def get_parent_hashes(self) -> list[str]:
        """Get list of non-None parent hashes."""
        parents = []
        if self.self_parent:
            parents.append(self.self_parent)
        if self.other_parent:
            parents.append(self.other_parent)
        return parents

    def get_event_type(self) -> EventType | None:
        """Extract event type from payload if present."""
        type_str = self.payload.get("type")
        if type_str:
            try:
                return EventType(type_str)
            except ValueError:
                return None
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "creator": self.creator,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "self_parent": self.self_parent,
            "other_parent": self.other_parent,
            "round_number": self.round_number,
            "event_hash": self.event_hash,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HashgraphEvent:
        """Reconstruct event from dictionary.

        Args:
            data: Dictionary representation of event

        Returns:
            HashgraphEvent instance

        Raises:
            ValueError: If event hash doesn't match contents
        """
        event = cls(
            creator=data["creator"],
            payload=data["payload"],
            timestamp=data["timestamp"],
            self_parent=data.get("self_parent"),
            other_parent=data.get("other_parent"),
            round_number=data.get("round_number", 0),
            event_hash=data["event_hash"],
            signature=data.get("signature"),
        )
        if not event.verify_hash():
            raise ValueError(f"Event hash mismatch: {event.event_hash}")
        return event

    def __repr__(self) -> str:
        event_type = self.get_event_type()
        type_str = event_type.value if event_type else "unknown"
        return (
            f"HashgraphEvent("
            f"creator={self.creator!r}, "
            f"type={type_str!r}, "
            f"round={self.round_number}, "
            f"hash={self.event_hash[:8]}...)"
        )


@dataclass
class EventBatch:
    """Batch of events for efficient gossip transmission.

    Groups multiple events together for network efficiency.
    """

    events: list[HashgraphEvent] = field(default_factory=list)
    sender: str = ""
    batch_timestamp: float = field(default_factory=time.time)

    def add(self, event: HashgraphEvent) -> None:
        """Add event to batch."""
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        """Serialize batch for transmission."""
        return {
            "events": [e.to_dict() for e in self.events],
            "sender": self.sender,
            "batch_timestamp": self.batch_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventBatch:
        """Deserialize batch from transmission."""
        return cls(
            events=[HashgraphEvent.from_dict(e) for e in data["events"]],
            sender=data["sender"],
            batch_timestamp=data.get("batch_timestamp", time.time()),
        )


__all__ = [
    "HashgraphEvent",
    "EventType",
    "EventBatch",
    "canonical_json",
    "compute_event_hash",
]
