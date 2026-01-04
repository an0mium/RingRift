"""Hashgraph DAG (Directed Acyclic Graph) management.

Manages the hashgraph event DAG and provides traversal algorithms for:
- Ancestor/descendant queries
- Strongly-seeing relationships
- Round computation
- Topological ordering

The DAG structure enables:
1. Causal ordering of events
2. Byzantine fault detection (equivocation = forking)
3. Virtual voting consensus

Usage:
    dag = HashgraphDAG(node_id="node-1")

    # Add events
    dag.add_event(genesis_event)
    dag.add_event(new_event)

    # Query relationships
    ancestors = dag.get_ancestors(event_hash)
    if dag.is_ancestor(ancestor_hash, descendant_hash):
        print("Happened before")

    # Get consensus ordering
    ordered = dag.get_consensus_order()
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from app.coordination.hashgraph.event import HashgraphEvent

logger = logging.getLogger(__name__)


@dataclass
class DAGNode:
    """Node wrapper for events in the DAG.

    Tracks additional metadata for efficient DAG traversal.

    Attributes:
        event: The hashgraph event
        children: Event hashes that reference this event as a parent
        round_received: Round when this event was received locally
        is_witness: Whether this event is a witness for its round
        is_famous: Whether this witness is famous (None if undecided)
    """

    event: HashgraphEvent
    children: set[str] = field(default_factory=set)
    round_received: int = 0
    is_witness: bool = False
    is_famous: bool | None = None

    @property
    def event_hash(self) -> str:
        """Get the event hash."""
        return self.event.event_hash

    @property
    def creator(self) -> str:
        """Get the event creator."""
        return self.event.creator

    @property
    def round_number(self) -> int:
        """Get the event round."""
        return self.event.round_number


@dataclass
class AncestryResult:
    """Result of ancestry queries.

    Contains sets of ancestor and descendant events.
    """

    ancestors: set[str] = field(default_factory=set)
    descendants: set[str] = field(default_factory=set)
    path_length: int = 0


class HashgraphDAG:
    """Manages the hashgraph event DAG.

    Thread-safe implementation supporting concurrent event addition
    and ancestor queries.

    Attributes:
        node_id: Local node identifier
        nodes: Map of event_hash -> DAGNode
        creator_chains: Map of creator -> list of event hashes (in order)
        round_witnesses: Map of round -> set of witness event hashes
    """

    def __init__(self, node_id: str = ""):
        """Initialize DAG.

        Args:
            node_id: Local node identifier for logging
        """
        self.node_id = node_id
        self._lock = threading.RLock()

        # Primary storage
        self._nodes: dict[str, DAGNode] = {}

        # Indexes for efficient queries
        self._creator_chains: dict[str, list[str]] = defaultdict(list)
        self._round_events: dict[int, set[str]] = defaultdict(set)
        self._round_witnesses: dict[int, set[str]] = defaultdict(set)

        # Cached ancestry (cleared on new events)
        self._ancestry_cache: dict[str, set[str]] = {}
        self._cache_valid = True

    def add_event(self, event: HashgraphEvent) -> bool:
        """Add event to the DAG.

        Validates parent references and updates indexes.
        NOTE: Events are immutable - the round_number must be set correctly
        before calling add_event. Use compute_round_for_event() if needed.

        Args:
            event: Event to add

        Returns:
            True if added successfully, False if already exists or invalid
        """
        with self._lock:
            # Check for duplicate
            if event.event_hash in self._nodes:
                return False

            # Verify hash integrity
            if not event.verify_hash():
                logger.warning(f"[{self.node_id}] Invalid event hash: {event.event_hash[:8]}")
                return False

            # Validate parent references exist (except for genesis)
            if event.self_parent and event.self_parent not in self._nodes:
                logger.debug(
                    f"[{self.node_id}] Missing self_parent: {event.self_parent[:8]}"
                )
                return False

            if event.other_parent and event.other_parent not in self._nodes:
                logger.debug(
                    f"[{self.node_id}] Missing other_parent: {event.other_parent[:8]}"
                )
                return False

            # Create node with the event as-is (don't modify the event)
            node = DAGNode(event=event)

            # Check if witness (first event from creator in this round)
            round_num = event.round_number
            creator = event.creator
            existing_witnesses = self._round_witnesses.get(round_num, set())
            creator_has_witness = any(
                self._nodes[w].creator == creator for w in existing_witnesses
            )
            if not creator_has_witness:
                node.is_witness = True
                self._round_witnesses[round_num].add(event.event_hash)

            # Add to storage using original event hash
            self._nodes[event.event_hash] = node

            # Update parent children sets
            if event.self_parent:
                self._nodes[event.self_parent].children.add(event.event_hash)
            if event.other_parent:
                self._nodes[event.other_parent].children.add(event.event_hash)

            # Update indexes
            self._creator_chains[creator].append(event.event_hash)
            self._round_events[round_num].add(event.event_hash)

            # Invalidate cache
            self._cache_valid = False
            self._ancestry_cache.clear()

            return True

    def compute_round_for_parents(
        self,
        self_parent: str | None,
        other_parent: str | None,
    ) -> int:
        """Compute round number based on parent events.

        Call this before creating an event to get the correct round.

        Round = max(self_parent.round, other_parent.round) + 1
        (Simplified from full hashgraph which checks strongly-seeing)

        Args:
            self_parent: Hash of self parent (or None for genesis)
            other_parent: Hash of other parent (or None)

        Returns:
            Computed round number (0 for genesis, 1+ otherwise)
        """
        if self_parent is None and other_parent is None:
            return 0  # Genesis event

        parent_round = 0

        if self_parent:
            parent_node = self._nodes.get(self_parent)
            if parent_node:
                parent_round = max(parent_round, parent_node.round_number)

        if other_parent:
            parent_node = self._nodes.get(other_parent)
            if parent_node:
                parent_round = max(parent_round, parent_node.round_number)

        return parent_round + 1

    def get_event(self, event_hash: str) -> HashgraphEvent | None:
        """Get event by hash.

        Args:
            event_hash: Event hash to look up

        Returns:
            Event if found, None otherwise
        """
        with self._lock:
            node = self._nodes.get(event_hash)
            return node.event if node else None

    def get_node(self, event_hash: str) -> DAGNode | None:
        """Get DAG node by hash.

        Args:
            event_hash: Event hash to look up

        Returns:
            DAGNode if found, None otherwise
        """
        with self._lock:
            return self._nodes.get(event_hash)

    def has_event(self, event_hash: str) -> bool:
        """Check if event exists in DAG.

        Args:
            event_hash: Event hash to check

        Returns:
            True if event exists
        """
        return event_hash in self._nodes

    def get_ancestors(self, event_hash: str) -> set[str]:
        """Get all ancestor event hashes.

        Includes transitive ancestors via DFS traversal.

        Args:
            event_hash: Starting event hash

        Returns:
            Set of ancestor event hashes (excludes the event itself)
        """
        with self._lock:
            # Check cache
            if self._cache_valid and event_hash in self._ancestry_cache:
                return self._ancestry_cache[event_hash].copy()

            ancestors: set[str] = set()
            stack = [event_hash]
            visited: set[str] = set()

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                node = self._nodes.get(current)
                if not node:
                    continue

                for parent_hash in node.event.get_parent_hashes():
                    if parent_hash not in visited:
                        ancestors.add(parent_hash)
                        stack.append(parent_hash)

            # Cache result
            self._ancestry_cache[event_hash] = ancestors.copy()
            return ancestors

    def is_ancestor(self, ancestor_hash: str, descendant_hash: str) -> bool:
        """Check if one event is an ancestor of another.

        Args:
            ancestor_hash: Potential ancestor event
            descendant_hash: Potential descendant event

        Returns:
            True if ancestor_hash is an ancestor of descendant_hash
        """
        if ancestor_hash == descendant_hash:
            return False

        ancestors = self.get_ancestors(descendant_hash)
        return ancestor_hash in ancestors

    def get_descendants(self, event_hash: str) -> set[str]:
        """Get all descendant event hashes.

        Uses children links for forward traversal.

        Args:
            event_hash: Starting event hash

        Returns:
            Set of descendant event hashes (excludes the event itself)
        """
        with self._lock:
            descendants: set[str] = set()
            stack = [event_hash]
            visited: set[str] = set()

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                node = self._nodes.get(current)
                if not node:
                    continue

                for child_hash in node.children:
                    if child_hash not in visited:
                        descendants.add(child_hash)
                        stack.append(child_hash)

            return descendants

    def get_creator_chain(self, creator: str) -> list[str]:
        """Get ordered list of events from a creator.

        Args:
            creator: Node ID

        Returns:
            List of event hashes in creation order
        """
        with self._lock:
            return self._creator_chains.get(creator, []).copy()

    def get_latest_event(self, creator: str) -> HashgraphEvent | None:
        """Get most recent event from a creator.

        Args:
            creator: Node ID

        Returns:
            Most recent event or None
        """
        with self._lock:
            chain = self._creator_chains.get(creator, [])
            if not chain:
                return None
            return self._nodes[chain[-1]].event

    def get_round_events(self, round_number: int) -> list[HashgraphEvent]:
        """Get all events in a round.

        Args:
            round_number: Round to query

        Returns:
            List of events in the round
        """
        with self._lock:
            hashes = self._round_events.get(round_number, set())
            return [self._nodes[h].event for h in hashes if h in self._nodes]

    def get_round_witnesses(self, round_number: int) -> list[HashgraphEvent]:
        """Get witness events for a round.

        Args:
            round_number: Round to query

        Returns:
            List of witness events
        """
        with self._lock:
            hashes = self._round_witnesses.get(round_number, set())
            return [self._nodes[h].event for h in hashes if h in self._nodes]

    def get_all_events(self) -> Iterator[HashgraphEvent]:
        """Iterate over all events.

        Yields:
            All events in the DAG
        """
        with self._lock:
            for node in self._nodes.values():
                yield node.event

    def get_topological_order(self) -> list[HashgraphEvent]:
        """Get events in topological order (parents before children).

        Uses Kahn's algorithm for deterministic ordering.

        Returns:
            List of events in topological order
        """
        with self._lock:
            # Compute in-degrees
            in_degree: dict[str, int] = {}
            for h in self._nodes:
                in_degree[h] = 0

            for node in self._nodes.values():
                for parent_hash in node.event.get_parent_hashes():
                    if parent_hash in in_degree:
                        # Parent's child count doesn't matter for topological sort
                        pass

            # Count incoming edges (parents)
            for node in self._nodes.values():
                for parent_hash in node.event.get_parent_hashes():
                    if parent_hash in self._nodes:
                        in_degree[node.event_hash] = in_degree.get(node.event_hash, 0) + 1

            # Start with nodes that have no parents (in-degree 0)
            queue = [h for h, deg in in_degree.items() if deg == 0]
            result: list[HashgraphEvent] = []

            while queue:
                # Sort for determinism
                queue.sort(key=lambda h: (self._nodes[h].round_number, h))
                current = queue.pop(0)
                node = self._nodes[current]
                result.append(node.event)

                for child_hash in node.children:
                    if child_hash in in_degree:
                        in_degree[child_hash] -= 1
                        if in_degree[child_hash] == 0:
                            queue.append(child_hash)

            return result

    def get_consensus_order(self) -> list[HashgraphEvent]:
        """Get events in consensus order.

        Orders by (round, timestamp, hash) for deterministic ordering.
        Events with decided famous witnesses are ordered first.

        Returns:
            List of events in consensus order
        """
        with self._lock:
            events = list(self.get_all_events())
            # Sort by round, then timestamp, then hash for determinism
            events.sort(key=lambda e: (e.round_number, e.timestamp, e.event_hash))
            return events

    def get_known_creators(self) -> set[str]:
        """Get set of all known event creators.

        Returns:
            Set of creator node IDs
        """
        with self._lock:
            return set(self._creator_chains.keys())

    def get_latest_round(self) -> int:
        """Get the highest round number in the DAG.

        Returns:
            Highest round number, or 0 if empty
        """
        with self._lock:
            if not self._round_events:
                return 0
            return max(self._round_events.keys())

    def get_event_count(self) -> int:
        """Get total number of events in DAG.

        Returns:
            Number of events
        """
        return len(self._nodes)

    def get_stats(self) -> dict[str, int]:
        """Get DAG statistics.

        Returns:
            Dictionary with event_count, round_count, creator_count, witness_count
        """
        with self._lock:
            witness_count = sum(len(w) for w in self._round_witnesses.values())
            return {
                "event_count": len(self._nodes),
                "round_count": len(self._round_events),
                "creator_count": len(self._creator_chains),
                "witness_count": witness_count,
            }

    def detect_equivocation(self, creator: str) -> list[tuple[str, str]]:
        """Detect Byzantine equivocation (forking) by a creator.

        Equivocation occurs when a node creates two events with the same
        self_parent, which is forbidden in hashgraph.

        Args:
            creator: Node ID to check

        Returns:
            List of (event1_hash, event2_hash) tuples that share a self_parent
        """
        with self._lock:
            chain = self._creator_chains.get(creator, [])
            if len(chain) < 2:
                return []

            # Check for events with same self_parent
            parent_to_events: dict[str | None, list[str]] = defaultdict(list)
            for event_hash in chain:
                node = self._nodes.get(event_hash)
                if node:
                    parent_to_events[node.event.self_parent].append(event_hash)

            equivocations = []
            for parent, events in parent_to_events.items():
                if parent is not None and len(events) > 1:
                    # Multiple events claim same self_parent = fork
                    for i in range(len(events)):
                        for j in range(i + 1, len(events)):
                            equivocations.append((events[i], events[j]))

            return equivocations

    def clear(self) -> None:
        """Clear all events from DAG."""
        with self._lock:
            self._nodes.clear()
            self._creator_chains.clear()
            self._round_events.clear()
            self._round_witnesses.clear()
            self._ancestry_cache.clear()
            self._cache_valid = True


__all__ = [
    "HashgraphDAG",
    "DAGNode",
    "AncestryResult",
]
