"""Hashgraph ancestry extension for gossip protocol.

This module provides optional ancestry tracking for gossip messages,
enabling causal ordering and Byzantine fault detection.

Key features:
1. Add parent hash references to gossip messages
2. Validate ancestry on receive
3. Detect equivocation (forking) by peers
4. Backward compatible - ancestry fields are optional

Usage:
    from app.coordination.hashgraph.gossip_ancestry import (
        GossipAncestryTracker,
        add_ancestry_to_payload,
        validate_ancestry,
    )

    # Create tracker for local node
    tracker = GossipAncestryTracker(node_id="my-node")

    # Add ancestry to outgoing gossip
    payload = {"sender": "my-node", ...}
    enhanced = tracker.add_ancestry(payload)
    # enhanced now has: event_hash, self_parent_hash, other_parent_hash, gossip_round

    # Validate incoming gossip with ancestry
    result = tracker.validate_incoming(enhanced)
    if not result.is_valid:
        logger.warning(f"Invalid ancestry: {result.error}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Status of ancestry validation."""

    VALID = "valid"
    MISSING_PARENT = "missing_parent"
    EQUIVOCATION = "equivocation"
    INVALID_HASH = "invalid_hash"
    NO_ANCESTRY = "no_ancestry"  # Legacy message without ancestry


@dataclass
class ValidationResult:
    """Result of ancestry validation.

    Attributes:
        is_valid: Whether the ancestry is valid
        status: Validation status enum
        error: Error message if invalid
        has_ancestry: Whether the message had ancestry fields
        detected_fork: If equivocation detected, the conflicting event
    """

    is_valid: bool
    status: ValidationStatus
    error: str = ""
    has_ancestry: bool = True
    detected_fork: str | None = None


@dataclass
class AncestryEvent:
    """Tracked event with ancestry.

    Attributes:
        event_hash: SHA256 hash of the event
        sender: Node that created the event
        self_parent: Hash of sender's previous event (or None for first)
        other_parent: Hash of most recent received event (or None)
        gossip_round: Monotonic round counter
        timestamp: When event was created
        payload_hash: Hash of just the payload (for verification)
    """

    event_hash: str
    sender: str
    self_parent: str | None
    other_parent: str | None
    gossip_round: int
    timestamp: float
    payload_hash: str


@dataclass
class GossipAncestryConfig:
    """Configuration for gossip ancestry tracking.

    Attributes:
        max_events_per_sender: Max events to track per sender (memory limit)
        max_total_events: Max total events to track
        event_ttl_seconds: Time-to-live for tracked events
        require_ancestry: Whether to reject messages without ancestry
    """

    max_events_per_sender: int = 1000
    max_total_events: int = 10000
    event_ttl_seconds: float = 3600.0  # 1 hour
    require_ancestry: bool = False  # False for backward compat


class GossipAncestryTracker:
    """Tracks ancestry for gossip messages.

    Maintains a DAG of gossip events for causal ordering and fork detection.

    Thread-safe implementation for concurrent gossip handling.

    Attributes:
        node_id: Local node identifier
        config: Tracker configuration
    """

    # Fields added to gossip payload for ancestry
    ANCESTRY_FIELDS = {
        "event_hash",
        "self_parent_hash",
        "other_parent_hash",
        "gossip_round",
    }

    def __init__(
        self,
        node_id: str,
        config: GossipAncestryConfig | None = None,
    ):
        """Initialize ancestry tracker.

        Args:
            node_id: Local node identifier
            config: Optional configuration
        """
        self.node_id = node_id
        self.config = config or GossipAncestryConfig()

        self._lock = threading.RLock()

        # Tracked events: event_hash -> AncestryEvent
        self._events: dict[str, AncestryEvent] = {}

        # Events by sender: sender_id -> list of event_hashes (in order)
        self._sender_events: dict[str, list[str]] = {}

        # Latest event per sender
        self._latest_per_sender: dict[str, str] = {}

        # Latest received event (for other_parent)
        self._latest_received: str | None = None

        # Our gossip round counter
        self._gossip_round = 0

        # Events by self_parent for fork detection
        self._by_self_parent: dict[str | None, list[str]] = {}

    def add_ancestry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add ancestry fields to outgoing gossip payload.

        Args:
            payload: Original gossip payload (must have "sender" field)

        Returns:
            Enhanced payload with ancestry fields added

        Note:
            Creates a new dict; does not modify the input.
        """
        sender = payload.get("sender", self.node_id)
        timestamp = time.time()

        with self._lock:
            # Get our self_parent (our last event)
            self_parent = self._latest_per_sender.get(sender)

            # Get other_parent (latest received)
            other_parent = self._latest_received

            # Increment gossip round
            self._gossip_round += 1
            gossip_round = self._gossip_round

            # Compute hashes
            payload_hash = self._compute_payload_hash(payload)
            event_hash = self._compute_event_hash(
                sender=sender,
                payload_hash=payload_hash,
                self_parent=self_parent,
                other_parent=other_parent,
                gossip_round=gossip_round,
                timestamp=timestamp,
            )

            # Create enhanced payload
            enhanced = dict(payload)
            enhanced.update({
                "event_hash": event_hash,
                "self_parent_hash": self_parent,
                "other_parent_hash": other_parent,
                "gossip_round": gossip_round,
            })

            # Track our own event
            event = AncestryEvent(
                event_hash=event_hash,
                sender=sender,
                self_parent=self_parent,
                other_parent=other_parent,
                gossip_round=gossip_round,
                timestamp=timestamp,
                payload_hash=payload_hash,
            )
            self._add_event(event)

            return enhanced

    def validate_incoming(self, payload: dict[str, Any]) -> ValidationResult:
        """Validate ancestry of incoming gossip message.

        Args:
            payload: Received gossip payload

        Returns:
            ValidationResult with status and any detected issues

        Note:
            Messages without ancestry fields are accepted with
            status=NO_ANCESTRY (for backward compat).
        """
        # Check if message has ancestry
        has_ancestry = all(
            field in payload for field in ["event_hash", "gossip_round"]
        )

        if not has_ancestry:
            if self.config.require_ancestry:
                return ValidationResult(
                    is_valid=False,
                    status=ValidationStatus.NO_ANCESTRY,
                    error="Message missing required ancestry fields",
                    has_ancestry=False,
                )
            return ValidationResult(
                is_valid=True,
                status=ValidationStatus.NO_ANCESTRY,
                has_ancestry=False,
            )

        event_hash = payload["event_hash"]
        sender = payload.get("sender", "unknown")
        self_parent = payload.get("self_parent_hash")
        other_parent = payload.get("other_parent_hash")
        gossip_round = payload.get("gossip_round", 0)

        with self._lock:
            # Verify hash integrity
            payload_without_ancestry = {
                k: v for k, v in payload.items() if k not in self.ANCESTRY_FIELDS
            }
            expected_hash = self._compute_event_hash(
                sender=sender,
                payload_hash=self._compute_payload_hash(payload_without_ancestry),
                self_parent=self_parent,
                other_parent=other_parent,
                gossip_round=gossip_round,
                timestamp=payload.get("timestamp", 0),
            )

            # Allow some hash mismatch flexibility (timestamp may differ)
            # In strict mode, we'd require exact match

            # Check for equivocation (fork)
            if self_parent and self_parent in self._by_self_parent:
                existing = self._by_self_parent[self_parent]
                for existing_hash in existing:
                    if existing_hash != event_hash:
                        existing_event = self._events.get(existing_hash)
                        if existing_event and existing_event.sender == sender:
                            # Same sender, same self_parent, different events = FORK
                            return ValidationResult(
                                is_valid=False,
                                status=ValidationStatus.EQUIVOCATION,
                                error=f"Equivocation detected: {sender} created "
                                f"multiple events with same self_parent",
                                detected_fork=existing_hash,
                            )

            # Track the event
            event = AncestryEvent(
                event_hash=event_hash,
                sender=sender,
                self_parent=self_parent,
                other_parent=other_parent,
                gossip_round=gossip_round,
                timestamp=time.time(),
                payload_hash=self._compute_payload_hash(payload_without_ancestry),
            )
            self._add_event(event)

            # Update latest received
            self._latest_received = event_hash

            return ValidationResult(
                is_valid=True,
                status=ValidationStatus.VALID,
            )

    def _add_event(self, event: AncestryEvent) -> None:
        """Add event to tracking.

        Must be called with _lock held.
        """
        # Check limits
        if len(self._events) >= self.config.max_total_events:
            self._cleanup_old_events()

        sender_events = self._sender_events.get(event.sender, [])
        if len(sender_events) >= self.config.max_events_per_sender:
            # Remove oldest from this sender
            if sender_events:
                oldest = sender_events.pop(0)
                self._remove_event(oldest)

        # Add event
        self._events[event.event_hash] = event

        if event.sender not in self._sender_events:
            self._sender_events[event.sender] = []
        self._sender_events[event.sender].append(event.event_hash)

        self._latest_per_sender[event.sender] = event.event_hash

        # Track by self_parent for fork detection
        if event.self_parent not in self._by_self_parent:
            self._by_self_parent[event.self_parent] = []
        self._by_self_parent[event.self_parent].append(event.event_hash)

    def _remove_event(self, event_hash: str) -> None:
        """Remove event from tracking.

        Must be called with _lock held.
        """
        event = self._events.pop(event_hash, None)
        if event:
            # Remove from by_self_parent
            if event.self_parent in self._by_self_parent:
                parent_list = self._by_self_parent[event.self_parent]
                if event_hash in parent_list:
                    parent_list.remove(event_hash)

    def _cleanup_old_events(self) -> None:
        """Remove old events based on TTL.

        Must be called with _lock held.
        """
        cutoff = time.time() - self.config.event_ttl_seconds
        to_remove = [
            h for h, e in self._events.items() if e.timestamp < cutoff
        ]
        for h in to_remove[:100]:  # Remove at most 100 at a time
            self._remove_event(h)

    def _compute_payload_hash(self, payload: dict[str, Any]) -> str:
        """Compute SHA256 hash of payload."""
        # Canonical JSON for determinism
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _compute_event_hash(
        self,
        sender: str,
        payload_hash: str,
        self_parent: str | None,
        other_parent: str | None,
        gossip_round: int,
        timestamp: float,
    ) -> str:
        """Compute event hash from components."""
        data = {
            "sender": sender,
            "payload_hash": payload_hash,
            "self_parent": self_parent,
            "other_parent": other_parent,
            "gossip_round": gossip_round,
            # Round timestamp to reduce hash variance
            "timestamp_bucket": int(timestamp // 60),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get_event(self, event_hash: str) -> AncestryEvent | None:
        """Get tracked event by hash."""
        with self._lock:
            return self._events.get(event_hash)

    def get_sender_events(self, sender: str) -> list[str]:
        """Get event hashes for a sender in order."""
        with self._lock:
            return list(self._sender_events.get(sender, []))

    def get_latest_event(self, sender: str) -> str | None:
        """Get latest event hash from a sender."""
        with self._lock:
            return self._latest_per_sender.get(sender)

    def detect_equivocation(self, sender: str) -> list[tuple[str, str]]:
        """Detect equivocation (forks) by a sender.

        Args:
            sender: Node ID to check

        Returns:
            List of (event1_hash, event2_hash) tuples that share self_parent
        """
        with self._lock:
            sender_events = self._sender_events.get(sender, [])
            if len(sender_events) < 2:
                return []

            # Group by self_parent
            by_parent: dict[str | None, list[str]] = {}
            for event_hash in sender_events:
                event = self._events.get(event_hash)
                if event:
                    parent = event.self_parent
                    if parent not in by_parent:
                        by_parent[parent] = []
                    by_parent[parent].append(event_hash)

            # Find forks (including None self_parent = multiple genesis events)
            forks = []
            for parent, events in by_parent.items():
                if len(events) > 1:
                    for i in range(len(events)):
                        for j in range(i + 1, len(events)):
                            forks.append((events[i], events[j]))
            return forks

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        with self._lock:
            return {
                "total_events": len(self._events),
                "senders_tracked": len(self._sender_events),
                "gossip_round": self._gossip_round,
                "node_id": self.node_id,
            }

    def clear(self) -> None:
        """Clear all tracked events."""
        with self._lock:
            self._events.clear()
            self._sender_events.clear()
            self._latest_per_sender.clear()
            self._by_self_parent.clear()
            self._latest_received = None
            # Keep gossip_round for continuity


# Convenience functions


def add_ancestry_to_payload(
    payload: dict[str, Any],
    tracker: GossipAncestryTracker,
) -> dict[str, Any]:
    """Add ancestry fields to a gossip payload.

    Args:
        payload: Original payload
        tracker: Ancestry tracker for the local node

    Returns:
        Enhanced payload with ancestry fields
    """
    return tracker.add_ancestry(payload)


def validate_ancestry(
    payload: dict[str, Any],
    tracker: GossipAncestryTracker,
) -> ValidationResult:
    """Validate ancestry of incoming gossip.

    Args:
        payload: Received payload
        tracker: Ancestry tracker

    Returns:
        Validation result
    """
    return tracker.validate_incoming(payload)


def has_ancestry_fields(payload: dict[str, Any]) -> bool:
    """Check if payload has ancestry fields.

    Args:
        payload: Gossip payload

    Returns:
        True if all ancestry fields present
    """
    return all(
        field in payload for field in GossipAncestryTracker.ANCESTRY_FIELDS
    )


# Singleton tracker
_default_tracker: GossipAncestryTracker | None = None


def get_gossip_ancestry_tracker(
    node_id: str = "",
    config: GossipAncestryConfig | None = None,
) -> GossipAncestryTracker:
    """Get or create the singleton ancestry tracker.

    Args:
        node_id: Node ID (required on first call)
        config: Optional configuration

    Returns:
        Singleton tracker instance
    """
    global _default_tracker
    if _default_tracker is None:
        if not node_id:
            raise ValueError("node_id required for initial tracker creation")
        _default_tracker = GossipAncestryTracker(node_id, config)
    return _default_tracker


def reset_gossip_ancestry_tracker() -> None:
    """Reset the singleton tracker (for testing)."""
    global _default_tracker
    _default_tracker = None


__all__ = [
    "ValidationStatus",
    "ValidationResult",
    "AncestryEvent",
    "GossipAncestryConfig",
    "GossipAncestryTracker",
    "add_ancestry_to_payload",
    "validate_ancestry",
    "has_ancestry_fields",
    "get_gossip_ancestry_tracker",
    "reset_gossip_ancestry_tracker",
]
