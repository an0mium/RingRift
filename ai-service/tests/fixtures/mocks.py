"""Reusable Test Mocks (Phase 5.1 - January 2026).

Provides standardized mock classes for testing coordination components.
These reduce test creation time and ensure consistent behavior across tests.

Usage:
    from tests.fixtures.mocks import (
        MockEventRouter,
        MockDaemonManager,
        MockP2PCluster,
        MockGameEngine,
    )

    # Create mock event router
    router = MockEventRouter()
    await router.emit("MY_EVENT", {"data": "value"})
    assert router.emitted[-1]["type"] == "MY_EVENT"

    # Create mock daemon manager
    dm = MockDaemonManager()
    await dm.start(DaemonType.AUTO_SYNC)
    assert "auto_sync" in dm.started

January 2026: Created as part of long-term stability improvements.
Expected impact: -50% test creation time, +25% test coverage.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock

__all__ = [
    "MockEventRouter",
    "MockDaemonManager",
    "MockP2PCluster",
    "MockGameEngine",
    "MockNodeInfo",
    "MockPeer",
    "MockHealthChecker",
    "create_mock_event_payload",
    "create_mock_training_state",
]


# =============================================================================
# Mock Event Router
# =============================================================================

@dataclass
class MockEventRouter:
    """In-memory event router for testing.

    Captures emitted events and allows subscription for handlers.
    Thread-safe for async tests.

    Attributes:
        emitted: List of all emitted events
        subscriptions: Dict mapping event types to handlers
        emit_delay: Optional delay before emitting (for timing tests)
    """

    emitted: list[dict[str, Any]] = field(default_factory=list)
    subscriptions: dict[str, list[Callable]] = field(default_factory=dict)
    emit_delay: float = 0.0
    _emit_count: int = 0

    async def emit(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event asynchronously.

        Args:
            event_type: The event type name
            payload: Event payload dict
        """
        if self.emit_delay > 0:
            await asyncio.sleep(self.emit_delay)

        event = {
            "type": event_type,
            "payload": payload or {},
            "timestamp": time.time(),
            "sequence": self._emit_count,
        }
        self.emitted.append(event)
        self._emit_count += 1

        # Notify subscribers
        for handler in self.subscriptions.get(event_type, []):
            try:
                result = handler(payload or {})
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Ignore handler errors in mock

    def emit_sync(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event synchronously (for sync tests)."""
        event = {
            "type": event_type,
            "payload": payload or {},
            "timestamp": time.time(),
            "sequence": self._emit_count,
        }
        self.emitted.append(event)
        self._emit_count += 1

    def subscribe(
        self,
        event_type: str,
        handler: Callable,
    ) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: The event type to subscribe to
            handler: The handler function (sync or async)
        """
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """Unsubscribe a handler from an event type."""
        if event_type in self.subscriptions:
            try:
                self.subscriptions[event_type].remove(handler)
                return True
            except ValueError:
                pass
        return False

    def get_events_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Get all emitted events of a specific type."""
        return [e for e in self.emitted if e["type"] == event_type]

    def clear(self) -> None:
        """Clear all emitted events and subscriptions."""
        self.emitted.clear()
        self.subscriptions.clear()
        self._emit_count = 0

    def assert_event_emitted(
        self,
        event_type: str,
        payload_contains: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assert that an event was emitted.

        Args:
            event_type: Expected event type
            payload_contains: Optional dict of expected payload keys/values

        Returns:
            The matched event

        Raises:
            AssertionError: If no matching event found
        """
        matching = self.get_events_by_type(event_type)
        assert matching, f"Expected event {event_type} was not emitted"

        if payload_contains:
            for event in matching:
                payload = event.get("payload", {})
                if all(payload.get(k) == v for k, v in payload_contains.items()):
                    return event
            raise AssertionError(
                f"Event {event_type} found but payload {payload_contains} not matched"
            )

        return matching[-1]


# =============================================================================
# Mock Daemon Manager
# =============================================================================

@dataclass
class MockDaemonManager:
    """Mock daemon lifecycle manager for testing.

    Tracks started/stopped daemons and provides health status.

    Attributes:
        started: Set of started daemon type names
        stopped: Set of stopped daemon type names
        health: Dict mapping daemon type to health status
        running: Set of currently running daemons
    """

    started: set[str] = field(default_factory=set)
    stopped: set[str] = field(default_factory=set)
    health: dict[str, str] = field(default_factory=dict)
    running: set[str] = field(default_factory=set)
    _start_times: dict[str, float] = field(default_factory=dict)

    async def start(self, daemon_type: Any) -> bool:
        """Start a daemon.

        Args:
            daemon_type: DaemonType enum or string

        Returns:
            True if started successfully
        """
        name = daemon_type.value if hasattr(daemon_type, "value") else str(daemon_type)
        self.started.add(name)
        self.running.add(name)
        self.health[name] = "healthy"
        self._start_times[name] = time.time()
        return True

    async def stop(self, daemon_type: Any) -> bool:
        """Stop a daemon.

        Args:
            daemon_type: DaemonType enum or string

        Returns:
            True if stopped successfully
        """
        name = daemon_type.value if hasattr(daemon_type, "value") else str(daemon_type)
        self.stopped.add(name)
        self.running.discard(name)
        self.health[name] = "stopped"
        return True

    async def restart(self, daemon_type: Any) -> bool:
        """Restart a daemon."""
        await self.stop(daemon_type)
        return await self.start(daemon_type)

    def is_running(self, daemon_type: Any) -> bool:
        """Check if a daemon is running."""
        name = daemon_type.value if hasattr(daemon_type, "value") else str(daemon_type)
        return name in self.running

    def get_health(self, daemon_type: Any) -> str:
        """Get health status of a daemon."""
        name = daemon_type.value if hasattr(daemon_type, "value") else str(daemon_type)
        return self.health.get(name, "unknown")

    def set_health(self, daemon_type: Any, status: str) -> None:
        """Set health status of a daemon (for test setup)."""
        name = daemon_type.value if hasattr(daemon_type, "value") else str(daemon_type)
        self.health[name] = status

    def get_all_daemon_health(self) -> dict[str, str]:
        """Get health status of all daemons."""
        return dict(self.health)

    def clear(self) -> None:
        """Reset all state."""
        self.started.clear()
        self.stopped.clear()
        self.health.clear()
        self.running.clear()
        self._start_times.clear()


# =============================================================================
# Mock P2P Cluster
# =============================================================================

@dataclass
class MockNodeInfo:
    """Mock node info for P2P testing."""

    node_id: str
    tailscale_ip: str = "100.0.0.1"
    ssh_host: str = "localhost"
    ssh_port: int = 22
    role: str = "gpu_selfplay"
    gpu: str | None = "RTX 4090"
    is_alive_flag: bool = True
    first_seen: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    error_count: int = 0
    request_count: int = 100
    peer_count: int = 10

    def is_alive(self) -> bool:
        """Check if node is alive."""
        return self.is_alive_flag


@dataclass
class MockPeer:
    """Simplified mock peer for quick test setup."""

    node_id: str
    is_alive_flag: bool = True
    leader_id: str | None = None

    def is_alive(self) -> bool:
        return self.is_alive_flag


@dataclass
class MockP2PCluster:
    """Mock P2P cluster for distributed testing.

    Simulates a P2P cluster with configurable nodes, leader election,
    and job distribution.

    Attributes:
        nodes: Dict of node_id -> MockNodeInfo
        leader_id: Current leader node ID
        jobs: List of submitted jobs
        voter_node_ids: List of voter node IDs
    """

    nodes: dict[str, MockNodeInfo] = field(default_factory=dict)
    leader_id: str | None = None
    jobs: list[dict[str, Any]] = field(default_factory=list)
    voter_node_ids: list[str] = field(default_factory=list)
    _job_counter: int = 0

    def add_node(
        self,
        node_id: str,
        role: str = "gpu_selfplay",
        gpu: str = "RTX 4090",
        is_voter: bool = False,
    ) -> MockNodeInfo:
        """Add a node to the cluster.

        Args:
            node_id: Unique node identifier
            role: Node role (gpu_selfplay, gpu_training, coordinator)
            gpu: GPU type or None for CPU-only
            is_voter: Whether this node is a voter

        Returns:
            The created MockNodeInfo
        """
        node = MockNodeInfo(
            node_id=node_id,
            role=role,
            gpu=gpu,
            tailscale_ip=f"100.0.0.{len(self.nodes) + 1}",
        )
        self.nodes[node_id] = node
        if is_voter:
            self.voter_node_ids.append(node_id)
        return node

    def set_leader(self, node_id: str) -> None:
        """Set the cluster leader."""
        self.leader_id = node_id

    def kill_node(self, node_id: str) -> None:
        """Simulate a node going offline."""
        if node_id in self.nodes:
            self.nodes[node_id].is_alive_flag = False

    def revive_node(self, node_id: str) -> None:
        """Simulate a node coming back online."""
        if node_id in self.nodes:
            self.nodes[node_id].is_alive_flag = True

    def get_alive_nodes(self) -> list[MockNodeInfo]:
        """Get all alive nodes."""
        return [n for n in self.nodes.values() if n.is_alive()]

    def get_alive_voters(self) -> list[MockNodeInfo]:
        """Get all alive voter nodes."""
        return [
            self.nodes[nid]
            for nid in self.voter_node_ids
            if nid in self.nodes and self.nodes[nid].is_alive()
        ]

    def submit_job(
        self,
        job_type: str,
        config_key: str,
        node_id: str | None = None,
    ) -> str:
        """Submit a job to the cluster.

        Returns:
            Job ID
        """
        job_id = f"job-{self._job_counter}"
        self._job_counter += 1

        self.jobs.append({
            "job_id": job_id,
            "type": job_type,
            "config_key": config_key,
            "node_id": node_id or self.leader_id,
            "status": "pending",
            "submitted_at": time.time(),
        })
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID."""
        for job in self.jobs:
            if job["job_id"] == job_id:
                return job
        return None

    def complete_job(self, job_id: str, success: bool = True) -> None:
        """Mark a job as complete."""
        job = self.get_job(job_id)
        if job:
            job["status"] = "completed" if success else "failed"
            job["completed_at"] = time.time()

    def clear(self) -> None:
        """Reset cluster state."""
        self.nodes.clear()
        self.leader_id = None
        self.jobs.clear()
        self.voter_node_ids.clear()
        self._job_counter = 0


# =============================================================================
# Mock Game Engine
# =============================================================================

@dataclass
class MockGameEngine:
    """Mock game engine for game state testing.

    Simulates a RingRift game engine with configurable board state.

    Attributes:
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players
        moves: List of applied moves
        current_player: Current player number
        game_over: Whether game is complete
    """

    board_type: str = "hex8"
    num_players: int = 2
    moves: list[dict[str, Any]] = field(default_factory=list)
    current_player: int = 1
    game_over: bool = False
    winner: int | None = None
    _move_counter: int = 0

    def apply_move(self, move: dict[str, Any]) -> dict[str, Any]:
        """Apply a move to the game state.

        Args:
            move: Move dict with type, from, to, etc.

        Returns:
            Updated game state
        """
        self.moves.append(move)
        self._move_counter += 1

        # Advance player
        self.current_player = (self.current_player % self.num_players) + 1

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Get current game state."""
        return {
            "board_type": self.board_type,
            "num_players": self.num_players,
            "current_player": self.current_player,
            "move_count": len(self.moves),
            "game_over": self.game_over,
            "winner": self.winner,
        }

    def get_legal_moves(self, player: int | None = None) -> list[dict[str, Any]]:
        """Get legal moves for a player.

        Returns a standard set of mock moves for testing.
        """
        if self.game_over:
            return []

        player = player or self.current_player
        return [
            {"type": "place_ring", "to": (0, 0), "player": player},
            {"type": "place_ring", "to": (1, 1), "player": player},
            {"type": "move_stack", "from": (0, 0), "to": (0, 1), "player": player},
        ]

    def set_winner(self, player: int) -> None:
        """Set the game winner."""
        self.game_over = True
        self.winner = player

    def reset(self) -> None:
        """Reset game to initial state."""
        self.moves.clear()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self._move_counter = 0


# =============================================================================
# Mock Health Checker
# =============================================================================

@dataclass
class MockHealthChecker:
    """Mock health checker for testing health-related code.

    Attributes:
        health_status: Dict of component -> health status
        is_healthy_override: Force all health checks to return this value
    """

    health_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    is_healthy_override: bool | None = None

    def set_health(
        self,
        component: str,
        is_healthy: bool,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Set health status for a component."""
        self.health_status[component] = {
            "is_healthy": is_healthy,
            "message": message,
            "details": details or {},
            "timestamp": time.time(),
        }

    def get_health(self, component: str) -> dict[str, Any]:
        """Get health status for a component."""
        if self.is_healthy_override is not None:
            return {
                "is_healthy": self.is_healthy_override,
                "message": "Override active",
                "details": {},
            }
        return self.health_status.get(component, {
            "is_healthy": True,
            "message": "Default healthy",
            "details": {},
        })

    def is_healthy(self, component: str) -> bool:
        """Check if a component is healthy."""
        if self.is_healthy_override is not None:
            return self.is_healthy_override
        status = self.health_status.get(component, {})
        return status.get("is_healthy", True)

    def get_system_health_score(self) -> float:
        """Get overall system health score (0.0 - 1.0)."""
        if self.is_healthy_override is not None:
            return 1.0 if self.is_healthy_override else 0.0

        if not self.health_status:
            return 1.0

        healthy_count = sum(
            1 for s in self.health_status.values()
            if s.get("is_healthy", True)
        )
        return healthy_count / len(self.health_status)


# =============================================================================
# Factory Functions
# =============================================================================

def create_mock_event_payload(
    event_type: str,
    config_key: str = "hex8_2p",
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a mock event payload with common fields.

    Args:
        event_type: The event type
        config_key: Configuration key (e.g., "hex8_2p")
        **kwargs: Additional payload fields

    Returns:
        Mock event payload dict
    """
    # Parse config_key
    parts = config_key.replace("_", " ").split()
    board_type = parts[0] if parts else "hex8"
    num_players = int(parts[1].rstrip("p")) if len(parts) > 1 else 2

    payload = {
        "config_key": config_key,
        "board_type": board_type,
        "num_players": num_players,
        "timestamp": time.time(),
        **kwargs,
    }
    return payload


def create_mock_training_state(
    config_key: str = "hex8_2p",
    training_in_progress: bool = False,
    npz_sample_count: int = 5000,
    last_training_time: float = 0.0,
    last_npz_update: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a mock training state for training trigger tests.

    Args:
        config_key: Configuration key
        training_in_progress: Whether training is in progress
        npz_sample_count: Number of samples in NPZ
        last_training_time: Last training timestamp
        last_npz_update: Last NPZ update timestamp
        **kwargs: Additional state fields

    Returns:
        Mock training state dict
    """
    # Parse config_key
    parts = config_key.replace("_", " ").split()
    board_type = parts[0] if parts else "hex8"
    num_players = int(parts[1].rstrip("p")) if len(parts) > 1 else 2

    now = time.time()
    return {
        "config_key": config_key,
        "board_type": board_type,
        "num_players": num_players,
        "training_in_progress": training_in_progress,
        "npz_sample_count": npz_sample_count,
        "last_training_time": last_training_time or now - 3600,
        "last_npz_update": last_npz_update or now - 1800,
        "training_intensity": "normal",
        "elo_velocity_trend": "stable",
        **kwargs,
    }
