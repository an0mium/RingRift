"""P2P Orchestrator Type Definitions.

This module contains enums used throughout the P2P orchestrator.
Extracted from p2p_orchestrator.py for better modularity.
"""

from enum import Enum


class NodeRole(str, Enum):
    """Role a node plays in the cluster.

    Jan 1, 2026: Added PROVISIONAL_LEADER for probabilistic fallback leadership.
    When normal elections fail repeatedly (e.g., voter quorum unavailable),
    nodes can claim provisional leadership with increasing probability.
    Provisional leaders can dispatch work but must be confirmed by quorum
    acknowledgment or node_id tiebreaker if contested.
    """
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    PROVISIONAL_LEADER = "provisional_leader"  # Jan 1, 2026: Fallback leadership before quorum confirmation


class NodeHealthState(str, Enum):
    """Health state of a node based on heartbeat timing.

    Dec 2025: Added SUSPECT state to reduce false-positive failures.
    Nodes transition: ALIVE -> SUSPECT -> DEAD

    With 15s heartbeats and 30s suspect timeout:
    - ALIVE: heartbeat within last 30s (2 heartbeats)
    - SUSPECT: heartbeat 30-60s ago (grace period)
    - DEAD: no heartbeat for 60s+ (4 missed heartbeats)
    """
    ALIVE = "alive"
    SUSPECT = "suspect"  # Grace period - node may be experiencing transient issues
    DEAD = "dead"


class JobType(str, Enum):
    """Types of jobs nodes can run."""
    SELFPLAY = "selfplay"
    GPU_SELFPLAY = "gpu_selfplay"  # GPU-accelerated parallel selfplay (pure GPU, high parity 2025-12)
    HYBRID_SELFPLAY = "hybrid_selfplay"  # Hybrid CPU/GPU selfplay (100% rule fidelity, GPU-accelerated eval)
    CPU_SELFPLAY = "cpu_selfplay"  # Pure CPU selfplay to utilize excess CPU on high-CPU/low-VRAM nodes
    GUMBEL_SELFPLAY = "gumbel_selfplay"  # Gumbel MCTS with NN policy for high-quality training data
    TRAINING = "training"
    CMAES = "cmaes"
    # Distributed job types
    DISTRIBUTED_CMAES_COORDINATOR = "distributed_cmaes_coordinator"
    DISTRIBUTED_CMAES_WORKER = "distributed_cmaes_worker"
    DISTRIBUTED_TOURNAMENT_COORDINATOR = "distributed_tournament_coordinator"
    DISTRIBUTED_TOURNAMENT_WORKER = "distributed_tournament_worker"
    IMPROVEMENT_LOOP = "improvement_loop"
    # CPU-intensive data processing jobs
    DATA_EXPORT = "data_export"  # NPZ export (CPU-intensive, route to high-CPU nodes)
    DATA_AGGREGATION = "data_aggregation"  # JSONL aggregation (CPU-intensive)


class JobLifecycleState(str, Enum):
    """Unified job lifecycle states for all job management loops.

    Jan 2, 2026 - Sprint 9: Consolidates job states from:
    - JobReaperLoop: claimed, started, running
    - JobReassignmentLoop: orphaned, stale
    - SpawnVerificationLoop: verified, failed, pending
    - General: completed, cancelled

    State transitions:
        PENDING -> CLAIMED -> STARTING -> RUNNING -> COMPLETED
                              |           |
                              v           v
                            STALE     ORPHANED / STUCK
                              |           |
                              v           v
                            FAILED      FAILED
    """
    # Initial states
    PENDING = "pending"      # Queued in work queue, not yet claimed
    CLAIMED = "claimed"      # Claimed by a node, not yet started

    # Active states
    STARTING = "starting"    # Process is being spawned
    RUNNING = "running"      # Actively executing

    # Verification state (from SpawnVerificationLoop)
    VERIFIED = "verified"    # Spawn confirmed successful

    # Problem states
    STALE = "stale"          # Claimed but not started within timeout
    STUCK = "stuck"          # Running too long (exceeds expected duration)
    ORPHANED = "orphaned"    # Lost contact with executing node

    # Terminal states
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Execution failed or spawn failed
    CANCELLED = "cancelled"  # Explicitly cancelled

    def is_terminal(self) -> bool:
        """Check if this is a terminal (final) state."""
        return self in (
            JobLifecycleState.COMPLETED,
            JobLifecycleState.FAILED,
            JobLifecycleState.CANCELLED,
        )

    def is_active(self) -> bool:
        """Check if the job is actively executing or about to execute."""
        return self in (
            JobLifecycleState.CLAIMED,
            JobLifecycleState.STARTING,
            JobLifecycleState.RUNNING,
            JobLifecycleState.VERIFIED,
        )

    def is_problem(self) -> bool:
        """Check if the job is in a problem state that needs attention."""
        return self in (
            JobLifecycleState.STALE,
            JobLifecycleState.STUCK,
            JobLifecycleState.ORPHANED,
        )
