# ADR-013: SWIM/Raft Protocol Evaluation

**Status**: Proposed
**Date**: 2025-12-28
**Context**: RingRift AI Training Infrastructure P2P Cluster

## Summary

This ADR evaluates replacing the current HTTP polling + Bully election with SWIM-based membership and Raft-based consensus for improved cluster reliability.

## Current State

### Membership Detection (HTTP Polling)

- **Protocol**: HTTP `/status` endpoint polling every 15s
- **Failure Detection**: 60-90s (4-6 missed polls)
- **Bandwidth**: O(n²) - each node polls all others
- **Pros**: Simple, no external dependencies
- **Cons**: Slow failure detection, high bandwidth at scale

### Leader Election (Bully Algorithm)

- **Protocol**: Highest node ID wins election
- **Failover Time**: 60-90s (detect + elect)
- **State Replication**: None (leader-only state)
- **Pros**: Simple, deterministic
- **Cons**: No state replication, slow failover

### Current Production State

- **Nodes**: ~36 configured, ~20-30 alive
- **Leader**: Stable (hetzner-cpu1)
- **Reliability**: ~95% uptime (5% from slow failure detection)
- **Issues**: Occasional split-brain during network partitions

## SWIM Protocol (Membership)

### How SWIM Works

1. **Ping**: Random node A pings random node B
2. **Indirect Ping**: If B fails, A asks k random nodes to ping B
3. **Suspicion**: Mark B as "suspected" if indirect pings fail
4. **Confirm**: Mark B as "failed" after suspicion timeout
5. **Gossip**: Propagate membership changes via piggybacked messages

### Benefits

| Metric            | HTTP Polling | SWIM                      |
| ----------------- | ------------ | ------------------------- |
| Failure Detection | 60-90s       | 5-10s                     |
| Bandwidth         | O(n²)        | O(1) per node             |
| Scalability       | ~100 nodes   | 1000+ nodes               |
| False Positives   | Rare         | Very rare (indirect ping) |

### Implementation Options

#### Option A: swim-p2p Library

```python
# swim-p2p >= 1.2.0 (not currently installed)
from swim import SwimMember, SwimCluster

cluster = SwimCluster(
    member_id=node_id,
    bind_port=8771,
    gossip_interval_ms=500,
    ping_timeout_ms=1000,
    indirect_ping_count=3,
    suspicion_timeout_ms=5000,
)

cluster.on_member_join(handle_join)
cluster.on_member_leave(handle_leave)
await cluster.start()
```

#### Option B: Custom SWIM Implementation

- ~500-1000 LOC
- Full control over protocol parameters
- No external dependency

**Recommendation**: Option A (library) for faster deployment, migrate to Option B if customization needed.

## Raft Protocol (Consensus)

### How Raft Works

1. **Leader Election**: Nodes vote for candidates with highest log index
2. **Log Replication**: Leader replicates entries to followers
3. **Commit**: Entry committed when majority acknowledge
4. **State Machine**: Committed entries applied to state machine

### Benefits

| Metric            | Bully    | Raft      |
| ----------------- | -------- | --------- |
| Failover Time     | 60-90s   | <2s       |
| State Replication | None     | Automatic |
| Split-brain       | Possible | Prevented |
| Consistency       | Eventual | Strong    |

### What Would Use Raft

1. **Work Queue**: Replicated across all voters
2. **Job Status**: All nodes see same job state
3. **Model Registry**: Consistent model metadata
4. **Training Locks**: Distributed exclusive locks

### Implementation Options

#### Option A: PySyncObj Library

```python
# pysyncobj >= 0.3.14 (not currently installed)
from pysyncobj import SyncObj, replicated

class RaftState(SyncObj):
    def __init__(self, node_id: str, voters: list[str]):
        super().__init__(
            f"tcp://{node_id}:8772",
            [f"tcp://{v}:8772" for v in voters],
        )
        self._work_queue: list[dict] = []
        self._job_status: dict[str, str] = {}

    @replicated
    def add_work(self, work: dict) -> None:
        self._work_queue.append(work)

    @replicated
    def claim_work(self, job_id: str, node_id: str) -> bool:
        if self._job_status.get(job_id) == "pending":
            self._job_status[job_id] = f"claimed:{node_id}"
            return True
        return False
```

#### Option B: etcd as External Coordinator

- Requires etcd cluster (3+ nodes)
- Well-tested consensus
- Higher operational complexity

**Recommendation**: Option A (PySyncObj) for tight integration with Python codebase.

## Comparison Matrix

| Aspect                 | Current (HTTP+Bully) | SWIM+Raft           |
| ---------------------- | -------------------- | ------------------- |
| Failure Detection      | 60-90s               | 5-10s               |
| Leader Failover        | 60-90s               | <2s                 |
| State Replication      | None                 | Automatic           |
| Split-brain Prevention | Manual               | Automatic           |
| Bandwidth (30 nodes)   | ~900 req/min         | ~60 msg/min         |
| Dependencies           | None                 | swim-p2p, pysyncobj |
| Complexity             | Low                  | Medium              |
| Production Risk        | Low (current)        | Medium (new)        |

## Decision

### Recommended: Hybrid Approach

Keep current protocols as fallback, enable SWIM/Raft via feature flags:

```bash
# Environment variables
export RINGRIFT_SWIM_ENABLED=false   # Default: keep HTTP polling
export RINGRIFT_RAFT_ENABLED=false   # Default: keep Bully election
export RINGRIFT_MEMBERSHIP_MODE=http  # Options: http, swim, hybrid
export RINGRIFT_CONSENSUS_MODE=bully  # Options: bully, raft, hybrid
```

### Migration Plan

#### Phase 1: SWIM Pilot (2 weeks)

1. Install `swim-p2p` dependency
2. Enable SWIM on 5 stable nodes (Hetzner + Nebius)
3. Run parallel with HTTP polling
4. Compare failure detection metrics
5. Validate no false positives

#### Phase 2: SWIM Production (1 week)

1. Enable SWIM cluster-wide
2. Disable HTTP polling (keep as fallback)
3. Monitor for 1 week

#### Phase 3: Raft Pilot (2 weeks)

1. Install `pysyncobj` dependency
2. Enable Raft on 5 voters
3. Replicate work queue only
4. Compare with current implementation

#### Phase 4: Raft Production (2 weeks)

1. Enable Raft for work queue + job status
2. Migrate model registry
3. Disable Bully election

### Rollback Plan

If issues occur:

1. Set `RINGRIFT_MEMBERSHIP_MODE=http`
2. Set `RINGRIFT_CONSENSUS_MODE=bully`
3. Restart P2P orchestrators

## Code Locations

### Current Implementation

- `scripts/p2p_orchestrator.py:8030-8100` - HTTP heartbeat handler
- `scripts/p2p/leader_election.py` - Bully election mixin
- `scripts/p2p/peer_manager.py` - Peer tracking

### SWIM Integration Points (Future)

- `scripts/p2p/membership_mixin.py` - SWIM adapter
- `scripts/p2p/handlers/swim.py` - SWIM status endpoints

### Raft Integration Points (Future)

- `scripts/p2p/consensus_mixin.py` - Raft adapter
- `scripts/p2p/handlers/raft.py` - Raft status endpoints
- `app/p2p/raft_state.py` - Replicated state machine

## Metrics to Track

| Metric                 | Target       | Current      |
| ---------------------- | ------------ | ------------ |
| Failure detection time | <10s         | 60-90s       |
| Leader failover time   | <5s          | 60-90s       |
| Split-brain incidents  | 0/month      | ~2/month     |
| False positive rate    | <0.1%        | ~0.5%        |
| Message overhead       | <100 msg/min | ~900 req/min |

## Decision Drivers

1. **Reliability**: 5% downtime from slow failure detection is too high
2. **Scalability**: HTTP polling won't scale beyond ~100 nodes
3. **Consistency**: Work queue conflicts from lack of consensus
4. **Operational**: Split-brain incidents require manual intervention

## Alternatives Considered

### Alternative 1: Kubernetes

- Pros: Built-in service discovery, automatic failover
- Cons: Requires K8s cluster, adds operational complexity
- Decision: Rejected - overhead not justified for current scale

### Alternative 2: HashiCorp Consul

- Pros: Production-tested service mesh, SWIM-based
- Cons: External dependency, complex setup
- Decision: Rejected - prefer in-process solution

### Alternative 3: Redis Cluster

- Pros: Simple leader election via SETNX
- Cons: External dependency, single point of failure
- Decision: Rejected - adds infrastructure complexity

## References

- [SWIM Paper](https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf)
- [Raft Paper](https://raft.github.io/raft.pdf)
- [PySyncObj Docs](https://github.com/bakwc/PySyncObj)
- [swim-p2p Docs](https://pypi.org/project/swim-p2p/)
