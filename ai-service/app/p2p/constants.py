"""P2P Protocol Constants.

Canonical source for P2P feature flags and configuration used by app layer.
These values are also available in scripts.p2p.constants for CLI tools.

Environment Variable Configuration:
    RINGRIFT_RAFT_ENABLED - Enable Raft consensus (default: false)
    RINGRIFT_SWIM_ENABLED - Enable SWIM membership (default: false)
    RINGRIFT_CONSENSUS_MODE - bully|raft|hybrid (default: bully)
    RINGRIFT_MEMBERSHIP_MODE - http|swim|hybrid (default: http)
"""

from __future__ import annotations

import os

# ============================================
# Raft Protocol Configuration
# ============================================
# Raft provides replicated state machines for work queue and job assignments.
# Sub-second leader failover, distributed locks, automatic log compaction.

RAFT_ENABLED = os.environ.get("RINGRIFT_RAFT_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}
RAFT_BIND_PORT = int(os.environ.get("RINGRIFT_RAFT_PORT", "4321"))
RAFT_COMPACTION_MIN_ENTRIES = int(
    os.environ.get("RINGRIFT_RAFT_COMPACTION_MIN", "1000")
)
RAFT_AUTO_UNLOCK_TIME = float(os.environ.get("RINGRIFT_RAFT_AUTO_UNLOCK", "300.0"))

# ============================================
# SWIM Protocol Configuration
# ============================================
# SWIM provides leaderless membership with O(1) bandwidth and <5s failure detection.
# Complements existing HTTP heartbeats - can run in hybrid mode during migration.

SWIM_ENABLED = os.environ.get("RINGRIFT_SWIM_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}
SWIM_BIND_PORT = int(os.environ.get("RINGRIFT_SWIM_PORT", "7947"))
SWIM_FAILURE_TIMEOUT = float(os.environ.get("RINGRIFT_SWIM_FAILURE_TIMEOUT", "5.0"))
SWIM_SUSPICION_TIMEOUT = float(os.environ.get("RINGRIFT_SWIM_SUSPICION_TIMEOUT", "3.0"))
SWIM_PING_INTERVAL = float(os.environ.get("RINGRIFT_SWIM_PING_INTERVAL", "1.0"))
SWIM_INDIRECT_PING_COUNT = int(os.environ.get("RINGRIFT_SWIM_INDIRECT_PINGS", "3"))

# ============================================
# Hybrid Mode Feature Flags
# ============================================
# These flags control gradual migration from Bully/HTTP to Raft/SWIM.
# Use hybrid modes for safe rollout with instant rollback capability.

# Consensus mode for leader election and work distribution
# - "bully": Current Bully algorithm (default, backward compatible)
# - "raft": Use PySyncObj Raft for consensus (requires RAFT_ENABLED)
# - "hybrid": Raft for work queue, Bully for leader election
CONSENSUS_MODE = os.environ.get("RINGRIFT_CONSENSUS_MODE", "bully")

# Membership mode for failure detection and peer discovery
# - "http": Current HTTP heartbeat-based (default, backward compatible)
# - "swim": Use SWIM gossip protocol (requires SWIM_ENABLED)
# - "hybrid": Use both, prefer SWIM when available
MEMBERSHIP_MODE = os.environ.get("RINGRIFT_MEMBERSHIP_MODE", "http")
