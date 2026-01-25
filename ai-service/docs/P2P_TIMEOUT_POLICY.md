# P2P Network Timeout Policy

**Last Updated**: January 25, 2026

This document defines the canonical timeout values for the P2P mesh network. These values should NOT be changed without updating this document and testing in a staging environment.

## Design Principles

1. **Provider-aware timeouts**: Different cloud providers have different network characteristics
2. **Conservative defaults**: Prefer false-negatives (slower detection) over false-positives (spurious failures)
3. **Relay tolerance**: NAT-blocked nodes need longer timeouts due to relay overhead
4. **Quorum stability**: Voter nodes should have tight timeouts to maintain quorum health

## Provider Timeout Configuration

| Provider     | Peer Timeout | Heartbeat Interval | Network Characteristics                             |
| ------------ | ------------ | ------------------ | --------------------------------------------------- |
| Lambda GH200 | 120s         | 15s                | NAT-blocked, relay-dependent, high latency variance |
| Vast.ai      | 90s          | 15s                | CGNAT, P2PD hole punching, consumer networks        |
| RunPod       | 90s          | 15s                | Variable connectivity, CGNAT possible               |
| Nebius       | 90s          | 10s                | Direct connection, stable cloud infrastructure      |
| Hetzner      | 60s          | 10s                | Always-online CPU voters, stable connections        |
| Vultr        | 60s          | 10s                | Stable GPU nodes, direct connectivity               |
| Mac (local)  | 60s          | 10s                | Local coordinator nodes, low latency                |

## Core Constants

```python
# ai-service/app/p2p/constants.py
PEER_TIMEOUT = 120          # Max time to consider peer alive without heartbeat
PEER_DEAD_TIMEOUT = 120     # MUST equal PEER_TIMEOUT for consistency
HEARTBEAT_INTERVAL = 15     # Base interval between heartbeats
ELECTION_TIMEOUT = 30       # Time to wait for election responses
GOSSIP_INTERVAL = 15        # Interval for gossip protocol messages
STARTUP_GRACE_PERIOD = 120  # Grace period for slow state loading at startup
```

## Relay Configuration

NAT-blocked nodes require relay through public-facing nodes:

1. **Primary relay**: First choice for relay (should be stable, low-latency)
2. **Secondary/Tertiary/Quaternary**: Fallbacks if primary fails

**Relay-capable nodes** (always online, public IP):

- hetzner-cpu1, hetzner-cpu2, hetzner-cpu3
- vultr-a100-20gb
- nebius-h100-1, nebius-backbone-1

**Relay rules**:

1. Never use offline nodes as relays
2. Diversify relay chains - don't put same node as primary for all hosts
3. Each NAT-blocked node needs at least 3 relays for resilience
4. Hetzner nodes are preferred relays (always online, no GPU contention)

## Leader Election

Using **bully consensus** (Raft disabled due to pysyncobj bugs):

```yaml
protocols:
  membership_mode: bully
  consensus_mode: bully
  raft:
    enabled: false # pysyncobj "unhashable type: dict" errors
```

**Voter quorum** (4 of 8 required):

- local-mac, mac-studio (coordinators)
- vultr-a100-20gb, nebius-h100-1, nebius-h100-3 (GPU nodes)
- hetzner-cpu1, hetzner-cpu2, hetzner-cpu3 (CPU voters)

## Dead Peer Detection

```python
# Detection thresholds
CONSECUTIVE_FAILURES_TO_RETIRE = 4    # Heartbeat failures before retirement
PROBE_RETIRED_INTERVAL = 300          # Seconds between probing retired peers
MIN_PROBE_RETIRED_INTERVAL = 60       # Minimum probe interval
```

**Retirement flow**:

1. Peer misses `CONSECUTIVE_FAILURES_TO_RETIRE` heartbeats
2. Peer marked as "retired" (not dead - may recover)
3. Retired peers are probed every `PROBE_RETIRED_INTERVAL` seconds
4. If probe succeeds, peer is un-retired and rejoins cluster

## Health Check Intervals

| Check Type      | Interval | Timeout | Purpose                       |
| --------------- | -------- | ------- | ----------------------------- |
| LeaderProbeLoop | 10s      | 5s      | Fast leader failure detection |
| Peer heartbeat  | 15s      | 10s     | Peer liveness                 |
| Manager health  | 30s      | 15s     | Internal manager health       |
| Daemon health   | 60s      | 30s     | Background daemon health      |

## Stability Metrics

**Target SLOs**:

- Leader failover: < 70s (via LeaderProbeLoop)
- Peer recovery MTTR: < 2.5 min
- Quorum maintenance: 99.9% uptime
- False-positive rate: < 0.1% per hour

## Change Control

To modify timeout values:

1. Update this document with rationale
2. Update `distributed_hosts.yaml` provider_timeouts section
3. Update `app/p2p/constants.py` if changing core constants
4. Test in staging with at least 10 nodes for 4+ hours
5. Deploy to production during low-activity period
6. Monitor for 24 hours before considering stable

## Historical Changes

| Date         | Change                       | Reason                            | Outcome                   |
| ------------ | ---------------------------- | --------------------------------- | ------------------------- |
| Jan 25, 2026 | PEER_DEAD_TIMEOUT 60s → 120s | Sync with PEER_TIMEOUT            | Reduced spurious failures |
| Jan 19, 2026 | Provider-specific timeouts   | Different network characteristics | Improved stability        |
| Jan 15, 2026 | Raft disabled                | pysyncobj bugs                    | Using bully consensus     |
| Dec 27, 2025 | Heartbeat 30s → 15s          | Faster peer discovery             | Improved responsiveness   |
| Dec 27, 2025 | Peer timeout 90s → 60s       | Faster dead node detection        | Faster failover           |
