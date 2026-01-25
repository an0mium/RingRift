# P2P Cluster Stability Recommendations

Generated: Jan 25, 2026

## Current State

- 13 nodes connected (target: 20+)
- Cluster quorum: OK
- Leader: hetzner-cpu1

## Root Causes Identified

### 1. Supervisor Lock Conflicts (High Impact)

**Problem:** P2P orchestrator fails to start if stale lock files exist in `/tmp/p2p_*`.

**Fix:**

```bash
# Add to P2P startup command
python scripts/p2p_orchestrator.py --force-supervisor --kill-duplicates
```

**Long-term:** Add automatic stale lock cleanup on startup (check PID validity).

### 2. Tailscale-Only Bootstrap Seeds (High Impact)

**Problem:** Vast.ai and other non-Tailscale nodes can't reach hardcoded seed IPs (all Tailscale addresses).

**Fix:**

```python
# In p2p_orchestrator.py, add public IP seeds
BOOTSTRAP_SEEDS = [
    "208.167.249.164:8770",  # vultr-a100-20gb (public)
    "46.62.147.150:8770",    # hetzner-cpu1 (public)
    "89.169.98.165:8770",    # nebius-h100-3 (public)
    # ... existing Tailscale seeds
]
```

**Or:** Start nodes with `--peers <public_ip>:8770,<public_ip2>:8770`

### 3. Coordinator Misconfiguration (Medium Impact)

**Problem:** Some GPU nodes marked as `coordinator` in YAML, disabling training capabilities.

**Fix:** In `distributed_hosts.yaml`, ensure GPU nodes have:

```yaml
role: gpu_training_primary # NOT coordinator
```

### 4. NAT Blocking (Medium Impact)

**Problem:** Lambda nodes behind CGNAT can't accept incoming connections.

**Current Mitigation:** P2P relay system is working. Nodes use `relay_via` peers.

**Enhancement:** Add more relay-capable nodes to improve coverage.

### 5. Gossip State Serialization Blocking (Low-Medium Impact)

**Problem:** 8MB+ gossip state JSON serialization blocks event loop.

**Fix:** Already partially implemented. Ensure `asyncio.to_thread()` wraps heavy operations.

## Recommended Startup Command

For non-Tailscale nodes:

```bash
RINGRIFT_NODE_ID=<node_id> python scripts/p2p_orchestrator.py \
    --force-supervisor \
    --kill-duplicates \
    --peers 208.167.249.164:8770,46.62.147.150:8770
```

## Quick Fixes to Implement

1. **Add public IP seeds** to `BOOTSTRAP_SEEDS` in `p2p_orchestrator.py`
2. **Add cleanup logic** for stale supervisor locks on startup
3. **Update distributed_hosts.yaml** to fix coordinator misconfigurations
4. **Create startup scripts** per provider type (Lambda, Vast.ai, etc.)

## Nodes to Fix

| Node            | Issue             | Fix                         |
| --------------- | ----------------- | --------------------------- |
| lambda-gh200-2  | Supervisor lock   | Use --force-supervisor      |
| lambda-gh200-5  | Supervisor lock   | Use --force-supervisor      |
| lambda-gh200-11 | Supervisor lock   | Use --force-supervisor      |
| vast-30274241   | Can't reach seeds | Add --peers with public IPs |
| vast-30274242   | Can't reach seeds | Add --peers with public IPs |
| vast-29128352   | Can't reach seeds | Add --peers with public IPs |
| vast-29129151   | Can't reach seeds | Add --peers with public IPs |
