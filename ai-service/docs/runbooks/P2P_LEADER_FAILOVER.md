# P2P Leader Failover

Runbook for handling P2P cluster leader election and failover scenarios.

## Overview

The P2P cluster uses a Bully election algorithm with 5 designated voter nodes.
Leader election requires a quorum of 3 voters to prevent split-brain.

### Voter Nodes (Dec 2025)

| Node              | Role  | Notes                         |
| ----------------- | ----- | ----------------------------- |
| nebius-backbone-1 | Voter | L40S backbone, usually leader |
| nebius-h100-3     | Voter | H100 80GB                     |
| hetzner-cpu1      | Voter | CPU-only, stable connectivity |
| hetzner-cpu2      | Voter | CPU-only, stable connectivity |
| vultr-a100-20gb   | Voter | A100 vGPU                     |

**Quorum**: 3 of 5 voters must be reachable for leader election.

## Quick Commands

```bash
# Check current leader
curl -s http://localhost:8770/status | jq '.leader_id'

# Check voter status
curl -s http://localhost:8770/status | jq '.voters'

# Check alive peers
curl -s http://localhost:8770/status | jq '.alive_peers'

# Force leader check (from any node)
curl -s http://localhost:8770/election/status
```

## Identifying Leader Issues

### Symptoms

- No leader elected (all nodes show `"leader_id": null`)
- Stale leader (leader node is offline but still reported)
- Split brain (different nodes report different leaders)
- Work queue not being processed

### Check Leader Health

```bash
# From any cluster node
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\", \"NONE\")}")
print(f"Role: {d.get(\"role\", \"unknown\")}")
print(f"Alive peers: {d.get(\"alive_peers\", 0)}")
print(f"Voters reachable: {len([v for v in d.get(\"voters\", {}).values() if v.get(\"alive\")])}/5")
'
```

## Resolution Procedures

### Scenario 1: No Leader Elected

**Cause**: Fewer than 3 voters are reachable.

1. Check voter connectivity:

   ```bash
   for host in nebius-backbone-1 nebius-h100-3 hetzner-cpu1 hetzner-cpu2 vultr-a100-20gb; do
     echo -n "$host: "
     curl -s --connect-timeout 3 "http://$host:8770/health" && echo "OK" || echo "UNREACHABLE"
   done
   ```

2. Restart P2P on unreachable voters:

   ```bash
   ssh user@voter-host "cd ~/ringrift/ai-service && pkill -f p2p_orchestrator; nohup python scripts/p2p_orchestrator.py > logs/p2p.log 2>&1 &"
   ```

3. Wait for election (typically 30-60 seconds)

### Scenario 2: Stale Leader

**Cause**: Leader crashed but peers still think it's alive.

1. Check if leader is actually responsive:

   ```bash
   curl -s --connect-timeout 5 http://LEADER_HOST:8770/health
   ```

2. If leader is dead, wait for heartbeat timeout (60s default)

3. Or manually trigger re-election:
   ```bash
   # On any voter node
   curl -X POST http://localhost:8770/election/trigger
   ```

### Scenario 3: Split Brain

**Cause**: Network partition causing different nodes to elect different leaders.

1. Identify all claimed leaders:

   ```bash
   for node in $(cat config/distributed_hosts.yaml | grep "tailscale_ip" | awk '{print $2}'); do
     leader=$(curl -s --connect-timeout 3 "http://$node:8770/status" | jq -r '.leader_id // "none"')
     echo "$node claims leader: $leader"
   done
   ```

2. Restart P2P on conflicting nodes to force re-election:
   ```bash
   ssh user@conflicting-node "pkill -f p2p_orchestrator"
   # Wait for restart via systemd or manual restart
   ```

### Scenario 4: Force Leadership Change

To force a specific node to become leader (use sparingly):

1. Stop P2P on current leader:

   ```bash
   ssh user@current-leader "pkill -f p2p_orchestrator"
   ```

2. The highest-priority available voter will become leader automatically.

## Voter Priority

Voters are prioritized by:

1. Node stability (uptime, failure count)
2. Network centrality (connectivity to other nodes)
3. GPU capability (GPU nodes preferred for data sync)

Current priority order (Dec 2025):

1. nebius-backbone-1 (L40S, excellent connectivity)
2. vultr-a100-20gb (A100, good connectivity)
3. nebius-h100-3 (H100, good connectivity)
4. hetzner-cpu1 (CPU, stable)
5. hetzner-cpu2 (CPU, stable)

## Monitoring

### Prometheus Metrics

```bash
curl -s http://localhost:8770/metrics/prometheus | grep -E "p2p_leader|p2p_election"
```

Key metrics:

- `ringrift_p2p_is_leader` - 1 if this node is leader
- `ringrift_p2p_alive_peers` - Number of reachable peers
- `ringrift_p2p_election_count` - Total elections participated in

### Alerts

| Alert           | Condition                       | Action                   |
| --------------- | ------------------------------- | ------------------------ |
| No Leader       | `leader_id == null` for > 5 min | Check voter connectivity |
| Leader Unstable | > 3 elections in 10 min         | Check leader node health |
| Low Quorum      | < 3 voters reachable            | Restart offline voters   |

## Configuration

Voter nodes are configured in `config/distributed_hosts.yaml`:

```yaml
voters:
  - nebius-backbone-1
  - nebius-h100-3
  - hetzner-cpu1
  - hetzner-cpu2
  - vultr-a100-20gb

election:
  heartbeat_interval: 15 # seconds
  peer_timeout: 60 # seconds
  election_timeout: 30 # seconds
```

## Related

- [CLUSTER_GPU_STUCK.md](./CLUSTER_GPU_STUCK.md) - GPU node issues
- [DAEMON_FAILURE_RECOVERY.md](./DAEMON_FAILURE_RECOVERY.md) - Daemon restart
- `scripts/p2p_orchestrator.py` - P2P orchestrator source

## History

- Dec 2025: Updated voter list (removed Lambda nodes)
- Dec 2025: Added vultr-a100-20gb as voter
- Dec 2025: Reduced heartbeat interval to 15s for faster failover
