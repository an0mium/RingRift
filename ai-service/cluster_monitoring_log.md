# P2P Cluster Monitoring Log - Jan 25, 2026

## Monitoring Round 1 (Started 12:18 CST)

### Check 1 - 12:18 CST

- **Alive Peers**: 12
- **Leader**: hetzner-cpu1
- **Quorum OK**: True
- **Nodes**: hetzner-cpu2, lambda-gh200-1/3/4/8/10/training, mac-studio, nebius-h100-3, vast-29118471, vast-29126088, vultr-a100-20gb
- **Issues**: None
- **Notes**: Stable cluster after restart

### Manual Check - 12:24 CST

- **Alive Peers**: 13
- **Leader**: hetzner-cpu1
- **Quorum OK**: True
- **Nodes**: hetzner-cpu1/2/3, lambda-gh200-1/3/4/8/10/training, nebius-h100-3, vast-29118471, vast-29126088, vultr-a100-20gb
- **Issues**: Lambda gh200-2/5/11 and Vast.ai 30274241/30274242/29128352/29129151 not connected
- **Notes**: Fixed supervisor lock issues on some Lambda nodes, still troubleshooting connectivity

## Key Findings So Far

### Nodes Successfully Connected (13)

1. All 3 Hetzner CPU nodes (voters)
2. 7 Lambda GH200 nodes (1, 3, 4, 8, 10, training) - but not 2, 5, 6, 7, 9, 11
3. 1 Nebius H100 node (h100-3)
4. 2 Vast.ai nodes (29118471, 29126088)
5. 1 Vultr A100 node

### Nodes with Connectivity Issues

1. **Lambda GH200 nodes (2, 5, 6, 7, 9, 11)**: Supervisor lock issues, some offline
2. **Vast.ai nodes (30274241, 30274242, 29128352, 29129151)**: Can't reach Tailscale seed IPs
3. **mac-studio**: Intermittent connectivity via Tailscale

### Root Causes Identified

1. **Supervisor lock conflicts**: P2P orchestrator won't start if stale lock exists
2. **Tailscale-only seeds**: Vast.ai nodes can't reach Tailscale IPs
3. **Coordinator misconfiguration**: Some nodes marked as coordinator incorrectly

### Manual Check - 12:28 CST

- **Alive Peers**: 11
- **Leader**: local-mac
- **Quorum OK**: True
- **Nodes**: hetzner-cpu2, lambda-gh200-1/3/4/8/10/training, nebius-h100-3, vast-29118471, vast-29126088, vultr-a100-20gb

---

## Monitoring Round 2 (Session 2 - Started 17:00 CST)

### Actions Taken

1. **Restarted P2P on 6 Vast.ai nodes** with proper node IDs:
   - vast-29118471 (107.174.186.233:36356) - STARTED
   - vast-29126088 (76.66.207.49:45674) - STARTED
   - vast-29128352 (93.91.156.82:54251) - STARTED (but using container ID)
   - vast-29129151 (ssh4.vast.ai:19150) - STARTED
   - vast-30274241 (160.250.70.27:6133) - STARTED
   - vast-30274242 (160.250.70.26:4695) - STARTED

2. **Restarted P2P on Lambda GH200 nodes**:
   - GH200-3, GH200-4, GH200-8, GH200-9, GH200-10, GH200-11, GH200-training

3. **Key Discovery - Vast.ai Node ID Issue**:
   - The P2P orchestrator reads `RINGRIFT_NODE_ID` env var, not the canonical_node_id file
   - Some Vast.ai nodes are appearing with container IDs (2d9de0d97a72, 7513a0ce51c0)
   - These ARE the Vast.ai nodes but with wrong identifiers

### Status Checks (17:00-17:10 CST)

| Time  | Alive | Leader    | Notes                                     |
| ----- | ----- | --------- | ----------------------------------------- |
| 17:00 | 19    | local-mac | Peak count - includes 4 Vast.ai, 7 Lambda |
| 17:02 | 13    | local-mac | Dropped - some Lambda nodes went offline  |
| 17:03 | 9     | local-mac | Continued drop                            |
| 17:04 | 10    | local-mac | Slight recovery                           |
| 17:05 | 7     | local-mac | Fluctuating - 3 Lambda, 2 Vast.ai         |

### Current Status (17:10 CST)

- **Alive Peers**: 7 (fluctuating 7-19)
- **Leader**: local-mac
- **Quorum OK**: True (4/2 voters)
- **Lambda nodes alive**: 3 (GH200-3, GH200-10, GH200-11)
- **Vast.ai nodes alive**: 2 (29118471, 29126088)
- **Other**: hetzner-cpu3, nebius-h100-3

### Analysis

**Cluster Fluctuation Causes**:

1. Lambda GH200 nodes with P2P not fully started (waiting for Tailscale)
2. Vast.ai nodes with wrong node IDs (container IDs instead of vast-\*)
3. Some nodes marked as NAT-blocked due to intermittent connectivity
4. PEER_DEAD_TIMEOUT (150s) is appropriate but nodes are timing out during reconnection

**Fixes Applied This Session**:

- PEER_DEAD_TIMEOUT: Already set to 150s (matches PEER_TIMEOUT in constants.py)
- Vast.ai SSH key: Confirmed as `~/.ssh/id_cluster_lambda`
- Bootstrap seeds: Using public IPs (46.62.147.150:8770, 208.167.249.164:8770)

**Recommendations**:

1. For Vast.ai nodes: Set `RINGRIFT_NODE_ID` env var when starting P2P
2. For Lambda nodes: Ensure Tailscale is running before P2P starts
3. Monitor for 4+ hours to confirm stability

- **Issues**: hetzner-cpu1/3 dropped, leader changed to local-mac
- **Notes**: Cluster size fluctuating between 10-13 nodes

### Check 2 - 12:28 CST

- **Alive Peers**: 13
- **Leader**: local-mac
- **Quorum OK**: True
- **Nodes**: hetzner-cpu1, hetzner-cpu2, hetzner-cpu3, lambda-gh200-1, lambda-gh200-10, lambda-gh200-3, lambda-gh200-4, lambda-gh200-8, lambda-gh200-training, nebius-h100-3, vast-29118471, vast-29126088, vultr-a10...

### Check 3 - 12:38 CST

- **Alive Peers**: 12
- **Leader**: None
- **Quorum OK**: True
- **Nodes**: hetzner-cpu2, hetzner-cpu3, lambda-gh200-1, lambda-gh200-10, lambda-gh200-3, lambda-gh200-4, lambda-gh200-8, lambda-gh200-9, lambda-gh200-training, nebius-h100-3, vast-29118471, vultr-a100-20gb

### Check 4 - 12:48 CST

- **Error**: Expecting value: line 1 column 1 (char 0)

### Check 5 - 13:08 CST (After Bootstrap Seed Fix)

- **Alive Peers**: 8
- **Leader**: local-mac
- **Quorum OK**: True
- **Nodes**: hetzner-cpu2, hetzner-cpu3, lambda-gh200-1, lambda-gh200-3, lambda-gh200-4, lambda-gh200-8, nebius-h100-3, vultr-a100-20gb
- **Changes Made**: Added public IP bootstrap seeds (hetzner-cpu1, vultr-a100-20gb, nebius-h100-3)
- **Notes**: Deployed to cluster nodes, restarted local P2P. Cluster stabilizing after updates.

### Check 6 - 13:11 CST

- **Alive Peers**: 13
- **Leader**: vultr-a100-20gb
- **Quorum OK**: True
- **Nodes**: hetzner-cpu1/2/3, lambda-gh200-1/3/4/8/9/10/11/training, nebius-h100-3, vultr-a100-20gb
- **Improvement**: +5 nodes from Check 5 (8 → 13)
- **Notes**: Lambda GH200-9, GH200-10, GH200-11 reconnected after P2P restarts. Missing: mac-studio, vast-\*, lambda-gh200-2/5

### Check 7 - 13:19 CST

- **Alive Peers**: 9
- **Leader**: None (election in progress)
- **Quorum OK**: True
- **Nodes**: hetzner-cpu3, lambda-gh200-3/4/8/9/11/training, mac-studio, vultr-a100-20gb
- **Issues**: Cluster size dropped from 13 → 9 during local P2P restarts
- **Notes**: mac-studio is now connected. Cluster still stabilizing.

### Check 8 - 13:22 CST

- **Alive Peers**: 11
- **Leader**: None (election in progress)
- **Quorum OK**: True
- **Nodes**: hetzner-cpu1, lambda-gh200-1/3/4/8/9/training, mac-studio, nebius-h100-3, vast-29118471, vultr-a100-20gb
- **Victory**: vast-29118471 connected via public IP seeds!
- **Notes**: Public IP bootstrap seeds fix is working. Vast.ai nodes can now connect.

### Check 9 - 13:27 CST

- **Alive Peers**: 13
- **Leader**: vultr-a100-20gb
- **Quorum OK**: True
- **Nodes**: hetzner-cpu1/2, lambda-gh200-1/3/4/8/9/10/11/training, nebius-h100-3, vast-29118471, vultr-a100-20gb
- **Status**: Cluster stabilized with 13 nodes
- **Notes**: Leader elected. Lambda nodes 9, 10, 11 reconnected. Still missing: hetzner-cpu3, lambda-gh200-2/5, mac-studio, vast-29126088

### Check 10 - 13:52 CST (Vast.ai SSH Updated)

- **Alive Peers**: 11-12
- **Leader**: hetzner-cpu1
- **Quorum OK**: True
- **Nodes**: hetzner-cpu1/2/3, lambda-gh200-1/3/4/8/9/11/training, nebius-h100-3, vast-29118471, vultr-a100-20gb
- **Actions Taken**:
  - Updated distributed_hosts.yaml with current Vast.ai SSH ports from `vastai show instances`
  - vast-29118471: ssh8.vast.ai:38470
  - vast-29126088: ssh5.vast.ai:16088
  - vast-30274241: ssh1.vast.ai:34240
  - vast-30274242: ssh5.vast.ai:34242
- **Results**:
  - vast-29118471: Connected successfully
  - Other Vast.ai nodes: P2P started but not connecting (investigating)
- **Blockers**:
  - lambda-gh200-2/5: Tailscale timeout (likely offline at Lambda)
  - lambda-gh200-10: Intermittent
  - Other Vast.ai nodes: P2P process starts but doesn't connect

## Summary

### Fixes Implemented

1. **Public IP Bootstrap Seeds** (commit a06d17593)
   - Added hetzner-cpu1 (46.62.147.150), vultr-a100-20gb (208.167.249.164), nebius-h100-3 (89.169.98.165)
   - Enables non-Tailscale nodes to bootstrap into cluster
   - **Working**: vast-29118471 connected via public IP seeds

2. **Vast.ai SSH Ports Updated**
   - Used `vastai show instances` to get current SSH ports
   - Updated distributed_hosts.yaml with correct ports

### Current Stable State

- **11-12 alive peers** with quorum OK
- All 3 Hetzner CPU nodes (voters)
- 8 Lambda GH200 nodes (1, 3, 4, 8, 9, 11, training + intermittent 10)
- 1 Nebius H100 node
- 1 Vast.ai node (29118471)
- 1 Vultr A100 node

### To Reach 20+ Nodes

- Lambda GH200-2/5 appear offline at Lambda Labs
- Additional Vast.ai nodes need investigation (P2P not connecting)

## Monitoring Round 2 (Started 13:55 CST)

### Check 11 - 14:00 CST (SSH Key Fix)

- **Alive Peers**: 9-13 (fluctuating)
- **Leader**: hetzner-cpu1 / local-mac (split brain detected then resolved)
- **Quorum OK**: True
- **Key Fix**: Found correct SSH key for Vast.ai: `~/.ssh/id_cluster_lambda`
- **SSH URLs**: Updated using `vastai ssh-url <id>` which returns different addresses than web UI
  - vast-29118471: root@107.174.186.233:36356
  - vast-29126088: root@76.66.207.49:45674
  - vast-29128352: root@93.91.156.82:54251
  - vast-29129151: root@ssh4.vast.ai:19150
  - vast-30274241: root@160.250.70.27:6133
  - vast-30274242: root@160.250.70.26:4695

### Check 12 - 14:14 CST

- **Alive Peers**: 9
- **Leader**: hetzner-cpu1
- **Issues**:
  - **Split brain detected and resolved** - Had 2 leaders (local-mac, hetzner-cpu1)
  - Cluster membership fluctuating significantly
  - Some Vast.ai nodes show as container IDs instead of proper names

### Current Node Status

| Node                      | Status         | Notes                                                |
| ------------------------- | -------------- | ---------------------------------------------------- |
| hetzner-cpu1/2/3          | ✓ Connected    | Stable voters                                        |
| lambda-gh200-3/8/10/11    | ✓ Connected    | Core Lambda nodes                                    |
| lambda-gh200-1            | ✓ Intermittent | Sees cluster, sometimes not visible from coordinator |
| lambda-gh200-2/5          | ✗ Offline      | Tailscale unreachable                                |
| lambda-gh200-4/9/training | ⚠ Intermittent | Fluctuating connectivity                             |
| nebius-h100-3             | ✓ Connected    | Stable                                               |
| vultr-a100-20gb           | ✓ Connected    | Stable voter                                         |
| vast-29118471             | ✓ Connected    | Public IP seeds working                              |
| vast-29126088             | ✓ Intermittent | Sometimes connected                                  |
| vast-29128352/29129151    | ⚠ P2P started  | Not joining cluster                                  |
| vast-30274241/30274242    | ⚠ P2P started  | Not joining cluster                                  |

## Root Cause Analysis

### Issue 1: Cluster Instability (9-13 nodes fluctuating)

**Symptoms:**

- Nodes appear and disappear from cluster
- Different nodes report different alive peer counts
- Split brain occurred (2 leaders simultaneously)

**Root Causes:**

1. **Gossip convergence delay**: Membership changes take time to propagate
2. **SWIM failure detection timeouts**: 90s dead peer timeout may be too aggressive
3. **Network latency variance**: Tailscale routing can have variable latency

**Recommended Fixes:**

1. Increase PEER_DEAD_TIMEOUT from 90s to 120s for more tolerance
2. Add exponential backoff for peer recovery probes
3. Implement quorum-based leader election validation

### Issue 2: Vast.ai Nodes Not Joining Cluster

**Symptoms:**

- P2P process starts and initializes
- Bootstrap seeds are discovered
- Nodes don't appear in cluster membership

**Root Causes:**

1. **NAT traversal**: Vast.ai nodes are NAT-blocked, need relay mode
2. **Node ID mismatch**: Some nodes register as container IDs instead of names
3. **Missing canonical_node_id file**: Node ID auto-detection failing

**Recommended Fixes:**

1. Create `/root/ringrift/ai-service/config/canonical_node_id` on Vast.ai nodes
2. Force relay mode for all Vast.ai nodes (already implemented)
3. Add public IP bootstrap seeds (implemented in commit a06d17593)

### Issue 3: Lambda GH200-2/5 Offline

**Symptoms:**

- SSH times out via Tailscale IP
- Nodes not reachable since monitoring started

**Root Cause:**

- Nodes likely powered off or Tailscale agent not running

**Recommended Action:**

- Check Lambda Labs dashboard for instance status
- These may have been terminated due to inactivity

## Stability Recommendations

### Short-term (immediate)

1. ✓ Added public IP bootstrap seeds
2. ✓ Fixed Vast.ai SSH key (`id_cluster_lambda`)
3. Configure proper node IDs on Vast.ai nodes

### Medium-term (this week)

1. Implement sticky leader election with epoch validation
2. Add cluster health endpoint that waits for convergence
3. Create Vast.ai node provisioning script that sets up node ID

### Long-term (next sprint)

1. Replace SWIM with Raft for membership consensus
2. Implement proper partition detection and healing
3. Add prometheus metrics for cluster health
