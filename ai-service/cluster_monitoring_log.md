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
