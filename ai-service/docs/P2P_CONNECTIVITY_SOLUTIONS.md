# P2P Connectivity Solutions for Vast.ai ↔ Cluster

## Problem Statement

Vast.ai containers cannot connect to cluster nodes for P2P orchestration:

- Vast containers are behind NAT (no inbound connections)
- Cannot reach Tailscale-only IPs (100.x.x.x) without Tailscale installed
- Cannot reach some public IPs on port 8770 (firewall blocks)
- Result: P2P mesh is fragmented, Vast nodes isolated

**Status (2025-12):** Lambda Labs account terminated (legacy hosts removed).
Use `ai-service/config/distributed_hosts.yaml` as the source of truth for active nodes.

## Solution Options

### Option 1: Tailscale in Vast Containers (RECOMMENDED)

**How it works:** Install Tailscale in each Vast container to join the same tailnet as cluster nodes.

**Pros:**

- Direct P2P connectivity (lowest latency)
- End-to-end encryption via WireGuard
- NAT traversal built-in
- Same network as cluster nodes

**Cons:**

- Requires auth key per container
- Containers are ephemeral - need to re-auth on restart
- Tailscale has 100-device limit on free tier

**Implementation:**

```bash
# On each Vast container at startup:
curl -fsSL https://tailscale.com/install.sh | sh

# Start in userspace mode (no systemd in containers)
mkdir -p /var/run/tailscale /var/lib/tailscale
tailscaled --tun=userspace-networking \
  --statedir=/var/lib/tailscale \
  --socket=/var/run/tailscale/tailscaled.sock &

# Authenticate with pre-generated auth key
tailscale --socket=/var/run/tailscale/tailscaled.sock up \
  --authkey=tskey-auth-xxxxx \
  --hostname=vast-$(hostname)
```

**Auth Key Setup:**

1. Go to https://login.tailscale.com/admin/settings/keys
2. Generate auth key with:
   - Reusable: Yes (for multiple containers)
   - Ephemeral: Yes (auto-cleanup on disconnect)
   - Tags: `tag:vast-nodes`
3. Store key in `config/.tailscale-authkey`

**Estimated Latency:** 5-50ms (direct P2P)

---

### Option 2: Cloudflare Zero Trust Tunnel

**How it works:** Each node runs `cloudflared` tunnel, exposing P2P API via Cloudflare's edge network.

**Pros:**

- No firewall changes needed
- Works through any NAT
- Free tier available
- Built-in DDoS protection

**Cons:**

- All traffic routes through Cloudflare (added latency)
- Requires Cloudflare account + domain
- More complex setup
- Not true P2P (hub-and-spoke through Cloudflare)

**Implementation:**

1. **Create Cloudflare Tunnel** (one-time):

```bash
cloudflared tunnel login
cloudflared tunnel create ringrift-p2p
```

2. **Configure tunnel** (`~/.cloudflared/config.yml`):

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: /root/.cloudflared/<TUNNEL_UUID>.json

ingress:
  # Cluster nodes expose their P2P via subdomains
  - hostname: runpod-h100.ringrift.example.com
    service: http://localhost:8770
  - hostname: vast-29128352.ringrift.example.com
    service: http://localhost:8770
  - service: http_status:404
```

3. **Run on each node:**

```bash
# Cluster side
cloudflared tunnel run ringrift-p2p

# Vast side
cloudflared tunnel run ringrift-p2p
```

4. **Update P2P config** to use Cloudflare hostnames:

```yaml
# distributed_hosts.yaml
runpod-h100:
  p2p_url: https://runpod-h100.ringrift.example.com
  # or use Cloudflare Access with WARP
```

**Estimated Latency:** 50-150ms (through Cloudflare edge)

---

### Option 3: Open Firewall on Coordinator

**How it works:** Configure the coordinator firewall to allow inbound connections on port 8770 from Vast's egress IPs.

**Pros:**

- Simple, direct connectivity
- No additional software
- Low latency

**Cons:**

- Exposes P2P API to internet
- Vast IPs are dynamic (containers can get new IPs)
- Requires firewall access on the provider nodes (may not be available)
- Security risk if P2P has vulnerabilities

**Implementation:**

```bash
# On provider nodes (requires root):
sudo iptables -A INPUT -p tcp --dport 8770 -j ACCEPT

# Or restrict to Vast IP ranges (if known):
sudo iptables -A INPUT -p tcp --dport 8770 -s 93.91.0.0/16 -j ACCEPT
```

**Note:** Some providers may not provide firewall access.

---

### Option 4: Relay/Proxy Server

**How it works:** Set up a relay server that both Runpod and Vast can reach, acting as a message broker.

**Pros:**

- Works with any NAT configuration
- Centralized logging/monitoring
- Can add authentication layer

**Cons:**

- Single point of failure
- Added latency (all traffic through relay)
- Additional infrastructure to maintain
- Not true mesh networking

**Implementation Options:**

**A. Simple HTTP Relay:**

```python
# relay_server.py - runs on a public server
from aiohttp import web

peers = {}

async def register(request):
    data = await request.json()
    peers[data['node_id']] = data['endpoint']
    return web.json_response({'peers': peers})

async def forward(request):
    target = request.match_info['target']
    if target not in peers:
        return web.json_response({'error': 'unknown peer'}, status=404)
    # Forward request to target
    ...
```

**B. Use existing solutions:**

- NATS.io - lightweight message broker
- Redis Pub/Sub - if already using Redis
- RabbitMQ - enterprise-grade but heavier

---

### Option 5: Hybrid Approach (RECOMMENDED FOR PRODUCTION)

**How it works:** Combine Tailscale for Runpod/self-hosted nodes + Cloudflare Access for Vast nodes.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                      Tailscale Network                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │runpod-h100│  │gh200-a   │  │gh200-b   │  │gh200-c   │    │
│  │100.78.x.x │  │100.83.x.x│  │100.88.x.x│  │100.x.x.x │    │
│  └─────┬─────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘    │
│        │              │             │             │          │
│        └──────────────┴─────┬───────┴─────────────┘          │
│                             │                                 │
│                    ┌────────▼────────┐                       │
│                    │  P2P Leader     │                       │
│                    │  (coordinator)  │                       │
│                    └────────┬────────┘                       │
└─────────────────────────────┼───────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Cloudflare Tunnel │
                    │ (p2p.ringrift.com)│
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │vast-5090│          │vast-4090│          │vast-3060│
   │(NAT)    │          │(NAT)    │          │(NAT)    │
   └─────────┘          └─────────┘          └─────────┘
```

**Benefits:**

- Runpod/self-hosted nodes communicate directly via Tailscale (low latency)
- Vast nodes connect through Cloudflare (works through NAT)
- Single tunnel endpoint simplifies Vast configuration
- Graceful degradation if Cloudflare unavailable

---

## Recommended Implementation Order

### Phase 1: Quick Win - Tailscale on Vast (Today)

1. Generate Tailscale auth key (ephemeral, reusable)
2. Add to Vast container startup script
3. Test P2P connectivity

```bash
# Generate auth key at: https://login.tailscale.com/admin/settings/keys
# Save to: config/.tailscale-authkey

# Add to Vast onstart.sh:
curl -fsSL https://tailscale.com/install.sh | sh
mkdir -p /var/run/tailscale /var/lib/tailscale
tailscaled --tun=userspace-networking --statedir=/var/lib/tailscale &
sleep 3
tailscale up --authkey=$(cat /workspace/ringrift/ai-service/config/.tailscale-authkey) \
  --hostname=vast-$(hostname | cut -d- -f2)
```

### Phase 2: Robustness - Cloudflare Backup (This Week)

1. Set up Cloudflare tunnel on coordinator node
2. Configure as backup when Tailscale unavailable
3. Update P2P to try Tailscale first, fallback to Cloudflare

### Phase 3: Automation (Next Sprint)

1. Auto-provisioning of Tailscale for new Vast instances
2. Health monitoring for connectivity
3. Automatic failover between Tailscale/Cloudflare

---

## Configuration Files Needed

### 1. Tailscale Auth Key Storage

```yaml
# config/secrets.yaml (gitignored)
tailscale:
  auth_key: tskey-auth-xxxxxxxxxx
  tags:
    - tag:vast-nodes
```

### 2. Cloudflare Tunnel Config

```yaml
# config/cloudflare-tunnel.yaml
tunnel: <uuid>
credentials-file: /etc/cloudflared/<uuid>.json
ingress:
  - hostname: p2p.ringrift.example.com
    service: http://localhost:8770
  - service: http_status:404
```

### 3. P2P Discovery Update

```yaml
# config/distributed_hosts.yaml additions
p2p_discovery:
  methods:
    - tailscale # Primary
    - cloudflare # Fallback
  cloudflare_endpoint: https://p2p.ringrift.example.com
```

---

## Security Considerations

1. **Tailscale:** Uses WireGuard encryption, requires auth key
2. **Cloudflare:** Can add Cloudflare Access for authentication
3. **Open Firewall:** Least secure, not recommended
4. **Relay:** Depends on relay implementation

## Cost Analysis

| Solution          | Cost               | Notes                      |
| ----------------- | ------------------ | -------------------------- |
| Tailscale         | Free (100 devices) | $5/user/mo for more        |
| Cloudflare Tunnel | Free               | Paid for advanced features |
| Open Firewall     | Free               | Security cost              |
| Custom Relay      | ~$10/mo            | VPS hosting                |

## Decision Matrix

| Criteria         | Tailscale | Cloudflare | Firewall | Relay  |
| ---------------- | --------- | ---------- | -------- | ------ |
| Setup Complexity | Medium    | Medium     | Low      | High   |
| Latency          | Low       | Medium     | Low      | Medium |
| Security         | High      | High       | Low      | Medium |
| Reliability      | High      | High       | Medium   | Medium |
| Cost             | Free\*    | Free       | Free     | Low    |
| NAT Support      | Yes       | Yes        | No       | Yes    |

**Recommendation:** Start with **Tailscale** (Option 1), add **Cloudflare** (Option 5 hybrid) for robustness.
