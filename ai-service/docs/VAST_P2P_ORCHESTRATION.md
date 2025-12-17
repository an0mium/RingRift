# Vast.ai P2P Orchestration System

## Overview

This document describes the automated P2P orchestration system for Vast.ai GPU instances, integrating Tailscale mesh networking, aria2 parallel downloads, and optional Cloudflare tunnels for NAT traversal.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        P2P Mesh Network                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ mac-studio  │    │ lambda-a10  │    │ lambda-h100 │             │
│  │ (Leader)    │◄──►│ (Voter)     │◄──►│ (Voter)     │             │
│  │ TS: 100.x   │    │ TS: 100.x   │    │ TS: 100.x   │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│         ┌──────────────────┼──────────────────┐                     │
│         │                  │                  │                     │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐             │
│  │ vast-5090x8 │    │ vast-a40    │    │ vast-4080s  │  ... x15    │
│  │ P2P:8770    │    │ P2P:8770    │    │ P2P:8770    │             │
│  │ aria2:6800  │    │ aria2:6800  │    │ aria2:6800  │             │
│  │ data:8766   │    │ data:8766   │    │ data:8766   │             │
│  │ SOCKS:1055  │    │ SOCKS:1055  │    │ SOCKS:1055  │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. P2P Orchestrator (`scripts/p2p_orchestrator.py`)

- Runs on every node (port 8770)
- Leader election via Bully algorithm
- Job distribution and monitoring
- Heartbeat-based failure detection

### 2. Tailscale SOCKS5 Proxy

- Userspace networking for containers without CAP_NET_ADMIN
- SOCKS5 server on `localhost:1055`
- Enables mesh connectivity through NAT/firewalls
- P2P uses via `RINGRIFT_SOCKS_PROXY=socks5://localhost:1055`

### 3. aria2 Data Server

- RPC server on port 6800 for programmatic downloads
- HTTP data server on port 8766 for file serving
- 16 parallel connections per source
- Multi-source metalink support

### 4. Cloudflare Tunnels (Optional)

- Quick tunnels for NAT bypass
- No open ports required
- URLs change on restart (use named tunnels for persistence)

## Setup Scripts

### Primary Script: `scripts/vast_p2p_setup.py`

```bash
# Check status of all Vast instances
python scripts/vast_p2p_setup.py --check-status

# Deploy all components
python scripts/vast_p2p_setup.py --deploy-to-vast --components tailscale aria2 p2p

# Deploy specific components
python scripts/vast_p2p_setup.py --deploy-to-vast --components aria2

# Add Cloudflare tunnels (optional)
python scripts/vast_p2p_setup.py --deploy-to-vast --components cloudflare
```

### Lifecycle Manager: `scripts/vast_lifecycle.py`

```bash
# Health check with P2P status
python scripts/vast_lifecycle.py --check

# Deploy P2P orchestrator
python scripts/vast_lifecycle.py --deploy-p2p

# Update distributed_hosts.yaml
python scripts/vast_lifecycle.py --update-config

# Start jobs on idle instances
python scripts/vast_lifecycle.py --start-jobs

# Full automation cycle
python scripts/vast_lifecycle.py --auto
```

## Configuration

### distributed_hosts.yaml

Vast instances are auto-discovered and added to `config/distributed_hosts.yaml`:

```yaml
hosts:
  vast-28928169:
    ssh_host: ssh5.vast.ai
    ssh_port: 18168
    ssh_user: root
    ssh_key: ~/.ssh/id_cluster
    ringrift_path: ~/ringrift/ai-service
    memory_gb: 773
    cpus: 512
    gpu: 8x RTX 5090
    role: nn_training_primary
    status: ready
    vast_instance_id: '28928169'
    tailscale_ip: 100.x.x.x # If available
```

### GPU to Role Mapping

| GPU                       | VRAM    | Role                | Board Type |
| ------------------------- | ------- | ------------------- | ---------- |
| RTX 3070, 2060S, 3060 Ti  | ≤8GB    | gpu_selfplay        | hex8       |
| RTX 4060 Ti, 4080S, 5080  | 12-16GB | gpu_selfplay        | square8    |
| RTX 5070, 5090, A40, H100 | 24GB+   | nn_training_primary | hexagonal  |

## Using aria2 for Parallel Downloads

### Via aria2_transport.py

```python
from app.distributed.aria2_transport import Aria2Transport, Aria2Config

transport = Aria2Transport(Aria2Config(
    connections_per_server=16,
    split=16,
    max_concurrent_downloads=5,
))

# Sync from multiple sources
result = await transport.sync_from_sources(
    sources=["http://vast-28928169:8766", "http://vast-28918742:8766"],
    local_dir=Path("data/models"),
    patterns=["*.pth"],
)
```

### Via aria2c CLI

```bash
# Download from multiple sources with 16 connections each
aria2c --max-connection-per-server=16 --split=16 \
    http://vast-28928169:8766/models/latest.pth \
    http://vast-28918742:8766/models/latest.pth
```

## Environment Variables

| Variable               | Description                    | Default       |
| ---------------------- | ------------------------------ | ------------- |
| `RINGRIFT_SOCKS_PROXY` | SOCKS5 proxy URL               | (none)        |
| `RINGRIFT_P2P_VERBOSE` | Enable verbose logging         | false         |
| `RINGRIFT_P2P_VOTERS`  | Comma-separated voter node IDs | (from config) |

## Monitoring

### Check P2P Health

```bash
# From any node
curl http://localhost:8770/health

# Get cluster status
curl http://localhost:8770/cluster/status
```

### Check aria2 Status

```bash
# Via JSON-RPC
curl http://localhost:6800/jsonrpc \
    -d '{"jsonrpc":"2.0","method":"aria2.getGlobalStat","id":1}'
```

## Troubleshooting

### P2P Won't Start

1. Check Python imports: `python -c "import scripts.p2p.types"`
2. Check venv: `source venv/bin/activate`
3. Check logs: `cat logs/p2p_orchestrator.log`

### Tailscale SOCKS Not Working

1. Check tailscaled is running: `pgrep tailscaled`
2. Test SOCKS: `curl --socks5 localhost:1055 http://100.107.168.125:8770/health`
3. Check auth: `tailscale status`

### aria2 Not Responding

1. Check process: `pgrep aria2c`
2. Test RPC: `curl http://localhost:6800/jsonrpc -d '{"jsonrpc":"2.0","method":"aria2.getVersion","id":1}'`
3. Check data server: `curl http://localhost:8766/`

## Related Files

- `scripts/vast_p2p_setup.py` - Unified setup for SOCKS, aria2, P2P
- `scripts/vast_lifecycle.py` - Instance lifecycle management
- `scripts/p2p_orchestrator.py` - Main P2P orchestrator
- `app/distributed/aria2_transport.py` - aria2 transport layer
- `scripts/setup_cloudflare_tunnel.sh` - Cloudflare tunnel setup
- `config/distributed_hosts.yaml` - Host configuration
