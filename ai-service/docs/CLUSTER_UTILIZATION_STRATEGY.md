# Cluster Utilization Strategy

**Date**: December 26, 2025
**Purpose**: Maximize AI training efficiency through optimal resource allocation

## Current Cluster Inventory

### Active Nodes (21 total)

| Provider | Count | GPUs                           | Total VRAM | Role                        |
| -------- | ----- | ------------------------------ | ---------- | --------------------------- |
| Vast.ai  | 12    | Mixed (RTX 3060 - 8x RTX 4090) | ~600GB     | Selfplay + Training         |
| RunPod   | 4     | H100, 2x A100, L40S            | ~288GB     | Primary Training + Selfplay |
| Nebius   | 1     | L40S                           | 48GB       | Backbone P2P Voter          |
| Vultr    | 1     | A100 (20GB vGPU)               | 20GB       | Selfplay                    |
| Local    | 1     | M3 Max/Ultra (MPS)             | 64GB       | Coordinator                 |

### GPU Tier Classification

**Tier 1 - Training Primary** (80GB+ VRAM):

- runpod-h100 (80GB H100)
- runpod-a100-1 (80GB A100)
- runpod-a100-2 (80GB A100)
- vast-29129529 (8x RTX 4090, 192GB)
- vast-29118471 (8x RTX 3090, 192GB)

**Tier 2 - Training/Selfplay Hybrid** (32-64GB VRAM):

- vast-29128352 (2x RTX 5090, 64GB)
- vast-28925166 (RTX 5090, 32GB)
- vast-29128356 (RTX 5090, 32GB)
- runpod-l40s-2 (L40S, 48GB)
- nebius-backbone-1 (L40S, 48GB)
- vast-28918742 (A40, 46GB)

**Tier 3 - Selfplay Only** (8-24GB VRAM):

- vast-29031159 (RTX 5080, 16GB)
- vast-29126088 (RTX 4060 Ti, 16GB)
- vast-29031161 (RTX 3060, 12GB)
- vast-28890015 (RTX 2080 Ti, 11GB)
- vast-28889766 (RTX 3060 Ti, 8GB)
- vast-29046315 (RTX 3060 Ti, 8GB)
- vultr-a100-20gb (A100 vGPU, 20GB)

---

## Resource Allocation Strategy

### 1. Board-to-GPU Matching

| Board Type            | Min VRAM | Recommended Tier | Batch Size |
| --------------------- | -------- | ---------------- | ---------- |
| hex8 (61 cells)       | 8GB      | Tier 3           | 256-512    |
| square8 (64 cells)    | 8GB      | Tier 3           | 256-512    |
| square19 (361 cells)  | 16GB     | Tier 2-3         | 128-256    |
| hexagonal (469 cells) | 32GB     | Tier 1-2         | 64-128     |

### 2. Job Type Allocation

**Training Jobs**:

- Primary: Tier 1 nodes (H100, A100, multi-GPU)
- Fallback: Tier 2 nodes when Tier 1 busy
- Max 1 training job per config at a time

**Selfplay Jobs**:

- Gumbel MCTS (high quality): Tier 1-2 nodes
- Heuristic (fast bootstrap): Tier 3 nodes
- GPU parallel games: Any GPU node

**Evaluation/Gauntlet**:

- Run on Tier 2 nodes to avoid interrupting training
- Can preempt selfplay if needed

### 3. Quota-Based Fair Allocation

```yaml
config_quotas:
  # Large boards get more time on big GPUs
  hexagonal_4p: 0.25 # 25% of Tier 1 time
  hexagonal_3p: 0.15
  hexagonal_2p: 0.10
  square19_4p: 0.15
  square19_3p: 0.10
  square19_2p: 0.05
  # Small boards primarily use Tier 3
  square8_*: 0.10
  hex8_*: 0.10
```

---

## Automation Components

### 1. SelfplayScheduler (`app/coordination/selfplay_scheduler.py`)

- Priority-based config selection
- Elo velocity weighting
- Curriculum weight integration
- Quality score gating

### 2. IdleResourceDaemon (`app/coordination/idle_resource_daemon.py`)

- Detects idle GPUs (>10 min idle threshold)
- Spawns appropriate selfplay based on GPU tier
- Emits IDLE_RESOURCE_DETECTED events

### 3. QueuePopulator (`app/coordination/queue_populator.py`)

- Maintains work queue until Elo targets met
- Target distribution: 60% selfplay, 30% training, 10% tournament
- Priority boosting for starved configs

### 4. UtilizationOptimizer (`app/coordination/utilization_optimizer.py`)

- Matches GPU capabilities to board sizes
- Cost-aware scheduling for multi-provider
- Ephemeral node handling (shorter jobs for Vast.ai)

---

## Data Flow Optimization

### 1. Sync Routing

```
Generator Node → Coordinator → Replication Targets

Priority Sync Triggers:
- New canonical games (immediate)
- Training data changes (5 min)
- Model updates (immediate)
```

### 2. Ephemeral Node Protection

Vast.ai nodes can be terminated without warning:

```python
# EphemeralSyncDaemon runs every 5 seconds
# Syncs incrementally to prevent data loss
ephemeral_sync_interval = 5  # seconds
min_games_to_sync = 1        # sync even single games
```

### 3. NPZ Export Pipeline

```
Selfplay Complete → Quality Gate → Export to NPZ → Training Ready

Triggers:
- SELFPLAY_COMPLETE event
- Quality score > 0.7
- Min games threshold (1000+)
```

---

## Monitoring & Alerting

### Key Metrics

| Metric               | Target               | Alert Threshold       |
| -------------------- | -------------------- | --------------------- |
| GPU Utilization      | >80%                 | <50% for 30 min       |
| Selfplay throughput  | 100+ games/hour/node | <20 games/hour        |
| Training queue depth | <5 jobs              | >20 jobs              |
| Elo velocity         | >5 Elo/day           | <1 Elo/day for 3 days |

### Dashboard Endpoints

```bash
# Cluster status
curl http://localhost:8770/status

# Queue status
curl http://localhost:8765/queue/status

# Daemon health
curl http://localhost:8765/daemons/health
```

---

## Long-Term Optimization

### 1. Predictive Allocation

Track job duration patterns:

```python
# HistoricalJobTracker stores:
# - config -> avg_duration
# - config -> variance
# Use for smarter scheduling
```

### 2. Cost Optimization

```yaml
provider_preferences:
  training:
    prefer: [runpod, nebius] # Stable, good perf
    avoid: [vast] # Ephemeral risk
  selfplay:
    prefer: [vast] # Cost effective
    fallback: [runpod, vultr]
```

### 3. Elastic Scaling

When additional GH200 nodes return:

- Add 19x GH200 (96GB each) = +1.8TB VRAM
- Shift training primary to the expanded GH200 fleet
- Use current nodes for increased selfplay

---

## Implementation Checklist

- [x] SelfplayScheduler with priority allocation
- [x] IdleResourceDaemon for GPU utilization
- [x] EphemeralSyncDaemon for data protection
- [x] QueuePopulator for work distribution
- [ ] Config quota enforcement
- [ ] Starvation prevention (priority boost after 4h)
- [ ] Cost-aware provider selection
- [ ] Predictive job duration tracking

---

## Quick Commands

```bash
# Check cluster utilization
python -m app.distributed.cluster_monitor --watch

# View queue status
python scripts/queue_status.py

# Force selfplay on idle node
python scripts/spawn_selfplay.py --host vast-29129529 --config hex8_2p

# Check daemon status
python scripts/launch_daemons.py --status
```
