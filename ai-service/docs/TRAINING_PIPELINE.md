# RingRift AI Training Pipeline

This document describes the self-improvement training loop architecture and operational procedures.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED AI TRAINING LOOP                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  GH200 x12   │    │  Vast.ai x4  │    │  Lambda H100 │    │ Lambda A10 │ │
│  │  (Selfplay)  │    │  (Selfplay)  │    │  (Training)  │    │ (Backup)   │ │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └────────────┘ │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │  JSONL Files  │                                       │
│                     │  (Raw Data)   │                                       │
│                     └───────┬───────┘                                       │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │  SQLite DBs   │                                       │
│                     │ (Aggregated)  │                                       │
│                     └───────┬───────┘                                       │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │   NPZ Files   │                                       │
│                     │  (Training)   │                                       │
│                     └───────┬───────┘                                       │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │   GPU Train   │                                       │
│                     │   (H100)      │                                       │
│                     └───────┬───────┘                                       │
│                             │                                               │
│                     ┌───────▼───────┐                                       │
│                     │  Elo Eval &   │                                       │
│                     │  Promotion    │                                       │
│                     └───────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Selfplay Generation

**Hosts**: GH200 cluster (12 nodes), Vast.ai instances (4 nodes)

**Output**: JSONL files in `data/selfplay/` directories

**Process**:

- Workers generate games via `run_gpu_selfplay.py` or `run_hybrid_selfplay.py`
- Games are written to JSONL format for efficiency
- Each worker writes to its own subdirectory

### 2. Data Synchronization

**Cron Job**: Every 15 minutes

```bash
# Sync from vast hosts via P2P or unified sync
python scripts/unified_data_sync.py --source vast --dest local
```

**Script**: `unified_data_sync.py` (supersedes deprecated `sync_vast_jsonl.sh`)

### 3. JSONL to SQLite Aggregation

**Cron Job**: Every 30 minutes

```bash
python3 scripts/aggregate_jsonl_to_db.py \
    --input-dir data/selfplay \
    --output-db data/games/cluster_merged.db
```

**Output**: `data/games/cluster_merged.db` (or config-specific DBs like `hex8_2p.db`)

### 4. NPZ Export

**Triggered by**: Unified loop when training threshold reached

```bash
python3 scripts/export_replay_dataset.py \
    --db data/games/all_jsonl_training.db \
    --output data/training/unified_{config}.npz
```

### 5. Model Training

**Host**: Lambda H100

**Script**: `scripts/unified_ai_loop.py`

**Config**: `config/unified_loop.yaml`

```yaml
training:
  trigger_threshold_games: 300
  min_interval_seconds: 1200
```

### 6. Evaluation & Promotion

**Requirement**: +20 Elo vs current best model

**Process**:

1. Shadow tournament against current best
2. If Elo gain >= threshold, promote to production
3. Distribute to all selfplay workers

## Key Configuration Files

| File                                        | Purpose                    |
| ------------------------------------------- | -------------------------- |
| `config/unified_loop.yaml`                  | Main loop configuration    |
| `config/remote_hosts.yaml`                  | SSH host definitions       |
| `config/distributed_hosts.yaml`             | Distributed training hosts |
| `logs/unified_loop/unified_loop_state.json` | Loop state persistence     |

## Monitoring

### Training Monitor

```bash
python3 scripts/training_monitor.py --verbose
```

### Database Health Check

```bash
python3 scripts/db_health_check.py
```

### GPU Status

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

### Process Status

```bash
ps aux | grep -E "unified_ai_loop|selfplay|training"
```

## Troubleshooting

### Training Not Triggering

1. Check `training_in_progress` flag:

   ```bash
   cat logs/unified_loop/unified_loop_state.json | grep training_in_progress
   ```

2. Reset if stuck:
   ```python
   import json
   with open("logs/unified_loop/unified_loop_state.json", "r+") as f:
       state = json.load(f)
       state["training_in_progress"] = False
       f.seek(0)
       json.dump(state, f, indent=2)
       f.truncate()
   ```

### Corrupted Timestamps

Check for unrealistic "hours since promotion":

```bash
grep "since last promotion" logs/unified_*.log
```

Fix by resetting timestamps:

```python
import json, time
with open("logs/unified_loop/unified_loop_state.json", "r+") as f:
    state = json.load(f)
    now = time.time()
    for config in state["configs"].values():
        if config["last_promotion_time"] == 0:
            config["last_promotion_time"] = now
    f.seek(0)
    json.dump(state, f, indent=2)
    f.truncate()
```

### SSH Key Issues

Vast hosts require SSH key configuration on H100:

```bash
# Add to ~/.ssh/config on H100
Host 100.118.201.85
    User root
    IdentityFile ~/.ssh/id_cluster
    StrictHostKeyChecking no
```

### Database Corruption

Run health check:

```bash
python3 scripts/db_health_check.py --repair --quarantine
```

## Cron Jobs (H100)

```cron
# Sync JSONL from vast hosts
*/15 * * * * ~/sync_vast_jsonl.sh >> ~/ringrift/ai-service/logs/vast_sync.log 2>&1

# Aggregate JSONL to SQLite
*/30 * * * * cd ~/ringrift/ai-service && python3 scripts/aggregate_jsonl_to_db.py --input-dir data/selfplay --output-db data/games/all_jsonl_training.db >> logs/aggregate_jsonl.log 2>&1

# Training monitor (hourly)
0 * * * * cd ~/ringrift/ai-service && python3 scripts/training_monitor.py --log >> logs/training_monitor.log 2>&1
```

## Metrics

Prometheus metrics available at: `http://H100_IP:9090/metrics`

Key metrics:

- `ringrift_training_runs_total`
- `ringrift_promotions_total`
- `ringrift_games_synced_total`
- `ringrift_gpu_utilization`

## Operational Procedures

### Starting the Training Loop

```bash
cd ~/ringrift/ai-service
source venv/bin/activate
python scripts/unified_ai_loop.py --start
```

### Stopping the Training Loop

```bash
python scripts/unified_ai_loop.py --stop
```

### Checking Status

```bash
python scripts/unified_ai_loop.py --status
```

### Emergency Halt

```bash
python scripts/unified_ai_loop.py --halt
```

### Resume After Halt

```bash
python scripts/unified_ai_loop.py --resume
```

## File Locations

| Path                         | Contents                  |
| ---------------------------- | ------------------------- |
| `data/selfplay/`             | Raw JSONL selfplay data   |
| `data/games/`                | SQLite training databases |
| `data/training/`             | NPZ training files        |
| `models/`                    | Trained model checkpoints |
| `models/ringrift_best_*.pth` | Production models         |
| `logs/unified_loop/`         | Loop logs and state       |

## Advanced Training Features (2025-12)

The training pipeline includes several advanced features for improved model quality.

### Policy Label Smoothing

Prevents overconfident predictions by mixing targets with uniform distribution:

```bash
python -m app.training.train \
  --data-path data/training/dataset.npz \
  --board-type hex8 \
  --policy-label-smoothing 0.05
```

**Configuration** (`unified_loop.yaml`):

```yaml
training:
  label_smoothing_warmup: 5 # Epochs before full smoothing
```

### Hex Board Augmentation

D6 symmetry augmentation provides 12x effective dataset size for hex boards:

```bash
python -m app.training.train \
  --data-path data/training/hex8_games.npz \
  --board-type hex8 \
  --augment-hex-symmetry
```

**Configuration**:

```yaml
training:
  use_hex_augmentation: true
```

### Soft Policy Targets (Gumbel MCTS)

Generate training data with soft policy targets from visit distributions:

```bash
python scripts/run_hybrid_selfplay.py \
  --board-type hex8 \
  --engine-mode gumbel-mcts \
  --nn-model-id ringrift_hex8_2p_v3_retrained \
  --num-games 200
```

### Training CLI Arguments

| Argument                    | Type  | Default | Description                              |
| --------------------------- | ----- | ------- | ---------------------------------------- |
| `--policy-label-smoothing`  | float | 0.0     | Label smoothing factor                   |
| `--augment-hex-symmetry`    | flag  | False   | Enable D6 hex augmentation               |
| `--warmup-epochs`           | int   | 0       | LR warmup epochs                         |
| `--lr-scheduler`            | str   | none    | none, step, cosine, cosine-warm-restarts |
| `--early-stopping-patience` | int   | 10      | Early stopping patience                  |
| `--sampling-weights`        | str   | uniform | uniform, late_game, phase_emphasis       |

### Advanced Regularization

| Feature          | Config Key             | Default | Description                 |
| ---------------- | ---------------------- | ------- | --------------------------- |
| SWA              | `use_swa`              | true    | Stochastic Weight Averaging |
| EMA              | `use_ema`              | true    | Exponential Moving Average  |
| Spectral Norm    | `use_spectral_norm`    | true    | Gradient stability          |
| Stochastic Depth | `use_stochastic_depth` | true    | Dropout regularization      |
| Focal Loss       | `focal_gamma`          | 2.0     | Hard sample mining          |

### See Also

- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Comprehensive training features reference
- [HEX_AUGMENTATION.md](HEX_AUGMENTATION.md) - D6 symmetry augmentation details
- [GUMBEL_MCTS.md](GUMBEL_MCTS.md) - Gumbel MCTS implementation

## Contact

For issues with the training pipeline, check:

1. This documentation
2. `logs/unified_ai_loop.log`
3. `logs/unified_loop/unified_loop_state.json`
