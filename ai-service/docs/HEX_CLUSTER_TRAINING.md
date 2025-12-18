# Hex Board Cluster Training Guide

This document covers training RingRift AI models for hexagonal board variants on the distributed GPU cluster.

## Quick Reference

```bash
# Start hex selfplay on worker nodes
python scripts/run_gpu_selfplay.py --board hexagonal --num-players 2

# Start hex8 training pipeline (monitors + trains)
python scripts/hex8_training_pipeline.py --monitor

# Train immediately from existing data
python scripts/hex8_training_pipeline.py --train

# Run unified loop with hex configuration
python scripts/unified_ai_loop.py --start --config config/hex_loop.yaml

# Run hex tournament
python scripts/run_tournament.py --board hexagonal --num-players 2 --games 100
```

## Board Configurations

RingRift supports multiple hexagonal board sizes:

| Board Type | Size | Total Spaces | Line Length | Rings/Player (2p) |
| ---------- | ---- | ------------ | ----------- | ----------------- |
| hexagonal  | 11   | 469          | 4           | 96                |
| hex8       | 8    | 217          | 4           | 44                |

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hex Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Worker Nodes              Primary Node              You         │
│  ┌──────────┐             ┌──────────┐         ┌──────────┐     │
│  │ GPU 1    │─selfplay───>│          │         │          │     │
│  │ GPU 2    │─selfplay───>│  Data    │─sync───>│ Monitor  │     │
│  │ GPU N    │─selfplay───>│  Sync    │         │ Dashboard │     │
│  └──────────┘             │          │         └──────────┘     │
│                           │    ↓     │                          │
│                           │ Training │                          │
│                           │    ↓     │                          │
│                           │ Evaluate │                          │
│                           │    ↓     │                          │
│                           │ Promote  │                          │
│                           └──────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Configure Hex Training

### Create hex configuration file

```yaml
# config/hex_loop.yaml
board_type: hexagonal
num_players: 2

training:
  batch_size: 256
  learning_rate: 0.001
  epochs: 50
  checkpoint_interval: 5
  min_games_threshold: 1000

selfplay:
  mcts_simulations: 800
  temperature: 1.0
  temperature_threshold: 30 # moves before temp drops
  parallel_games: 64

evaluation:
  gauntlet_games: 50
  elo_threshold: 30 # Elo improvement needed for promotion
```

### Set environment variables

```bash
# config/hex.env
export RINGRIFT_BOARD_TYPE=hexagonal
export RINGRIFT_NUM_PLAYERS=2
export RINGRIFT_MCTS_SIMS=800
export RINGRIFT_BATCH_SIZE=256
```

## Step 2: Start Selfplay on Worker Nodes

SSH into each GPU worker node and start hex selfplay:

```bash
# Lambda node with H100
ssh ubuntu@lambda-h100

cd /home/ubuntu/ringrift/ai-service
source venv/bin/activate

# Start GPU-accelerated hex selfplay
python scripts/run_gpu_selfplay.py \
    --board hexagonal \
    --num-players 2 \
    --parallel-games 128 \
    --mcts-sims 800 \
    --output-dir data/games/hex \
    --continuous
```

### Multi-GPU selfplay

For nodes with multiple GPUs:

```bash
# Distribute across all GPUs
python scripts/run_multi_gpu_selfplay.py \
    --board hexagonal \
    --num-players 2 \
    --games-per-gpu 64
```

### Vast.ai node setup

```bash
ssh -p 14364 root@ssh5.vast.ai

# Install if needed
cd /workspace
git clone https://github.com/an0mium/RingRift.git
pip install -r RingRift/ai-service/requirements.txt

# Start selfplay
cd RingRift/ai-service
python scripts/run_gpu_selfplay.py \
    --board hexagonal \
    --num-players 2 \
    --parallel-games 32 \
    --continuous
```

## Step 3: Monitor Data Collection

### Via hex8_training_pipeline.py

```bash
# On primary training node
python scripts/hex8_training_pipeline.py --monitor

# Output:
# [Pipeline] Collecting from 8 remote databases...
# [Pipeline] Total hex games: 15,234 (need 1,000 for training)
# [Pipeline] Training triggered with 15,234 games
```

### Via unified_ai_loop.py

```bash
python scripts/unified_ai_loop.py \
    --foreground \
    --verbose \
    --config config/hex_loop.yaml
```

### Manual data collection

```bash
# Collect data from all nodes
python scripts/hex8_training_pipeline.py --collect

# Check data status
python -c "
from pathlib import Path
import sqlite3

db = Path('data/games/hex_consolidated.db')
if db.exists():
    conn = sqlite3.connect(db)
    count = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
    print(f'Hex games collected: {count}')
"
```

## Step 4: Train Hex Model

### Automatic training (via pipeline)

```bash
# Pipeline auto-triggers when threshold met
python scripts/hex8_training_pipeline.py --monitor
```

### Manual training

```bash
# Export to NPZ format
python scripts/export_replay_dataset.py \
    --db data/games/hex_consolidated.db \
    --output data/training/hex_training.npz \
    --board hexagonal

# Train HexNeuralNet
python scripts/run_nn_training_baseline.py \
    --board hexagonal \
    --num-players 2 \
    --data data/training/hex_training.npz \
    --epochs 50 \
    --batch-size 256 \
    --run-dir runs/hex_$(date +%Y%m%d)
```

### Distributed training (multi-GPU)

```bash
# On multi-GPU node
torchrun --nproc_per_node=4 \
    scripts/run_nn_training_baseline.py \
    --board hexagonal \
    --distributed \
    --data data/training/hex_training.npz
```

## Step 5: Evaluate and Promote

### Run hex gauntlet

```bash
python scripts/run_gauntlet.py \
    --candidate models/hex/checkpoint_latest.pt \
    --baseline models/hex/production.pt \
    --board hexagonal \
    --games 100
```

### Two-stage gauntlet (efficient evaluation)

```bash
# Stage 1: Quick screen (10 games)
# Stage 2: Deep evaluation (50 games) if Stage 1 passes
python scripts/two_stage_gauntlet.py \
    --candidate models/hex/checkpoint_latest.pt \
    --board hexagonal
```

### Auto-promotion

```bash
# Via unified loop (handles automatically)
python scripts/unified_ai_loop.py --start

# Manual promotion
python scripts/auto_model_promotion.py \
    --model models/hex/checkpoint_latest.pt \
    --board hexagonal \
    --elo-threshold 30
```

## Step 6: Deploy to Production

```bash
# Sync promoted model to production
python scripts/sync_models.py \
    --source models/hex/promoted_latest.pt \
    --destination /var/www/ringrift/models/hex_production.pt \
    --board hexagonal

# Verify deployment
curl -X POST http://localhost:3001/api/ai/inference \
    -H "Content-Type: application/json" \
    -d '{"board_type": "hexagonal", "state": {...}}'
```

## Common Commands Reference

### Selfplay

```bash
# Basic hex selfplay
python scripts/run_gpu_selfplay.py --board hexagonal --num-players 2

# With custom MCTS settings
python scripts/run_gpu_selfplay.py \
    --board hexagonal \
    --mcts-sims 1600 \
    --temperature 0.8 \
    --parallel-games 64

# Continuous selfplay (recommended for cluster)
python scripts/run_gpu_selfplay.py --board hexagonal --continuous

# 3-player hex (experimental)
python scripts/run_gpu_selfplay.py --board hexagonal --num-players 3
```

### Training

```bash
# Standard training
python scripts/run_nn_training_baseline.py --board hexagonal

# With hex augmentation (D6 symmetry)
python scripts/run_nn_training_baseline.py \
    --board hexagonal \
    --augmentation hex_symmetry

# Transfer learning from square model
python scripts/transfer_learning.py \
    --source models/square8/production.pt \
    --target-board hexagonal
```

### Evaluation

```bash
# Elo tournament
python scripts/run_tournament.py --board hexagonal --games 200

# Specific model comparison
python scripts/run_tournament.py \
    --board hexagonal \
    --model-a models/hex/v1.pt \
    --model-b models/hex/v2.pt \
    --games 100
```

### Monitoring

```bash
# Check training progress
python scripts/training_monitor.py --board hexagonal

# View selfplay stats
python scripts/analyze_game_statistics.py \
    --db data/games/hex_selfplay.db

# Dashboard
python scripts/dashboard_server.py --port 8080
# Open http://localhost:8080
```

## Troubleshooting

### "No hex games found"

```bash
# Check if selfplay is running
ps aux | grep "gpu_selfplay.*hex"

# Check database location
ls -la data/games/hex*.db

# Verify board type in selfplay
python scripts/run_gpu_selfplay.py --board hexagonal --dry-run
```

### Training loss not decreasing

```bash
# Validate training data
python -m app.training.data_validation --file data/training/hex_training.npz

# Check for data issues
python scripts/analyze_game_statistics.py \
    --db data/games/hex_consolidated.db \
    --check-quality
```

### Out of memory during training

```bash
# Reduce batch size
python scripts/run_nn_training_baseline.py \
    --board hexagonal \
    --batch-size 128  # Default is 256

# Enable gradient checkpointing
python scripts/run_nn_training_baseline.py \
    --board hexagonal \
    --gradient-checkpointing

# Use smaller model variant
python scripts/run_nn_training_baseline.py \
    --board hexagonal \
    --model HexNeuralNet_v2  # vs v3
```

### Slow selfplay on hex

```bash
# Hex has more positions - reduce MCTS sims
python scripts/run_gpu_selfplay.py \
    --board hexagonal \
    --mcts-sims 400  # vs 800 default

# Use more parallel games (better GPU utilization)
python scripts/run_gpu_selfplay.py \
    --board hexagonal \
    --parallel-games 128
```

## Configuration Files

### config/hex_loop.yaml (full example)

```yaml
board_type: hexagonal
num_players: 2

data_ingestion:
  sync_interval_seconds: 60
  min_games_threshold: 1000
  consolidation_interval: 300

training:
  batch_size: 256
  learning_rate: 0.001
  lr_warmup_epochs: 5
  epochs: 50
  checkpoint_interval: 5
  early_stopping_patience: 10
  validation_split: 0.1
  auto_tune_batch_size: true

selfplay:
  mcts_simulations: 800
  temperature: 1.0
  temperature_threshold: 30
  parallel_games: 64
  cpuct: 2.5
  noise_alpha: 0.3
  noise_weight: 0.25

evaluation:
  stage1_games: 10
  stage2_games: 50
  elo_threshold: 30
  pass_rate_threshold: 0.55

promotion:
  auto_promote: true
  backup_before_promote: true
  max_models_to_keep: 10

curriculum:
  enabled: true
  elo_weighted: true
  recency_weight: 0.3
```

### config/distributed_hosts.yaml

```yaml
hosts:
  lambda-h100:
    status: active
    tailscale_ip: 100.78.101.123
    ssh_user: ubuntu
    db_path: /home/ubuntu/ringrift/ai-service/data/games/hex_selfplay.db
    role: training

  lambda-gh200-a:
    status: active
    tailscale_ip: 100.88.176.74
    ssh_user: ubuntu
    db_path: /home/ubuntu/ringrift/ai-service/data/games/hex_selfplay.db
    role: selfplay

  vast-3070:
    status: active
    ssh_host: ssh5.vast.ai
    ssh_port: 14364
    ssh_user: root
    db_path: /workspace/RingRift/ai-service/data/games/hex_selfplay.db
    role: selfplay
```

## Metrics and Monitoring

### Prometheus metrics

```bash
# Export hex training metrics
python scripts/elo_metrics_exporter.py --board hexagonal --port 9090
```

Key metrics:

- `ringrift_hex_games_total`: Total hex selfplay games
- `ringrift_hex_training_loss`: Current training loss
- `ringrift_hex_model_elo`: Production model Elo
- `ringrift_hex_selfplay_games_per_hour`: Selfplay throughput

### Grafana dashboards

Import `monitoring/grafana/dashboards/hex-training.json` for:

- Selfplay throughput by node
- Training loss curves
- Model Elo progression
- Data quality metrics
