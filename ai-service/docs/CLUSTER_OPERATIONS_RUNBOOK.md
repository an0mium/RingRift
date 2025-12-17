# RingRift Cluster Operations Runbook

Quick reference for daily cluster operations and troubleshooting.

## Cluster Overview

| Host               | Role               | GPU     | Notes                |
| ------------------ | ------------------ | ------- | -------------------- |
| lambda-gh200-a     | Primary, Training  | GH200   | Main coordinator     |
| lambda-gh200-{b-l} | Workers            | GH200   | Selfplay workers     |
| lambda-2xh100      | Training + Workers | 2x H100 | High-throughput node |

## Quick Status Check

### 1. Check All Nodes

```bash
# From local machine
./ai-service/scripts/cluster_status.sh

# Or manually
for host in lambda-gh200-{a,b,c,d,e,g,h,i,k,l} lambda-2xh100; do
    echo -n "$host: "
    ssh -o ConnectTimeout=3 $host 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader' 2>/dev/null || echo "offline"
done
```

### 2. Check Running Services

```bash
ssh lambda-gh200-a 'systemctl status ringrift-*'
```

### 3. Check Worker Count

```bash
ssh lambda-gh200-a 'ps aux | grep -E "(selfplay|train)" | grep -v grep | wc -l'
```

## Common Operations

### Start/Stop GPU Selfplay Workers

```bash
# Start on a node
ssh lambda-gh200-b 'cd ~/ringrift/ai-service && \
  nohup venv/bin/python -m scripts.run_gpu_selfplay \
    --board square8 --num-players 2 --num-games 1000 \
    --output-dir /tmp/gpu_selfplay > /tmp/gpu_selfplay.log 2>&1 &'

# Stop workers on a node
ssh lambda-gh200-b 'pkill -f run_gpu_selfplay'
```

### Trigger Training

```bash
ssh lambda-gh200-a 'cd ~/ringrift/ai-service && \
  venv/bin/python -m scripts.train_nnue \
    --db data/games/jsonl_aggregated.db \
    --board-type square8 --num-players 2 \
    --epochs 50 --batch-size 2048 \
    --save-path models/nnue/square8_2p_new.pt'
```

### Run Model Tournament

```bash
ssh lambda-gh200-a 'cd ~/ringrift/ai-service && \
  venv/bin/python scripts/run_model_elo_tournament.py \
    --board square8 --players 2 --games 50 \
    --include-nnue --run'
```

### Sync Models Across Cluster

```bash
ssh lambda-gh200-a 'cd ~/ringrift/ai-service && \
  venv/bin/python scripts/sync_models.py --sync'
```

## Troubleshooting

### Low GPU Utilization

1. Check if workers are running:

   ```bash
   ssh $HOST 'ps aux | grep selfplay | grep -v grep'
   ```

2. Check for errors in logs:

   ```bash
   ssh $HOST 'tail -50 /tmp/gpu_selfplay.log'
   ```

3. Restart workers if needed:
   ```bash
   ssh $HOST 'pkill -f selfplay; sleep 5'
   # Then start new workers
   ```

### Node Unreachable

1. Check Tailscale:

   ```bash
   tailscale status
   ```

2. Try direct SSH:

   ```bash
   ssh -v $HOST
   ```

3. If still unreachable, contact cloud provider.

### Disk Full

1. Check usage:

   ```bash
   ssh $HOST 'df -h'
   ```

2. Find large files:

   ```bash
   ssh $HOST 'du -sh ~/ringrift/ai-service/data/*'
   ```

3. Clean up old data:
   ```bash
   ssh $HOST 'find ~/ringrift/ai-service/data/selfplay -name "*.jsonl" -mtime +7 -delete'
   ```

### Training Stuck

1. Check training log:

   ```bash
   ssh lambda-gh200-a 'tail -100 /tmp/train_*.log'
   ```

2. Check GPU memory:

   ```bash
   ssh lambda-gh200-a 'nvidia-smi'
   ```

3. Kill and restart if needed:
   ```bash
   ssh lambda-gh200-a 'pkill -f train_nnue'
   ```

## Data Management

### Aggregate JSONL to Database

```bash
ssh lambda-gh200-a 'cd ~/ringrift/ai-service && \
  venv/bin/python scripts/aggregate_jsonl_to_db.py --run'
```

### Check Holdout Validation Set

```bash
# View holdout stats
cd ai-service
PYTHONPATH=. python scripts/holdout_validation.py --stats

# Evaluate model on holdout
PYTHONPATH=. python scripts/holdout_validation.py \
  --evaluate --model models/nnue/square8_2p.pt

# Check for overfitting
PYTHONPATH=. python scripts/holdout_validation.py --check-overfitting
```

### Backup to External Drive

```bash
# Backup models
rsync -avz ai-service/models/nnue/ /Volumes/RingRift-Data/model_backups/

# Backup training database
rsync -avz ai-service/data/games/merged_training.db /Volumes/RingRift-Data/db_backups/
```

## Alerting

### Test Cluster Alerting

```bash
cd ai-service
./scripts/cluster_alert.sh
```

### Run as Daemon

```bash
RINGRIFT_WEBHOOK_URL="https://your-webhook-url" \
./scripts/cluster_alert.sh --cron
```

## Logs

| Log         | Location                  | Description            |
| ----------- | ------------------------- | ---------------------- |
| Training    | `/tmp/train_*.log`        | NNUE training progress |
| Tournament  | `/tmp/elo_tournament.log` | Elo evaluation results |
| Selfplay    | `/tmp/gpu_selfplay.log`   | Worker output          |
| Aggregation | `/tmp/aggregation.log`    | Data aggregation       |

## Emergency Procedures

### Stop All Cluster Activity

```bash
for host in lambda-gh200-{a,b,c,d,e,g,h,i,k,l} lambda-2xh100; do
    echo "Stopping $host..."
    ssh $host 'pkill -f ringrift' 2>/dev/null || true
done
```

### Recovery After Outage

1. Check which nodes are back online:

   ```bash
   ./scripts/cluster_status.sh
   ```

2. Restart services on primary:

   ```bash
   ssh lambda-gh200-a 'sudo systemctl restart ringrift-*'
   ```

3. Start workers on all nodes:
   ```bash
   ssh lambda-gh200-a 'cd ~/ringrift/ai-service && venv/bin/python scripts/p2p_orchestrator.py --start'
   ```

## Key Thresholds

| Metric          | Warning | Critical |
| --------------- | ------- | -------- |
| GPU Utilization | < 50%   | < 20%    |
| Disk Usage      | > 70%   | > 85%    |
| Worker Count    | < 5     | < 2      |
| Training Loss   | > 1.0   | > 2.0    |
| Holdout Gap     | > 0.10  | > 0.15   |

## Contacts

For infrastructure issues, check cloud provider status page first.
