# Soft Target Training Workflow

This document describes the soft target training workflow for RingRift AI models.

## Overview

**Soft targets** are policy targets derived from MCTS visit distributions rather than single-action one-hot vectors. They improve training by:

- Providing richer supervision signal
- Reducing overfitting
- Better representing action uncertainty

**Results**: Soft targets achieve **13.8% lower validation loss** compared to hard targets.

## Workflow Steps

### 1. Generate Soft Target Data

#### Option A: GPU MCTS Selfplay (Recommended)

```bash
# On GPU cluster node
PYTHONPATH=. python3 -m app.training.gpu_mcts_selfplay \
  --board-type hex8 \
  --num-players 2 \
  --num-games 100 \
  --output data/selfplay/hex8_2p_soft.jsonl \
  --device cuda
```

#### Option B: Gumbel MCTS Selfplay

```bash
PYTHONPATH=. python3 scripts/run_gumbel_mcts_selfplay.py \
  --board-type square8 \
  --num-players 2 \
  --num-games 100 \
  --output data/selfplay/gumbel_sq8_2p.jsonl
```

### 2. Convert JSONL to NPZ

The conversion script automatically detects and extracts soft targets:

```bash
PYTHONPATH=. python3 scripts/jsonl_to_npz.py \
  --input data/selfplay/hex8_2p_soft.jsonl \
  --output data/training/hex8_2p_soft.npz \
  --board-type hex8 \
  --num-players 2 \
  --gpu-selfplay
```

**Key Features:**

- Handles numeric policy indices (e.g., `'21'`, `'111'`)
- Handles coordinate-based keys (e.g., `'place_ring_2,-4'`)
- Automatically normalizes probability distributions
- Falls back to hard targets if soft not available

### 3. Train Model

```bash
PYTHONPATH=. python3 -m app.training.train \
  --data-path data/training/hex8_2p_soft.npz \
  --board-type hex8 \
  --num-players 2 \
  --model-version v2 \
  --epochs 20 \
  --batch-size 128 \
  --save-path models/hex8_2p_soft.pt
```

### 4. Evaluate Model

```bash
PYTHONPATH=. python3 -c "
from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent
from app.models import BoardType

results = run_baseline_gauntlet(
    model_path='models/hex8_2p_soft.pt',
    board_type=BoardType.HEX8,
    opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
    games_per_opponent=20,
    num_players=2,
)
for opp, res in results.items():
    print(f'{opp}: {res[\"win_rate\"]*100:.1f}% win rate')
"
```

## Data Format

### JSONL Format (mcts_policy field)

```json
{
  "moves": [
    {
      "type": "place_ring",
      "to": { "x": 2, "y": -4 },
      "mcts_policy": {
        "21": 0.35,
        "18": 0.25,
        "111": 0.2,
        "45": 0.12,
        "67": 0.08
      }
    }
  ]
}
```

### NPZ Format (sparse policy)

```python
# policy_indices: array of arrays, variable length per sample
# policy_values: array of arrays, probabilities (sum to 1)

data = np.load('training.npz', allow_pickle=True)
print(data['policy_indices'][0])  # [21, 18, 111, 45, 67]
print(data['policy_values'][0])   # [0.35, 0.25, 0.20, 0.12, 0.08]
```

## Validation

Check soft target stats in NPZ:

```bash
PYTHONPATH=. python3 -c "
import numpy as np
data = np.load('data/training/hex8_2p_soft.npz', allow_pickle=True)
multi = sum(1 for p in data['policy_indices'] if len(p) > 1)
print(f'Samples with soft targets: {multi}/{len(data[\"features\"])}')
print(f'Avg actions per sample: {np.mean([len(p) for p in data[\"policy_indices\"]]):.2f}')
"
```

## Key Files

- `scripts/jsonl_to_npz.py`: JSONL to NPZ converter with soft target support
- `app/training/datasets.py`: Dataset loader (handles sparse policy)
- `app/training/gpu_mcts_selfplay.py`: GPU MCTS with visit distribution capture
- `app/training/train.py`: Training loop (uses soft targets automatically)

## Comparison Results (Dec 2025)

| Metric     | Soft Targets | Hard Targets | Improvement |
| ---------- | ------------ | ------------ | ----------- |
| Val Loss   | 6.42         | 7.45         | -13.8%      |
| Train Loss | 5.72         | 6.73         | -15.0%      |

## Troubleshooting

### All moves return INVALID_MOVE_INDEX

- **Cause**: mcts_policy keys are numeric indices, not coordinate strings
- **Fix**: Script now handles both formats (Dec 24, 2025 fix)

### Low soft target percentage

- **Cause**: Selfplay mode doesn't capture visit distributions
- **Fix**: Use `--gpu-selfplay` flag and ensure MCTS mode is enabled
