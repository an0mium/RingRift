# OWC Drive Data Value Assessment

**Date:** December 26, 2025
**Total Storage:** 7.3TB (5.3TB used after cleanup)

## Executive Summary

The OWC drive contains **327,000+ high-quality selfplay games** from Lambda GH200 nodes that are NOT currently being used for training. This represents a 10-100x increase in available training data compared to the sparse canonical databases.

### Critical Finding

| Data Source | Games | Status |
|-------------|-------|--------|
| canonical_games (current training) | 35,000 | In use |
| cluster_collected_backup (Lambda) | **327,000+** | **UNUSED** |
| cluster_aggregated/selfplay.db | 28,419 | Partially used |

## Current Training Data Gap

### Canonical Games (Sparse!)
| Config | Games | Quality |
|--------|-------|---------|
| hex8_2p | 20 | CRITICAL GAP |
| hex8_3p | 11,731 | OK |
| hex8_4p | 7,059 | OK |
| hexagonal_2p | 21 | CRITICAL GAP |
| hexagonal_3p | 0 | CRITICAL GAP |
| hexagonal_4p | 8 | CRITICAL GAP |
| square19_2p | 77 | CRITICAL GAP |
| square19_3p | 150 | CRITICAL GAP |
| square19_4p | 0 | CRITICAL GAP |
| square8_2p | 15,491 | OK |
| square8_3p | 494 | LOW |
| square8_4p | 0 | CRITICAL GAP |

### Lambda Backup (Untapped!)
| Node | Games | Top Config |
|------|-------|------------|
| lambda-gh200-g | 143,119 | square8_2p: 132K |
| lambda-gh200-h | 155,368 | square8_2p: ~140K |
| lambda-gh200-e | 28,838 | square8_2p: ~25K |
| **TOTAL** | **327,000+** | - |

## Recommended Actions

### Immediate: Consolidate Lambda Data

```bash
# 1. Create consolidated training database from Lambda backup
PYTHONPATH=. python3 scripts/consolidate_lambda_backup.py \
  --source /Volumes/RingRift-Data/cluster_collected_backup \
  --output /Volumes/RingRift-Data/consolidated_lambda_games.db

# 2. Export to NPZ for training
PYTHONPATH=. python scripts/export_replay_dataset.py \
  --db /Volumes/RingRift-Data/consolidated_lambda_games.db \
  --board-type square8 --num-players 2 \
  --output data/training/lambda_sq8_2p.npz
```

### Training Acceleration Strategy

1. **Merge Lambda data into canonical:**
   - Add 130K+ square8_2p games to training
   - Add 5K+ square8_4p games (currently 0!)
   - Add 3K+ square8_3p games
   - Add 2K+ hex8_2p games

2. **Expected Impact:**
   - 10x more training data for square8 configs
   - Fill critical gaps in 4-player and large board configs
   - Faster Elo improvement (more diverse positions)

3. **Quality Check:**
   - Lambda games from Dec 15-19, 2025
   - Generated on GH200 GPUs with Gumbel MCTS
   - High-quality selfplay (not heuristic fallback)

## Directory Value Summary

### CRITICAL (Keep Forever)
| Directory | Size | Purpose |
|-----------|------|---------|
| canonical_models/ | 1.1GB | Production AI models |
| canonical_games/ | 6.4GB | Validated training games |
| cluster_collected_backup/ | 2.9TB | Lambda node data (327K+ games) |

### VALUABLE (Active Training)
| Directory | Size | Purpose |
|-----------|------|---------|
| selfplay_repository/ | 1.7TB | Ongoing selfplay collection |
| cluster_aggregated/ | 270GB | Consolidated training pool |
| canonical_data/ | 11GB | NPZ training files |

### ARCHIVED
| Directory | Size | Purpose |
|-----------|------|---------|
| archived/model_checkpoints*.tar.gz | ~30GB | Historical checkpoints |
| fallback_archive/ | 3GB | CPU selfplay backup |

### DELETED (Dec 26, 2025)
| Directory | Size | Reason |
|-----------|------|--------|
| cluster-backups/ | 90GB | Partial old backup |
| local_machine_sync/ | 123GB | Dev snapshots |
| models/ | 21GB | Sync copies |
| model_backups/ | 12GB | Duplicates |
| cluster_backup/ | 268GB | Older backup |
| cluster_sync/ | 178GB | Stale sync |

**Total Freed:** ~700GB

## Data Pipeline Integration

### To use Lambda data for training:

1. **Consolidate databases:**
   ```python
   # In scripts/consolidate_lambda_backup.py
   from app.db.game_replay_db import GameReplayDB

   sources = [
       "/Volumes/RingRift-Data/cluster_collected_backup/lambda-gh200-g/selfplay.db",
       "/Volumes/RingRift-Data/cluster_collected_backup/lambda-gh200-h/selfplay.db",
       "/Volumes/RingRift-Data/cluster_collected_backup/lambda-gh200-e/selfplay.db",
   ]

   dest = GameReplayDB("/Volumes/RingRift-Data/consolidated_lambda_training.db")
   for src in sources:
       dest.merge_from(src)
   ```

2. **Export to NPZ:**
   ```bash
   python scripts/export_replay_dataset.py \
     --db /Volumes/RingRift-Data/consolidated_lambda_training.db \
     --use-discovery \
     --output data/training/lambda_consolidated.npz
   ```

3. **Train with combined data:**
   ```bash
   python -m app.training.train \
     --board-type square8 --num-players 2 \
     --data-path data/training/lambda_consolidated.npz \
     --epochs 75
   ```

## Monitoring

Use `scripts/check_sync_health.py` to monitor:
- OWC drive mount status
- Sync daemon health
- Data freshness

```bash
python scripts/check_sync_health.py --watch
```
