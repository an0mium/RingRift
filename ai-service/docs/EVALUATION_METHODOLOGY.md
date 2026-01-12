# RingRift AI Evaluation Methodology

Last Updated: January 12, 2026

## Overview

This document describes how RingRift AI models are evaluated to measure improvement over training iterations. The goal is to demonstrate that models iteratively improve through self-play training.

## Baseline Opponents

### 1. Random Baseline

- **Description**: Selects a uniformly random legal move
- **Implementation**: `app/ai/random_ai.py`
- **Expected Performance**: Models should achieve >85% win rate
- **Purpose**: Sanity check - any trained model should beat random

### 2. Heuristic Baseline

- **Description**: Rule-based player using domain knowledge
- **Implementation**: `app/ai/heuristic_ai.py`
- **Features Used**:
  - Territory control (cells owned)
  - Stack height advantage
  - Board position (center vs edge)
  - Mobility (number of legal moves)
- **Expected Performance**: Trained models should achieve >55% win rate
- **Purpose**: Primary benchmark for model quality

### 3. Neural Network Descent (nn-descent)

- **Description**: Previous generation model with greedy move selection
- **Implementation**: Policy network, argmax selection
- **Expected Performance**: New models should achieve >50% win rate
- **Purpose**: Prove generation-over-generation improvement

### 4. Weak Neural Network

- **Description**: Model with reduced MCTS simulations (budget=16)
- **Implementation**: Same model, limited search
- **Expected Performance**: Full model should achieve >60% win rate
- **Purpose**: Verify MCTS search adds value beyond raw policy

## Evaluation Metrics

### Primary Metrics

| Metric                   | Definition              | Target               |
| ------------------------ | ----------------------- | -------------------- |
| Win Rate vs Random       | Wins / Total Games      | >85%                 |
| Win Rate vs Heuristic    | Wins / Total Games      | >55%                 |
| Win Rate vs Previous Gen | Wins / Total Games      | >50%                 |
| Elo Rating               | Relative strength score | Increasing over time |

### Game Counting Rules

- **Win**: Model achieves highest score at game end
- **Loss**: Model does not achieve highest score
- **Draw**: Multiple players tie for highest score
- **Draw Handling**: Draws count as 0.5 wins for win rate calculation

### Statistical Significance

For gauntlet results to be considered significant:

| Games Played | 95% CI Width | Interpretation      |
| ------------ | ------------ | ------------------- |
| 25 games     | ±20%         | Rough estimate only |
| 50 games     | ±14%         | Moderate confidence |
| 100 games    | ±10%         | Good confidence     |
| 200 games    | ±7%          | High confidence     |

**Minimum requirement**: 50 games for promotion decisions

## Elo Rating System

### Implementation

RingRift uses the standard Elo rating system:

```
Expected Score = 1 / (1 + 10^((Ra - Rb) / 400))
New Rating = Old Rating + K * (Actual - Expected)
```

**Parameters**:

- Initial Elo: 1500
- K-factor: 32 (standard)
- Database: `data/unified_elo.db`

### Elo Interpretation

| Elo Range | Interpretation                           |
| --------- | ---------------------------------------- |
| <1400     | Below baseline (weaker than heuristic)   |
| 1400-1500 | Baseline level (comparable to heuristic) |
| 1500-1600 | Above baseline (stronger than heuristic) |
| 1600-1700 | Strong (beats heuristic >60%)            |
| >1700     | Very strong (beats heuristic >70%)       |

### Heuristic Baseline Elo

The heuristic baseline is anchored at approximately **1500 Elo**. This means:

- Models with Elo >1500 are stronger than heuristic
- Models with Elo <1500 are weaker than heuristic

## Gauntlet Evaluation Process

### Standard Gauntlet

```bash
python -m app.training.game_gauntlet \
  --board-type hex8 --num-players 2 \
  --model-path models/canonical_hex8_2p.pth \
  --games 50
```

### Gauntlet Stages

1. **vs Random** (25 games)
   - Must pass: >85% win rate
   - Failure: Model is broken

2. **vs Heuristic** (50 games)
   - Must pass: >55% win rate
   - Failure: Model underperforms baseline

3. **vs Previous Generation** (50 games, if available)
   - Must pass: >50% win rate
   - Failure: Regression detected

### Promotion Thresholds

For a model to be promoted to production:

| Opponent     | Required Win Rate | Games |
| ------------ | ----------------- | ----- |
| Random       | >85%              | 25    |
| Heuristic    | >55%              | 50    |
| Previous Gen | >50%              | 50    |

## Generation Tracking

### Database Schema

Results are stored in `data/generation_tracking.db`:

```sql
-- Each trained model
CREATE TABLE model_generations (
    generation_id INTEGER PRIMARY KEY,
    model_path TEXT NOT NULL,
    parent_generation INTEGER,
    board_type TEXT NOT NULL,
    num_players INTEGER NOT NULL,
    created_at REAL,
    training_games INTEGER,
    training_samples INTEGER
);

-- Head-to-head tournaments
CREATE TABLE generation_tournaments (
    gen_a INTEGER NOT NULL,
    gen_b INTEGER NOT NULL,
    gen_a_wins INTEGER,
    gen_b_wins INTEGER,
    draws INTEGER,
    gen_a_elo REAL,
    gen_b_elo REAL
);

-- Elo over time
CREATE TABLE elo_progression (
    generation_id INTEGER NOT NULL,
    elo REAL NOT NULL,
    games_played INTEGER NOT NULL,
    timestamp REAL
);
```

### Demonstrating Improvement

To show iterative improvement:

```bash
python scripts/track_elo_progress.py --generations
```

Expected output:

```
=== RingRift Generation Progress Report ===

### hex8_2p ###
  Generations: 3
  Elo: 1450 -> 1520 (+70)
  Lineage: v1 -> v2 -> v3
  Training: 15,000 games, 750,000 samples
```

## Reproducibility

### Training Command

To reproduce a model:

```bash
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p_quality.npz \
  --epochs 20 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --model-version v2 \
  --save-path models/canonical_hex8_2p_v2.pth
```

### Evaluation Command

To evaluate a model:

```bash
python -m app.training.game_gauntlet \
  --board-type hex8 --num-players 2 \
  --model-path models/canonical_hex8_2p_v2.pth \
  --games 100 \
  --seed 42  # For reproducibility
```

### Environment Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU inference)
- See `requirements.txt` for full list

## Common Issues

### Issue: Model underperforms heuristic

**Possible causes**:

1. Training data quality too low (budget <150 simulations)
2. Insufficient training samples (<50,000)
3. Overfitting (validation loss increasing)
4. Architecture mismatch (wrong value head size)

**Solution**:

- Use `--quality-tier quality` when exporting
- Increase simulation budget to 800+
- Enable early stopping

### Issue: Elo not improving across generations

**Possible causes**:

1. Training on same data repeatedly
2. No opponent diversity during selfplay
3. Curriculum not advancing

**Solution**:

- Generate fresh selfplay data with new model
- Mix 50% heuristic + 30% descent + 20% weak opponents
- Check curriculum progress metrics

### Issue: Large variance in gauntlet results

**Possible causes**:

1. Too few games (need 50+ for stability)
2. High Elo uncertainty
3. Stochastic opponent behavior

**Solution**:

- Run 100+ games for reliable results
- Report confidence intervals
- Use fixed random seed for reproducibility

## Related Documents

- [Model Zoo](MODEL_ZOO.md) - List of all trained models
- [Deployment Checklist](../docs/DEPLOYMENT_CHECKLIST.md) - Production deployment
- [CLAUDE.md](../CLAUDE.md) - Development context
