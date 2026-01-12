# RingRift Model Zoo

Last Updated: January 12, 2026

## Overview

This document catalogs all canonical RingRift AI models. Each model is trained via self-play and evaluated against baselines before deployment.

## Canonical Models

All canonical models are stored in `ai-service/models/` with naming convention `canonical_{board}_{n}p.pth`.

### Summary Table

| Board     | Players | Size  | Elo (b800) | Games | Architecture   | Status     |
| --------- | ------- | ----- | ---------- | ----- | -------------- | ---------- |
| hex8      | 2       | 33MB  | 1484       | 150   | v2             | Production |
| hex8      | 3       | 38MB  | 1468       | 150   | v2             | Production |
| hex8      | 4       | 38MB  | 1434       | 150   | v2             | Production |
| square8   | 2       | 20MB  | 1502       | 50    | v2             | Production |
| square8   | 3       | 15MB  | 1498       | 50    | v2             | Production |
| square8   | 4       | 366MB | 1478       | 150   | v5-heavy-large | Production |
| square19  | 2       | 103MB | 1506       | 50    | v5-heavy       | Production |
| square19  | 3       | 103MB | 1500       | 50    | v5-heavy       | Production |
| square19  | 4       | 103MB | 1486       | 50    | v5-heavy       | Production |
| hexagonal | 2       | 166MB | 1479       | 150   | v5-heavy-large | Production |
| hexagonal | 3       | 166MB | 1500       | 150   | v5-heavy-large | Production |
| hexagonal | 4       | 166MB | 1474       | 50    | v5-heavy-large | Production |

**Notes:**

- Elo (b800): Rating with Gumbel MCTS budget=800 simulations
- Games: Total games played in Elo pool

## Model Details

### hex8 (Small Hexagonal Board)

Board geometry: Radius 4, 61 cells

| Config   | File                    | Size | Elo  | Training Date |
| -------- | ----------------------- | ---- | ---- | ------------- |
| 2-player | `canonical_hex8_2p.pth` | 33MB | 1484 | Jan 11, 2026  |
| 3-player | `canonical_hex8_3p.pth` | 38MB | 1468 | Dec 28, 2025  |
| 4-player | `canonical_hex8_4p.pth` | 38MB | 1434 | Jan 6, 2026   |

**Architecture**: HexNeuralNet_v2

- 96 channels, 6 residual blocks
- SE attention (squeeze-excitation)
- Position-aware policy head
- Per-player value head (softmax)

### square8 (Standard Square Board)

Board geometry: 8x8, 64 cells

| Config   | File                       | Size  | Elo  | Training Date |
| -------- | -------------------------- | ----- | ---- | ------------- |
| 2-player | `canonical_square8_2p.pth` | 20MB  | 1502 | Jan 12, 2026  |
| 3-player | `canonical_square8_3p.pth` | 15MB  | 1498 | Dec 28, 2025  |
| 4-player | `canonical_square8_4p.pth` | 366MB | 1478 | Dec 25, 2025  |

**Notes:**

- 2p model recently retrained with quality selfplay data
- 4p uses larger v5-heavy-large architecture for multiplayer complexity

### square19 (Large Square Board)

Board geometry: 19x19, 361 cells

| Config   | File                        | Size  | Elo  | Training Date |
| -------- | --------------------------- | ----- | ---- | ------------- |
| 2-player | `canonical_square19_2p.pth` | 103MB | 1506 | Dec 25, 2025  |
| 3-player | `canonical_square19_3p.pth` | 103MB | 1500 | Dec 28, 2025  |
| 4-player | `canonical_square19_4p.pth` | 103MB | 1486 | Dec 28, 2025  |

**Architecture**: HexNeuralNet_v5_Heavy

- 49 heuristic input features
- Wider network for complex positions
- Optimized for large board evaluation

### hexagonal (Large Hexagonal Board)

Board geometry: Radius 12, 469 cells

| Config   | File                         | Size  | Elo  | Training Date |
| -------- | ---------------------------- | ----- | ---- | ------------- |
| 2-player | `canonical_hexagonal_2p.pth` | 166MB | 1479 | Dec 25, 2025  |
| 3-player | `canonical_hexagonal_3p.pth` | 166MB | 1500 | Jan 9, 2026   |
| 4-player | `canonical_hexagonal_4p.pth` | 166MB | 1474 | Dec 25, 2025  |

**Architecture**: HexNeuralNet_v5_Heavy_Large

- 256 filters (vs 128 in v5-heavy)
- Scaled for largest board geometry
- Most computationally intensive models

## Elo Interpretation

| Elo Range | Interpretation | Expected Win Rate vs Heuristic |
| --------- | -------------- | ------------------------------ |
| <1400     | Below baseline | <40%                           |
| 1400-1450 | Weak           | 40-45%                         |
| 1450-1500 | Baseline       | 45-55%                         |
| 1500-1550 | Above baseline | 55-60%                         |
| 1550-1600 | Strong         | 60-65%                         |
| >1600     | Very strong    | >65%                           |

**Note**: Heuristic baseline is anchored at ~1500 Elo.

## Architecture Reference

### v2 (Standard)

```
Parameters: ~2-4M
Channels: 96
Residual blocks: 6
Attention: SE (squeeze-excitation)
Policy: Position-aware
Value: Per-player softmax
```

### v5-heavy

```
Parameters: ~8-12M
Channels: 128
Input features: 49 (full heuristics)
Residual blocks: 8
Attention: SE
Policy: Position-aware
Value: Per-player softmax
```

### v5-heavy-large

```
Parameters: ~25-35M
Channels: 256
Input features: 49 (full heuristics)
Residual blocks: 10
Attention: SE
Policy: Position-aware
Value: Per-player softmax
```

## Training Data Requirements

| Quality Tier | Simulation Budget | Min Games | Use Case            |
| ------------ | ----------------- | --------- | ------------------- |
| Bootstrap    | 64-150            | 100       | Quick iteration     |
| Standard     | 200               | 500       | Development         |
| Quality      | 800               | 1000      | Production training |
| Ultimate     | 1600              | 2000+     | Research            |

## Promotion Thresholds

For a model to become canonical:

| Opponent     | Required Win Rate | Games |
| ------------ | ----------------- | ----- |
| Random       | >85%              | 25    |
| Heuristic    | >55%              | 50    |
| Previous Gen | >50%              | 50    |

## Version History

### Generation 1 (December 2025 - January 2026)

Initial canonical models trained via self-play with Gumbel MCTS:

- Dec 25, 2025: First complete set of 12 canonical models
- Dec 28, 2025: hex8_3p, hex8_4p, square8_3p, square19_3p, square19_4p retrained
- Jan 4-6, 2026: v2 variants created for hex8, square19
- Jan 9, 2026: hexagonal_3p averaged model
- Jan 11-12, 2026: hex8_2p, square8_2p retrained with quality data

## Loading Models

```python
from app.ai.hex_neural_net import HexNeuralNetFactory
from app.utils.torch_utils import safe_load_checkpoint

# Load checkpoint safely
checkpoint = safe_load_checkpoint("models/canonical_hex8_2p.pth")

# Create model from checkpoint
model = HexNeuralNetFactory.from_checkpoint(
    checkpoint,
    board_type="hex8",
    num_players=2,
)
model.eval()
```

## Related Documents

- [Evaluation Methodology](EVALUATION_METHODOLOGY.md) - How models are evaluated
- [CLAUDE.md](../CLAUDE.md) - Development context
- [Deployment Checklist](../../docs/DEPLOYMENT_CHECKLIST.md) - Production deployment

## Model Storage

Canonical models are stored in:

- **Local**: `ai-service/models/canonical_*.pth`
- **Cluster**: Synced to all training nodes via P2P distribution
- **Production**: `ringrift-web-prod:/home/ubuntu/RingRift/ai-service/models/`

Symlinks (`ringrift_best_*.pth`) point to canonical models for backward compatibility.
