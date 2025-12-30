# Transfer Learning Guide

This guide covers transfer learning between RingRift model configurations.

## Overview

Transfer learning allows leveraging trained models to accelerate training on new configurations. The main transfer paths are:

| Transfer Type | Source → Target                      | Complexity |
| ------------- | ------------------------------------ | ---------- |
| Player Count  | 2p → 3p, 3p → 4p, 2p → 4p            | Medium     |
| Board Size    | hex8 → hexagonal, square8 → square19 | High       |
| Board Type    | hex → square (or vice versa)         | Very High  |

## Player Count Transfer (Recommended)

The most common and reliable transfer path is changing player count while keeping board geometry.

### 2-Player → 4-Player Transfer

```bash
cd ai-service

# Step 1: Resize value head (2 outputs → 4 outputs)
python scripts/transfer_2p_to_4p.py \
  --source models/canonical_hex8_2p.pth \
  --output models/transfer_hex8_4p_init.pth \
  --board-type hex8

# Step 2: Train with transferred weights
python -m app.training.train \
  --board-type hex8 --num-players 4 \
  --init-weights models/transfer_hex8_4p_init.pth \
  --data-path data/training/hex8_4p.npz \
  --save-path models/hex8_4p_from_2p.pth
```

### What Gets Transferred

| Component                  | Transferred? | Notes                     |
| -------------------------- | ------------ | ------------------------- |
| Encoder (ResNet blocks)    | ✅ Yes       | Fully transferred         |
| Policy Head (pre-softmax)  | ✅ Yes       | Fully transferred         |
| Value Head (hidden layers) | ✅ Yes       | Fully transferred         |
| Value Head (output layer)  | ⚠️ Partial   | Output size changes (2→4) |

### Value Head Resizing

The `transfer_2p_to_4p.py` script handles value head resizing:

```python
# Before: value_head outputs 2 values [p1_win, p2_win]
# After: value_head outputs 4 values [p1_win, p2_win, p3_win, p4_win]

# Strategy: Initialize new outputs from mean of existing outputs
# This gives a neutral starting point for new players
old_output = checkpoint["model"]["value_head.output.weight"]  # [2, hidden]
new_output = torch.zeros(4, hidden)
new_output[0] = old_output[0]  # Player 1 preserved
new_output[1] = old_output[1]  # Player 2 preserved
new_output[2] = old_output.mean(dim=0)  # Player 3 = average
new_output[3] = old_output.mean(dim=0)  # Player 4 = average
```

### 2p → 3p Transfer

```bash
# Similar process with 3-output value head
python scripts/transfer_2p_to_3p.py \
  --source models/canonical_hex8_2p.pth \
  --output models/transfer_hex8_3p_init.pth \
  --board-type hex8
```

### 3p → 4p Transfer

```bash
# Transfer from 3-player to 4-player
python scripts/transfer_3p_to_4p.py \
  --source models/canonical_hex8_3p.pth \
  --output models/transfer_hex8_4p_init.pth \
  --board-type hex8
```

## Board Size Transfer (Advanced)

Transferring between board sizes requires handling different input dimensions.

### Small → Large Board (hex8 → hexagonal)

```bash
# This requires architecture adaptation - input channels differ
python scripts/transfer_board_size.py \
  --source models/canonical_hex8_2p.pth \
  --output models/transfer_hexagonal_2p_init.pth \
  --source-board hex8 \
  --target-board hexagonal \
  --num-players 2
```

**What Gets Transferred:**

- Encoder: Only layers with matching dimensions
- Policy Head: Requires position encoding change
- Value Head: Fully transferred

**Challenges:**

- Different board cell counts (61 vs 469 for hex)
- Position encodings need remapping
- Policy output size changes

### Implementation

```python
# Pseudocode for board size transfer
def transfer_board_size(source_ckpt, source_board, target_board):
    # 1. Load source checkpoint
    source_model = load_checkpoint(source_ckpt)

    # 2. Create target model with different input size
    target_model = create_model(
        board_type=target_board,
        num_players=source_model.num_players,
    )

    # 3. Transfer compatible layers
    for name, param in source_model.named_parameters():
        if name in target_model.state_dict():
            target_param = target_model.state_dict()[name]
            if param.shape == target_param.shape:
                target_model.state_dict()[name].copy_(param)
            else:
                # Handle shape mismatch with interpolation
                target_model.state_dict()[name].copy_(
                    interpolate_weights(param, target_param.shape)
                )

    return target_model
```

## Board Type Transfer (Experimental)

Transferring between hex and square boards is experimental due to fundamental geometric differences.

### hex8 → square8 (Same Cell Count)

```bash
# Experimental: hex (61 cells) → square (64 cells)
python scripts/transfer_board_type.py \
  --source models/canonical_hex8_2p.pth \
  --output models/transfer_square8_2p_init.pth \
  --source-board hex8 \
  --target-board square8
```

**What Transfers:**

- Value head: Fully compatible
- Encoder hidden layers: Partially (need retraining)
- Position encodings: Incompatible (remapped)
- Adjacency logic: Incompatible (6-neighbor vs 4/8-neighbor)

**Expected Results:**

- Initial model performs better than random
- Requires significant retraining (50%+ of full training)
- Not recommended unless no same-board-type model available

## Partial Loading (Fallback)

When transfer scripts aren't available, use partial loading:

```bash
python -m app.training.train \
  --board-type hex8 --num-players 4 \
  --init-weights models/canonical_hex8_2p.pth \
  --data-path data/training/hex8_4p.npz \
  --partial-load
```

With `--partial-load`, the trainer will:

1. Load all weights that match by name and shape
2. Initialize mismatched weights randomly
3. Log which weights were/weren't loaded

## Training After Transfer

### Recommended Hyperparameters

| Parameter     | Full Training | Transfer Training   |
| ------------- | ------------- | ------------------- |
| Learning Rate | 1e-3          | 1e-4 (10x lower)    |
| Epochs        | 50-100        | 20-30               |
| Batch Size    | 512           | 256 (for stability) |
| Warmup        | 5 epochs      | 2 epochs            |

### Learning Rate Schedule

```bash
python -m app.training.train \
  --board-type hex8 --num-players 4 \
  --init-weights models/transfer_hex8_4p_init.pth \
  --lr 1e-4 \
  --lr-schedule cosine \
  --warmup-epochs 2 \
  --epochs 25
```

### Frozen Layers (Optional)

For very limited data, freeze encoder and train only heads:

```bash
python -m app.training.train \
  --board-type hex8 --num-players 4 \
  --init-weights models/transfer_hex8_4p_init.pth \
  --freeze-encoder \
  --lr 5e-4 \
  --epochs 15
```

## Transfer Learning Matrix

### Recommended Paths (High Success Rate)

```
hex8_2p ──────┬──── hex8_3p ──── hex8_4p
              │
              └──── hex8_4p (direct)

square8_2p ───┬──── square8_3p ──── square8_4p
              │
              └──── square8_4p (direct)
```

### Experimental Paths (Moderate Success)

```
hex8_2p ────────── hexagonal_2p
square8_2p ─────── square19_2p
```

### Not Recommended (Low Success)

```
hex8_* ──X──> square*_*  (geometry mismatch)
square*_* ──X──> hex*_*  (geometry mismatch)
```

## Evaluation After Transfer

Always evaluate transferred models before deploying:

```bash
# Quick gauntlet (50 games per baseline)
python scripts/quick_gauntlet.py \
  --model models/hex8_4p_from_2p.pth \
  --board-type hex8 --num-players 4 \
  --games 50

# Full gauntlet (100 games per baseline)
python -m app.gauntlet.runner \
  --model models/hex8_4p_from_2p.pth \
  --board-type hex8 --num-players 4 \
  --games 100
```

**Promotion Thresholds:**

Thresholds are per-config (board + players). See `app/config/thresholds.py`
(`PROMOTION_THRESHOLDS_BY_CONFIG` and `PROMOTION_MINIMUM_THRESHOLDS`).

## Troubleshooting

### Model Won't Load

```
RuntimeError: Error loading state dict: size mismatch for value_head.output.weight
```

**Solution**: Use the appropriate transfer script to resize weights before loading.

### Poor Initial Performance

If transferred model performs worse than random:

1. Check board type matches
2. Verify value head was properly resized
3. Try lower learning rate (1e-5)

### Training Instability

If loss spikes or diverges:

1. Reduce learning rate
2. Increase warmup epochs
3. Reduce batch size
4. Try frozen encoder training

## Scripts Reference

| Script                           | Purpose                        |
| -------------------------------- | ------------------------------ |
| `scripts/transfer_2p_to_4p.py`   | Resize value head 2→4 outputs  |
| `scripts/transfer_2p_to_3p.py`   | Resize value head 2→3 outputs  |
| `scripts/transfer_3p_to_4p.py`   | Resize value head 3→4 outputs  |
| `scripts/transfer_board_size.py` | Handle board dimension changes |
| `scripts/transfer_board_type.py` | Experimental hex↔square        |

## See Also

- `CLAUDE.md` - Main documentation
- `docs/TRAINING_GUIDE.md` - Full training documentation
- `app/training/train.py` - Training implementation
