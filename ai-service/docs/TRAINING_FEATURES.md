# RingRift Training Features Reference

> **Last Updated**: 2025-12-17
> **Status**: Active

This document provides a comprehensive reference for all training features, parameters, and techniques available in the RingRift AI training pipeline.

## Table of Contents

1. [Training Configuration](#training-configuration)
2. [Label Smoothing](#label-smoothing)
3. [Hex Board Augmentation](#hex-board-augmentation)
4. [Advanced Regularization](#advanced-regularization)
5. [Learning Rate Scheduling](#learning-rate-scheduling)
6. [Batch Size Management](#batch-size-management)
7. [Model Architecture Selection](#model-architecture-selection)
8. [CLI Arguments Reference](#cli-arguments-reference)

---

## Training Configuration

### TrainConfig Parameters (`scripts/unified_loop/config.py`)

| Parameter                   | Type  | Default        | Description                                   |
| --------------------------- | ----- | -------------- | --------------------------------------------- |
| `learning_rate`             | float | 1e-3           | Initial learning rate                         |
| `batch_size`                | int   | 256            | Training batch size (optimized for GPU)       |
| `epochs`                    | int   | 50             | Number of training epochs                     |
| `policy_weight`             | float | 1.0            | Weight of policy loss in total loss           |
| `value_weight`              | float | 1.0            | Weight of value loss in total loss            |
| `policy_label_smoothing`    | float | 0.05           | Label smoothing factor (0.05-0.1 recommended) |
| `warmup_epochs`             | int   | 5              | Epochs for learning rate warmup               |
| `early_stopping_patience`   | int   | 15             | Epochs without improvement before stopping    |
| `lr_scheduler`              | str   | "cosine"       | Learning rate scheduler type                  |
| `lr_min`                    | float | 1e-6           | Minimum learning rate for cosine annealing    |
| `sampling_weights`          | str   | "victory_type" | Sample balancing strategy                     |
| `use_optimized_hyperparams` | bool  | true           | Load board-specific hyperparameters           |

### Environment Variables

| Variable                       | Default | Description                               |
| ------------------------------ | ------- | ----------------------------------------- |
| `RINGRIFT_AUTO_BATCH_SCALE`    | 1       | Auto-scale batch size based on GPU memory |
| `RINGRIFT_DISABLE_GPU_DATAGEN` | 0       | Disable GPU parallel data generation      |

---

## Label Smoothing

Label smoothing is a regularization technique that prevents the model from becoming overconfident in its predictions.

### How It Works

Instead of training with hard targets (one-hot encoded), label smoothing mixes the target distribution with a uniform distribution:

```
smoothed_target = (1 - epsilon) * target + epsilon / num_classes
```

### Configuration

```python
# In TrainConfig or via CLI
policy_label_smoothing = 0.05  # Typical range: 0.05-0.1
```

### CLI Usage

```bash
python -m app.training.train \
  --data-path data/training/dataset.npz \
  --board-type hex8 \
  --policy-label-smoothing 0.05
```

### Benefits

- Prevents overconfident predictions
- Improves model calibration
- Better generalization to unseen positions
- Reduces overfitting on noisy labels

### Label Smoothing Warmup

To prevent training instability early on, label smoothing can be gradually introduced:

```yaml
# In unified_loop.yaml
training:
  label_smoothing_warmup: 5 # Apply full smoothing after 5 epochs
```

---

## Hex Board Augmentation

D6 dihedral symmetry augmentation for hexagonal boards provides 12x effective dataset expansion.

### Symmetry Transformations

The D6 group consists of 12 transformations:

- **6 Rotations**: 0°, 60°, 120°, 180°, 240°, 300°
- **6 Reflections**: Mirror across 6 axes

### Supported Board Sizes

| Board Type | Bounding Box | Radius | Hex Cells | Policy Size |
| ---------- | ------------ | ------ | --------- | ----------- |
| hex8       | 9×9          | 4      | 61        | ~4,500      |
| hexagonal  | 25×25        | 12     | 469       | ~92,000     |

### Configuration

```yaml
# In unified_loop.yaml
training:
  use_hex_augmentation: true
```

### CLI Usage

```bash
python -m app.training.train \
  --data-path data/training/hex8_games.npz \
  --board-type hex8 \
  --augment-hex-symmetry
```

### Implementation Details

The `HexSymmetryTransform` class (`app/training/hex_augmentation.py`) provides:

```python
from app.training.hex_augmentation import HexSymmetryTransform, augment_hex_sample

# Create transform for hex8 (9x9) board
transform = HexSymmetryTransform(board_size=9)

# Transform a single sample with all 12 symmetries
augmented = augment_hex_sample(features, globals_vec, policy_indices, policy_values)
# Returns list of 12 (features, globals, policy_indices, policy_values) tuples
```

### Key Functions

| Function                                                 | Description                                  |
| -------------------------------------------------------- | -------------------------------------------- |
| `get_hex_policy_layout(board_size)`                      | Compute policy layout for custom board sizes |
| `transform_board(board, transform_id)`                   | Transform board features                     |
| `transform_policy(policy, transform_id)`                 | Transform policy vector                      |
| `transform_sparse_policy(indices, values, transform_id)` | Transform sparse policy                      |
| `get_inverse_transform(transform_id)`                    | Get inverse of a transformation              |

---

## Advanced Regularization

### Stochastic Weight Averaging (SWA)

Averages model weights over training trajectory for better generalization.

```yaml
training:
  use_swa: true
  swa_start_fraction: 0.75 # Start averaging at 75% of training
```

### Exponential Moving Average (EMA)

Maintains a smoothed version of model weights.

```yaml
training:
  use_ema: true
  ema_decay: 0.999 # Decay rate for EMA
```

### Stochastic Depth

Randomly drops residual blocks during training for regularization.

```yaml
training:
  use_stochastic_depth: true
  stochastic_depth_prob: 0.1 # Drop probability
```

### Value Whitening

Normalizes value head outputs for stable training.

```yaml
training:
  use_value_whitening: true
  value_whitening_momentum: 0.99
```

### Spectral Normalization

Constrains weight matrices for gradient stability.

```yaml
training:
  use_spectral_norm: true
```

---

## Learning Rate Scheduling

### Available Schedulers

| Scheduler              | Description                   |
| ---------------------- | ----------------------------- |
| `none`                 | Constant learning rate        |
| `step`                 | Step decay at fixed intervals |
| `cosine`               | Cosine annealing to lr_min    |
| `cosine-warm-restarts` | Cosine with periodic restarts |

### Configuration

```bash
python -m app.training.train \
  --lr-scheduler cosine \
  --lr-min 1e-6 \
  --warmup-epochs 5
```

### Cyclic Learning Rate

Triangular wave cycling for escaping local minima:

```yaml
training:
  use_cyclic_lr: true
  cyclic_lr_period: 5 # Cycle every 5 epochs
```

### Adaptive Warmup

Automatically adjusts warmup duration based on dataset size:

```yaml
training:
  use_adaptive_warmup: true
```

---

## Batch Size Management

### Progressive Batch Sizing

Gradually increases batch size during training:

```yaml
training:
  use_progressive_batch: true
  min_batch_size: 64 # Start with small batches
  max_batch_size: 512 # Ramp up to large batches
```

### Dynamic Batch Scheduling

```yaml
training:
  use_dynamic_batch: true
  dynamic_batch_schedule: 'linear' # linear, exponential, or step
```

### GPU Memory Auto-Scaling

Batch size automatically scales based on detected GPU:

| GPU             | Multiplier |
| --------------- | ---------- |
| H100 (80GB)     | 16x        |
| A100 (40GB)     | 8x         |
| RTX 4090 (24GB) | 4x         |
| RTX 3090 (24GB) | 4x         |
| Default         | 1x         |

---

## Model Architecture Selection

### Square Board Models

| Version | Description                           | Recommended For |
| ------- | ------------------------------------- | --------------- |
| v2      | Flat policy head                      | square19        |
| v3      | Spatial policy with rank distribution | square8         |
| v4      | NAS-optimized with attention          | Experimental    |

### Hex Board Models

| Version         | Channels      | Description               |
| --------------- | ------------- | ------------------------- |
| HexNeuralNet_v2 | 10 per player | Original hex architecture |
| HexNeuralNet_v3 | 16 per player | Improved, recommended     |

### Configuration

```yaml
# In unified_loop.yaml
training:
  hex_encoder_version: 'v3' # Use HexStateEncoderV3
```

### CLI Usage

```bash
python -m app.training.train \
  --board-type hex8 \
  --model-version hex  # Auto-selects HexNeuralNet
```

---

## CLI Arguments Reference

### Data and Model

| Argument          | Type | Default  | Description                        |
| ----------------- | ---- | -------- | ---------------------------------- |
| `--data-path`     | str  | Required | Path to training NPZ file          |
| `--save-path`     | str  | Auto     | Output model path                  |
| `--board-type`    | str  | Required | square8, square19, hex8, hexagonal |
| `--model-version` | str  | Auto     | v2, v3, v4, hex                    |

### Training Parameters

| Argument          | Type  | Default | Description           |
| ----------------- | ----- | ------- | --------------------- |
| `--epochs`        | int   | 50      | Number of epochs      |
| `--batch-size`    | int   | 64      | Batch size            |
| `--learning-rate` | float | 1e-3    | Initial learning rate |
| `--seed`          | int   | None    | Random seed           |

### Regularization

| Argument                    | Type  | Default | Description                |
| --------------------------- | ----- | ------- | -------------------------- |
| `--policy-label-smoothing`  | float | 0.0     | Label smoothing factor     |
| `--augment-hex-symmetry`    | flag  | False   | Enable D6 hex augmentation |
| `--early-stopping-patience` | int   | 10      | Early stopping patience    |

### Learning Rate Schedule

| Argument          | Type  | Default | Description              |
| ----------------- | ----- | ------- | ------------------------ |
| `--warmup-epochs` | int   | 0       | LR warmup epochs         |
| `--lr-scheduler`  | str   | none    | Scheduler type           |
| `--lr-min`        | float | 1e-6    | Minimum learning rate    |
| `--lr-t0`         | int   | 10      | T_0 for warm restarts    |
| `--lr-t-mult`     | int   | 2       | T_mult for warm restarts |

### Checkpointing

| Argument                | Type | Default     | Description            |
| ----------------------- | ---- | ----------- | ---------------------- |
| `--checkpoint-dir`      | str  | checkpoints | Checkpoint directory   |
| `--checkpoint-interval` | int  | 5           | Save every N epochs    |
| `--resume`              | str  | None        | Resume from checkpoint |

### Sampling

| Argument             | Type | Default | Description                                  |
| -------------------- | ---- | ------- | -------------------------------------------- |
| `--sampling-weights` | str  | uniform | uniform, late_game, phase_emphasis, combined |

### Distributed Training

| Argument          | Type | Default | Description                  |
| ----------------- | ---- | ------- | ---------------------------- |
| `--distributed`   | flag | False   | Enable DDP                   |
| `--local-rank`    | int  | -1      | Local rank (set by torchrun) |
| `--scale-lr`      | flag | False   | Scale LR by world size       |
| `--lr-scale-mode` | str  | linear  | linear or sqrt               |

---

## Example Training Commands

### Basic Hex8 Training

```bash
python -m app.training.train \
  --data-path data/training/hex8_games.npz \
  --board-type hex8 \
  --epochs 50 \
  --batch-size 64
```

### Advanced Hex8 Training with All Features

```bash
python -m app.training.train \
  --data-path data/training/hex8_games.npz \
  --board-type hex8 \
  --save-path models/ringrift_hex8_2p_v9.pth \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 2e-3 \
  --policy-label-smoothing 0.05 \
  --augment-hex-symmetry \
  --warmup-epochs 5 \
  --lr-scheduler cosine \
  --lr-min 1e-6 \
  --early-stopping-patience 10
```

### Distributed Training

```bash
torchrun --nproc_per_node=4 -m app.training.train \
  --data-path data/training/large_dataset.npz \
  --board-type square8 \
  --distributed \
  --scale-lr \
  --batch-size 256
```

---

## See Also

- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) - Training infrastructure overview
- [NEURAL_AI_ARCHITECTURE.md](NEURAL_AI_ARCHITECTURE.md) - Model architectures
- [UNIFIED_AI_LOOP.md](UNIFIED_AI_LOOP.md) - Automated training loop
- [HEX_AUGMENTATION.md](HEX_AUGMENTATION.md) - Detailed hex augmentation guide
