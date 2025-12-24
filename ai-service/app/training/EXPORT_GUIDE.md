# Export Pipeline Guide

**Created**: December 2025
**Purpose**: Clarify export module roles and usage

## Current Architecture

| Module                  | Lines | Role                                                      |
| ----------------------- | ----- | --------------------------------------------------------- |
| `export_core.py`        | 536   | **Core utilities** - value computation, encoding, NPZ I/O |
| `export_cache.py`       | 396   | Caching layer for expensive computations                  |
| `dynamic_export.py`     | 294   | Feature version management                                |
| `incremental_export.py` | 494   | Streaming/incremental export                              |
| `game_record_export.py` | 191   | DB → game record extraction                               |

### Scripts

| Script                            | Lines | Role                                |
| --------------------------------- | ----- | ----------------------------------- |
| `export_replay_dataset.py`        | 1,312 | **Primary CLI** - full pipeline     |
| `export_gumbel_kl_dataset.py`     | 405   | Gumbel MCTS policy targets          |
| `export_training_from_cluster.py` | 288   | Cluster-wide data collection        |
| `export_state_pool.py`            | 300   | State pool for distributed training |
| `export_canonical_to_jsonl.py`    | 192   | DB → JSONL conversion               |
| `export_from_metadata_json.py`    | 275   | Metadata-based export               |
| `export_heuristic_weights.py`     | 148   | Heuristic weight extraction         |

## Recommended Usage

### Standard Training Data Export

```bash
# Using GameDiscovery to find all databases
python scripts/export_replay_dataset.py \
    --use-discovery \
    --board-type hex8 \
    --num-players 2 \
    --output data/training/hex8_2p.npz

# From specific database
python scripts/export_replay_dataset.py \
    --db data/games/canonical_hex8_2p.db \
    --board-type hex8 \
    --num-players 2 \
    --output data/training/hex8_2p.npz
```

### Gumbel MCTS Training Data

```bash
# Export with KL-divergence policy targets from Gumbel MCTS
python scripts/export_gumbel_kl_dataset.py \
    --db data/games/gumbel_selfplay.db \
    --board-type square8 \
    --output data/training/sq8_gumbel.npz
```

### Cluster-Wide Export

```bash
# Collect data from all cluster nodes
python scripts/export_training_from_cluster.py \
    --board-type hex8 \
    --num-players 2 \
    --output data/training/hex8_2p_cluster.npz
```

## Module Roles

### export_core.py (Primary)

Core utilities used by all export scripts:

```python
from app.training.export_core import (
    # Value computation
    compute_value,           # (winner, perspective) → float
    value_from_final_winner, # GameState → value for 2p
    value_from_final_ranking,# GameState → values for N players
    compute_multi_player_values,  # Full multiplayer value vector

    # State encoding
    encode_state_with_history,  # GameState → numpy features

    # NPZ I/O
    NPZDatasetWriter,        # Streaming writer
    load_npz_dataset,        # Load for training
    validate_npz_dataset,    # Check integrity
)
```

### export_cache.py

Caches expensive computations (feature extraction, policy targets):

```python
from app.training.export_cache import ExportCache

cache = ExportCache(cache_dir="data/cache")
features = cache.get_or_compute(
    key=f"{game_id}_{move_idx}",
    compute_fn=lambda: encode_state_with_history(state, history),
)
```

### dynamic_export.py

Manages feature versions for compatibility:

```python
from app.training.dynamic_export import ExportSettings, get_feature_version

# Get current feature version
version = get_feature_version(board_type="hex8")

# Configure export settings
settings = ExportSettings(
    board_type="hex8",
    num_players=2,
    feature_version="v3",
    include_policy=True,
)
```

### incremental_export.py

Stream export for large datasets:

```python
from app.training.incremental_export import IncrementalExporter

exporter = IncrementalExporter(
    output_path="data/training/large_dataset.npz",
    chunk_size=10000,
)

for game_record in iterate_games(db_path):
    exporter.add_game(game_record)

exporter.finalize()
```

## Data Formats

### NPZ Format

Standard training data format:

```python
data = np.load("data/training/hex8_2p.npz")

# Features: [N, C, H, W] tensor
features = data["features"]  # Shape: (positions, channels, height, width)

# Policy targets: [N, policy_size] softmax distribution
policy = data["policy"]  # Shape: (positions, num_cells)

# Value targets: [N] for 2p, [N, num_players] for multiplayer
value = data["value"]  # Shape: (positions,) or (positions, num_players)

# Metadata
metadata = data["metadata"].item()  # Dict with board_type, num_players, etc.
```

### JSONL Format

Intermediate format for game records:

```json
{"game_id": "abc123", "board_type": "hex8", "num_players": 2, "winner": 1, "moves": [...]}
```

## Quality Checking

Always validate exported data before training:

```bash
# Validate NPZ file
python -m app.training.data_quality --npz data/training/hex8_2p.npz --detailed

# Validate database
python -m app.training.data_quality --db data/games/selfplay.db
```

## Migration Notes

### From Old Export Scripts

If using older export patterns:

```python
# Old: Multiple scattered scripts
python scripts/export_from_sqlite.py ...
python scripts/convert_to_npz.py ...

# New: Single unified script
python scripts/export_replay_dataset.py \
    --use-discovery \
    --board-type hex8 \
    --num-players 2 \
    --output data/training/hex8_2p.npz
```

### Feature Compatibility

When loading data with different feature versions:

```python
from app.training.dynamic_export import get_feature_adapter

# Adapt old features to new format
adapter = get_feature_adapter(from_version="v2", to_version="v3")
new_features = adapter.adapt(old_features)
```

## Files to Archive (future)

Once all users migrate to unified pipeline:

- `scripts/export_canonical_to_jsonl.py` → use `--format jsonl` flag
- `scripts/export_from_metadata_json.py` → use `--use-discovery`
