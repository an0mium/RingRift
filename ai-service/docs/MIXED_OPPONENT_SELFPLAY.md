# Mixed Opponent Selfplay Guide

## Overview

Mixed opponent selfplay enables training data generation against diverse opponent strengths, improving model robustness and generalization. Instead of always playing against the same engine, the model faces a mix of:

- **Random opponents (30% default)** - Maximum exploration and position diversity
- **Heuristic opponents (40% default)** - Fast tactical play and common patterns
- **MCTS opponents (30% default)** - Strong strategic depth

## Quick Start

### Command Line

Default mix (30% random, 40% heuristic, 30% MCTS):

```bash
python scripts/selfplay.py --board hex8 --num-players 2 --num-games 1000 --mixed-opponents
```

Custom mix (50% random, 30% heuristic, 20% MCTS):

```bash
python scripts/selfplay.py --board square8 --num-players 2 --num-games 500 \
  --mixed-opponents --opponent-mix "random:0.5,heuristic:0.3,mcts:0.2"
```

### Programmatic Usage

```python
from app.training.mixed_opponent_selfplay import MixedOpponentSelfplayRunner
from app.training.selfplay_config import SelfplayConfig

# Create configuration
config = SelfplayConfig(
    board_type="hex8",
    num_players=2,
    num_games=1000,
    mixed_opponents=True,
    opponent_mix={"random": 0.3, "heuristic": 0.4, "mcts": 0.3},
    record_db="data/games/hex8_2p_mixed.db"
)

# Run selfplay
runner = MixedOpponentSelfplayRunner(config)
stats = runner.run()

print(f"Generated {stats.games_completed} games")
print(f"Throughput: {stats.games_per_second:.2f} games/sec")
```

### Curriculum Training

Use the "robust_diverse" curriculum stage for automated configuration:

```python
from app.training.selfplay_config import get_curriculum_config

# Get pre-configured mixed opponent settings
config = get_curriculum_config("robust_diverse", board_type="hex8", num_players=2)

# This creates:
# - 600 games with MIXED engine
# - Temperature 1.0
# - 400 MCTS simulations (for MCTS opponents)
# - 1 random opening move
# - Default 30/40/30 opponent mix
```

## Configuration Options

### Opponent Mix

Customize the opponent distribution via `opponent_mix` dict or `--opponent-mix` flag:

```python
# Python
config.opponent_mix = {
    "random": 0.2,      # 20% random opponents
    "heuristic": 0.5,   # 50% heuristic opponents
    "mcts": 0.3         # 30% MCTS opponents
}
```

```bash
# CLI
--opponent-mix "random:0.2,heuristic:0.5,mcts:0.3"
```

**Note**: Probabilities must sum to ~1.0 (automatically normalized if not).

### Opponent Characteristics

| Opponent Type | Speed     | Strength | Use Case                         |
| ------------- | --------- | -------- | -------------------------------- |
| **Random**    | Very Fast | Weakest  | Exploration, opening diversity   |
| **Heuristic** | Fast      | Medium   | Tactical patterns, quick games   |
| **MCTS**      | Slower    | Strong   | Strategic depth, quality targets |

## Integration with Existing Features

### PFSP (Prioritized Fictitious Self-Play)

Mixed opponent selfplay is fully compatible with PFSP. The opponent type refers to the **engine**, while PFSP selects specific **model checkpoints**:

```python
# PFSP selects model versions automatically
# Mixed opponents determine engine type per game
config = SelfplayConfig(
    board_type="hex8",
    num_players=2,
    mixed_opponents=True,
    use_pfsp=True  # Enabled by default
)
```

### Temperature Scheduling

Elo-adaptive temperature scheduling works seamlessly:

```python
config = SelfplayConfig(
    board_type="hex8",
    num_players=2,
    mixed_opponents=True,
    model_elo=1650.0,  # Enables Elo-adaptive budget and temperature
)
```

### Pipeline Automation

Mixed opponent selfplay emits `SELFPLAY_COMPLETE` events for pipeline automation:

```bash
python scripts/selfplay.py --board hex8 --num-players 2 \
  --mixed-opponents --emit-pipeline-events
```

This triggers automatic export → train → evaluate → promote workflow.

## Statistics and Monitoring

The runner tracks and reports opponent usage distribution:

```
INFO:app.training.mixed_opponent_selfplay:Opponent usage distribution:
INFO:app.training.mixed_opponent_selfplay:  random: 305/1000 (30.5%, target: 30.0%)
INFO:app.training.mixed_opponent_selfplay:  heuristic: 398/1000 (39.8%, target: 40.0%)
INFO:app.training.mixed_opponent_selfplay:  mcts: 297/1000 (29.7%, target: 30.0%)
```

Metadata is stored in game database for analysis:

```python
# Query games by opponent type
import sqlite3
conn = sqlite3.connect("data/games/hex8_2p_mixed.db")
cursor = conn.execute("""
    SELECT metadata->>'opponent_type', COUNT(*)
    FROM games
    GROUP BY metadata->>'opponent_type'
""")
for opponent_type, count in cursor:
    print(f"{opponent_type}: {count} games")
```

## Performance Considerations

### Speed

- **Random opponents**: ~100-200 games/sec (fastest)
- **Heuristic opponents**: ~20-50 games/sec (fast)
- **MCTS opponents**: ~2-5 games/sec (slower, higher quality)

Default 30/40/30 mix balances speed (~30 games/sec average) with quality.

### Quality vs. Throughput

Adjust mix based on training phase:

**Bootstrap phase** (need lots of data fast):

```python
opponent_mix = {"random": 0.6, "heuristic": 0.3, "mcts": 0.1}
```

**Refinement phase** (need quality targets):

```python
opponent_mix = {"random": 0.1, "heuristic": 0.3, "mcts": 0.6}
```

**Balanced training** (default):

```python
opponent_mix = {"random": 0.3, "heuristic": 0.4, "mcts": 0.3}
```

## Best Practices

1. **Start with default mix** - The 30/40/30 split is well-tested and balanced
2. **Use during mid-training** - After bootstrap, before fine-tuning
3. **Monitor distribution** - Check actual vs target opponent percentages
4. **Combine with PFSP** - Let PFSP select model checkpoints, mix selects engine
5. **Track metadata** - Store opponent_type for downstream analysis

## Troubleshooting

### Import Error

```python
from app.training.mixed_opponent_selfplay import MixedOpponentSelfplayRunner
# ImportError: No module named 'app.training.mixed_opponent_selfplay'
```

**Solution**: Ensure `PYTHONPATH=.` when running from ai-service directory.

### Opponent Mix Doesn't Sum to 1.0

```
WARNING: Opponent mix probabilities sum to 0.8, not 1.0. Normalizing...
```

**Solution**: Mix is automatically normalized. To avoid warning, ensure probabilities sum to 1.0:

```python
opponent_mix = {"random": 0.3, "heuristic": 0.4, "mcts": 0.3}  # Sums to 1.0
```

### Actual Distribution Differs from Target

```
random: 350/1000 (35.0%, target: 30.0%)
```

**Solution**: This is expected variance. Over 1000+ games, distribution converges to target. Small deviations (<5%) are normal for sample size.

## Examples

### Example 1: Bootstrap Training with High Random Mix

```bash
# Fast data generation for initial training
python scripts/selfplay.py \
  --board hex8 --num-players 2 --num-games 10000 \
  --mixed-opponents --opponent-mix "random:0.7,heuristic:0.2,mcts:0.1"
```

### Example 2: Quality Training with High MCTS Mix

```bash
# High-quality games for model refinement
python scripts/selfplay.py \
  --board square19 --num-players 2 --num-games 1000 \
  --mixed-opponents --opponent-mix "random:0.1,heuristic:0.2,mcts:0.7"
```

### Example 3: Balanced Multi-Player Training

```python
from app.training.mixed_opponent_selfplay import MixedOpponentSelfplayRunner
from app.training.selfplay_config import SelfplayConfig

# 4-player with balanced mix
config = SelfplayConfig(
    board_type="square8",
    num_players=4,
    num_games=5000,
    mixed_opponents=True,
    # Default 30/40/30 mix used
    record_db="data/games/square8_4p_mixed.db"
)

runner = MixedOpponentSelfplayRunner(config)
stats = runner.run()
```

## See Also

- `/Users/armand/Development/RingRift/ai-service/app/training/selfplay_config.py` - Configuration options
- `/Users/armand/Development/RingRift/ai-service/app/training/selfplay_runner.py` - Base runner class
- `/Users/armand/Development/RingRift/CLAUDE.md` - General RingRift documentation
