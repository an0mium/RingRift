# Harness Selection Guide

This guide helps you choose the right AI harness for your use case. A harness wraps an AI evaluation algorithm (Gumbel MCTS, Minimax, etc.) and provides a unified interface for move selection with metadata capture.

## Quick Reference

| Harness       | Player Count | Model Type | Policy Required | Best For                    |
| ------------- | ------------ | ---------- | --------------- | --------------------------- |
| `GUMBEL_MCTS` | 2-4          | NN         | Yes             | Training data, high quality |
| `GPU_GUMBEL`  | 2-4          | NN         | Yes             | High throughput selfplay    |
| `MINIMAX`     | **2 only**   | NN, NNUE   | No              | Fast 2-player evaluation    |
| `MAXN`        | **3-4 only** | NN, NNUE   | No              | Multiplayer search          |
| `BRS`         | **3-4 only** | NN, NNUE   | No              | Fast multiplayer            |
| `POLICY_ONLY` | 2-4          | NN, NNUE\* | Yes             | Fast play, baselines        |
| `DESCENT`     | 2-4          | NN         | Yes             | Exploration, research       |
| `HEURISTIC`   | 2-4          | None       | No              | Baselines, bootstrap        |

\*NNUE with policy head only

## Decision Tree

### For 2-Player Games

```
Need training data?
├─ Yes → GUMBEL_MCTS (quality) or GPU_GUMBEL (throughput)
└─ No
   ├─ Have NNUE model? → MINIMAX (fastest)
   ├─ Have NN model?
   │  ├─ Speed critical? → POLICY_ONLY
   │  └─ Quality critical? → GUMBEL_MCTS
   └─ No model? → HEURISTIC
```

### For 3-4 Player Games

```
Need training data?
├─ Yes → GUMBEL_MCTS or GPU_GUMBEL
└─ No
   ├─ Have NNUE model?
   │  ├─ Speed critical? → BRS (fast, greedy)
   │  └─ Quality critical? → MAXN (thorough)
   ├─ Have NN model?
   │  ├─ Speed critical? → POLICY_ONLY or BRS
   │  └─ Quality critical? → MAXN or GUMBEL_MCTS
   └─ No model? → HEURISTIC
```

## Harness Details

### GUMBEL_MCTS

**Purpose**: Gumbel AlphaZero MCTS with Sequential Halving. The primary harness for generating high-quality training data.

**When to use**:

- Generating selfplay training data
- When move quality is more important than speed
- When you need visit distributions for soft policy targets

**Configuration**:

```python
harness = create_harness(
    HarnessType.GUMBEL_MCTS,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
    simulations=200,  # Higher = better quality, slower
)
```

**Key parameters**:

- `simulations`: Number of MCTS simulations (64-1600 typical)
- `difficulty`: Maps to simulation budget tiers

**Budget tiers** (from `gumbel_common.py`):
| Tier | Budget | Use Case |
|------|--------|----------|
| THROUGHPUT | 64 | Fast bootstrap |
| STANDARD | 150 | Balanced |
| QUALITY | 800 | High quality training |
| ULTIMATE | 1600 | Maximum quality |

---

### GPU_GUMBEL

**Purpose**: GPU-accelerated Gumbel MCTS for batch selfplay. Uses tensor operations for parallel tree search.

**When to use**:

- High-throughput selfplay on GPU
- Batch evaluation of multiple positions
- When you have CUDA available

**Configuration**:

```python
harness = create_harness(
    HarnessType.GPU_GUMBEL,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
    simulations=200,
)
```

**Performance**: 6-57x speedup vs CPU depending on GPU and batch size.

---

### MINIMAX

**Purpose**: Alpha-beta minimax search. Classic game tree search with pruning.

**Restrictions**: **2-player games only**. Alpha-beta pruning assumes zero-sum, which doesn't hold for 3+ players.

**When to use**:

- Fast evaluation in 2-player games
- With NNUE models (value-only, no policy needed)
- When deterministic play is preferred

**Configuration**:

```python
harness = create_harness(
    HarnessType.MINIMAX,
    model_path="models/nnue_hex8_2p.pt",  # NNUE model
    board_type="hex8",
    num_players=2,  # Must be 2
    depth=4,  # Search depth
)
```

**Key parameters**:

- `depth`: Search depth (3-6 typical)
- Higher depth = better play, slower

---

### MAXN

**Purpose**: Max-N search for multiplayer games. Each player maximizes their own score.

**Restrictions**: **3-4 players only**. For 2-player games, use MINIMAX instead.

**When to use**:

- Multiplayer games (3-4 players)
- When thorough search is needed
- With either NN or NNUE evaluation

**Configuration**:

```python
harness = create_harness(
    HarnessType.MAXN,
    model_path="models/canonical_hex8_4p.pth",
    board_type="hex8",
    num_players=4,  # Must be 3 or 4
    depth=3,
)
```

---

### BRS (Best-Reply Search)

**Purpose**: Fast greedy multiplayer search. Assumes opponents play best responses.

**Restrictions**: **3-4 players only**.

**When to use**:

- Fast multiplayer evaluation
- When speed matters more than optimality
- Tournament play with time limits

**Configuration**:

```python
harness = create_harness(
    HarnessType.BRS,
    model_path="models/canonical_hex8_3p.pth",
    board_type="hex8",
    num_players=3,  # Must be 3 or 4
    depth=4,
)
```

**Comparison with MAXN**:

- BRS: Faster, assumes opponents cooperate against you
- MAXN: Slower, models each player independently

---

### POLICY_ONLY

**Purpose**: Direct policy sampling without tree search. Uses the neural network's policy head directly.

**When to use**:

- Fast baseline play
- When exploring move diversity
- Quick sanity checks

**Configuration**:

```python
harness = create_harness(
    HarnessType.POLICY_ONLY,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
)
```

**Note**: Requires a model with policy head. NNUE without policy won't work.

---

### DESCENT

**Purpose**: Gradient descent on value function for move selection. Experimental/research use.

**When to use**:

- Exploring alternative move selection strategies
- Research into gradient-based planning
- When you want to study value landscapes

**Configuration**:

```python
harness = create_harness(
    HarnessType.DESCENT,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
)
```

**Note**: Requires full neural network (not NNUE).

---

### HEURISTIC

**Purpose**: Hand-crafted heuristic evaluation only. No neural network required.

**When to use**:

- Bootstrapping (no models yet)
- Baseline opponent for gauntlet
- Fast sanity checks
- Environments without GPU

**Configuration**:

```python
harness = create_harness(
    HarnessType.HEURISTIC,
    board_type="hex8",
    num_players=2,
    # No model_path needed
)
```

---

## Model Type Compatibility

| Model Type        | Description               | Compatible Harnesses              |
| ----------------- | ------------------------- | --------------------------------- |
| `NEURAL_NET` (NN) | Full neural network v2-v6 | All except HEURISTIC              |
| `NNUE`            | Efficiently Updatable NN  | MINIMAX, MAXN, BRS, POLICY_ONLY\* |
| `HEURISTIC`       | Hand-crafted evaluation   | HEURISTIC only                    |

\*NNUE with policy head only

### Auto-Detection

The `create_harness()` factory auto-detects model type from the path:

- `*nnue*` in path → `ModelType.NNUE`
- No path → `ModelType.HEURISTIC`
- Otherwise → `ModelType.NEURAL_NET`

Override with `model_type` parameter if needed:

```python
harness = create_harness(
    HarnessType.MINIMAX,
    model_path="models/custom_model.pth",
    model_type=ModelType.NNUE,  # Explicit override
    ...
)
```

---

## Factory Function Reference

```python
from app.ai.harness import create_harness, HarnessType, ModelType

harness = create_harness(
    harness_type: HarnessType,      # Required: which algorithm
    model_path: str | Path | None,  # Path to model checkpoint
    model_id: str = "",             # ID for Elo tracking
    board_type: BoardType | None,   # Board configuration
    num_players: int = 2,           # 2, 3, or 4
    difficulty: int = 5,            # 1-10 difficulty
    think_time_ms: int | None,      # Time limit per move
    simulations: int = 200,         # MCTS simulations
    depth: int = 3,                 # Search depth (minimax)
    model_type: ModelType | None,   # Override auto-detection
    extra: dict | None,             # Harness-specific options
)
```

---

## Common Patterns

### Selfplay Data Generation

```python
# High-quality training data
harness = create_harness(
    HarnessType.GUMBEL_MCTS,
    model_path="models/canonical_hex8_2p.pth",
    board_type="hex8",
    num_players=2,
    simulations=800,  # QUALITY tier
)

for _ in range(num_games):
    game_state = initial_state()
    while not game_state.is_terminal:
        move, metadata = harness.evaluate(game_state, current_player)
        # metadata.visit_distribution gives soft policy targets
        record_training_sample(game_state, move, metadata)
        game_state = apply_move(game_state, move)
    harness.reset()  # Clear state between games
```

### Gauntlet Evaluation

```python
from app.ai.harness import get_harnesses_for_model_and_players

# Get all compatible harnesses for evaluation
harnesses = get_harnesses_for_model_and_players(
    model_type=ModelType.NEURAL_NET,
    num_players=4,
)
# Returns: [GUMBEL_MCTS, GPU_GUMBEL, MAXN, BRS, POLICY_ONLY, DESCENT]

for harness_type in harnesses:
    harness = create_harness(harness_type, model_path=model, ...)
    elo = run_gauntlet(harness)
    record_elo(model, harness_type, elo)
```

### Composite Participant ID

Each harness generates a composite ID for Elo tracking:

```python
harness = create_harness(...)
participant_id = harness.get_composite_participant_id()
# Format: "{model_id}:{harness_type}:{config_hash}"
# Example: "ringrift_hex8_2p:gumbel_mcts:d4abc123"
```

This ensures separate Elo ratings per (model, harness) combination.

---

## Common Pitfalls

### 1. Wrong Player Count

```python
# ERROR: Minimax with 4 players
harness = create_harness(
    HarnessType.MINIMAX,
    num_players=4,  # Will raise ValueError
    ...
)
```

**Fix**: Use MAXN or BRS for 3-4 player games.

### 2. NNUE with Policy-Required Harness

```python
# ERROR: NNUE without policy head + GUMBEL_MCTS
harness = create_harness(
    HarnessType.GUMBEL_MCTS,  # Requires policy
    model_path="models/nnue_value_only.pt",  # No policy
    ...
)
```

**Fix**: Use MINIMAX, MAXN, or BRS with value-only NNUE.

### 3. No Model for GUMBEL_MCTS

```python
# ERROR: No model path
harness = create_harness(
    HarnessType.GUMBEL_MCTS,
    model_path=None,  # Gumbel needs NN
    ...
)
```

**Fix**: Use HEURISTIC harness or provide a model path.

---

## Helper Functions

```python
from app.ai.harness import (
    get_harness_compatibility,      # Get HarnessCompatibility for type
    get_compatible_harnesses,       # Get all harnesses for model type
    get_harnesses_for_model_and_players,  # Filter by model + players
    get_harness_player_range,       # Get (min, max) players
    is_harness_valid_for_player_count,  # Check player count
    get_harness_matrix,             # Full compatibility matrix
)

# Example: Check if harness works for your setup
if is_harness_valid_for_player_count(HarnessType.MINIMAX, num_players=4):
    harness = create_harness(HarnessType.MINIMAX, ...)
else:
    harness = create_harness(HarnessType.MAXN, ...)  # Use multiplayer harness
```

---

## See Also

- `app/ai/harness/base_harness.py` - Base class and config
- `app/ai/harness/harness_registry.py` - Factory and compatibility
- `app/ai/harness/implementations.py` - Harness implementations
- `app/training/multi_harness_gauntlet.py` - Multi-harness evaluation
- `ai-service/CLAUDE.md` - Section on Harness Abstraction Layer
