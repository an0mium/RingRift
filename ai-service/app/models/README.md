# Models Module

Data models, neural network architectures, and model management for the RingRift AI service. This module provides the **canonical Python representation** of game state mirroring TypeScript types.

## Table of Contents

1. [Overview](#overview)
2. [Core Game Models](#core-game-models)
   - [Enums](#enums)
   - [Position & Board](#position--board)
   - [Game State](#game-state)
   - [AI Configuration](#ai-configuration)
3. [Game Records](#game-records)
   - [Record Types](#record-types)
   - [RingRift Notation](#ringrift-notation-rrn)
4. [Neural Network Architectures](#neural-network-architectures)
   - [Transformer Models](#transformer-models)
   - [Multi-Task Learning](#multi-task-learning)
5. [Model Discovery](#model-discovery)
6. [Model Loading](#model-loading)
7. [Usage Examples](#usage-examples)
8. [Integration](#integration)

---

## Overview

The models module provides three main capabilities:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Models Module                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│   │   Core Models   │  │   NN Architectures  │  │ Model Management │ │
│   │                 │  │                 │  │                 │    │
│   │ • GameState     │  │ • Transformer   │  │ • Discovery     │    │
│   │ • Move          │  │ • Multi-Task    │  │ • Loading       │    │
│   │ • BoardState    │  │ • CNN-Hybrid    │  │ • Caching       │    │
│   │ • AIConfig      │  │ • Linear Attn   │  │ • Sidecars      │    │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                      │
│   Mirrors TypeScript       Experimental         Production           │
│   src/shared/types/        architectures        model APIs           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Structure

| File                   | Lines | Description                                     |
| ---------------------- | ----- | ----------------------------------------------- |
| `core.py`              | ~1000 | Pydantic models mirroring TypeScript game types |
| `game_record.py`       | ~516  | Game record storage and RingRift Notation       |
| `multitask_heads.py`   | ~860  | Multi-task auxiliary prediction heads           |
| `transformer_model.py` | ~709  | Transformer neural network architectures        |
| `discovery.py`         | ~487  | Model discovery and sidecar management          |
| `loader.py`            | ~590  | Model loading with caching                      |
| `__init__.py`          | ~148  | Lazy-loading exports                            |

---

## Core Game Models

### Enums

The module provides canonical enumerations mirroring TypeScript types:

```python
from app.models import BoardType, GamePhase, GameStatus, MoveType, AIType

# Board types
BoardType.SQUARE8     # 8×8 (64 cells)
BoardType.SQUARE19    # 19×19 (361 cells)
BoardType.HEX8        # Radius 4 hex (61 cells)
BoardType.HEXAGONAL   # Radius 12 hex (469 cells)

# Game phases (per RR-CANON-R070)
GamePhase.RING_PLACEMENT       # Initial ring placement
GamePhase.MOVEMENT             # Stack movement
GamePhase.CAPTURE              # Overtaking capture
GamePhase.CHAIN_CAPTURE        # Multi-jump capture
GamePhase.LINE_PROCESSING      # Line reward decisions
GamePhase.TERRITORY_PROCESSING # Territory claim decisions
GamePhase.FORCED_ELIMINATION   # No-action forced elimination
GamePhase.GAME_OVER           # Terminal phase

# Move types
MoveType.PLACE_RING            # Place ring on board
MoveType.MOVE_STACK            # Move stack (canonical)
MoveType.OVERTAKING_CAPTURE    # Capture opponent stack
MoveType.CHOOSE_LINE_OPTION    # Line reward choice
MoveType.SWAP_SIDES            # Pie rule swap
# ... and many more for all game actions
```

#### AI Types

Extensive AI implementation types for the difficulty ladder:

```python
from app.models import AIType

# Classic AI
AIType.RANDOM           # Random legal moves
AIType.HEURISTIC        # Handcrafted evaluation
AIType.MINIMAX          # Paranoid search
AIType.MCTS             # Monte Carlo Tree Search

# GPU-Accelerated
AIType.GPU_MINIMAX      # Batched leaf evaluation
AIType.GUMBEL_MCTS      # AlphaZero Sequential Halving

# Neural Network
AIType.POLICY_ONLY      # Direct NN policy
AIType.DESCENT          # Gradient descent search
AIType.HYBRID_NN        # Fast heuristic + NN ranking

# Experimental
AIType.GMO              # Gradient Move Optimization
AIType.CAGE             # Constraint-Aware Graph Energy
AIType.GNN              # Graph Neural Network
```

### Position & Board

```python
from app.models import Position, RingStack, MarkerInfo, BoardState

# Position (2D or 3D for hex)
pos = Position(x=3, y=4)
hex_pos = Position(x=0, y=0, z=0)

# Position to string key (cached for performance)
key = pos.to_key()  # "3,4"

# Ring stack
stack = RingStack(
    position=pos,
    rings=[1, 2, 1],  # Player numbers bottom to top
    stack_height=3,
    cap_height=1,     # Rings of controlling player
    controlling_player=1,
)

# Markers
marker = MarkerInfo(
    player=1,
    position=pos,
    type="regular",  # or "collapsed"
)

# Board state
board = BoardState(
    type=BoardType.SQUARE8,
    size=8,
    stacks={"3,4": stack},
    markers={"2,2": marker},
    formed_lines=[],
    territories={},
)
```

### Game State

Complete game state matching TypeScript `GameState`:

```python
from app.models import GameState, Player, TimeControl

state = GameState(
    id="game-123",
    board_type=BoardType.SQUARE8,
    board=board,
    players=[player1, player2],
    current_phase=GamePhase.RING_PLACEMENT,
    current_player=0,
    move_history=[],
    time_control=TimeControl(initial_time=600000, increment=5000, type="fischer"),
    game_status=GameStatus.ACTIVE,
    max_players=2,
    victory_threshold=25,
    territory_victory_threshold=33,
    lps_rounds_required=3,  # Last Player Standing config
    # ... many more fields
)
```

### AI Configuration

Extensive configuration for AI behavior:

```python
from app.models import AIConfig

config = AIConfig(
    difficulty=7,
    think_time=5000,
    randomness=0.1,

    # Self-play exploration
    self_play=True,
    root_dirichlet_alpha=0.3,
    root_noise_fraction=0.25,
    temperature=1.0,

    # Neural network
    use_neural_net=True,
    nn_model_id="hex8_2p_v2",

    # Gumbel MCTS
    gumbel_num_sampled_actions=16,
    gumbel_simulation_budget=100,
    use_gpu_tree=True,

    # Hybrid evaluation
    heuristic_blend_alpha=0.6,  # 60% NN, 40% heuristic
    heuristic_fallback_enabled=True,

    # NNUE policy priors
    use_policy_ordering=True,
    use_nnue_policy_priors=True,
)
```

---

## Game Records

### Record Types

Complete game record format for training data and replay:

```python
from app.models.game_record import (
    GameRecord,
    MoveRecord,
    PlayerRecordInfo,
    GameOutcome,
    RecordSource,
)

# Player information
player = PlayerRecordInfo(
    player_number=0,
    username="gumbel_mcts",
    player_type="ai",
    rating_before=1500,
    rating_after=1525,
    ai_difficulty=8,
    ai_type="gumbel_mcts",
)

# Move record (lightweight for storage)
move = MoveRecord(
    move_number=1,
    player=0,
    type=MoveType.PLACE_RING,
    to=Position(x=4, y=4),
    think_time_ms=150,
    mcts_policy={4*8+4: 0.8, 3*8+3: 0.1, 5*8+5: 0.1},  # MCTS visit distribution
)

# Complete game record
record = GameRecord(
    id="game-abc123",
    board_type=BoardType.SQUARE8,
    num_players=2,
    rng_seed=12345,
    is_rated=False,
    players=[player1, player2],
    winner=0,
    outcome=GameOutcome.TERRITORY_CONTROL,
    moves=[move1, move2, ...],
    total_moves=85,
    metadata=GameRecordMetadata(
        source=RecordSource.SELF_PLAY,
        fsm_validated=True,
    ),
)

# Serialize for training pipeline
jsonl_line = record.to_jsonl_line()

# Deserialize
record = GameRecord.from_jsonl_line(jsonl_line)
```

### RingRift Notation (RRN)

Compact notation for moves and game records:

```python
from app.models.game_record import (
    RRNMove,
    RRNCoordinate,
    game_record_to_rrn,
    rrn_to_moves,
    parse_rrn_move,
)

# Notation format:
# - Placement: Pa1, Pa1x2 (multi-ring)
# - Movement: e4-e6
# - Capture: d4xd5-d6
# - Chain capture: d4xd5-d6+
# - Line processing: La3
# - Territory: Tb2
# - Skip: -
# - Swap sides: S

# Convert game record to RRN
rrn_string = game_record_to_rrn(record)
# "square8:2:12345:Pa4 Ph5 d4-d6 d8-d4 d6xd5-d4"

# Parse RRN string
board_type, num_players, seed, moves = rrn_to_moves(rrn_string)

# Parse single move
move_type, from_pos, to_pos = parse_rrn_move("e4-e6", BoardType.SQUARE8)

# Coordinate conversion
coord = RRNCoordinate.from_position(Position(x=0, y=0), BoardType.SQUARE8)
print(coord.notation)  # "a1"

coord = RRNCoordinate.from_position(Position(x=0, y=0, z=0), BoardType.HEX8)
print(coord.notation)  # "(0,0,0)"
```

---

## Neural Network Architectures

### Transformer Models

Attention-based architectures for board game AI:

```python
from app.models import TransformerConfig, create_transformer_model

# Create transformer model
config = TransformerConfig(
    board_size=(8, 8),
    input_channels=18,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    mlp_ratio=4.0,
    dropout=0.1,
    use_cls_token=True,
    positional_encoding_type="learned",  # or "2d", "sinusoidal"
    num_policy_actions=64,
)

# Available architectures
model = create_transformer_model("pure", board_size=(8, 8))      # Pure Transformer
model = create_transformer_model("hybrid", board_size=(8, 8))    # CNN + Transformer
model = create_transformer_model("efficient", board_size=(8, 8)) # Linear attention

# Forward pass
policy, value = model(board_tensor)

# Get intermediate features for auxiliary heads
policy, value, features = model.forward_with_features(board_tensor)
```

#### Architecture Comparison

| Model                       | Complexity | Best For                 |
| --------------------------- | ---------- | ------------------------ |
| `BoardTransformer`          | O(n²)      | Maximum accuracy         |
| `HybridCNNTransformer`      | O(n²)      | Local + global reasoning |
| `EfficientBoardTransformer` | O(n)       | Large boards (19×19+)    |

### Multi-Task Learning

Auxiliary prediction heads to improve representation learning:

```python
from app.models import (
    AuxiliaryTask,
    TaskConfig,
    MultiTaskConfig,
    create_default_multitask_config,
)
from app.models.multitask_heads import MultiTaskModel, AuxiliaryDataGenerator

# Configure auxiliary tasks
config = MultiTaskConfig(
    tasks=[
        TaskConfig(AuxiliaryTask.MOVE_COUNT, loss_weight=0.05),
        TaskConfig(AuxiliaryTask.GAME_PHASE, loss_weight=0.1, num_classes=3),
        TaskConfig(AuxiliaryTask.TERRITORY_CONTROL, loss_weight=0.1),
        TaskConfig(AuxiliaryTask.DEFENSIVE_NEED, loss_weight=0.05),
    ],
    uncertainty_weighting=True,  # Learn loss weights automatically
    gradient_normalization=False,
)

# Or use defaults
config = create_default_multitask_config()

# Wrap base model
mt_model = MultiTaskModel(
    base_model=my_policy_value_model,
    config=config,
    feature_dim=256,
    board_size=(8, 8),
)

# Forward pass returns all outputs
outputs = mt_model(board_tensor)
# {
#     'policy': tensor,
#     'value': tensor,
#     'move_count': tensor,
#     'game_phase': tensor,
#     'territory': tensor,
#     'defense': tensor,
# }

# Compute losses
total_loss, individual_losses = mt_model.compute_losses(outputs, targets)

# Generate auxiliary labels from game state
labels = AuxiliaryDataGenerator.generate_labels(
    game_state,
    value=0.5,
    board_size=(8, 8),
)
```

#### Available Auxiliary Tasks

| Task                | Output      | Use Case                     |
| ------------------- | ----------- | ---------------------------- |
| `MOVE_COUNT`        | Scalar      | Estimate remaining moves     |
| `GAME_PHASE`        | 3-class     | Opening/midgame/endgame      |
| `TERRITORY_CONTROL` | Board-sized | Influence map                |
| `THREAT_DETECTION`  | Multi-label | Identify threats             |
| `LINE_COMPLETION`   | 2-class     | Line completion probability  |
| `DEFENSIVE_NEED`    | Scalar      | Urgency of defense           |
| `WINNING_PATH`      | Board-sized | Moves on winning path        |
| `POSITION_TYPE`     | 3-class     | Tactical/positional/balanced |

---

## Model Discovery

Unified API for finding models across the codebase:

```python
from app.models.discovery import (
    ModelInfo,
    discover_models,
    get_model_info,
    write_model_sidecar,
    generate_all_sidecars,
)

# Discover all models
all_models = discover_models()

# Filter by configuration
hex8_models = discover_models(
    board_type="hex8",
    num_players=2,
    model_type="nn",  # or "nnue"
)

# Get info for a specific model
info = get_model_info(Path("models/my_model.pth"))
print(f"Board: {info.board_type}, Players: {info.num_players}")
print(f"Source: {info.source}")  # "sidecar", "checkpoint", or "filename"

# Write sidecar JSON for better discovery
write_model_sidecar(
    model_path=Path("models/my_model.pth"),
    board_type="square8",
    num_players=2,
    elo=1650.0,
    architecture_version="v2",
)

# Generate sidecars for all models
count = generate_all_sidecars(overwrite=False)
print(f"Generated {count} sidecars")
```

### Detection Priority

1. **Sidecar JSON** (fastest, most reliable) - `model.pth.json`
2. **Filename parsing** (fast) - Extracts from naming convention
3. **Checkpoint metadata** (slow) - Loads checkpoint to read `_versioning_metadata`

### ModelInfo Fields

```python
@dataclass
class ModelInfo:
    path: str               # Full path to model file
    name: str               # Model name (stem)
    model_type: str         # "nn" or "nnue"
    board_type: str         # "square8", "hex8", etc.
    num_players: int        # 2, 3, or 4
    elo: float | None       # ELO rating if known
    architecture_version: str | None
    created_at: str | None
    size_bytes: int
    source: str             # How board_type was determined
```

---

## Model Loading

Production model loading with caching:

```python
from app.models.loader import (
    ModelLoader,
    ModelCache,
    get_model,
    get_latest_model,
    clear_model_cache,
)

# Use convenience functions
model, info = get_latest_model(
    board_type="square8",
    num_players=2,
    model_type="nnue",
    stage="production",
)

model, info = get_model(
    model_id="hex8_2p_v3",
    model_type="nnue",
    board_type="hex8",
    num_players=2,
)

# Or use loader directly
loader = ModelLoader(
    base_path=Path("/path/to/ai-service"),
    use_cache=True,
    device="cuda",
)

# Load NNUE model
model, info = loader.load_nnue("square8", 2, stage="production")

# Load policy model
model, info = loader.load_policy("hex8", 2)

# List available models
available = loader.get_available_models(board_type="square8")
print(available["nnue"])  # [{"name": "...", "path": "...", "size_mb": ...}]

# Clear cache and free memory
loader.clear_cache()
# or
clear_model_cache()
```

### ModelCache

Thread-safe LRU cache with configurable limits:

```python
from app.models.loader import ModelCache

cache = ModelCache()

# Default limits
cache.MAX_NNUE_MODELS = 4
cache.MAX_POLICY_MODELS = 2
cache.MAX_VALUE_MODELS = 2

# Get cache stats
stats = cache.stats()
print(stats)  # {"nnue_models": 2, "policy_models": 1, "value_models": 0}

# Clear cache
cache.clear()
```

---

## Usage Examples

### Training Data Pipeline

```python
from app.models import GameState, Move, MoveType
from app.models.game_record import GameRecord, MoveRecord

# Convert game to training format
def game_to_record(game_state: GameState, moves: list[Move]) -> GameRecord:
    move_records = [
        MoveRecord(
            move_number=i,
            player=m.player,
            type=m.type,
            from_pos=m.from_pos,
            to=m.to,
            think_time_ms=m.think_time or 0,
            mcts_policy=getattr(m, 'mcts_policy', None),
        )
        for i, m in enumerate(moves)
    ]

    return GameRecord(
        id=game_state.id,
        board_type=game_state.board_type,
        num_players=len(game_state.players),
        moves=move_records,
        # ... other fields
    )
```

### Multi-Task Training

```python
from app.models.multitask_heads import MultiTaskModel, GradNormBalancer

# Setup model with auxiliary heads
mt_model = MultiTaskModel(base_model, config)

# Setup GradNorm for gradient balancing
balancer = GradNormBalancer(
    mt_model,
    task_names=["policy", "value", "move_count", "game_phase"],
    alpha=1.5,
)

# Training loop
for batch in dataloader:
    outputs = mt_model(batch["board"])

    total_loss, losses = mt_model.compute_losses(outputs, batch)

    optimizer.zero_grad()
    total_loss.backward()

    # Update task weights based on gradient norms
    balancer.update_weights(losses, shared_layer=mt_model.base_model.backbone)

    optimizer.step()

    # Log task weights
    weights = mt_model.get_task_weights()
    print(f"Task weights: {weights}")
```

### AI Service Integration

```python
from app.models import GameState, AIConfig, AIType
from app.models.loader import get_latest_model

# Load model for AI service
model, info = get_latest_model(
    board_type="square8",
    num_players=2,
    model_type="nnue",
)

# Configure AI
config = AIConfig(
    difficulty=8,
    use_neural_net=True,
    nn_state_dict=model.state_dict(),  # In-memory weights
    gumbel_simulation_budget=150,
)

# Use in AI decision making
# (Actual AI implementation in app/ai/)
```

---

## Integration

### TypeScript Parity

All core models mirror TypeScript types:

| Python                 | TypeScript                           |
| ---------------------- | ------------------------------------ |
| `app.models.GameState` | `src/shared/types/game.ts:GameState` |
| `app.models.Move`      | `src/shared/types/game.ts:Move`      |
| `app.models.BoardType` | `src/shared/types/game.ts:BoardType` |
| `app.models.GamePhase` | `src/shared/types/game.ts:GamePhase` |
| `app.models.MoveType`  | `src/shared/types/game.ts:MoveType`  |

### Pydantic Configuration

All models use `populate_by_name=True` for camelCase/snake_case interop:

```python
# JSON with camelCase
json_data = {"boardType": "square8", "currentPlayer": 0}

# Python model uses snake_case
state = GameState.model_validate(json_data)
print(state.board_type)  # "square8"
print(state.current_player)  # 0

# Serialize back to camelCase
json_out = state.model_dump(by_alias=True, mode="json")
```

### Lazy Loading

Torch-dependent modules load lazily to avoid import overhead:

```python
from app.models import AuxiliaryTask  # No torch import yet

# Torch imports when needed
from app.models import MultiTaskConfig  # Now torch loads
```

---

## See Also

- `app/ai/README.md` - AI implementations using these models
- `app/training/README.md` - Training pipeline
- `app/db/README.md` - Database storage
- `src/shared/types/game.ts` - TypeScript source of truth

---

_Last updated: December 2025_
