# Rules Engine

This module implements the Python rules engine for RingRift, mirroring the TypeScript game engine to ensure parity for training data generation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
   - [RulesEngine Interface](#rulesengine-interface)
   - [DefaultRulesEngine](#defaultrulesengine)
   - [Validators](#validators)
   - [Mutators](#mutators)
3. [Game State](#game-state)
4. [Move Processing](#move-processing)
5. [Phase Machine](#phase-machine)
6. [Parity with TypeScript](#parity-with-typescript)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The rules engine mirrors the TypeScript architecture with validators and mutators:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Rules Engine Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Move Input ─────► Validators ─────► Mutators ─────► New State     │
│                         │                │                          │
│                    (is_valid?)     (apply changes)                  │
│                         │                │                          │
│                         ▼                ▼                          │
│   Validators:      Mutators:                                        │
│   • Placement      • Placement                                      │
│   • Movement       • Movement                                       │
│   • Capture        • Capture                                        │
│   • Line           • Line                                           │
│   • Territory      • Territory                                      │
│                    • Turn                                           │
│                                                                     │
│   Supporting Modules:                                               │
│   • FSM (Finite State Machine for game phases)                     │
│   • MutableState (efficient state manipulation)                    │
│   • Geometry (board coordinate systems)                            │
│   • Serialization (state encoding/decoding)                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **TypeScript Parity**: Python rules must produce identical results to TS
2. **Validator/Mutator Split**: Validation separate from state mutation
3. **Shadow Contracts**: Continuous validation against canonical engine
4. **Immutable by Default**: State changes return new state objects

---

## Core Components

### RulesEngine Interface

The abstract interface all engines implement:

```python
from app.rules import RulesEngine, get_rules_engine

class RulesEngine:
    """Abstract interface for game rules."""

    def is_valid_move(self, state: GameState, move: Move) -> bool:
        """Check if a move is valid in current state."""
        ...

    def apply_move(self, state: GameState, move: Move) -> GameState:
        """Apply move and return new state."""
        ...

    def get_valid_moves(self, state: GameState) -> list[Move]:
        """Get all valid moves for current player."""
        ...

    def is_game_over(self, state: GameState) -> bool:
        """Check if game has ended."""
        ...

    def get_winner(self, state: GameState) -> int | None:
        """Get winning player index or None."""
        ...
```

### DefaultRulesEngine

The production implementation:

```python
from app.rules.default_engine import DefaultRulesEngine

engine = DefaultRulesEngine(
    mutator_first=False,      # Use canonical GameEngine
    skip_shadow_contracts=True,  # Skip validation for speed
)

# Check move validity
if engine.is_valid_move(state, move):
    new_state = engine.apply_move(state, move)

# Get all valid moves
moves = engine.get_valid_moves(state)
```

#### Shadow Contracts

Shadow contracts validate mutators against the canonical engine:

```python
# Enable shadow contracts for debugging
engine = DefaultRulesEngine(skip_shadow_contracts=False)

# Or via environment
# RINGRIFT_SKIP_SHADOW_CONTRACTS=false
```

When enabled, each mutator's output is compared to the canonical result, catching any parity issues.

### Validators

Validators check move legality without modifying state:

```python
from app.rules.validators.placement import PlacementValidator
from app.rules.validators.movement import MovementValidator
from app.rules.validators.capture import CaptureValidator
from app.rules.validators.line import LineValidator
from app.rules.validators.territory import TerritoryValidator

class PlacementValidator:
    def can_handle(self, move_type: MoveType) -> bool:
        return move_type == MoveType.PLACEMENT

    def is_valid(self, state: GameState, move: Move) -> bool:
        # Check cell is empty and in bounds
        return state.board.is_empty(move.cell) and state.board.in_bounds(move.cell)
```

| Validator            | Move Types      | Checks                              |
| -------------------- | --------------- | ----------------------------------- |
| `PlacementValidator` | PLACEMENT       | Empty cell, in bounds, phase allows |
| `MovementValidator`  | MOVEMENT        | Owns piece, valid destination       |
| `CaptureValidator`   | CAPTURE         | Adjacent enemy, valid capture rules |
| `LineValidator`      | LINE_CLAIM      | Valid line completion               |
| `TerritoryValidator` | TERRITORY_CLAIM | Enclosed territory                  |

### Mutators

Mutators apply state changes:

```python
from app.rules.mutators.placement import PlacementMutator
from app.rules.mutators.movement import MovementMutator
from app.rules.mutators.capture import CaptureMutator
from app.rules.mutators.line import LineMutator
from app.rules.mutators.territory import TerritoryMutator
from app.rules.mutators.turn import TurnMutator

class PlacementMutator:
    def can_handle(self, move_type: MoveType) -> bool:
        return move_type == MoveType.PLACEMENT

    def apply(self, state: MutableState, move: Move) -> None:
        # Place piece on board
        state.board.set_cell(move.cell, move.player_id)
        state.players[move.player_id].pieces_placed += 1
```

| Mutator            | Responsibilities         |
| ------------------ | ------------------------ |
| `PlacementMutator` | Place new pieces         |
| `MovementMutator`  | Move existing pieces     |
| `CaptureMutator`   | Remove captured pieces   |
| `LineMutator`      | Record line completions  |
| `TerritoryMutator` | Update territory control |
| `TurnMutator`      | Advance turn/phase       |

---

## Game State

### MutableState

Efficient mutable state for move application:

```python
from app.rules.mutable_state import MutableState

# Convert from immutable
mutable = MutableState.from_game_state(state)

# Make changes
mutable.board.set_cell(5, player_id=0)
mutable.players[0].score += 10
mutable.current_player = 1

# Convert back to immutable
new_state = mutable.to_game_state()
```

### State Components

| Component        | Description                            |
| ---------------- | -------------------------------------- |
| `board`          | Cell ownership grid                    |
| `players`        | Per-player stats (pieces, score, etc.) |
| `current_player` | Active player index                    |
| `phase`          | Current game phase                     |
| `turn`           | Turn number                            |
| `move_count`     | Total moves played                     |
| `lines`          | Completed lines                        |
| `territories`    | Claimed territories                    |

### Serialization

State serialization for storage:

```python
from app.rules.serialization import (
    serialize_state,
    deserialize_state,
    compute_state_hash,
)

# Serialize to JSON
json_str = serialize_state(state)

# Deserialize
state = deserialize_state(json_str)

# Compute hash for parity checking
hash_val = compute_state_hash(state)
```

---

## Move Processing

### Move Types

| Type              | Description          | Example                                       |
| ----------------- | -------------------- | --------------------------------------------- |
| `PLACEMENT`       | Place new piece      | {"type": "placement", "cell": 42}             |
| `MOVEMENT`        | Move existing piece  | {"type": "movement", "from": 10, "to": 15}    |
| `CAPTURE`         | Capture enemy piece  | {"type": "capture", "cell": 20}               |
| `LINE_CLAIM`      | Claim completed line | {"type": "line_claim", "line_id": 3}          |
| `TERRITORY_CLAIM` | Claim territory      | {"type": "territory_claim", "cells": [1,2,3]} |
| `PASS`            | Pass turn            | {"type": "pass"}                              |
| `RESIGN`          | Forfeit game         | {"type": "resign"}                            |

### Move Validation Flow

```python
def apply_move(state: GameState, move: Move) -> GameState:
    # 1. Find validator
    for validator in self.validators:
        if validator.can_handle(move.type):
            if not validator.is_valid(state, move):
                raise InvalidMoveError(f"Move {move} is invalid")
            break

    # 2. Create mutable copy
    mutable = MutableState.from_game_state(state)

    # 3. Apply mutators
    for mutator in self.mutators:
        if mutator.can_handle(move.type):
            mutator.apply(mutable, move)

    # 4. Update phase/turn
    self.phase_machine.advance(mutable, move)

    # 5. Return new immutable state
    return mutable.to_game_state()
```

---

## Phase Machine

The FSM manages game phase transitions:

```python
from app.rules.fsm import GameFSM, GamePhase
from app.rules.phase_machine import PhaseMachine

fsm = GameFSM()

# Get current phase requirements
requirements = fsm.get_phase_requirements(state)
print(f"Phase: {state.phase}")
print(f"Required moves: {requirements}")

# Check if phase complete
if fsm.is_phase_complete(state):
    next_phase = fsm.get_next_phase(state)
```

### Game Phases

| Phase       | Description             | Valid Moves             |
| ----------- | ----------------------- | ----------------------- |
| `PLACEMENT` | Initial piece placement | PLACEMENT               |
| `MAIN`      | Main gameplay           | MOVEMENT, CAPTURE, PASS |
| `SCORING`   | Final scoring           | TERRITORY_CLAIM         |
| `GAME_OVER` | Game ended              | None                    |

### Phase Transitions

```
PLACEMENT ──► MAIN ──► SCORING ──► GAME_OVER
    │           │          │
    └───────────┴──────────┴─► (early termination)
```

---

## Parity with TypeScript

### The Source of Truth

**TypeScript is the source of truth for game rules.** The Python rules engine must produce identical results.

### Parity Validation

```python
from app.db import validate_game_parity, ParityMode

# Enable strict parity checking
os.environ["RINGRIFT_PARITY_VALIDATION"] = "strict"

# Validate game replay
result = validate_game_parity(
    game_id="abc123",
    db_path="data/games/selfplay.db",
)

if not result.valid:
    print(f"Divergence at move {result.divergence_move}")
    print(f"Python hash: {result.python_hash}")
    print(f"TS hash: {result.typescript_hash}")
```

### Common Parity Issues

| Issue               | Cause           | Fix                  |
| ------------------- | --------------- | -------------------- |
| Hash mismatch       | Float precision | Use integer math     |
| Move order          | Set iteration   | Sort consistently    |
| Score difference    | Rounding        | Match TS rounding    |
| Missing elimination | Chain logic     | Update capture chain |

### Testing Parity

```bash
# Run parity tests
python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_hex8.db \
  --sample 100

# Run canonical parity gate
python scripts/run_canonical_selfplay_parity_gate.py \
  --board-type hex8 --num-players 2
```

---

## Configuration

### Environment Variables

| Variable                         | Description             | Default |
| -------------------------------- | ----------------------- | ------- |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | Skip mutator validation | `true`  |
| `RINGRIFT_RULES_MUTATOR_FIRST`   | Use mutator-first mode  | `false` |
| `RINGRIFT_PARITY_VALIDATION`     | Parity check mode       | `off`   |

### Performance Tuning

```python
# High-performance selfplay
engine = DefaultRulesEngine(
    mutator_first=False,
    skip_shadow_contracts=True,
)

# Development/debugging
engine = DefaultRulesEngine(
    mutator_first=True,
    skip_shadow_contracts=False,
)
```

---

## Usage Examples

### Basic Game Loop

```python
from app.rules import get_rules_engine
from app.models import GameState, Move

engine = get_rules_engine()

# Initialize game
state = GameState.create(board_type="hex8", num_players=2)

# Game loop
while not engine.is_game_over(state):
    # Get valid moves
    moves = engine.get_valid_moves(state)

    # Select move (AI or random)
    move = select_move(moves)

    # Apply move
    state = engine.apply_move(state, move)

# Get winner
winner = engine.get_winner(state)
print(f"Player {winner} wins!")
```

### Move Validation

```python
from app.rules import get_rules_engine
from app.models import Move, MoveType

engine = get_rules_engine()

# Check single move
move = Move(type=MoveType.PLACEMENT, cell=42, player_id=0)
if engine.is_valid_move(state, move):
    state = engine.apply_move(state, move)
else:
    print("Invalid move!")

# Filter valid placements
placements = [
    Move(type=MoveType.PLACEMENT, cell=i, player_id=state.current_player)
    for i in range(state.board.size)
    if engine.is_valid_move(state, Move(type=MoveType.PLACEMENT, cell=i, player_id=state.current_player))
]
```

### Custom Validator

```python
from app.rules.interfaces import Validator
from app.models import GameState, Move, MoveType

class CustomValidator(Validator):
    def can_handle(self, move_type: MoveType) -> bool:
        return move_type == MoveType.CUSTOM

    def is_valid(self, state: GameState, move: Move) -> bool:
        # Custom validation logic
        return True

# Register with engine
engine.validators.append(CustomValidator())
```

### State Inspection

```python
from app.rules.mutable_state import MutableState

# Analyze state
state = engine.apply_move(initial_state, move)

# Check board
for cell in range(state.board.size):
    owner = state.board.get_cell(cell)
    if owner is not None:
        print(f"Cell {cell}: Player {owner}")

# Check player stats
for i, player in enumerate(state.players):
    print(f"Player {i}: {player.pieces_placed} pieces, {player.score} score")

# Check lines
for line in state.lines:
    print(f"Line by player {line.owner}: {line.cells}")
```

---

## Troubleshooting

### Invalid Move Errors

```python
from app.rules import get_rules_engine

engine = get_rules_engine()

# Debug why move is invalid
move = Move(type=MoveType.PLACEMENT, cell=42, player_id=0)

if not engine.is_valid_move(state, move):
    # Check each validator
    for validator in engine.validators:
        if validator.can_handle(move.type):
            # Validator-specific debug
            print(f"{validator.__class__.__name__}: {validator.is_valid(state, move)}")
```

### Shadow Contract Failures

```bash
# Enable verbose shadow contract logging
RINGRIFT_SKIP_SHADOW_CONTRACTS=false \
PYTHONPATH=. python your_script.py

# Look for assertion errors comparing mutator vs canonical results
```

### Parity Debugging

```python
from app.rules.serialization import serialize_state, compute_state_hash

# Serialize states for comparison
python_state_json = serialize_state(python_state)
print(f"Python state: {python_state_json[:200]}...")

# Compare hashes
python_hash = compute_state_hash(python_state)
print(f"Python hash: {python_hash}")

# Export for TS comparison
with open("debug_state.json", "w") as f:
    f.write(python_state_json)
```

### Performance Issues

```python
import time
from app.rules import get_rules_engine

# Profile move generation
engine = get_rules_engine()

start = time.time()
for _ in range(1000):
    moves = engine.get_valid_moves(state)
elapsed = time.time() - start

print(f"1000 get_valid_moves: {elapsed:.2f}s ({elapsed/1000*1000:.2f}ms each)")

# Ensure shadow contracts are disabled for performance
engine = DefaultRulesEngine(skip_shadow_contracts=True)
```

---

## Module Reference

| Module                | Lines | Description                        |
| --------------------- | ----- | ---------------------------------- |
| `mutable_state.py`    | 80KB  | Efficient mutable game state       |
| `default_engine.py`   | 60KB  | Main rules engine implementation   |
| `recovery.py`         | 55KB  | State recovery from corrupted data |
| `fsm.py`              | 34KB  | Finite state machine for phases    |
| `phase_machine.py`    | 25KB  | Phase transition logic             |
| `serialization.py`    | 21KB  | State serialization                |
| `capture_chain.py`    | 16KB  | Capture chain resolution           |
| `core.py`             | 16KB  | Core types and utilities           |
| `placement.py`        | 12KB  | Placement rules                    |
| `elimination.py`      | 12KB  | Player elimination                 |
| `global_actions.py`   | 12KB  | Global game actions                |
| `geometry.py`         | 7KB   | Board geometry helpers             |
| `history_contract.py` | 5KB   | Move history validation            |
| `interfaces.py`       | 2KB   | Abstract interfaces                |
| `factory.py`          | 2KB   | Engine factory                     |

---

## See Also

- `src/shared/engine/` - TypeScript source of truth
- `app/game_engine/` - Canonical Python game engine
- `app/db/README.md` - Game storage and parity validation
- `scripts/check_ts_python_replay_parity.py` - Parity testing

---

_Last updated: December 2025_
