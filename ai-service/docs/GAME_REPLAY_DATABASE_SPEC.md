# Game Replay Database Specification

## Overview

This document specifies the schema and API for a database storing complete
RingRift games from self-play for training, analysis, and replay functionality.

The database supports:

- Full game replay with step-by-step navigation (forward/backward)
- Efficient querying by game metadata (board type, winner, outcome type, etc.)
- Training data extraction for neural network and heuristic optimization
- Integration with the sandbox UI for game replay and analysis

## Design Principles

1. **Completeness**: Store all data needed to reconstruct any game state at any turn
2. **Compactness**: Avoid redundancy; derive state from initial state + move history
3. **Queryability**: Index on useful dimensions (board type, player count, outcome)
4. **Extensibility**: Schema supports future metadata (AI profiles, evaluation scores)

## Storage Format

### Option 1: SQLite Database (Recommended for local development)

A single SQLite database with the following tables:

### Option 2: JSONL Files with Index (Recommended for bulk storage)

- One JSONL file per board type with complete game records
- A separate SQLite index for fast metadata queries

---

## Schema Definition

### Table: `games`

Primary game metadata, one row per game.

| Column               | Type             | Description                                                          |
| -------------------- | ---------------- | -------------------------------------------------------------------- |
| `game_id`            | TEXT PRIMARY KEY | Unique game identifier (UUID)                                        |
| `board_type`         | TEXT NOT NULL    | 'square8', 'square19', 'hexagonal'                                   |
| `num_players`        | INTEGER NOT NULL | 2, 3, or 4                                                           |
| `rng_seed`           | INTEGER          | Seed used for any stochastic elements                                |
| `created_at`         | TIMESTAMP        | When the game was created                                            |
| `completed_at`       | TIMESTAMP        | When the game ended                                                  |
| `game_status`        | TEXT NOT NULL    | 'finished', 'completed', 'abandoned'                                 |
| `winner`             | INTEGER          | Player number of winner (NULL for draws)                             |
| `termination_reason` | TEXT             | 'ring_elimination', 'territory', 'last_player_standing', 'stalemate' |
| `total_moves`        | INTEGER NOT NULL | Number of moves in the game                                          |
| `total_turns`        | INTEGER NOT NULL | Number of full turn cycles                                           |
| `duration_ms`        | INTEGER          | Total game duration in milliseconds                                  |
| `source`             | TEXT             | 'self_play', 'human', 'tournament', 'training'                       |
| `schema_version`     | INTEGER NOT NULL | Schema version for forward compatibility                             |

**Indexes:**

- `idx_games_board_type` on `board_type`
- `idx_games_winner` on `winner`
- `idx_games_termination` on `termination_reason`
- `idx_games_created` on `created_at`

---

### Table: `game_players`

Per-player metadata for each game.

| Column                   | Type             | Description                                        |
| ------------------------ | ---------------- | -------------------------------------------------- |
| `game_id`                | TEXT NOT NULL    | FK to games.game_id                                |
| `player_number`          | INTEGER NOT NULL | 1, 2, 3, or 4                                      |
| `player_type`            | TEXT NOT NULL    | 'ai', 'human'                                      |
| `ai_type`                | TEXT             | 'heuristic', 'minimax', 'mcts', 'random', 'neural' |
| `ai_difficulty`          | INTEGER          | 1-10 difficulty level                              |
| `ai_profile_id`          | TEXT             | Heuristic weight profile or NN checkpoint ID       |
| `final_eliminated_rings` | INTEGER          | Rings eliminated by this player                    |
| `final_territory_spaces` | INTEGER          | Territory spaces controlled                        |
| `final_rings_in_hand`    | INTEGER          | Rings remaining in hand                            |

**Primary Key:** (`game_id`, `player_number`)

---

### Table: `game_initial_state`

Initial game state for reconstruction. Stored as JSON blob.

| Column               | Type                  | Description                      |
| -------------------- | --------------------- | -------------------------------- |
| `game_id`            | TEXT PRIMARY KEY      | FK to games.game_id              |
| `initial_state_json` | TEXT NOT NULL         | Full GameState as JSON           |
| `compressed`         | BOOLEAN DEFAULT FALSE | If TRUE, JSON is gzip compressed |

---

### Table: `game_moves`

Move history with full metadata for replay.

| Column          | Type             | Description                          |
| --------------- | ---------------- | ------------------------------------ |
| `game_id`       | TEXT NOT NULL    | FK to games.game_id                  |
| `move_number`   | INTEGER NOT NULL | 0-indexed move sequence              |
| `turn_number`   | INTEGER NOT NULL | Which turn this move belongs to      |
| `player`        | INTEGER NOT NULL | Player who made the move             |
| `phase`         | TEXT NOT NULL    | Game phase when move was made        |
| `move_type`     | TEXT NOT NULL    | MoveType enum value                  |
| `move_json`     | TEXT NOT NULL    | Full Move object as JSON             |
| `timestamp`     | TIMESTAMP        | When move was made                   |
| `think_time_ms` | INTEGER          | AI think time or human decision time |

**Primary Key:** (`game_id`, `move_number`)

**Indexes:**

- `idx_moves_game_turn` on (`game_id`, `turn_number`)

---

### Table: `game_state_snapshots`

Optional state snapshots at key points for fast seeking.

| Column        | Type                  | Description                        |
| ------------- | --------------------- | ---------------------------------- |
| `game_id`     | TEXT NOT NULL         | FK to games.game_id                |
| `move_number` | INTEGER NOT NULL      | Move number this snapshot is AFTER |
| `state_json`  | TEXT NOT NULL         | Full GameState as JSON             |
| `compressed`  | BOOLEAN DEFAULT FALSE | If TRUE, JSON is gzip compressed   |

**Primary Key:** (`game_id`, `move_number`)

**Note:** Snapshots are created every N moves (default: 20) to allow fast
seeking without replaying from the beginning.

---

### Table: `game_choices`

Player choices during decision phases (line reward, ring elimination, etc.).

| Column                 | Type             | Description                                                                          |
| ---------------------- | ---------------- | ------------------------------------------------------------------------------------ |
| `game_id`              | TEXT NOT NULL    | FK to games.game_id                                                                  |
| `move_number`          | INTEGER NOT NULL | Associated move number                                                               |
| `choice_type`          | TEXT NOT NULL    | 'line_reward', 'ring_elimination', 'line_order', 'region_order', 'capture_direction' |
| `player`               | INTEGER NOT NULL | Player who made the choice                                                           |
| `options_json`         | TEXT NOT NULL    | Available options as JSON array                                                      |
| `selected_option_json` | TEXT NOT NULL    | Selected option as JSON                                                              |
| `ai_reasoning`         | TEXT             | Optional: AI reasoning for the choice                                                |

**Primary Key:** (`game_id`, `move_number`, `choice_type`)

---

## Data Structures (JSON Schemas)

### GameState JSON

Matches the existing Pydantic `GameState` model serialization with these fields:

```json
{
  "id": "game_id",
  "boardType": "square8",
  "rngSeed": 12345,
  "board": {
    "type": "square8",
    "size": 8,
    "stacks": { "3,3": {...}, ... },
    "markers": { "2,2": {...}, ... },
    "collapsedSpaces": { "1,1": 1, ... },
    "eliminatedRings": { "1": 5, "2": 3 },
    "formedLines": [],
    "territories": {}
  },
  "players": [...],
  "currentPhase": "movement",
  "currentPlayer": 1,
  "moveHistory": [],  // Empty in initial_state, populated in snapshots
  "timeControl": {...},
  "gameStatus": "active",
  "winner": null,
  "createdAt": "2024-12-01T00:00:00Z",
  "lastMoveAt": "2024-12-01T00:00:00Z",
  "maxPlayers": 2,
  "totalRingsInPlay": 36,
  "totalRingsEliminated": 0,
  "victoryThreshold": 19,
  "territoryVictoryThreshold": 33,
  "lpsRoundIndex": 0,
  "lpsCurrentRoundActorMask": {},
  "rulesOptions": { "swapRuleEnabled": false }
}
```

### Move JSON

Matches the existing Pydantic `Move` model serialization.

---

## API Specification

### Python API (ai-service)

```python
from typing import Optional, List, Iterator
from app.models import GameState, Move, BoardType

class GameReplayDB:
    """Database interface for game storage and replay."""

    def __init__(self, db_path: str):
        """Initialize database connection."""
        ...

    # Write operations
    def store_game(
        self,
        game_id: str,
        initial_state: GameState,
        final_state: GameState,
        moves: List[Move],
        choices: List[dict],
        metadata: dict,
    ) -> None:
        """Store a complete game with all associated data."""
        ...

    def store_game_incremental(
        self,
        game_id: str,
        initial_state: GameState,
    ) -> "GameWriter":
        """Begin incremental game storage (for live games)."""
        ...

    # Read operations
    def get_game_metadata(self, game_id: str) -> Optional[dict]:
        """Get game metadata without loading full state."""
        ...

    def get_initial_state(self, game_id: str) -> Optional[GameState]:
        """Get the initial game state."""
        ...

    def get_moves(
        self,
        game_id: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> List[Move]:
        """Get moves in a range."""
        ...

    def get_state_at_move(
        self,
        game_id: str,
        move_number: int,
    ) -> Optional[GameState]:
        """Reconstruct state at a specific move number."""
        ...

    def get_choices_at_move(
        self,
        game_id: str,
        move_number: int,
    ) -> List[dict]:
        """Get player choices made at a specific move."""
        ...

    # Query operations
    def query_games(
        self,
        board_type: Optional[BoardType] = None,
        num_players: Optional[int] = None,
        winner: Optional[int] = None,
        termination_reason: Optional[str] = None,
        source: Optional[str] = None,
        min_moves: Optional[int] = None,
        max_moves: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Query games by metadata filters."""
        ...

    def iterate_games(
        self,
        **filters,
    ) -> Iterator[tuple[dict, GameState, List[Move]]]:
        """Iterate over games matching filters (for bulk processing)."""
        ...

    # Maintenance
    def vacuum(self) -> None:
        """Optimize database storage."""
        ...

    def get_stats(self) -> dict:
        """Get database statistics."""
        ...


class GameWriter:
    """Incremental game writer for live games."""

    def add_move(self, move: Move) -> None:
        """Add a move to the game."""
        ...

    def add_choice(
        self,
        move_number: int,
        choice_type: str,
        options: List[dict],
        selected: dict,
        reasoning: Optional[str] = None,
    ) -> None:
        """Record a player choice."""
        ...

    def finalize(
        self,
        final_state: GameState,
        metadata: dict,
    ) -> None:
        """Finalize and close the game record."""
        ...

    def abort(self) -> None:
        """Abort an incomplete game."""
        ...
```

### TypeScript Integration (Sandbox Replay)

The sandbox UI can consume replays via a REST API:

```typescript
// GET /api/replay/:gameId/metadata
interface GameMetadata {
  gameId: string;
  boardType: BoardType;
  numPlayers: number;
  winner: number | null;
  terminationReason: string;
  totalMoves: number;
  createdAt: string;
  players: PlayerMetadata[];
}

// GET /api/replay/:gameId/state?moveNumber=N
interface ReplayState {
  gameState: GameState;
  moveNumber: number;
  availableChoices?: Choice[]; // Choices available at this state
}

// GET /api/replay/:gameId/moves?start=0&end=100
interface MovesResponse {
  moves: Move[];
  hasMore: boolean;
}
```

---

## Integration with Self-Play

### Automatic Storage Hook

The self-play soak and pool generation scripts will automatically store
completed games:

```python
# In run_self_play_soak.py

def on_game_complete(
    game_id: str,
    initial_state: GameState,
    final_state: GameState,
    moves: List[Move],
    choices: List[dict],
    outcome: str,
    metadata: dict,
) -> None:
    """Called when a game completes successfully."""
    if db is not None and final_state.game_status in (GameStatus.FINISHED, GameStatus.COMPLETED):
        db.store_game(
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            choices=choices,
            metadata={
                **metadata,
                "source": "self_play",
                "termination_reason": outcome,
            },
        )
```

### Validation Before Storage

Games are only stored if they meet these criteria:

1. `game_status` is `FINISHED` or `COMPLETED` (not `ACTIVE` or `ABANDONED`)
2. A winner is determined (or valid stalemate with tiebreaker)
3. Move history is non-empty and consistent
4. Final state passes invariant checks (S is monotonic, no invalid termination)

---

## Replay Navigation in Sandbox

The sandbox UI supports game replay with these controls:

1. **Step Forward**: Apply next move, update state
2. **Step Backward**: Reconstruct state at previous move number
3. **Jump to Move**: Seek to any move number using snapshots + replay
4. **Play/Pause**: Automatic playback at configurable speed
5. **Choice Inspection**: View available choices and the selected option at decision points

### State Reconstruction Algorithm

```python
def get_state_at_move(game_id: str, target_move: int) -> GameState:
    """Reconstruct game state at a specific move."""

    # Find nearest snapshot before target
    snapshot = find_nearest_snapshot(game_id, target_move)

    if snapshot:
        state = GameState.model_validate_json(snapshot.state_json)
        start_move = snapshot.move_number + 1
    else:
        state = get_initial_state(game_id)
        start_move = 0

    # Replay moves from snapshot to target
    moves = get_moves(game_id, start=start_move, end=target_move + 1)
    for move in moves:
        state = GameEngine.apply_move(state, move)

    return state
```

---

## Storage Estimates

### Per-Game Storage

| Component                 | Size (approx)      |
| ------------------------- | ------------------ |
| Initial state JSON        | 2-10 KB            |
| Move (avg)                | 200-500 bytes      |
| Snapshot (every 20 moves) | 2-10 KB            |
| Choices                   | 100-500 bytes each |

### Estimated Database Sizes

| Games     | Avg Moves | Est. Size     |
| --------- | --------- | ------------- |
| 1,000     | 50        | 50-100 MB     |
| 10,000    | 50        | 500 MB - 1 GB |
| 100,000   | 50        | 5-10 GB       |
| 1,000,000 | 50        | 50-100 GB     |

---

## Migration Path

### Phase 1: Core Implementation

1. Implement SQLite schema and GameReplayDB class
2. Add storage hook to self-play soak script
3. Validate storage/retrieval with existing self-play runs

### Phase 2: Query and Analysis

1. Implement query_games and iterate_games
2. Add training data extraction utilities
3. Build CLI tools for database inspection

### Phase 3: Sandbox Integration

1. Add REST API endpoints for replay
2. Implement sandbox replay UI controls
3. Add choice inspection visualization

---

## File Locations

- Database file: `data/games/ringrift_games.db`
- Backup location: `data/games/backups/`
- Schema migrations: `app/db/migrations/`
- API implementation: `app/db/game_replay.py`

---

## Version History

| Version | Date       | Changes               |
| ------- | ---------- | --------------------- |
| 1.0     | 2024-12-01 | Initial specification |
