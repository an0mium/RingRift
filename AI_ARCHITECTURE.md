# RingRift AI Architecture & Strategy

**Last Updated:** November 21, 2025
**Scope:** AI Service, Algorithms, Training Pipeline, and Integration

This document consolidates the architectural overview, technical assessment, and improvement plans for the RingRift AI system. It serves as the definitive guide for AI development.

---

## 1. Architecture Overview

### System Context

The AI system operates as a dedicated microservice (`ai-service`) built with Python/FastAPI, communicating with the main Node.js backend via HTTP.

- **Microservice:** `ai-service/` (Python 3.11+)
- **Communication:** REST API (`/ai/move`, `/ai/evaluate`, `/rules/evaluate_move`)
- **Integration:**
  - `AIEngine` (TypeScript) delegates to `AIServiceClient` for AI moves.
  - `RulesBackendFacade` (TypeScript) delegates to `PythonRulesClient` for rules validation (shadow/authoritative modes).
- **Fallback:** Local heuristics in `AIEngine` provide immediate responses if the service is unavailable or for simple decisions.
- **UI Integration:** Full lobby and game UI support for AI opponent configuration and visualization.

### Difficulty-to-AI-Type Mapping

The system provides a unified difficulty scale (1-10) that automatically selects the appropriate AI type:

| Difficulty | Label        | AI Type     | Description                                                |
| ---------- | ------------ | ----------- | ---------------------------------------------------------- |
| 1-2        | Beginner     | RandomAI    | Random move selection with filtering                       |
| 3-5        | Intermediate | HeuristicAI | Rule-based evaluation (stack control, territory, mobility) |
| 6-8        | Advanced     | MinimaxAI   | Alpha-beta search with move ordering                       |
| 9-10       | Expert       | MCTSAI      | Monte Carlo Tree Search with neural network guidance       |

Users can optionally override the AI type for specific testing or gameplay scenarios.

### AI Implementations

**Production-supported tactical engines (behind the `AIType`/`AIServiceClient.AIType` enum):**

1.  **RandomAI** (`random`): Baseline engine for testing and very low difficulty.
2.  **HeuristicAI** (`heuristic`): Rule-based evaluation using weighted factors (stack control, territory, mobility).
3.  **MinimaxAI** (`minimax`): Alpha–beta search with move ordering and quiescence. **Note:** Currently implemented but not instantiated by default in the factory; mid-high difficulties currently fall back to HeuristicAI.
4.  **MCTSAI** (`mcts`): Monte Carlo Tree Search with PUCT and RAVE, using the shared neural network for value/policy where weights are available.
5.  **DescentAI** (`descent`): UBFM/Descent-style tree search that also consumes the shared neural network for guidance and learning logs.

**Supporting / experimental components:**

- **NeuralNetAI:** CNN-based evaluation (value and policy heads) shared across board types (8×8, 19×19, hex) and used internally by `MCTSAI` and `DescentAI`.
- Training-side helpers and analysis tools under `ai-service/app/training/` (self-play data generation, tournaments, overfit tests).

The Python `ai-service` exposes these tactical engines via the `AIType` enum, and the TypeScript backend selects them through [`AIServiceClient.AIType`](src/server/services/AIServiceClient.ts:16) and the profile-driven mapping in [`AIEngine`](src/server/game/ai/AIEngine.ts:26).

### Neural Network Status

- **Architecture:** ResNet-style CNN (10 residual blocks).
- **Input:** 10-channel board representation + 10 global features.
- **Output:** Value (scalar) and Policy (probability distribution over ~55k moves).
- **Training:** Basic training loop implemented (`train.py`), but data generation (`generate_data.py`) needs improvement to use self-play with the current best model.

### UI Integration

**Lobby (Game Creation)**

- AI opponent configuration panel with visual difficulty selector
- Support for 0-3 AI opponents per game
- Difficulty slider (1-10) with clear labels (Beginner/Intermediate/Advanced/Expert)
- Optional AI type and control mode overrides
- Clear visual feedback showing AI configuration before game creation

**Game Display**

- AI opponent indicator badges in game header and player cards
- Color-coded difficulty labels (green=Beginner, blue=Intermediate, purple=Advanced, red=Expert)
- AI type display (Random/Heuristic/Minimax/MCTS)
- Animated "thinking" indicators during AI turns
- Distinct styling for AI players vs human players

**Game Lifecycle**

- AI games auto-start immediately upon creation (no waiting for human opponents)
- AI moves are automatically triggered by GameSession when it's an AI player's turn
- AI games are unrated by default to prevent rating manipulation

---

## 2. Technical Assessment & Code Review

### Operational Stability

- **Status:** **Stable**
- **Verification:** Environment setup (`setup.sh`, `run.sh`) and dependencies are correct. Unit tests for MCTS (`tests/test_mcts_ai.py`) are passing.
- **Issues:** Runtime warning regarding model architecture mismatch (handled via fallback).

### Component Analysis

#### Heuristic AI (`heuristic_ai.py`)

- **Status:** **Improved**
- **Optimizations:** Mobility evaluation bottleneck resolved using "pseudo-mobility" heuristic.
- **Issues:** Hardcoded weights make dynamic tuning difficult. Redundant line-of-sight logic.

#### Minimax AI (`minimax_ai.py`)

- **Status:** **Significantly Improved**
- **Optimizations:** Safe time management, enhanced move ordering (MVV-LVA), optimized quiescence search.
- **Critical Issue:** Zobrist hashing is O(N) instead of O(1), negating transposition table benefits.
- **Wiring Gap:** MinimaxAI is not currently instantiated in `_create_ai_instance` (commented out), so requests for it fall back to HeuristicAI.

#### MCTS Agent (`mcts_ai.py`)

- **Status:** **Improved**
- **Strengths:** Implements PUCT with RAVE heuristics. Supports batched inference.
- **Weaknesses:** Fallback rollout policy is weak. Tree reuse is not fully implemented. State copying during simulation is expensive.

#### Neural Network (`neural_net.py`)

- **Strengths:** ResNet-style CNN with adaptive pooling.
- **Weaknesses:** Architecture mismatch with saved checkpoint. History handling in training pipeline is flawed (see below).

---

## 5. RNG Determinism & Replay System

### Overview

RingRift implements comprehensive per-game RNG seeding to enable:

- **Deterministic replay:** Same seed + same inputs = same outputs
- **Cross-language parity:** TypeScript and Python produce identical sequences
- **Debugging:** Reproducible AI behavior for troubleshooting
- **Tournament validation:** Verify results by replaying games
- **Testing:** Reliable parity tests between engines

### Architecture

**Per-Game Seeding:**

- Each [`GameState`](src/shared/types/game.ts:469) contains an optional `rngSeed` field
- Seeds are auto-generated during game creation if not explicitly provided
- Database schema includes `rngSeed` for persistence and replay

**TypeScript Implementation:**

- [`SeededRNG`](src/shared/utils/rng.ts:11) class using xorshift128+ algorithm
- Provides `next()`, `nextInt()`, `shuffle()`, and `choice()` methods
- Used by [`GameSession`](src/server/game/GameSession.ts:35), [`AIEngine`](src/server/game/ai/AIEngine.ts:135), and [`ClientSandboxEngine`](src/client/sandbox/ClientSandboxEngine.ts:128)

**Python Implementation:**

- Python's built-in `random.Random` class provides deterministic sequences
- [`BaseAI`](ai-service/app/ai/base.py:16) initializes per-instance `self.rng` from `AIConfig.rngSeed`
- All AI implementations (RandomAI, HeuristicAI, MinimaxAI, MCTSAI) use `self.rng` instead of global `random`

**API Integration:**

- [`AIServiceClient`](src/server/services/AIServiceClient.ts:106) propagates `gameState.rngSeed` to Python service
- Python `/ai/move` endpoint accepts optional `seed` parameter
- When seed is provided, creates per-request AI instance instead of caching

### Determinism Guarantees

**What is deterministic:**

- AI move selection with same seed + same game state
- Random tie-breaking in move evaluation
- MCTS exploration with same seed
- Line reward and territory processing choices

**What is NOT deterministic:**

- Network timing (latency, timeouts)
- Wall-clock timestamps
- Concurrent game execution order
- User input timing

### Testing

**TypeScript Tests:**

- [`RNGDeterminism.test.ts`](tests/unit/RNGDeterminism.test.ts:1): Core SeededRNG algorithm tests
- AI parity tests verify sandbox and backend produce identical sequences

**Python Tests:**

- [`test_determinism.py`](ai-service/tests/test_determinism.py:1): AI determinism with seeded configs
- Verify RandomAI, HeuristicAI produce identical moves with same seed

### Known Limitations

1. **Python Neural Network:** Some NN operations may use non-seeded GPU operations
2. **External Services:** Network calls introduce non-determinism in timing
3. **Process Isolation:** Python global state requires careful seed management

### Migration & Backward Compatibility

- **Existing games:** Migration sets `rngSeed` to NULL (games created before this feature)
- **API:** `seed` parameter is optional in all requests
- **Fallback:** Games without seed generate one automatically and log it for debugging
- **No breaking changes:** All existing code paths continue to work

---

## 3. Improvement Plan & Roadmap

### Phase 1: Enhanced Training Pipeline (Immediate)

1.  **Fix History Handling:** Update `generate_data.py` to store full stacked feature tensors or reconstruct history correctly. Remove hacks in `train.py`.
2.  **Self-Play Loop:** Update `generate_data.py` to use the latest model for self-play data generation.
3.  **Rich Policy Targets:** Modify MCTS to return visit counts/probabilities instead of just the best move.
4.  **Data Augmentation:** Implement dihedral symmetries (rotation/reflection) for board data.

### Phase 2: Engine Optimization (Short-term)

1.  **Incremental Hashing:** Fix Zobrist hashing in `MinimaxAI` to be truly incremental (O(1)).
2.  **Batched Inference:** Ensure MCTS evaluates leaf nodes in batches to maximize throughput.
3.  **Tree Reuse:** Implement MCTS tree persistence between moves.

### Phase 3: Architecture Refinement (Medium-term)

1.  **Input Features:** Add history planes (last 3-5 moves) to capture dynamics.
2.  **Network Size:** Experiment with MobileNet vs ResNet-50 for different difficulty levels.
3.  **In-place State Updates:** Refactor `GameEngine` or create a specialized `FastGameEngine` for MCTS to eliminate copying overhead.

### Phase 4: Production Readiness (Long-term)

1.  **Model Versioning:** Implement a system to manage and serve different model versions.
2.  **Async Inference:** Use an async task queue (e.g., Celery/Redis) for heavy AI computations.

---

## 4. Rules Completeness in AI Service

- **Status:** **Mostly Complete**
- **Implemented:** Ring placement, movement, capturing (including chains), line formation, territory claiming, forced elimination, and victory conditions.
- **Simplifications:**
  - **Line Formation:** Automatically chooses to collapse all markers and eliminate from the largest stack (biasing against "Option 2").
  - **Territory Claim:** Automatically claims territory without user choice nuances.

**Recommendation:** Update `GameEngine` to support branching for Line Formation and Territory Claim choices to fully match the game's strategic depth.
