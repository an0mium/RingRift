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
  - [`AIEngine`](src/server/game/ai/AIEngine.ts:135) (TypeScript) delegates to [`AIServiceClient`](src/server/services/AIServiceClient.ts:170) for AI moves.
  - [`RulesBackendFacade`](src/server/game/RulesBackendFacade.ts:1) (TypeScript) delegates to [`PythonRulesClient`](src/server/services/PythonRulesClient.ts:1) for rules validation (shadow/authoritative modes).
- **Resilience:** Multi-tier fallback system ensures games never get stuck due to AI failures.
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

## 6. Error Handling & Resilience

### Tiered Fallback Architecture

The AI system implements a robust three-tier fallback hierarchy to ensure games never get stuck due to AI service failures:

```
Level 1: Python AI Service (RemoteAI)
   ↓ (on failure: timeout, error, invalid move)
Level 2: Local Heuristic AI (TypeScript)
   ↓ (on failure: exception in local selection)
Level 3: Random Valid Move Selection
```

**Implementation:** [`AIEngine.getAIMove()`](src/server/game/ai/AIEngine.ts:228)

### Error Scenarios Handled

#### Network & Service Failures

- **Connection Refused:** AI service unreachable or not started
  - Circuit breaker opens after 5 consecutive failures
  - Automatic fallback to local heuristics
  - Service availability re-tested after 60-second cooldown

- **Timeouts:** AI service taking too long to respond
  - Default timeout: 30 seconds (configured in [`AIServiceClient`](src/server/services/AIServiceClient.ts:179))
  - Automatic fallback to local heuristics
  - Logged with latency metrics for monitoring

- **HTTP Errors:** Server errors (500, 503) from AI service
  - Categorized and logged with error type
  - Immediate fallback without retries
  - Circuit breaker tracks failure patterns

#### Invalid Move Responses

- **Move Validation:** All AI-suggested moves are validated against the legal move list from [`RuleEngine`](src/server/game/RuleEngine.ts:1)
  - Validates move type, player, positions, and special properties
  - Deep equality check including hexagonal coordinates
  - Invalid moves trigger automatic fallback

- **Malformed Responses:** AI service returns null or unparseable moves
  - Handled as service failure
  - Immediate fallback to local heuristics

- **Wrong Phase/Player:** AI suggests moves for incorrect game state
  - Caught by move validation
  - Fallback maintains game flow

### Circuit Breaker Pattern

**Implementation:** [`CircuitBreaker`](src/server/services/AIServiceClient.ts:20) class in AIServiceClient

**Behavior:**
- **Closed:** Normal operation, all requests attempt service
- **Opening:** After 5 consecutive failures within 60 seconds
- **Open:** Rejects requests immediately for 60 seconds
- **Half-Open:** After timeout, allows test request to check recovery

**Benefits:**
- Prevents hammering failing AI service
- Reduces cascade failures
- Automatic recovery detection
- Minimal latency when service is down

### Fallback Strategy

#### Level 1: Remote AI Service

- Uses Python microservice for sophisticated AI
- Supports all AI types (Random, Heuristic, Minimax, MCTS, Descent)
- Provides evaluation scores and thinking time metrics
- Protected by circuit breaker

#### Level 2: Local Heuristic AI

**Implementation:** [`AIEngine.selectLocalHeuristicMove()`](src/server/game/ai/AIEngine.ts:352)

- Uses shared [`chooseLocalMoveFromCandidates()`](src/shared/engine/localAIMoveSelection.ts:1)
- Prioritizes captures over movements
- Prefers moves that advance game state
- Deterministic with provided RNG
- Always produces valid moves

**Shared Policy:**
- Same heuristics used by sandbox AI and backend fallback
- Ensures consistent behavior across test/production
- Maintains game parity for debugging

#### Level 3: Random Selection

- Last resort when both service and heuristics fail
- Selects uniformly from valid moves using provided RNG
- Guarantees game progression
- Logs warning for monitoring

### Diagnostics & Monitoring

#### Per-Player Diagnostics

**[`AIDiagnostics`](src/server/game/ai/AIEngine.ts:50) Interface:**

```typescript
{
  serviceFailureCount: number;  // Times AI service failed
  localFallbackCount: number;   // Times local heuristic was used
}
```

**Access:** [`AIEngine.getDiagnostics(playerNumber)`](src/server/game/ai/AIEngine.ts:722)

#### Per-Game Quality Mode

[`GameSession`](src/server/game/GameSession.ts:42) tracks aggregate AI quality:

- `normal`: AI service working as expected
- `fallbackLocalAI`: Using local heuristics due to service issues
- `rulesServiceDegraded`: Python rules engine failures detected

**Access:** [`GameSession.getAIDiagnosticsSnapshotForTesting()`](src/server/game/GameSession.ts:832)

#### Logging

All AI failures are logged with context:

```typescript
logger.warn('Remote AI service failed, falling back to local heuristics', {
  error: error.message,
  playerNumber,
  difficulty,
});
```

**Log Levels:**
- `info`: Normal operation, successful fallbacks
- `warn`: Service failures, invalid moves, fallback usage
- `error`: Fatal errors, game abandonment

### Client-Side Error Handling

#### Error Events

[`GameSession`](src/server/game/GameSession.ts:759) emits `game_error` events when AI encounters fatal failures:

```typescript
socket.emit('game_error', {
  message: 'AI encountered a fatal error. Game cannot continue.',
  technical: error.message,
  gameId
});
```

#### UI Feedback

[`GamePage`](src/client/pages/GamePage.tsx:1) displays error banners:
- User-friendly error message
- Technical details in development mode
- Dismissible notification
- Game marked as completed with abandonment

### Sandbox AI Resilience

[`sandboxAI.ts`](src/client/sandbox/sandboxAI.ts:1) implements comprehensive error handling:

- Top-level try-catch in [`maybeRunAITurnSandbox()`](src/client/sandbox/sandboxAI.ts:437)
- Error recovery in [`selectSandboxMovementMove()`](src/client/sandbox/sandboxAI.ts:392)
- Fallback to random selection on errors
- Never propagates exceptions to game engine
- Logs all errors for debugging

### Testing

#### Unit Tests

[`tests/unit/AIEngine.fallback.test.ts`](tests/unit/AIEngine.fallback.test.ts:1):
- Service failure handling
- Invalid move rejection
- Circuit breaker behavior
- Move validation logic
- Diagnostics tracking
- RNG determinism

#### Integration Tests

[`tests/integration/AIResilience.test.ts`](tests/integration/AIResilience.test.ts:1):
- Complete game with AI service down
- Intermittent failures
- Circuit breaker integration
- Performance under failure
- Error recovery patterns

### Operational Monitoring

**Health Checks:**

Endpoint: `/health/ai-service` (when implemented)
- Checks [`AIServiceClient.healthCheck()`](src/server/services/AIServiceClient.ts:455)
- Returns status: `healthy`, `degraded`, or `unavailable`

**Metrics to Monitor:**

1. **AI Service Availability:** Success rate of AI service calls
2. **Fallback Usage:** Frequency of local heuristic usage
3. **Circuit Breaker State:** Open/closed status and failure counts
4. **Move Validation Failures:** Rate of invalid moves from AI service
5. **Random Fallback Usage:** Should be near zero in production

**Alert Thresholds:**

- Service availability < 95%: Investigate AI service health
- Fallback usage > 20%: Check network or service degradation
- Circuit breaker open: Critical - AI service down
- Invalid moves > 1%: AI service logic issue

### Known Limitations

1. **Fatal Failures:** If all three tiers fail (extremely rare), game is abandoned
2. **Quality Degradation:** Local heuristics are weaker than trained AI
3. **No Retry Logic:** Service failures trigger immediate fallback (by design for responsiveness)
4. **Circuit Breaker State:** Shared across all games (not per-game isolation)

### Future Enhancements

1. **Adaptive Timeout:** Adjust timeout based on AI type and difficulty
2. **Quality Metrics:** Track move quality when using fallbacks
3. **Graceful Degradation:** Warn users when AI quality is degraded
4. **Service Pool:** Load balance across multiple AI service instances
5. **Caching:** Cache positions for common opening/endgame patterns

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
