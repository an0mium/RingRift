# Sandbox Replay & ClientSandboxEngine Refactor Plan

**Status:** Approved for Implementation
**Author:** Architecture Team
**Created:** 2025-12-07
**Last Updated:** 2025-12-07
**Scope:** Sandbox replay/parity architecture, ClientSandboxEngine decomposition, long-term maintainability

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Architecture Options Analysis](#architecture-options-analysis)
5. [Recommended Solution](#recommended-solution)
6. [Decomposition Patterns Analysis](#decomposition-patterns-analysis)
7. [Detailed Implementation Plan](#detailed-implementation-plan)
8. [Interface Contracts](#interface-contracts)
9. [Testing Strategy](#testing-strategy)
10. [Risk Assessment & Mitigations](#risk-assessment--mitigations)
11. [Success Criteria](#success-criteria)
12. [Related Files & Dependencies](#related-files--dependencies)
13. [Appendix: Code Examples](#appendix-code-examples)

---

## Executive Summary

`ClientSandboxEngine.ts` has grown to ~4,351 lines mixing interactive sandbox play, AI integration, decision handling, history management, and replay/parity testing logic. Iterative "phase/player coercion" fixes attempting to align TypeScript replay with Python recordings are **not converging** and risk masking real engine divergences.

**Core Insight:** Parity testing and interactive play have fundamentally different requirements. Trying to make one engine serve both well is the root cause of complexity.

**Recommendation:** Implement a **heavy upfront refactor** (Option E + Pattern 6 Hybrid) that:
1. Creates a purpose-built `CanonicalReplayEngine` for parity testing
2. Decomposes `ClientSandboxEngine` into focused, testable modules
3. Consolidates all move processing through the canonical `TurnEngineAdapter`
4. Removes all host-side phase/player coercions

This maximizes **long-term value**, **correctness**, **maintainability**, and **debuggability** by enforcing separation of concerns and keeping rules enforcement in the shared orchestrator.

---

## Problem Statement

### Current State

| Metric | Value | Concern |
|--------|-------|---------|
| `ClientSandboxEngine.ts` | 4,351 lines | Critical - monolithic |
| Public/Private methods | 60+ | High - hard to navigate |
| Replay coercion blocks | 15+ | Critical - not converging |
| Test coverage (replay paths) | ~40% | Medium - hard to test |

### Symptoms

1. **Parity failures persist** despite multiple iterations of phase/player coercion fixes
2. **Structural errors**: `[PHASE_MOVE_INVARIANT] Cannot apply move type 'no_placement_action' in phase 'territory_processing'`
3. **Semantic divergences**: TS state lags in decision phases while Python has advanced
4. **Player mismatches**: `Move player 1 does not match current player 2`
5. **Complexity growth**: Each fix adds more conditional branches

### Business Impact

- **Blocked parity validation**: Cannot reliably verify TS engine matches Python
- **Hidden bugs**: Coercions may mask real divergences that affect production
- **Maintenance burden**: Changes to engine require understanding 4K+ lines
- **Debugging difficulty**: State flows through many code paths

---

## Root Cause Analysis

### Why Coercions Don't Converge

```
Python Recording Path:
  ┌─────────────────────────────────────────────────────┐
  │ Records EVERY phase transition as explicit Move     │
  │ no_placement_action, no_territory_action, etc.     │
  │ Decision phases have explicit resolution moves      │
  └─────────────────────────────────────────────────────┘
                          │
                          ▼
              Recorded Move Stream
                          │
                          ▼
TypeScript Replay (Current):
  ┌─────────────────────────────────────────────────────┐
  │ ClientSandboxEngine designed for INTERACTIVE play   │
  │ Auto-resolves some phases, skips others            │
  │ Phase state can lag behind recorded stream         │
  └─────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │ COERCION HACKS try to force alignment              │
  │ - Coerce phase to match move type                  │
  │ - Coerce player to match move.player               │
  │ - Skip validation, append history-only             │
  │                                                     │
  │ PROBLEM: Fighting against engine's state machine   │
  └─────────────────────────────────────────────────────┘
```

### Decision Lifecycle Mismatch

| Aspect | Python (Recording) | TS Orchestrator | Mismatch |
|--------|-------------------|-----------------|----------|
| Decision phases | Explicit moves recorded | Can auto-resolve | YES |
| No-action moves | Always recorded | Sometimes synthesized | YES |
| Phase transitions | After each move | May batch | YES |
| Player advancement | Explicit | Can be implicit | YES |

### Canonical Rules Violations

The coercion approach violates RR-CANON principles:

- **RR-CANON-R075**: Every phase transition requires explicit Move
- **RR-CANON-R076**: No synthetic moves without recording
- **Separation of Concerns**: Host should not rewrite phase/player state

---

## Architecture Options Analysis

### Option A: Create CanonicalReplayEngine (New Purpose-Built Class)

Create a minimal (~200-400 line) replay engine wrapping `TurnEngineAdapter` directly.

**Implementation:**
```typescript
class CanonicalReplayEngine {
  private state: GameState;
  private adapter: TurnEngineAdapter;

  async applyMove(move: Move): Promise<ReplayResult> {
    // No coercions - direct delegation to canonical orchestrator
    return this.adapter.processMove(move);
  }
}
```

| Aspect | Assessment |
|--------|------------|
| **Pros** | Clean separation; doesn't touch production code; TurnEngineAdapter already canonical; fast to implement |
| **Cons** | Another class to maintain; doesn't fix underlying ClientSandboxEngine complexity |
| **Risk** | Low - new isolated code |
| **Effort** | Low-Medium (2-4 hours) |
| **Long-term Value** | Medium - solves parity, doesn't improve sandbox |
| **Correctness** | High - uses canonical path |
| **Maintainability** | High for replay; neutral for sandbox |
| **Debuggability** | High - simple, linear flow |

---

### Option B: Full ClientSandboxEngine Decomposition

Break ClientSandboxEngine into composable modules with clear responsibilities.

**Target Architecture:**
```
src/client/sandbox/
├── ClientSandboxEngine.ts (facade ~800-1000 lines)
├── modules/
│   ├── SandboxStateManager.ts (~400 lines)
│   ├── SandboxMoveProcessor.ts (~600 lines)
│   ├── SandboxReplayEngine.ts (~600 lines)
│   ├── SandboxDecisionHandler.ts (~500 lines)
│   ├── SandboxTurnManager.ts (~400 lines)
│   └── SandboxBoardOps.ts (~300 lines)
```

| Aspect | Assessment |
|--------|------------|
| **Pros** | Best long-term maintainability; testable components; clear separation of concerns |
| **Cons** | Large effort; regression risk; many call sites to update |
| **Risk** | Medium-High - touching critical production code |
| **Effort** | High (1-2 weeks) |
| **Long-term Value** | Very High - sustainable architecture |
| **Correctness** | High - when complete |
| **Maintainability** | Very High - isolated concerns |
| **Debuggability** | Very High - clear boundaries |

---

### Option C: Continue In-Place Coercion Fixes

Keep adding targeted phase/player alignment fixes to `processMoveViaAdapter`.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Incremental; familiar code; no new architecture |
| **Cons** | Not converging; complexity growing; violates canonical rules |
| **Risk** | Medium - unclear endpoint; may mask real bugs |
| **Effort** | Unknown (already spent significant time without convergence) |
| **Long-term Value** | Low - technical debt accumulation |
| **Correctness** | Low - coercions can mask bugs |
| **Maintainability** | Low - increasingly fragile |
| **Debuggability** | Low - complex conditional flows |

**RECOMMENDATION: REJECT** - Evidence shows this approach does not converge.

---

### Option D: Python-Only Parity (Accept Divergence)

Use Python as authoritative replay engine; test TS only for interactive scenarios.

| Aspect | Assessment |
|--------|------------|
| **Pros** | Simplest; no TS changes |
| **Cons** | Doesn't validate TS engine; undetected production bugs; contradicts "TS is SSoT" principle |
| **Risk** | High - undetected divergence in production |
| **Effort** | None |
| **Long-term Value** | Negative - avoids problem |
| **Correctness** | Unknown - TS unvalidated |
| **Maintainability** | N/A |
| **Debuggability** | N/A |

**RECOMMENDATION: REJECT** - Violates SSoT principle and risks production bugs.

---

### Option E: Hybrid Approach (RECOMMENDED)

Combine Option A + Option B in phased execution:

1. **Phase 1 (Immediate)**: Create CanonicalReplayEngine for parity testing
2. **Phase 2 (Short-term)**: Roll back coercions from ClientSandboxEngine
3. **Phase 3 (Medium-term)**: Decompose ClientSandboxEngine into modules
4. **Phase 4 (Ongoing)**: Continuous improvement

| Aspect | Assessment |
|--------|------------|
| **Pros** | Immediate parity fix + long-term cleanup; low initial risk; validates architecture |
| **Cons** | Two paths temporarily; requires discipline to complete |
| **Risk** | Low initially; medium for full migration |
| **Effort** | Medium-High (but spread over time) |
| **Long-term Value** | Very High - best of both approaches |
| **Correctness** | High - canonical replay validates engine |
| **Maintainability** | Very High - when complete |
| **Debuggability** | Very High - clear separation |

---

## Recommended Solution

### Primary Strategy: Option E (Hybrid) + Pattern 6 (Adapter Consolidation)

```
                      ┌─────────────────────────────────────┐
                      │     ClientSandboxEngine             │
                      │     (Facade: ~1000 lines)           │
                      │     - Coordinates modules           │
                      │     - Public API surface            │
                      │     - Mode selection                │
                      └──────────────┬────────────────────-─┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │ SandboxState    │     │ SandboxDecision │     │ SandboxMove     │
  │ Manager         │     │ Handler         │     │ Strategies      │
  │ (~400 lines)    │     │ (~500 lines)    │     │ (~400 lines)    │
  │                 │     │                 │     │                 │
  │ - State access  │     │ - Choice mapping│     │ - Interactive   │
  │ - History       │     │ - UI integration│     │ - AI Simulation │
  │ - Snapshots     │     │ - Decision wire │     │ - Replay (UI)   │
  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────────────────┐
                      │     SandboxOrchestratorAdapter      │
                      │     (Existing: ~800 lines)          │
                      │     - Adapts sandbox → orchestrator │
                      └──────────────┬──────────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────────────┐
                      │     TurnEngineAdapter               │
                      │     (Canonical backend adapter)     │
                      └──────────────┬──────────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────────────┐
                      │     Shared Orchestrator             │
                      │     (turnOrchestrator.ts)           │
                      │     - CANONICAL RULES ENFORCEMENT   │
                      └─────────────────────────────────────┘


  SEPARATE PATH FOR PARITY:

  ┌─────────────────────────────────────┐
  │     CanonicalReplayEngine           │
  │     (NEW: ~300-400 lines)           │
  │     - Parity testing only           │
  │     - No UI/AI/coercions            │
  │     - Direct TurnEngineAdapter      │
  │     - Fail fast on errors           │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │     TurnEngineAdapter               │
  │     (replayMode: true)              │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │     Shared Orchestrator             │
  └─────────────────────────────────────┘
```

### Key Principles

1. **Single Source of Truth**: All rules enforcement in shared orchestrator
2. **Explicit Moves Only**: No synthetic or coerced moves in hosts
3. **Separation of Concerns**: Parity testing separate from interactive play
4. **Fail Fast**: Replay engine fails on invariant violations (no masking)
5. **Testability**: Each module independently testable
6. **Canonical Compliance**: Aligns with RR-CANON rules specification

---

## Decomposition Patterns Analysis

### Pattern 1: Vertical Slice (Feature Modules)

Extract cohesive feature slices that own their state and logic.

```typescript
// Each module owns its domain completely
class SandboxCaptureModule {
  enumerateCaptures(state: GameState): CaptureMove[];
  applyCapture(state: GameState, move: CaptureMove): GameState;
  hasAnyCapturesForPlayer(state: GameState, player: number): boolean;
}
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| Long-term Value | 8/10 | Good isolation, natural boundaries |
| Correctness | 7/10 | Clear ownership prevents bugs |
| Maintainability | 8/10 | Easy to understand individual modules |
| Debuggability | 8/10 | Can trace issues to specific module |
| Effort | Medium | 3-5 days |
| Risk | Medium | Well-understood pattern |

---

### Pattern 2: Strategy Pattern (Pluggable Behaviors)

Extract variable behaviors into strategy interfaces.

```typescript
interface MoveProcessingStrategy {
  processMove(context: SandboxContext, move: Move): Promise<MoveResult>;
  getValidMoves(context: SandboxContext): Move[];
}

class InteractiveMoveStrategy implements MoveProcessingStrategy {
  // UI-driven, graceful error handling
}

class ReplayMoveStrategy implements MoveProcessingStrategy {
  // Fail-fast, no coercions, canonical
}

class AISimulationStrategy implements MoveProcessingStrategy {
  // Fast path for AI search
}
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| Long-term Value | 9/10 | Easy to add new modes |
| Correctness | 8/10 | Clear contracts per mode |
| Maintainability | 8/10 | Open/Closed principle |
| Debuggability | 7/10 | Need to identify which strategy active |
| Effort | Medium | 2-4 days |
| Risk | Low-Medium | Well-known GoF pattern |

---

### Pattern 3: State Machine Refactor

Formalize phase/turn flow as explicit state machine with handlers per phase.

```typescript
interface PhaseHandler {
  phase: GamePhase;
  enter(context: SandboxContext): void;
  getValidMoves(context: SandboxContext): Move[];
  processMove(context: SandboxContext, move: Move): PhaseTransition;
  exit(context: SandboxContext): void;
}

class RingPlacementHandler implements PhaseHandler { ... }
class MovementHandler implements PhaseHandler { ... }
class CaptureHandler implements PhaseHandler { ... }
class LineProcessingHandler implements PhaseHandler { ... }
class TerritoryProcessingHandler implements PhaseHandler { ... }
class ForcedEliminationHandler implements PhaseHandler { ... }
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| Long-term Value | 10/10 | Matches game domain perfectly |
| Correctness | 9/10 | Explicit transitions, testable per phase |
| Maintainability | 9/10 | Clear structure matches rules |
| Debuggability | 9/10 | Can debug individual phase handlers |
| Effort | High | 1-2 weeks |
| Risk | Medium-High | Significant architectural change |

---

### Pattern 4: Composition over Inheritance

Compose ClientSandboxEngine from smaller trait-like objects via dependency injection.

```typescript
class ClientSandboxEngine {
  constructor(
    private readonly stateManager: ISandboxStateManager,
    private readonly moveProcessor: ISandboxMoveProcessor,
    private readonly decisionHandler: ISandboxDecisionHandler,
    private readonly turnManager: ISandboxTurnManager,
  ) {}
}

// Easy to swap implementations for testing
const testEngine = new ClientSandboxEngine(
  new MockStateManager(),
  new MockMoveProcessor(),
  // ...
);
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| Long-term Value | 8/10 | SOLID principles |
| Correctness | 7/10 | Good with proper interfaces |
| Maintainability | 8/10 | Components replaceable |
| Debuggability | 8/10 | Clear dependencies |
| Effort | Low-Medium | 2-3 days per component |
| Risk | Low | Incremental migration |

---

### Pattern 5: Event-Driven Architecture

Decouple components via event bus.

```typescript
class SandboxEventBus {
  emit(event: SandboxEvent): void;
  on<T extends SandboxEvent>(type: T['type'], handler: (e: T) => void): void;
}

// Components react to events
stateManager.on('move:applied', (e) => this.snapshotHistory(e.state));
victoryChecker.on('state:changed', (e) => this.checkVictory(e.state));
telemetry.on('move:applied', (e) => this.recordMetrics(e));
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| Long-term Value | 6/10 | Good for UI, risky for rules |
| Correctness | 5/10 | Event ordering can cause bugs |
| Maintainability | 6/10 | Can become "event spaghetti" |
| Debuggability | 4/10 | Hard to trace async event flows |
| Effort | High | 1+ week |
| Risk | Medium-High | Easy to misuse |

**RECOMMENDATION**: Use sparingly for UI notifications only, not core logic.

---

### Pattern 6: Adapter Consolidation (RECOMMENDED)

Consolidate all move processing through `TurnEngineAdapter` as single source of truth.

```typescript
// ALL move processing goes through this path:
ClientSandboxEngine
  → SandboxOrchestratorAdapter
    → TurnEngineAdapter
      → processTurnAsync (shared orchestrator)

// No bypass paths, no special cases
```

| Criterion | Score | Notes |
|-----------|-------|-------|
| Long-term Value | 10/10 | Single source of truth |
| Correctness | 10/10 | Canonical rules enforcement |
| Maintainability | 9/10 | Clear layering |
| Debuggability | 9/10 | One path to trace |
| Effort | Medium | 3-5 days |
| Risk | Low-Medium | Builds on existing pattern |

---

### Pattern Selection Summary

| Pattern | Long-term | Correct | Maintain | Debug | Effort | Risk | TOTAL |
|---------|-----------|---------|----------|-------|--------|------|-------|
| 1. Vertical Slice | 8 | 7 | 8 | 8 | Med | Med | 31 |
| 2. Strategy | 9 | 8 | 8 | 7 | Med | Low | 32 |
| 3. State Machine | 10 | 9 | 9 | 9 | High | Med | 37 |
| 4. Composition | 8 | 7 | 8 | 8 | Low | Low | 31 |
| 5. Event-Driven | 6 | 5 | 6 | 4 | High | High | 21 |
| **6. Adapter Consol.** | **10** | **10** | **9** | **9** | **Med** | **Low** | **38** |

**SELECTED: Pattern 6 (Adapter Consolidation) + Pattern 2 (Strategy) + Pattern 4 (Composition)**

---

## Detailed Implementation Plan

### Phase 1: Canonical Replay Engine (Day 1-2)

**Goal:** Unblock parity testing with canonical path

**Deliverables:**
- `src/shared/replay/CanonicalReplayEngine.ts` (~300-400 lines)
- Updated `scripts/selfplay-db-ts-replay.ts`
- Passing parity gate on test DBs

**Implementation Steps:**

1. **Create CanonicalReplayEngine class**
   ```typescript
   // src/shared/replay/CanonicalReplayEngine.ts
   export interface ReplayEngineConfig {
     initialState: GameState;
     boardType: BoardType;
     numPlayers: number;
   }

   export interface ReplayStepResult {
     success: boolean;
     state: GameState;
     error?: string;
     stateHash: string;
   }

   export class CanonicalReplayEngine {
     private state: GameState;
     private adapter: TurnEngineAdapter;

     constructor(config: ReplayEngineConfig);
     getState(): GameState;
     async applyMove(move: Move): Promise<ReplayStepResult>;
   }
   ```

2. **Implement TurnEngineAdapter replay mode**
   - Add `replayMode` option to `TurnEngineAdapterDeps`
   - In replay mode: stop on `awaiting_decision`, don't auto-resolve
   - Return control to caller for next decision move

3. **Update selfplay-db-ts-replay.ts**
   - Replace `ClientSandboxEngine` instantiation with `CanonicalReplayEngine`
   - Remove all phase/player coercion logic
   - Use direct recorded moves as decision resolutions

4. **Validation**
   - Run parity gate on `/tmp/db_debug.db`
   - Verify zero structural/semantic divergences
   - Confirm fail-fast on invariant violations

**Acceptance Criteria:**
- [ ] CanonicalReplayEngine instantiates from initial state
- [ ] Applies recorded moves via TurnEngineAdapter
- [ ] No phase/player coercions in engine code
- [ ] Parity gate passes with 5+ games
- [ ] Clear error messages on failures

---

### Phase 2: Sandbox Cleanup Foundations (Day 3-5)

**Goal:** Remove coercions, establish module boundaries

**Deliverables:**
- Cleaned `ClientSandboxEngine.processMoveViaAdapter`
- `SandboxStateManager.ts` module
- `SandboxDecisionHandler.ts` module

**Implementation Steps:**

1. **Remove replay coercions from ClientSandboxEngine**
   - Delete all `coercedPhase` logic blocks
   - Delete all `coercedPlayer` logic blocks
   - Delete "history-only" append paths
   - Delete `autoResolvePendingDecisionPhasesForReplay`
   - Keep simple post-move advancement for UI

2. **Extract SandboxStateManager**
   ```typescript
   // src/client/sandbox/modules/SandboxStateManager.ts
   export interface ISandboxStateManager {
     getState(): GameState;
     setState(state: GameState): void;
     cloneState(): GameState;
     getStateAtIndex(index: number): GameState | null;
     appendHistory(before: GameState, move: Move): void;
     getSerializedState(): SerializedGameState;
   }
   ```

3. **Extract SandboxDecisionHandler**
   ```typescript
   // src/client/sandbox/modules/SandboxDecisionHandler.ts
   export interface ISandboxDecisionHandler {
     mapDecisionToChoice(decision: PendingDecision): PlayerChoice;
     mapChoiceResponseToMove(response: PlayerChoiceResponse): Move;
     getValidTerritoryMoves(): Move[];
     getValidLineMoves(): Move[];
     getValidEliminationMoves(): Move[];
   }
   ```

4. **Update ClientSandboxEngine to delegate**
   - Instantiate modules in constructor
   - Replace direct state access with stateManager calls
   - Replace decision mapping with decisionHandler calls

**Acceptance Criteria:**
- [ ] No phase/player coercions in ClientSandboxEngine
- [ ] SandboxStateManager handles all state operations
- [ ] SandboxDecisionHandler handles all decision mapping
- [ ] Interactive sandbox tests still pass
- [ ] Parity uses CanonicalReplayEngine (not affected)

---

### Phase 3: Strategy Pattern for Move Processing (Day 6-8)

**Goal:** Clean separation of interactive vs AI vs replay paths

**Deliverables:**
- `MoveProcessingStrategy` interface
- `InteractiveMoveStrategy` implementation
- `AISimulationStrategy` implementation
- `UIReplayStrategy` implementation (for UI playback, distinct from parity)

**Implementation Steps:**

1. **Define strategy interface**
   ```typescript
   // src/client/sandbox/modules/strategies/types.ts
   export interface MoveProcessingStrategy {
     processMove(move: Move): Promise<MoveResult>;
     getValidMoves(): Move[];
     canUndo(): boolean;
   }

   export interface MoveResult {
     success: boolean;
     newState: GameState;
     victoryResult?: GameResult;
     pendingDecision?: PendingDecision;
     error?: string;
   }
   ```

2. **Implement InteractiveMoveStrategy**
   - Graceful error handling
   - UI state updates
   - Undo support
   - Decision prompts via handler

3. **Implement AISimulationStrategy**
   - Fast path without UI updates
   - Auto-resolve decisions with first option
   - No history append during search

4. **Implement UIReplayStrategy**
   - For SelfPlayBrowser UI playback
   - Uses recorded decisions
   - Supports pause/step/resume

5. **Update ClientSandboxEngine**
   ```typescript
   class ClientSandboxEngine {
     private strategy: MoveProcessingStrategy;

     setMode(mode: 'interactive' | 'ai' | 'replay'): void {
       this.strategy = this.createStrategy(mode);
     }

     async processMove(move: Move): Promise<MoveResult> {
       return this.strategy.processMove(move);
     }
   }
   ```

**Acceptance Criteria:**
- [ ] Three strategy implementations complete
- [ ] ClientSandboxEngine delegates to active strategy
- [ ] Mode switching works correctly
- [ ] All sandbox tests pass
- [ ] AI simulation path is measurably faster

---

### Phase 4: Turn/Phase Management & Tests (Day 9-11)

**Goal:** Extract remaining concerns, comprehensive test coverage

**Deliverables:**
- `SandboxTurnManager.ts` module
- Unit tests for all new modules
- Integration tests for strategy switching

**Implementation Steps:**

1. **Extract SandboxTurnManager**
   ```typescript
   // src/client/sandbox/modules/SandboxTurnManager.ts
   export interface ISandboxTurnManager {
     startTurnForPlayer(player: number): void;
     advanceTurnAndPhase(): void;
     getNextPlayer(): number;
     isInteractivePhase(): boolean;
     updateLpsTracking(): void;
   }
   ```

2. **Create unit tests**
   ```
   tests/unit/sandbox/
   ├── SandboxStateManager.test.ts
   ├── SandboxDecisionHandler.test.ts
   ├── SandboxTurnManager.test.ts
   └── strategies/
       ├── InteractiveMoveStrategy.test.ts
       ├── AISimulationStrategy.test.ts
       └── UIReplayStrategy.test.ts
   ```

3. **Create integration tests**
   ```typescript
   describe('ClientSandboxEngine Integration', () => {
     it('processes full game in interactive mode', async () => { ... });
     it('processes full game in AI simulation mode', async () => { ... });
     it('replays recorded game correctly', async () => { ... });
     it('switches modes mid-game', async () => { ... });
   });
   ```

4. **Verify canonical compliance**
   - No synthetic moves generated
   - No silent phase skips
   - All moves go through orchestrator

**Acceptance Criteria:**
- [ ] SandboxTurnManager extracted
- [ ] >80% coverage on new modules
- [ ] Integration tests for mode switching
- [ ] No regression in existing tests
- [ ] ClientSandboxEngine now ~1000-1200 lines

---

### Phase 5: Backend Alignment (Day 12-14)

**Goal:** Apply similar patterns to server-side GameEngine

**Deliverables:**
- `GameEngine.ts` delegates fully to `TurnEngineAdapter`
- Remove duplicate rule logic from GameEngine
- Shared test helpers for both engines

**Implementation Steps:**

1. **Audit GameEngine.ts** (2,721 lines)
   - Identify rule logic that should be in orchestrator
   - Identify UI/session logic that should stay
   - Map methods to shared engine equivalents

2. **Migrate GameEngine to adapter pattern**
   - All move processing via TurnEngineAdapter
   - Remove makeMove bypass paths
   - Delegate victory checking to shared aggregate

3. **Create shared test utilities**
   ```typescript
   // src/shared/testing/engineTestHelpers.ts
   export function createTestEngine(config: TestConfig): IGameEngine;
   export function applyMoveSequence(engine: IGameEngine, moves: Move[]): void;
   export function assertStateEquals(a: GameState, b: GameState): void;
   ```

**Acceptance Criteria:**
- [ ] GameEngine uses TurnEngineAdapter for all moves
- [ ] No duplicate rule logic
- [ ] Backend tests still pass
- [ ] Parity tests validate backend path

---

### Phase 6: Continuous Improvement (Ongoing)

**Goal:** Maintain architecture, address tech debt

**Activities:**
- Monitor ClientSandboxEngine size (target: <1200 lines)
- Add tests for edge cases discovered in production
- Refactor GameSession.ts as needed
- Document architectural decisions

---

## Interface Contracts

### CanonicalReplayEngine

```typescript
/**
 * Purpose-built engine for parity testing.
 * Uses canonical TurnEngineAdapter path with no coercions.
 * Fails fast on any invariant violation.
 */
export interface ICanonicalReplayEngine {
  /** Current game state (read-only view) */
  getState(): Readonly<GameState>;

  /** Apply a recorded move. Returns result with state hash for parity. */
  applyMove(move: Move): Promise<ReplayStepResult>;

  /** Get canonical state hash for parity comparison */
  getStateHash(): string;

  /** Check if game has ended */
  isGameOver(): boolean;

  /** Get victory result if game ended */
  getVictoryResult(): GameResult | null;
}

export interface ReplayStepResult {
  success: boolean;
  state: GameState;
  stateHash: string;
  error?: {
    type: 'invariant' | 'validation' | 'internal';
    message: string;
    phase: GamePhase;
    player: number;
    moveType: string;
  };
}
```

### SandboxStateManager

```typescript
/**
 * Manages sandbox game state, history, and snapshots.
 * Single source of truth for state access.
 */
export interface ISandboxStateManager {
  /** Get current state (mutable for sandbox use) */
  getState(): GameState;

  /** Update state */
  setState(state: GameState): void;

  /** Deep clone current state */
  cloneState(): GameState;

  /** Get serialized state for persistence */
  getSerializedState(): SerializedGameState;

  /** Initialize from serialized state */
  initFromSerialized(serialized: SerializedGameState): void;

  /** History operations */
  appendHistoryEntry(before: GameState, move: Move): void;
  getHistoryLength(): number;
  getStateAtIndex(index: number): GameState | null;

  /** Snapshot operations for undo */
  createSnapshot(): StateSnapshot;
  restoreSnapshot(snapshot: StateSnapshot): void;
}
```

### SandboxDecisionHandler

```typescript
/**
 * Handles player decision mapping between UI choices
 * and engine Move objects.
 */
export interface ISandboxDecisionHandler {
  /** Map orchestrator decision to UI choice format */
  mapDecisionToChoice(decision: PendingDecision): PlayerChoice;

  /** Map UI choice response back to Move */
  mapChoiceResponseToMove(
    response: PlayerChoiceResponse,
    decision: PendingDecision
  ): Move;

  /** Get valid moves for decision phases */
  getValidTerritoryMoves(state: GameState): Move[];
  getValidLineMoves(state: GameState): Move[];
  getValidEliminationMoves(state: GameState): Move[];

  /** Request decision from UI */
  requestChoice(choice: PlayerChoice): Promise<PlayerChoiceResponse>;
}
```

### MoveProcessingStrategy

```typescript
/**
 * Strategy for processing moves in different modes.
 */
export interface IMoveProcessingStrategy {
  /** Mode identifier */
  readonly mode: 'interactive' | 'ai' | 'replay';

  /** Process a move in this mode */
  processMove(move: Move): Promise<MoveResult>;

  /** Get valid moves for current player */
  getValidMoves(): Move[];

  /** Whether undo is supported in this mode */
  canUndo(): boolean;

  /** Error handling behavior */
  onError(error: Error): MoveResult;
}

export interface MoveResult {
  success: boolean;
  newState: GameState;
  victoryResult?: GameResult;
  pendingDecision?: PendingDecision;
  error?: string;
  /** Metrics for debugging */
  metrics?: {
    processingTimeMs: number;
    orchestratorCalls: number;
  };
}
```

---

## Testing Strategy

### Unit Test Coverage Targets

| Module | Target | Priority |
|--------|--------|----------|
| CanonicalReplayEngine | 95% | P0 |
| SandboxStateManager | 90% | P1 |
| SandboxDecisionHandler | 85% | P1 |
| MoveProcessingStrategies | 90% | P1 |
| SandboxTurnManager | 85% | P2 |
| ClientSandboxEngine (facade) | 80% | P2 |

### Test Categories

1. **Parity Tests** (CanonicalReplayEngine)
   - Replay recorded games from Python DBs
   - Compare state hashes at each step
   - Verify victory detection matches

2. **Strategy Tests**
   - Interactive mode with simulated UI
   - AI mode with timing validation
   - Replay mode with recorded sequences

3. **Integration Tests**
   - Full game flows in each mode
   - Mode switching scenarios
   - Error recovery paths

4. **Regression Tests**
   - Existing sandbox test suite
   - UI component integration
   - Self-play scenarios

### Parity Gate Criteria

```bash
# Must pass before merging
PYTHONPATH=. python scripts/run_canonical_selfplay_parity_gate.py \
  --board-type square8 \
  --num-games 10 \
  --db /tmp/parity_gate.db

# Expected output:
# - 0 structural errors
# - 0 semantic divergences
# - 0 end-of-game divergences
```

---

## Risk Assessment & Mitigations

### Risk 1: Divergence Between Replay and Interactive Engines

**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Both use `TurnEngineAdapter` as single path
- Replay engine is minimal, delegates everything
- Shared orchestrator is canonical SSoT
- Run parity tests before every release

### Risk 2: Regression in Interactive Sandbox UX

**Likelihood:** Medium
**Impact:** Medium
**Mitigation:**
- Parity uses new engine; sandbox cleanup is separate
- Module extraction is incremental with tests
- Feature flags for gradual rollout
- Comprehensive UI test suite

### Risk 3: Time Overrun

**Likelihood:** Low-Medium
**Impact:** Medium
**Mitigation:**
- Phase 1 delivers value quickly (2 days)
- Subsequent phases are independent
- Can pause after any phase if needed
- Clear acceptance criteria per phase

### Risk 4: Performance Regression in AI Mode

**Likelihood:** Low
**Impact:** Medium
**Mitigation:**
- AISimulationStrategy skips unnecessary work
- Benchmark before/after
- Profiling during development

### Risk 5: Incomplete Module Extraction

**Likelihood:** Medium
**Impact:** Low
**Mitigation:**
- Track ClientSandboxEngine line count
- Code review for new additions
- Periodic architectural reviews

---

## Success Criteria

### Phase 1 (Parity)
- [ ] CanonicalReplayEngine processes recorded games
- [ ] No phase/player coercions in replay path
- [ ] Parity gate passes with 10+ square8 games
- [ ] Clear error messages with state context

### Phase 2 (Foundations)
- [ ] SandboxStateManager module extracted
- [ ] SandboxDecisionHandler module extracted
- [ ] ClientSandboxEngine delegates to modules
- [ ] All existing tests pass

### Phase 3 (Strategies)
- [ ] Three strategy implementations
- [ ] Mode switching works correctly
- [ ] AI simulation is measurably faster
- [ ] 85%+ coverage on strategy code

### Phase 4 (Tests)
- [ ] SandboxTurnManager extracted
- [ ] Unit tests for all modules
- [ ] Integration tests for modes
- [ ] ClientSandboxEngine < 1200 lines

### Overall
- [ ] Parity testing works reliably
- [ ] Sandbox UX unchanged for users
- [ ] Codebase more maintainable
- [ ] Debugging easier with clear layers
- [ ] Team can extend without full context

---

## Related Files & Dependencies

### Primary Files (Refactor Targets)

| File | Lines | Action |
|------|-------|--------|
| `src/client/sandbox/ClientSandboxEngine.ts` | 4,351 | Decompose |
| `src/client/sandbox/SandboxOrchestratorAdapter.ts` | 808 | Keep, enhance |
| `src/server/game/turn/TurnEngineAdapter.ts` | 400 | Add replay mode |
| `scripts/selfplay-db-ts-replay.ts` | 717 | Update to use replay engine |

### Secondary Files (May Need Updates)

| File | Lines | Potential Changes |
|------|-------|-------------------|
| `src/server/game/GameEngine.ts` | 2,721 | Future: adapter consolidation |
| `src/shared/engine/orchestration/turnOrchestrator.ts` | 2,145 | None - canonical SSoT |
| `src/client/sandbox/sandboxAI.ts` | 1,651 | Integrate with AISimulationStrategy |
| `src/server/game/GameSession.ts` | 2,039 | Future: separate concerns |

### New Files to Create

```
src/shared/replay/
├── CanonicalReplayEngine.ts
├── types.ts
└── index.ts

src/client/sandbox/modules/
├── SandboxStateManager.ts
├── SandboxDecisionHandler.ts
├── SandboxTurnManager.ts
└── strategies/
    ├── types.ts
    ├── InteractiveMoveStrategy.ts
    ├── AISimulationStrategy.ts
    └── UIReplayStrategy.ts
```

---

## Appendix: Code Examples

### Example: CanonicalReplayEngine Implementation

```typescript
// src/shared/replay/CanonicalReplayEngine.ts
import { TurnEngineAdapter, StateAccessor, DecisionHandler } from '../../server/game/turn/TurnEngineAdapter';
import { GameState, Move, GameResult, BoardType } from '../types/game';
import { hashGameStateSHA256 } from '../engine';
import { createInitialGameState } from '../engine/initialState';

export interface ReplayEngineConfig {
  initialState?: GameState;
  boardType: BoardType;
  numPlayers: number;
  gameId: string;
}

export interface ReplayStepResult {
  success: boolean;
  state: GameState;
  stateHash: string;
  error?: {
    type: 'invariant' | 'validation' | 'internal';
    message: string;
    phase: string;
    player: number;
    moveType: string;
  };
}

export class CanonicalReplayEngine {
  private state: GameState;
  private adapter: TurnEngineAdapter;
  private moveIndex: number = 0;

  constructor(config: ReplayEngineConfig) {
    this.state = config.initialState ?? createInitialGameState(
      config.gameId,
      config.boardType,
      this.createPlayers(config.numPlayers),
      { type: 'rapid', initialTime: 600, increment: 0 },
      false
    );

    const stateAccessor: StateAccessor = {
      getGameState: () => this.state,
      updateGameState: (s) => { this.state = s; },
      getPlayerInfo: () => ({ type: 'ai' as const }),
    };

    const decisionHandler: DecisionHandler = {
      requestDecision: async (decision) => {
        // In replay mode, this should not be called
        // The next recorded move should resolve the decision
        throw new Error(
          `[CanonicalReplayEngine] Unexpected decision request: ${decision.type}. ` +
          `Replay should provide explicit decision moves.`
        );
      },
    };

    this.adapter = new TurnEngineAdapter({
      stateAccessor,
      decisionHandler,
      replayMode: true, // New option: don't auto-resolve
    });
  }

  getState(): Readonly<GameState> {
    return this.state;
  }

  getStateHash(): string {
    return hashGameStateSHA256(this.state);
  }

  isGameOver(): boolean {
    return this.state.gameStatus !== 'active';
  }

  getVictoryResult(): GameResult | null {
    if (!this.isGameOver()) return null;
    return {
      winner: this.state.winner,
      reason: this.state.winReason ?? 'unknown',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      },
    };
  }

  async applyMove(move: Move): Promise<ReplayStepResult> {
    this.moveIndex++;
    const beforePhase = this.state.currentPhase;
    const beforePlayer = this.state.currentPlayer;

    try {
      const result = await this.adapter.processMove(move);

      if (!result.success) {
        return {
          success: false,
          state: this.state,
          stateHash: this.getStateHash(),
          error: {
            type: 'validation',
            message: result.error ?? 'Move validation failed',
            phase: beforePhase,
            player: beforePlayer,
            moveType: move.type,
          },
        };
      }

      return {
        success: true,
        state: this.state,
        stateHash: this.getStateHash(),
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        success: false,
        state: this.state,
        stateHash: this.getStateHash(),
        error: {
          type: message.includes('INVARIANT') ? 'invariant' : 'internal',
          message,
          phase: beforePhase,
          player: beforePlayer,
          moveType: move.type,
        },
      };
    }
  }

  private createPlayers(count: number) {
    return Array.from({ length: count }, (_, i) => ({
      id: `player-${i + 1}`,
      username: `Player ${i + 1}`,
      type: 'ai' as const,
      playerNumber: i + 1,
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));
  }
}
```

### Example: Strategy Interface Usage

```typescript
// Usage in ClientSandboxEngine
class ClientSandboxEngine {
  private strategy: IMoveProcessingStrategy;
  private stateManager: ISandboxStateManager;

  async processMove(move: Move): Promise<MoveResult> {
    const before = this.stateManager.cloneState();
    const result = await this.strategy.processMove(move);

    if (result.success) {
      this.stateManager.appendHistoryEntry(before, move);
    }

    return result;
  }

  setMode(mode: 'interactive' | 'ai' | 'replay'): void {
    switch (mode) {
      case 'interactive':
        this.strategy = new InteractiveMoveStrategy(
          this.stateManager,
          this.decisionHandler,
          this.adapter
        );
        break;
      case 'ai':
        this.strategy = new AISimulationStrategy(
          this.stateManager,
          this.adapter
        );
        break;
      case 'replay':
        this.strategy = new UIReplayStrategy(
          this.stateManager,
          this.adapter
        );
        break;
    }
  }
}
```

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-07 | Architecture Team | Initial comprehensive plan |

---

*End of Document*
