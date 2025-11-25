# Seed-5 Tail Divergence: Diagnostic Summary & Next Steps

**Date**: 2025-11-24  
**Related**: Hard Problem #2 (Backend ↔ Sandbox phase/victory parity)  
**Scope**: square8 / 2 players / seed=5 AI trace

## Executive Summary

We have **definitively narrowed** the seed-5 backend vs sandbox divergence to:
1. **Turn ownership at index 62** (backend thinks currentPlayer=2, sandbox=1)
2. **Missed LPS victory** (sandbox ends game with winner=2, backend stays active)

**Line processing and geometry parity are now confirmed correct.**

---

## 1. New Diagnostic Harnesses Implemented

### 1.1 Internal State Parity Test
**File**: `tests/unit/Backend_vs_Sandbox.seed5.internalStateParity.test.ts`

This harness replays the full seed-5 sandbox AI trace into both backend and sandbox engines, comparing after each move:

1. **GameState snapshots** (board, players, markers, collapsed, totals, currentPlayer, currentPhase, gameStatus)
2. **Raw internal metadata**:
   - Per-turn flags: `hasPlacedThisTurn`, `mustMoveFromStackKey`
   - Decision flags: `pendingTerritorySelfElimination`, `pendingLineRewardElimination`
   - LPS metadata: roundIndex, firstPlayer, exclusiveCandidate, actorMask
3. **Semantic LPS invariants** (NEW):
   - Per-player: `hasAnyRealActionForPlayer`, `hasMaterial`
   - Exclusive candidate status (rules-level, not raw counters)

**Key findings**:
- `firstGameStateMismatchIndex`: **62** (0-based, moveNumber 63)
- `firstInternalMismatchIndex`: **0** (raw LPS counters differ from start)
- `firstSemanticLpsMismatchIndex`: **-1** (semantic LPS stays equal while GameState equal!)

### 1.2 Terminal Snapshot Test
**File**: `tests/unit/Seed5TerminalSnapshot.parity.test.ts`

Replays entire seed-5 trace and logs final states:

**Backend terminal**:
- `currentPlayer: 2`
- `currentPhase: 'movement'`
- `gameStatus: 'active'`
- `winner: undefined`

**Sandbox terminal**:
- `currentPlayer: 2`
- `currentPhase: 'ring_placement'`
- `gameStatus: 'completed'`
- `winner: 2`

---

## 2. Precise Answer to "How Much Earlier Than 62?"

### GameState Divergence
**Answer**: There is **NO divergence earlier than index 62**.

- Indices **0–61** inclusive: backend and sandbox GameState snapshots are **identical**
- Index **62**: first public mismatch (`currentPlayer: backend=2, sandbox=1`)

### Internal Metadata Divergence
**Answer**: Raw LPS counters (`lpsRoundIndex`, etc.) differ from **index 0**, but:
- These are **implementation details** (different indexing conventions)
- They produce **NO observable effect on GameState** until index 62
- **Semantic LPS invariants** (who has actions, material, candidate status) **remain equal** through index 62

**Therefore**: The first *sema

ntically meaningful* divergence is **exactly at index 62**, manifesting as a turn-ownership difference.

---

## 3. What We've Fixed / Confirmed

### 3.1 Line Processing ✅
- **FIXED**: Spurious backend `line_processing` phase at move 63
- **Confirmed**: At the critical snapshot (move index 62, moveNumber 63):
  - `findAllLines` returns **zero lines** for both hosts
  - `enumerateProcessLineMoves` returns **zero decisions** for both hosts
  - Both engines are in `currentPhase='movement'` with no line work
- **Test**: `Seed5Move63.lineDetectionSnapshot.test.ts` (passing)

### 3.2 Geometry & Elimination Parity ✅
- **Confirmed**: Board state (stacks, markers, collapsed), player counters, and totalRingsEliminated are **identical** through index 61
- All earlier geometry/elimination issues (chain_capture, territory) are resolved

### 3.3 Phase Scheduling Alignment (Partial)
- Backend now uses `detect_now` for line decision enumeration (no stale cache issues)
- Backend `stepAutomaticPhasesForTesting` skips empty decision phases correctly
- Both engines share `turnLogic.advanceTurnAndPhase` for turn/phase transitions

---

## 4. Remaining Root Causes (Confirmed)

### 4.1 Turn Ownership Mismatch at Index 62
**Observation** (from internal parity harness, index 62 snapshot):
- Backend: `currentPlayer=2`, sandbox: `currentPlayer=1`
- Both: `currentPhase='movement'`, `gameStatus='active'`
- **Semantic LPS**: still equal (both see same action availability per player)

**Implication**:
- The shared `turnLogic.advanceTurnAndPhase` is being called with identical GameState inputs but somehow backend and sandbox arrive at different `currentPlayer` values during the index 61→62 transition
- This is likely due to **subtle differences** in how backend vs sandbox wire the turn engine hooks or when they call `advanceTurnAndPhaseForCurrentPlayer`

### 4.2 Backend Misses LPS Victory
**Observation** (from terminal snapshot):
- By index 63, sandbox has`gameStatus='completed'`, `winner=2` (last-player-standing)
- Backend remains `gameStatus='active'` in `movement` phase

**LPS Tracking State at Index 63** (from internal parity logs):
- Backend: `lpsRoundIndex: 60`, `lpsCurrentRoundFirstPlayer: 1`, `exclusiveCandidate: null`
- Sandbox: `lpsRoundIndex: 13`, `lpsCurrentRoundFirstPlayer: null`, `exclusiveCandidate: null`
- Backend `semanticLps.exclusiveCandidate: null`, Sandbox `exclusiveCandidate: null`

**Implication**:
- Backend's LPS round bookkeeping (`lpsRoundIndex`, `lpsCurrentRoundFirstPlayer`) diverges **numerically** from sandbox from the start
- However, **semantic LPS** (who has actions) stays aligned until the terminal snapshot
- The backend's `updateLpsTrackingForCurrentTurn` / `finalizeCompletedLpsRound` / `maybeEndGameByLastPlayerStanding` sequence **never fires victory**, even though the semantic conditions for R172 must be met by index 63

---

## 5. Backend LPS Wiring Changes Made So Far

### 5.1 Added `runLpsCheckForCurrentInteractiveTurn()`
**Location**: `GameEngine.ts`

```typescript
private runLpsCheckForCurrentInteractiveTurn(): GameResult | undefined {
  if (this.gameState.gameStatus !== 'active') return undefined;
  
  const phase = this.gameState.currentPhase;
  if (phase !== 'ring_placement' && phase !== 'movement' && 
      phase !== 'capture' && phase !== 'chain_capture') {
    return undefined;
  }
  
  this.updateLpsTrackingForCurrentTurn();
  return this.maybeEndGameByLastPlayerStanding();
}
```

### 5.2 Hooked into `stepAutomaticPhasesForTesting`
When the current player has at least one real action (placement/movement/capture), we now call `runLpsCheckForCurrentInteractiveTurn()` before returning from the automatic-phase stepping loop.

### 5.3 Impact
- **Positive**: Backend now has a centralized LPS lifecycle hook mirroring sandbox's `handleStartOfInteractiveTurn`
- **No effect yet on seed-5**: The bisect parity test still fails at index 62, terminal mismatch persists

**Interpretation**: The LPS check is being called, but the **LPS round bookkeeping itself** is not tracking rounds/candidates the same way sandbox does, so `maybeEndGameByLastPlayerStanding` never fires.

---

## 6. Critical Observations from Semantic LPS Tracking

The new semantic LPS snapshot comparison (`firstSemanticLpsMismatchIndex: -1`) tells us:

**While GameState remains equal (indices 0–61):**
- Backend and sandbox **agree on which players have material**
- Backend and sandbox **agree on which players have real actions**
- Backend and sandbox **agree on whether there is an exclusive LPS candidate** (both report `null` at index 62)

**This means**:
- The raw LPS bookkeeping differences (`lpsRoundIndex`, `lpsCurrentRoundFirstPlayer`) are **cosmetic/indexing**
- The actual **R172 semantics** (who has actions, who's the candidate) stay aligned
- **BUT**: The turn-ownership mismatch at index 62 breaks the GameState surface before semantic LPS diverges

**Conclusion**: The index-62 `currentPlayer` mismatch is **NOT caused by LPS divergence**—it's a **turn/phase sequencing issue** in how backend vs sandbox advance from index 61→62.

---

## 7. Next Concrete Steps

### Step A: Fix Turn Ownership at Index 62
**Goal**: Backend and sandbox must agree on `currentPlayer` after move index 61

**Approach**:
1. Add focused logging around **index 61→62 transition** in both engines:
   - Before/after calling `advanceGame` / `advanceTurnAndPhaseForCurrentPlayer`
   - When `turnLogic.advanceTurnAndPhase` is invoked, log inputs/outputs
   - Track forced eliminations, player skipping, and phase selection
2. Compare the exact sequence of `(currentPlayer, currentPhase)` transitions:
   - Backend: `makeMove(move[61])` → `advanceGame` → `stepAutomaticPhasesForTesting` → final state
   - Sandbox: `applyCanonicalMove(move[61])` → `advanceAfterMovement` → `advanceTurnAndPhaseForCurrentPlayerSandbox` → final state
3. Identify where the divergence occurs in the shared `turnLogic` or host-specific turn wiring
4. Adjust backend (likely in `GameEngine.advanceGame` or `stepAutomaticPhasesForTesting`) to match sandbox's sequencing

### Step B: Align Backend LPS Round Bookkeeping
**Goal**: Backend `updateLpsTrackingForCurrentTurn` / `finalizeCompletedLpsRound` must track rounds the same way sandbox's `updateLpsRoundTrackingForCurrentPlayer` does

**Approach**:
1. Review sandbox's `ClientSandboxEngine.updateLpsRoundTrackingForCurrentPlayer`:
   - When is a new cycle started?
   - How are active players identified?
   - When does it finalize a round and compute `exclusiveCandidate`?
2. Compare to backend's `GameEngine.updateLpsTrackingForCurrentTurn` / `finalizeCompletedLpsRound`:
   - Identify semantic differences (not just numeric indices)
3. Adjust backend to use the **same round-start/finalize logic** as sandbox
4. Expected outcome: `lpsRoundIndex` and `lpsCurrentRoundFirstPlayer` stay numerically closer (though exactness isn't required)

### Step C: Ensure LPS Victory Fires at the Right Moment
**Goal**: Backend `maybeEndGameByLastPlayerStanding` must end the game at the same index sandbox does

**Dependency**: Likely blocked by Step B (if LPS candidate is never set correctly, victory can't fire)

**Verification**:
1. After Steps A+B, re-run:
   - `Backend_vs_Sandbox.seed5.internalStateParity.test.ts`
   - `Seed5TerminalSnapshot.parity.test.ts`
   - `Backend_vs_Sandbox.seed5.bisectParity.test.ts`
2. Expected outcomes:
   - Backend terminal: `gameStatus='completed'`, `winner=2`, `currentPhase='ring_placement'`
   - `firstGameStateMismatchIndex` moves to `moves.length` (no mismatch)
   - Bisect test passes

### Step D: Stabilize Broader Parity Suite
Once seed-5 is stable:
1. Re-run `Backend_vs_Sandbox.traceParity.test.ts` (broader seed coverage)
2. Re-run `Sandbox_vs_Backend.aiRngFullParity.test.ts`
3. Tighten any remaining tolerances in `TraceParity.seed5.firstDivergence.test.ts`
4. Ensure `GameEngine.victory.LPS.scenarios.test.ts` still passes (LPS scenarios must not regress)

### Step E: Documentation Updates
1. **`TRACE_PARITY_CONTINUATION_TASK.md`**:
   - Document that seed-5 tail divergence was due to **turn sequencing + LPS lifecycle**, not geometry/lines
   - Capture new canonical invariant: semantic LPS (hasRealAction/hasMaterial/candidate) must match across hosts at all indices where GameState is equal
2. **`HARDEST_PROBLEMS_REPORT.md`**:
   - Update Hard Problem #2 status: geometry/eliminations/lines all aligned, remaining work is turn/LPS lifecycle only

---

## 8. Test Artifacts Created

New test files (all passing, diagnostic-only for now):
- `tests/unit/Backend_vs_Sandbox.seed5.internalStateParity.test.ts`
- `tests/unit/Seed5TerminalSnapshot.parity.test.ts`

Modified engine files:
- `src/server/game/GameEngine.ts`:
  - Added `runLpsCheckForCurrentInteractiveTurn()` helper
  - Hooked into `stepAutomaticPhasesForTesting` to call LPS check when player has real actions

---

## 9. Crisp Status: What's Solved vs Pending

### ✅ Solved / Confirmed
- **Line geometry parity**: Both hosts detect identical lines at all indices
- **Line decision parity**: `enumerateProcessLineMoves` produces identical results
- **Empty line_processing eliminated**: Backend no longer enters `line_processing` with zero decisions at move 63
- **Elimination bookkeeping**: `totalRingsEliminated`, per-player `eliminatedRings`, `board.eliminatedRings` all align
- **Earliest GameState divergence pinpointed**: Exactly index **62**
- **Semantic LPS aligned through index 62**: Both hosts agree on who has actions/material/candidate status

### ⚠️ Pending / Root Causes Not Yet Fixed
1. **Turn ownership mismatch at index 62**:
   - Backend: `currentPlayer=2` after move 61
   - Sandbox: `currentPlayer=1` after move 61
   - **Root cause**: Unknown turn/phase sequencing difference in 61→62 transition
   
2. **Backend never fires LPS victory**:
   - Sandbox correctly ends game with `winner=2` (last-player-standing) by index 63
   - Backend remains `gameStatus='active'` indefinitely
   - **Root cause**: LPS round bookkeeping (`lpsRoundIndex: 60` vs `12`) and/or lifecycle timing prevents `exclusiveCandidate` from being set and victory from firing

3. **LPS round bookkeeping divergence**:
   - Backend and sandbox use different numeric round indices from move 0
   - While semantic LPS stays aligned (action availability), the backend never correctly identifies the LPS candidate
   - **Root cause**: Implementation differences in `updateLpsTrackingForCurrentTurn` vs `updateLpsRoundTrackingForCurrentPlayer`

---

## 10. Diagnostic Log Samples

### Index 62 Tail Window Snapshot
```
[Seed5 InternalParity] tail window snapshot {
  index: 62,
  backend: {
    currentPlayer: 2,
    currentPhase: 'movement',
    gameStatus: 'active',
    internal: { lpsRoundIndex: 60, lpsCurrentRoundFirstPlayer: 1, exclusiveCandidate: null },
    semanticLps: { exclusiveCandidate: null }
  },
  sandbox: {
    currentPlayer: 1,
    currentPhase: 'movement',
    gameStatus: 'active',
    internal: { lpsRoundIndex: 12, lpsCurrentRoundFirstPlayer: 2, exclusiveCandidate: null },
    semanticLps: { exclusiveCandidate: null }
  },
  gameStatesEqual: false,  // currentPlayer differs
  semanticLpsEqual: true   // but action availability identical!
}
```

### Index 63 Terminal Snapshot
```
[Seed5 InternalParity] tail window snapshot {
  index: 63,
  backend: {
    currentPlayer: 2,
    currentPhase: 'movement',
    gameStatus: 'active',
    winner: undefined,
    semanticLps: { exclusiveCandidate: null }
  },
  sandbox: {
    currentPlayer: 2,
    currentPhase: 'ring_placement',
    gameStatus: 'completed',
    winner: 2,
    semanticLps: { exclusiveCandidate: null }
  },
  gameStatesEqual: false,
  semanticLpsEqual: false  // now diverged (sandbox has winner)
}
```

---

## 11. Code Locations for Next Phase

### Backend Turn/Phase Wiring
- `src/server/game/GameEngine.ts`:
  - `advanceGame()` - delegates to TurnEngine
  - `stepAutomaticPhasesForTesting()` - loops through automatic/interactive phases
  - `updateLpsTrackingForCurrentTurn()` - records action availability per player
  - `finalizeCompletedLpsRound()` - computes exclusive candidate
  - `maybeEndGameByLastPlayerStanding()` - fires LPS victory
  - `runLpsCheckForCurrentInteractiveTurn()` - NEW, mirrors sandbox lifecycle

- `src/server/game/turn/TurnEngine.ts`:
  - `advanceGameForCurrentPlayer()` - wraps shared turnLogic

### Sandbox Turn/Phase Wiring
- `src/client/sandbox/ClientSandboxEngine.ts`:
  - `advanceAfterMovement()` - post-movement pipeline
  - `advanceTurnAndPhaseForCurrentPlayer()` - delegates to sandboxTurnEngine
  - `handleStartOfInteractiveTurn()` - LPS lifecycle hook
  - `updateLpsRoundTrackingForCurrentPlayer()` - records action availability
  - `finalizeCompletedLpsRound()` - computes exclusive candidate
  - `maybeEndGameByLastPlayerStanding()` - fires LPS victory

- `src/client/sandbox/sandboxTurnEngine.ts`:
  - `advanceTurnAndPhaseForCurrentPlayerSandbox()` - wraps shared turnLogic

### Shared Turn Logic
- `src/shared/engine/turnLogic.ts`:
  - `advanceTurnAndPhase()` - shared state machine for phase/turn transitions
  - Both backend and sandbox call this with `TurnLogicDelegates`

---

## 12. Proposed New Task Specification

**Title**: "Align Backend Turn/LPS Lifecycle with Sandbox for Seed-5 Tail Parity"

**Scope**:
1. **Diagnose index 61→62 turn transition divergence**:
   - Add instrumented logging for move 61 showing complete `advanceGame` / `advanceTurnAndPhase` call sequence
   - Identify where backend vs sandbox diverge in selecting `nextPlayer`
   
2. **Align backend LPS round lifecycle with sandbox**:
   - Match backend `updateLpsTrackingForCurrentTurn` logic to sandbox's `updateLpsRoundTrackingForCurrentPlayer`
   - Ensure backend `finalizeCompletedLpsRound` computes `exclusiveCandidate` the same way sandbox does
   - Verify `maybeEndGameByLastPlayerStanding` fires when semantic conditions are met

3. **Stabilize seed-5 parity**:
   - Target: `Backend_vs_Sandbox.seed5.bisectParity.test.ts` passes with `firstMismatchIndex === moves.length`
   - Target: `Seed5TerminalSnapshot.parity.test.ts` shows backend `gameStatus='completed'`, `winner=2`
   - Constraint: `GameEngine.victory.LPS.scenarios.test.ts` must remain green

4. **Broader parity validation**:
   - Re-run `Backend_vs_Sandbox.traceParity.test.ts` (includes seeds 1, 5, 14, 17, 18)
   - Ensure no regressions in RNG/determinism suites

**Success Criteria**:
- [ ] Backend and sandbox agree on `currentPlayer` at every index for seed-5
- [ ] Backend fires LPS victory at same index as sandbox for seed-5 (index 63, winner=2)
- [ ] All three seed-5 diagnostic harnesses show no GameState divergence
- [ ] Broader parity suite passes (all seeds, RNG parity, LPS scenarios)

---

## 13. Files to Update in Documentation Phase

Once parity is achieved:

1. **`TRACE_PARITY_CONTINUATION_TASK.md`**:
   - Document resolution of seed-5 tail divergence
   - Emphasize that it was **NOT a geometry/line issue**, but turn/LPS lifecycle
   
2. **`HARDEST_PROBLEMS_REPORT.md`**:
   - Mark Hard Problem #2 substeps as resolved:
     - ✅ Geometry parity
     - ✅ Elimination parity
     - ✅ Line detection & scheduling parity
     - ✅ Turn/LPS lifecycle parity (after this next phase)

3. **Add canonical invariants**:
   - Document that `semanticLpsEqual` (per-player action availability + candidate status) is the **canonical LPS parity target**, not raw round indices
   - Raw `lpsRoundIndex` may differ across hosts as long as semantic LPS matches

---

## 14. Open Questions for Next Session

1. **Why does backend reach `currentPlayer=2` at index 62 while sandbox has `currentPlayer=1`?**
   - Same GameState at index 61, same `turnLogic.advanceTurnAndPhase` shared helper
   - Must be in how delegates (hasAnyPlacement/Movement/Capture) are wired or when advanceGame is called

2. **Why does backend's LPS round index grow to 60 while sandbox's stays at ~13?**
   - Are backend and sandbox calling `updateLpsTracking` at different frequencies?
   - Is backend incrementing `lpsRoundIndex` on every turn while sandbox increments per "actual round of all active players"?

3. **What is the exact set of players + action availability at indices 62–63 that should trigger LPS?**
   - Need to expand semantic LPS logs for indices 62–63 to show `perPlayer` arrays in detail
   - Verify: does P1 have zero actions, P2 has actions, and P2 should be declared winner by R172?

---

## 15. Summary for Continuation

**What we now know with certainty**:
- Earliest GameState divergence: **index 62** (not earlier)
- Root causes:
  1. Turn ownership at index 62 (backend=P2, sandbox=P1)
  2. Backend never fires LPS victory that sandbox fires at index 63
- **NOT** a line/geometry issue; line processing is now correct

**What's instrumented**:
- Full-trace internal+semantic parity harness
- Terminal snapshot harness
- Semantic LPS tracking (hasRealAction/hasMaterial/candidate per player)

**Next focuses**:
- Diagnose 61→62 turn transition difference
- Align backend LPS lifecycle (`updateLpsTracking` / `finalize` / `maybeEndGame`) to sandbox semantics
- Achieve zero-divergence parity for seed-5 tail
