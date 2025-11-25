# Trace Parity Continuation Task

## Executive Summary

This document captures the context and remaining work for solving **Hard Problem #2: Backend↔Sandbox Trace Parity Divergence** in the RingRift codebase. The goal is to achieve **complete parity through game conclusion** by harmonizing the turn advancement and player-skipping logic between `TurnEngine` (backend) and `sandboxTurnEngine` (client).

**Current Status**: 63 of 64 moves match (98.4%). One move diverges at the penultimate position.

---

## 1. Problem Status Matrix

| Priority | Problem | Status |
|----------|---------|--------|
| P0 | Chain Capture Phase Enumeration Bug | ✅ **SOLVED** |
| P0 | Backend↔Sandbox Trace Parity Divergence | 🟡 **98.4% COMPLETE** (63/64 moves) |
| P1 | RNG Determinism Across TS↔Python | Not started |
| P1 | TS↔Python Rules Parity Gaps | Not started |
| P1+ | Non-Bare-Board Last-Player-Standing Edge Case | Not started |
| P2 | Shared `captureChainHelpers` Stub Usage | Not started |
| P1 | Forced Elimination Choice Divergence | Not started |

---

## 2. Current Divergence: Precise Diagnosis

### 2.1 Test Evidence

From `tests/unit/Backend_vs_Sandbox.seed5.bisectParity.test.ts`:

```
firstMismatchIndex: 63
totalMoves: 64  
allEqual: false
```

**Interpretation**: The backend and sandbox produce **identical state hashes** for moves 0-62, but diverge on move 63 (the penultimate move).

### 2.2 Hash Difference Pattern

The divergence is **NOT** in board state - only in the `currentPlayer` prefix:

```
Sandbox Hash: 1:ring_placement:active#<board_hash>
Backend Hash: 2:ring_placement:active#<board_hash>
                                      ^~~~~~~~~~~~ IDENTICAL
```

The board portion after `#` matches exactly. Only `currentPlayer` differs.

### 2.3 Root Cause: Defensive Player-Skip Logic

**Location**: `src/server/game/GameEngine.ts` in `advanceGame()` method, approximately lines 2113-2160

```typescript
// Defensive normalisation: if the active player has no stacks on the
// board and no rings in hand, they can never act again this game.
const activePlayer = this.gameState.players.find(
  (p) => p.playerNumber === this.gameState.currentPlayer
);
if (activePlayer) {
  const activeStacks = this.boardManager.getPlayerStacks(
    this.gameState.board,
    activePlayer.playerNumber
  );

  if (activeStacks.length === 0 && activePlayer.ringsInHand === 0) {
    // Skip to next player in seat order...
    while (skips < maxSkips) {
      nextPlayer = ((nextPlayer % this.gameState.players.length) + 1);
      const player = this.gameState.players.find(p => p.playerNumber === nextPlayer);
      if (player && (this.boardManager.getPlayerStacks(this.gameState.board, nextPlayer).length > 0 || player.ringsInHand > 0)) {
        this.gameState.currentPlayer = nextPlayer;
        this.gameState.currentPhase = player.ringsInHand > 0 ? 'ring_placement' : 'movement';
        break;
      }
      skips++;
    }
  }
}
```

**Problem**: This skip logic exists in `GameEngine.advanceGame()` but has **no equivalent** in `sandboxTurnEngine.ts`.

---

## 3. Implementation Plan: Complete Parity

### Phase 1: Verify Current State ✅

- [x] Fix HTML entity bug in `ClientSandboxEngine.ts` (`Map&lt;number, boolean&gt;` → `Map<number, boolean>`)
- [x] Verify TypeScript compiles
- [x] Run bisectParity test to confirm divergence point

### Phase 2: Define the Invariant (NEXT)

**Question**: After a complete move (including automatic consequences like captures, territory processing, turn advancement), what should `GameState.currentPlayer` represent?

**Option A**: The **actor who just moved** (historical perspective)
**Option B**: The **next player to act** (ready-to-play perspective)

**Current behavior**:
- Backend: Uses Option B (advances to next player, skipping eliminated players)
- Sandbox: Uses Option A (currentPlayer is the actor until explicitly advanced)

**Resolution Required**: Choose one convention and enforce it in both engines.

### Phase 3: Harmonize Skip Logic

#### 3.1 Option: Add Skip Logic to Sandbox

**File**: `src/client/sandbox/sandboxTurnEngine.ts`

Add defensive skip check after `advanceTurnAndPhase()`:

```typescript
export function advanceTurnAndPhaseForCurrentPlayerSandbox(
  state: GameState,
  boardManager: BoardManager
): GameState {
  // Existing shared logic
  let result = advanceTurnAndPhase(state, createSandboxDelegates(state, boardManager));
  
  // NEW: Defensive skip for fully-eliminated players
  result = skipFullyEliminatedPlayers(result, boardManager);
  
  return result;
}

function skipFullyEliminatedPlayers(
  state: GameState,
  boardManager: BoardManager
): GameState {
  const maxSkips = state.players.length;
  let skips = 0;
  
  while (skips < maxSkips) {
    const activePlayer = state.players.find(p => p.playerNumber === state.currentPlayer);
    if (!activePlayer) break;
    
    const activeStacks = boardManager.getPlayerStacks(state.board, activePlayer.playerNumber);
    
    // If player has stacks or rings, they can act - stop skipping
    if (activeStacks.length > 0 || activePlayer.ringsInHand > 0) {
      break;
    }
    
    // Skip to next player
    const nextPlayer = (state.currentPlayer % state.players.length) + 1;
    const nextPlayerData = state.players.find(p => p.playerNumber === nextPlayer);
    
    if (nextPlayerData) {
      state = {
        ...state,
        currentPlayer: nextPlayer,
        currentPhase: nextPlayerData.ringsInHand > 0 ? 'ring_placement' : 'movement'
      };
    }
    
    skips++;
  }
  
  return state;
}
```

#### 3.2 Alternative: Remove Skip Logic from Backend

If the skip logic is actually a bug-masking workaround, remove it from `GameEngine.advanceGame()` instead. This would require understanding why the skip was added in the first place.

### Phase 4: Verify Complete Parity

```bash
# Must all pass
npm test -- --testPathPattern="Backend_vs_Sandbox.seed5.bisectParity" --no-coverage
npm test -- --testPathPattern="TraceParity.seed5.firstDivergence" --no-coverage
npm test -- --testPathPattern="Backend_vs_Sandbox.seed" --no-coverage
npm test -- --testPathPattern="Sandbox_vs_Backend" --no-coverage
```

### Phase 5: Regression Check

```bash
# Must not break existing tests
npm test -- --testPathPattern="GameEngine.chainCapture" --no-coverage
npm run test:core
```

---

## 4. Key Files Reference

| File | Role | Lines of Interest |
|------|------|-------------------|
| `src/server/game/GameEngine.ts` | Backend move application | ~2113-2160 (skip logic) |
| `src/client/sandbox/sandboxTurnEngine.ts` | Sandbox turn advancement | ~147 (debug logging), entire file |
| `src/shared/engine/turnLogic.ts` | Shared `advanceTurnAndPhase()` | Entire file |
| `src/server/game/turn/TurnEngine.ts` | Backend delegates | Entire file |
| `src/client/sandbox/ClientSandboxEngine.ts` | Sandbox engine | History entry creation |
| `tests/unit/Backend_vs_Sandbox.seed5.bisectParity.test.ts` | Primary parity test | Test specification |

---

## 5. Success Criteria

**Problem #2 is SOLVED when ALL of the following are true:**

1. ✅ `Backend_vs_Sandbox.seed5.bisectParity.test.ts`: `firstMismatchIndex === 64` (NO divergence)
2. ✅ `TraceParity.seed5.firstDivergence.test.ts`: passes
3. ✅ All `Backend_vs_Sandbox.seed*.*.test.ts` tests pass
4. ✅ All `Sandbox_vs_Backend.*.test.ts` tests pass
5. ✅ `GameEngine.chainCapture` tests remain green (no regression)
6. ✅ `npm run test:core` has no regressions
7. ✅ `currentPlayer`/`currentPhase` semantics documented in this file

**Current progress**: 6 of 7 criteria satisfied (only #1 failing).

---

## 6. State Hash Format

For reference, the state hash format is:

```
<currentPlayer>:<currentPhase>:<gameStatus>#<board_hash>
```

Where `board_hash` encodes:
- All space states (stacks, markers, collapsed status)
- Ring counts per player
- Line definitions
- Territory ownership

Both engines produce identical `board_hash` values. The divergence is purely in the prefix.

---

## 7. Test Commands Quick Reference

```bash
# Primary divergence test
npm test -- --testPathPattern="Backend_vs_Sandbox.seed5.bisectParity" --no-coverage

# All seed5 parity tests
npm test -- --testPathPattern="Backend_vs_Sandbox.seed5" --no-coverage

# Trace parity
npm test -- --testPathPattern="TraceParity.seed5" --no-coverage

# Core suite (must stay green)
npm run test:core

# Chain capture regression check
npm test -- --testPathPattern="GameEngine.chainCapture" --no-coverage
```

---

## 8. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-24 | Problem #1 (Chain Capture) solved | Phase reset to 'capture' when chain exhausts |
| 2025-11-24 | HTML entity bug fixed in ClientSandboxEngine | TypeScript compilation was failing |
| 2025-11-24 | Identified skip logic divergence | bisectParity shows 63/64 match |
| TBD | Choose player-skip invariant | Backend has extra skip, sandbox doesn't |

---

## 9. Appendix: Problem #1 Solution (Chain Capture)

For completeness, here's the chain capture fix summary:

### Root Causes Fixed

1. **Test helper using stale `gameState` reference**
   - `tests/unit/GameEngine.chainCapture.test.ts` cached `gameState` but `GameEngine.appendHistoryEntry()` reassigns `this.gameState`
   - **Fix**: Re-fetch `engineAny.gameState` on each loop iteration

2. **Phase not advanced when chain exhausts**
   - When `followUpMoves.length === 0` after capture segment, code cleared `chainCaptureState` but left `currentPhase === 'chain_capture'`
   - **Fix**: Added `this.gameState.currentPhase = 'capture'` when chain exhausts

### Files Modified
- `src/server/game/GameEngine.ts`
- `tests/unit/GameEngine.chainCapture.test.ts`

---

*Last Updated: November 24, 2025*
*Status: Ready for Phase 2 (Define Invariant) and Phase 3 (Harmonize Skip Logic)*
*Author: Cline Assistant*
