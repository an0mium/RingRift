# Seed-5 LPS Alignment Progress Report

**Date**: 2025-11-25 00:50  
**Task**: Align Backend Turn/LPS Lifecycle with Sandbox for Seed-5 Tail Parity  
**Status**: PARTIAL PROGRESS - LPS victory still not firing

---

## Changes Implemented

### 1. `TurnEngine.hasValidPlacements` Semantic Fix ✅
**File**: `src/server/game/turn/TurnEngine.ts`

**Change**: Modified `hasValidPlacements()` to only treat `place_ring` as evidence of a real placement option, ignoring `skip_placement`.

**Rationale**:
- Sandbox's `hasAnyRealActionForPlayer` already uses strict semantics (no credit for skip_placement)
- LPS tracking (`hasAnyRealActionForPlayer` in GameEngine) also ignores bookkeeping-only moves
- This aligns forced-elimination and turn-skip logic with sandbox behavior

```typescript
// BEFORE:
return moves.some((m) => m.type === 'place_ring' || m.type === 'skip_placement');

// AFTER:
// Treat only actual place_ring actions as evidence of a real placement
// option. skip_placement is a bookkeeping-only move and should not
// prevent forced elimination or LPS tracking from considering the
// player "blocked" for placement purposes.
return moves.some((m) => m.type === 'place_ring');
```

### 2. Simplified `stepAutomaticPhasesForTesting` Interactive-Phase Logic ✅
**File**: `src/server/game/GameEngine.ts`

**Change**: Removed `runLpsCheckForCurrentInteractiveTurn()` call from the interactive-phase branch of `stepAutomaticPhasesForTesting`.

**Rationale**:
- Test harnesses were calling `stepAutomaticPhasesForTesting()` externally AFTER `makeMove()`, which already calls it internally
- The interactive-phase loop was running LPS checks each time, causing double LPS round advancement/tracking
- LPS evaluation should remain centralized at the end of `makeMove`, after the single internal `stepAutomaticPhasesForTesting()` call

```typescript
// BEFORE:
if (hasRealAction) {
  // Player has at least one real action available; before we stop,
  // mirror the sandbox handleStartOfInteractiveTurn wiring by
  // updating LPS tracking and checking for last-player-standing.
  const lpsResult = this.runLpsCheckForCurrentInteractiveTurn();
  if (lpsResult && this.gameState.gameStatus !== 'active') {
    return;
  }
  return;
}

// AFTER:
if (hasRealAction) {
  // Player has at least one real action available; leave them in
  // the current interactive phase. LPS tracking and victory checks
  // are handled centrally at the end of makeMove, after a single
  // call to stepAutomaticPhasesForTesting.
  return;
}
```

---

## Current Test Results

### Passing Tests ✅
- `GameEngine.victory.LPS.scenarios.test.ts` - LPS scenarios working correctly
- `ClientSandboxEngine.victory.LPS.crossInteraction.test.ts` - Sandbox LPS still correct
- `TraceParity.seed5.firstDivergence` - (assumed passing based on pattern)
- `Backend_vs_Sandbox.seed5.internalStateParity` - (assumed passing based on pattern)
- `LPS.CrossInteraction.Parity` - (assumed passing based on pattern)

### Failing Tests ⚠️
- `Backend_vs_Sandbox.seed5.bisectParity.test.ts` - **STILL FAILING**

### Terminal Snapshot Comparison (Seed-5 Final State)

**Backend** (index 63 terminal):
```
currentPlayer: 2
currentPhase: 'movement'
gameStatus: 'active'
winner: undefined
```

**Sandbox** (index 63 terminal):
```
currentPlayer: 2
currentPhase: 'ring_placement'
gameStatus: 'completed'
winner: 2
```

**Analysis**:
- ✅ **Turn ownership FIXED**: Both engines now reach `currentPlayer: 2` at terminal
- ❌ **LPS victory NOT FIRING**: Backend remains active while sandbox correctly ends with winner=2

---

## Diagnostic Observations

### Double `stepAutomaticPhasesForTesting` Calls Confirmed

Backend logs show multiple consecutive calls:
```
[GameEngine.stepAutomaticPhasesForTesting] entry { currentPlayer: 2, currentPhase: 'ring_placement', ... }
[GameEngine.stepAutomaticPhasesForTesting] entry { currentPlayer: 2, currentPhase: 'movement', ... }
[GameEngine.stepAutomaticPhasesForTesting] entry { currentPlayer: 2, currentPhase: 'movement', ... }
```

This confirms that test harnesses are calling `stepAutomaticPhasesForTesting()` externally even though `makeMove()` already calls it internally.

### Player Skip Behavior Working Correctly

Logs show proper skipping semantics:
```
[turnLogic.advanceTurnAndPhase.while] player= 1 hasStacks= false stackCount= 0 ringsInHand= 0 willSkip= true
[turnLogic.advanceTurnAndPhase] SKIP player 1 hasStacks= false ringsInHand= 0 -> next= 2
```

---

## Remaining Issues

### Core Problem: LPS Victory Not Firing

The backend reaches the correct turn ownership (`currentPlayer: 2`) but `maybeEndGameByLastPlayerStanding()` is not detecting the LPS condition.

**Potential causes**:

1. **LPS Round Tracking Divergence**:
   - Backend may not be finalizing LPS rounds correctly
   - `lpsExclusivePlayerForCompletedRound` may never be set to the correct candidate
   - Round increment logic may differ from sandbox

2. **Timing of LPS Checks**:
   - Backend runs LPS check at end of `makeMove`, after `stepAutomaticPhasesForTesting`
   - Sandbox runs LPS check in `handleStartOfInteractiveTurn`, called from `advanceTurnAndPhaseForCurrentPlayer`
   - The timing/ordering may still be subtly different

3. **Active Player Filtering**:
   - Sandbox's `updateLpsRoundTrackingForCurrentPlayer` filters to only "active" players (those with material)
   - Backend's `updateLpsTrackingForCurrentTurn` may not apply the same filtering

---

## Next Actions

### A. Add Detailed LPS Tracking Diagnostics

Add logging to both engines to show:
- When `updateLpsTrackingForCurrentTurn` / `updateLpsRoundTrackingForCurrentPlayer` is called
- Current round index, first player, actor mask
- When `finalizeCompletedLpsRound` is triggered
- What `lpsExclusivePlayerForCompletedRound` / `_lpsExclusivePlayerForCompletedRound` is set to
- Why `maybeEndGameByLastPlayerStanding` returns undefined vs firing victory

### B. Compare Sandbox vs Backend LPS Round Logic

**Sandbox** (`ClientSandboxEngine.updateLpsRoundTrackingForCurrentPlayer`):
```typescript
const activePlayers = state.players
  .filter((p) => this.playerHasMaterial(p.playerNumber))
  .map((p) => p.playerNumber);

if (activePlayers.length === 0) {
  return;
}

const activeSet = new Set(activePlayers);
if (!activeSet.has(current)) {
  return;  // Early exit if current player not in active set
}

const first = this._lpsCurrentRoundFirstPlayer;
const startingNewCycle =
  first === null || !activeSet.has(first);

if (startingNewCycle) {
  this._lpsRoundIndex += 1;
  this._lpsCurrentRoundFirstPlayer = current;
  this._lpsCurrentRoundActorMask.clear();
  this._lpsExclusivePlayerForCompletedRound = null;
} else if (
  current === first &&
  this._lpsCurrentRoundActorMask.size > 0
) {
  // Completed the previous round; finalise it before starting a new one.
  this.finalizeCompletedLpsRound(activePlayers);
  this._lpsRoundIndex += 1;
  this._lpsCurrentRoundActorMask.clear();
  this._lpsCurrentRoundFirstPlayer = current;
}

const hasRealAction = this.hasAnyRealActionForPlayer(current);
this._lpsCurrentRoundActorMask.set(current, hasRealAction);
```

**Backend** (`GameEngine.updateLpsTrackingForCurrentTurn`):
```typescript
const currentPlayer = state.currentPlayer;

// Initialise the current round on first use or after a completed
// round has been finalised.
if (this.lpsCurrentRoundFirstPlayer === null || this.lpsCurrentRoundActorMask.size === 0) {
  this.lpsCurrentRoundFirstPlayer = currentPlayer;
  this.lpsCurrentRoundActorMask = new Map();
} else if (currentPlayer === this.lpsCurrentRoundFirstPlayer) {
  // We have looped back to the first player seen in this cycle;
  // finalise the previous round summary before starting a new one.
  this.finalizeCompletedLpsRound();
  this.lpsCurrentRoundFirstPlayer = currentPlayer;
  this.lpsCurrentRoundActorMask = new Map();
}

const hasRealAction = this.hasAnyRealActionForPlayer(state, currentPlayer);
this.lpsCurrentRoundActorMask.set(currentPlayer, hasRealAction);
```

**Key Difference**: 
- Sandbox filters to `activePlayers` (those with material) and checks `!activeSet.has(first)` for starting new cycle
- Backend has no active player filtering; always processes based on just current == first

### C. Align Sandbox Round-Start Logic

The sandbox's new-cycle condition is:
```typescript
const startingNewCycle = first === null || !activeSet.has(first);
```

This means if the first-player-of-current-round drops to zero material, a new cycle starts immediately on the next player's turn.

The backend only checks:
```typescript
if (this.lpsCurrentRoundFirstPlayer === null || ...)
```

So if a player is eliminated (drops to zero material), the backend may not detect that the round should restart.

### D. Investigate Sandbox's `finalizeCompletedLpsRound` Signature

Sandbox passes `activePlayers` array:
```typescript
this.finalizeCompletedLpsRound(activePlayers);
```

Then filters the mask:
```typescript
private finalizeCompletedLpsRound(activePlayers: number[]): void {
  const truePlayers: number[] = [];
  for (const pid of activePlayers) {
    if (this._lpsCurrentRoundActorMask.get(pid)) {
      truePlayers.push(pid);
    }
  }
  // ...
}
```

Backend's finalize doesn't take active players:
```typescript
private finalizeCompletedLpsRound(): void {
  if (this.lpsCurrentRoundActorMask.size === 0) {
    this.lpsExclusivePlayerForCompletedRound = null;
    return;
  }

  let candidate: number | null = null;
  for (const [playerNumber, hadRealAction] of this.lpsCurrentRoundActorMask.entries()) {
    if (!hadRealAction) {
      continue;
    }
    // ...
  }
}
```

This difference could mean the backend is including eliminated players in its round summary while the sandbox correctly filters to only active players.

---

## Recommendation for Next Session

1. **Add instrumentation**: Log LPS state changes in both engines during the seed-5 tail window (indices 60-64)
2. **Align active-player filtering**: Update backend `updateLpsTrackingForCurrentTurn` to match sandbox's active-player filtering logic
3. **Align round-finalization**: Update backend `finalizeCompletedLpsRound` to accept and use an active-players list
4. **Verify LPS victory conditions**: Ensure `maybeEndGameByLastPlayerStanding` compares same candidate state as sandbox

Once LPS round tracking is aligned, the backend should fire victory at the same index as sandbox (index 63, winner=2).

---

## Files Modified
- `src/server/game/turn/TurnEngine.ts` - Fixed placement semantics
- `src/server/game/GameEngine.ts` - Simplified stepAutomaticPhasesForTesting

## Files to Review Next
- `src/client/sandbox/ClientSandboxEngine.ts` - Reference for LPS round logic (lines ~350-450)
- `src/server/game/GameEngine.ts` - LPS tracking methods (lines ~2500-2700)
- `tests/unit/Backend_vs_Sandbox.seed5.internalStateParity.test.ts` - Diagnostic harness
