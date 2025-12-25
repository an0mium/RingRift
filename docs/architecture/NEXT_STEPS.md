# Next Steps: Architecture Improvements

> **Doc Status (2025-12-11): Active**
>
> This document tracks prioritized improvements identified through codebase analysis.
> Updated after each work session.

**Created:** 2025-12-11
**Last Updated:** 2025-12-11

---

## Overview

This document tracks actionable improvements to the RingRift codebase, prioritized by impact and aligned with the canonical rules as SSOT. The goal is to maintain a simple, debuggable architecture with clear responsibilities.

## Related Documents

| Document                                                             | Purpose                                  |
| -------------------------------------------------------------------- | ---------------------------------------- |
| [ARCHITECTURAL_DEBT_ASSESSMENT.md](ARCHITECTURAL_DEBT_ASSESSMENT.md) | Detailed debt tracking with progress     |
| [REFACTORING_RECOMMENDATIONS.md](REFACTORING_RECOMMENDATIONS.md)     | Bookkeeping architecture recommendations |
| [VICTORY_REFACTORING_PLAN.md](VICTORY_REFACTORING_PLAN.md)           | Victory consolidation (COMPLETE)         |
| [TEST_HYGIENE_NOTES.md](TEST_HYGIENE_NOTES.md)                       | Test organization and cleanup            |

---

## Priority Summary

| Priority | Item                                    | Status       | Impact                                |
| -------- | --------------------------------------- | ------------ | ------------------------------------- |
| P0.1     | Validator/Aggregate Consolidation       | **BLOCKED**  | Requires GameState type unification   |
| P0.2     | Deprecated Phase Orchestrator Cleanup   | Deferred     | Blocked until FSM migration complete  |
| P1.1     | Test Suite Cleanup (Orchestrator Skips) | **COMPLETE** | 36 files documented as legacy         |
| P1.2     | Dead Code Removal                       | **COMPLETE** | Major cleanup done; 447 LOC blocked   |
| P2.1     | AI Training Pipeline Improvements       | Documented   | Checkpoint persistence, loss tracking |

---

## P0.1: Validator/Aggregate Consolidation

### Problem

The codebase has two parallel structures for validation/mutation:

- `/src/shared/engine/validators/` - Original validators (~30K LOC total)
- `/src/shared/engine/aggregates/` - Consolidated aggregates (canonical)

### Status: BLOCKED by GameState Type Unification

**Attempted:** Dec 11, 2025 - Simple re-export approach failed due to type incompatibility.

**Root Cause:** The codebase has TWO incompatible `GameState` types:

1. `shared/engine/types.ts:GameState` - Used by validators and GameEngine.ts
2. `shared/types/game.ts:GameState` - Used by aggregates (has additional fields: `boardType`, `history`, `spectators`)

These types are structurally different, causing TypeScript errors when validators re-export from aggregates.

### Technical Details

```typescript
// shared/engine/types.ts (used by validators)
export interface GameState {
  readonly id: string;
  readonly board: BoardState;
  readonly players: ReadonlyArray<Player>;
  // ... 14 fields total
}

// shared/types/game.ts (used by aggregates)
export interface GameState {
  // ... all of the above PLUS:
  boardType: BoardType; // Additional field
  history: GameEvent[]; // Additional field
  spectators: string[]; // Additional field
}
```

### Resolution Options

1. **Unify GameState types** (Recommended, High Effort)
   - Merge the two types into one canonical definition
   - Update all consumers to use the unified type
   - Most correct long-term solution

2. **Make aggregates accept both types** (Medium Effort)
   - Use generic type parameters or type unions
   - More complex type signatures

3. **Keep validators as-is** (Low Effort)
   - Accept the duplication for now
   - Mark as future consolidation opportunity

### Current Decision: Keep validators as-is

The validator consolidation is deferred until GameState type unification is addressed. The validators remain functional and are not blocking other work.

### Files (For Reference)

| File                               | Status    | Notes                       |
| ---------------------------------- | --------- | --------------------------- |
| `validators/PlacementValidator.ts` | ACTIVE    | Uses engine/types.GameState |
| `validators/MovementValidator.ts`  | ACTIVE    | Uses engine/types.GameState |
| `validators/CaptureValidator.ts`   | ACTIVE    | Uses engine/types.GameState |
| `validators/LineValidator.ts`      | ACTIVE    | Uses engine/types.GameState |
| `validators/TerritoryValidator.ts` | ACTIVE    | Uses engine/types.GameState |
| `validators/utils.ts`              | ACTIVE    | Shared utilities (keep)     |
| `aggregates/*.ts`                  | CANONICAL | Uses types/game.GameState   |

### Estimated Effort: 2-3 sessions (after GameState unification)

---

## P0.2: Deprecated Phase Orchestrator Cleanup

### Problem

Two competing phase transition systems existed:

1. **Legacy**: `phaseStateMachine.ts` (447 LOC) + `phase_machine.py` (380 LOC)
2. **Canonical**: `TurnStateMachine.ts` (500+ LOC) via FSM

### Status: COMPLETE (PASS30-R1)

**Completed:** Dec 2025

The legacy `phaseStateMachine.ts` was fully removed in PASS30-R1. All phase transition logic now uses the canonical FSM (`TurnStateMachine.ts`, `FSMAdapter.ts`).

### Completed Steps

1. [x] Complete FSM migration in turnOrchestrator
2. [x] Verify all phase transitions work through FSMAdapter
3. [x] Remove phaseStateMachine.ts exports from engine/index.ts
4. [x] Delete phaseStateMachine.ts (removed in PASS30-R1)
5. [ ] Plan Python FSM equivalent (ongoing, `phase_machine.py` remains)

### Remaining Work

| File                                    | Lines | Status                         |
| --------------------------------------- | ----- | ------------------------------ |
| `ai-service/app/rules/phase_machine.py` | 380   | Active - Python FSM equivalent |

---

## P1.1: Test Suite Cleanup (Orchestrator Skips)

### Problem

Multiple test suites have conditional skips based on `ORCHESTRATOR_ADAPTER_ENABLED`:

```typescript
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';
(skipWithOrchestrator ? describe.skip : describe)('...', () => {...});
```

Since `ORCHESTRATOR_ADAPTER_ENABLED=true` is set in `.env`, these tests ARE being skipped. The orchestrator is now canonical, so these tests either:

1. Test legacy code paths that are no longer used
2. Need to be updated to work with the orchestrator
3. Should be archived/documented as historical

### Files with Orchestrator Skips (20+)

| File                                                                       | Skip Type              | Action Needed |
| -------------------------------------------------------------------------- | ---------------------- | ------------- |
| `TerritoryDecisions.HexRegionThenElim.GameEngine_vs_Sandbox.test.ts`       | `skipWithOrchestrator` | Evaluate      |
| `Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts`                         | `skipWithOrchestrator` | Evaluate      |
| `Seed5Move63.lineDetectionSnapshot.test.ts`                                | `skipWithOrchestrator` | Evaluate      |
| `TerritoryDecisions.SquareTwoRegionThenElim.GameEngine_vs_Sandbox.test.ts` | `skipWithOrchestrator` | Evaluate      |
| `GameEngine.chainCapture.test.ts`                                          | `orchestratorEnabled`  | Evaluate      |
| `TerritoryDecisions.SquareRegionThenElim.GameEngine_vs_Sandbox.test.ts`    | `skipWithOrchestrator` | Evaluate      |
| `ClientSandboxEngine.chainCapture.scenarios.test.ts`                       | `orchestratorEnabled`  | Evaluate      |
| `ClientSandboxEngine.regionOrderChoice.test.ts`                            | `skipWithOrchestrator` | Evaluate      |
| `ClientSandboxEngine.traceStructure.test.ts`                               | `skipWithOrchestrator` | Evaluate      |

### Resolution Plan

1. [ ] Audit each skipped test to determine if it tests:
   - Legacy code path (archive)
   - Feature that should work with orchestrator (fix/update)
   - Historical debugging scenario (document and archive)
2. [ ] For each test, either:
   - Remove skip and verify passing
   - Update to work with orchestrator
   - Move to archive with documentation
3. [ ] Update TEST_HYGIENE_NOTES.md with decisions

### Estimated Effort: 1 session

---

## P1.2: Dead Code Removal

### Status: COMPLETE (No further action needed)

Several modules had `@deprecated` annotations. Analysis and cleanup completed:

1. **phaseStateMachine.ts** (447 LOC) - **REMOVED** in PASS30-R1 (Dec 2025)
2. **hashGameState** - Not actually dead; actively used for parity validation
3. **No-op methods in GameEngine** - Backward compat shims called by test suites
4. **Legacy move types** - Required for historical game recordings

### Completed Removals

| File                   | Lines | Status  | Notes                                       |
| ---------------------- | ----- | ------- | ------------------------------------------- |
| `phaseStateMachine.ts` | 447   | REMOVED | ✅ Deleted in PASS30-R1 (FSM now canonical) |
| `victoryLogic.ts`      | 418   | REMOVED | ✅ Done Dec 11                              |

### Already Completed

- [x] Removed `victoryLogic.ts` (418 LOC) - Dec 11, 2025
- [x] Removed unused helpers from TurnEngine.ts (~50 LOC)
- [x] Removed `hasAnyMovementOrCaptureForPlayer` from ClientSandboxEngine
- [x] Removed line direction helpers from sandboxLines.ts (~100 LOC)
- [x] Removed deprecated `RecoverySlideTarget.cost` field
- [x] Removed `phaseStateMachine.ts` (447 LOC) - PASS30-R1
- [x] Audited remaining @deprecated code - all in use

---

## P2.1: AI Training Pipeline Improvements

### Problem

Identified TODOs in training pipeline that affect iteration speed:

- Line 516 (curriculum.py): Checkpoint path support for curriculum training
- Line 2307 (train.py): Loss capture from training loop

### Status: DOCUMENTED

These are documented in the AI pipeline docs and queued for future work. Not blocking current development.

### Resolution Plan

1. [ ] Implement checkpoint persistence in curriculum training
2. [ ] Add loss tracking to monitoring system
3. [ ] Profile AI response times at difficulties 7-10

---

## Change Log

| Date       | Change                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------- |
| 2025-12-11 | Initial document created from codebase analysis                                              |
| 2025-12-11 | Identified P0.1 (validator consolidation) as immediate actionable item                       |
| 2025-12-11 | P0.1 BLOCKED: Discovered GameState type incompatibility between validators and aggregates    |
| 2025-12-11 | Documented GameState type unification as prerequisite for validator consolidation            |
| 2025-12-11 | P1.1 COMPLETE: Analyzed 36 orchestrator-skipped tests, documented as legacy diagnostic tests |
| 2025-12-11 | P1.2 COMPLETE: Audited @deprecated code - all either removed, in use, or blocked by P0.2     |

---

## Validation Checklist

Before marking any item complete:

- [ ] TypeScript compilation passes
- [ ] All Jest tests pass
- [ ] No new @deprecated annotations introduced
- [ ] Documentation updated
- [ ] Change log entry added
