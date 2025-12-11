# Architectural Improvement Plan

**Created:** 2025-12-11
**Status:** Active
**Priority:** Medium-High (technical debt reduction)

## Overview

This document captures architectural refactoring opportunities identified during the production readiness audit. The goal is to improve maintainability, debuggability, and testability while keeping the codebase as simple as possible.

## Guiding Principles

1. **Canonical Rules as SSOT**: All changes must preserve compliance with RULES_CANONICAL_SPEC.md
2. **Simplicity Over Cleverness**: Prefer straightforward code over abstractions
3. **One Domain = One Module**: Each aggregate handles one game concept
4. **Test-Driven Refactoring**: Improve coverage before restructuring
5. **Incremental Progress**: Small, reversible changes over large rewrites

---

## Priority 1: Quick Wins (Low Complexity, High Impact)

### 1.1 Remove Console.log Statements from Production Code

**Files:** `turnOrchestrator.ts` (lines 1408, 1419, 1469, 1850, 2200, 2946+)
**Effort:** 1 hour
**Impact:** Code quality, performance

Console statements pollute browser console and impact performance. Replace with `debugLog()` utility.

**Status Update (2025-12-11):**
Investigated - all console.log statements in turnOrchestrator.ts are already properly guarded:

- Line 1408+: Guarded by `process.env.RINGRIFT_TRACE_DEBUG === '1'`
- Line 1850: Guarded by `process.env.NODE_ENV === 'test'`
- Line 2200: Guarded by `process.env.NODE_ENV === 'test'`
- Line 2946: Guarded by `process.env.RINGRIFT_TRACE_DEBUG === '1'`

**No action needed** - existing guards prevent production pollution.

**Status:** ✅ Complete (No changes required)

### 1.2 Extract Shared Decision Helpers

**Files:** `lineDecisionHelpers.ts`, `territoryDecisionHelpers.ts`
**Effort:** 2-4 hours
**Impact:** DRY principle, maintainability

Identical `computeNextMoveNumber()` function duplicated in both files. Create shared module.

**Completed (2025-12-11):**

- Created `src/shared/engine/sharedDecisionHelpers.ts` with centralized `computeNextMoveNumber()`
- Updated `lineDecisionHelpers.ts` and `territoryDecisionHelpers.ts` to import from shared module
- Added dedicated test file `tests/unit/sharedDecisionHelpers.test.ts` (7 test cases)
- All 105 related tests passing

**Status:** ✅ Complete

### 1.3 Create Board View Factory

**Files:** `globalActions.ts`, `MovementAggregate.ts`, `CaptureAggregate.ts`
**Effort:** 4-6 hours
**Impact:** Consistency, testability

Board view adapters created inline in multiple files. Centralize into factory.

**Status Update (2025-12-11):**
Already implemented! `src/client/sandbox/boardViewFactory.ts` provides:

- `createBoardView()` - Unified adapter factory
- `createMovementBoardView()` - Movement-specific adapter
- `createCaptureBoardAdapters()` - Capture-specific adapter
- `createSandboxBoardView()` - Sandbox-style adapter (board as param)
- `bindSandboxViewToBoard()` - Adapter conversion utility
- Test file exists: `tests/unit/sandbox/boardViewFactory.test.ts`

**Status:** ✅ Complete (Already implemented)

---

## Priority 2: Coverage Improvements (Before Major Refactoring)

### 2.1 TurnOrchestrator Coverage (42.7% → 80%)

**Key Gaps Identified:**

- `buildGameEndExplanationForVictory` (318 lines) - victory explanation edge cases
- `processPostMovePhases` (414 lines) - phase transitions
- `resolveANMForCurrentPlayer` - ANM resolution loop
- Decision creation functions (11 functions at 0% coverage)

**Functions Needing Tests:**
| Function | Lines | Current Coverage | Priority |
|----------|-------|------------------|----------|
| computeNextNonEliminatedPlayer | 160-183 | 0% | P1 |
| createForcedEliminationDecision | 1034-1087 | ✅ Tested | P1 |
| createLineOrderDecision | 971-983 | ✅ Tested | P1 |
| createRegionOrderDecision | 988-1022 | ✅ Tested | P1 |
| createChainCaptureDecision | 1092-1101 | ✅ Tested | P1 |
| detectTerritoryMiniRegions | 817-850 | 0% | P2 |
| groupIntoConnectedRegions | 871-912 | 0% | P2 |

**Progress (2025-12-11):**

- Added 5 new test cases for decision creation functions
- Fixed 2 flaky tests in phaseTransitions that assumed specific winner values
- Added ANM resolution and turn advancement test cases
- All 235 turnOrchestrator tests passing
- Current coverage: **74.57% statements, 69.12% branches, 84.84% functions**
- Remaining gaps primarily in:
  - Victory explanation edge cases (lines 576-612)
  - Decision surface building internals (lines 1128-1182)
  - Phase transition internals (lines 2484-2521)
  - ANM resolution loop edge cases (lines 254-272)

**Status:** ⏳ In Progress (74.57% → 80% target)

### 2.2 CaptureAggregate Coverage (51.7% → 80%)

**Key Gaps Identified:**

- `mutateCapture()` (lines 631-780) - CRITICAL, 0% coverage
- `enumerateCaptureMoves()` (lines 381-489) - CRITICAL, 0% coverage
- `validateCapture()` (lines 316-365) - HIGH, 0% coverage

**Estimated New Tests Needed:** 60-80 test cases

**Status Update (2025-12-11):**

- Current coverage: **96.23% statements, 92.85% branches** ✅
- 96 tests passing across 3 test files
- Remaining uncovered lines are defensive code paths (unreachable in normal game)

**Status:** ✅ Target Met (96%)

### 2.3 MovementAggregate Coverage (51.7% → 80%)

Similar gaps to CaptureAggregate. Core mutation and enumeration logic needs testing.

**Status Update (2025-12-11):**

- Current coverage: **93.51% statements, 88.15% branches** ✅
- 74 tests passing across 2 test files
- Remaining uncovered lines are edge case defensive code

**Status:** ✅ Target Met (93%)

### 2.4 LineAggregate Coverage (67.2% → 80%)

Line detection and collapse logic needs additional edge case tests.

**Status Update (2025-12-11):**

- Current coverage: **94.31% statements, 82.66% branches** ✅
- 104 tests passing across 3 test files
- Remaining uncovered lines are edge case scenarios

**Status:** ✅ Target Met (94%)

---

## Priority 3: Medium Complexity Refactoring

### 3.1 Consistent Error Handling

**Effort:** 2-3 days
**Impact:** Debugging, user feedback

Create structured `EngineError` base class with:

- `RulesViolation` - invalid moves per game rules
- `InvalidState` - corrupted game state
- `BoardConstraintViolation` - geometry/topology issues

**Status:** ⬜ Not Started

### 3.2 Strong Typing for Decisions

**Effort:** 1-2 days
**Impact:** Type safety, IDE support

Create discriminated unions for `PendingDecision` types:

```typescript
export type PendingDecision =
  | { type: 'line_order'; options: LineOrderMove[]; ... }
  | { type: 'territory_region'; options: TerritoryMove[]; ... }
  | { type: 'forced_elimination'; options: EliminationMove[]; ... }
```

**Status:** ⬜ Not Started

### 3.3 Consolidate Validator/Mutator Pairs

**Effort:** 1-2 days
**Impact:** Single source of truth

Move remaining standalone validators into aggregates. Document "one domain = one aggregate" principle.

**Status:** ⬜ Not Started

---

## Priority 4: Large Refactoring (After Coverage Goals Met)

### 4.1 Split TurnOrchestrator (3,232 lines)

**Effort:** 2-3 days
**Impact:** Maintainability, testability

Extract into focused modules:

- `VictoryOrchestrator.ts` - victory evaluation and explanation
- `ANMResolution.ts` - active no-moves handling
- `DecisionSurfaceBuilder.ts` - decision construction
- `TurnMetadata.ts` - history/metadata recording

**Prerequisites:**

- [ ] TurnOrchestrator coverage ≥80%
- [ ] All parity tests passing
- [ ] Comprehensive integration tests

**Status:** ⬜ Blocked (coverage prerequisite)

### 4.2 Extract Heuristic Helpers (1,450 lines)

**Effort:** 1 day
**Impact:** Reusability, clarity

Create `boardTraversal.ts` and `PositionHelpers.ts` for shared utilities.

**Status:** ⬜ Not Started

### 4.3 Resolve FSM Duality

**Effort:** 1 day
**Impact:** Clarity, remove dead code

Migrate fully to `TurnStateMachine`, deprecate `PhaseStateMachine`.

**Status:** ⬜ Not Started

---

## Progress Tracking

| Phase           | Items | Complete | Status                                         |
| --------------- | ----- | -------- | ---------------------------------------------- |
| Quick Wins      | 3     | 3        | ✅ Complete                                    |
| Coverage        | 4     | 3        | ⏳ In Progress (TurnOrchestrator 74.57% → 80%) |
| Medium Refactor | 3     | 0        | ⬜ Not Started                                 |
| Large Refactor  | 3     | 0        | ⬜ Blocked                                     |

---

## Implementation Notes

### When Adding Tests

1. Use existing test file patterns (_.branchCoverage.test.ts for coverage, _.shared.test.ts for integration)
2. Reference canonical rules by ID (e.g., RR-CANON-R022)
3. Prefer realistic scenarios over synthetic edge cases
4. Update WEAK_ASSERTION_AUDIT.md if strengthening assertions

### When Refactoring

1. Run full test suite before and after
2. Use feature flags for gradual rollout if needed
3. Update MODULE_RESPONSIBILITIES.md for any responsibility changes
4. Document architectural decisions in this file

---

## Related Documents

- [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md) - Single source of truth for game rules
- [MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md) - Current module breakdown
- [RULES_ENGINE_ARCHITECTURE.md](RULES_ENGINE_ARCHITECTURE.md) - Architecture overview
- [WEAK_ASSERTION_AUDIT.md](WEAK_ASSERTION_AUDIT.md) - Test assertion quality tracking
- [PRODUCTION_READINESS_CHECKLIST.md](PRODUCTION_READINESS_CHECKLIST.md) - Launch criteria
