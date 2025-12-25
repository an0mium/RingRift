# Phase Orchestration Architecture

**Created**: 2025-12-11
**Updated**: 2025-12-25
**Status**: Reference Documentation

## Overview

RingRift uses a unified FSM-based architecture for phase orchestration. The `TurnStateMachine` and `FSMAdapter` provide type-safe validation and state tracking for all phase transitions.

## The FSM System

### TurnStateMachine - Canonical Orchestration

**Location**: `src/shared/engine/fsm/TurnStateMachine.ts`

The FSM is a type-safe finite state machine that:

- Defines all valid (state, event) → nextState transitions
- Provides compile-time guarantees against invalid transitions
- Is the **canonical validator** for all moves via `validateMoveWithFSM()`

```typescript
// FSM validation is always used
const fsmValidationResult = validateMoveWithFSM(state, move);
if (!fsmValidationResult.valid) {
  throw new Error(`Invalid move: ${fsmValidationResult.reason}`);
}
```

### FSMAdapter - Bridge Layer

**Location**: `src/shared/engine/fsm/FSMAdapter.ts`

The FSMAdapter:

- Bridges the type-safe FSM with game types
- Computes orchestration results (next phase, player rotation, pending decisions)
- Provides the integration layer between FSM states and game state

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        processTurn()                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. VALIDATION (FSM)                                             │
│     ┌──────────────────────────────────────────┐                 │
│     │ validateMoveWithFSM(state, move)         │                 │
│     │ - Phase-aware validation                 │                 │
│     │ - Type-safe transition guards            │                 │
│     │ - Returns FSMValidationResult            │                 │
│     └──────────────────────────────────────────┘                 │
│                         │                                        │
│                         ▼                                        │
│  2. MOVE APPLICATION (Domain Aggregates)                         │
│     ┌──────────────────────────────────────────┐                 │
│     │ applyMoveWithChainInfo()                 │                 │
│     │ - Delegates to PlacementAggregate,       │                 │
│     │   MovementAggregate, CaptureAggregate    │                 │
│     │ - Returns updated state + chain info     │                 │
│     └──────────────────────────────────────────┘                 │
│                         │                                        │
│                         ▼                                        │
│  3. POST-MOVE PROCESSING                                         │
│     ┌──────────────────────────────────────────┐                 │
│     │ processPostMovePhases()                  │                 │
│     │ - Line detection, territory processing   │                 │
│     │ - Victory evaluation                     │                 │
│     │ - Inline state tracking (no external dep)│                 │
│     └──────────────────────────────────────────┘                 │
│                         │                                        │
│                         ▼                                        │
│  4. PHASE RESOLUTION (FSM)                                       │
│     ┌──────────────────────────────────────────┐                 │
│     │ computeFSMOrchestration()                │                 │
│     │ - Determines next phase                  │                 │
│     │ - Computes pending decisions             │                 │
│     │ - Handles player rotation                │                 │
│     └──────────────────────────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Historical Context

The legacy `phaseStateMachine.ts` module was deprecated and removed in PASS30-R1. Its functionality has been consolidated into:

- `TurnStateMachine.ts` for phase transition rules
- `FSMAdapter.ts` for bridging FSM with game types
- Inline state tracking in `turnOrchestrator.ts`

## Guidelines for Developers

### When Adding New Validation Logic

**Use FSM** (`TurnStateMachine.ts` / `FSMAdapter.ts`):

```typescript
// Add new phase guards in TurnStateMachine
// Add validation helpers in FSMAdapter
```

### When Adding New State Tracking

Extend the inline tracking in `turnOrchestrator.ts` or add helpers to the FSM adapter layer.

## File Reference

| File                                | Purpose                          | Status    |
| ----------------------------------- | -------------------------------- | --------- |
| `fsm/TurnStateMachine.ts`           | Type-safe FSM states/transitions | Canonical |
| `fsm/FSMAdapter.ts`                 | Bridges FSM with game types      | Canonical |
| `fsm/index.ts`                      | FSM public API                   | Canonical |
| `orchestration/turnOrchestrator.ts` | Main orchestrator                | Canonical |

## Python Parity

| TypeScript            | Python             | Status        |
| --------------------- | ------------------ | ------------- |
| `TurnStateMachine.ts` | `fsm.py`           | Experimental  |
| `FSMAdapter.ts`       | (partial)          | In progress   |
| (removed)             | `phase_machine.py` | Legacy parity |

Python currently uses `phase_machine.py` for phase logic. Full FSM parity is in progress.
