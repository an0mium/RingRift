/**
 * FSM Module - Canonical Orchestrator for RingRift turn phases
 *
 * This module is the **canonical orchestrator** for all phase transitions and
 * move validation in RingRift. It implements a finite state machine that
 * governs the flow of game phases during a turn.
 *
 * ## Architecture
 *
 * The FSM consists of three main components:
 *
 * 1. **TurnStateMachine** - The pure FSM implementation with:
 *    - Type-safe phase states (RingPlacementState, MovementState, etc.)
 *    - Event-driven transitions (PLACE_RING, MOVE_STACK, CAPTURE, etc.)
 *    - Guard conditions that validate move legality
 *    - Action generation for game state mutations
 *
 * 2. **FSMAdapter** - Bridges the FSM with existing game types:
 *    - Converts Move objects to TurnEvents
 *    - Derives FSM state from GameState
 *    - Provides validation functions (validateMoveWithFSM)
 *    - Computes orchestration results (computeFSMOrchestration)
 *
 * 3. **FSMDecisionSurface** - Exposes pending decisions:
 *    - Lines requiring reward choices
 *    - Territory regions requiring processing
 *    - Chain capture continuations
 *    - Forced elimination targets
 *
 * ## Usage
 *
 * ```typescript
 * import {
 *   validateMoveWithFSM,
 *   computeFSMOrchestration,
 *   isMoveTypeValidForPhase,
 * } from './fsm';
 *
 * // Validate a move
 * const result = validateMoveWithFSM(gameState, move);
 * if (!result.valid) {
 *   console.error(result.errorCode, result.reason);
 * }
 *
 * // Compute next phase after a move
 * const orchestration = computeFSMOrchestration(gameState, move);
 * if (orchestration.success) {
 *   // Apply actions and transition to orchestration.nextPhase
 * }
 * ```
 *
 * ## Migration from Legacy Orchestration
 *
 * The FSM replaces the legacy `PhaseStateMachine` in
 * `../orchestration/phaseStateMachine.ts`. That module is deprecated but
 * retained for backwards compatibility. New code should always use this
 * FSM module.
 *
 * @module fsm
 */

export {
  TurnStateMachine,
  transition,
  type TurnState,
  type TurnEvent,
  type TransitionResult,
  type TransitionError,
  type Action,
  type GameContext,
  // Phase states
  type RingPlacementState,
  type MovementState,
  type CaptureState,
  type ChainCaptureState,
  type LineProcessingState,
  type TerritoryProcessingState,
  type ForcedEliminationState,
  type TurnEndState,
  type GameOverState,
  // Context types
  type DetectedLine,
  type DisconnectedRegion,
  type CaptureContext,
  type EliminationTarget,
  // Enums / helpers
  type VictoryReason,
  type Direction,
  type LineRewardChoice,
  // High-level phase completion helpers used by orchestrators / parity tooling
  type PhaseAfterLineProcessing,
  type PhaseAfterTerritoryProcessing,
  onLineProcessingComplete,
  onTerritoryProcessingComplete,
} from './TurnStateMachine';

// Adapter for bridging FSM with existing Move types
export {
  moveToEvent,
  eventToMove,
  deriveStateFromGame,
  deriveGameContext,
  validateEvent,
  getValidEvents,
  describeActionEffects,
  // FSM-based validation
  validateMoveWithFSM,
  validateMoveWithFSMAndCompare,
  isMoveTypeValidForPhase,
  type FSMValidationResult,
  // Debug logging
  setFSMDebugLogger,
  consoleFSMDebugLogger,
  type FSMDebugLogger,
  type FSMDebugContext,
  // Orchestration integration
  determineNextPhaseFromFSM,
  attemptFSMTransition,
  getCurrentFSMState,
  isFSMTerminalState,
  type PhaseTransitionContext,
  type FSMTransitionAttemptResult,
  // FSM-driven orchestration
  computeFSMOrchestration,
  compareFSMWithLegacy,
  type FSMOrchestrationResult,
  type FSMDecisionSurface,
  // Line reward extraction
  extractLineRewardChoice,
} from './FSMAdapter';
