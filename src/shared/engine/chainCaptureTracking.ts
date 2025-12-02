/**
 * Chain Capture State Tracking
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Shared helpers for managing chain capture state across GameEngine and
 * ClientSandboxEngine. This module provides a clean abstraction over the
 * lower-level CaptureAggregate functions.
 *
 * Rule Reference: RR-CANON-R084, R085 - Chain captures (consecutive captures
 * from same stack, must extend from previous capture position)
 *
 * @module chainCaptureTracking
 */

import type { GameState, Position, Move, GamePhase } from '../types/game';
import { positionToString } from '../types/game';
import {
  ChainCaptureState,
  ChainCaptureSegment,
  getChainCaptureContinuationInfo,
  updateChainCaptureStateAfterCapture,
} from './aggregates/CaptureAggregate';

// Re-export types for convenience
export type { ChainCaptureState, ChainCaptureSegment };

/**
 * Minimal chain capture tracking state for engines that don't need
 * full segment history (like ClientSandboxEngine).
 */
export interface MinimalChainCaptureState {
  /** Player performing the capture chain */
  playerNumber: number;
  /** Current position of the capturing stack */
  currentPosition: Position;
  /** Whether the chain is currently active */
  isActive: boolean;
}

/**
 * Result of evaluating whether a chain capture should continue.
 */
export interface ChainCaptureEvaluation {
  /** True if chain must continue (captures available) */
  mustContinue: boolean;
  /** Available continuation moves */
  availableMoves: Move[];
  /** Recommended phase for the game state */
  recommendedPhase: GamePhase;
}

/**
 * Create an empty/inactive chain capture state.
 * Use this to initialize the chain capture tracking in an engine.
 */
export function createEmptyChainCaptureState(): MinimalChainCaptureState | null {
  return null;
}

/**
 * Create a full chain capture state from the first capture segment.
 * This is used by GameEngine which needs full segment history.
 */
export function createFullChainCaptureState(
  move: Move,
  capturedCapHeight: number
): ChainCaptureState | undefined {
  return updateChainCaptureStateAfterCapture(undefined, move, capturedCapHeight);
}

/**
 * Create a minimal chain capture state for simpler tracking.
 * This is used by ClientSandboxEngine which only needs position tracking.
 */
export function createMinimalChainCaptureState(
  playerNumber: number,
  currentPosition: Position
): MinimalChainCaptureState {
  return {
    playerNumber,
    currentPosition,
    isActive: true,
  };
}

/**
 * Update the chain capture state after a capture segment is applied.
 *
 * For full state (GameEngine):
 * - Updates currentPosition to the landing
 * - Adds segment to history
 * - Tracks visited positions
 *
 * For minimal state (ClientSandboxEngine):
 * - Simply updates currentPosition
 */
export function updateChainCapturePosition(
  state: MinimalChainCaptureState,
  newPosition: Position
): MinimalChainCaptureState {
  return {
    ...state,
    currentPosition: newPosition,
  };
}

/**
 * Update full chain capture state after a capture.
 * Delegates to CaptureAggregate's updateChainCaptureStateAfterCapture.
 */
export function updateFullChainCaptureState(
  state: ChainCaptureState | undefined,
  move: Move,
  capturedCapHeight: number
): ChainCaptureState | undefined {
  return updateChainCaptureStateAfterCapture(state, move, capturedCapHeight);
}

/**
 * Evaluate whether a chain capture must continue from the current position.
 * This is the core decision function used by both engines.
 *
 * @param gameState - Current game state after applying the capture
 * @param playerNumber - Player performing the chain
 * @param currentPosition - Landing position of the last capture
 * @returns Evaluation result with mustContinue flag and available moves
 */
export function evaluateChainCaptureContinuation(
  gameState: GameState,
  playerNumber: number,
  currentPosition: Position
): ChainCaptureEvaluation {
  const info = getChainCaptureContinuationInfo(gameState, playerNumber, currentPosition);

  return {
    mustContinue: info.mustContinue,
    availableMoves: info.availableContinuations,
    recommendedPhase: info.mustContinue ? 'chain_capture' : 'line_processing',
  };
}

/**
 * Check if the current phase is a chain capture phase.
 */
export function isChainCapturePhase(phase: GamePhase): boolean {
  return phase === 'chain_capture';
}

/**
 * Determine if a chain capture state is active.
 */
export function isChainCaptureActive(
  state: ChainCaptureState | MinimalChainCaptureState | null | undefined
): boolean {
  if (!state) return false;
  if ('isActive' in state) return state.isActive;
  // Full ChainCaptureState is always active if it exists
  return true;
}

/**
 * Get the current position from a chain capture state.
 */
export function getChainCapturePosition(
  state: ChainCaptureState | MinimalChainCaptureState | null | undefined
): Position | null {
  if (!state) return null;
  return state.currentPosition;
}

/**
 * Get the player number from a chain capture state.
 */
export function getChainCapturePlayer(
  state: ChainCaptureState | MinimalChainCaptureState | null | undefined
): number | null {
  if (!state) return null;
  return state.playerNumber;
}

/**
 * Clear/deactivate chain capture state.
 * Returns null to indicate no active chain.
 */
export function clearChainCaptureState(): null {
  return null;
}

/**
 * Process the result of a capture and determine chain state updates.
 * This is a convenience function that combines evaluation and state updates.
 *
 * @param gameState - Game state after the capture was applied
 * @param move - The capture move that was applied
 * @param capturedCapHeight - Cap height of the captured stack
 * @param existingState - Current chain capture state (or undefined for first capture)
 * @returns Object with updated state and continuation info
 */
export function processChainCaptureResult(
  gameState: GameState,
  move: Move,
  capturedCapHeight: number,
  existingState: ChainCaptureState | undefined
): {
  chainState: ChainCaptureState | undefined;
  evaluation: ChainCaptureEvaluation;
  shouldTransitionToChainPhase: boolean;
} {
  // Update the chain state
  const chainState = updateChainCaptureStateAfterCapture(existingState, move, capturedCapHeight);

  // Evaluate continuation from the landing position
  const evaluation = evaluateChainCaptureContinuation(gameState, move.player, move.to);

  // Cache available moves in the chain state if it exists
  if (chainState) {
    chainState.availableMoves = evaluation.availableMoves;
  }

  return {
    chainState: evaluation.mustContinue ? chainState : undefined,
    evaluation,
    shouldTransitionToChainPhase: evaluation.mustContinue,
  };
}

/**
 * Validate that a continue_capture_segment move is legal for the current
 * chain capture state.
 *
 * @param move - The continuation move to validate
 * @param chainState - Current chain capture state
 * @returns True if the move is a valid continuation
 */
export function validateChainCaptureContinuation(
  move: Move,
  chainState: ChainCaptureState | MinimalChainCaptureState | null | undefined
): boolean {
  if (!chainState) return false;
  if (move.type !== 'continue_capture_segment') return false;
  if (move.player !== chainState.playerNumber) return false;

  // Check that the move starts from the current position
  if (!move.from) return false;
  const currentKey = positionToString(chainState.currentPosition);
  const moveFromKey = positionToString(move.from);

  return currentKey === moveFromKey;
}

/**
 * Get valid moves during chain_capture phase.
 * This is used by getValidMoves() implementations.
 *
 * @param gameState - Current game state
 * @param chainState - Current chain capture state
 * @returns Array of valid continuation moves, or empty if chain is complete
 */
export function getChainCaptureMoves(
  gameState: GameState,
  chainState: ChainCaptureState | MinimalChainCaptureState | null | undefined
): Move[] {
  if (!chainState) return [];

  // If we have cached moves in a full state, use them
  if ('availableMoves' in chainState && chainState.availableMoves.length > 0) {
    return chainState.availableMoves;
  }

  // Otherwise, enumerate from current position
  const evaluation = evaluateChainCaptureContinuation(
    gameState,
    chainState.playerNumber,
    chainState.currentPosition
  );

  return evaluation.availableMoves;
}
