/**
 * Shared history entry creation helpers for RingRift engine.
 *
 * These functions are pure and can be used by both server-side GameEngine
 * and client-side ClientSandboxEngine to create consistent GameHistoryEntry
 * records. Hosts may wrap these with additional debug/trace logic.
 *
 * @module historyHelpers
 */

import type { GameState, Move, GameHistoryEntry, ProgressSnapshot } from '../types/game';
import { computeProgressSnapshot, summarizeBoard, hashGameState } from './core';

/**
 * Options for creating a history entry.
 */
export interface CreateHistoryEntryOptions {
  /**
   * When true, normalize moveNumber to the current history length + 1.
   * This ensures contiguous 1..N sequence in the history.
   * Default: false (use action.moveNumber as-is)
   */
  normalizeMoveNumber?: boolean;

  /**
   * Custom progress snapshot for the "before" state.
   * If not provided, computes from the before state.
   * This allows hosts to provide pre-computed or adjusted snapshots.
   */
  progressBefore?: ProgressSnapshot;

  /**
   * Custom progress snapshot for the "after" state.
   * If not provided, computes from the after state.
   */
  progressAfter?: ProgressSnapshot;
}

/**
 * Create a GameHistoryEntry for a move applied between two game states.
 *
 * This function creates the core entry structure used by both server and client
 * engines. It computes progress snapshots, board summaries, and state hashes
 * from the provided before/after states.
 *
 * @param before - Game state before the move was applied
 * @param after - Game state after the move was applied
 * @param action - The move that was applied
 * @param options - Optional configuration for entry creation
 * @returns A complete GameHistoryEntry
 *
 * @example
 * ```typescript
 * const entry = createHistoryEntry(beforeState, afterState, move);
 * gameState.history.push(entry);
 * ```
 */
export function createHistoryEntry(
  before: GameState,
  after: GameState,
  action: Move,
  options: CreateHistoryEntryOptions = {}
): GameHistoryEntry {
  const { normalizeMoveNumber = false, progressBefore, progressAfter } = options;

  // Compute progress snapshots if not provided
  const computedProgressBefore = progressBefore ?? computeProgressSnapshot(before);
  const computedProgressAfter = progressAfter ?? computeProgressSnapshot(after);

  // Compute board summaries
  const boardBeforeSummary = summarizeBoard(before.board);
  const boardAfterSummary = summarizeBoard(after.board);

  // Determine move number
  const moveNumber = normalizeMoveNumber ? before.history.length + 1 : action.moveNumber;

  // Create normalized action if moveNumber changed
  const normalizedAction: Move = normalizeMoveNumber ? { ...action, moveNumber } : action;

  return {
    moveNumber,
    action: normalizedAction,
    actor: normalizedAction.player,
    phaseBefore: before.currentPhase,
    phaseAfter: after.currentPhase,
    statusBefore: before.gameStatus,
    statusAfter: after.gameStatus,
    progressBefore: computedProgressBefore,
    progressAfter: computedProgressAfter,
    stateHashBefore: hashGameState(before),
    stateHashAfter: hashGameState(after),
    boardBeforeSummary,
    boardAfterSummary,
  };
}

/**
 * Create a normalized progress snapshot from a board summary.
 *
 * This creates a progress snapshot where the S-invariant components
 * (markers, collapsed, eliminated) are derived from the board summary
 * to ensure consistency between progress tracking and board geometry.
 *
 * @param boardSummary - Board summary to derive progress from
 * @param eliminatedRings - Total eliminated rings count
 * @returns A ProgressSnapshot with consistent S-invariant
 */
export function createProgressFromBoardSummary(
  boardSummary: ReturnType<typeof summarizeBoard>,
  eliminatedRings: number
): ProgressSnapshot {
  return {
    markers: boardSummary.markers.length,
    collapsed: boardSummary.collapsedSpaces.length,
    eliminated: eliminatedRings,
    S: boardSummary.markers.length + boardSummary.collapsedSpaces.length + eliminatedRings,
  };
}

/**
 * Append a history entry to a game state immutably.
 *
 * @param state - Current game state
 * @param entry - History entry to append
 * @returns New game state with the entry appended
 */
export function appendHistoryEntryToState(state: GameState, entry: GameHistoryEntry): GameState {
  return {
    ...state,
    history: [...state.history, entry],
  };
}
