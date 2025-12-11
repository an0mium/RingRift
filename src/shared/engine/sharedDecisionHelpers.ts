/**
 * Shared Decision Helpers - Common utilities for decision enumeration modules
 *
 * @module sharedDecisionHelpers
 *
 * This module consolidates shared utilities used by multiple decision helper modules:
 * - lineDecisionHelpers.ts
 * - territoryDecisionHelpers.ts
 *
 * By centralizing these utilities, we:
 * 1. Eliminate code duplication (DRY principle)
 * 2. Ensure consistent move numbering across all decision types
 * 3. Simplify future maintenance
 *
 * Related modules:
 * - {@link lineDecisionHelpers} - Line-processing decision enumeration
 * - {@link territoryDecisionHelpers} - Territory-processing decision enumeration
 */

import type { GameState } from '../types/game';

/**
 * Compute the next canonical moveNumber for decision moves based on the
 * existing history/moveHistory. This keeps numbering stable across hosts
 * without requiring callers to thread an explicit counter.
 *
 * The function checks both `state.history` (canonical) and `state.moveHistory`
 * (legacy) arrays to determine the appropriate next move number.
 *
 * @param state - The current game state
 * @returns The next sequential move number (minimum 1)
 *
 * @example
 * ```typescript
 * const nextNum = computeNextMoveNumber(state);
 * const move = { type: 'process_line', moveNumber: nextNum, ... };
 * ```
 */
export function computeNextMoveNumber(state: GameState): number {
  if (state.history && state.history.length > 0) {
    const last = state.history[state.history.length - 1];
    if (typeof last.moveNumber === 'number' && last.moveNumber > 0) {
      return last.moveNumber + 1;
    }
  }

  if (state.moveHistory && state.moveHistory.length > 0) {
    const lastLegacy = state.moveHistory[state.moveHistory.length - 1];
    if (typeof lastLegacy.moveNumber === 'number' && lastLegacy.moveNumber > 0) {
      return lastLegacy.moveNumber + 1;
    }
  }

  return 1;
}
