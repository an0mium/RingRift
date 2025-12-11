/**
 * Tests for sharedDecisionHelpers.ts
 *
 * Verifies the shared utility functions used by lineDecisionHelpers
 * and territoryDecisionHelpers work correctly.
 */

import { computeNextMoveNumber } from '../../src/shared/engine/sharedDecisionHelpers';
import type { GameState } from '../../src/shared/types/game';

describe('sharedDecisionHelpers', () => {
  describe('computeNextMoveNumber', () => {
    it('returns 1 when state has no history', () => {
      const state = {
        history: [],
        moveHistory: [],
      } as unknown as GameState;

      expect(computeNextMoveNumber(state)).toBe(1);
    });

    it('returns next number from history when available', () => {
      const state = {
        history: [{ moveNumber: 5, type: 'movement', player: 1 }],
        moveHistory: [],
      } as unknown as GameState;

      expect(computeNextMoveNumber(state)).toBe(6);
    });

    it('returns next number from moveHistory as fallback', () => {
      const state = {
        history: [],
        moveHistory: [{ moveNumber: 10, type: 'capture', player: 2 }],
      } as unknown as GameState;

      expect(computeNextMoveNumber(state)).toBe(11);
    });

    it('prefers history over moveHistory when both exist', () => {
      const state = {
        history: [{ moveNumber: 3, type: 'movement', player: 1 }],
        moveHistory: [{ moveNumber: 7, type: 'capture', player: 2 }],
      } as unknown as GameState;

      expect(computeNextMoveNumber(state)).toBe(4);
    });

    it('handles history with invalid moveNumber gracefully', () => {
      const state = {
        history: [
          { moveNumber: 0, type: 'movement', player: 1 }, // Invalid: 0
        ],
        moveHistory: [{ moveNumber: 5, type: 'capture', player: 2 }],
      } as unknown as GameState;

      // Falls back to moveHistory
      expect(computeNextMoveNumber(state)).toBe(6);
    });

    it('handles undefined history arrays', () => {
      const state = {} as GameState;

      expect(computeNextMoveNumber(state)).toBe(1);
    });

    it('handles history entries without moveNumber', () => {
      const state = {
        history: [
          { type: 'movement', player: 1 }, // No moveNumber
        ],
        moveHistory: [],
      } as unknown as GameState;

      expect(computeNextMoveNumber(state)).toBe(1);
    });
  });
});
