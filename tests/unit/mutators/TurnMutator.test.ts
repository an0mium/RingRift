/**
 * Unit tests for TurnMutator.ts
 *
 * Tests for mutateTurnChange and mutatePhaseChange functions.
 */

import {
  mutateTurnChange,
  mutatePhaseChange,
} from '../../../src/shared/engine/mutators/TurnMutator';
import { createTestGameState, createTestPlayer } from '../../utils/fixtures';
import type { GamePhase } from '../../../src/shared/types/game';

describe('TurnMutator', () => {
  describe('mutateTurnChange', () => {
    it('rotates to next player in 2-player game', () => {
      const state = createTestGameState({
        currentPlayer: 1,
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutateTurnChange(state);

      expect(result.currentPlayer).toBe(2);
    });

    it('wraps around from last player to first player', () => {
      const state = createTestGameState({
        currentPlayer: 2,
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutateTurnChange(state);

      expect(result.currentPlayer).toBe(1);
    });

    it('handles 3-player rotation', () => {
      const state = createTestGameState({
        currentPlayer: 2,
        players: [createTestPlayer(1), createTestPlayer(2), createTestPlayer(3)],
      });

      const result = mutateTurnChange(state);

      expect(result.currentPlayer).toBe(3);
    });

    it('resets phase to ring_placement', () => {
      const state = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutateTurnChange(state);

      expect(result.currentPhase).toBe('ring_placement');
    });

    it('updates lastMoveAt timestamp', () => {
      const oldDate = new Date('2020-01-01');
      const state = createTestGameState({
        currentPlayer: 1,
        players: [createTestPlayer(1), createTestPlayer(2)],
        lastMoveAt: oldDate,
      });

      const result = mutateTurnChange(state);

      expect(result.lastMoveAt.getTime()).toBeGreaterThan(oldDate.getTime());
    });

    it('preserves original state immutably', () => {
      const state = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutateTurnChange(state);

      expect(state.currentPlayer).toBe(1);
      expect(state.currentPhase).toBe('movement');
      expect(result).not.toBe(state);
    });

    it('creates new players array', () => {
      const state = createTestGameState({
        currentPlayer: 1,
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutateTurnChange(state);

      expect(result.players).not.toBe(state.players);
      expect(result.players[0]).not.toBe(state.players[0]);
    });

    it('creates new moveHistory array', () => {
      const state = createTestGameState({
        currentPlayer: 1,
        players: [createTestPlayer(1), createTestPlayer(2)],
        moveHistory: [{ type: 'placement' } as any],
      });

      const result = mutateTurnChange(state);

      expect(result.moveHistory).not.toBe(state.moveHistory);
      expect(result.moveHistory).toEqual(state.moveHistory);
    });
  });

  describe('mutatePhaseChange', () => {
    it('changes to movement phase', () => {
      const state = createTestGameState({
        currentPhase: 'ring_placement',
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutatePhaseChange(state, 'movement');

      expect(result.currentPhase).toBe('movement');
    });

    it('changes to line_processing phase', () => {
      const state = createTestGameState({
        currentPhase: 'movement',
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutatePhaseChange(state, 'line_processing');

      expect(result.currentPhase).toBe('line_processing');
    });

    it('changes to territory_processing phase', () => {
      const state = createTestGameState({
        currentPhase: 'line_processing',
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutatePhaseChange(state, 'territory_processing');

      expect(result.currentPhase).toBe('territory_processing');
    });

    it('updates lastMoveAt timestamp', () => {
      const oldDate = new Date('2020-01-01');
      const state = createTestGameState({
        currentPhase: 'ring_placement',
        players: [createTestPlayer(1), createTestPlayer(2)],
        lastMoveAt: oldDate,
      });

      const result = mutatePhaseChange(state, 'movement');

      expect(result.lastMoveAt.getTime()).toBeGreaterThan(oldDate.getTime());
    });

    it('preserves original state immutably', () => {
      const state = createTestGameState({
        currentPhase: 'ring_placement',
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const result = mutatePhaseChange(state, 'movement');

      expect(state.currentPhase).toBe('ring_placement');
      expect(result).not.toBe(state);
    });

    it('creates new moveHistory array', () => {
      const state = createTestGameState({
        currentPhase: 'ring_placement',
        players: [createTestPlayer(1), createTestPlayer(2)],
        moveHistory: [{ type: 'placement' } as any],
      });

      const result = mutatePhaseChange(state, 'movement');

      expect(result.moveHistory).not.toBe(state.moveHistory);
    });
  });
});
