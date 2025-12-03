/**
 * Unit tests for sandboxVictory.ts
 *
 * Tests for checkSandboxVictory function, focusing on edge cases
 * and branch coverage for victory detection scenarios.
 */

import { checkSandboxVictory } from '../../src/client/sandbox/sandboxVictory';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addMarker,
} from '../utils/fixtures';
import type { BoardType } from '../../src/shared/types/game';

describe('sandboxVictory', () => {
  const boardType: BoardType = 'square8';

  describe('checkSandboxVictory', () => {
    it('returns null when game is not over (active stacks on board)', () => {
      const board = createTestBoard(boardType);
      // Add a stack so the game is not a bare board
      board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        controllingPlayer: 1,
        stackHeight: 3,
        capHeight: 1,
        rings: [1, 1, 1],
      });

      const state = createTestGameState({
        boardType,
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 15, eliminatedRings: 0 }),
          createTestPlayer(2, { ringsInHand: 18, eliminatedRings: 0 }),
        ],
        victoryThreshold: 18,
        territoryVictoryThreshold: 64,
      });

      const result = checkSandboxVictory(state);
      expect(result).toBeNull();
    });

    it('returns ring_elimination reason when victory threshold reached', () => {
      const board = createTestBoard(boardType);

      const state = createTestGameState({
        boardType,
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 10, eliminatedRings: 18 }),
          createTestPlayer(2, { ringsInHand: 8, eliminatedRings: 0 }),
        ],
        victoryThreshold: 18,
      });

      const result = checkSandboxVictory(state);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('ring_elimination');
    });

    it('returns territory_control reason when territory threshold reached', () => {
      const board = createTestBoard(boardType);

      const state = createTestGameState({
        boardType,
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 10, territorySpaces: 33 }),
          createTestPlayer(2, { ringsInHand: 10, territorySpaces: 5 }),
        ],
        territoryVictoryThreshold: 33,
        victoryThreshold: 100,
      });

      const result = checkSandboxVictory(state);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('territory_control');
    });

    it('returns last_player_standing (via markers) on bare board with equal territory and eliminations', () => {
      // Bare board scenario (no stacks) where marker count is the tie-breaker
      const board = createTestBoard(boardType);
      board.stacks.clear();

      // Add markers - player 1 has more markers
      addMarker(board, { x: 0, y: 0 }, 1);
      addMarker(board, { x: 1, y: 1 }, 1);
      addMarker(board, { x: 2, y: 2 }, 2);

      const state = createTestGameState({
        boardType,
        board,
        players: [
          // No rings in hand, no territory, no eliminations - only markers differ
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 0, territorySpaces: 0 }),
          createTestPlayer(2, { ringsInHand: 0, eliminatedRings: 0, territorySpaces: 0 }),
        ],
        victoryThreshold: 100,
        territoryVictoryThreshold: 100,
      });

      const result = checkSandboxVictory(state);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('last_player_standing');
    });

    it('returns last_player_standing (via last actor) on bare board with all tie-breakers equal', () => {
      // Bare board with no markers, no territory, no eliminations - last actor wins
      const board = createTestBoard(boardType);
      board.stacks.clear();
      board.markers.clear();

      const state = createTestGameState({
        boardType,
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 0, territorySpaces: 0 }),
          createTestPlayer(2, { ringsInHand: 0, eliminatedRings: 0, territorySpaces: 0 }),
        ],
        currentPlayer: 1, // Player 1's turn means player 2 was last actor
        victoryThreshold: 100,
        territoryVictoryThreshold: 100,
      });

      const result = checkSandboxVictory(state);

      expect(result).not.toBeNull();
      // Last actor is the player before currentPlayer (player 2)
      expect(result!.winner).toBe(2);
      expect(result!.reason).toBe('last_player_standing');
    });

    it('returns territory_control via stalemate ladder when bare board has territory leader', () => {
      // Bare board where territory is the first tie-breaker
      const board = createTestBoard(boardType);
      board.stacks.clear();

      const state = createTestGameState({
        boardType,
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 0, territorySpaces: 5 }),
          createTestPlayer(2, { ringsInHand: 0, eliminatedRings: 0, territorySpaces: 3 }),
        ],
        victoryThreshold: 100,
        territoryVictoryThreshold: 100,
      });

      const result = checkSandboxVictory(state);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('territory_control');
    });

    it('returns ring_elimination via stalemate ladder when bare board has elimination leader', () => {
      // Bare board where territory is tied but eliminations differ
      const board = createTestBoard(boardType);
      board.stacks.clear();

      const state = createTestGameState({
        boardType,
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 5, territorySpaces: 3 }),
          createTestPlayer(2, { ringsInHand: 0, eliminatedRings: 10, territorySpaces: 3 }),
        ],
        victoryThreshold: 100,
        territoryVictoryThreshold: 100,
      });

      const result = checkSandboxVictory(state);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(2);
      expect(result!.reason).toBe('ring_elimination');
    });

    it('includes final score stats in result', () => {
      const board = createTestBoard(boardType);

      const state = createTestGameState({
        boardType,
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 5, eliminatedRings: 18, territorySpaces: 10 }),
          createTestPlayer(2, { ringsInHand: 13, eliminatedRings: 0, territorySpaces: 5 }),
        ],
        victoryThreshold: 18,
      });

      const result = checkSandboxVictory(state);

      expect(result).not.toBeNull();
      expect(result!.finalScore).toBeDefined();
      expect(result!.finalScore.ringsEliminated[1]).toBe(18);
      expect(result!.finalScore.ringsEliminated[2]).toBe(0);
      expect(result!.finalScore.territorySpaces[1]).toBe(10);
      expect(result!.finalScore.territorySpaces[2]).toBe(5);
    });

    it('returns null when players array is empty', () => {
      const board = createTestBoard(boardType);

      const state = createTestGameState({
        boardType,
        board,
        players: [],
        victoryThreshold: 18,
        territoryVictoryThreshold: 64,
      });

      const result = checkSandboxVictory(state);
      expect(result).toBeNull();
    });
  });
});
