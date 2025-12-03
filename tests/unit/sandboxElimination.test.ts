/**
 * Unit tests for sandboxElimination.ts
 *
 * Tests for forceEliminateCapOnBoard function, focusing on edge cases
 * and branch coverage for forced cap elimination scenarios.
 */

import { forceEliminateCapOnBoard } from '../../src/client/sandbox/sandboxElimination';
import { createTestBoard, createTestPlayer, pos, posStr } from '../utils/fixtures';
import type { BoardState, Player, RingStack } from '../../src/shared/types/game';

describe('sandboxElimination', () => {
  describe('forceEliminateCapOnBoard', () => {
    it('returns unchanged state when player not found', () => {
      const board = createTestBoard('square8');
      const players = [createTestPlayer(1), createTestPlayer(2)];

      const stack: RingStack = {
        position: pos(0, 0),
        rings: [3, 3, 3],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 3,
      };

      // Player 99 doesn't exist
      const result = forceEliminateCapOnBoard(board, players, 99, [stack]);

      expect(result.totalRingsEliminatedDelta).toBe(0);
      expect(result.board).toBe(board);
      expect(result.players).toBe(players);
    });

    it('returns unchanged state when stacks array is empty', () => {
      const board = createTestBoard('square8');
      const players = [createTestPlayer(1), createTestPlayer(2)];

      const result = forceEliminateCapOnBoard(board, players, 1, []);

      expect(result.totalRingsEliminatedDelta).toBe(0);
      expect(result.board).toBe(board);
      expect(result.players).toBe(players);
    });

    it('returns unchanged state when stack has no cap height (empty rings)', () => {
      const board = createTestBoard('square8');
      const players = [createTestPlayer(1), createTestPlayer(2)];

      // Stack with empty rings - calculateCapHeight returns 0
      const stack: RingStack = {
        position: pos(0, 0),
        rings: [],
        stackHeight: 0,
        capHeight: 0,
        controllingPlayer: 1,
      };

      const result = forceEliminateCapOnBoard(board, players, 1, [stack]);

      expect(result.totalRingsEliminatedDelta).toBe(0);
    });

    it('eliminates cap and updates board eliminatedRings', () => {
      const board = createTestBoard('square8');
      board.eliminatedRings = { 1: 5, 2: 0 };
      const players = [
        createTestPlayer(1, { eliminatedRings: 5 }),
        createTestPlayer(2, { eliminatedRings: 0 }),
      ];

      const stack: RingStack = {
        position: pos(2, 3),
        rings: [1, 1, 1, 2, 2], // Cap of 3 player 1 rings
        stackHeight: 5,
        capHeight: 3,
        controllingPlayer: 1,
      };
      board.stacks.set(posStr(2, 3), stack);

      const result = forceEliminateCapOnBoard(board, players, 1, [stack]);

      expect(result.totalRingsEliminatedDelta).toBe(3);
      expect(result.board.eliminatedRings[1]).toBe(8); // 5 + 3
      expect(result.players.find((p) => p.playerNumber === 1)?.eliminatedRings).toBe(8);
    });

    it('removes stack from board when all rings are eliminated', () => {
      const board = createTestBoard('square8');
      const players = [createTestPlayer(1), createTestPlayer(2)];

      // Stack where all rings belong to same player (full cap)
      const stack: RingStack = {
        position: pos(4, 4),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      };
      board.stacks.set(posStr(4, 4), stack);

      const result = forceEliminateCapOnBoard(board, players, 1, [stack]);

      expect(result.totalRingsEliminatedDelta).toBe(3);
      expect(result.board.stacks.has(posStr(4, 4))).toBe(false);
    });

    it('updates stack when remaining rings exist after cap elimination', () => {
      const board = createTestBoard('square8');
      const players = [createTestPlayer(1), createTestPlayer(2)];

      // Stack with cap of 2 player 1 rings, then 3 player 2 rings
      const stack: RingStack = {
        position: pos(1, 1),
        rings: [1, 1, 2, 2, 2],
        stackHeight: 5,
        capHeight: 2,
        controllingPlayer: 1,
      };
      board.stacks.set(posStr(1, 1), stack);

      const result = forceEliminateCapOnBoard(board, players, 1, [stack]);

      expect(result.totalRingsEliminatedDelta).toBe(2);
      expect(result.board.stacks.has(posStr(1, 1))).toBe(true);

      const updatedStack = result.board.stacks.get(posStr(1, 1))!;
      expect(updatedStack.rings).toEqual([2, 2, 2]);
      expect(updatedStack.stackHeight).toBe(3);
      expect(updatedStack.controllingPlayer).toBe(2);
    });

    it('prefers stack with positive capHeight when multiple stacks provided', () => {
      const board = createTestBoard('square8');
      const players = [createTestPlayer(1), createTestPlayer(2)];

      // First stack has no cap
      const stackNoCap: RingStack = {
        position: pos(0, 0),
        rings: [1, 2, 1],
        stackHeight: 3,
        capHeight: 0,
        controllingPlayer: 1,
      };

      // Second stack has cap
      const stackWithCap: RingStack = {
        position: pos(1, 1),
        rings: [1, 1, 2],
        stackHeight: 3,
        capHeight: 2,
        controllingPlayer: 1,
      };

      board.stacks.set(posStr(0, 0), stackNoCap);
      board.stacks.set(posStr(1, 1), stackWithCap);

      const result = forceEliminateCapOnBoard(board, players, 1, [stackNoCap, stackWithCap]);

      expect(result.totalRingsEliminatedDelta).toBe(2);
      // Stack at 1,1 should be updated (remaining: [2])
      expect(result.board.stacks.get(posStr(1, 1))?.rings).toEqual([2]);
    });

    it('falls back to first stack when no stacks have positive capHeight', () => {
      const board = createTestBoard('square8');
      const players = [createTestPlayer(1), createTestPlayer(2)];

      // All stacks have capHeight of 0 in RingStack but rings actually form a cap
      // (simulates mismatched capHeight field - calculateCapHeight recalculates)
      const stack1: RingStack = {
        position: pos(0, 0),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 0, // Wrong value, calculateCapHeight will find 3
        controllingPlayer: 1,
      };

      const stack2: RingStack = {
        position: pos(1, 1),
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 0,
        controllingPlayer: 2,
      };

      board.stacks.set(posStr(0, 0), stack1);
      board.stacks.set(posStr(1, 1), stack2);

      // Both stacks have capHeight 0, so it uses first one (stack1)
      // calculateCapHeight([1,1,1]) = 3
      const result = forceEliminateCapOnBoard(board, players, 1, [stack1, stack2]);

      expect(result.totalRingsEliminatedDelta).toBe(3);
      expect(result.board.stacks.has(posStr(0, 0))).toBe(false);
    });

    it('initializes eliminatedRings for player if not present', () => {
      const board = createTestBoard('square8');
      board.eliminatedRings = {}; // No entries yet
      const players = [createTestPlayer(1, { eliminatedRings: 0 }), createTestPlayer(2)];

      const stack: RingStack = {
        position: pos(3, 3),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      };
      board.stacks.set(posStr(3, 3), stack);

      const result = forceEliminateCapOnBoard(board, players, 1, [stack]);

      expect(result.totalRingsEliminatedDelta).toBe(2);
      expect(result.board.eliminatedRings[1]).toBe(2);
    });

    it('correctly copies board Maps and formedLines array', () => {
      const board = createTestBoard('square8');
      board.markers.set('5,5', { position: pos(5, 5), player: 2, type: 'regular' });
      board.collapsedSpaces.set('6,6', { position: pos(6, 6), collapsed: true });
      board.territories.set('7,7', {
        position: pos(7, 7),
        owner: 2,
        type: 'home',
        convertedAt: 1,
      });
      board.formedLines = [{ cells: [pos(0, 0), pos(0, 1)], player: 2, processedAt: 1 }];

      const players = [createTestPlayer(1), createTestPlayer(2)];

      const stack: RingStack = {
        position: pos(2, 2),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      };
      board.stacks.set(posStr(2, 2), stack);

      const result = forceEliminateCapOnBoard(board, players, 1, [stack]);

      // Original board should be unchanged
      expect(board.stacks.has(posStr(2, 2))).toBe(true);

      // Result board should have copied structures
      expect(result.board.markers.get('5,5')).toBeDefined();
      expect(result.board.collapsedSpaces.get('6,6')).toBeDefined();
      expect(result.board.territories.get('7,7')).toBeDefined();
      expect(result.board.formedLines).toHaveLength(1);

      // Result board should have eliminated the stack
      expect(result.board.stacks.has(posStr(2, 2))).toBe(false);
    });
  });
});
