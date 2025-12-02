/**
 * PlacementMutator Unit Tests
 *
 * Tests the shared engine placement mutation logic including:
 * - Ring placement on empty cells
 * - Ring placement on existing stacks
 * - Marker clearing on placement
 * - Player ringsInHand updates
 * - CapHeight recalculation
 */

import {
  applyPlacementOnBoard,
  mutatePlacement,
} from '../../../src/shared/engine/mutators/PlacementMutator';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  pos,
  posStr,
} from '../../utils/fixtures';
import { BoardState, RingStack, MarkerInfo } from '../../../src/shared/types/game';
import { PlaceRingAction, GameState } from '../../../src/shared/engine/types';

describe('PlacementMutator', () => {
  describe('applyPlacementOnBoard', () => {
    let board: BoardState;

    beforeEach(() => {
      board = createTestBoard('square8');
    });

    describe('empty cell placement', () => {
      it('creates a new stack with single ring', () => {
        const result = applyPlacementOnBoard(board, pos(3, 3), 1, 1);
        const stack = result.stacks.get(posStr(3, 3));

        expect(stack).toBeDefined();
        expect(stack!.rings).toEqual([1]);
        expect(stack!.stackHeight).toBe(1);
        expect(stack!.capHeight).toBe(1);
        expect(stack!.controllingPlayer).toBe(1);
        expect(stack!.position).toEqual(pos(3, 3));
      });

      it('creates a new stack with multiple rings', () => {
        const result = applyPlacementOnBoard(board, pos(3, 3), 2, 3);
        const stack = result.stacks.get(posStr(3, 3));

        expect(stack).toBeDefined();
        expect(stack!.rings).toEqual([2, 2, 2]);
        expect(stack!.stackHeight).toBe(3);
        expect(stack!.capHeight).toBe(3);
        expect(stack!.controllingPlayer).toBe(2);
      });

      it('does not mutate the original board', () => {
        const result = applyPlacementOnBoard(board, pos(3, 3), 1, 1);

        expect(board.stacks.has(posStr(3, 3))).toBe(false);
        expect(result.stacks.has(posStr(3, 3))).toBe(true);
      });
    });

    describe('existing stack placement', () => {
      beforeEach(() => {
        const existingStack: RingStack = {
          position: pos(3, 3),
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2,
        };
        board.stacks.set(posStr(3, 3), existingStack);
      });

      it('adds rings on top of existing stack', () => {
        const result = applyPlacementOnBoard(board, pos(3, 3), 1, 1);
        const stack = result.stacks.get(posStr(3, 3));

        expect(stack).toBeDefined();
        expect(stack!.rings).toEqual([1, 2]); // New ring on top
        expect(stack!.stackHeight).toBe(2);
        expect(stack!.controllingPlayer).toBe(1); // Controller changes to top ring owner
      });

      it('recalculates capHeight based on new ring sequence', () => {
        // Add 2 rings of player 1 on top of player 2's stack
        const result = applyPlacementOnBoard(board, pos(3, 3), 1, 1);
        const stack = result.stacks.get(posStr(3, 3));

        // Rings are [1, 2], capHeight is 1 (only consecutive same-color from top)
        expect(stack!.capHeight).toBe(1);

        // Now add another ring of player 1
        const result2 = applyPlacementOnBoard(result, pos(3, 3), 1, 1);
        const stack2 = result2.stacks.get(posStr(3, 3));

        // Rings are [1, 1, 2], capHeight is 2
        expect(stack2!.rings).toEqual([1, 1, 2]);
        expect(stack2!.capHeight).toBe(2);
      });
    });

    describe('marker clearing', () => {
      it('removes marker when placing on its position', () => {
        const marker: MarkerInfo = { player: 1, position: pos(3, 3), type: 'regular' };
        board.markers.set(posStr(3, 3), marker);

        const result = applyPlacementOnBoard(board, pos(3, 3), 1, 1);

        expect(result.markers.has(posStr(3, 3))).toBe(false);
        expect(result.stacks.has(posStr(3, 3))).toBe(true);
      });
    });

    describe('edge cases', () => {
      it('handles count of 0 by placing at least 1 ring', () => {
        const result = applyPlacementOnBoard(board, pos(3, 3), 1, 0);
        const stack = result.stacks.get(posStr(3, 3));

        expect(stack).toBeDefined();
        expect(stack!.stackHeight).toBe(1);
      });

      it('handles negative count by placing at least 1 ring', () => {
        const result = applyPlacementOnBoard(board, pos(3, 3), 1, -5);
        const stack = result.stacks.get(posStr(3, 3));

        expect(stack).toBeDefined();
        expect(stack!.stackHeight).toBe(1);
      });
    });
  });

  describe('mutatePlacement (GameState wrapper)', () => {
    let state: GameState;

    beforeEach(() => {
      state = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        players: [
          createTestPlayer(1, { ringsInHand: 18 }),
          createTestPlayer(2, { ringsInHand: 18 }),
        ],
      });
    });

    it('decrements player ringsInHand', () => {
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 2,
      };

      const result = mutatePlacement(state, action);
      const player = result.players.find((p) => p.playerNumber === 1);

      expect(player!.ringsInHand).toBe(16); // 18 - 2
    });

    it('places rings on board', () => {
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 3,
      };

      const result = mutatePlacement(state, action);
      const stack = result.board.stacks.get(posStr(3, 3));

      expect(stack).toBeDefined();
      expect(stack!.rings).toEqual([1, 1, 1]);
      expect(stack!.controllingPlayer).toBe(1);
    });

    it('updates lastMoveAt timestamp', () => {
      const originalTime = state.lastMoveAt;
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 1,
      };

      const result = mutatePlacement(state, action);

      expect(result.lastMoveAt.getTime()).toBeGreaterThanOrEqual(originalTime.getTime());
    });

    it('does not mutate original state', () => {
      const originalRingsInHand = state.players[0].ringsInHand;
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 1,
      };

      mutatePlacement(state, action);

      expect(state.players[0].ringsInHand).toBe(originalRingsInHand);
      expect(state.board.stacks.has(posStr(3, 3))).toBe(false);
    });

    it('throws error for unknown player', () => {
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 99,
        position: pos(3, 3),
        count: 1,
      };

      expect(() => mutatePlacement(state, action)).toThrow('Player not found');
    });

    it('returns state unchanged when player has no rings', () => {
      state.players[0].ringsInHand = 0;
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 1,
      };

      const result = mutatePlacement(state, action);

      expect(result.board.stacks.has(posStr(3, 3))).toBe(false);
      expect(result.players[0].ringsInHand).toBe(0);
    });

    it('clamps placement to available rings', () => {
      state.players[0].ringsInHand = 2;
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 5, // More than available
      };

      const result = mutatePlacement(state, action);
      const stack = result.board.stacks.get(posStr(3, 3));

      expect(stack!.stackHeight).toBe(2); // Only 2 placed
      expect(result.players[0].ringsInHand).toBe(0);
    });
  });
});
