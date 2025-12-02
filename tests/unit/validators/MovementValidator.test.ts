/**
 * MovementValidator Unit Tests
 *
 * Tests the shared engine movement validation logic including:
 * - Phase and turn validation
 * - Stack ownership verification
 * - Direction validation
 * - Minimum distance constraints
 * - Path obstruction checks
 * - Landing position validation
 */

import { validateMovement } from '../../../src/shared/engine/validators/MovementValidator';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  pos,
  posStr,
} from '../../utils/fixtures';
import { BoardState, RingStack, MarkerInfo } from '../../../src/shared/types/game';
import { MoveStackAction, GameState } from '../../../src/shared/engine/types';

describe('MovementValidator', () => {
  let state: GameState;

  beforeEach(() => {
    state = createTestGameState({
      currentPhase: 'movement',
      currentPlayer: 1,
      players: [createTestPlayer(1, { ringsInHand: 10 }), createTestPlayer(2, { ringsInHand: 10 })],
    });

    // Add a stack for player 1 at (3, 3)
    const stack: RingStack = {
      position: pos(3, 3),
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    };
    state.board.stacks.set(posStr(3, 3), stack);
  });

  describe('phase and turn validation', () => {
    it('allows movement in movement phase', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3), // Move 2 spaces (stack height = 2)
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects movement in wrong phase', () => {
      state.currentPhase = 'ring_placement';
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects movement when not player turn', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 2,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });
  });

  describe('position validation', () => {
    it('rejects movement from off-board position', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(-1, 3),
        to: pos(1, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects movement to off-board position', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(10, 3), // Off board for square8
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });
  });

  describe('stack ownership', () => {
    it('rejects movement from empty position', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0), // No stack here
        to: pos(2, 0),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_STACK');
    });

    it('rejects movement of opponent stack', () => {
      // Add opponent stack
      const opponentStack: RingStack = {
        position: pos(5, 5),
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(5, 5), opponentStack);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(5, 5),
        to: pos(7, 5),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_STACK');
    });
  });

  describe('collapsed space checks', () => {
    it('rejects movement to collapsed space', () => {
      state.board.collapsedSpaces.set(posStr(5, 3), 1);
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('COLLAPSED_SPACE');
    });
  });

  describe('direction validation', () => {
    it('allows horizontal movement', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows vertical movement', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(3, 5),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows diagonal movement', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 5),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects non-aligned movement', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 4), // Not aligned on any axis
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_DIRECTION');
    });
  });

  describe('minimum distance constraint', () => {
    it('allows movement equal to stack height', () => {
      // Stack height is 2, move distance is 2
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows movement greater than stack height', () => {
      // Stack height is 2, move distance is 4
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(7, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects movement less than stack height', () => {
      // Stack height is 2, move distance is 1
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(4, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INSUFFICIENT_DISTANCE');
    });
  });

  describe('path obstruction', () => {
    it('allows movement over empty path', () => {
      // Move from (3,3) to (6,3), path is clear
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects movement through collapsed space', () => {
      state.board.collapsedSpaces.set(posStr(4, 3), 1);
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('PATH_BLOCKED');
    });

    it('rejects movement through other stack', () => {
      const blockingStack: RingStack = {
        position: pos(4, 3),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(4, 3), blockingStack);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(6, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('PATH_BLOCKED');
    });
  });

  describe('landing position validation', () => {
    it('allows landing on empty space', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows landing on own marker', () => {
      const marker: MarkerInfo = { player: 1, position: pos(5, 3), type: 'regular' };
      state.board.markers.set(posStr(5, 3), marker);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects landing on opponent marker', () => {
      const marker: MarkerInfo = { player: 2, position: pos(5, 3), type: 'regular' };
      state.board.markers.set(posStr(5, 3), marker);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LANDING');
    });

    it('rejects landing on existing stack', () => {
      const landingStack: RingStack = {
        position: pos(5, 3),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(5, 3), landingStack);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_LANDING');
    });
  });

  describe('hexagonal board movement', () => {
    beforeEach(() => {
      state = createTestGameState({
        boardType: 'hexagonal',
        currentPhase: 'movement',
        currentPlayer: 1,
        players: [
          createTestPlayer(1, { ringsInHand: 36 }),
          createTestPlayer(2, { ringsInHand: 36 }),
        ],
      });
      state.board = createTestBoard('hexagonal');

      // Add a stack at hex center
      const stack: RingStack = {
        position: pos(0, 0, 0),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      };
      state.board.stacks.set(posStr(0, 0, 0), stack);
    });

    it('allows movement along hex axis', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0, 0),
        to: pos(2, -2, 0), // Along x-y axis
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects off-axis movement on hex board', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0, 0),
        to: pos(2, -1, -1), // Not aligned on any hex axis
      };
      const result = validateMovement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_DIRECTION');
    });
  });
});
