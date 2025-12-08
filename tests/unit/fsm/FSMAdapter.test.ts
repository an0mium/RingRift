/**
 * FSMAdapter integration tests
 *
 * Tests the bidirectional conversion between Move types and FSM events,
 * as well as deriving FSM state from game state.
 */

import {
  moveToEvent,
  eventToMove,
  deriveStateFromGame,
  deriveGameContext,
  validateEvent,
  getValidEvents,
  validateMoveWithFSM,
  isMoveTypeValidForPhase,
} from '../../../src/shared/engine/fsm/FSMAdapter';
import type { Move, GameState } from '../../../src/shared/types/game';
import { createInitialGameState } from '../../../src/shared/engine/initialState';

describe('FSMAdapter', () => {
  describe('moveToEvent', () => {
    it('should convert place_ring to PLACE_RING event', () => {
      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'PLACE_RING', to: { x: 3, y: 3 } });
    });

    it('should convert skip_placement to SKIP_PLACEMENT event', () => {
      const move: Move = {
        id: 'test-2',
        type: 'skip_placement',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'SKIP_PLACEMENT' });
    });

    it('should convert move_stack to MOVE_STACK event', () => {
      const move: Move = {
        id: 'test-3',
        type: 'move_stack',
        player: 1,
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({
        type: 'MOVE_STACK',
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
      });
    });

    it('should return null for move_stack without from position', () => {
      const move: Move = {
        id: 'test-4',
        type: 'move_stack',
        player: 1,
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };

      const event = moveToEvent(move);

      expect(event).toBeNull();
    });

    it('should convert overtaking_capture to CAPTURE event', () => {
      const move: Move = {
        id: 'test-5',
        type: 'overtaking_capture',
        player: 1,
        from: { x: 2, y: 2 },
        captureTarget: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 3,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'CAPTURE', target: { x: 3, y: 3 } });
    });

    it('should convert skip_capture to END_CHAIN event', () => {
      const move: Move = {
        id: 'test-6',
        type: 'skip_capture',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 4,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'END_CHAIN' });
    });

    it('should convert forced_elimination to FORCED_ELIMINATE event', () => {
      const move: Move = {
        id: 'test-7',
        type: 'forced_elimination',
        player: 1,
        to: { x: 5, y: 5 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 5,
      };

      const event = moveToEvent(move);

      expect(event).toEqual({ type: 'FORCED_ELIMINATE', target: { x: 5, y: 5 } });
    });

    it('should return null for swap_sides (meta-move)', () => {
      const move: Move = {
        id: 'test-8',
        type: 'swap_sides',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const event = moveToEvent(move);

      expect(event).toBeNull();
    });
  });

  describe('eventToMove', () => {
    it('should convert PLACE_RING event to place_ring move', () => {
      const event = { type: 'PLACE_RING' as const, to: { x: 3, y: 3 } };

      const move = eventToMove(event, 1, 1);

      expect(move).not.toBeNull();
      expect(move!.type).toBe('place_ring');
      expect(move!.to).toEqual({ x: 3, y: 3 });
      expect(move!.player).toBe(1);
    });

    it('should convert MOVE_STACK event to move_stack move', () => {
      const event = { type: 'MOVE_STACK' as const, from: { x: 2, y: 2 }, to: { x: 4, y: 4 } };

      const move = eventToMove(event, 1, 2);

      expect(move).not.toBeNull();
      expect(move!.type).toBe('move_stack');
      expect(move!.from).toEqual({ x: 2, y: 2 });
      expect(move!.to).toEqual({ x: 4, y: 4 });
    });

    it('should return null for RESIGN event', () => {
      const event = { type: 'RESIGN' as const, player: 1 };

      const move = eventToMove(event, 1, 5);

      expect(move).toBeNull();
    });

    it('should return null for _ADVANCE_TURN event', () => {
      const event = { type: '_ADVANCE_TURN' as const };

      const move = eventToMove(event, 1, 5);

      expect(move).toBeNull();
    });
  });

  describe('deriveGameContext', () => {
    it('should derive correct context for square8 2-player game', () => {
      const players = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human' as const,
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'ai' as const,
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const state = createInitialGameState('test-game', 'square8', players, {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      });

      const context = deriveGameContext(state);

      expect(context.boardType).toBe('square8');
      expect(context.numPlayers).toBe(2);
      expect(context.ringsPerPlayer).toBe(18);
      expect(context.lineLength).toBe(3);
    });
  });

  describe('deriveStateFromGame', () => {
    it('should derive ring_placement state correctly', () => {
      const players = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human' as const,
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'ai' as const,
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const state = createInitialGameState('test-game', 'square8', players, {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      });

      const fsmState = deriveStateFromGame(state);

      expect(fsmState.phase).toBe('ring_placement');
      expect(fsmState.player).toBe(1);
      if (fsmState.phase === 'ring_placement') {
        expect(fsmState.canPlace).toBe(true);
        expect(fsmState.validPositions.length).toBeGreaterThan(0);
      }
    });
  });

  describe('roundtrip conversion', () => {
    it('should preserve move semantics through Move → Event → Move conversion', () => {
      const originalMove: Move = {
        id: 'original-1',
        type: 'place_ring',
        player: 1,
        to: { x: 5, y: 5 },
        timestamp: new Date(),
        thinkTime: 100,
        moveNumber: 3,
      };

      const event = moveToEvent(originalMove);
      expect(event).not.toBeNull();

      const convertedMove = eventToMove(event!, 1, 3);
      expect(convertedMove).not.toBeNull();

      // Key fields should be preserved
      expect(convertedMove!.type).toBe(originalMove.type);
      expect(convertedMove!.to).toEqual(originalMove.to);
      expect(convertedMove!.player).toBe(originalMove.player);
    });

    it('should preserve movement semantics', () => {
      const originalMove: Move = {
        id: 'original-2',
        type: 'move_stack',
        player: 2,
        from: { x: 3, y: 3 },
        to: { x: 6, y: 6 },
        timestamp: new Date(),
        thinkTime: 200,
        moveNumber: 7,
      };

      const event = moveToEvent(originalMove);
      expect(event).not.toBeNull();

      const convertedMove = eventToMove(event!, 2, 7);
      expect(convertedMove).not.toBeNull();

      expect(convertedMove!.type).toBe('move_stack');
      expect(convertedMove!.from).toEqual({ x: 3, y: 3 });
      expect(convertedMove!.to).toEqual({ x: 6, y: 6 });
    });
  });

  describe('validateMoveWithFSM', () => {
    const createTestGameState = (): GameState => {
      const players = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human' as const,
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'ai' as const,
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      return createInitialGameState('test-game', 'square8', players, {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      });
    };

    it('should validate place_ring in ring_placement phase', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      expect(result.valid).toBe(true);
      expect(result.currentPhase).toBe('ring_placement');
    });

    it('should reject move_stack in ring_placement phase', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-2',
        type: 'move_stack',
        player: 1,
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      expect(result.valid).toBe(false);
      expect(result.currentPhase).toBe('ring_placement');
      expect(result.errorCode).toBe('INVALID_EVENT');
      expect(result.validEventTypes).toBeDefined();
    });

    it('should return CONVERSION_FAILED for swap_sides', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-3',
        type: 'swap_sides',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      expect(result.valid).toBe(false);
      expect(result.errorCode).toBe('CONVERSION_FAILED');
      expect(result.reason).toContain('swap_sides');
    });

    it('should reject skip_placement when player can place rings', () => {
      const state = createTestGameState();
      const move: Move = {
        id: 'test-4',
        type: 'skip_placement',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = validateMoveWithFSM(state, move);

      // Should fail because player has rings to place
      expect(result.valid).toBe(false);
      expect(result.errorCode).toBe('GUARD_FAILED');
    });
  });

  describe('isMoveTypeValidForPhase', () => {
    const createTestGameState = (): GameState => {
      const players = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human' as const,
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'ai' as const,
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      return createInitialGameState('test-game', 'square8', players, {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      });
    };

    it('should return true for place_ring in ring_placement phase', () => {
      const state = createTestGameState();

      expect(isMoveTypeValidForPhase(state, 'place_ring')).toBe(true);
    });

    it('should return true for skip_placement in ring_placement phase', () => {
      const state = createTestGameState();

      expect(isMoveTypeValidForPhase(state, 'skip_placement')).toBe(true);
    });

    it('should return false for move_stack in ring_placement phase', () => {
      const state = createTestGameState();

      expect(isMoveTypeValidForPhase(state, 'move_stack')).toBe(false);
    });

    it('should return false for overtaking_capture in ring_placement phase', () => {
      const state = createTestGameState();

      expect(isMoveTypeValidForPhase(state, 'overtaking_capture')).toBe(false);
    });

    it('should return false for forced_elimination in ring_placement phase', () => {
      const state = createTestGameState();

      expect(isMoveTypeValidForPhase(state, 'forced_elimination')).toBe(false);
    });
  });
});
