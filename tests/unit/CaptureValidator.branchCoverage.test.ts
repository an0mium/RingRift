/**
 * CaptureValidator.branchCoverage.test.ts
 *
 * Branch coverage tests for CaptureValidator.ts targeting uncovered branches:
 * - Phase check (movement, capture, chain_capture valid; others invalid)
 * - Turn check
 * - Position validity (from, to, captureTarget)
 * - Core validator result (valid vs invalid capture)
 */

import { validateCapture } from '../../src/shared/engine/validators/CaptureValidator';
import type { GameState, OvertakingCaptureAction, RingStack } from '../../src/shared/engine/types';
import type { Position, BoardType, Marker } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

// Helper to create a position
const pos = (x: number, y: number): Position => ({ x, y });

// Helper to create a minimal game state for testing
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultState: GameState = {
    id: 'test-game',
    board: {
      type: 'square8' as BoardType,
      size: 8,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      formedLines: [],
      territories: new Map(),
      eliminatedRings: { 1: 0, 2: 0 },
    },
    players: [
      {
        id: 'p1',
        username: 'Player1',
        playerNumber: 1,
        type: 'human',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'Player2',
        playerNumber: 2,
        type: 'human',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ],
    currentPlayer: 1,
    currentPhase: 'movement',
    gameStatus: 'active',
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    moveHistory: [],
    spectators: [],
    boardType: 'square8',
  };

  return { ...defaultState, ...overrides } as GameState;
}

// Helper to add a stack to the board
function addStack(
  state: GameState,
  position: Position,
  controllingPlayer: number,
  stackHeight: number,
  capHeight?: number
): void {
  const key = positionToString(position);
  const rings = Array(stackHeight).fill(controllingPlayer);
  const stack: RingStack = {
    position,
    rings,
    stackHeight,
    capHeight: capHeight ?? stackHeight,
    controllingPlayer,
  };
  state.board.stacks.set(key, stack);
}

// Helper to add a marker
function addMarker(state: GameState, position: Position, player: number): void {
  const key = positionToString(position);
  const marker: Marker = { position, player, type: 'regular' };
  state.board.markers.set(key, marker);
}

describe('CaptureValidator branch coverage', () => {
  describe('phase check', () => {
    it('accepts capture in movement phase', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      // Set up a valid capture scenario: player 1 stack at (0,0), player 2 target at (1,0), landing at (2,0)
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      // Result depends on core validator, but phase check passes
      expect(result.code).not.toBe('INVALID_PHASE');
    });

    it('accepts capture in capture phase', () => {
      const state = makeGameState({ currentPhase: 'capture' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.code).not.toBe('INVALID_PHASE');
    });

    it('accepts capture in chain_capture phase', () => {
      const state = makeGameState({ currentPhase: 'chain_capture' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.code).not.toBe('INVALID_PHASE');
    });

    it('rejects capture in ring_placement phase', () => {
      const state = makeGameState({ currentPhase: 'ring_placement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects capture in territory_processing phase', () => {
      const state = makeGameState({ currentPhase: 'territory_processing' });

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });
  });

  describe('turn check', () => {
    it('rejects capture when not player turn', () => {
      const state = makeGameState({ currentPlayer: 2, currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1, // Player 1 trying to capture
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });
  });

  describe('position validity', () => {
    it('rejects when from position is off board', () => {
      const state = makeGameState({ currentPhase: 'movement' });

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(-1, 0), // Off board
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects when to position is off board', () => {
      const state = makeGameState({ currentPhase: 'movement' });

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(10, 0), // Off board
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects when captureTarget position is off board', () => {
      const state = makeGameState({ currentPhase: 'movement' });

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(-1, -1), // Off board
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });
  });

  describe('core validator', () => {
    it('accepts valid capture scenario', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      // Player 1 stack at (0,0) with height 2
      addStack(state, pos(0, 0), 1, 2);
      // Player 2 target stack at (1,0) with height 1 (capturable)
      addStack(state, pos(1, 0), 2, 1);
      // Landing position (2,0) is empty and reachable

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects invalid capture (no stack at from)', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      // No stack at from position
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0), // No stack here
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });

    it('allows capture of own stack (self-capture is valid)', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 1, 1); // Same player as attacker

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      // Self-capture is allowed by the rules
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects invalid capture (attacking stack shorter than target)', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 1); // Height 1
      addStack(state, pos(1, 0), 2, 2); // Height 2 - taller than attacker

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });

    it('rejects invalid capture (landing on collapsed space)', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);
      state.board.collapsedSpaces.set('2,0', 1); // Landing is collapsed

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });

    it('rejects invalid capture (landing on existing stack)', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);
      addStack(state, pos(2, 0), 1, 1); // Landing has a stack

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_CAPTURE');
    });

    it('handles marker at landing position', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);
      addMarker(state, pos(2, 0), 2); // Marker at landing

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      // Landing on marker should be valid
      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });
  });

  describe('edge cases', () => {
    it('handles diagonal capture', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 1), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 1),
        to: pos(2, 2),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('validates result of core capture validator', () => {
      // This test verifies that the CaptureValidator passes through
      // the result from the core validator correctly
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      // The result should be valid for this standard capture scenario
      expect(result.valid).toBe(true);
    });
  });

  describe('additional edge cases', () => {
    it('rejects capture with invalid from position (negative)', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(1, 1), 1, 2);
      addStack(state, pos(2, 1), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(-1, 0),
        captureTarget: pos(2, 1),
        to: pos(3, 1),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects capture with invalid to position (off board)', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(6, 0), 1, 2);
      addStack(state, pos(7, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(6, 0),
        captureTarget: pos(7, 0),
        to: pos(8, 0), // Off board
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects capture with invalid captureTarget position', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(-1, -1),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_POSITION');
    });

    it('rejects capture in ring_placement phase', () => {
      const state = makeGameState({ currentPhase: 'ring_placement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects capture in line_processing phase', () => {
      const state = makeGameState({ currentPhase: 'line_processing' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects capture in territory_processing phase', () => {
      const state = makeGameState({ currentPhase: 'territory_processing' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('allows capture in capture phase', () => {
      const state = makeGameState({ currentPhase: 'capture' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('allows capture in chain_capture phase', () => {
      const state = makeGameState({ currentPhase: 'chain_capture' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects capture when landing on collapsed space', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);
      state.board.collapsedSpaces.set(positionToString(pos(2, 0)), 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
    });

    it('rejects capture when no stack at from position', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0), // No stack here
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
    });

    it('rejects capture when no stack at capture target', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0), // No stack here
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
    });

    it('handles capture of own stack (allowed when overtaking)', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 1, 1); // Own smaller stack

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      // Overtaking own stack may be valid if heights allow
      expect(result.valid).toBe(true);
    });

    it('rejects capture when attacker height is not greater than target', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 1); // Smaller
      addStack(state, pos(1, 0), 2, 2); // Bigger

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(false);
    });

    it('validates capture when positions form a valid line', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      addStack(state, pos(0, 0), 1, 2);
      addStack(state, pos(1, 0), 2, 1);

      const action: OvertakingCaptureAction = {
        type: 'OVERTAKING_CAPTURE',
        playerId: 1,
        from: pos(0, 0),
        captureTarget: pos(1, 0),
        to: pos(2, 0),
      };

      const result = validateCapture(state, action);
      expect(result.valid).toBe(true);
    });
  });
});
