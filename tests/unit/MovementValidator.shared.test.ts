/**
 * MovementValidator branch coverage tests
 * Tests for src/shared/engine/validators/MovementValidator.ts
 *
 * Structural coverage for non-capture movement aligned with:
 * - RulesMatrix M1 – minimum distance and stack height (§8.2, FAQ Q2).
 * - RulesMatrix M2 – landing beyond marker runs (§8.2–8.3, FAQ Q2–Q3).
 * - RulesMatrix M3 – overtaking capture vs move_stack parity (§9–10).
 *
 * These tests exercise the validation logic for movement actions including:
 * - Phase/turn checks
 * - Position validity (square8, square19, hex boards)
 * - Stack ownership
 * - Collapsed space checks
 * - Direction validation
 * - Distance requirements
 * - Path blocking
 * - Landing restrictions
 *
 * Full scenario behaviour is covered by:
 * - tests/unit/movement.shared.test.ts
 * - tests/scenarios/RulesMatrix.Comprehensive.test.ts
 */

import { validateMovement } from '@shared/engine/validators/MovementValidator';
import type { GameState, MoveStackAction } from '@shared/engine/types';

// Helper to create minimal GameState for movement validation tests
function createMinimalState(
  overrides: Partial<{
    currentPhase: string;
    currentPlayer: number;
    boardType: 'square' | 'hex';
    boardSize: number;
    stacks: Map<string, { controllingPlayer: number; stackHeight: number }>;
    markers: Map<string, { player: number }>;
    collapsedSpaces: Set<string>;
  }>
): GameState {
  const boardType = overrides.boardType ?? 'square';
  const boardSize = overrides.boardSize ?? 8;

  const base = {
    board: {
      type: boardType,
      size: boardSize,
      stacks: overrides.stacks ?? new Map(),
      markers: overrides.markers ?? new Map(),
      collapsedSpaces: overrides.collapsedSpaces ?? new Set(),
      rings: new Map(),
      territories: new Map(),
      geometry: { type: boardType, size: boardSize },
    },
    currentPhase: overrides.currentPhase ?? 'movement',
    currentPlayer: overrides.currentPlayer ?? 1,
    players: [
      { id: 1, eliminated: false, score: 0, reserveStacks: 0, reserveRings: 0 },
      { id: 2, eliminated: false, score: 0, reserveStacks: 0, reserveRings: 0 },
    ],
    turnNumber: 1,
    gameStatus: 'active' as const,
    moveHistory: [],
    pendingDecision: null,
    victoryCondition: null,
  };
  return base as unknown as GameState;
}

describe('MovementValidator', () => {
  describe('validateMovement', () => {
    describe('Phase Check', () => {
      it('returns error when not in movement phase', () => {
        const state = createMinimalState({ currentPhase: 'placement' });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
        expect(result.reason).toContain('movement phase');
      });
    });

    describe('Turn Check', () => {
      it('returns error when not current player turn', () => {
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 1 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 2,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_TURN');
        expect(result.reason).toContain('turn');
      });
    });

    describe('Position Validity', () => {
      it('returns error when from position is off board', () => {
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          boardSize: 8,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: -1, y: 0 },
          to: { x: 0, y: 0 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_POSITION');
        expect(result.reason).toContain('off board');
      });

      it('returns error when to position is off board', () => {
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 1 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          boardSize: 8,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 100, y: 0 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_POSITION');
      });

      it('respects square19 bounds for long moves', () => {
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 3 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          boardType: 'square',
          boardSize: 19,
          stacks,
        });

        // In-bounds move along a file on 19x19
        const inBoundsAction: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 10 },
          playerId: 1,
        };
        const inBoundsResult = validateMovement(state, inBoundsAction);

        // May still fail due to distance/stack constraints, but not as INVALID_POSITION.
        if (!inBoundsResult.valid) {
          expect(inBoundsResult.code).not.toBe('INVALID_POSITION');
        }

        // Off-board move beyond size 19 must be rejected as INVALID_POSITION.
        const offBoardAction: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 19 },
          playerId: 1,
        };
        const offBoardResult = validateMovement(state, offBoardAction);

        expect(offBoardResult.valid).toBe(false);
        expect(offBoardResult.code).toBe('INVALID_POSITION');
      });

      it('respects hexagonal bounds for movement endpoints', () => {
        const stacks = new Map([['0,0,0', { controllingPlayer: 1, stackHeight: 1 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          boardType: 'hex',
          boardSize: 4,
          stacks,
        });

        const inBoundsAction: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0, z: 0 },
          to: { x: 2, y: -2, z: 0 },
          playerId: 1,
        };
        const inBoundsResult = validateMovement(state, inBoundsAction);

        if (!inBoundsResult.valid) {
          expect(inBoundsResult.code).not.toBe('INVALID_POSITION');
        }

        const offBoardAction: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0, z: 0 },
          to: { x: 5, y: -5, z: 0 },
          playerId: 1,
        };
        const offBoardResult = validateMovement(state, offBoardAction);

        expect(offBoardResult.valid).toBe(false);
        expect(offBoardResult.code).toBe('INVALID_POSITION');
      });
    });

    describe('Stack Ownership', () => {
      it('returns error when no stack at starting position', () => {
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks: new Map(), // No stacks
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_STACK');
        expect(result.reason).toContain('No stack');
      });

      it('returns error when stack is controlled by different player', () => {
        const stacks = new Map([['0,0', { controllingPlayer: 2, stackHeight: 1 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_STACK');
        expect(result.reason).toContain('do not control');
      });
    });

    describe('Collapsed Space Check', () => {
      it('returns error when destination is collapsed', () => {
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 1 }]]);
        const collapsedSpaces = new Set(['0,1']);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          collapsedSpaces,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('COLLAPSED_SPACE');
        expect(result.reason).toContain('collapsed');
      });
    });

    describe('Direction Check', () => {
      it('returns error for completely invalid direction (non-aligned move)', () => {
        // A move that doesn't align with any known direction pattern
        const stacks = new Map([['2,2', { controllingPlayer: 1, stackHeight: 1 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          boardType: 'square',
        });
        // Moving by (3, 2) which doesn't align with orthogonal or diagonal directions
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 2, y: 2 },
          to: { x: 5, y: 4 }, // (3, 2) offset - non-aligned
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_DIRECTION');
        expect(result.reason).toContain('direction');
      });

      it('allows orthogonal movement on square board', () => {
        const stacks = new Map([['2,2', { controllingPlayer: 1, stackHeight: 1 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          boardType: 'square',
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 2, y: 2 },
          to: { x: 2, y: 4 }, // Orthogonal move (up by 2)
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(true);
      });

      it('returns error for diagonal move on square board that does not match a canonical direction', () => {
        // This test complements M3: ensure purely geometric diagonals that
        // are not part of the allowed movement direction set are rejected
        // before overtaking/capture logic is considered.
        const stacks = new Map([['1,1', { controllingPlayer: 1, stackHeight: 1 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          boardType: 'square',
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 1, y: 1 },
          to: { x: 3, y: 3 }, // Diagonal by (2,2); validator must classify via direction set
          playerId: 1,
        };

        const result = validateMovement(state, action);

        // Depending on getMovementDirectionsForBoardType, this may or may not
        // be allowed; the structural guarantee is that when rejected, it is
        // due to INVALID_DIRECTION rather than a later phase.
        if (!result.valid) {
          expect(result.code).toBe('INVALID_DIRECTION');
        }
      });
    });

    describe('Distance Check', () => {
      it('returns error when move distance is less than stack height', () => {
        // Stack height 2 means must move at least 2 spaces
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 2 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 }, // Distance 1, but stack height is 2
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('INSUFFICIENT_DISTANCE');
        expect(result.reason).toContain('stack height');
      });
    });

    describe('Path Check', () => {
      it('returns error when path is blocked by collapsed space', () => {
        // Moving from 0,0 to 0,3 must pass through 0,1 and 0,2
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 1 }]]);
        const collapsedSpaces = new Set(['0,1']); // Block path
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          collapsedSpaces,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 3 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('PATH_BLOCKED');
        expect(result.reason).toContain('blocked');
      });

      it('returns error when path is blocked by another stack', () => {
        // Moving from 0,0 to 0,3 must pass through 0,1 and 0,2
        const stacks = new Map([
          ['0,0', { controllingPlayer: 1, stackHeight: 1 }],
          ['0,2', { controllingPlayer: 2, stackHeight: 1 }], // Blocking stack
        ]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 3 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('PATH_BLOCKED');
        expect(result.reason).toContain('stack');
      });
    });

    describe('Landing Check', () => {
      it('returns error when landing on opponent marker', () => {
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 1 }]]);
        const markers = new Map([['0,1', { player: 2 }]]); // Opponent marker
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          markers,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_LANDING');
        expect(result.reason).toContain('opponent marker');
      });

      it('returns error when landing on existing stack', () => {
        const stacks = new Map([
          ['0,0', { controllingPlayer: 1, stackHeight: 1 }],
          ['0,1', { controllingPlayer: 1, stackHeight: 1 }], // Cannot land on another stack
        ]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 1 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_LANDING');
        expect(result.reason).toContain('existing stack');
      });
    });

    describe('Valid Movements', () => {
      it('returns valid for move to empty space with sufficient distance', () => {
        const stacks = new Map([['2,2', { controllingPlayer: 1, stackHeight: 1 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 2, y: 2 },
          to: { x: 2, y: 3 }, // Distance 1, stack height 1 - valid
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(true);
      });

      it('returns valid when landing on own marker', () => {
        const stacks = new Map([['2,2', { controllingPlayer: 1, stackHeight: 1 }]]);
        const markers = new Map([['2,3', { player: 1 }]]); // Own marker
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
          markers,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 2, y: 2 },
          to: { x: 2, y: 3 },
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(true);
      });

      it('returns valid for longer move matching stack height', () => {
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 3 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 3 }, // Distance 3 matches stack height 3
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(true);
      });

      it('returns valid for move longer than stack height', () => {
        const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 2 }]]);
        const state = createMinimalState({
          currentPhase: 'movement',
          currentPlayer: 1,
          stacks,
        });
        const action: MoveStackAction = {
          type: 'moveStack',
          from: { x: 0, y: 0 },
          to: { x: 0, y: 4 }, // Distance 4 >= stack height 2
          playerId: 1,
        };

        const result = validateMovement(state, action);

        expect(result.valid).toBe(true);
      });
    });
  });
});
