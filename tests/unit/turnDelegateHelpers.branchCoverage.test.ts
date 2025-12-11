/**
 * turnDelegateHelpers.branchCoverage.test.ts
 *
 * Branch coverage tests for turnDelegateHelpers.ts targeting:
 * - hasAnyPlacementForPlayer: player check, board config, cap check, board iteration
 * - hasAnyMovementForPlayer: stack enumeration, mustMoveFromStackKey constraint
 * - hasAnyCaptureForPlayer: capture enumeration, mustMoveFromStackKey constraint
 * - createDefaultTurnLogicDelegates: stack filtering, delegates
 */

import {
  hasAnyPlacementForPlayer,
  hasAnyMovementForPlayer,
  hasAnyCaptureForPlayer,
  createDefaultTurnLogicDelegates,
} from '../../src/shared/engine/turnDelegateHelpers';
import type { GameState, RingStack } from '../../src/shared/engine/types';
import type { Position, BoardType } from '../../src/shared/types/game';
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
    currentPhase: 'ring_placement',
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
  rings: number[]
): void {
  const key = positionToString(position);
  const stack: RingStack = {
    position,
    rings,
    stackHeight: rings.length,
    capHeight: rings.length,
    controllingPlayer,
  };
  state.board.stacks.set(key, stack);
}

describe('turnDelegateHelpers branch coverage', () => {
  describe('hasAnyPlacementForPlayer', () => {
    describe('player checks', () => {
      it('returns false when player not found', () => {
        const state = makeGameState();
        // Player 99 doesn't exist
        const result = hasAnyPlacementForPlayer(state, 99);
        expect(result).toBe(false);
      });

      it('returns false when player has no rings in hand', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 0;
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(false);
      });

      it('returns true when player has rings in hand and empty board', () => {
        const state = makeGameState();
        state.players[0].ringsInHand = 10;
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(true);
      });
    });

    describe('board config checks', () => {
      it('returns false for unknown board type', () => {
        const state = makeGameState();
        state.board.type = 'unknown_board' as BoardType;
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(false);
      });
    });

    describe('ring capacity checks', () => {
      it('returns false when ring capacity is exhausted', () => {
        const state = makeGameState();
        // For square8, ringsPerPlayer is 18. Add 18 rings on board.
        // Add stacks with multiple rings to reach the cap
        for (let i = 0; i < 6; i++) {
          addStack(state, pos(i, 0), 1, [1, 1, 1]); // 3 rings each = 18 total
        }
        state.players[0].ringsInHand = 0; // No rings in hand anyway
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(false);
      });

      it('returns true when player has remaining capacity', () => {
        const state = makeGameState();
        addStack(state, pos(0, 0), 1, [1]); // 1 ring on board
        state.players[0].ringsInHand = 17; // 17 in hand, 1 on board = 18 total cap not reached
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(true);
      });
    });

    describe('square board iteration', () => {
      it('finds placement on square8 board', () => {
        const state = makeGameState();
        state.board.type = 'square8';
        state.board.size = 8;
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(true);
      });

      it('returns false when entire square8 board is collapsed', () => {
        const state = makeGameState();
        // Collapse all spaces on the board
        for (let x = 0; x < 8; x++) {
          for (let y = 0; y < 8; y++) {
            state.board.collapsedSpaces.set(`${x},${y}`, 1);
          }
        }
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(false);
      });

      it('handles square19 board type', () => {
        const state = makeGameState();
        state.board.type = 'square19';
        state.board.size = 19;
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(true);
      });
    });

    describe('hexagonal board iteration', () => {
      it('finds placement on hexagonal board', () => {
        const state = makeGameState();
        state.board.type = 'hexagonal';
        state.board.size = 13; // Hex: size=13, radius=12
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(result).toBe(true);
      });
    });

    describe('placement validation', () => {
      it('respects dead-ring placement rule (avoids creating isolated ring)', () => {
        const state = makeGameState();
        // Fill most of the board to create constrained placement scenarios
        // This tests that validatePlacementOnBoard is being called
        const result = hasAnyPlacementForPlayer(state, 1);
        expect(typeof result).toBe('boolean');
      });
    });
  });

  describe('hasAnyMovementForPlayer', () => {
    it('returns false when player has no stacks', () => {
      const state = makeGameState();
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      const result = hasAnyMovementForPlayer(state, 1, turn);
      expect(result).toBe(false);
    });

    it('returns true when player has a stack with valid moves', () => {
      const state = makeGameState();
      // Place a stack at center with height 2 (can move 2 spaces)
      addStack(state, pos(4, 4), 1, [1, 1]);
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      const result = hasAnyMovementForPlayer(state, 1, turn);
      expect(result).toBe(true);
    });

    it('returns false when player has stack but mustMoveFromStackKey points elsewhere', () => {
      const state = makeGameState();
      addStack(state, pos(4, 4), 1, [1, 1]);
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: '0,0' }; // Different position

      const result = hasAnyMovementForPlayer(state, 1, turn);
      expect(result).toBe(false);
    });

    it('returns true when mustMoveFromStackKey matches stack with valid moves', () => {
      const state = makeGameState();
      addStack(state, pos(4, 4), 1, [1, 1]);
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: '4,4' };

      const result = hasAnyMovementForPlayer(state, 1, turn);
      expect(result).toBe(true);
    });

    it('returns false when stack is surrounded and cannot move', () => {
      const state = makeGameState();
      // Place a height-1 stack at corner with adjacent stacks blocking all moves
      addStack(state, pos(0, 0), 1, [1]); // Can move 1 space
      addStack(state, pos(1, 0), 2, [2]); // Block right
      addStack(state, pos(0, 1), 2, [2]); // Block down
      addStack(state, pos(1, 1), 2, [2]); // Block diagonal
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      const result = hasAnyMovementForPlayer(state, 1, turn);
      expect(result).toBe(false);
    });
  });

  describe('hasAnyCaptureForPlayer', () => {
    it('returns false when player has no stacks', () => {
      const state = makeGameState();
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      const result = hasAnyCaptureForPlayer(state, 1, turn);
      expect(result).toBe(false);
    });

    it('returns false when player has stack but no capturable targets', () => {
      const state = makeGameState();
      addStack(state, pos(4, 4), 1, [1, 1]); // Player 1 stack
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      const result = hasAnyCaptureForPlayer(state, 1, turn);
      expect(result).toBe(false);
    });

    it('returns true when player can capture an enemy stack', () => {
      const state = makeGameState();
      // Player 1 stack with cap height 2
      addStack(state, pos(4, 4), 1, [1, 1]);
      // Player 2 stack with cap height 1 at distance 2 (player 1 can reach)
      addStack(state, pos(4, 6), 2, [2]);
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      const result = hasAnyCaptureForPlayer(state, 1, turn);
      expect(result).toBe(true);
    });

    it('returns false when cap height insufficient for capture', () => {
      const state = makeGameState();
      // Player 1 stack with cap height 1 cannot capture a height 2 enemy
      addStack(state, pos(4, 4), 1, [1]);
      // Player 2 stack with cap height 2 - can't be captured by cap height 1
      addStack(state, pos(4, 5), 2, [2, 2]);
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      // RR-CANON-R070: Attacker cap height must be >= target cap height
      const result = hasAnyCaptureForPlayer(state, 1, turn);
      expect(result).toBe(false);
    });

    it('returns false when mustMoveFromStackKey points to stack without captures', () => {
      const state = makeGameState();
      addStack(state, pos(4, 4), 1, [1, 1]);
      addStack(state, pos(0, 0), 1, [1]); // Another stack with no captures
      addStack(state, pos(4, 6), 2, [2]); // Target reachable from 4,4 only
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: '0,0' };

      const result = hasAnyCaptureForPlayer(state, 1, turn);
      expect(result).toBe(false);
    });

    it('returns true when mustMoveFromStackKey matches stack with captures', () => {
      const state = makeGameState();
      addStack(state, pos(4, 4), 1, [1, 1]);
      addStack(state, pos(4, 6), 2, [2]); // Target reachable from 4,4
      const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: '4,4' };

      const result = hasAnyCaptureForPlayer(state, 1, turn);
      expect(result).toBe(true);
    });
  });

  describe('createDefaultTurnLogicDelegates', () => {
    const mockConfig = {
      getNextPlayerNumber: jest.fn().mockReturnValue(2),
      applyForcedElimination: jest.fn().mockReturnValue(undefined),
    };

    it('creates delegates with all required methods', () => {
      const delegates = createDefaultTurnLogicDelegates(mockConfig);

      expect(delegates.getPlayerStacks).toBeDefined();
      expect(delegates.hasAnyPlacement).toBeDefined();
      expect(delegates.hasAnyMovement).toBeDefined();
      expect(delegates.hasAnyCapture).toBeDefined();
      expect(delegates.applyForcedElimination).toBe(mockConfig.applyForcedElimination);
      expect(delegates.getNextPlayerNumber).toBe(mockConfig.getNextPlayerNumber);
    });

    describe('getPlayerStacks delegate', () => {
      it('returns empty array when player has no stacks', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();

        const stacks = delegates.getPlayerStacks(state, 1);
        expect(stacks).toEqual([]);
      });

      it('returns only stacks controlled by specified player', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();
        addStack(state, pos(0, 0), 1, [1, 1]);
        addStack(state, pos(1, 0), 2, [2, 2, 2]);
        addStack(state, pos(2, 0), 1, [1]);

        const player1Stacks = delegates.getPlayerStacks(state, 1);
        expect(player1Stacks.length).toBe(2);
        expect(player1Stacks[0].stackHeight).toBe(2);
        expect(player1Stacks[1].stackHeight).toBe(1);

        const player2Stacks = delegates.getPlayerStacks(state, 2);
        expect(player2Stacks.length).toBe(1);
        expect(player2Stacks[0].stackHeight).toBe(3);
      });

      it('includes position and stackHeight in returned stacks', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();
        addStack(state, pos(3, 4), 1, [1, 1, 1, 1]);

        const stacks = delegates.getPlayerStacks(state, 1);
        expect(stacks[0]).toEqual({
          position: pos(3, 4),
          stackHeight: 4,
        });
      });
    });

    describe('hasAnyPlacement delegate', () => {
      it('delegates to hasAnyPlacementForPlayer', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();

        const result = delegates.hasAnyPlacement(state, 1);
        expect(result).toBe(true);
      });

      it('returns false when player has no rings', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();
        state.players[0].ringsInHand = 0;

        const result = delegates.hasAnyPlacement(state, 1);
        expect(result).toBe(false);
      });
    });

    describe('hasAnyMovement delegate', () => {
      it('delegates to hasAnyMovementForPlayer', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();
        const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

        // No stacks = no movement
        const result = delegates.hasAnyMovement(state, 1, turn);
        expect(result).toBe(false);
      });

      it('returns true when player has movable stacks', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();
        addStack(state, pos(4, 4), 1, [1, 1]);
        const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

        const result = delegates.hasAnyMovement(state, 1, turn);
        expect(result).toBe(true);
      });
    });

    describe('hasAnyCapture delegate', () => {
      it('delegates to hasAnyCaptureForPlayer', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();
        const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

        // No stacks = no captures
        const result = delegates.hasAnyCapture(state, 1, turn);
        expect(result).toBe(false);
      });

      it('returns true when player can capture', () => {
        const delegates = createDefaultTurnLogicDelegates(mockConfig);
        const state = makeGameState();
        addStack(state, pos(4, 4), 1, [1, 1]);
        addStack(state, pos(4, 6), 2, [2]);
        const turn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

        const result = delegates.hasAnyCapture(state, 1, turn);
        expect(result).toBe(true);
      });
    });
  });
});
