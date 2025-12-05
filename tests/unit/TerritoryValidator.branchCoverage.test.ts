/**
 * TerritoryValidator.branchCoverage.test.ts
 *
 * Branch coverage tests for TerritoryValidator.ts targeting uncovered branches:
 * - validateProcessTerritory: phase check, turn check, region existence, disconnection check
 * - validateEliminateStack: phase check, turn check, stack existence, ownership, height check
 */

import {
  validateProcessTerritory,
  validateEliminateStack,
} from '../../src/shared/engine/validators/TerritoryValidator';
import type {
  GameState,
  ProcessTerritoryAction,
  EliminateStackAction,
  RingStack,
} from '../../src/shared/engine/types';
import type { Position, BoardType, TerritoryRegion } from '../../src/shared/types/game';
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
    currentPhase: 'territory_processing',
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
  stackHeight: number
): void {
  const key = positionToString(position);
  const rings = Array(stackHeight).fill(controllingPlayer);
  const stack: RingStack = {
    position,
    rings,
    stackHeight,
    capHeight: stackHeight,
    controllingPlayer,
  };
  state.board.stacks.set(key, stack);
}

// Helper to add a territory region
function addTerritory(
  state: GameState,
  regionId: string,
  owner: number,
  spaces: Position[],
  isDisconnected: boolean = true
): void {
  const region: TerritoryRegion = {
    id: regionId,
    owner,
    spaces,
    borderPositions: [],
    size: spaces.length,
    isDisconnected,
  };
  state.board.territories.set(regionId, region);
}

describe('TerritoryValidator branch coverage', () => {
  describe('validateProcessTerritory', () => {
    describe('phase check', () => {
      it('rejects when not in territory_processing phase (ring_placement)', () => {
        const state = makeGameState({ currentPhase: 'ring_placement' });
        addTerritory(state, 'region-1', 1, [pos(0, 0)], true);

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
        expect(result.reason).toBe('Not in territory processing phase');
      });

      it('rejects when in movement phase', () => {
        const state = makeGameState({ currentPhase: 'movement' });

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });

      it('rejects when in capture phase', () => {
        const state = makeGameState({ currentPhase: 'capture' });

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });
    });

    describe('turn check', () => {
      it('rejects when not the player turn', () => {
        const state = makeGameState({ currentPlayer: 2 });
        addTerritory(state, 'region-1', 1, [pos(0, 0)], true);

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1, // Player 1 trying to act
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_TURN');
        expect(result.reason).toBe('Not your turn');
      });

      it('accepts when player matches current player', () => {
        const state = makeGameState({ currentPlayer: 1 });
        addTerritory(state, 'region-1', 1, [pos(0, 0)], true);

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('region existence check', () => {
      it('rejects when region not found', () => {
        const state = makeGameState();

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'non-existent-region',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('REGION_NOT_FOUND');
        expect(result.reason).toBe('Region not found');
      });

      it('accepts when region exists', () => {
        const state = makeGameState();
        addTerritory(state, 'region-1', 1, [pos(0, 0)], true);

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('disconnection check', () => {
      it('rejects when region is not disconnected', () => {
        const state = makeGameState();
        addTerritory(state, 'region-1', 1, [pos(0, 0)], false); // isDisconnected = false

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('REGION_NOT_DISCONNECTED');
        expect(result.reason).toBe('Region is not disconnected');
      });

      it('accepts when region is disconnected', () => {
        const state = makeGameState();
        addTerritory(state, 'region-1', 1, [pos(0, 0)], true);

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('valid actions', () => {
      it('accepts valid process_territory action', () => {
        const state = makeGameState();
        addTerritory(state, 'region-1', 1, [pos(0, 0), pos(1, 0)], true);

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 1,
          regionId: 'region-1',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(true);
        expect(result.reason).toBeUndefined();
        expect(result.code).toBeUndefined();
      });

      it('accepts action for player 2', () => {
        const state = makeGameState({ currentPlayer: 2 });
        addTerritory(state, 'region-2', 2, [pos(5, 5)], true);

        const action: ProcessTerritoryAction = {
          type: 'process_territory',
          playerId: 2,
          regionId: 'region-2',
        };

        const result = validateProcessTerritory(state, action);
        expect(result.valid).toBe(true);
      });
    });
  });

  describe('validateEliminateStack', () => {
    describe('phase check', () => {
      it('rejects when not in territory_processing phase (ring_placement)', () => {
        const state = makeGameState({ currentPhase: 'ring_placement' });
        addStack(state, pos(0, 0), 1, 2);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
        expect(result.reason).toBe('Not in territory processing phase');
      });

      it('rejects when in movement phase', () => {
        const state = makeGameState({ currentPhase: 'movement' });
        addStack(state, pos(0, 0), 1, 2);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });

      it('rejects when in capture phase', () => {
        const state = makeGameState({ currentPhase: 'capture' });
        addStack(state, pos(0, 0), 1, 2);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });
    });

    describe('turn check', () => {
      it('rejects when not the player turn', () => {
        const state = makeGameState({ currentPlayer: 2 });
        addStack(state, pos(0, 0), 1, 2);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1, // Player 1 trying to act
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_TURN');
        expect(result.reason).toBe('Not your turn');
      });

      it('accepts when player matches current player', () => {
        const state = makeGameState({ currentPlayer: 1 });
        addStack(state, pos(0, 0), 1, 2);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('stack existence check', () => {
      it('rejects when stack not found', () => {
        const state = makeGameState();
        // No stack at position

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('STACK_NOT_FOUND');
        expect(result.reason).toBe('Stack not found');
      });

      it('accepts when stack exists', () => {
        const state = makeGameState();
        addStack(state, pos(0, 0), 1, 2);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('stack ownership check', () => {
      it('rejects when stack not controlled by player', () => {
        const state = makeGameState();
        addStack(state, pos(0, 0), 2, 2); // Player 2's stack

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1, // Player 1 trying to eliminate
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_STACK');
        expect(result.reason).toBe('Stack is not controlled by player');
      });

      it('accepts when stack is controlled by player', () => {
        const state = makeGameState();
        addStack(state, pos(0, 0), 1, 2);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('stack height check', () => {
      it('rejects when stack height is 0', () => {
        const state = makeGameState();
        // Create a stack with 0 height (edge case)
        const key = positionToString(pos(0, 0));
        state.board.stacks.set(key, {
          position: pos(0, 0),
          rings: [],
          stackHeight: 0,
          capHeight: 0,
          controllingPlayer: 1,
        });

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('EMPTY_STACK');
        expect(result.reason).toBe('Stack is empty');
      });

      it('rejects when stack height is negative (edge case)', () => {
        const state = makeGameState();
        // Create a stack with negative height (edge case)
        const key = positionToString(pos(0, 0));
        state.board.stacks.set(key, {
          position: pos(0, 0),
          rings: [],
          stackHeight: -1,
          capHeight: 0,
          controllingPlayer: 1,
        });

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('EMPTY_STACK');
      });

      it('accepts when stack height is 1', () => {
        const state = makeGameState();
        addStack(state, pos(0, 0), 1, 1);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(true);
      });

      it('accepts when stack height is greater than 1', () => {
        const state = makeGameState();
        addStack(state, pos(0, 0), 1, 5);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(0, 0),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('valid actions', () => {
      it('accepts valid eliminate_stack action', () => {
        const state = makeGameState();
        addStack(state, pos(3, 4), 1, 3);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 1,
          stackPosition: pos(3, 4),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(true);
        expect(result.reason).toBeUndefined();
        expect(result.code).toBeUndefined();
      });

      it('accepts action for player 2', () => {
        const state = makeGameState({ currentPlayer: 2 });
        addStack(state, pos(5, 5), 2, 2);

        const action: EliminateStackAction = {
          type: 'eliminate_stack',
          playerId: 2,
          stackPosition: pos(5, 5),
        };

        const result = validateEliminateStack(state, action);
        expect(result.valid).toBe(true);
      });
    });
  });

  describe('edge cases', () => {
    it('validates multiple regions in board territories', () => {
      const state = makeGameState();
      addTerritory(state, 'region-1', 1, [pos(0, 0)], true);
      addTerritory(state, 'region-2', 2, [pos(1, 0)], true);

      const action1: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-1',
      };

      const result1 = validateProcessTerritory(state, action1);
      expect(result1.valid).toBe(true);

      // Cannot process region-2 as player 1
      const action2: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-2',
      };

      const result2 = validateProcessTerritory(state, action2);
      // Should still be valid since region exists and is disconnected
      // The actual owner check is not performed in the validator
      expect(result2.valid).toBe(true);
    });

    it('handles position string conversion correctly', () => {
      const state = makeGameState();
      // Add stack at non-origin position
      addStack(state, pos(7, 7), 1, 2);

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(7, 7),
      };

      const result = validateEliminateStack(state, action);
      expect(result.valid).toBe(true);
    });

    it('handles various board sizes', () => {
      const state = makeGameState();
      state.board.size = 19;
      addStack(state, pos(18, 18), 1, 1);

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(18, 18),
      };

      const result = validateEliminateStack(state, action);
      expect(result.valid).toBe(true);
    });
  });
});
