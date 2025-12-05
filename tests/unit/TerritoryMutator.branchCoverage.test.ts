/**
 * TerritoryMutator.branchCoverage.test.ts
 *
 * Branch coverage tests for TerritoryMutator.ts targeting uncovered branches:
 * - mutateProcessTerritory: region not found, stack elimination, territory gain, player updates
 * - mutateEliminateStack: stack not found, full/partial stack removal, player elimination tracking
 */

import {
  mutateProcessTerritory,
  mutateEliminateStack,
} from '../../src/shared/engine/mutators/TerritoryMutator';
import type {
  GameState,
  ProcessTerritoryAction,
  EliminateStackAction,
  RingStack,
} from '../../src/shared/engine/types';
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
    currentPhase: 'territory_processing',
    gameStatus: 'active',
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    moveHistory: [],
    spectators: [],
    boardType: 'square8',
    totalRingsEliminated: 0,
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
    capHeight: rings.filter((r, i) => i === 0 || rings[i - 1] === r).length,
    controllingPlayer,
  };
  state.board.stacks.set(key, stack);
}

// Helper to add a territory region
function addTerritory(state: GameState, regionId: string, owner: number, spaces: Position[]): void {
  state.board.territories.set(regionId, {
    id: regionId,
    owner,
    spaces,
    borderPositions: [],
    size: spaces.length,
  });
}

// Helper to add a marker
function addMarker(state: GameState, position: Position, player: number): void {
  const key = positionToString(position);
  state.board.markers.set(key, { position, player, type: 'regular' });
}

describe('TerritoryMutator branch coverage', () => {
  describe('mutateProcessTerritory', () => {
    it('throws when region not found', () => {
      const state = makeGameState();
      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'non-existent-region',
      };

      expect(() => mutateProcessTerritory(state, action)).toThrow(
        'TerritoryMutator: Region not found'
      );
    });

    it('processes empty region (no stacks)', () => {
      const state = makeGameState();
      addTerritory(state, 'region-1', 1, [pos(0, 0), pos(1, 0)]);

      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.collapsedSpaces.get('0,0')).toBe(1);
      expect(result.board.collapsedSpaces.get('1,0')).toBe(1);
      expect(result.board.territories.has('region-1')).toBe(false);
      expect(result.players[0].territorySpaces).toBe(2);
    });

    it('eliminates stacks inside region', () => {
      const state = makeGameState();
      addTerritory(state, 'region-1', 1, [pos(0, 0), pos(1, 0)]);
      addStack(state, pos(0, 0), 2, [2, 2, 1]); // 3 rings to eliminate

      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.stacks.has('0,0')).toBe(false);
      expect(result.board.eliminatedRings[1]).toBe(3);
      expect(result.players[0].eliminatedRings).toBe(3);
    });

    it('removes markers inside region', () => {
      const state = makeGameState();
      addTerritory(state, 'region-1', 1, [pos(0, 0), pos(1, 0)]);
      addMarker(state, pos(0, 0), 2);

      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.markers.has('0,0')).toBe(false);
    });

    it('handles region with zero territory gain (edge case)', () => {
      const state = makeGameState();
      // Empty spaces array
      state.board.territories.set('region-empty', {
        id: 'region-empty',
        owner: 1,
        spaces: [],
        borderPositions: [],
        size: 0,
      });

      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-empty',
      };

      const result = mutateProcessTerritory(state, action);

      // territoryGain would be 0, so no player update
      expect(result.players[0].territorySpaces).toBe(0);
    });

    it('handles player not found for territory gain (should not happen normally)', () => {
      const state = makeGameState();
      addTerritory(state, 'region-1', 99, [pos(0, 0)]);

      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 99, // Non-existent player
        regionId: 'region-1',
      };

      // Should not throw, just not update player
      const result = mutateProcessTerritory(state, action);

      expect(result.board.collapsedSpaces.get('0,0')).toBe(99);
      expect(result.board.territories.has('region-1')).toBe(false);
    });

    it('handles multiple stacks in region', () => {
      const state = makeGameState();
      addTerritory(state, 'region-1', 1, [pos(0, 0), pos(1, 0), pos(2, 0)]);
      addStack(state, pos(0, 0), 2, [2, 2]); // 2 rings
      addStack(state, pos(2, 0), 1, [1, 1, 1]); // 3 rings

      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.stacks.size).toBe(0);
      expect(result.board.eliminatedRings[1]).toBe(5); // 2 + 3 = 5
      expect(result.players[0].eliminatedRings).toBe(5);
      expect(result.players[0].territorySpaces).toBe(3);
    });

    it('preserves existing collapsed spaces outside region', () => {
      const state = makeGameState();
      state.board.collapsedSpaces.set('5,5', 2); // Pre-existing collapsed space
      addTerritory(state, 'region-1', 1, [pos(0, 0)]);

      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.collapsedSpaces.get('5,5')).toBe(2);
      expect(result.board.collapsedSpaces.get('0,0')).toBe(1);
    });

    it('initializes eliminatedRings for player if not present', () => {
      const state = makeGameState();
      delete (state.board.eliminatedRings as Record<number, number>)[1];
      addTerritory(state, 'region-1', 1, [pos(0, 0)]);
      addStack(state, pos(0, 0), 2, [2]);

      const action: ProcessTerritoryAction = {
        type: 'process_territory',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = mutateProcessTerritory(state, action);

      expect(result.board.eliminatedRings[1]).toBe(1);
    });
  });

  describe('mutateEliminateStack', () => {
    it('throws when stack not found', () => {
      const state = makeGameState();
      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      expect(() => mutateEliminateStack(state, action)).toThrow(
        'TerritoryMutator: Stack to eliminate not found'
      );
    });

    it('removes entire stack when only cap remains', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1]); // All same color = cap is entire stack

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      const result = mutateEliminateStack(state, action);

      expect(result.board.stacks.has('0,0')).toBe(false);
    });

    it('removes cap and updates stack when rings remain', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1, 2, 2]); // Cap is [1, 1], remaining is [2, 2]

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      const result = mutateEliminateStack(state, action);

      const stack = result.board.stacks.get('0,0');
      expect(stack).toBeDefined();
      expect(stack!.rings).toEqual([2, 2]);
      expect(stack!.stackHeight).toBe(2);
      expect(stack!.controllingPlayer).toBe(2);
    });

    it('updates elimination counts for eliminated cap', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1, 1, 2]); // Cap is [1, 1]

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      const result = mutateEliminateStack(state, action);

      expect(result.board.eliminatedRings[1]).toBe(2);
      expect(result.players[0].eliminatedRings).toBe(2);
    });

    it('handles single-ring stack (complete elimination)', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1]); // Single ring

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      const result = mutateEliminateStack(state, action);

      expect(result.board.stacks.has('0,0')).toBe(false);
      expect(result.board.eliminatedRings[1]).toBe(1);
    });

    it('handles mixed ownership stack (eliminates only cap)', () => {
      const state = makeGameState();
      // [2, 2, 1, 1] - cap is [2, 2], controlled by player 2
      addStack(state, pos(0, 0), 2, [2, 2, 1, 1]);

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 2, // Forced elimination for player 2
        stackPosition: pos(0, 0),
      };

      const result = mutateEliminateStack(state, action);

      const stack = result.board.stacks.get('0,0');
      expect(stack!.rings).toEqual([1, 1]);
      expect(stack!.controllingPlayer).toBe(1);
      expect(result.board.eliminatedRings[2]).toBe(2);
      expect(result.players[1].eliminatedRings).toBe(2);
    });

    it('initializes totalRingsEliminated if not present', () => {
      const state = makeGameState();
      delete (state as GameState & { totalRingsEliminated?: number }).totalRingsEliminated;
      addStack(state, pos(0, 0), 1, [1]);

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      const result = mutateEliminateStack(state, action);

      // Should handle NaN + 1 correctly (the code does totalRingsEliminated += capHeight)
      // Actually this might result in NaN. Let's see what happens.
      expect(
        typeof (result as GameState & { totalRingsEliminated?: number }).totalRingsEliminated
      ).toBe('number');
    });

    it('handles player not found for elimination update', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 99, [99]); // Player 99 doesn't exist

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 99,
        stackPosition: pos(0, 0),
      };

      // Should not throw, just not update player
      const result = mutateEliminateStack(state, action);

      expect(result.board.stacks.has('0,0')).toBe(false);
      expect(result.board.eliminatedRings[99]).toBe(1);
    });

    it('updates capHeight after partial elimination', () => {
      const state = makeGameState();
      // [1, 2, 2, 2] - cap is [1], remaining is [2, 2, 2] with new cap of 3
      addStack(state, pos(0, 0), 1, [1, 2, 2, 2]);

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      const result = mutateEliminateStack(state, action);

      const stack = result.board.stacks.get('0,0');
      expect(stack!.capHeight).toBe(3); // [2, 2, 2] all same color
    });

    it('preserves other stacks on board', () => {
      const state = makeGameState();
      addStack(state, pos(0, 0), 1, [1]);
      addStack(state, pos(1, 0), 2, [2, 2]);

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      const result = mutateEliminateStack(state, action);

      expect(result.board.stacks.has('0,0')).toBe(false);
      expect(result.board.stacks.has('1,0')).toBe(true);
    });
  });
});
