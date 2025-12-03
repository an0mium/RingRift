/**
 * TerritoryValidator Unit Tests
 *
 * Exercises the shared engine territory validation logic to cover:
 * - Phase and turn checks for PROCESS_TERRITORY and ELIMINATE_STACK
 * - Region existence and disconnection predicates
 * - Stack existence, ownership, and non-empty constraints
 */

import {
  validateProcessTerritory,
  validateEliminateStack,
} from '../../../src/shared/engine/validators/TerritoryValidator';
import { createTestBoard, createTestGameState, pos } from '../../utils/fixtures';
import type { GameState, Territory, RingStack } from '../../../src/shared/types/game';
import type {
  ProcessTerritoryAction,
  EliminateStackAction,
} from '../../../src/shared/engine/types';

describe('TerritoryValidator', () => {
  describe('validateProcessTerritory', () => {
    let state: GameState;

    beforeEach(() => {
      const board = createTestBoard('square8');
      const region: Territory = {
        spaces: [pos(0, 0)],
        controllingPlayer: 1,
        isDisconnected: true,
      };
      // Region id is a string; use "region-1" for clarity.
      board.territories.set('region-1', region);

      state = createTestGameState({
        board,
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
    });

    it('rejects when not in territory_processing phase', () => {
      state.currentPhase = 'movement';
      const action: ProcessTerritoryAction = {
        type: 'process_territory_region',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = validateProcessTerritory(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects when it is not the acting player turn', () => {
      const action: ProcessTerritoryAction = {
        type: 'process_territory_region',
        playerId: 2,
        regionId: 'region-1',
      };

      const result = validateProcessTerritory(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('rejects when the referenced region does not exist', () => {
      const action: ProcessTerritoryAction = {
        type: 'process_territory_region',
        playerId: 1,
        regionId: 'missing-region',
      };

      const result = validateProcessTerritory(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('REGION_NOT_FOUND');
    });

    it('rejects when the region is not marked disconnected', () => {
      const region = state.board.territories.get('region-1');
      if (!region) {
        throw new Error('expected region-1 to exist');
      }
      region.isDisconnected = false;

      const action: ProcessTerritoryAction = {
        type: 'process_territory_region',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = validateProcessTerritory(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('REGION_NOT_DISCONNECTED');
    });

    it('accepts a valid territory processing action', () => {
      const action: ProcessTerritoryAction = {
        type: 'process_territory_region',
        playerId: 1,
        regionId: 'region-1',
      };

      const result = validateProcessTerritory(state, action);
      expect(result.valid).toBe(true);
    });
  });

  describe('validateEliminateStack', () => {
    let state: GameState;

    beforeEach(() => {
      const board = createTestBoard('square8');
      const stack: RingStack = {
        position: pos(3, 3),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      };
      board.stacks.set('3,3', stack);

      state = createTestGameState({
        board,
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
    });

    it('rejects elimination when not in territory_processing phase', () => {
      state.currentPhase = 'movement';
      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(3, 3),
      };

      const result = validateEliminateStack(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects elimination when it is not the acting player turn', () => {
      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 2,
        stackPosition: pos(3, 3),
      };

      const result = validateEliminateStack(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('rejects elimination when the stack does not exist', () => {
      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(0, 0),
      };

      const result = validateEliminateStack(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('STACK_NOT_FOUND');
    });

    it('rejects elimination when the stack is not controlled by the player', () => {
      const stack = state.board.stacks.get('3,3');
      if (!stack) {
        throw new Error('expected stack at 3,3');
      }
      stack.controllingPlayer = 2;

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(3, 3),
      };

      const result = validateEliminateStack(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_STACK');
    });

    it('rejects elimination when the stack is empty', () => {
      const stack = state.board.stacks.get('3,3');
      if (!stack) {
        throw new Error('expected stack at 3,3');
      }
      stack.stackHeight = 0;
      stack.rings = [];

      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(3, 3),
      };

      const result = validateEliminateStack(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('EMPTY_STACK');
    });

    it('accepts a valid elimination action', () => {
      const action: EliminateStackAction = {
        type: 'eliminate_stack',
        playerId: 1,
        stackPosition: pos(3, 3),
      };

      const result = validateEliminateStack(state, action);
      expect(result.valid).toBe(true);
    });
  });
});
