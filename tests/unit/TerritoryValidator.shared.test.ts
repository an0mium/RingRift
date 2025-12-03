/**
 * TerritoryValidator branch coverage tests
 * Tests for src/shared/engine/validators/TerritoryValidator.ts
 *
 * These tests exercise the minimal structural contract for territory
 * decisions (phase/turn/region/stack checks). Full territory discovery
 * and scoring semantics are covered by the suites listed under
 * "5. Territory disconnection & chain reactions" in RULES_SCENARIO_MATRIX.md.
 */

import {
  validateProcessTerritory,
  validateEliminateStack,
} from '@shared/engine/validators/TerritoryValidator';
import type { GameState, ProcessTerritoryAction, EliminateStackAction } from '@shared/engine/types';

// Helper to create minimal GameState for territory validation tests
function createMinimalState(
  overrides: Partial<{
    currentPhase: string;
    currentPlayer: number;
    territories: Map<string, { controllingPlayer: number | null; isDisconnected?: boolean }>;
    stacks: Map<string, { controllingPlayer: number; stackHeight: number }>;
  }>
): GameState {
  const base = {
    board: {
      territories: overrides.territories ?? new Map(),
      stacks: overrides.stacks ?? new Map(),
      markers: new Map(),
      rings: new Map(),
      geometry: { type: 'square' as const, size: 8 },
    },
    currentPhase: overrides.currentPhase ?? 'territory_processing',
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

describe('TerritoryValidator', () => {
  describe('validateProcessTerritory', () => {
    it('returns error for invalid phase (not territory_processing)', () => {
      const state = createMinimalState({ currentPhase: 'placement' });
      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region_1',
        playerId: 1,
      };

      const result = validateProcessTerritory(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('phase');
    });

    it('returns error when player is not current player', () => {
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 2,
      });
      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region_1',
        playerId: 1, // wrong player
      };

      const result = validateProcessTerritory(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('turn');
    });

    it('returns error when region does not exist', () => {
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        territories: new Map(), // empty - no regions
      });
      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'nonexistent_region',
        playerId: 1,
      };

      const result = validateProcessTerritory(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('not found');
    });

    it('returns error when region is not disconnected', () => {
      const territories = new Map([['region_1', { controllingPlayer: 1, isDisconnected: false }]]);
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        territories,
      });
      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region_1',
        playerId: 1,
      };

      const result = validateProcessTerritory(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('disconnected');
    });

    it('returns valid when all conditions are met', () => {
      const territories = new Map([['region_1', { controllingPlayer: 1, isDisconnected: true }]]);
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        territories,
      });
      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region_1',
        playerId: 1,
      };

      const result = validateProcessTerritory(state, action);

      expect(result.valid).toBe(true);
    });

    it('accepts processing of a disconnected neutral region (no controlling player)', () => {
      const territories = new Map([
        ['region_1', { controllingPlayer: null, isDisconnected: true }],
      ]);
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        territories,
      });
      const action: ProcessTerritoryAction = {
        type: 'processTerritory',
        regionId: 'region_1',
        playerId: 1,
      };

      const result = validateProcessTerritory(state, action);

      expect(result.valid).toBe(true);
    });
  });

  describe('validateEliminateStack', () => {
    it('returns error for invalid phase', () => {
      const state = createMinimalState({ currentPhase: 'placement' });
      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 0, y: 0 },
        playerId: 1,
      };

      const result = validateEliminateStack(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('phase');
    });

    it('returns error when player is not current player', () => {
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 2,
      });
      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 0, y: 0 },
        playerId: 1,
      };

      const result = validateEliminateStack(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('turn');
    });

    it('returns error when stack does not exist at position', () => {
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        stacks: new Map(), // no stacks
      });
      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 0, y: 0 },
        playerId: 1,
      };

      const result = validateEliminateStack(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('not found');
    });

    it('returns error when stack is not controlled by player', () => {
      const stacks = new Map([
        ['0,0', { controllingPlayer: 2, stackHeight: 1 }], // owned by player 2, not player 1
      ]);
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        stacks,
      });
      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 0, y: 0 },
        playerId: 1,
      };

      const result = validateEliminateStack(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('not controlled');
    });

    it('returns error when stack height is zero', () => {
      const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 0 }]]);
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        stacks,
      });
      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 0, y: 0 },
        playerId: 1,
      };

      const result = validateEliminateStack(state, action);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('empty');
    });

    it('returns valid when all conditions are met', () => {
      const stacks = new Map([['0,0', { controllingPlayer: 1, stackHeight: 1 }]]);
      const state = createMinimalState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        stacks,
      });
      const action: EliminateStackAction = {
        type: 'eliminateStack',
        stackPosition: { x: 0, y: 0 },
        playerId: 1,
      };

      const result = validateEliminateStack(state, action);

      expect(result.valid).toBe(true);
    });
  });
});
