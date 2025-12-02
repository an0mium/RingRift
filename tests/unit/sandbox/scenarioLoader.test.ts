/**
 * Tests for scenarioLoader utilities.
 */

import {
  vectorToScenario,
  loadCustomScenarios,
  saveCustomScenario,
  deleteCustomScenario,
  filterScenarios,
} from '../../../src/client/sandbox/scenarioLoader';
import type { ContractTestVector } from '../../../src/client/sandbox/scenarioTypes';
import type { LoadableScenario } from '../../../src/client/sandbox/scenarioTypes';
import {
  CUSTOM_SCENARIOS_STORAGE_KEY,
  MAX_CUSTOM_SCENARIOS,
} from '../../../src/client/sandbox/scenarioTypes';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Mock fetch for async loading functions
global.fetch = jest.fn();

describe('scenarioLoader', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.clear();
  });

  describe('vectorToScenario', () => {
    const createMockVector = (id: string): ContractTestVector => ({
      id,
      description: 'Test vector description',
      category: 'placement',
      tags: ['test', 'unit'],
      createdAt: '2024-01-01T00:00:00.000Z',
      input: {
        state: {
          board: {
            type: 'square8',
            cells: [],
          },
          players: [{ id: 1 }, { id: 2 }],
          currentPlayer: 1,
          currentPhase: 'ring_placement',
        },
        move: { type: 'place_ring', to: { x: 0, y: 0 } },
      },
      expected: {
        valid: true,
      },
    });

    it('should convert vector to loadable scenario', () => {
      const vector = createMockVector('placement.initial.center');
      const scenario = vectorToScenario(vector);

      expect(scenario.id).toBe('placement.initial.center');
      expect(scenario.description).toBe('Test vector description');
      expect(scenario.category).toBe('placement');
      expect(scenario.boardType).toBe('square8');
      expect(scenario.playerCount).toBe(2);
      expect(scenario.source).toBe('vector');
    });

    it('should format vector name from ID', () => {
      const vector = createMockVector('capture.basic_capture.diagonal');
      const scenario = vectorToScenario(vector);

      expect(scenario.name).toBe('Capture: basic capture: diagonal');
    });

    it('should preserve tags from vector', () => {
      const vector = createMockVector('test.vector');
      const scenario = vectorToScenario(vector);

      expect(scenario.tags).toEqual(['test', 'unit']);
    });

    it('should include suggested move', () => {
      const vector = createMockVector('test.vector');
      const scenario = vectorToScenario(vector);

      expect(scenario.suggestedMove).toEqual({ type: 'place_ring', to: { x: 0, y: 0 } });
    });

    it('should handle empty tags', () => {
      const vector = createMockVector('test.vector');
      delete (vector as any).tags;
      const scenario = vectorToScenario(vector);

      expect(scenario.tags).toEqual([]);
    });
  });

  describe('loadCustomScenarios', () => {
    const createMockScenario = (id: string): LoadableScenario => ({
      id,
      name: `Scenario ${id}`,
      description: 'Test scenario',
      category: 'custom',
      tags: [],
      boardType: 'square8',
      playerCount: 2,
      state: {} as any,
    });

    it('should return empty array when no scenarios stored', () => {
      const scenarios = loadCustomScenarios();
      expect(scenarios).toEqual([]);
    });

    it('should return stored scenarios', () => {
      const mockScenarios = [createMockScenario('1'), createMockScenario('2')];
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(mockScenarios));

      const scenarios = loadCustomScenarios();

      expect(scenarios).toEqual(mockScenarios);
    });

    it('should handle invalid JSON gracefully', () => {
      localStorageMock.getItem.mockReturnValueOnce('invalid json {{{');

      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      const scenarios = loadCustomScenarios();

      expect(scenarios).toEqual([]);
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('should use correct storage key', () => {
      loadCustomScenarios();
      expect(localStorageMock.getItem).toHaveBeenCalledWith(CUSTOM_SCENARIOS_STORAGE_KEY);
    });
  });

  describe('saveCustomScenario', () => {
    const createMockScenario = (id: string): LoadableScenario => ({
      id,
      name: `Scenario ${id}`,
      description: 'Test scenario',
      category: 'custom',
      tags: [],
      boardType: 'square8',
      playerCount: 2,
      state: {} as any,
    });

    it('should save new scenario', () => {
      const scenario = createMockScenario('new-1');
      saveCustomScenario(scenario);

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        CUSTOM_SCENARIOS_STORAGE_KEY,
        expect.stringContaining('new-1')
      );
    });

    it('should add new scenario at front', () => {
      const existing = [createMockScenario('old-1')];
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(existing));

      const newScenario = createMockScenario('new-1');
      saveCustomScenario(newScenario);

      const savedArg = localStorageMock.setItem.mock.calls[0][1];
      const saved = JSON.parse(savedArg);
      expect(saved[0].id).toBe('new-1');
      expect(saved[1].id).toBe('old-1');
    });

    it('should replace scenario with same ID', () => {
      const existing = [createMockScenario('same-id')];
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(existing));

      const updated = createMockScenario('same-id');
      updated.name = 'Updated Name';
      saveCustomScenario(updated);

      const savedArg = localStorageMock.setItem.mock.calls[0][1];
      const saved = JSON.parse(savedArg);
      expect(saved).toHaveLength(1);
      expect(saved[0].name).toBe('Updated Name');
    });

    it('should limit to MAX_CUSTOM_SCENARIOS', () => {
      // Create max scenarios
      const existing = Array.from({ length: MAX_CUSTOM_SCENARIOS }, (_, i) =>
        createMockScenario(`existing-${i}`)
      );
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(existing));

      const newScenario = createMockScenario('new-one');
      saveCustomScenario(newScenario);

      const savedArg = localStorageMock.setItem.mock.calls[0][1];
      const saved = JSON.parse(savedArg);
      expect(saved).toHaveLength(MAX_CUSTOM_SCENARIOS);
      expect(saved[0].id).toBe('new-one');
    });

    it('should throw on localStorage error', () => {
      localStorageMock.setItem.mockImplementationOnce(() => {
        throw new Error('QuotaExceededError');
      });

      const scenario = createMockScenario('test');
      expect(() => saveCustomScenario(scenario)).toThrow('Failed to save scenario');
    });
  });

  describe('deleteCustomScenario', () => {
    const createMockScenario = (id: string): LoadableScenario => ({
      id,
      name: `Scenario ${id}`,
      description: 'Test scenario',
      category: 'custom',
      tags: [],
      boardType: 'square8',
      playerCount: 2,
      state: {} as any,
    });

    it('should delete scenario by ID', () => {
      const existing = [
        createMockScenario('keep-1'),
        createMockScenario('delete-me'),
        createMockScenario('keep-2'),
      ];
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(existing));

      deleteCustomScenario('delete-me');

      const savedArg = localStorageMock.setItem.mock.calls[0][1];
      const saved = JSON.parse(savedArg);
      expect(saved).toHaveLength(2);
      expect(saved.find((s: LoadableScenario) => s.id === 'delete-me')).toBeUndefined();
    });

    it('should handle deleting non-existent ID gracefully', () => {
      const existing = [createMockScenario('existing')];
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(existing));

      deleteCustomScenario('non-existent');

      const savedArg = localStorageMock.setItem.mock.calls[0][1];
      const saved = JSON.parse(savedArg);
      expect(saved).toHaveLength(1);
    });
  });

  describe('filterScenarios', () => {
    const scenarios: LoadableScenario[] = [
      {
        id: '1',
        name: 'Ring Placement Tutorial',
        description: 'Learn basic placement',
        category: 'placement',
        tags: ['beginner', 'tutorial'],
        boardType: 'square8',
        playerCount: 2,
        state: {} as any,
      },
      {
        id: '2',
        name: 'Capture Mechanics',
        description: 'Advanced capture strategies',
        category: 'capture',
        tags: ['advanced', 'capture'],
        boardType: 'square8',
        playerCount: 2,
        state: {} as any,
      },
      {
        id: '3',
        name: 'Movement Guide',
        description: 'How to move pieces',
        category: 'movement',
        tags: ['beginner', 'guide'],
        boardType: 'hex11',
        playerCount: 2,
        state: {} as any,
      },
    ];

    it('should return all scenarios with no filters', () => {
      const result = filterScenarios(scenarios, {});
      expect(result).toHaveLength(3);
    });

    it('should filter by category', () => {
      const result = filterScenarios(scenarios, { category: 'placement' });
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Ring Placement Tutorial');
    });

    it('should return all with category "all"', () => {
      const result = filterScenarios(scenarios, { category: 'all' });
      expect(result).toHaveLength(3);
    });

    it('should filter by boardType', () => {
      const result = filterScenarios(scenarios, { boardType: 'hex11' });
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Movement Guide');
    });

    it('should return all with boardType "all"', () => {
      const result = filterScenarios(scenarios, { boardType: 'all' });
      expect(result).toHaveLength(3);
    });

    it('should filter by search query matching name', () => {
      const result = filterScenarios(scenarios, { searchQuery: 'ring' });
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Ring Placement Tutorial');
    });

    it('should filter by search query matching description', () => {
      const result = filterScenarios(scenarios, { searchQuery: 'strategies' });
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Capture Mechanics');
    });

    it('should filter by search query matching tags', () => {
      const result = filterScenarios(scenarios, { searchQuery: 'tutorial' });
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Ring Placement Tutorial');
    });

    it('should be case-insensitive for search', () => {
      const result = filterScenarios(scenarios, { searchQuery: 'CAPTURE' });
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Capture Mechanics');
    });

    it('should combine multiple filters', () => {
      const result = filterScenarios(scenarios, {
        category: 'placement',
        boardType: 'square8',
        searchQuery: 'tutorial',
      });
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Ring Placement Tutorial');
    });

    it('should return empty array when no matches', () => {
      const result = filterScenarios(scenarios, { searchQuery: 'nonexistent' });
      expect(result).toHaveLength(0);
    });
  });
});
