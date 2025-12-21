/**
 * Tests for statePersistence utilities.
 */

import {
  saveCurrentGameState,
  exportScenarioToFile,
  importScenarioFromFile,
} from '../../../src/client/sandbox/statePersistence';
import type { GameState } from '../../../src/shared/types/game';
import type { LoadableScenario } from '../../../src/client/sandbox/scenarioTypes';
import * as scenarioLoader from '../../../src/client/sandbox/scenarioLoader';
import * as serialization from '../../../src/shared/engine/contracts/serialization';

// Mock dependencies
jest.mock('../../../src/client/sandbox/scenarioLoader', () => ({
  saveCustomScenario: jest.fn(),
}));

jest.mock('../../../src/shared/engine/contracts/serialization', () => ({
  serializeGameState: jest.fn((state) => state),
}));

const mockSaveCustomScenario = scenarioLoader.saveCustomScenario as jest.Mock;
const mockSerializeGameState = serialization.serializeGameState as jest.Mock;

describe('statePersistence', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('saveCurrentGameState', () => {
    const createMockGameState = (overrides: Partial<GameState> = {}): GameState =>
      ({
        id: 'test-game',
        boardType: 'square8',
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [{ id: 1 }, { id: 2 }],
        board: {
          cells: [],
          boardType: 'square8',
        },
        moveHistory: [],
        ...overrides,
      }) as GameState;

    it('should create scenario with provided name', () => {
      const gameState = createMockGameState();
      const scenario = saveCurrentGameState(gameState, { name: 'My Test Save' });

      expect(scenario.name).toBe('My Test Save');
    });

    it('should create scenario with provided description', () => {
      const gameState = createMockGameState();
      const scenario = saveCurrentGameState(gameState, {
        name: 'Test',
        description: 'Custom description',
      });

      expect(scenario.description).toBe('Custom description');
    });

    it('should generate default description when not provided', () => {
      const gameState = createMockGameState({ moveHistory: [{}, {}, {}] as any[] });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.description).toContain('turn 4');
      expect(scenario.description).toContain('ring_placement');
    });

    it('should infer category from phase - ring_placement', () => {
      const gameState = createMockGameState({ currentPhase: 'ring_placement' });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.category).toBe('placement');
    });

    it('should infer category from phase - movement', () => {
      const gameState = createMockGameState({ currentPhase: 'movement' });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.category).toBe('movement');
    });

    it('should infer category from phase - chain_capture', () => {
      const gameState = createMockGameState({ currentPhase: 'chain_capture' });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.category).toBe('chain_capture');
    });

    it('should infer category from phase - process_line', () => {
      const gameState = createMockGameState({ currentPhase: 'process_line' });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.category).toBe('line_processing');
    });

    it('should infer category from phase - choose_line_option', () => {
      const gameState = createMockGameState({ currentPhase: 'choose_line_option' });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.category).toBe('line_processing');
    });

    it('should infer category from phase - choose_territory_option', () => {
      const gameState = createMockGameState({ currentPhase: 'choose_territory_option' });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.category).toBe('territory_processing');
    });

    it('should use custom category when unknown phase', () => {
      const gameState = createMockGameState({ currentPhase: 'unknown_phase' as any });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.category).toBe('custom');
    });

    it('should use provided category override', () => {
      const gameState = createMockGameState();
      const scenario = saveCurrentGameState(gameState, {
        name: 'Test',
        category: 'capture',
      });

      expect(scenario.category).toBe('capture');
    });

    it('should generate unique ID', () => {
      const gameState = createMockGameState();
      const scenario1 = saveCurrentGameState(gameState, { name: 'Test 1' });
      const scenario2 = saveCurrentGameState(gameState, { name: 'Test 2' });

      expect(scenario1.id).not.toBe(scenario2.id);
      expect(scenario1.id).toMatch(/^custom_/);
    });

    it('should set correct boardType', () => {
      const gameState = createMockGameState({ boardType: 'hex11' });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.boardType).toBe('hex11');
    });

    it('should set correct playerCount', () => {
      const gameState = createMockGameState({
        players: [{ id: 1 }, { id: 2 }, { id: 3 }] as any[],
      });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.playerCount).toBe(3);
    });

    it('should set source as custom', () => {
      const gameState = createMockGameState();
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.source).toBe('custom');
    });

    it('should set createdAt timestamp', () => {
      const beforeTime = new Date().toISOString();
      const gameState = createMockGameState();
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });
      const afterTime = new Date().toISOString();

      expect(scenario.createdAt).toBeDefined();
      expect(scenario.createdAt! >= beforeTime).toBe(true);
      expect(scenario.createdAt! <= afterTime).toBe(true);
    });

    it('should call serializeGameState', () => {
      const gameState = createMockGameState();
      saveCurrentGameState(gameState, { name: 'Test' });

      expect(mockSerializeGameState).toHaveBeenCalledWith(gameState);
    });

    it('should call saveCustomScenario', () => {
      const gameState = createMockGameState();
      saveCurrentGameState(gameState, { name: 'Test' });

      expect(mockSaveCustomScenario).toHaveBeenCalled();
    });

    it('should include default tags', () => {
      const gameState = createMockGameState({ currentPhase: 'movement' });
      const scenario = saveCurrentGameState(gameState, { name: 'Test' });

      expect(scenario.tags).toContain('saved');
      expect(scenario.tags).toContain('movement');
    });

    it('should use provided tags', () => {
      const gameState = createMockGameState();
      const scenario = saveCurrentGameState(gameState, {
        name: 'Test',
        tags: ['custom-tag', 'another-tag'],
      });

      expect(scenario.tags).toEqual(['custom-tag', 'another-tag']);
    });
  });

  describe('exportScenarioToFile', () => {
    let mockCreateElement: jest.SpyInstance;
    let mockAppendChild: jest.SpyInstance;
    let mockRemoveChild: jest.SpyInstance;
    let mockAnchorClick: jest.Mock;
    let originalCreateObjectURL: typeof URL.createObjectURL;
    let originalRevokeObjectURL: typeof URL.revokeObjectURL;

    beforeEach(() => {
      mockAnchorClick = jest.fn();
      mockCreateElement = jest.spyOn(document, 'createElement').mockReturnValue({
        href: '',
        download: '',
        click: mockAnchorClick,
      } as unknown as HTMLAnchorElement);
      mockAppendChild = jest.spyOn(document.body, 'appendChild').mockImplementation();
      mockRemoveChild = jest.spyOn(document.body, 'removeChild').mockImplementation();

      // Mock URL methods globally since they don't exist in jsdom
      originalCreateObjectURL = URL.createObjectURL;
      originalRevokeObjectURL = URL.revokeObjectURL;
      URL.createObjectURL = jest.fn().mockReturnValue('blob:test-url');
      URL.revokeObjectURL = jest.fn();
    });

    afterEach(() => {
      mockCreateElement.mockRestore();
      mockAppendChild.mockRestore();
      mockRemoveChild.mockRestore();
      URL.createObjectURL = originalCreateObjectURL;
      URL.revokeObjectURL = originalRevokeObjectURL;
    });

    const createMockScenario = (): LoadableScenario => ({
      id: 'test-scenario',
      name: 'Test Scenario Name',
      description: 'Test description',
      category: 'custom',
      tags: [],
      boardType: 'square8',
      playerCount: 2,
      state: {} as any,
    });

    it('should create download link with correct filename', () => {
      const scenario = createMockScenario();
      exportScenarioToFile(scenario);

      const anchor = mockCreateElement.mock.results[0].value;
      expect(anchor.download).toBe('ringrift_scenario_test_scenario_name.json');
    });

    it('should sanitize filename - remove special characters', () => {
      const scenario = createMockScenario();
      scenario.name = 'Test!@#$%Name';
      exportScenarioToFile(scenario);

      const anchor = mockCreateElement.mock.results[0].value;
      expect(anchor.download).toBe('ringrift_scenario_test_name.json');
    });

    it('should create Blob with JSON content', () => {
      const scenario = createMockScenario();
      exportScenarioToFile(scenario);

      expect(URL.createObjectURL).toHaveBeenCalledWith(expect.any(Blob));
    });

    it('should trigger download click', () => {
      const scenario = createMockScenario();
      exportScenarioToFile(scenario);

      expect(mockAnchorClick).toHaveBeenCalled();
    });

    it('should cleanup after download', () => {
      const scenario = createMockScenario();
      exportScenarioToFile(scenario);

      expect(mockRemoveChild).toHaveBeenCalled();
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-url');
    });
  });

  describe('importScenarioFromFile', () => {
    const createMockFile = (content: string): File => {
      const file = new File([content], 'test.json', { type: 'application/json' });
      // Mock text() method since it's not available in jsdom
      file.text = jest.fn().mockResolvedValue(content);
      return file;
    };

    const validScenarioData = {
      id: 'imported-1',
      name: 'Imported Scenario',
      description: 'Test import',
      category: 'custom',
      boardType: 'square8',
      state: {
        board: { cells: [] },
        players: [{ id: 1 }, { id: 2 }],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
      },
    };

    it('should import valid scenario file', async () => {
      const file = createMockFile(JSON.stringify(validScenarioData));
      const scenario = await importScenarioFromFile(file);

      expect(scenario.name).toBe('Imported Scenario');
      expect(scenario.description).toBe('Test import');
    });

    it('should generate new ID with imported prefix', async () => {
      const file = createMockFile(JSON.stringify(validScenarioData));
      const scenario = await importScenarioFromFile(file);

      expect(scenario.id).toMatch(/^imported_/);
    });

    it('should set source as custom', async () => {
      const file = createMockFile(JSON.stringify(validScenarioData));
      const scenario = await importScenarioFromFile(file);

      expect(scenario.source).toBe('custom');
    });

    it('should throw on invalid JSON', async () => {
      const file = createMockFile('not valid json {{');

      await expect(importScenarioFromFile(file)).rejects.toThrow('Invalid JSON file');
    });

    it('should throw on missing id field', async () => {
      const data = { ...validScenarioData };
      delete (data as any).id;
      const file = createMockFile(JSON.stringify(data));

      await expect(importScenarioFromFile(file)).rejects.toThrow('Missing or invalid "id" field');
    });

    it('should throw on missing name field', async () => {
      const data = { ...validScenarioData };
      delete (data as any).name;
      const file = createMockFile(JSON.stringify(data));

      await expect(importScenarioFromFile(file)).rejects.toThrow('Missing or invalid "name" field');
    });

    it('should throw on missing state field', async () => {
      const data = { ...validScenarioData };
      delete (data as any).state;
      const file = createMockFile(JSON.stringify(data));

      await expect(importScenarioFromFile(file)).rejects.toThrow(
        'Missing or invalid "state" field'
      );
    });

    it('should throw on missing boardType field', async () => {
      const data = { ...validScenarioData };
      delete (data as any).boardType;
      const file = createMockFile(JSON.stringify(data));

      await expect(importScenarioFromFile(file)).rejects.toThrow(
        'Missing or invalid "boardType" field'
      );
    });

    it('should throw on missing state.board', async () => {
      const data = {
        ...validScenarioData,
        state: { ...validScenarioData.state, board: undefined },
      };
      const file = createMockFile(JSON.stringify(data));

      await expect(importScenarioFromFile(file)).rejects.toThrow('Invalid state: missing "board"');
    });

    it('should throw on missing state.players', async () => {
      const data = {
        ...validScenarioData,
        state: { ...validScenarioData.state, players: undefined },
      };
      const file = createMockFile(JSON.stringify(data));

      await expect(importScenarioFromFile(file)).rejects.toThrow(
        'Invalid state: missing "players"'
      );
    });

    it('should throw on missing state.currentPlayer', async () => {
      const data = {
        ...validScenarioData,
        state: { ...validScenarioData.state, currentPlayer: undefined },
      };
      const file = createMockFile(JSON.stringify(data));

      await expect(importScenarioFromFile(file)).rejects.toThrow(
        'Invalid state: missing "currentPlayer"'
      );
    });

    it('should throw on missing state.currentPhase', async () => {
      const data = {
        ...validScenarioData,
        state: { ...validScenarioData.state, currentPhase: undefined },
      };
      const file = createMockFile(JSON.stringify(data));

      await expect(importScenarioFromFile(file)).rejects.toThrow(
        'Invalid state: missing "currentPhase"'
      );
    });

    it('should handle missing optional fields', async () => {
      const minimalData = {
        id: 'minimal',
        name: 'Minimal',
        boardType: 'square8',
        state: validScenarioData.state,
      };
      const file = createMockFile(JSON.stringify(minimalData));
      const scenario = await importScenarioFromFile(file);

      expect(scenario.description).toBe('');
      expect(scenario.category).toBe('custom');
      expect(scenario.tags).toEqual([]);
      expect(scenario.playerCount).toBe(2); // default
    });

    it('should preserve tags if provided', async () => {
      const data = { ...validScenarioData, tags: ['tag1', 'tag2'] };
      const file = createMockFile(JSON.stringify(data));
      const scenario = await importScenarioFromFile(file);

      expect(scenario.tags).toEqual(['tag1', 'tag2']);
    });

    it('should throw on non-object input', async () => {
      const file = createMockFile(JSON.stringify('just a string'));

      await expect(importScenarioFromFile(file)).rejects.toThrow('not an object');
    });

    it('should throw on null input', async () => {
      const file = createMockFile(JSON.stringify(null));

      await expect(importScenarioFromFile(file)).rejects.toThrow('not an object');
    });
  });
});
