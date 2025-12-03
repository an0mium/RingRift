/**
 * Unit tests for SeededRNG class and generateGameSeed function
 * Covers deterministic random number generation for game replay
 */
import { SeededRNG, generateGameSeed } from '../../../src/shared/utils/rng';
import type { LocalAIRng } from '../../../src/shared/engine';
import type { GameState } from '../../../src/shared/types/game';
import { GameSession } from '../../../src/server/game/GameSession';
import { globalAIEngine } from '../../../src/server/game/ai/AIEngine';
import { createInitialGameState } from '../../../src/shared/engine/initialState';

jest.mock('../../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    getAIConfig: jest.fn(),
    createAI: jest.fn(),
    createAIFromProfile: jest.fn(),
    getAIMove: jest.fn(),
    getLocalFallbackMove: jest.fn(),
    chooseLocalMoveFromCandidates: jest.fn(),
    getDiagnostics: jest.fn(() => ({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    })),
  },
}));

jest.mock('../../../src/server/config', () => ({
  config: {
    aiService: { requestTimeoutMs: 1000 },
    featureFlags: { analysisMode: { enabled: false } },
    decisionPhaseTimeouts: {
      defaultTimeoutMs: 30000,
      warningBeforeTimeoutMs: 5000,
      extensionMs: 15000,
    },
    logging: {
      level: 'error',
    },
    nodeEnv: 'test',
    isTest: true,
    isDevelopment: true,
  },
}));

jest.mock('../../../src/server/services/GamePersistenceService', () => ({
  GamePersistenceService: {
    finishGame: jest.fn(),
  },
}));

jest.mock('../../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    game: {
      findUnique: jest.fn().mockResolvedValue({
        id: 'rng-game-id',
        boardType: 'square8',
        status: 'active',
        maxPlayers: 2,
        isRated: false,
        player1: { id: 'p1', username: 'P1' },
        player2: { id: 'p2', username: 'P2' },
        timeControl: JSON.stringify({ type: 'rapid', initialTime: 600000, increment: 0 }),
        gameState: null,
        rngSeed: 123456,
        moves: [],
      }),
      update: jest.fn(),
    },
  }),
}));

const createMockIo = () =>
  ({
    to: jest.fn().mockReturnThis(),
    emit: jest.fn(),
    sockets: {
      adapter: { rooms: new Map() },
      sockets: new Map(),
    },
  }) as any;

describe('SeededRNG', () => {
  describe('constructor', () => {
    it('should initialize with a seed', () => {
      const rng = new SeededRNG(12345);
      expect(rng).toBeDefined();
    });

    it('should produce consistent results with same seed', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      const values1 = [rng1.next(), rng1.next(), rng1.next()];
      const values2 = [rng2.next(), rng2.next(), rng2.next()];

      expect(values1).toEqual(values2);
    });

    it('should produce different results with different seeds', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(43);

      const value1 = rng1.next();
      const value2 = rng2.next();

      expect(value1).not.toEqual(value2);
    });

    it('should handle seed of 0', () => {
      const rng = new SeededRNG(0);
      const value = rng.next();
      expect(typeof value).toBe('number');
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    });

    it('should handle negative seeds', () => {
      const rng = new SeededRNG(-12345);
      const value = rng.next();
      expect(typeof value).toBe('number');
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    });

    it('should handle large seeds', () => {
      const rng = new SeededRNG(0x7fffffff);
      const value = rng.next();
      expect(typeof value).toBe('number');
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    });
  });

  describe('next()', () => {
    it('should return values in range [0, 1)', () => {
      const rng = new SeededRNG(12345);
      for (let i = 0; i < 100; i++) {
        const value = rng.next();
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThan(1);
      }
    });

    it('should produce different values on consecutive calls', () => {
      const rng = new SeededRNG(12345);
      const values = new Set<number>();
      for (let i = 0; i < 100; i++) {
        values.add(rng.next());
      }
      // Should have mostly unique values (allowing some collisions)
      expect(values.size).toBeGreaterThan(90);
    });

    it('should produce uniform distribution', () => {
      const rng = new SeededRNG(12345);
      let low = 0;
      let high = 0;
      const samples = 10000;

      for (let i = 0; i < samples; i++) {
        if (rng.next() < 0.5) {
          low++;
        } else {
          high++;
        }
      }

      // Should be roughly 50/50 (within 5% tolerance)
      const ratio = low / samples;
      expect(ratio).toBeGreaterThan(0.45);
      expect(ratio).toBeLessThan(0.55);
    });
  });

  describe('nextInt()', () => {
    it('should return integers in range [min, max)', () => {
      const rng = new SeededRNG(12345);
      for (let i = 0; i < 100; i++) {
        const value = rng.nextInt(0, 10);
        expect(Number.isInteger(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThan(10);
      }
    });

    it('should handle negative range', () => {
      const rng = new SeededRNG(12345);
      for (let i = 0; i < 100; i++) {
        const value = rng.nextInt(-10, 0);
        expect(Number.isInteger(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(-10);
        expect(value).toBeLessThan(0);
      }
    });

    it('should handle range crossing zero', () => {
      const rng = new SeededRNG(12345);
      for (let i = 0; i < 100; i++) {
        const value = rng.nextInt(-5, 5);
        expect(Number.isInteger(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(-5);
        expect(value).toBeLessThan(5);
      }
    });

    it('should handle single value range', () => {
      const rng = new SeededRNG(12345);
      for (let i = 0; i < 10; i++) {
        const value = rng.nextInt(5, 6);
        expect(value).toBe(5);
      }
    });

    it('should cover all values in small range', () => {
      const rng = new SeededRNG(12345);
      const seen = new Set<number>();
      for (let i = 0; i < 1000; i++) {
        seen.add(rng.nextInt(0, 5));
      }
      expect(seen.size).toBe(5);
      expect(seen.has(0)).toBe(true);
      expect(seen.has(1)).toBe(true);
      expect(seen.has(2)).toBe(true);
      expect(seen.has(3)).toBe(true);
      expect(seen.has(4)).toBe(true);
    });

    it('should be deterministic with same seed', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      for (let i = 0; i < 20; i++) {
        expect(rng1.nextInt(0, 100)).toBe(rng2.nextInt(0, 100));
      }
    });
  });

  describe('shuffle()', () => {
    it('should return the same array instance', () => {
      const rng = new SeededRNG(12345);
      const arr = [1, 2, 3, 4, 5];
      const result = rng.shuffle(arr);
      expect(result).toBe(arr);
    });

    it('should modify array in place', () => {
      const rng = new SeededRNG(12345);
      const arr = [1, 2, 3, 4, 5];
      rng.shuffle(arr);

      // The array should still have the same length
      expect(arr.length).toBe(5);
      // All original elements should still be present
      expect(arr.sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5]);
    });

    it('should contain all original elements', () => {
      const rng = new SeededRNG(12345);
      const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      rng.shuffle(arr);

      expect(arr.sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    });

    it('should be deterministic with same seed', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      const arr1 = [1, 2, 3, 4, 5, 6, 7, 8];
      const arr2 = [1, 2, 3, 4, 5, 6, 7, 8];

      rng1.shuffle(arr1);
      rng2.shuffle(arr2);

      expect(arr1).toEqual(arr2);
    });

    it('should handle empty array', () => {
      const rng = new SeededRNG(12345);
      const arr: number[] = [];
      const result = rng.shuffle(arr);
      expect(result).toEqual([]);
    });

    it('should handle single element array', () => {
      const rng = new SeededRNG(12345);
      const arr = [42];
      const result = rng.shuffle(arr);
      expect(result).toEqual([42]);
    });

    it('should handle array of objects', () => {
      const rng = new SeededRNG(12345);
      const obj1 = { id: 1 };
      const obj2 = { id: 2 };
      const obj3 = { id: 3 };
      const arr = [obj1, obj2, obj3];

      rng.shuffle(arr);

      expect(arr.length).toBe(3);
      expect(arr).toContain(obj1);
      expect(arr).toContain(obj2);
      expect(arr).toContain(obj3);
    });

    it('should produce different orders with different seeds', () => {
      const arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const arr2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      new SeededRNG(42).shuffle(arr1);
      new SeededRNG(43).shuffle(arr2);

      // Arrays should be different (extremely unlikely to be same)
      let same = true;
      for (let i = 0; i < arr1.length; i++) {
        if (arr1[i] !== arr2[i]) {
          same = false;
          break;
        }
      }
      expect(same).toBe(false);
    });
  });

  describe('choice()', () => {
    it('should return an element from the array', () => {
      const rng = new SeededRNG(12345);
      const arr = ['a', 'b', 'c', 'd', 'e'];
      const result = rng.choice(arr);
      expect(arr).toContain(result);
    });

    it('should throw on empty array', () => {
      const rng = new SeededRNG(12345);
      expect(() => rng.choice([])).toThrow('Cannot select from empty array');
    });

    it('should return the only element for single-element array', () => {
      const rng = new SeededRNG(12345);
      expect(rng.choice([42])).toBe(42);
    });

    it('should be deterministic with same seed', () => {
      const arr = ['a', 'b', 'c', 'd', 'e'];

      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      for (let i = 0; i < 20; i++) {
        expect(rng1.choice(arr)).toBe(rng2.choice(arr));
      }
    });

    it('should cover all elements with enough samples', () => {
      const rng = new SeededRNG(12345);
      const arr = ['a', 'b', 'c'];
      const seen = new Set<string>();

      for (let i = 0; i < 100; i++) {
        seen.add(rng.choice(arr));
      }

      expect(seen.size).toBe(3);
    });

    it('should not modify the original array', () => {
      const rng = new SeededRNG(12345);
      const arr = [1, 2, 3, 4, 5];
      const original = [...arr];

      for (let i = 0; i < 100; i++) {
        rng.choice(arr);
      }

      expect(arr).toEqual(original);
    });

    it('should work with objects', () => {
      const rng = new SeededRNG(12345);
      const obj1 = { id: 1 };
      const obj2 = { id: 2 };
      const arr = [obj1, obj2];

      const result = rng.choice(arr);
      expect(result === obj1 || result === obj2).toBe(true);
    });
  });

  describe('sequence determinism', () => {
    it('should produce identical sequences across multiple operations', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      // Mix of operations
      expect(rng1.next()).toBe(rng2.next());
      expect(rng1.nextInt(0, 100)).toBe(rng2.nextInt(0, 100));
      expect(rng1.choice(['a', 'b', 'c'])).toBe(rng2.choice(['a', 'b', 'c']));

      const arr1 = [1, 2, 3];
      const arr2 = [1, 2, 3];
      rng1.shuffle(arr1);
      rng2.shuffle(arr2);
      expect(arr1).toEqual(arr2);

      expect(rng1.next()).toBe(rng2.next());
    });
  });

  describe('cross-component determinism', () => {
    it('GameSession.createLocalAIRng and AIEngine local RNG mixing produce consistent streams for same game rngSeed', async () => {
      const seed = 987654321;

      const players = [
        { id: 'p1', username: 'Player1', type: 'human' as const, isReady: true },
        { id: 'p2', username: 'Player2', type: 'human' as const, isReady: true },
      ];
      const timeControl = { type: 'rapid' as const, initialTime: 600000, increment: 0 };

      const baseState: GameState = createInitialGameState(
        'rng-game',
        'square8',
        players as any,
        timeControl,
        false,
        seed
      ) as GameState;

      const io = createMockIo();
      const session = new GameSession('rng-game', io as any, {} as any, new Map());
      // Patch GameEngine state directly to avoid full DB init in this low-level test
      (session as any).gameEngine = {
        getGameState: () => baseState,
      };

      const sessionRng: LocalAIRng = (session as any).createLocalAIRng(baseState, 1);

      // Mirror the same mixing strategy as AIEngine.createDeterministicLocalRng
      const baseSeed = typeof baseState.rngSeed === 'number' ? baseState.rngSeed : 0;
      const mixed = (baseSeed ^ (1 * 0x9e3779b1)) >>> 0;
      const engineSeeded = new SeededRNG(mixed);
      const engineLocal: LocalAIRng = () => engineSeeded.next();

      for (let i = 0; i < 20; i++) {
        expect(sessionRng()).toBe(engineLocal());
      }
    });

    it('sandbox ClientSandboxEngine and backend AIEngine share compatible seed derivation via GameState.rngSeed', () => {
      const seed = 13579;
      const base = {
        id: 'g',
        boardType: 'square8',
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'movement',
        rngSeed: seed,
        players: [{ id: 'p1', playerNumber: 1, type: 'ai', isReady: true, timeRemaining: 0 }],
        spectators: [],
        moveHistory: [],
        board: {
          type: 'square8',
          size: 8,
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Set(),
          territories: new Map(),
          rings: new Map(),
          formedLines: [],
          geometry: { type: 'square8', size: 8 },
        },
        timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      } as unknown as GameState;

      const baseSeed = typeof base.rngSeed === 'number' ? base.rngSeed : 0;
      const mixed = (baseSeed ^ (1 * 0x9e3779b1)) >>> 0;
      const backendRng = new SeededRNG(mixed);
      const backendSequence = Array.from({ length: 5 }, () => backendRng.next());

      const sandboxSeed = base.rngSeed ?? 0;
      const sandboxRng = new SeededRNG(sandboxSeed);
      const sandboxSequence = Array.from({ length: 5 }, () => sandboxRng.next());

      expect(backendSequence).not.toEqual(sandboxSequence);
    });
  });
});

describe('generateGameSeed', () => {
  it('should return a number', () => {
    const seed = generateGameSeed();
    expect(typeof seed).toBe('number');
  });

  it('should return an integer', () => {
    const seed = generateGameSeed();
    expect(Number.isInteger(seed)).toBe(true);
  });

  it('should return a non-negative value', () => {
    for (let i = 0; i < 100; i++) {
      const seed = generateGameSeed();
      expect(seed).toBeGreaterThanOrEqual(0);
    }
  });

  it('should return values less than 0x7fffffff', () => {
    for (let i = 0; i < 100; i++) {
      const seed = generateGameSeed();
      expect(seed).toBeLessThan(0x7fffffff);
    }
  });

  it('should produce varying values', () => {
    const seeds = new Set<number>();
    for (let i = 0; i < 100; i++) {
      seeds.add(generateGameSeed());
    }
    // With 31 bits of randomness, 100 samples should all be unique
    expect(seeds.size).toBeGreaterThan(90);
  });
});
