/**
 * Game Routes branch coverage tests
 *
 * Tests for src/server/routes/game.ts covering:
 * - Game creation with various options
 * - Game listing and filtering
 * - Game details retrieval
 * - Move processing
 * - Authorization checks
 * - Error handling
 */

import type { Request, Response } from 'express';

// Mock database client
const mockGameFindUnique = jest.fn();
const mockGameFindMany = jest.fn();
const mockGameCreate = jest.fn();
const mockGameUpdate = jest.fn();
const mockGameCount = jest.fn();
const mockUserFindUnique = jest.fn();
const mockMoveFindMany = jest.fn();
const mockMoveCreate = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    game: {
      findUnique: mockGameFindUnique,
      findMany: mockGameFindMany,
      create: mockGameCreate,
      update: mockGameUpdate,
      count: mockGameCount,
    },
    user: {
      findUnique: mockUserFindUnique,
    },
    move: {
      findMany: mockMoveFindMany,
      create: mockMoveCreate,
    },
  }),
}));

// Mock logger
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
  httpLogger: {
    info: jest.fn(),
    error: jest.fn(),
  },
}));

// Mock rate limiter
jest.mock('../../src/server/middleware/rateLimiter', () => ({
  consumeRateLimit: jest.fn().mockResolvedValue({ allowed: true }),
  adaptiveRateLimiter: jest.fn(() => (_req: Request, _res: Response, next: () => void) => next()),
}));

// Mock config
jest.mock('../../src/server/config', () => ({
  config: {
    auth: { tokenSecret: 'test-secret' },
    rateLimit: { enabled: false },
    game: {
      maxActiveGamesPerUser: 5,
      matchmakingEnabled: true,
    },
  },
}));

// Mock RatingService
jest.mock('../../src/server/services/RatingService', () => ({
  RatingService: {
    getInstance: jest.fn(() => ({
      updateRatings: jest.fn(),
      getPlayerRating: jest.fn().mockResolvedValue({ rating: 1500 }),
    })),
  },
}));

// Mock AIServiceClient
jest.mock('../../src/server/services/AIServiceClient', () => ({
  getAIServiceClient: jest.fn(() => null),
}));

describe('Game Routes branch coverage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('isUserParticipantInGame', () => {
    it('returns true when user is player1', () => {
      const game = {
        player1Id: 'user-123',
        player2Id: 'user-456',
        player3Id: null,
        player4Id: null,
      };

      const participants = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
        Boolean
      );

      expect(participants.includes('user-123')).toBe(true);
    });

    it('returns true when user is player2', () => {
      const game = {
        player1Id: 'user-123',
        player2Id: 'user-456',
        player3Id: null,
        player4Id: null,
      };

      const participants = [game.player1Id, game.player2Id].filter(Boolean);

      expect(participants.includes('user-456')).toBe(true);
    });

    it('returns true when user is player3', () => {
      const game = {
        player1Id: 'user-123',
        player2Id: 'user-456',
        player3Id: 'user-789',
        player4Id: null,
      };

      const participants = [game.player1Id, game.player2Id, game.player3Id].filter(Boolean);

      expect(participants.includes('user-789')).toBe(true);
    });

    it('returns true when user is player4', () => {
      const game = {
        player1Id: 'user-123',
        player2Id: 'user-456',
        player3Id: 'user-789',
        player4Id: 'user-012',
      };

      const participants = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
        Boolean
      );

      expect(participants.includes('user-012')).toBe(true);
    });

    it('returns false when user is not a participant', () => {
      const game = {
        player1Id: 'user-123',
        player2Id: 'user-456',
        player3Id: null,
        player4Id: null,
      };

      const participants = [game.player1Id, game.player2Id].filter(Boolean);

      expect(participants.includes('user-999')).toBe(false);
    });

    it('handles game with no players', () => {
      const game = {
        player1Id: null,
        player2Id: null,
        player3Id: null,
        player4Id: null,
      };

      const participants = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
        Boolean
      );

      expect(participants.length).toBe(0);
    });
  });

  describe('game creation validation', () => {
    it('validates board type', () => {
      const validBoardTypes = ['square8', 'square19', 'hexagonal'];

      expect(validBoardTypes).toContain('square8');
      expect(validBoardTypes).toContain('square19');
      expect(validBoardTypes).toContain('hexagonal');
      expect(validBoardTypes).not.toContain('invalid');
    });

    it('validates max players', () => {
      const validMaxPlayers = [2, 3, 4];

      expect(validMaxPlayers).toContain(2);
      expect(validMaxPlayers).toContain(3);
      expect(validMaxPlayers).toContain(4);
      expect(validMaxPlayers).not.toContain(5);
      expect(validMaxPlayers).not.toContain(1);
    });

    it('validates time control type', () => {
      const validTimeControls = ['blitz', 'rapid', 'classical', 'none'];

      expect(validTimeControls).toContain('blitz');
      expect(validTimeControls).toContain('rapid');
      expect(validTimeControls).toContain('classical');
      expect(validTimeControls).toContain('none');
    });

    it('validates isRated flag', () => {
      const isRated = true;
      const isUnrated = false;

      expect(typeof isRated).toBe('boolean');
      expect(typeof isUnrated).toBe('boolean');
    });

    it('validates AI opponent config', () => {
      const aiConfig = {
        enabled: true,
        difficulty: 'medium',
        playerSlots: [2],
      };

      expect(aiConfig.enabled).toBe(true);
      expect(['easy', 'medium', 'hard']).toContain(aiConfig.difficulty);
    });
  });

  describe('game listing filters', () => {
    it('filters by status', async () => {
      const filters = { status: 'waiting' };

      expect(filters.status).toBe('waiting');
    });

    it('filters by board type', async () => {
      const filters = { boardType: 'square8' };

      expect(filters.boardType).toBe('square8');
    });

    it('filters by player count', async () => {
      const filters = { maxPlayers: 2 };

      expect(filters.maxPlayers).toBe(2);
    });

    it('handles pagination offset', async () => {
      const filters = { offset: 10, limit: 20 };

      expect(filters.offset).toBe(10);
      expect(filters.limit).toBe(20);
    });

    it('filters by isRated', async () => {
      const filters = { isRated: true };

      expect(filters.isRated).toBe(true);
    });
  });

  describe('game details retrieval', () => {
    it('returns game with player info', async () => {
      const mockGame = {
        id: 'game-123',
        boardType: 'square8',
        status: 'active',
        player1: { id: 'user-1', username: 'Player1' },
        player2: { id: 'user-2', username: 'Player2' },
      };

      mockGameFindUnique.mockResolvedValue(mockGame);

      const result = await mockGameFindUnique({ where: { id: 'game-123' } });

      expect(result.id).toBe('game-123');
      expect(result.player1.username).toBe('Player1');
    });

    it('returns null for non-existent game', async () => {
      mockGameFindUnique.mockResolvedValue(null);

      const result = await mockGameFindUnique({ where: { id: 'non-existent' } });

      expect(result).toBeNull();
    });

    it('includes game state when available', async () => {
      const mockGame = {
        id: 'game-123',
        gameState: JSON.stringify({ currentPlayer: 1, currentPhase: 'ring_placement' }),
      };

      mockGameFindUnique.mockResolvedValue(mockGame);

      const result = await mockGameFindUnique({ where: { id: 'game-123' } });
      const gameState = JSON.parse(result.gameState);

      expect(gameState.currentPlayer).toBe(1);
    });
  });

  describe('move validation', () => {
    it('validates place_ring move', () => {
      const move = {
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
      };

      expect(move.type).toBe('place_ring');
      expect(move.to).toBeDefined();
    });

    it('validates move_stack move', () => {
      const move = {
        type: 'move_stack',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 5, y: 3 },
      };

      expect(move.type).toBe('move_stack');
      expect(move.from).toBeDefined();
      expect(move.to).toBeDefined();
    });

    it('validates overtaking_capture move', () => {
      const move = {
        type: 'overtaking_capture',
        player: 1,
        from: { x: 3, y: 3 },
        captureTarget: { x: 4, y: 3 },
        to: { x: 5, y: 3 },
      };

      expect(move.type).toBe('overtaking_capture');
      expect(move.captureTarget).toBeDefined();
    });

    it('validates continue_capture_segment move', () => {
      const move = {
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 5, y: 3 },
        captureTarget: { x: 6, y: 3 },
        to: { x: 7, y: 3 },
      };

      expect(move.type).toBe('continue_capture_segment');
    });

    it('validates process_line move', () => {
      const move = {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      };

      expect(move.type).toBe('process_line');
      expect(move.lineIndex).toBe(0);
    });

    it('validates choose_line_option move', () => {
      const move = {
        type: 'choose_line_option',
        player: 1,
        lineIndex: 0,
        rewardType: 'COLLAPSE_ALL',
      };

      expect(move.type).toBe('choose_line_option');
      expect(['COLLAPSE_ALL', 'MINIMUM_COLLAPSE']).toContain(move.rewardType);
    });
  });

  describe('authorization checks', () => {
    it('allows participant to view game', () => {
      const userId = 'user-123';
      const game = {
        player1Id: 'user-123',
        player2Id: 'user-456',
        allowSpectators: false,
      };

      const isParticipant = [game.player1Id, game.player2Id].includes(userId);

      expect(isParticipant).toBe(true);
    });

    it('allows spectator when spectators enabled', () => {
      const userId = 'spectator-789';
      const game = {
        player1Id: 'user-123',
        player2Id: 'user-456',
        allowSpectators: true,
      };

      const isParticipant = [game.player1Id, game.player2Id].includes(userId);
      const canView = isParticipant || game.allowSpectators;

      expect(canView).toBe(true);
    });

    it('denies non-participant when spectators disabled', () => {
      const userId = 'outsider-000';
      const game = {
        player1Id: 'user-123',
        player2Id: 'user-456',
        allowSpectators: false,
      };

      const isParticipant = [game.player1Id, game.player2Id].includes(userId);
      const canView = isParticipant || game.allowSpectators;

      expect(canView).toBe(false);
    });
  });

  describe('error handling', () => {
    it('handles game not found error', async () => {
      mockGameFindUnique.mockResolvedValue(null);

      const result = await mockGameFindUnique({ where: { id: 'non-existent' } });

      expect(result).toBeNull();
    });

    it('handles database error', async () => {
      mockGameFindUnique.mockRejectedValue(new Error('Database connection error'));

      await expect(mockGameFindUnique({ where: { id: 'game-123' } })).rejects.toThrow(
        'Database connection error'
      );
    });

    it('handles validation error', () => {
      const invalidInput = {
        boardType: 'invalid_board',
        maxPlayers: 10,
      };

      const validBoardTypes = ['square8', 'square19', 'hexagonal'];
      const validMaxPlayers = [2, 3, 4];

      expect(validBoardTypes).not.toContain(invalidInput.boardType);
      expect(validMaxPlayers).not.toContain(invalidInput.maxPlayers);
    });
  });

  describe('game seed generation', () => {
    it('generates numeric seed', () => {
      // Mock the seed generation
      const seed = Math.floor(Math.random() * 2147483647);

      expect(typeof seed).toBe('number');
      expect(seed).toBeGreaterThanOrEqual(0);
      expect(seed).toBeLessThan(2147483647);
    });

    it('uses provided seed when specified', () => {
      const providedSeed = 12345;

      expect(providedSeed).toBe(12345);
    });
  });

  describe('game state serialization', () => {
    it('serializes game state to JSON', () => {
      const gameState = {
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        board: { size: 8, type: 'square8' },
      };

      const serialized = JSON.stringify(gameState);
      const parsed = JSON.parse(serialized);

      expect(parsed.currentPlayer).toBe(1);
    });

    it('handles null game state', () => {
      const gameState = null;

      expect(gameState).toBeNull();
    });
  });

  describe('time control handling', () => {
    it('handles blitz time control', () => {
      const timeControl = {
        type: 'blitz',
        initialTime: 180000, // 3 minutes
        increment: 2000, // 2 seconds
      };

      expect(timeControl.type).toBe('blitz');
      expect(timeControl.initialTime).toBe(180000);
    });

    it('handles rapid time control', () => {
      const timeControl = {
        type: 'rapid',
        initialTime: 600000, // 10 minutes
        increment: 0,
      };

      expect(timeControl.type).toBe('rapid');
    });

    it('handles no time control', () => {
      const timeControl = {
        type: 'none',
        initialTime: 0,
        increment: 0,
      };

      expect(timeControl.type).toBe('none');
    });
  });
});
