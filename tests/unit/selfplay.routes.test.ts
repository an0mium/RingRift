import express from 'express';
import request from 'supertest';

// --- Mock data -----------------------------------------------------------

const mockDatabases = [
  { path: '/data/games/test.db', name: 'test.db', gameCount: 10, createdAt: '2024-01-01' },
  { path: '/data/games/training.db', name: 'training.db', gameCount: 100, createdAt: '2024-02-01' },
];

const mockGames = [
  { gameId: 'game-1', boardType: 'square8', numPlayers: 2, winner: 1 },
  { gameId: 'game-2', boardType: 'square8', numPlayers: 2, winner: 2 },
];

const mockGameDetails = {
  gameId: 'game-1',
  boardType: 'square8',
  numPlayers: 2,
  winner: 1,
  totalMoves: 50,
  totalTurns: 25,
  createdAt: '2024-01-01T00:00:00Z',
  completedAt: '2024-01-01T00:10:00Z',
  initialState: { board: {} },
  moves: [{ type: 'place', position: { x: 0, y: 0 } }],
  players: [{ playerId: 1, type: 'ai' }],
};

const mockState = { board: { stacks: {} }, currentPlayer: 1 };

const mockStats = {
  totalGames: 100,
  byBoardType: { square8: 80, square19: 20 },
  byNumPlayers: { 2: 90, 3: 10 },
  byWinner: { p1: 50, p2: 40, draw: 10 },
  avgMoves: 45,
  avgDuration: 120000,
};

// --- Mocks (must be before imports that use them) --------------------------

const mockListDatabases = jest.fn();
const mockListGames = jest.fn();
const mockGetGame = jest.fn();
const mockGetStateAtMove = jest.fn();
const mockGetStats = jest.fn();

jest.mock('../../src/server/services/SelfPlayGameService', () => ({
  getSelfPlayGameService: () => ({
    listDatabases: mockListDatabases,
    listGames: mockListGames,
    getGame: mockGetGame,
    getStateAtMove: mockGetStateAtMove,
    getStats: mockGetStats,
  }),
}));

// Mock logger
jest.mock('../../src/server/utils/logger', () => ({
  httpLogger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

// Now import routes (after mocks are set up)
import selfplayRoutes from '../../src/server/routes/selfplay';

// --- Test app factory ---------------------------------------------------

function createTestApp() {
  const app = express();
  app.use(express.json());
  app.use('/api/selfplay', selfplayRoutes);
  return app;
}

// --- Tests --------------------------------------------------------------

describe('Selfplay HTTP routes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Set default mock implementations
    mockListDatabases.mockReturnValue(mockDatabases);
    mockListGames.mockReturnValue(mockGames);
    mockGetGame.mockReturnValue(mockGameDetails);
    mockGetStateAtMove.mockReturnValue(mockState);
    mockGetStats.mockReturnValue(mockStats);
  });

  describe('GET /api/selfplay/databases', () => {
    it('returns list of databases', async () => {
      const app = createTestApp();
      const res = await request(app).get('/api/selfplay/databases').expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.databases).toEqual(mockDatabases);
      expect(mockListDatabases).toHaveBeenCalled();
    });

    it('returns 500 on service error', async () => {
      mockListDatabases.mockImplementation(() => {
        throw new Error('Database scan failed');
      });

      const app = createTestApp();
      const res = await request(app).get('/api/selfplay/databases').expect(500);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Failed to list databases');
    });
  });

  describe('GET /api/selfplay/games', () => {
    it('returns list of games', async () => {
      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games')
        .query({ db: '/path/to/test.db' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.games).toEqual(mockGames);
      expect(res.body.pagination).toMatchObject({
        limit: 50,
        offset: 0,
        returned: 2,
      });
    });

    it('returns 400 when db param is missing', async () => {
      const app = createTestApp();
      const res = await request(app).get('/api/selfplay/games').expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Missing required parameter: db');
    });

    it('passes filter options to service', async () => {
      const app = createTestApp();
      await request(app)
        .get('/api/selfplay/games')
        .query({
          db: '/path/to/test.db',
          boardType: 'square8',
          numPlayers: '2',
          source: 'cmaes',
          hasWinner: 'true',
          limit: '10',
          offset: '5',
        })
        .expect(200);

      expect(mockListGames).toHaveBeenCalledWith('/path/to/test.db', {
        boardType: 'square8',
        numPlayers: 2,
        source: 'cmaes',
        hasWinner: true,
        limit: 10,
        offset: 5,
      });
    });

    it('returns 500 on service error', async () => {
      mockListGames.mockImplementation(() => {
        throw new Error('Query failed');
      });

      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games')
        .query({ db: '/path/to/test.db' })
        .expect(500);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Query failed');
    });
  });

  describe('GET /api/selfplay/games/:gameId', () => {
    it('returns game details', async () => {
      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1')
        .query({ db: '/path/to/test.db' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.game).toEqual(mockGameDetails);
      expect(mockGetGame).toHaveBeenCalledWith('/path/to/test.db', 'game-1');
    });

    it('returns 400 when db param is missing', async () => {
      const app = createTestApp();
      const res = await request(app).get('/api/selfplay/games/game-1').expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Missing required parameter: db');
    });

    it('returns 404 when game not found', async () => {
      mockGetGame.mockReturnValue(null);

      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/nonexistent')
        .query({ db: '/path/to/test.db' })
        .expect(404);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Game not found');
    });

    it('returns 500 on service error', async () => {
      mockGetGame.mockImplementation(() => {
        throw new Error('Database error');
      });

      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1')
        .query({ db: '/path/to/test.db' })
        .expect(500);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Database error');
    });
  });

  describe('GET /api/selfplay/games/:gameId/state', () => {
    it('returns state at move number', async () => {
      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1/state')
        .query({ db: '/path/to/test.db', move: '5' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.moveNumber).toBe(5);
      expect(res.body.state).toEqual(mockState);
      expect(mockGetStateAtMove).toHaveBeenCalledWith('/path/to/test.db', 'game-1', 5);
    });

    it('returns 400 when db param is missing', async () => {
      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1/state')
        .query({ move: '5' })
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Missing required parameter: db');
    });

    it('returns 400 when move param is missing', async () => {
      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1/state')
        .query({ db: '/path/to/test.db' })
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Missing required parameter: move');
    });

    it('returns 400 when move is invalid (negative)', async () => {
      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1/state')
        .query({ db: '/path/to/test.db', move: '-1' })
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Invalid move number');
    });

    it('returns 400 when move is invalid (non-numeric)', async () => {
      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1/state')
        .query({ db: '/path/to/test.db', move: 'abc' })
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Invalid move number');
    });

    it('returns 404 when state not found', async () => {
      mockGetStateAtMove.mockReturnValue(null);

      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1/state')
        .query({ db: '/path/to/test.db', move: '999' })
        .expect(404);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('State not found');
    });

    it('returns 500 on service error', async () => {
      mockGetStateAtMove.mockImplementation(() => {
        throw new Error('Snapshot error');
      });

      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/games/game-1/state')
        .query({ db: '/path/to/test.db', move: '5' })
        .expect(500);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Snapshot error');
    });
  });

  describe('GET /api/selfplay/stats', () => {
    it('returns database statistics', async () => {
      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/stats')
        .query({ db: '/path/to/test.db' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.stats).toEqual(mockStats);
      expect(mockGetStats).toHaveBeenCalledWith('/path/to/test.db');
    });

    it('returns 400 when db param is missing', async () => {
      const app = createTestApp();
      const res = await request(app).get('/api/selfplay/stats').expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Missing required parameter: db');
    });

    it('returns 500 on service error', async () => {
      mockGetStats.mockImplementation(() => {
        throw new Error('Stats calculation failed');
      });

      const app = createTestApp();
      const res = await request(app)
        .get('/api/selfplay/stats')
        .query({ db: '/path/to/test.db' })
        .expect(500);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Stats calculation failed');
    });
  });
});
