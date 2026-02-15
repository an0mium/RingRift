import express from 'express';
import request from 'supertest';

// Simple error handler
function testErrorHandler(err: any, _req: any, res: any, _next: any) {
  const statusCode = err.statusCode || 500;
  const code = err.code || 'SERVER_ERROR';
  res.status(statusCode).json({
    success: false,
    error: { code, message: err.message || 'Internal server error' },
  });
}

// --- Mock state ---
let mockAuthUser: any = null;
let mockAuthError: any = null;

// Mock database responses
const mockFindUnique = jest.fn();
const mockFindFirst = jest.fn();
const mockFindMany = jest.fn();
const mockUpdate = jest.fn();
const mockCount = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    user: {
      findUnique: mockFindUnique,
      findFirst: mockFindFirst,
      findMany: mockFindMany,
      update: mockUpdate,
      count: mockCount,
    },
    game: { findMany: mockFindMany },
    ratingHistory: { findMany: mockFindMany, count: mockCount },
  })),
  withQueryTimeoutStrict: jest.fn(async (promise: Promise<any>) => ({
    success: true,
    data: await promise,
  })),
  TransactionClient: class {},
}));

jest.mock('../../src/server/middleware/auth', () => ({
  authenticate: (req: any, _res: any, next: any) => {
    if (mockAuthError) return next(mockAuthError);
    req.user = mockAuthUser;
    next();
  },
  getAuthUserId: (req: any) => req.user?.id,
  AuthenticatedRequest: {},
}));

jest.mock('../../src/server/middleware/rateLimiter', () => ({
  dataExportRateLimiter: (_req: any, _res: any, next: any) => next(),
  userRatingRateLimiter: (_req: any, _res: any, next: any) => next(),
  userSearchRateLimiter: (_req: any, _res: any, next: any) => next(),
}));

jest.mock('../../src/server/utils/logger', () => ({
  logger: { info: jest.fn(), error: jest.fn(), warn: jest.fn(), debug: jest.fn() },
  httpLogger: { info: jest.fn(), error: jest.fn(), warn: jest.fn() },
  getRequestContext: jest.fn(() => ({})),
  withRequestContext: jest.fn((_req: any, meta: any) => meta),
}));

jest.mock('../../src/server/config', () => ({
  config: {
    isDevelopment: true,
    isProduction: false,
    isTest: true,
    auth: { jwtSecret: 'test-secret-at-least-32-characters-long' },
  },
}));

jest.mock('../../src/server/services/RatingService', () => ({
  RatingService: {
    getRatingHistory: jest.fn().mockResolvedValue({ history: [], total: 0 }),
  },
}));

// Import after mocks
import userRoutes from '../../src/server/routes/user';
import { authenticate } from '../../src/server/middleware/auth';

function createTestApp() {
  const app = express();
  app.use(express.json());
  const { setWebSocketServer } = require('../../src/server/routes/user');
  setWebSocketServer(null);
  // Add authenticate middleware like index.ts does at the mount level
  app.use('/api/users', authenticate as any, userRoutes);
  app.use(testErrorHandler);
  return app;
}

const mockUser = {
  id: 'user-123',
  email: 'test@example.com',
  username: 'testuser',
  role: 'USER',
  rating: 1500,
  gamesPlayed: 10,
  gamesWon: 6,
  createdAt: new Date('2025-01-01'),
  lastLoginAt: new Date('2025-06-01'),
  emailVerified: true,
  isActive: true,
};

describe('User HTTP routes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockAuthUser = {
      id: 'user-123',
      email: 'test@example.com',
      username: 'testuser',
      role: 'USER',
    };
    mockAuthError = null;
  });

  describe('GET /api/users/profile', () => {
    it('returns user profile for authenticated user', async () => {
      mockFindUnique.mockResolvedValue(mockUser);
      const app = createTestApp();
      const res = await request(app).get('/api/users/profile').expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.data.user.username).toBe('testuser');
      expect(res.body.data.user.rating).toBe(1500);
    });

    it('returns 401 when not authenticated', async () => {
      mockAuthError = Object.assign(new Error('Auth required'), {
        statusCode: 401,
        code: 'AUTH_REQUIRED',
      });
      const app = createTestApp();
      await request(app).get('/api/users/profile').expect(401);
    });

    it('returns 404 when user not found in database', async () => {
      mockFindUnique.mockResolvedValue(null);
      const app = createTestApp();
      const res = await request(app).get('/api/users/profile').expect(404);

      expect(res.body.success).toBe(false);
    });
  });

  describe('PUT /api/users/profile', () => {
    it('updates username successfully', async () => {
      mockFindFirst.mockResolvedValue(null); // username not taken
      mockUpdate.mockResolvedValue({ ...mockUser, username: 'newname' });
      const app = createTestApp();
      const res = await request(app)
        .put('/api/users/profile')
        .send({ username: 'newname' })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.data.user.username).toBe('newname');
    });

    it('returns 409 when username is taken', async () => {
      mockFindFirst.mockResolvedValue({ id: 'other-user' }); // username taken
      const app = createTestApp();
      const res = await request(app)
        .put('/api/users/profile')
        .send({ username: 'takenname' })
        .expect(409);

      expect(res.body.success).toBe(false);
    });

    it('returns 401 when not authenticated', async () => {
      mockAuthError = Object.assign(new Error('Auth required'), {
        statusCode: 401,
        code: 'AUTH_REQUIRED',
      });
      const app = createTestApp();
      await request(app).put('/api/users/profile').send({ username: 'test' }).expect(401);
    });
  });

  describe('GET /api/users/stats', () => {
    it('returns user stats', async () => {
      mockFindUnique.mockResolvedValue({ rating: 1500, gamesPlayed: 10, gamesWon: 6 });
      mockFindMany.mockResolvedValue([]); // recent games
      const app = createTestApp();
      const res = await request(app).get('/api/users/stats').expect(200);

      expect(res.body.success).toBe(true);
    });

    it('returns 404 when user not found', async () => {
      mockFindUnique.mockResolvedValue(null);
      const app = createTestApp();
      await request(app).get('/api/users/stats').expect(404);
    });
  });
});
