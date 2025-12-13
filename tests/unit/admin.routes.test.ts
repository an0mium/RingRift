import express from 'express';
import request from 'supertest';

// Simple error handler for testing that returns proper JSON
function testErrorHandler(err: any, _req: any, res: any, _next: any) {
  const statusCode = err.statusCode || 500;
  const code = err.code || 'SERVER_ERROR';
  res.status(statusCode).json({
    success: false,
    error: {
      code,
      message: err.message || 'Internal server error',
    },
  });
}

// --- Mocks (must be before imports that use them) --------------------------

// Store for controlling middleware behavior
let mockAuthUser: any = null;
let mockAuthError: any = null;
let mockAuthorizeError: any = null;

// Mock rate limiter
jest.mock('../../src/server/middleware/rateLimiter', () => ({
  rateLimiter: (_req: any, _res: any, next: any) => next(),
}));

// Mock OrchestratorRolloutService
const mockCircuitBreakerState = {
  isOpen: false,
  errorCount: 2,
  requestCount: 100,
  windowStart: Date.now(),
};

const mockGetCircuitBreakerState = jest.fn(() => mockCircuitBreakerState);
const mockGetErrorRate = jest.fn(() => 2.0);

jest.mock('../../src/server/services/OrchestratorRolloutService', () => ({
  orchestratorRollout: {
    getCircuitBreakerState: mockGetCircuitBreakerState,
    getErrorRate: mockGetErrorRate,
  },
}));

// Mock config
jest.mock('../../src/server/config', () => ({
  config: {
    isDevelopment: false,
    featureFlags: {
      orchestrator: {
        adapterEnabled: true,
        allowlistUsers: ['user-1', 'user-2'],
        denylistUsers: ['user-3'],
        circuitBreaker: {
          enabled: true,
          errorThresholdPercent: 50,
          errorWindowSeconds: 60,
        },
      },
    },
  },
}));

// Mock auth middleware - these functions are called at route registration time
jest.mock('../../src/server/middleware/auth', () => ({
  // authenticate is called per-request
  authenticate: (req: any, _res: any, next: any) => {
    if (mockAuthError) {
      return next(mockAuthError);
    }
    req.user = mockAuthUser;
    next();
  },
  // authorize is called at registration time, returns middleware for per-request use
  authorize: (_roles: string[]) => (req: any, _res: any, next: any) => {
    if (mockAuthorizeError) {
      return next(mockAuthorizeError);
    }
    if (!req.user) {
      const err = new Error('Authentication required') as any;
      err.statusCode = 401;
      err.code = 'AUTH_TOKEN_REQUIRED';
      return next(err);
    }
    if (req.user.role !== 'admin') {
      const err = new Error('Access denied') as any;
      err.statusCode = 403;
      err.code = 'AUTH_FORBIDDEN';
      return next(err);
    }
    next();
  },
}));

// Mock logger
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

// Now import routes (after mocks are set up)
import adminRoutes from '../../src/server/routes/admin';

// --- Test app factory ---------------------------------------------------

function createTestApp() {
  const app = express();
  app.use(express.json());
  app.use('/api/admin', adminRoutes);
  app.use(testErrorHandler);
  return app;
}

// --- Tests --------------------------------------------------------------

describe('Admin HTTP routes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset to default admin user
    mockAuthUser = {
      id: 'admin-user-1',
      email: 'admin@example.com',
      username: 'admin',
      role: 'admin',
    };
    mockAuthError = null;
    mockAuthorizeError = null;
  });

  describe('GET /api/admin/orchestrator/status', () => {
    describe('authentication', () => {
      it('returns 401 when token is missing', async () => {
        mockAuthError = Object.assign(new Error('Authentication token required'), {
          statusCode: 401,
          code: 'AUTH_TOKEN_REQUIRED',
        });

        const app = createTestApp();
        const res = await request(app).get('/api/admin/orchestrator/status').expect(401);

        expect(res.body.success).toBe(false);
        expect(res.body.error.code).toBe('AUTH_TOKEN_REQUIRED');
      });

      it('returns 401 when token is invalid', async () => {
        mockAuthError = Object.assign(new Error('Invalid authentication token'), {
          statusCode: 401,
          code: 'AUTH_TOKEN_INVALID',
        });

        const app = createTestApp();
        const res = await request(app).get('/api/admin/orchestrator/status').expect(401);

        expect(res.body.success).toBe(false);
        expect(res.body.error.code).toBe('AUTH_TOKEN_INVALID');
      });

      it('returns 401 when token is expired', async () => {
        mockAuthError = Object.assign(new Error('Token expired'), {
          statusCode: 401,
          code: 'AUTH_TOKEN_EXPIRED',
        });

        const app = createTestApp();
        const res = await request(app).get('/api/admin/orchestrator/status').expect(401);

        expect(res.body.success).toBe(false);
        expect(res.body.error.code).toBe('AUTH_TOKEN_EXPIRED');
      });
    });

    describe('authorization', () => {
      it('returns 403 when user is not an admin', async () => {
        mockAuthUser = {
          id: 'regular-user-1',
          email: 'user@example.com',
          username: 'regularuser',
          role: 'USER',
        };

        const app = createTestApp();
        const res = await request(app).get('/api/admin/orchestrator/status').expect(403);

        expect(res.body.success).toBe(false);
        expect(res.body.error.code).toBe('AUTH_FORBIDDEN');
      });

      it('returns 200 when user has admin role', async () => {
        const app = createTestApp();
        const res = await request(app).get('/api/admin/orchestrator/status').expect(200);

        expect(res.body.success).toBe(true);
      });
    });

    describe('response validation', () => {
      it('returns orchestrator config with all expected fields', async () => {
        const app = createTestApp();
        const res = await request(app).get('/api/admin/orchestrator/status').expect(200);

        expect(res.body.success).toBe(true);
        expect(res.body.data.config).toBeDefined();
        expect(res.body.data.config.adapterEnabled).toBe(true);
        expect(res.body.data.config.allowlistUsers).toEqual(['user-1', 'user-2']);
        expect(res.body.data.config.denylistUsers).toEqual(['user-3']);
        expect(res.body.data.config.circuitBreaker).toEqual({
          enabled: true,
          errorThresholdPercent: 50,
          errorWindowSeconds: 60,
        });
      });

      it('returns circuit breaker state with all expected fields', async () => {
        const app = createTestApp();
        const res = await request(app).get('/api/admin/orchestrator/status').expect(200);

        expect(res.body.success).toBe(true);
        expect(res.body.data.circuitBreaker).toBeDefined();
        expect(res.body.data.circuitBreaker.isOpen).toBe(false);
        expect(res.body.data.circuitBreaker.errorCount).toBe(2);
        expect(res.body.data.circuitBreaker.requestCount).toBe(100);
        expect(res.body.data.circuitBreaker.windowStart).toBeDefined();
        expect(res.body.data.circuitBreaker.errorRatePercent).toBe(2.0);
      });

      it('returns windowStart as ISO8601 timestamp', async () => {
        const app = createTestApp();
        const res = await request(app).get('/api/admin/orchestrator/status').expect(200);

        const windowStart = res.body.data.circuitBreaker.windowStart;
        expect(windowStart).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
      });
    });

    describe('service integration', () => {
      it('calls OrchestratorRolloutService methods', async () => {
        const app = createTestApp();
        await request(app).get('/api/admin/orchestrator/status').expect(200);

        expect(mockGetCircuitBreakerState).toHaveBeenCalled();
        expect(mockGetErrorRate).toHaveBeenCalled();
      });
    });
  });
});
