/**
 * Rate Limiter Tests
 *
 * Tests for the enhanced rate limiting middleware:
 * - Rate limit headers on successful requests
 * - Rate limit headers on exceeded limits
 * - Authentication-based differentiation
 * - Fallback in-memory limiting
 * - Environment-driven configuration
 */

import { Request, Response, NextFunction } from 'express';
import {
  initializeMemoryRateLimiters,
  rateLimiter,
  authRateLimiter,
  authLoginRateLimiter,
  authRegisterRateLimiter,
  authPasswordResetRateLimiter,
  gameRateLimiter,
  gameMovesRateLimiter,
  adaptiveRateLimiter,
  consumeRateLimit,
  fallbackRateLimiter,
  setRateLimitHeaders,
  getRateLimitConfigs,
  getRateLimitConfig,
  __testResetRateLimiters,
} from '../../src/server/middleware/rateLimiter';

// Initialize in-memory rate limiters for testing
beforeAll(() => {
  initializeMemoryRateLimiters();
});

// Reset rate limiters before each test to ensure isolation
beforeEach(() => {
  __testResetRateLimiters();
});

/**
 * Mock Express request object
 */
const createMockRequest = (overrides: Partial<Request> = {}): Request => {
  return {
    ip: '127.0.0.1',
    path: '/test',
    ...overrides,
  } as Request;
};

/**
 * Mock Express response object with header tracking
 */
const createMockResponse = (): Response & { _headers: Record<string, string>; _statusCode: number; _body: any } => {
  const headers: Record<string, string> = {};
  const res: any = {
    _headers: headers,
    _statusCode: 200,
    _body: null,
    set: jest.fn((key: string, value: string) => {
      headers[key] = value;
      return res;
    }),
    status: jest.fn((code: number) => {
      res._statusCode = code;
      return res;
    }),
    json: jest.fn((body: any) => {
      res._body = body;
      return res;
    }),
  };
  return res;
};

/**
 * Mock next function
 */
const createMockNext = (): NextFunction => jest.fn();

describe('Rate Limiter Configuration', () => {
  describe('getRateLimitConfigs', () => {
    it('should return default configuration', () => {
      const configs = getRateLimitConfigs();

      expect(configs).toHaveProperty('api');
      expect(configs).toHaveProperty('apiAuthenticated');
      expect(configs).toHaveProperty('auth');
      expect(configs).toHaveProperty('authLogin');
      expect(configs).toHaveProperty('authRegister');
      expect(configs).toHaveProperty('authPasswordReset');
      expect(configs).toHaveProperty('game');
      expect(configs).toHaveProperty('gameMoves');
      expect(configs).toHaveProperty('websocket');
      expect(configs).toHaveProperty('gameCreateUser');
      expect(configs).toHaveProperty('gameCreateIp');
    });

    it('should have expected structure for each config', () => {
      const configs = getRateLimitConfigs();

      for (const [key, config] of Object.entries(configs)) {
        expect(config).toHaveProperty('keyPrefix');
        expect(config).toHaveProperty('points');
        expect(config).toHaveProperty('duration');
        expect(config).toHaveProperty('blockDuration');

        expect(typeof config.keyPrefix).toBe('string');
        expect(typeof config.points).toBe('number');
        expect(typeof config.duration).toBe('number');
        expect(typeof config.blockDuration).toBe('number');
      }
    });

    it('should have differentiated limits based on endpoint type', () => {
      const configs = getRateLimitConfigs();

      // Auth endpoints should have stricter limits than general API
      expect(configs.auth.points).toBeLessThanOrEqual(configs.api.points);
      expect(configs.authLogin.points).toBeLessThanOrEqual(configs.auth.points);

      // Authenticated users should have higher limits
      expect(configs.apiAuthenticated.points).toBeGreaterThan(configs.api.points);
    });
  });

  describe('getRateLimitConfig', () => {
    it('should return config for valid limiter key', () => {
      const config = getRateLimitConfig('api');
      expect(config).toBeDefined();
      expect(config?.keyPrefix).toBe('api_limit');
    });

    it('should return undefined for invalid limiter key', () => {
      const config = getRateLimitConfig('nonexistent');
      expect(config).toBeUndefined();
    });
  });
});

describe('Rate Limit Headers', () => {
  describe('setRateLimitHeaders', () => {
    it('should set all rate limit headers', () => {
      const res = createMockResponse();
      const limit = 100;
      const remaining = 50;
      const resetTimestamp = Math.ceil(Date.now() / 1000) + 60;

      setRateLimitHeaders(res, limit, remaining, resetTimestamp);

      expect(res.set).toHaveBeenCalledWith('X-RateLimit-Limit', '100');
      expect(res.set).toHaveBeenCalledWith('X-RateLimit-Remaining', '50');
      expect(res.set).toHaveBeenCalledWith('X-RateLimit-Reset', String(resetTimestamp));
    });

    it('should ensure remaining is never negative', () => {
      const res = createMockResponse();
      setRateLimitHeaders(res, 100, -5, 1234567890);

      expect(res.set).toHaveBeenCalledWith('X-RateLimit-Remaining', '0');
    });
  });
});

describe('Rate Limiter Middleware', () => {
  describe('rateLimiter (general API)', () => {
    it('should allow requests within limit and set headers', async () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      await rateLimiter(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(res._headers).toHaveProperty('X-RateLimit-Limit');
      expect(res._headers).toHaveProperty('X-RateLimit-Remaining');
      expect(res._headers).toHaveProperty('X-RateLimit-Reset');
    });

    it('should block requests that exceed limit', async () => {
      const config = getRateLimitConfig('api');
      const maxRequests = config?.points || 50;
      const req = createMockRequest();

      // Exhaust the limit
      for (let i = 0; i < maxRequests; i++) {
        const res = createMockResponse();
        const next = createMockNext();
        await rateLimiter(req, res, next);
      }

      // Next request should be blocked
      const res = createMockResponse();
      const next = createMockNext();
      await rateLimiter(req, res, next);

      expect(next).not.toHaveBeenCalled();
      expect(res._statusCode).toBe(429);
      expect(res._body).toHaveProperty('success', false);
      expect(res._body.error).toHaveProperty('code', 'RATE_LIMIT_EXCEEDED');
      expect(res._body.error).toHaveProperty('retryAfter');
      expect(res._headers).toHaveProperty('Retry-After');
    });
  });

  describe('authLoginRateLimiter', () => {
    it('should have stricter limits than general API', async () => {
      const apiConfig = getRateLimitConfig('api');
      const loginConfig = getRateLimitConfig('authLogin');

      expect(loginConfig?.points).toBeLessThan(apiConfig?.points || 50);
    });

    it('should block after login limit exceeded', async () => {
      const config = getRateLimitConfig('authLogin');
      const maxRequests = config?.points || 5;
      const req = createMockRequest({ path: '/api/auth/login' });

      // Exhaust the limit
      for (let i = 0; i < maxRequests; i++) {
        const res = createMockResponse();
        const next = createMockNext();
        await authLoginRateLimiter(req, res, next);
      }

      // Next request should be blocked
      const res = createMockResponse();
      const next = createMockNext();
      await authLoginRateLimiter(req, res, next);

      expect(next).not.toHaveBeenCalled();
      expect(res._statusCode).toBe(429);
      expect(res._body.error.code).toBe('RATE_LIMIT_EXCEEDED');
    });
  });

  describe('authRegisterRateLimiter', () => {
    it('should have strict limits for registration', async () => {
      const config = getRateLimitConfig('authRegister');
      expect(config?.points).toBeLessThanOrEqual(5);
      // Duration should be 1 hour
      expect(config?.duration).toBe(3600);
    });
  });
});

describe('Adaptive Rate Limiter', () => {
  it('should use higher limits for authenticated users', async () => {
    const authenticatedReq = createMockRequest({
      user: { id: 'user-123', email: 'test@example.com' },
    } as any);
    const anonymousReq = createMockRequest();

    const authConfig = getRateLimitConfig('apiAuthenticated');
    const anonConfig = getRateLimitConfig('api');

    expect(authConfig?.points).toBeGreaterThan(anonConfig?.points || 50);
  });

  it('should select appropriate limiter based on authentication', async () => {
    const middleware = adaptiveRateLimiter('apiAuthenticated', 'api');

    // Authenticated request
    const authReq = createMockRequest({
      user: { id: 'user-123' },
    } as any);
    const authRes = createMockResponse();
    const authNext = createMockNext();

    await middleware(authReq, authRes, authNext);

    expect(authNext).toHaveBeenCalled();
    // Should have higher limit for authenticated user
    expect(parseInt(authRes._headers['X-RateLimit-Limit'], 10)).toBeGreaterThanOrEqual(200);

    // Anonymous request
    const anonReq = createMockRequest({ ip: '192.168.1.100' });
    const anonRes = createMockResponse();
    const anonNext = createMockNext();

    await middleware(anonReq, anonRes, anonNext);

    expect(anonNext).toHaveBeenCalled();
    // Should have lower limit for anonymous user
    expect(parseInt(anonRes._headers['X-RateLimit-Limit'], 10)).toBeLessThanOrEqual(100);
  });
});

describe('consumeRateLimit', () => {
  it('should return allowed:true when within limit', async () => {
    const result = await consumeRateLimit('api', 'test-key-1');

    expect(result.allowed).toBe(true);
    expect(result.limit).toBeDefined();
    expect(result.remaining).toBeDefined();
    expect(result.reset).toBeDefined();
  });

  it('should return quota info in result', async () => {
    const result = await consumeRateLimit('api', 'test-key-2');

    expect(result).toHaveProperty('limit');
    expect(result).toHaveProperty('remaining');
    expect(result).toHaveProperty('reset');

    if (result.limit !== undefined && result.remaining !== undefined) {
      expect(result.remaining).toBeLessThanOrEqual(result.limit);
    }
  });

  it('should return allowed:false and retryAfter when limit exceeded', async () => {
    const config = getRateLimitConfig('api');
    const maxRequests = config?.points || 50;
    const testKey = 'exhausted-key';

    // Exhaust the limit
    for (let i = 0; i < maxRequests; i++) {
      await consumeRateLimit('api', testKey);
    }

    // Next request should be denied
    const result = await consumeRateLimit('api', testKey);

    expect(result.allowed).toBe(false);
    expect(result.retryAfter).toBeGreaterThan(0);
    expect(result.remaining).toBe(0);
  });

  it('should allow request for non-existent limiter with warning', async () => {
    const result = await consumeRateLimit('nonexistent', 'test-key');

    expect(result.allowed).toBe(true);
  });
});

describe('Fallback Rate Limiter', () => {
  it('should allow requests within limit', () => {
    const req = createMockRequest();
    const res = createMockResponse();
    const next = createMockNext();

    fallbackRateLimiter(req, res, next);

    expect(next).toHaveBeenCalled();
    expect(res._headers).toHaveProperty('X-RateLimit-Limit');
    expect(res._headers).toHaveProperty('X-RateLimit-Remaining');
    expect(res._headers).toHaveProperty('X-RateLimit-Reset');
  });

  it('should set correct rate limit response on exceeded', () => {
    // Use unique IP to avoid cross-test pollution
    const testIp = '10.0.0.99';

    // Make 100 requests (fallback default)
    for (let i = 0; i < 100; i++) {
      const req = createMockRequest({ ip: testIp });
      const res = createMockResponse();
      const next = createMockNext();
      fallbackRateLimiter(req, res, next);
    }

    // 101st request should be blocked
    const req = createMockRequest({ ip: testIp });
    const res = createMockResponse();
    const next = createMockNext();
    fallbackRateLimiter(req, res, next);

    expect(next).not.toHaveBeenCalled();
    expect(res._statusCode).toBe(429);
    expect(res._body.error.code).toBe('RATE_LIMIT_EXCEEDED');
    expect(res._headers).toHaveProperty('Retry-After');
  });
});

describe('Different IPs get separate limits', () => {
  it('should track limits separately per IP', async () => {
    const config = getRateLimitConfig('api');
    const maxRequests = config?.points || 50;

    // Exhaust limit for IP 1
    for (let i = 0; i < maxRequests; i++) {
      const req = createMockRequest({ ip: '192.168.1.1' });
      const res = createMockResponse();
      const next = createMockNext();
      await rateLimiter(req, res, next);
    }

    // IP 1 should be blocked
    const req1 = createMockRequest({ ip: '192.168.1.1' });
    const res1 = createMockResponse();
    const next1 = createMockNext();
    await rateLimiter(req1, res1, next1);
    expect(next1).not.toHaveBeenCalled();

    // IP 2 should still be allowed
    const req2 = createMockRequest({ ip: '192.168.1.2' });
    const res2 = createMockResponse();
    const next2 = createMockNext();
    await rateLimiter(req2, res2, next2);
    expect(next2).toHaveBeenCalled();
  });
});

describe('Error Response Format', () => {
  it('should return proper error response structure on rate limit exceeded', async () => {
    const config = getRateLimitConfig('authLogin');
    const maxRequests = config?.points || 5;
    const req = createMockRequest({ ip: '10.0.0.1', path: '/api/auth/login' });

    // Exhaust the limit
    for (let i = 0; i < maxRequests; i++) {
      const res = createMockResponse();
      const next = createMockNext();
      await authLoginRateLimiter(req, res, next);
    }

    // Check error response
    const res = createMockResponse();
    const next = createMockNext();
    await authLoginRateLimiter(req, res, next);

    expect(res._body).toEqual(
      expect.objectContaining({
        success: false,
        error: expect.objectContaining({
          message: expect.any(String),
          code: 'RATE_LIMIT_EXCEEDED',
          retryAfter: expect.any(Number),
          timestamp: expect.any(String),
        }),
      })
    );
  });
});