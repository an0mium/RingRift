/**
 * Rate Limiter + Error Handler Integration Test
 *
 * Wires the HTTP rate limiter middleware together with the global
 * errorHandler in an Express app, and asserts that when the API
 * limiter is exceeded the client sees:
 * - HTTP 429 status,
 * - Standard rate limit headers (X-RateLimit-*, Retry-After),
 * - A JSON body with code RATE_LIMIT_EXCEEDED and timestamp.
 *
 * This complements the unit tests in rateLimiter.test.ts and
 * errorHandler.standardized.test.ts by exercising the end-to-end
 * HTTP surface.
 */

import express from 'express';
import request from 'supertest';
import { errorHandler } from '../../../src/server/middleware/errorHandler';

// Keep metrics side effects lightweight for this integration test.
jest.mock('../../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    recordRateLimitHit: jest.fn(),
  }),
}));

describe('Rate limiter + errorHandler integration', () => {
  let originalBypassEnabled: string | undefined;
  let originalBypassIPs: string | undefined;
  let originalBypassUserPattern: string | undefined;
  let rateLimiterModule: typeof import('../../../src/server/middleware/rateLimiter');

  beforeEach(() => {
    // Ensure load-test bypass env vars from other suites (or developer shells)
    // cannot disable rate limiting for this integration test.
    originalBypassEnabled = process.env.RATE_LIMIT_BYPASS_ENABLED;
    originalBypassIPs = process.env.RATE_LIMIT_BYPASS_IPS;
    originalBypassUserPattern = process.env.RATE_LIMIT_BYPASS_USER_PATTERN;

    process.env.RATE_LIMIT_BYPASS_ENABLED = 'false';
    delete process.env.RATE_LIMIT_BYPASS_IPS;
    delete process.env.RATE_LIMIT_BYPASS_USER_PATTERN;

    // Defensive: other suites may have mocked the rate limiter module (or
    // initialized Redis-backed limiters) in the same Jest worker process.
    // Always pull the real module implementation and re-initialize memory
    // limiters so this integration test is deterministic.
    rateLimiterModule = jest.requireActual(
      '../../../src/server/middleware/rateLimiter'
    ) as typeof import('../../../src/server/middleware/rateLimiter');
    rateLimiterModule.initializeMemoryRateLimiters();
    rateLimiterModule.__testResetRateLimiters();
  });

  afterEach(() => {
    if (originalBypassEnabled === undefined) {
      delete process.env.RATE_LIMIT_BYPASS_ENABLED;
    } else {
      process.env.RATE_LIMIT_BYPASS_ENABLED = originalBypassEnabled;
    }

    if (originalBypassIPs === undefined) {
      delete process.env.RATE_LIMIT_BYPASS_IPS;
    } else {
      process.env.RATE_LIMIT_BYPASS_IPS = originalBypassIPs;
    }

    if (originalBypassUserPattern === undefined) {
      delete process.env.RATE_LIMIT_BYPASS_USER_PATTERN;
    } else {
      process.env.RATE_LIMIT_BYPASS_USER_PATTERN = originalBypassUserPattern;
    }
  });

  function createTestApp() {
    const app = express();

    // Apply the API rate limiter just as in the main server pipeline.
    app.get('/limited', rateLimiterModule.rateLimiter, (_req, res) => {
      res.json({ success: true });
    });

    // Attach the global error handler to mirror src/server/index.ts wiring.
    app.use(errorHandler as any);

    return app;
  }

  it('returns a 429 RATE_LIMIT_EXCEEDED response with standard headers when limit is exceeded', async () => {
    const app = createTestApp();
    const config = rateLimiterModule.getRateLimitConfig('api');
    const maxRequests = config?.points ?? 50;

    // Exhaust the in-memory quota for this IP.
    for (let i = 0; i < maxRequests; i++) {
      await request(app).get('/limited');
    }

    // Next request should be rate-limited.
    const res = await request(app).get('/limited');

    expect(res.status).toBe(429);

    // Standard rate limit headers (see docs/RATE_LIMITING.md).
    expect(res.headers['x-ratelimit-limit']).toBe(String(config?.points ?? 50));
    expect(res.headers['x-ratelimit-remaining']).toBe('0');
    expect(res.headers['x-ratelimit-reset']).toBeDefined();
    expect(res.headers['retry-after']).toBeDefined();

    // Body shape consistent with rateLimiter.ts contract.
    expect(res.body).toEqual(
      expect.objectContaining({
        success: false,
        error: expect.objectContaining({
          code: 'RATE_LIMIT_EXCEEDED',
          message: expect.any(String),
          retryAfter: expect.any(Number),
          timestamp: expect.any(String),
        }),
      })
    );
  });
});
