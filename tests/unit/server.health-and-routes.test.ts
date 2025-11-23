import express from 'express';
import request from 'supertest';
import client from 'prom-client';
import setupRoutes from '../../src/server/routes';
import '../../src/server/utils/rulesParityMetrics';

/**
 * Lightweight server harness for testing the public HTTP surface:
 * - /health
 * - /api (root API info endpoint)
 *
 * We intentionally avoid importing src/server/index.ts here to keep
 * tests isolated from real DB/Redis connections and actual network
 * listeners. Instead we mirror the minimal wiring from index.ts that
 * is relevant for these endpoints.
 */

function createTestApp() {
  const app = express();
  app.use(express.json());

  // Health check endpoint (lightweight mirror of src/server/index.ts)
  app.get('/health', (_req, res) => {
    res.status(200).json({
      status: 'healthy',
      // Use concrete runtime values; the test asserts only on types and
      // the presence of these fields, not their exact values.
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.npm_package_version || 'test',
    });
  });

  // Prometheus metrics endpoint (lightweight mirror of src/server/index.ts).
  // This uses the shared default registry from prom-client so that metrics
  // declared in src/server/utils/rulesParityMetrics.ts are exposed.
  app.get('/metrics', async (_req, res) => {
    res.set('Content-Type', client.register.contentType);
    const metrics = await client.register.metrics();
    res.send(metrics);
  });

  // API routes mounted at /api
  app.use('/api', setupRoutes());

  return app;
}

describe('Server health and API info routes', () => {
  it('GET /health responds with healthy status and basic metadata', async () => {
    const app = createTestApp();

    const res = await request(app).get('/health').expect(200);

    expect(res.body).toMatchObject({
      status: 'healthy',
    });

    // Ensure the shape matches the contract without asserting exact values.
    expect(typeof res.body.timestamp).toBe('string');
    expect(typeof res.body.uptime).toBe('number');
    expect(typeof res.body.version).toBe('string');
  });

  it('GET /api returns API metadata and endpoint map', async () => {
    const app = createTestApp();

    const res = await request(app).get('/api').expect(200);

    expect(res.body).toMatchObject({
      success: true,
      message: 'RingRift API',
      version: expect.any(String),
      endpoints: {
        auth: '/api/auth',
        games: '/api/games',
        users: '/api/users',
      },
    });

    expect(typeof res.body.timestamp).toBe('string');
  });

  it('GET /metrics exposes core Prometheus metrics', async () => {
    const app = createTestApp();

    const res = await request(app).get('/metrics').expect(200);
    const body = res.text;

    expect(body).toContain('ai_move_latency_ms');
    expect(body).toContain('ai_fallback_total');
    expect(body).toContain('game_move_latency_ms');
    expect(body).toContain('websocket_connections_current');
  });
});
