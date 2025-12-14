import express from 'express';
import request from 'supertest';

import { errorHandler } from '../../src/server/middleware/errorHandler';
import { sandboxHelperRoutes } from '../../src/server/routes/game';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';

jest.mock('../../src/server/services/AIServiceClient', () => ({
  getAIServiceClient: jest.fn(),
}));

function createTestApp() {
  const app = express();
  app.use(express.json());
  app.use('/api/games', sandboxHelperRoutes);
  app.use(errorHandler as any);
  return app;
}

describe('GET /api/games/sandbox/ai/ladder/health (contract)', () => {
  it('proxies query params into AIServiceClient.getLadderHealth', async () => {
    const app = createTestApp();

    const mockGetLadderHealth = jest.fn(async () => ({
      generated_at: '2025-01-01T00:00:00Z',
      summary: { tiers: 9 },
      tiers: [],
    }));

    (getAIServiceClient as unknown as jest.Mock).mockReturnValue({
      getLadderHealth: mockGetLadderHealth,
    });

    const res = await request(app)
      .get('/api/games/sandbox/ai/ladder/health')
      .query({ boardType: 'square8', numPlayers: '2', difficulty: '6' })
      .expect(200);

    expect(mockGetLadderHealth).toHaveBeenCalledTimes(1);
    expect(mockGetLadderHealth).toHaveBeenCalledWith({
      boardType: 'square8',
      numPlayers: 2,
      difficulty: 6,
    });

    expect(res.body).toMatchObject({
      generated_at: '2025-01-01T00:00:00Z',
      summary: { tiers: 9 },
    });
  });
});
