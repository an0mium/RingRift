/**
 * CacheService unit tests
 *
 * These tests exercise the CacheService wrapper around the Redis client to
 * ensure it correctly:
 * - Serializes/deserializes values
 * - Returns sensible fallbacks on errors
 * - Records cache hit/miss metrics via MetricsService
 * - Normalizes Redis set membership responses to booleans
 */

import { CacheService } from '../../src/server/cache/redis';

const mockClient = {
  get: jest.fn(),
  set: jest.fn(),
  setEx: jest.fn(),
  del: jest.fn(),
  exists: jest.fn(),
  expire: jest.fn(),
  keys: jest.fn(),
  flushAll: jest.fn(),
  hGet: jest.fn(),
  hSet: jest.fn(),
  hGetAll: jest.fn(),
  hDel: jest.fn(),
  lPush: jest.fn(),
  rPush: jest.fn(),
  lPop: jest.fn(),
  rPop: jest.fn(),
  lRange: jest.fn(),
  sAdd: jest.fn(),
  sRem: jest.fn(),
  sMembers: jest.fn(),
  sIsMember: jest.fn(),
  setEx: jest.fn(),
} as unknown as ReturnType<typeof jest.fn>;

const mockMetrics = {
  recordCacheHit: jest.fn(),
  recordCacheMiss: jest.fn(),
};

jest.mock('../../src/server/services/MetricsService', () => ({
  getMetricsService: () => mockMetrics,
}));

describe('CacheService', () => {
  let cache: CacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    cache = new CacheService(mockClient as any);
  });

  describe('get', () => {
    it('returns parsed value and records cache hit when key exists', async () => {
      (mockClient.get as jest.Mock).mockResolvedValueOnce(JSON.stringify({ foo: 'bar' }));

      const result = await cache.get<{ foo: string }>('key');

      expect(result).toEqual({ foo: 'bar' });
      expect(mockMetrics.recordCacheHit).toHaveBeenCalled();
      expect(mockMetrics.recordCacheMiss).not.toHaveBeenCalled();
    });

    it('returns null and records cache miss when key does not exist', async () => {
      (mockClient.get as jest.Mock).mockResolvedValueOnce(null);

      const result = await cache.get('missing');

      expect(result).toBeNull();
      expect(mockMetrics.recordCacheMiss).toHaveBeenCalled();
    });

    it('returns null and logs on error', async () => {
      (mockClient.get as jest.Mock).mockRejectedValueOnce(new Error('boom'));

      const result = await cache.get('key');

      expect(result).toBeNull();
    });
  });

  describe('set', () => {
    it('sets value without TTL when ttlSeconds is undefined', async () => {
      (mockClient.set as jest.Mock).mockResolvedValueOnce('OK');

      const ok = await cache.set('key', { a: 1 });

      expect(ok).toBe(true);
      expect(mockClient.set).toHaveBeenCalledWith('key', JSON.stringify({ a: 1 }));
    });

    it('sets value with TTL when ttlSeconds is provided', async () => {
      (mockClient.setEx as jest.Mock).mockResolvedValueOnce('OK');

      const ok = await cache.set('key', { a: 1 }, 60);

      expect(ok).toBe(true);
      expect(mockClient.setEx).toHaveBeenCalledWith('key', 60, JSON.stringify({ a: 1 }));
    });

    it('returns false on error', async () => {
      (mockClient.set as jest.Mock).mockRejectedValueOnce(new Error('boom'));

      const ok = await cache.set('key', { a: 1 });

      expect(ok).toBe(false);
    });
  });

  describe('exists', () => {
    it('returns true and records hit when Redis returns 1', async () => {
      (mockClient.exists as jest.Mock).mockResolvedValueOnce(1);

      const exists = await cache.exists('key');

      expect(exists).toBe(true);
      expect(mockMetrics.recordCacheHit).toHaveBeenCalled();
    });

    it('returns false and records miss when Redis returns 0', async () => {
      (mockClient.exists as jest.Mock).mockResolvedValueOnce(0);

      const exists = await cache.exists('key');

      expect(exists).toBe(false);
      expect(mockMetrics.recordCacheMiss).toHaveBeenCalled();
    });

    it('returns false on error', async () => {
      (mockClient.exists as jest.Mock).mockRejectedValueOnce(new Error('boom'));

      const exists = await cache.exists('key');

      expect(exists).toBe(false);
    });
  });

  describe('sIsMember', () => {
    it('returns true and records hit when Redis reports member present', async () => {
      (mockClient.sIsMember as jest.Mock).mockResolvedValueOnce(1);

      const isMember = await cache.sIsMember('set', 'value');

      expect(isMember).toBe(true);
      expect(mockMetrics.recordCacheHit).toHaveBeenCalled();
    });

    it('returns false and records miss when Redis reports member absent', async () => {
      (mockClient.sIsMember as jest.Mock).mockResolvedValueOnce(0);

      const isMember = await cache.sIsMember('set', 'value');

      expect(isMember).toBe(false);
      expect(mockMetrics.recordCacheMiss).toHaveBeenCalled();
    });

    it('returns false on error', async () => {
      (mockClient.sIsMember as jest.Mock).mockRejectedValueOnce(new Error('boom'));

      const isMember = await cache.sIsMember('set', 'value');

      expect(isMember).toBe(false);
    });
  });

  describe('expire', () => {
    it('returns true when expire succeeds', async () => {
      (mockClient.expire as jest.Mock).mockResolvedValueOnce(1);

      const ok = await cache.expire('key', 10);

      expect(ok).toBe(true);
    });

    it('returns false on error', async () => {
      (mockClient.expire as jest.Mock).mockRejectedValueOnce(new Error('boom'));

      const ok = await cache.expire('key', 10);

      expect(ok).toBe(false);
    });
  });
});
