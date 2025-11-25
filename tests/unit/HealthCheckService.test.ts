/**
 * Tests for HealthCheckService
 *
 * Covers:
 * - Liveness probe (minimal checks)
 * - Readiness probe with dependency health status
 * - Healthy, degraded, and unhealthy scenarios
 * - Timeout handling for slow checks
 */

// Mock the logger module BEFORE any imports that might use it
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

// Mock the config module BEFORE it's used by other modules
jest.mock('../../src/server/config', () => ({
  config: {
    app: {
      version: '1.0.0-test',
    },
    aiService: {
      url: 'http://localhost:8001',
    },
    logging: {
      level: 'info',
    },
    nodeEnv: 'test',
  },
}));

// Mock the database connection module
jest.mock('../../src/server/database/connection', () => ({
  checkDatabaseHealth: jest.fn(),
  getDatabaseClient: jest.fn(),
}));

// Mock the redis cache module
jest.mock('../../src/server/cache/redis', () => ({
  getRedisClient: jest.fn(),
}));

// Mock global fetch for AI service health checks
const mockFetch = jest.fn();
global.fetch = mockFetch as any;

// Import mocked modules
import { checkDatabaseHealth, getDatabaseClient } from '../../src/server/database/connection';
import { getRedisClient } from '../../src/server/cache/redis';

// Import HealthCheckService AFTER mocks are set up
import {
  getLivenessStatus,
  getReadinessStatus,
  isServiceReady,
  HealthCheckService,
  HealthCheckResponse,
} from '../../src/server/services/HealthCheckService';

const mockCheckDatabaseHealth = checkDatabaseHealth as jest.MockedFunction<typeof checkDatabaseHealth>;
const mockGetDatabaseClient = getDatabaseClient as jest.MockedFunction<typeof getDatabaseClient>;
const mockGetRedisClient = getRedisClient as jest.MockedFunction<typeof getRedisClient>;

describe('HealthCheckService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  describe('getLivenessStatus', () => {
    it('returns healthy status with basic metadata', () => {
      const status = getLivenessStatus();

      expect(status.status).toBe('healthy');
      expect(status.version).toBe('1.0.0-test');
      expect(typeof status.timestamp).toBe('string');
      expect(typeof status.uptime).toBe('number');
      expect(status.uptime).toBeGreaterThanOrEqual(0);
      // Liveness check should not include detailed checks
      expect(status.checks).toBeUndefined();
    });

    it('returns consistent structure on multiple calls', () => {
      const status1 = getLivenessStatus();
      const status2 = getLivenessStatus();

      expect(status1.status).toBe(status2.status);
      expect(status1.version).toBe(status2.version);
    });
  });

  describe('getReadinessStatus', () => {
    describe('when all dependencies are healthy', () => {
      beforeEach(() => {
        // Mock database as healthy
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockResolvedValue(true);

        // Mock Redis as healthy with PING response
        const mockRedisClient = {
          ping: jest.fn().mockResolvedValue('PONG'),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);

        // Mock AI service as healthy
        mockFetch.mockResolvedValue({
          ok: true,
          status: 200,
        } as Response);
      });

      it('returns healthy status with all checks passing', async () => {
        const status = await getReadinessStatus();

        expect(status.status).toBe('healthy');
        expect(status.version).toBe('1.0.0-test');
        expect(typeof status.timestamp).toBe('string');
        expect(typeof status.uptime).toBe('number');

        // Check individual dependency statuses
        expect(status.checks).toBeDefined();
        expect(status.checks?.database?.status).toBe('healthy');
        expect(status.checks?.redis?.status).toBe('healthy');
        expect(status.checks?.aiService?.status).toBe('healthy');

        // Latency should be recorded
        expect(typeof status.checks?.database?.latency).toBe('number');
        expect(typeof status.checks?.redis?.latency).toBe('number');
        expect(typeof status.checks?.aiService?.latency).toBe('number');
      });

      it('isServiceReady returns true for healthy status', async () => {
        const status = await getReadinessStatus();
        expect(isServiceReady(status)).toBe(true);
      });
    });

    describe('when database is unavailable', () => {
      beforeEach(() => {
        // Mock database as unavailable
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockResolvedValue(false);

        // Mock Redis as healthy
        const mockRedisClient = {
          ping: jest.fn().mockResolvedValue('PONG'),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);

        // Mock AI service as healthy
        mockFetch.mockResolvedValue({
          ok: true,
          status: 200,
        } as Response);
      });

      it('returns unhealthy status', async () => {
        const status = await getReadinessStatus();

        expect(status.status).toBe('unhealthy');
        expect(status.checks?.database?.status).toBe('unhealthy');
        expect(status.checks?.database?.error).toBeDefined();
      });

      it('isServiceReady returns false for unhealthy status', async () => {
        const status = await getReadinessStatus();
        expect(isServiceReady(status)).toBe(false);
      });
    });

    describe('when database client is not initialized', () => {
      beforeEach(() => {
        mockGetDatabaseClient.mockReturnValue(null);

        // Mock Redis as healthy
        const mockRedisClient = {
          ping: jest.fn().mockResolvedValue('PONG'),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);

        // Mock AI service as healthy
        mockFetch.mockResolvedValue({
          ok: true,
          status: 200,
        } as Response);
      });

      it('returns unhealthy status', async () => {
        const status = await getReadinessStatus();

        expect(status.status).toBe('unhealthy');
        expect(status.checks?.database?.status).toBe('unhealthy');
        expect(status.checks?.database?.error).toContain('not initialized');
      });
    });

    describe('when Redis is unavailable', () => {
      beforeEach(() => {
        // Mock database as healthy
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockResolvedValue(true);

        // Mock Redis as unavailable
        mockGetRedisClient.mockReturnValue(null);

        // Mock AI service as healthy
        mockFetch.mockResolvedValue({
          ok: true,
          status: 200,
        } as Response);
      });

      it('returns degraded status (Redis is non-critical)', async () => {
        const status = await getReadinessStatus();

        expect(status.status).toBe('degraded');
        expect(status.checks?.database?.status).toBe('healthy');
        expect(status.checks?.redis?.status).toBe('degraded');
        expect(status.checks?.redis?.error).toContain('not connected');
      });

      it('isServiceReady returns true for degraded status', async () => {
        const status = await getReadinessStatus();
        // Degraded means the service can still serve traffic
        expect(isServiceReady(status)).toBe(true);
      });
    });

    describe('when Redis PING fails', () => {
      beforeEach(() => {
        // Mock database as healthy
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockResolvedValue(true);

        // Mock Redis PING rejection
        const mockRedisClient = {
          ping: jest.fn().mockRejectedValue(new Error('Connection refused')),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);

        // Mock AI service as healthy
        mockFetch.mockResolvedValue({
          ok: true,
          status: 200,
        } as Response);
      });

      it('returns degraded status with error message', async () => {
        const status = await getReadinessStatus();

        expect(status.status).toBe('degraded');
        expect(status.checks?.redis?.status).toBe('degraded');
        expect(status.checks?.redis?.error).toContain('Connection refused');
      });
    });

    describe('when AI service is unavailable', () => {
      beforeEach(() => {
        // Mock database as healthy
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockResolvedValue(true);

        // Mock Redis as healthy
        const mockRedisClient = {
          ping: jest.fn().mockResolvedValue('PONG'),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);

        // Mock AI service as unavailable
        mockFetch.mockRejectedValue(new Error('ECONNREFUSED'));
      });

      it('returns degraded status (AI service is non-critical)', async () => {
        const status = await getReadinessStatus();

        expect(status.status).toBe('degraded');
        expect(status.checks?.database?.status).toBe('healthy');
        expect(status.checks?.redis?.status).toBe('healthy');
        expect(status.checks?.aiService?.status).toBe('degraded');
        expect(status.checks?.aiService?.error).toContain('ECONNREFUSED');
      });
    });

    describe('when AI service returns non-200 status', () => {
      beforeEach(() => {
        // Mock database as healthy
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockResolvedValue(true);

        // Mock Redis as healthy
        const mockRedisClient = {
          ping: jest.fn().mockResolvedValue('PONG'),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);

        // Mock AI service returning 503
        mockFetch.mockResolvedValue({
          ok: false,
          status: 503,
        } as Response);
      });

      it('returns degraded status with status code in error', async () => {
        const status = await getReadinessStatus();

        expect(status.status).toBe('degraded');
        expect(status.checks?.aiService?.status).toBe('degraded');
        expect(status.checks?.aiService?.error).toContain('503');
      });
    });

    describe('when includeAIService is false', () => {
      beforeEach(() => {
        // Mock database as healthy
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockResolvedValue(true);

        // Mock Redis as healthy
        const mockRedisClient = {
          ping: jest.fn().mockResolvedValue('PONG'),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);
      });

      it('excludes AI service check from results', async () => {
        const status = await getReadinessStatus({ includeAIService: false });

        expect(status.checks?.database).toBeDefined();
        expect(status.checks?.redis).toBeDefined();
        expect(status.checks?.aiService).toBeUndefined();
        expect(mockFetch).not.toHaveBeenCalled();
      });
    });

    describe('timeout handling', () => {
      beforeEach(() => {
        // Mock database as healthy but slow
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockImplementation(
          () =>
            new Promise((resolve) => {
              setTimeout(() => resolve(true), 100); // 100ms delay
            })
        );

        // Mock Redis as healthy
        const mockRedisClient = {
          ping: jest.fn().mockResolvedValue('PONG'),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);

        // Mock AI service as healthy
        mockFetch.mockResolvedValue({
          ok: true,
          status: 200,
        } as Response);
      });

      it('uses custom timeout when provided', async () => {
        // Use a short timeout that should succeed
        const status = await getReadinessStatus({ timeoutMs: 500 });

        expect(status.status).toBe('healthy');
      });
    });

    describe('database query error', () => {
      beforeEach(() => {
        // Mock database query throwing an error
        mockGetDatabaseClient.mockReturnValue({} as any);
        mockCheckDatabaseHealth.mockRejectedValue(new Error('Query timeout'));

        // Mock Redis as healthy
        const mockRedisClient = {
          ping: jest.fn().mockResolvedValue('PONG'),
        };
        mockGetRedisClient.mockReturnValue(mockRedisClient as any);

        // Mock AI service as healthy
        mockFetch.mockResolvedValue({
          ok: true,
          status: 200,
        } as Response);
      });

      it('returns unhealthy status with error message', async () => {
        const status = await getReadinessStatus();

        expect(status.status).toBe('unhealthy');
        expect(status.checks?.database?.status).toBe('unhealthy');
        expect(status.checks?.database?.error).toContain('Query timeout');
      });
    });
  });

  describe('isServiceReady helper', () => {
    it('returns true for healthy status', () => {
      const response: HealthCheckResponse = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: 100,
      };
      expect(isServiceReady(response)).toBe(true);
    });

    it('returns true for degraded status', () => {
      const response: HealthCheckResponse = {
        status: 'degraded',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: 100,
      };
      expect(isServiceReady(response)).toBe(true);
    });

    it('returns false for unhealthy status', () => {
      const response: HealthCheckResponse = {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: 100,
      };
      expect(isServiceReady(response)).toBe(false);
    });
  });

  describe('HealthCheckService namespace export', () => {
    it('exports all methods through HealthCheckService object', () => {
      expect(HealthCheckService.getLivenessStatus).toBe(getLivenessStatus);
      expect(HealthCheckService.getReadinessStatus).toBe(getReadinessStatus);
      expect(HealthCheckService.isServiceReady).toBe(isServiceReady);
    });
  });
});