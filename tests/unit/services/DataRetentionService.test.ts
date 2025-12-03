/**
 * DataRetentionService Unit Tests
 *
 * Tests for the data retention service including:
 * - Running retention tasks (success and error paths)
 * - Hard deleting expired users
 * - Cleaning up expired tokens
 * - Soft-deleting unverified accounts
 * - Configuration management
 * - Factory function with environment variables
 */

import {
  DataRetentionService,
  DEFAULT_RETENTION,
  createDataRetentionService,
  RetentionConfig,
} from '../../../src/server/services/DataRetentionService';

// Mock logger
jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

import { logger } from '../../../src/server/utils/logger';

describe('DataRetentionService', () => {
  let mockPrisma: {
    user: {
      deleteMany: jest.Mock;
      updateMany: jest.Mock;
    };
    refreshToken: {
      deleteMany: jest.Mock;
    };
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockPrisma = {
      user: {
        deleteMany: jest.fn().mockResolvedValue({ count: 0 }),
        updateMany: jest.fn().mockResolvedValue({ count: 0 }),
      },
      refreshToken: {
        deleteMany: jest.fn().mockResolvedValue({ count: 0 }),
      },
    };
  });

  describe('constructor', () => {
    it('should initialize with default config when no config provided', () => {
      const service = new DataRetentionService(mockPrisma as any);
      const config = service.getConfig();

      expect(config).toEqual(DEFAULT_RETENTION);
    });

    it('should merge custom config with defaults', () => {
      const customConfig: Partial<RetentionConfig> = {
        deletedUserRetentionDays: 60,
        expiredTokenRetentionDays: 14,
      };

      const service = new DataRetentionService(mockPrisma as any, customConfig);
      const config = service.getConfig();

      expect(config.deletedUserRetentionDays).toBe(60);
      expect(config.expiredTokenRetentionDays).toBe(14);
      // Defaults should be preserved
      expect(config.inactiveUserThresholdDays).toBe(DEFAULT_RETENTION.inactiveUserThresholdDays);
    });

    it('should log initialization', () => {
      new DataRetentionService(mockPrisma as any);

      expect(logger.debug).toHaveBeenCalledWith(
        'DataRetentionService initialized',
        expect.objectContaining({ config: expect.any(Object) })
      );
    });
  });

  describe('runRetentionTasks', () => {
    it('should run all cleanup tasks in parallel', async () => {
      mockPrisma.user.deleteMany.mockResolvedValue({ count: 5 });
      mockPrisma.refreshToken.deleteMany.mockResolvedValue({ count: 10 });
      mockPrisma.user.updateMany.mockResolvedValue({ count: 3 });

      const service = new DataRetentionService(mockPrisma as any);
      const report = await service.runRetentionTasks();

      expect(report.hardDeletedUsers).toBe(5);
      expect(report.deletedTokens).toBe(10);
      expect(report.deletedUnverified).toBe(3);
      expect(report.executedAt).toBeInstanceOf(Date);
      expect(report.durationMs).toBeGreaterThanOrEqual(0);
    });

    it('should log start and completion', async () => {
      const service = new DataRetentionService(mockPrisma as any);
      await service.runRetentionTasks();

      expect(logger.info).toHaveBeenCalledWith('Starting data retention cleanup tasks');
      expect(logger.info).toHaveBeenCalledWith(
        'Data retention cleanup tasks completed',
        expect.objectContaining({ report: expect.any(Object) })
      );
    });

    it('should log error and rethrow when a task fails', async () => {
      const error = new Error('Database connection failed');
      mockPrisma.user.deleteMany.mockRejectedValue(error);

      const service = new DataRetentionService(mockPrisma as any);

      await expect(service.runRetentionTasks()).rejects.toThrow('Database connection failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Data retention cleanup tasks failed',
        expect.objectContaining({
          error: 'Database connection failed',
          durationMs: expect.any(Number),
        })
      );
    });

    it('should handle non-Error thrown values', async () => {
      mockPrisma.user.deleteMany.mockRejectedValue('string error');

      const service = new DataRetentionService(mockPrisma as any);

      await expect(service.runRetentionTasks()).rejects.toBe('string error');

      expect(logger.error).toHaveBeenCalledWith(
        'Data retention cleanup tasks failed',
        expect.objectContaining({
          error: 'string error',
        })
      );
    });
  });

  describe('hardDeleteExpiredUsers', () => {
    it('should delete users soft-deleted past retention period', async () => {
      mockPrisma.user.deleteMany.mockResolvedValue({ count: 3 });

      const service = new DataRetentionService(mockPrisma as any, {
        deletedUserRetentionDays: 30,
      });

      const result = await service.hardDeleteExpiredUsers();

      expect(result).toBe(3);
      expect(mockPrisma.user.deleteMany).toHaveBeenCalledWith({
        where: {
          deletedAt: {
            not: null,
            lt: expect.any(Date),
          },
        },
      });
    });

    it('should log when users are deleted', async () => {
      mockPrisma.user.deleteMany.mockResolvedValue({ count: 5 });

      const service = new DataRetentionService(mockPrisma as any);
      await service.hardDeleteExpiredUsers();

      expect(logger.info).toHaveBeenCalledWith(
        'Hard deleted 5 expired user accounts',
        expect.objectContaining({ count: 5 })
      );
    });

    it('should not log when no users are deleted', async () => {
      mockPrisma.user.deleteMany.mockResolvedValue({ count: 0 });

      const service = new DataRetentionService(mockPrisma as any);
      await service.hardDeleteExpiredUsers();

      // Only debug log should be called, not info
      const infoCalls = (logger.info as jest.Mock).mock.calls.filter((call) =>
        call[0].includes('Hard deleted')
      );
      expect(infoCalls).toHaveLength(0);
    });
  });

  describe('cleanupExpiredTokens', () => {
    it('should delete expired and revoked tokens', async () => {
      mockPrisma.refreshToken.deleteMany.mockResolvedValue({ count: 15 });

      const service = new DataRetentionService(mockPrisma as any, {
        expiredTokenRetentionDays: 7,
      });

      const result = await service.cleanupExpiredTokens();

      expect(result).toBe(15);
      expect(mockPrisma.refreshToken.deleteMany).toHaveBeenCalledWith({
        where: {
          OR: [
            { expiresAt: { lt: expect.any(Date) } },
            { revokedAt: { not: null, lt: expect.any(Date) } },
          ],
        },
      });
    });

    it('should log when tokens are deleted', async () => {
      mockPrisma.refreshToken.deleteMany.mockResolvedValue({ count: 10 });

      const service = new DataRetentionService(mockPrisma as any);
      await service.cleanupExpiredTokens();

      expect(logger.info).toHaveBeenCalledWith(
        'Deleted 10 expired/revoked refresh tokens',
        expect.objectContaining({ count: 10 })
      );
    });

    it('should not log when no tokens are deleted', async () => {
      mockPrisma.refreshToken.deleteMany.mockResolvedValue({ count: 0 });

      const service = new DataRetentionService(mockPrisma as any);
      await service.cleanupExpiredTokens();

      const infoCalls = (logger.info as jest.Mock).mock.calls.filter(
        (call) => call[0].includes('Deleted') && call[0].includes('tokens')
      );
      expect(infoCalls).toHaveLength(0);
    });
  });

  describe('cleanupUnverifiedAccounts', () => {
    it('should soft-delete unverified accounts past threshold', async () => {
      mockPrisma.user.updateMany.mockResolvedValue({ count: 7 });

      const service = new DataRetentionService(mockPrisma as any, {
        unverifiedAccountRetentionDays: 7,
      });

      const result = await service.cleanupUnverifiedAccounts();

      expect(result).toBe(7);
      expect(mockPrisma.user.updateMany).toHaveBeenCalledWith({
        where: {
          emailVerified: false,
          createdAt: { lt: expect.any(Date) },
          deletedAt: null,
        },
        data: {
          deletedAt: expect.any(Date),
          isActive: false,
        },
      });
    });

    it('should log when accounts are soft-deleted', async () => {
      mockPrisma.user.updateMany.mockResolvedValue({ count: 4 });

      const service = new DataRetentionService(mockPrisma as any);
      await service.cleanupUnverifiedAccounts();

      expect(logger.info).toHaveBeenCalledWith(
        'Soft-deleted 4 unverified accounts',
        expect.objectContaining({ count: 4 })
      );
    });

    it('should not log when no accounts are soft-deleted', async () => {
      mockPrisma.user.updateMany.mockResolvedValue({ count: 0 });

      const service = new DataRetentionService(mockPrisma as any);
      await service.cleanupUnverifiedAccounts();

      const infoCalls = (logger.info as jest.Mock).mock.calls.filter((call) =>
        call[0].includes('Soft-deleted')
      );
      expect(infoCalls).toHaveLength(0);
    });
  });

  describe('getConfig', () => {
    it('should return a copy of the config', () => {
      const service = new DataRetentionService(mockPrisma as any);
      const config1 = service.getConfig();
      const config2 = service.getConfig();

      // Should be equal but not the same object
      expect(config1).toEqual(config2);
      expect(config1).not.toBe(config2);
    });
  });

  describe('updateConfig', () => {
    it('should update only provided fields', () => {
      const service = new DataRetentionService(mockPrisma as any);
      const originalConfig = service.getConfig();

      service.updateConfig({ deletedUserRetentionDays: 90 });

      const updatedConfig = service.getConfig();
      expect(updatedConfig.deletedUserRetentionDays).toBe(90);
      expect(updatedConfig.inactiveUserThresholdDays).toBe(
        originalConfig.inactiveUserThresholdDays
      );
    });

    it('should log configuration update', () => {
      const service = new DataRetentionService(mockPrisma as any);
      (logger.info as jest.Mock).mockClear();

      service.updateConfig({ expiredTokenRetentionDays: 14 });

      expect(logger.info).toHaveBeenCalledWith(
        'Retention configuration updated',
        expect.objectContaining({ config: expect.any(Object) })
      );
    });
  });
});

describe('createDataRetentionService', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('should use default config when no env vars set', () => {
    delete process.env.DATA_RETENTION_DELETED_USERS_DAYS;
    delete process.env.DATA_RETENTION_INACTIVE_USERS_DAYS;
    delete process.env.DATA_RETENTION_UNVERIFIED_DAYS;
    delete process.env.DATA_RETENTION_GAME_DATA_MONTHS;
    delete process.env.DATA_RETENTION_SESSION_HOURS;
    delete process.env.DATA_RETENTION_EXPIRED_TOKEN_DAYS;

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config).toEqual(DEFAULT_RETENTION);
  });

  it('should parse valid deletedUserRetentionDays from env', () => {
    process.env.DATA_RETENTION_DELETED_USERS_DAYS = '45';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.deletedUserRetentionDays).toBe(45);
  });

  it('should ignore invalid (NaN) deletedUserRetentionDays', () => {
    process.env.DATA_RETENTION_DELETED_USERS_DAYS = 'invalid';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.deletedUserRetentionDays).toBe(DEFAULT_RETENTION.deletedUserRetentionDays);
  });

  it('should ignore negative deletedUserRetentionDays', () => {
    process.env.DATA_RETENTION_DELETED_USERS_DAYS = '-5';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.deletedUserRetentionDays).toBe(DEFAULT_RETENTION.deletedUserRetentionDays);
  });

  it('should ignore zero deletedUserRetentionDays', () => {
    process.env.DATA_RETENTION_DELETED_USERS_DAYS = '0';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.deletedUserRetentionDays).toBe(DEFAULT_RETENTION.deletedUserRetentionDays);
  });

  it('should parse valid inactiveUserThresholdDays from env', () => {
    process.env.DATA_RETENTION_INACTIVE_USERS_DAYS = '180';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.inactiveUserThresholdDays).toBe(180);
  });

  it('should ignore invalid inactiveUserThresholdDays', () => {
    process.env.DATA_RETENTION_INACTIVE_USERS_DAYS = 'abc';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.inactiveUserThresholdDays).toBe(DEFAULT_RETENTION.inactiveUserThresholdDays);
  });

  it('should parse valid unverifiedAccountRetentionDays from env', () => {
    process.env.DATA_RETENTION_UNVERIFIED_DAYS = '14';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.unverifiedAccountRetentionDays).toBe(14);
  });

  it('should parse valid gameDataRetentionMonths from env', () => {
    process.env.DATA_RETENTION_GAME_DATA_MONTHS = '36';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.gameDataRetentionMonths).toBe(36);
  });

  it('should parse valid sessionDataRetentionHours from env', () => {
    process.env.DATA_RETENTION_SESSION_HOURS = '48';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.sessionDataRetentionHours).toBe(48);
  });

  it('should parse valid expiredTokenRetentionDays from env', () => {
    process.env.DATA_RETENTION_EXPIRED_TOKEN_DAYS = '14';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.expiredTokenRetentionDays).toBe(14);
  });

  it('should parse all env vars when set', () => {
    process.env.DATA_RETENTION_DELETED_USERS_DAYS = '60';
    process.env.DATA_RETENTION_INACTIVE_USERS_DAYS = '730';
    process.env.DATA_RETENTION_UNVERIFIED_DAYS = '3';
    process.env.DATA_RETENTION_GAME_DATA_MONTHS = '12';
    process.env.DATA_RETENTION_SESSION_HOURS = '12';
    process.env.DATA_RETENTION_EXPIRED_TOKEN_DAYS = '30';

    const mockPrisma = {} as any;
    const service = createDataRetentionService(mockPrisma);
    const config = service.getConfig();

    expect(config.deletedUserRetentionDays).toBe(60);
    expect(config.inactiveUserThresholdDays).toBe(730);
    expect(config.unverifiedAccountRetentionDays).toBe(3);
    expect(config.gameDataRetentionMonths).toBe(12);
    expect(config.sessionDataRetentionHours).toBe(12);
    expect(config.expiredTokenRetentionDays).toBe(30);
  });
});
