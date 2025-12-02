/**
 * AIUserService Unit Tests
 *
 * Tests the AI user management service including:
 * - Getting existing AI user
 * - Creating AI user when not exists
 * - Error handling when database unavailable
 */

// Mock the database connection module
jest.mock('../../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(),
}));

// Mock the logger
jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

import { getOrCreateAIUser } from '../../../src/server/services/AIUserService';
import { getDatabaseClient } from '../../../src/server/database/connection';
import { logger } from '../../../src/server/utils/logger';

describe('AIUserService', () => {
  let mockPrisma: any;

  beforeEach(() => {
    jest.clearAllMocks();

    // Setup mock Prisma client
    mockPrisma = {
      user: {
        findUnique: jest.fn(),
        create: jest.fn(),
      },
    };

    (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
  });

  describe('getOrCreateAIUser', () => {
    it('returns existing AI user if found', async () => {
      const existingUser = {
        id: 'ai-user-123',
        email: 'ai@ringrift.internal',
        username: 'RingRift AI',
        role: 'USER',
        rating: 1500,
      };

      mockPrisma.user.findUnique.mockResolvedValue(existingUser);

      const result = await getOrCreateAIUser();

      expect(result).toBe(existingUser);
      expect(mockPrisma.user.findUnique).toHaveBeenCalledTimes(1);
      expect(mockPrisma.user.findUnique).toHaveBeenCalledWith({
        where: { email: 'ai@ringrift.internal' },
      });
      expect(mockPrisma.user.create).not.toHaveBeenCalled();
    });

    it('creates AI user when not found', async () => {
      const newUser = {
        id: 'new-ai-user-456',
        email: 'ai@ringrift.internal',
        username: 'RingRift AI',
        passwordHash: 'AI_USER_NO_LOGIN',
        role: 'USER',
        rating: 1500,
        isActive: true,
        emailVerified: true,
      };

      mockPrisma.user.findUnique.mockResolvedValue(null);
      mockPrisma.user.create.mockResolvedValue(newUser);

      const result = await getOrCreateAIUser();

      expect(result).toBe(newUser);
      expect(mockPrisma.user.findUnique).toHaveBeenCalledTimes(1);
      expect(mockPrisma.user.create).toHaveBeenCalledTimes(1);
      expect(mockPrisma.user.create).toHaveBeenCalledWith({
        data: {
          email: 'ai@ringrift.internal',
          username: 'RingRift AI',
          passwordHash: 'AI_USER_NO_LOGIN',
          role: 'USER',
          rating: 1500,
          isActive: true,
          emailVerified: true,
        },
      });
      expect(logger.info).toHaveBeenCalledWith(
        'AI system user created',
        expect.objectContaining({ userId: 'new-ai-user-456' })
      );
    });

    it('throws error when database is not available', async () => {
      (getDatabaseClient as jest.Mock).mockReturnValue(null);

      await expect(getOrCreateAIUser()).rejects.toThrow('Database not available');
    });

    it('propagates database errors', async () => {
      const dbError = new Error('Connection failed');
      mockPrisma.user.findUnique.mockRejectedValue(dbError);

      await expect(getOrCreateAIUser()).rejects.toThrow('Connection failed');
    });

    it('propagates user creation errors', async () => {
      mockPrisma.user.findUnique.mockResolvedValue(null);
      mockPrisma.user.create.mockRejectedValue(new Error('Unique constraint violation'));

      await expect(getOrCreateAIUser()).rejects.toThrow('Unique constraint violation');
    });
  });
});
