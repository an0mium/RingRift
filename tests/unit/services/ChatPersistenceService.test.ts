/**
 * ChatPersistenceService Unit Tests
 *
 * Tests for the chat message persistence service including:
 * - Saving messages (success and validation)
 * - Fetching messages for a game
 * - Fetching messages since a timestamp
 * - Message count
 * - Deleting messages
 * - Database unavailable handling
 * - Singleton factory function
 */

import {
  ChatPersistenceService,
  getChatPersistenceService,
} from '../../../src/server/services/ChatPersistenceService';

// Mock dependencies
const mockPrisma = {
  chatMessage: {
    create: jest.fn(),
    findMany: jest.fn(),
    count: jest.fn(),
    deleteMany: jest.fn(),
  },
};

jest.mock('../../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => mockPrisma),
}));

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

import { logger } from '../../../src/server/utils/logger';

describe('ChatPersistenceService', () => {
  let service: ChatPersistenceService;

  const mockMessage = {
    id: 'msg-1',
    gameId: 'game-1',
    userId: 'user-1',
    message: 'Hello world!',
    createdAt: new Date(),
    user: { username: 'TestUser' },
  };

  beforeEach(() => {
    jest.clearAllMocks();
    service = new ChatPersistenceService();
  });

  describe('saveMessage', () => {
    it('should save a message successfully', async () => {
      mockPrisma.chatMessage.create.mockResolvedValue(mockMessage);

      const result = await service.saveMessage({
        gameId: 'game-1',
        userId: 'user-1',
        message: 'Hello world!',
      });

      expect(result.id).toBe('msg-1');
      expect(result.message).toBe('Hello world!');
      expect(result.username).toBe('TestUser');
      expect(mockPrisma.chatMessage.create).toHaveBeenCalledWith({
        data: {
          gameId: 'game-1',
          userId: 'user-1',
          message: 'Hello world!',
        },
        include: {
          user: {
            select: { username: true },
          },
        },
      });
    });

    it('should trim whitespace from message', async () => {
      mockPrisma.chatMessage.create.mockResolvedValue({
        ...mockMessage,
        message: 'Trimmed message',
      });

      await service.saveMessage({
        gameId: 'game-1',
        userId: 'user-1',
        message: '  Trimmed message  ',
      });

      expect(mockPrisma.chatMessage.create).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            message: 'Trimmed message',
          }),
        })
      );
    });

    it('should reject message exceeding max length', async () => {
      const longMessage = 'a'.repeat(501);

      await expect(
        service.saveMessage({
          gameId: 'game-1',
          userId: 'user-1',
          message: longMessage,
        })
      ).rejects.toThrow('exceeds maximum length');
    });

    it('should reject empty message', async () => {
      await expect(
        service.saveMessage({
          gameId: 'game-1',
          userId: 'user-1',
          message: '',
        })
      ).rejects.toThrow('cannot be empty');
    });

    it('should reject whitespace-only message', async () => {
      await expect(
        service.saveMessage({
          gameId: 'game-1',
          userId: 'user-1',
          message: '   ',
        })
      ).rejects.toThrow('cannot be empty');
    });

    it('should log debug message on success', async () => {
      mockPrisma.chatMessage.create.mockResolvedValue(mockMessage);

      await service.saveMessage({
        gameId: 'game-1',
        userId: 'user-1',
        message: 'Test',
      });

      expect(logger.debug).toHaveBeenCalledWith(
        'Chat message saved',
        expect.objectContaining({
          messageId: 'msg-1',
          gameId: 'game-1',
          userId: 'user-1',
        })
      );
    });

    it('should log error and rethrow on database error', async () => {
      const dbError = new Error('Database connection failed');
      mockPrisma.chatMessage.create.mockRejectedValue(dbError);

      await expect(
        service.saveMessage({
          gameId: 'game-1',
          userId: 'user-1',
          message: 'Test',
        })
      ).rejects.toThrow('Database connection failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to save chat message',
        expect.objectContaining({
          error: dbError,
          gameId: 'game-1',
          userId: 'user-1',
        })
      );
    });
  });

  describe('getMessagesForGame', () => {
    it('should return messages for a game', async () => {
      const messages = [mockMessage, { ...mockMessage, id: 'msg-2' }];
      mockPrisma.chatMessage.findMany.mockResolvedValue(messages);

      const result = await service.getMessagesForGame('game-1');

      expect(result).toHaveLength(2);
      expect(result[0].gameId).toBe('game-1');
      expect(mockPrisma.chatMessage.findMany).toHaveBeenCalledWith({
        where: { gameId: 'game-1' },
        orderBy: { createdAt: 'asc' },
        take: 100,
        include: {
          user: {
            select: { username: true },
          },
        },
      });
    });

    it('should respect custom limit up to max', async () => {
      mockPrisma.chatMessage.findMany.mockResolvedValue([]);

      await service.getMessagesForGame('game-1', 50);

      expect(mockPrisma.chatMessage.findMany).toHaveBeenCalledWith(
        expect.objectContaining({ take: 50 })
      );
    });

    it('should cap limit at max messages per fetch', async () => {
      mockPrisma.chatMessage.findMany.mockResolvedValue([]);

      await service.getMessagesForGame('game-1', 500);

      expect(mockPrisma.chatMessage.findMany).toHaveBeenCalledWith(
        expect.objectContaining({ take: 100 })
      );
    });

    it('should return empty array when no messages', async () => {
      mockPrisma.chatMessage.findMany.mockResolvedValue([]);

      const result = await service.getMessagesForGame('game-1');

      expect(result).toEqual([]);
    });

    it('should log error and rethrow on database error', async () => {
      const dbError = new Error('Query failed');
      mockPrisma.chatMessage.findMany.mockRejectedValue(dbError);

      await expect(service.getMessagesForGame('game-1')).rejects.toThrow('Query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to fetch chat messages',
        expect.objectContaining({
          error: dbError,
          gameId: 'game-1',
        })
      );
    });
  });

  describe('getMessagesSince', () => {
    it('should return messages after the given date', async () => {
      const afterDate = new Date('2024-01-01');
      mockPrisma.chatMessage.findMany.mockResolvedValue([mockMessage]);

      const result = await service.getMessagesSince('game-1', afterDate);

      expect(result).toHaveLength(1);
      expect(mockPrisma.chatMessage.findMany).toHaveBeenCalledWith({
        where: {
          gameId: 'game-1',
          createdAt: { gt: afterDate },
        },
        orderBy: { createdAt: 'asc' },
        take: 100,
        include: {
          user: {
            select: { username: true },
          },
        },
      });
    });

    it('should respect custom limit', async () => {
      mockPrisma.chatMessage.findMany.mockResolvedValue([]);

      await service.getMessagesSince('game-1', new Date(), 25);

      expect(mockPrisma.chatMessage.findMany).toHaveBeenCalledWith(
        expect.objectContaining({ take: 25 })
      );
    });

    it('should cap limit at max messages per fetch', async () => {
      mockPrisma.chatMessage.findMany.mockResolvedValue([]);

      await service.getMessagesSince('game-1', new Date(), 200);

      expect(mockPrisma.chatMessage.findMany).toHaveBeenCalledWith(
        expect.objectContaining({ take: 100 })
      );
    });

    it('should log error and rethrow on database error', async () => {
      const dbError = new Error('Query failed');
      const afterDate = new Date();
      mockPrisma.chatMessage.findMany.mockRejectedValue(dbError);

      await expect(service.getMessagesSince('game-1', afterDate)).rejects.toThrow('Query failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to fetch messages since date',
        expect.objectContaining({
          error: dbError,
          gameId: 'game-1',
          afterDate,
        })
      );
    });
  });

  describe('getMessageCount', () => {
    it('should return the message count', async () => {
      mockPrisma.chatMessage.count.mockResolvedValue(42);

      const result = await service.getMessageCount('game-1');

      expect(result).toBe(42);
      expect(mockPrisma.chatMessage.count).toHaveBeenCalledWith({
        where: { gameId: 'game-1' },
      });
    });

    it('should return 0 when no messages', async () => {
      mockPrisma.chatMessage.count.mockResolvedValue(0);

      const result = await service.getMessageCount('game-1');

      expect(result).toBe(0);
    });

    it('should log error and rethrow on database error', async () => {
      const dbError = new Error('Count failed');
      mockPrisma.chatMessage.count.mockRejectedValue(dbError);

      await expect(service.getMessageCount('game-1')).rejects.toThrow('Count failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to count chat messages',
        expect.objectContaining({
          error: dbError,
          gameId: 'game-1',
        })
      );
    });
  });

  describe('deleteMessagesForGame', () => {
    it('should delete messages and return count', async () => {
      mockPrisma.chatMessage.deleteMany.mockResolvedValue({ count: 10 });

      const result = await service.deleteMessagesForGame('game-1');

      expect(result).toBe(10);
      expect(mockPrisma.chatMessage.deleteMany).toHaveBeenCalledWith({
        where: { gameId: 'game-1' },
      });
    });

    it('should log deletion', async () => {
      mockPrisma.chatMessage.deleteMany.mockResolvedValue({ count: 5 });

      await service.deleteMessagesForGame('game-1');

      expect(logger.info).toHaveBeenCalledWith(
        'Deleted chat messages for game',
        expect.objectContaining({
          gameId: 'game-1',
          deletedCount: 5,
        })
      );
    });

    it('should return 0 when no messages to delete', async () => {
      mockPrisma.chatMessage.deleteMany.mockResolvedValue({ count: 0 });

      const result = await service.deleteMessagesForGame('game-1');

      expect(result).toBe(0);
    });

    it('should log error and rethrow on database error', async () => {
      const dbError = new Error('Delete failed');
      mockPrisma.chatMessage.deleteMany.mockRejectedValue(dbError);

      await expect(service.deleteMessagesForGame('game-1')).rejects.toThrow('Delete failed');

      expect(logger.error).toHaveBeenCalledWith(
        'Failed to delete chat messages',
        expect.objectContaining({
          error: dbError,
          gameId: 'game-1',
        })
      );
    });
  });
});

describe('ChatPersistenceService - Database unavailable', () => {
  let service: ChatPersistenceService;
  const { getDatabaseClient } = require('../../../src/server/database/connection');

  beforeEach(() => {
    jest.clearAllMocks();
    service = new ChatPersistenceService();
    // Reset to null for database unavailable tests
    getDatabaseClient.mockReturnValue(null);
  });

  afterEach(() => {
    // Restore mock to return prisma
    getDatabaseClient.mockReturnValue(mockPrisma);
  });

  it('should throw on saveMessage when database unavailable', async () => {
    await expect(
      service.saveMessage({
        gameId: 'game-1',
        userId: 'user-1',
        message: 'Test',
      })
    ).rejects.toThrow('Database not available');
  });

  it('should throw on getMessagesForGame when database unavailable', async () => {
    await expect(service.getMessagesForGame('game-1')).rejects.toThrow('Database not available');
  });

  it('should throw on getMessagesSince when database unavailable', async () => {
    await expect(service.getMessagesSince('game-1', new Date())).rejects.toThrow(
      'Database not available'
    );
  });

  it('should throw on getMessageCount when database unavailable', async () => {
    await expect(service.getMessageCount('game-1')).rejects.toThrow('Database not available');
  });

  it('should throw on deleteMessagesForGame when database unavailable', async () => {
    await expect(service.deleteMessagesForGame('game-1')).rejects.toThrow('Database not available');
  });
});

describe('getChatPersistenceService singleton', () => {
  it('should return a ChatPersistenceService instance', () => {
    const instance = getChatPersistenceService();
    expect(instance).toBeInstanceOf(ChatPersistenceService);
  });

  it('should return the same instance on multiple calls', () => {
    const instance1 = getChatPersistenceService();
    const instance2 = getChatPersistenceService();
    expect(instance1).toBe(instance2);
  });
});
