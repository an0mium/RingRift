/**
 * @file ChatPersistenceService.test.ts
 * @description Comprehensive unit tests for ChatPersistenceService
 * covering all branch paths including database availability checks,
 * message validation, and limit handling.
 */

// Mock dependencies before imports
const mockLogger = {
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  debug: jest.fn(),
};

jest.mock('../../src/server/utils/logger', () => ({
  logger: mockLogger,
}));

// Mock database connection
let mockPrismaClient: {
  chatMessage: {
    create: jest.Mock;
    findMany: jest.Mock;
    count: jest.Mock;
    deleteMany: jest.Mock;
  };
} | null = null;

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => mockPrismaClient),
}));

import {
  ChatPersistenceService,
  getChatPersistenceService,
} from '../../src/server/services/ChatPersistenceService';

describe('ChatPersistenceService', () => {
  let service: ChatPersistenceService;

  beforeEach(() => {
    jest.clearAllMocks();
    service = new ChatPersistenceService();
    // Default to having a valid prisma client
    mockPrismaClient = {
      chatMessage: {
        create: jest.fn(),
        findMany: jest.fn(),
        count: jest.fn(),
        deleteMany: jest.fn(),
      },
    };
  });

  describe('getChatPersistenceService singleton', () => {
    it('should return the same instance on multiple calls', () => {
      const instance1 = getChatPersistenceService();
      const instance2 = getChatPersistenceService();
      expect(instance1).toBe(instance2);
    });
  });

  describe('saveMessage', () => {
    const validInput = {
      gameId: 'game-123',
      userId: 'user-456',
      message: 'Hello, world!',
    };

    it('should throw when database is not available', async () => {
      mockPrismaClient = null;

      await expect(service.saveMessage(validInput)).rejects.toThrow('Database not available');
    });

    it('should throw when message exceeds maximum length', async () => {
      const longMessage = 'a'.repeat(501);

      await expect(service.saveMessage({ ...validInput, message: longMessage })).rejects.toThrow(
        'Message exceeds maximum length of 500 characters'
      );
    });

    it('should throw when message is empty after trimming', async () => {
      await expect(service.saveMessage({ ...validInput, message: '   ' })).rejects.toThrow(
        'Message cannot be empty'
      );
    });

    it('should throw when message is only whitespace', async () => {
      await expect(service.saveMessage({ ...validInput, message: '\t\n  ' })).rejects.toThrow(
        'Message cannot be empty'
      );
    });

    it('should save message and return DTO on success', async () => {
      const mockCreatedMessage = {
        id: 'msg-123',
        gameId: 'game-123',
        userId: 'user-456',
        message: 'Hello, world!',
        createdAt: new Date('2024-01-01T12:00:00Z'),
        user: { username: 'testuser' },
      };
      mockPrismaClient!.chatMessage.create.mockResolvedValueOnce(mockCreatedMessage);

      const result = await service.saveMessage(validInput);

      expect(mockPrismaClient!.chatMessage.create).toHaveBeenCalledWith({
        data: {
          gameId: 'game-123',
          userId: 'user-456',
          message: 'Hello, world!',
        },
        include: {
          user: {
            select: {
              username: true,
            },
          },
        },
      });

      expect(result).toEqual({
        id: 'msg-123',
        gameId: 'game-123',
        userId: 'user-456',
        username: 'testuser',
        message: 'Hello, world!',
        createdAt: new Date('2024-01-01T12:00:00Z'),
      });

      expect(mockLogger.debug).toHaveBeenCalledWith('Chat message saved', {
        messageId: 'msg-123',
        gameId: 'game-123',
        userId: 'user-456',
      });
    });

    it('should trim message before saving', async () => {
      const mockCreatedMessage = {
        id: 'msg-123',
        gameId: 'game-123',
        userId: 'user-456',
        message: 'Trimmed message',
        createdAt: new Date('2024-01-01T12:00:00Z'),
        user: { username: 'testuser' },
      };
      mockPrismaClient!.chatMessage.create.mockResolvedValueOnce(mockCreatedMessage);

      await service.saveMessage({ ...validInput, message: '  Trimmed message  ' });

      expect(mockPrismaClient!.chatMessage.create).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            message: 'Trimmed message',
          }),
        })
      );
    });

    it('should log and rethrow database errors', async () => {
      const dbError = new Error('Foreign key constraint failed');
      mockPrismaClient!.chatMessage.create.mockRejectedValueOnce(dbError);

      await expect(service.saveMessage(validInput)).rejects.toThrow(
        'Foreign key constraint failed'
      );

      expect(mockLogger.error).toHaveBeenCalledWith('Failed to save chat message', {
        error: dbError,
        gameId: 'game-123',
        userId: 'user-456',
      });
    });

    it('should accept message at exactly maximum length', async () => {
      const exactLengthMessage = 'a'.repeat(500);
      const mockCreatedMessage = {
        id: 'msg-123',
        gameId: 'game-123',
        userId: 'user-456',
        message: exactLengthMessage,
        createdAt: new Date(),
        user: { username: 'testuser' },
      };
      mockPrismaClient!.chatMessage.create.mockResolvedValueOnce(mockCreatedMessage);

      const result = await service.saveMessage({ ...validInput, message: exactLengthMessage });

      expect(result.message).toBe(exactLengthMessage);
    });
  });

  describe('getMessagesForGame', () => {
    const gameId = 'game-123';

    it('should throw when database is not available', async () => {
      mockPrismaClient = null;

      await expect(service.getMessagesForGame(gameId)).rejects.toThrow('Database not available');
    });

    it('should use default limit when not provided', async () => {
      const mockMessages: Array<{
        id: string;
        gameId: string;
        userId: string;
        message: string;
        createdAt: Date;
        user: { username: string };
      }> = [];
      mockPrismaClient!.chatMessage.findMany.mockResolvedValueOnce(mockMessages);

      await service.getMessagesForGame(gameId);

      expect(mockPrismaClient!.chatMessage.findMany).toHaveBeenCalledWith({
        where: { gameId },
        orderBy: { createdAt: 'asc' },
        take: 100,
        include: {
          user: {
            select: {
              username: true,
            },
          },
        },
      });
    });

    it('should use provided limit when less than max', async () => {
      const mockMessages: Array<{
        id: string;
        gameId: string;
        userId: string;
        message: string;
        createdAt: Date;
        user: { username: string };
      }> = [];
      mockPrismaClient!.chatMessage.findMany.mockResolvedValueOnce(mockMessages);

      await service.getMessagesForGame(gameId, 50);

      expect(mockPrismaClient!.chatMessage.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 50,
        })
      );
    });

    it('should cap limit at MAX_MESSAGES_PER_FETCH when limit exceeds it', async () => {
      const mockMessages: Array<{
        id: string;
        gameId: string;
        userId: string;
        message: string;
        createdAt: Date;
        user: { username: string };
      }> = [];
      mockPrismaClient!.chatMessage.findMany.mockResolvedValueOnce(mockMessages);

      await service.getMessagesForGame(gameId, 500);

      expect(mockPrismaClient!.chatMessage.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 100, // Capped at MAX_MESSAGES_PER_FETCH
        })
      );
    });

    it('should return mapped DTOs', async () => {
      const mockMessages = [
        {
          id: 'msg-1',
          gameId: 'game-123',
          userId: 'user-1',
          message: 'First message',
          createdAt: new Date('2024-01-01T12:00:00Z'),
          user: { username: 'alice' },
        },
        {
          id: 'msg-2',
          gameId: 'game-123',
          userId: 'user-2',
          message: 'Second message',
          createdAt: new Date('2024-01-01T12:01:00Z'),
          user: { username: 'bob' },
        },
      ];
      mockPrismaClient!.chatMessage.findMany.mockResolvedValueOnce(mockMessages);

      const result = await service.getMessagesForGame(gameId);

      expect(result).toEqual([
        {
          id: 'msg-1',
          gameId: 'game-123',
          userId: 'user-1',
          username: 'alice',
          message: 'First message',
          createdAt: new Date('2024-01-01T12:00:00Z'),
        },
        {
          id: 'msg-2',
          gameId: 'game-123',
          userId: 'user-2',
          username: 'bob',
          message: 'Second message',
          createdAt: new Date('2024-01-01T12:01:00Z'),
        },
      ]);
    });

    it('should log and rethrow database errors', async () => {
      const dbError = new Error('Connection lost');
      mockPrismaClient!.chatMessage.findMany.mockRejectedValueOnce(dbError);

      await expect(service.getMessagesForGame(gameId)).rejects.toThrow('Connection lost');

      expect(mockLogger.error).toHaveBeenCalledWith('Failed to fetch chat messages', {
        error: dbError,
        gameId,
      });
    });
  });

  describe('getMessagesSince', () => {
    const gameId = 'game-123';
    const afterDate = new Date('2024-01-01T12:00:00Z');

    it('should throw when database is not available', async () => {
      mockPrismaClient = null;

      await expect(service.getMessagesSince(gameId, afterDate)).rejects.toThrow(
        'Database not available'
      );
    });

    it('should use default limit when not provided', async () => {
      const mockMessages: Array<{
        id: string;
        gameId: string;
        userId: string;
        message: string;
        createdAt: Date;
        user: { username: string };
      }> = [];
      mockPrismaClient!.chatMessage.findMany.mockResolvedValueOnce(mockMessages);

      await service.getMessagesSince(gameId, afterDate);

      expect(mockPrismaClient!.chatMessage.findMany).toHaveBeenCalledWith({
        where: {
          gameId,
          createdAt: { gt: afterDate },
        },
        orderBy: { createdAt: 'asc' },
        take: 100,
        include: {
          user: {
            select: {
              username: true,
            },
          },
        },
      });
    });

    it('should use provided limit when less than max', async () => {
      const mockMessages: Array<{
        id: string;
        gameId: string;
        userId: string;
        message: string;
        createdAt: Date;
        user: { username: string };
      }> = [];
      mockPrismaClient!.chatMessage.findMany.mockResolvedValueOnce(mockMessages);

      await service.getMessagesSince(gameId, afterDate, 25);

      expect(mockPrismaClient!.chatMessage.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 25,
        })
      );
    });

    it('should cap limit at MAX_MESSAGES_PER_FETCH', async () => {
      const mockMessages: Array<{
        id: string;
        gameId: string;
        userId: string;
        message: string;
        createdAt: Date;
        user: { username: string };
      }> = [];
      mockPrismaClient!.chatMessage.findMany.mockResolvedValueOnce(mockMessages);

      await service.getMessagesSince(gameId, afterDate, 200);

      expect(mockPrismaClient!.chatMessage.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 100, // Capped
        })
      );
    });

    it('should return mapped DTOs', async () => {
      const mockMessages = [
        {
          id: 'msg-3',
          gameId: 'game-123',
          userId: 'user-1',
          message: 'New message',
          createdAt: new Date('2024-01-01T12:05:00Z'),
          user: { username: 'alice' },
        },
      ];
      mockPrismaClient!.chatMessage.findMany.mockResolvedValueOnce(mockMessages);

      const result = await service.getMessagesSince(gameId, afterDate);

      expect(result).toEqual([
        {
          id: 'msg-3',
          gameId: 'game-123',
          userId: 'user-1',
          username: 'alice',
          message: 'New message',
          createdAt: new Date('2024-01-01T12:05:00Z'),
        },
      ]);
    });

    it('should log and rethrow database errors', async () => {
      const dbError = new Error('Timeout');
      mockPrismaClient!.chatMessage.findMany.mockRejectedValueOnce(dbError);

      await expect(service.getMessagesSince(gameId, afterDate)).rejects.toThrow('Timeout');

      expect(mockLogger.error).toHaveBeenCalledWith('Failed to fetch messages since date', {
        error: dbError,
        gameId,
        afterDate,
      });
    });
  });

  describe('getMessageCount', () => {
    const gameId = 'game-123';

    it('should throw when database is not available', async () => {
      mockPrismaClient = null;

      await expect(service.getMessageCount(gameId)).rejects.toThrow('Database not available');
    });

    it('should return count from database', async () => {
      mockPrismaClient!.chatMessage.count.mockResolvedValueOnce(42);

      const result = await service.getMessageCount(gameId);

      expect(mockPrismaClient!.chatMessage.count).toHaveBeenCalledWith({
        where: { gameId },
      });
      expect(result).toBe(42);
    });

    it('should log and rethrow database errors', async () => {
      const dbError = new Error('Count failed');
      mockPrismaClient!.chatMessage.count.mockRejectedValueOnce(dbError);

      await expect(service.getMessageCount(gameId)).rejects.toThrow('Count failed');

      expect(mockLogger.error).toHaveBeenCalledWith('Failed to count chat messages', {
        error: dbError,
        gameId,
      });
    });
  });

  describe('deleteMessagesForGame', () => {
    const gameId = 'game-123';

    it('should throw when database is not available', async () => {
      mockPrismaClient = null;

      await expect(service.deleteMessagesForGame(gameId)).rejects.toThrow('Database not available');
    });

    it('should delete messages and return count', async () => {
      mockPrismaClient!.chatMessage.deleteMany.mockResolvedValueOnce({ count: 15 });

      const result = await service.deleteMessagesForGame(gameId);

      expect(mockPrismaClient!.chatMessage.deleteMany).toHaveBeenCalledWith({
        where: { gameId },
      });
      expect(result).toBe(15);

      expect(mockLogger.info).toHaveBeenCalledWith('Deleted chat messages for game', {
        gameId,
        deletedCount: 15,
      });
    });

    it('should log and rethrow database errors', async () => {
      const dbError = new Error('Delete failed');
      mockPrismaClient!.chatMessage.deleteMany.mockRejectedValueOnce(dbError);

      await expect(service.deleteMessagesForGame(gameId)).rejects.toThrow('Delete failed');

      expect(mockLogger.error).toHaveBeenCalledWith('Failed to delete chat messages', {
        error: dbError,
        gameId,
      });
    });
  });
});
