import { ChatMessage as PrismaChatMessage } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';

/**
 * DTO for creating a chat message
 */
export interface CreateChatMessageInput {
  gameId: string;
  userId: string;
  message: string;
}

/**
 * Chat message returned to clients
 */
export interface ChatMessageDTO {
  id: string;
  gameId: string;
  userId: string;
  username: string;
  message: string;
  createdAt: Date;
}

/**
 * Maximum message length (enforced in DB as VARCHAR(500))
 */
const MAX_MESSAGE_LENGTH = 500;

/**
 * Maximum number of messages to return per fetch (prevents memory issues)
 */
const MAX_MESSAGES_PER_FETCH = 100;

/**
 * Service for persisting and retrieving chat messages.
 *
 * Chat messages are stored per-game and can be retrieved on reconnect.
 * This ensures that players who disconnect and reconnect can see the full
 * chat history from their game.
 */
export class ChatPersistenceService {
  /**
   * Save a chat message to the database.
   *
   * @param input - The message data to save
   * @returns The created message DTO
   * @throws Error if message is too long or user/game doesn't exist
   */
  async saveMessage(input: CreateChatMessageInput): Promise<ChatMessageDTO> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    // Validate message length
    if (input.message.length > MAX_MESSAGE_LENGTH) {
      throw new Error(`Message exceeds maximum length of ${MAX_MESSAGE_LENGTH} characters`);
    }

    // Validate message is not empty
    const trimmedMessage = input.message.trim();
    if (trimmedMessage.length === 0) {
      throw new Error('Message cannot be empty');
    }

    try {
      const chatMessage = await prisma.chatMessage.create({
        data: {
          gameId: input.gameId,
          userId: input.userId,
          message: trimmedMessage,
        },
        include: {
          user: {
            select: {
              username: true,
            },
          },
        },
      });

      logger.debug('Chat message saved', {
        messageId: chatMessage.id,
        gameId: input.gameId,
        userId: input.userId,
      });

      return this.toDTO(chatMessage);
    } catch (error) {
      logger.error('Failed to save chat message', {
        error,
        gameId: input.gameId,
        userId: input.userId,
      });
      throw error;
    }
  }

  /**
   * Get all chat messages for a game, ordered by creation time.
   *
   * @param gameId - The game ID to fetch messages for
   * @param limit - Maximum number of messages to return (default: 100)
   * @returns Array of chat message DTOs, oldest first
   */
  async getMessagesForGame(gameId: string, limit?: number): Promise<ChatMessageDTO[]> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }
    const effectiveLimit = Math.min(limit ?? MAX_MESSAGES_PER_FETCH, MAX_MESSAGES_PER_FETCH);

    try {
      const messages = await prisma.chatMessage.findMany({
        where: { gameId },
        orderBy: { createdAt: 'asc' },
        take: effectiveLimit,
        include: {
          user: {
            select: {
              username: true,
            },
          },
        },
      });

      return messages.map((m) => this.toDTO(m));
    } catch (error) {
      logger.error('Failed to fetch chat messages', { error, gameId });
      throw error;
    }
  }

  /**
   * Get messages after a specific timestamp (for incremental loading).
   *
   * @param gameId - The game ID
   * @param afterDate - Only return messages created after this date
   * @param limit - Maximum number of messages to return
   * @returns Array of new chat messages
   */
  async getMessagesSince(
    gameId: string,
    afterDate: Date,
    limit?: number
  ): Promise<ChatMessageDTO[]> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }
    const effectiveLimit = Math.min(limit ?? MAX_MESSAGES_PER_FETCH, MAX_MESSAGES_PER_FETCH);

    try {
      const messages = await prisma.chatMessage.findMany({
        where: {
          gameId,
          createdAt: { gt: afterDate },
        },
        orderBy: { createdAt: 'asc' },
        take: effectiveLimit,
        include: {
          user: {
            select: {
              username: true,
            },
          },
        },
      });

      return messages.map((m) => this.toDTO(m));
    } catch (error) {
      logger.error('Failed to fetch messages since date', { error, gameId, afterDate });
      throw error;
    }
  }

  /**
   * Get the count of messages in a game.
   *
   * @param gameId - The game ID
   * @returns The number of messages
   */
  async getMessageCount(gameId: string): Promise<number> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      return await prisma.chatMessage.count({
        where: { gameId },
      });
    } catch (error) {
      logger.error('Failed to count chat messages', { error, gameId });
      throw error;
    }
  }

  /**
   * Delete all messages for a game (used for cleanup/data retention).
   *
   * @param gameId - The game ID
   * @returns Number of deleted messages
   */
  async deleteMessagesForGame(gameId: string): Promise<number> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const result = await prisma.chatMessage.deleteMany({
        where: { gameId },
      });

      logger.info('Deleted chat messages for game', {
        gameId,
        deletedCount: result.count,
      });

      return result.count;
    } catch (error) {
      logger.error('Failed to delete chat messages', { error, gameId });
      throw error;
    }
  }

  /**
   * Convert a Prisma ChatMessage to a DTO.
   */
  private toDTO(message: PrismaChatMessage & { user: { username: string } }): ChatMessageDTO {
    return {
      id: message.id,
      gameId: message.gameId,
      userId: message.userId,
      username: message.user.username,
      message: message.message,
      createdAt: message.createdAt,
    };
  }
}

// Singleton instance
let chatPersistenceServiceInstance: ChatPersistenceService | null = null;

/**
 * Get the singleton ChatPersistenceService instance.
 */
export function getChatPersistenceService(): ChatPersistenceService {
  if (!chatPersistenceServiceInstance) {
    chatPersistenceServiceInstance = new ChatPersistenceService();
  }
  return chatPersistenceServiceInstance;
}
