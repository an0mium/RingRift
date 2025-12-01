import { RematchRequest as PrismaRematchRequest, RematchRequestStatus } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { logger } from '../utils/logger';

/**
 * Rematch request timeout in milliseconds (30 seconds)
 */
const REMATCH_TIMEOUT_MS = 30_000;

/**
 * DTO for rematch request data sent to clients
 */
export interface RematchRequestDTO {
  id: string;
  gameId: string;
  requesterId: string;
  requesterUsername: string;
  status: RematchRequestStatus;
  expiresAt: Date;
  createdAt: Date;
  respondedAt?: Date | null;
  newGameId?: string | null;
}

/**
 * Result of creating a rematch request
 */
export interface CreateRematchResult {
  success: boolean;
  request?: RematchRequestDTO;
  error?: string;
}

/**
 * Result of responding to a rematch request
 */
export interface RematchResponseResult {
  success: boolean;
  request?: RematchRequestDTO;
  newGameId?: string;
  error?: string;
}

/**
 * Service for managing rematch requests.
 *
 * Rematch flow:
 * 1. After a game ends, any player can request a rematch
 * 2. For 2-player games: single accept starts a new game immediately
 * 3. For 3-4 player games: all other players must accept within 30 seconds
 * 4. Requests expire after 30 seconds if not fully accepted
 */
export class RematchService {
  /**
   * Create a rematch request for a completed game.
   *
   * @param gameId - The ID of the completed game
   * @param requesterId - The ID of the player requesting rematch
   * @returns The created rematch request or error
   */
  async createRematchRequest(gameId: string, requesterId: string): Promise<CreateRematchResult> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      return { success: false, error: 'Database not available' };
    }

    try {
      // Check if there's already a pending rematch for this game
      const existingPending = await prisma.rematchRequest.findFirst({
        where: {
          gameId,
          status: 'pending',
        },
      });

      if (existingPending) {
        return {
          success: false,
          error: 'A rematch request is already pending for this game',
        };
      }

      // Verify the game exists and is completed
      const game = await prisma.game.findUnique({
        where: { id: gameId },
        select: {
          id: true,
          status: true,
          player1Id: true,
          player2Id: true,
          player3Id: true,
          player4Id: true,
        },
      });

      if (!game) {
        return { success: false, error: 'Game not found' };
      }

      if (game.status !== 'completed' && game.status !== 'finished') {
        return { success: false, error: 'Can only request rematch for completed games' };
      }

      // Verify the requester was a player in the game
      const playerIds = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
        Boolean
      );
      if (!playerIds.includes(requesterId)) {
        return { success: false, error: 'Only players in the game can request a rematch' };
      }

      // Create the rematch request
      const expiresAt = new Date(Date.now() + REMATCH_TIMEOUT_MS);
      const request = await prisma.rematchRequest.create({
        data: {
          gameId,
          requesterId,
          status: 'pending',
          expiresAt,
        },
        include: {
          requester: {
            select: { username: true },
          },
        },
      });

      logger.info('Rematch request created', {
        requestId: request.id,
        gameId,
        requesterId,
        expiresAt,
      });

      return {
        success: true,
        request: this.toDTO(request),
      };
    } catch (error) {
      logger.error('Failed to create rematch request', { error, gameId, requesterId });
      return { success: false, error: 'Failed to create rematch request' };
    }
  }

  /**
   * Accept a rematch request.
   *
   * @param requestId - The rematch request ID
   * @param accepterId - The ID of the player accepting
   * @param createGameFn - Function to create the new game
   * @returns Result with the new game ID if all players accepted
   */
  async acceptRematch(
    requestId: string,
    accepterId: string,
    createGameFn: (originalGameId: string) => Promise<string>
  ): Promise<RematchResponseResult> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      return { success: false, error: 'Database not available' };
    }

    try {
      const request = await prisma.rematchRequest.findUnique({
        where: { id: requestId },
        include: {
          game: {
            select: {
              id: true,
              boardType: true,
              maxPlayers: true,
              player1Id: true,
              player2Id: true,
              player3Id: true,
              player4Id: true,
            },
          },
          requester: {
            select: { username: true },
          },
        },
      });

      if (!request) {
        return { success: false, error: 'Rematch request not found' };
      }

      if (request.status !== 'pending') {
        return { success: false, error: `Rematch request is already ${request.status}` };
      }

      if (new Date() > request.expiresAt) {
        // Mark as expired
        await prisma.rematchRequest.update({
          where: { id: requestId },
          data: { status: 'expired', respondedAt: new Date() },
        });
        return { success: false, error: 'Rematch request has expired' };
      }

      // Verify the accepter was a player in the game
      const playerIds = [
        request.game.player1Id,
        request.game.player2Id,
        request.game.player3Id,
        request.game.player4Id,
      ].filter(Boolean);

      if (!playerIds.includes(accepterId)) {
        return { success: false, error: 'Only players in the game can accept a rematch' };
      }

      // Cannot accept your own request
      if (accepterId === request.requesterId) {
        return { success: false, error: 'Cannot accept your own rematch request' };
      }

      // For 2-player games, immediately create new game
      // For 3-4 player games, would need more complex tracking (simplified for now)
      const newGameId = await createGameFn(request.gameId);

      // Update request as accepted
      const updatedRequest = await prisma.rematchRequest.update({
        where: { id: requestId },
        data: {
          status: 'accepted',
          respondedAt: new Date(),
          newGameId,
        },
        include: {
          requester: {
            select: { username: true },
          },
        },
      });

      logger.info('Rematch accepted', {
        requestId,
        accepterId,
        newGameId,
      });

      return {
        success: true,
        request: this.toDTO(updatedRequest),
        newGameId,
      };
    } catch (error) {
      logger.error('Failed to accept rematch', { error, requestId, accepterId });
      return { success: false, error: 'Failed to accept rematch' };
    }
  }

  /**
   * Decline a rematch request.
   *
   * @param requestId - The rematch request ID
   * @param declinerId - The ID of the player declining
   * @returns Result of the decline
   */
  async declineRematch(requestId: string, declinerId: string): Promise<RematchResponseResult> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      return { success: false, error: 'Database not available' };
    }

    try {
      const request = await prisma.rematchRequest.findUnique({
        where: { id: requestId },
        include: {
          game: {
            select: {
              player1Id: true,
              player2Id: true,
              player3Id: true,
              player4Id: true,
            },
          },
          requester: {
            select: { username: true },
          },
        },
      });

      if (!request) {
        return { success: false, error: 'Rematch request not found' };
      }

      if (request.status !== 'pending') {
        return { success: false, error: `Rematch request is already ${request.status}` };
      }

      // Verify the decliner was a player in the game
      const playerIds = [
        request.game.player1Id,
        request.game.player2Id,
        request.game.player3Id,
        request.game.player4Id,
      ].filter(Boolean);

      if (!playerIds.includes(declinerId)) {
        return { success: false, error: 'Only players in the game can decline a rematch' };
      }

      const updatedRequest = await prisma.rematchRequest.update({
        where: { id: requestId },
        data: {
          status: 'declined',
          respondedAt: new Date(),
        },
        include: {
          requester: {
            select: { username: true },
          },
        },
      });

      logger.info('Rematch declined', { requestId, declinerId });

      return {
        success: true,
        request: this.toDTO(updatedRequest),
      };
    } catch (error) {
      logger.error('Failed to decline rematch', { error, requestId, declinerId });
      return { success: false, error: 'Failed to decline rematch' };
    }
  }

  /**
   * Get the current pending rematch request for a game, if any.
   *
   * @param gameId - The game ID
   * @returns The pending request or null
   */
  async getPendingRequest(gameId: string): Promise<RematchRequestDTO | null> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const request = await prisma.rematchRequest.findFirst({
        where: {
          gameId,
          status: 'pending',
          expiresAt: { gt: new Date() },
        },
        include: {
          requester: {
            select: { username: true },
          },
        },
      });

      if (!request) return null;

      return this.toDTO(request);
    } catch (error) {
      logger.error('Failed to get pending rematch request', { error, gameId });
      throw error;
    }
  }

  /**
   * Expire all pending rematch requests that have passed their expiration time.
   * Should be called periodically by a cleanup job.
   *
   * @returns Number of expired requests
   */
  async expireOldRequests(): Promise<number> {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw new Error('Database not available');
    }

    try {
      const result = await prisma.rematchRequest.updateMany({
        where: {
          status: 'pending',
          expiresAt: { lt: new Date() },
        },
        data: {
          status: 'expired',
          respondedAt: new Date(),
        },
      });

      if (result.count > 0) {
        logger.info('Expired old rematch requests', { count: result.count });
      }

      return result.count;
    } catch (error) {
      logger.error('Failed to expire old rematch requests', { error });
      throw error;
    }
  }

  /**
   * Convert a Prisma RematchRequest to a DTO.
   */
  private toDTO(
    request: PrismaRematchRequest & { requester: { username: string } }
  ): RematchRequestDTO {
    return {
      id: request.id,
      gameId: request.gameId,
      requesterId: request.requesterId,
      requesterUsername: request.requester.username,
      status: request.status,
      expiresAt: request.expiresAt,
      createdAt: request.createdAt,
      respondedAt: request.respondedAt,
      newGameId: request.newGameId,
    };
  }
}

// Singleton instance
let rematchServiceInstance: RematchService | null = null;

/**
 * Get the singleton RematchService instance.
 */
export function getRematchService(): RematchService {
  if (!rematchServiceInstance) {
    rematchServiceInstance = new RematchService();
  }
  return rematchServiceInstance;
}
