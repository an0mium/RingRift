import {
  MatchmakingPreferences,
  MatchmakingStatus,
  WebSocketErrorPayload,
} from '../../shared/types/websocket';
import { BoardType as PrismaBoardType, GameStatus, MatchmakingOutcome } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { WebSocketServer } from '../websocket/server';
import { logger } from '../utils/logger';
import { v4 as uuidv4 } from 'uuid';
import crypto from 'crypto';
import os from 'os';
import { generateGameSeed } from '../../shared/utils/rng';

interface QueueEntry {
  id: string;
  ticketId: string;
  userId: string;
  socketId: string;
  preferences: MatchmakingPreferences;
  rating: number;
  joinedAt: Date;
  matchCreationInProgress?: boolean;
}

/**
 * Persistent Matchmaking Service
 *
 * Extends the basic MatchmakingService with database persistence:
 * - Queue state survives server restarts
 * - Multi-instance support via heartbeats
 * - Analytics tracking for match quality
 */
export class PersistentMatchmakingService {
  private localQueue: Map<string, QueueEntry> = new Map(); // userId -> entry
  private matchCheckInterval: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private readonly MATCH_CHECK_INTERVAL_MS = 5000;
  private readonly HEARTBEAT_INTERVAL_MS = 10000;
  private readonly RATING_EXPANSION_RATE = 50;
  private readonly MAX_WAIT_TIME_MS = 60000;
  private readonly serverId: string;

  constructor(private wsServer: WebSocketServer) {
    this.serverId = `${os.hostname()}-${process.pid}`;
    this.initialize();
  }

  private async initialize() {
    await this.loadQueueFromDB();
    this.startMatchmakingLoop();
    this.startHeartbeatLoop();

    logger.info('PersistentMatchmakingService initialized', {
      serverId: this.serverId,
      localQueueSize: this.localQueue.size,
    });
  }

  /**
   * Load queue entries from database on startup.
   * Claims orphaned entries whose server has stopped heartbeating.
   */
  private async loadQueueFromDB() {
    try {
      const prisma = getDatabaseClient();
      if (!prisma) return;

      const staleThreshold = new Date(Date.now() - this.HEARTBEAT_INTERVAL_MS * 3);

      // Find active entries that belong to us or are orphaned
      const entries = await prisma.matchmakingQueue.findMany({
        where: {
          status: 'searching',
          OR: [
            { serverId: this.serverId },
            { serverId: null },
            { lastHeartbeat: { lt: staleThreshold } },
          ],
        },
        include: {
          user: { select: { id: true, rating: true } },
        },
      });

      for (const entry of entries) {
        // Claim this entry for our server
        await prisma.matchmakingQueue.update({
          where: { id: entry.id },
          data: {
            serverId: this.serverId,
            lastHeartbeat: new Date(),
          },
        });

        // Check if user is still connected
        const connectedUsers = this.wsServer.getConnectedUsers();
        if (!connectedUsers.includes(entry.userId)) {
          // User disconnected while we were down - cancel their entry
          await this.cancelQueueEntry(entry.id, 'cancelled_disconnect');
          continue;
        }

        // Add to local queue
        const socketId = this.wsServer.getSocketIdForUser(entry.userId);
        if (socketId) {
          this.localQueue.set(entry.userId, {
            id: entry.id,
            ticketId: entry.ticketId,
            userId: entry.userId,
            socketId,
            preferences: {
              boardType: entry.boardType,
              ratingRange: {
                min: entry.ratingRangeMin,
                max: entry.ratingRangeMax,
              },
              timeControl: {
                min: entry.timeControlMin,
                max: entry.timeControlMax,
              },
            },
            rating: entry.rating,
            joinedAt: entry.joinedAt,
          });
        }
      }

      logger.info('Loaded queue from database', {
        loaded: this.localQueue.size,
        total: entries.length,
      });
    } catch (err) {
      logger.error('Failed to load queue from database', {
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  public async addToQueue(
    userId: string,
    socketId: string,
    preferences: MatchmakingPreferences,
    rating: number
  ): Promise<string> {
    // Remove existing entry if present
    await this.removeFromQueue(userId);

    const ticketId = uuidv4();
    const prisma = getDatabaseClient();

    try {
      // Persist to database
      const queueSize = this.localQueue.size;
      const dbEntry = await prisma?.matchmakingQueue.create({
        data: {
          ticketId,
          userId,
          boardType: preferences.boardType as PrismaBoardType,
          ratingRangeMin: preferences.ratingRange.min,
          ratingRangeMax: preferences.ratingRange.max,
          timeControlMin: preferences.timeControl.min,
          timeControlMax: preferences.timeControl.max,
          rating,
          serverId: this.serverId,
          status: 'searching',
        },
      });

      // Record metrics
      await prisma?.matchmakingMetrics.create({
        data: {
          ticketId,
          userId,
          boardType: preferences.boardType as PrismaBoardType,
          rating,
          joinedAt: new Date(),
          outcome: 'matched', // Will be updated when match completes
          queueSizeAtJoin: queueSize,
        },
      });

      const entry: QueueEntry = {
        id: dbEntry?.id || ticketId,
        ticketId,
        userId,
        socketId,
        preferences,
        rating,
        joinedAt: new Date(),
      };

      this.localQueue.set(userId, entry);
      this.emitStatus(entry);

      logger.info('User added to matchmaking queue', {
        userId,
        rating,
        preferences,
        queueSize: this.localQueue.size,
      });

      // Try to find a match immediately
      this.findMatch(entry);

      return ticketId;
    } catch (err) {
      logger.error('Failed to add user to queue', {
        userId,
        error: err instanceof Error ? err.message : String(err),
      });
      throw err;
    }
  }

  public async removeFromQueue(userId: string): Promise<void> {
    const entry = this.localQueue.get(userId);
    if (!entry) return;

    this.localQueue.delete(userId);

    try {
      const prisma = getDatabaseClient();
      if (prisma) {
        await this.cancelQueueEntry(entry.id, 'cancelled_user');
      }
    } catch (err) {
      logger.warn('Failed to remove queue entry from DB', {
        userId,
        error: err instanceof Error ? err.message : String(err),
      });
    }

    logger.info('User removed from matchmaking queue', { userId });
  }

  private async cancelQueueEntry(
    entryId: string,
    outcome: 'cancelled_user' | 'cancelled_disconnect' | 'expired'
  ) {
    const prisma = getDatabaseClient();
    if (!prisma) return;

    const now = new Date();

    // Update queue entry
    const entry = await prisma.matchmakingQueue.update({
      where: { id: entryId },
      data: {
        status: outcome === 'expired' ? 'expired' : 'cancelled',
        cancelledAt: now,
      },
    });

    // Update metrics
    await prisma.matchmakingMetrics.updateMany({
      where: { ticketId: entry.ticketId },
      data: {
        outcome: outcome as MatchmakingOutcome,
        waitTimeMs: now.getTime() - entry.joinedAt.getTime(),
      },
    });
  }

  private startMatchmakingLoop() {
    if (this.matchCheckInterval) return;

    this.matchCheckInterval = setInterval(() => {
      this.processQueue();
    }, this.MATCH_CHECK_INTERVAL_MS);
  }

  private startHeartbeatLoop() {
    if (this.heartbeatInterval) return;

    this.heartbeatInterval = setInterval(async () => {
      await this.updateHeartbeat();
      await this.cleanupStaleEntries();
    }, this.HEARTBEAT_INTERVAL_MS);
  }

  private async updateHeartbeat() {
    const prisma = getDatabaseClient();
    if (!prisma) return;

    try {
      const userIds = Array.from(this.localQueue.keys());
      if (userIds.length === 0) return;

      await prisma.matchmakingQueue.updateMany({
        where: {
          userId: { in: userIds },
          status: 'searching',
        },
        data: {
          lastHeartbeat: new Date(),
        },
      });
    } catch (err) {
      logger.warn('Failed to update heartbeat', {
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  private async processQueue() {
    await this.cleanupStaleEntries();

    const entries = Array.from(this.localQueue.values()).sort(
      (a, b) => a.joinedAt.getTime() - b.joinedAt.getTime()
    );

    const matchedUserIds = new Set<string>();

    for (const entry of entries) {
      if (matchedUserIds.has(entry.userId)) continue;
      if (entry.matchCreationInProgress) continue;

      const match = await this.findMatch(entry);
      if (match) {
        matchedUserIds.add(entry.userId);
        matchedUserIds.add(match.userId);
      } else {
        this.emitStatus(entry);
      }
    }
  }

  private async cleanupStaleEntries(): Promise<void> {
    const now = Date.now();
    const MAX_QUEUE_TIME_MS = this.MAX_WAIT_TIME_MS * 2;
    const MAX_MATCH_CREATION_MS = 30_000;

    const connectedUsers = new Set(this.wsServer.getConnectedUsers());
    const staleEntries: QueueEntry[] = [];

    for (const entry of this.localQueue.values()) {
      const waitTime = now - entry.joinedAt.getTime();

      if (!connectedUsers.has(entry.userId)) {
        staleEntries.push(entry);
        continue;
      }

      if (waitTime > MAX_QUEUE_TIME_MS) {
        staleEntries.push(entry);
        continue;
      }

      if (entry.matchCreationInProgress && waitTime > MAX_MATCH_CREATION_MS) {
        entry.matchCreationInProgress = false;
        logger.warn('Reset stuck matchCreationInProgress flag', {
          userId: entry.userId,
          ticketId: entry.ticketId,
          waitTime,
        });
      }
    }

    for (const entry of staleEntries) {
      this.localQueue.delete(entry.userId);
      const reason = !connectedUsers.has(entry.userId) ? 'cancelled_disconnect' : 'expired';
      await this.cancelQueueEntry(entry.id, reason as 'cancelled_disconnect' | 'expired');
      logger.info('Removed stale matchmaking entry', {
        userId: entry.userId,
        ticketId: entry.ticketId,
        reason,
      });
    }
  }

  private async findMatch(player: QueueEntry): Promise<QueueEntry | null> {
    const now = Date.now();
    const waitTime = now - player.joinedAt.getTime();

    const cappedWait = Math.min(
      waitTime,
      this.MATCH_CHECK_INTERVAL_MS * Math.ceil(this.MAX_WAIT_TIME_MS / this.MATCH_CHECK_INTERVAL_MS)
    );
    const expansionFactor = Math.floor(cappedWait / this.MATCH_CHECK_INTERVAL_MS);
    const ratingBuffer = this.RATING_EXPANSION_RATE * expansionFactor;

    const minRating = player.preferences.ratingRange.min - ratingBuffer;
    const maxRating = player.preferences.ratingRange.max + ratingBuffer;

    let opponent: QueueEntry | null = null;

    for (const other of this.localQueue.values()) {
      if (other.userId === player.userId) continue;
      if (other.matchCreationInProgress) continue;
      if (other.preferences.boardType !== player.preferences.boardType) continue;

      // Check time control range overlap (both players' ranges must intersect)
      const tcOverlap =
        player.preferences.timeControl.min <= other.preferences.timeControl.max &&
        other.preferences.timeControl.min <= player.preferences.timeControl.max;
      if (!tcOverlap) continue;

      const otherWaitTime = now - other.joinedAt.getTime();
      const otherExpansion = Math.floor(otherWaitTime / this.MATCH_CHECK_INTERVAL_MS);
      const otherBuffer = this.RATING_EXPANSION_RATE * otherExpansion;

      const otherMin = other.preferences.ratingRange.min - otherBuffer;
      const otherMax = other.preferences.ratingRange.max + otherBuffer;

      const playerFitsOther = player.rating >= otherMin && player.rating <= otherMax;
      const otherFitsPlayer = other.rating >= minRating && other.rating <= maxRating;

      if (playerFitsOther && otherFitsPlayer) {
        opponent = other;
        break;
      }
    }

    if (opponent) {
      await this.createMatch(player, opponent);
      return opponent;
    }

    return null;
  }

  private async createMatch(player1: QueueEntry, player2: QueueEntry) {
    player1.matchCreationInProgress = true;
    player2.matchCreationInProgress = true;

    const prisma = getDatabaseClient();
    if (!prisma) {
      player1.matchCreationInProgress = false;
      player2.matchCreationInProgress = false;
      return;
    }

    try {
      // Update queue entries to 'matching' status
      await prisma.matchmakingQueue.updateMany({
        where: {
          id: { in: [player1.id, player2.id] },
        },
        data: {
          status: 'matching',
        },
      });

      // Generate per-game RNG seed and invite code
      const rngSeed = generateGameSeed();
      const inviteCode = crypto.randomBytes(6).toString('base64url').slice(0, 8);

      // Create game in DB
      const game = await prisma.game.create({
        data: {
          boardType: player1.preferences.boardType as PrismaBoardType,
          maxPlayers: 2,
          timeControl: {
            type: 'rapid',
            initialTime: player1.preferences.timeControl.min,
            increment: 0,
          },
          isRated: true,
          allowSpectators: true,
          player1Id: player1.userId,
          player2Id: player2.userId,
          status: GameStatus.active,
          startedAt: new Date(),
          gameState: {},
          rngSeed,
          inviteCode,
        },
        include: {
          player1: { select: { id: true, username: true, rating: true } },
          player2: { select: { id: true, username: true, rating: true } },
        },
      });

      const now = new Date();
      const ratingDiff = Math.abs(player1.rating - player2.rating);
      const maxRatingDiff = this.RATING_EXPANSION_RATE * 12; // ~600 after 1 minute
      const matchQualityScore = Math.max(0, 1 - ratingDiff / maxRatingDiff);

      // Update queue entries to 'matched'
      await prisma.matchmakingQueue.updateMany({
        where: {
          id: { in: [player1.id, player2.id] },
        },
        data: {
          status: 'matched',
          matchedAt: now,
          gameId: game.id,
          matchedWithId: player1.id, // Will be overwritten for player2
        },
      });

      // Update individual matchedWithId
      await prisma.matchmakingQueue.update({
        where: { id: player1.id },
        data: { matchedWithId: player2.id },
      });
      await prisma.matchmakingQueue.update({
        where: { id: player2.id },
        data: { matchedWithId: player1.id },
      });

      // Update metrics for both players
      const queueSize = this.localQueue.size;
      for (const player of [player1, player2]) {
        const waitTimeMs = now.getTime() - player.joinedAt.getTime();
        await prisma.matchmakingMetrics.updateMany({
          where: { ticketId: player.ticketId },
          data: {
            outcome: 'matched',
            matchedAt: now,
            waitTimeMs,
            ratingDiff,
            matchQualityScore,
            queueSizeAtMatch: queueSize,
          },
        });
      }

      // Remove from local queue
      this.localQueue.delete(player1.userId);
      this.localQueue.delete(player2.userId);

      // Notify players
      this.wsServer.sendToUser(player1.userId, 'match-found', { gameId: game.id });
      this.wsServer.sendToUser(player2.userId, 'match-found', { gameId: game.id });

      logger.info('Match created', {
        gameId: game.id,
        player1: player1.userId,
        player2: player2.userId,
        ratingDiff,
        matchQualityScore,
      });
    } catch (err) {
      logger.error('Failed to create match', {
        error: err instanceof Error ? err.message : String(err),
        player1: player1.userId,
        player2: player2.userId,
      });

      player1.matchCreationInProgress = false;
      player2.matchCreationInProgress = false;

      // Revert queue status
      await prisma.matchmakingQueue.updateMany({
        where: {
          id: { in: [player1.id, player2.id] },
        },
        data: {
          status: 'searching',
        },
      });

      const errorPayload: WebSocketErrorPayload = {
        type: 'error',
        code: 'INTERNAL_ERROR',
        message: 'Match creation failed temporarily. You remain in the queue.',
      };
      this.wsServer.sendToUser(player1.userId, 'error', errorPayload);
      this.wsServer.sendToUser(player2.userId, 'error', errorPayload);

      this.emitStatus(player1);
      this.emitStatus(player2);
    }
  }

  private emitStatus(entry: QueueEntry) {
    const now = Date.now();
    const waitTime = now - entry.joinedAt.getTime();
    const entries = Array.from(this.localQueue.values());
    const position = entries.findIndex((e) => e.userId === entry.userId) + 1;

    const baseEstimate = 30000;
    const estimatedWaitTime = Math.max(5000, baseEstimate - waitTime);

    const status: MatchmakingStatus = {
      inQueue: true,
      estimatedWaitTime,
      queuePosition: position,
      searchCriteria: entry.preferences,
    };

    this.wsServer.sendToUser(entry.userId, 'matchmaking-status', status);
  }

  /**
   * Get current queue statistics
   */
  public getQueueStats(): {
    queueSize: number;
    byBoardType: Record<string, number>;
    avgWaitTime: number;
  } {
    const entries = Array.from(this.localQueue.values());
    const now = Date.now();

    const byBoardType: Record<string, number> = {};
    let totalWaitTime = 0;

    for (const entry of entries) {
      byBoardType[entry.preferences.boardType] =
        (byBoardType[entry.preferences.boardType] || 0) + 1;
      totalWaitTime += now - entry.joinedAt.getTime();
    }

    return {
      queueSize: entries.length,
      byBoardType,
      avgWaitTime: entries.length > 0 ? totalWaitTime / entries.length : 0,
    };
  }

  /**
   * Clean up on shutdown
   */
  public async shutdown() {
    if (this.matchCheckInterval) {
      clearInterval(this.matchCheckInterval);
      this.matchCheckInterval = null;
    }
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Don't cancel queue entries on shutdown - they'll be picked up by another instance
    // or when this instance restarts
    logger.info('PersistentMatchmakingService shutdown', {
      serverId: this.serverId,
      queueSize: this.localQueue.size,
    });
  }
}
