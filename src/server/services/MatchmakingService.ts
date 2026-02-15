import {
  MatchmakingPreferences,
  MatchmakingStatus,
  WebSocketErrorPayload,
} from '../../shared/types/websocket';
import { BoardType as PrismaBoardType, GameStatus } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import { WebSocketServer } from '../websocket/server';
import { logger } from '../utils/logger';
import { v4 as uuidv4 } from 'uuid';
import crypto from 'crypto';
import { generateGameSeed } from '../../shared/utils/rng';

interface QueueEntry {
  userId: string;
  socketId: string;
  preferences: MatchmakingPreferences;
  rating: number;
  joinedAt: Date;
  ticketId: string;
  // P0 FIX: Flag to indicate this entry is currently in match creation.
  // Entries with this flag should not be matched with other players.
  matchCreationInProgress?: boolean;
}

export class MatchmakingService {
  private queue: QueueEntry[] = [];
  private matchCheckInterval: NodeJS.Timeout | null = null;
  private readonly MATCH_CHECK_INTERVAL_MS = 5000;
  private readonly RATING_EXPANSION_RATE = 50; // Rating range expands by this amount per interval
  private readonly MAX_WAIT_TIME_MS = 60000; // 1 minute max wait before expanding to any rating

  constructor(private wsServer: WebSocketServer) {
    this.startMatchmakingLoop();
  }

  public addToQueue(
    userId: string,
    socketId: string,
    preferences: MatchmakingPreferences,
    rating: number
  ): string {
    // Remove existing entry if present
    this.removeFromQueue(userId);

    const ticketId = uuidv4();
    const entry: QueueEntry = {
      userId,
      socketId,
      preferences,
      rating,
      joinedAt: new Date(),
      ticketId,
    };

    this.queue.push(entry);
    this.emitStatus(entry);

    logger.info('User added to matchmaking queue', { userId, rating, preferences });

    // Try to find a match immediately
    this.findMatch(entry);

    return ticketId;
  }

  public removeFromQueue(userId: string): void {
    const index = this.queue.findIndex((e) => e.userId === userId);
    if (index !== -1) {
      this.queue.splice(index, 1);
      logger.info('User removed from matchmaking queue', { userId });
    }
  }

  private startMatchmakingLoop() {
    if (this.matchCheckInterval) return;

    this.matchCheckInterval = setInterval(() => {
      this.processQueue();
    }, this.MATCH_CHECK_INTERVAL_MS);
  }

  private processQueue() {
    // P2 FIX: Clean up stale entries before processing
    this.cleanupStaleEntries();

    // Sort queue by join time (FCFS)
    this.queue.sort((a, b) => a.joinedAt.getTime() - b.joinedAt.getTime());

    // Try to match each player
    // Note: We iterate backwards or use a while loop to handle removals safely
    // but for simplicity here we just iterate and skip if already matched
    const matchedUserIds = new Set<string>();

    for (const entry of this.queue) {
      if (matchedUserIds.has(entry.userId)) continue;
      // P0 FIX: Skip entries already in match creation process
      if (entry.matchCreationInProgress) continue;

      const match = this.findMatch(entry);
      if (match) {
        matchedUserIds.add(entry.userId);
        matchedUserIds.add(match.userId);
      } else {
        // Update status for unmatched players (e.g. expanded range)
        this.emitStatus(entry);
      }
    }
  }

  /**
   * P2 FIX: Clean up stale queue entries to prevent memory leaks.
   * Removes entries for:
   * - Users whose sockets are no longer connected
   * - Users who have been in queue too long (2x max wait time)
   * - Entries stuck in matchCreationInProgress for too long (30s timeout)
   */
  private cleanupStaleEntries(): void {
    const now = Date.now();
    const MAX_QUEUE_TIME_MS = this.MAX_WAIT_TIME_MS * 2; // 2 minutes
    const MAX_MATCH_CREATION_MS = 30_000; // 30 seconds for DB operation

    const connectedUsers = this.wsServer.getConnectedUsers();
    const connectedSet = new Set(connectedUsers);

    const staleEntries: QueueEntry[] = [];

    for (const entry of this.queue) {
      const waitTime = now - entry.joinedAt.getTime();

      // Check if user's socket is disconnected
      if (!connectedSet.has(entry.userId)) {
        staleEntries.push(entry);
        continue;
      }

      // Check if user has been waiting too long
      if (waitTime > MAX_QUEUE_TIME_MS) {
        staleEntries.push(entry);
        continue;
      }

      // Check if match creation has been stuck for too long
      if (entry.matchCreationInProgress && waitTime > MAX_MATCH_CREATION_MS) {
        // Reset the flag instead of removing - give them another chance
        entry.matchCreationInProgress = false;
        logger.warn('Reset stuck matchCreationInProgress flag', {
          userId: entry.userId,
          ticketId: entry.ticketId,
          waitTime,
        });
      }
    }

    // Remove stale entries
    for (const entry of staleEntries) {
      this.removeFromQueue(entry.userId);
      logger.info('Removed stale matchmaking entry', {
        userId: entry.userId,
        ticketId: entry.ticketId,
        reason: !connectedSet.has(entry.userId) ? 'disconnected' : 'timeout',
      });
    }
  }

  private findMatch(player: QueueEntry): QueueEntry | null {
    const now = Date.now();
    const waitTime = now - player.joinedAt.getTime();

    // Calculate expanded rating range based on wait time, capped by a
    // maximum window so expansion does not grow without bound.
    const cappedWait = Math.min(
      waitTime,
      this.MATCH_CHECK_INTERVAL_MS * Math.ceil(this.MAX_WAIT_TIME_MS / this.MATCH_CHECK_INTERVAL_MS)
    );
    const expansionFactor = Math.floor(cappedWait / this.MATCH_CHECK_INTERVAL_MS);
    const ratingBuffer = this.RATING_EXPANSION_RATE * expansionFactor;

    const minRating = player.preferences.ratingRange.min - ratingBuffer;
    const maxRating = player.preferences.ratingRange.max + ratingBuffer;

    // Find a compatible opponent
    const opponent = this.queue.find((other) => {
      if (other.userId === player.userId) return false;

      // P0 FIX: Skip players already in match creation to prevent double-matching
      if (other.matchCreationInProgress) return false;

      // Check board type compatibility
      if (other.preferences.boardType !== player.preferences.boardType) return false;

      // Check time control range overlap (both players' ranges must intersect)
      const tcOverlap =
        player.preferences.timeControl.min <= other.preferences.timeControl.max &&
        other.preferences.timeControl.min <= player.preferences.timeControl.max;
      if (!tcOverlap) return false;

      // Check rating compatibility (bidirectional)
      const otherWaitTime = now - other.joinedAt.getTime();
      const otherExpansion = Math.floor(otherWaitTime / this.MATCH_CHECK_INTERVAL_MS);
      const otherBuffer = this.RATING_EXPANSION_RATE * otherExpansion;

      const otherMin = other.preferences.ratingRange.min - otherBuffer;
      const otherMax = other.preferences.ratingRange.max + otherBuffer;

      const playerFitsOther = player.rating >= otherMin && player.rating <= otherMax;
      const otherFitsPlayer = other.rating >= minRating && other.rating <= maxRating;

      return playerFitsOther && otherFitsPlayer;
    });

    if (opponent) {
      this.createMatch(player, opponent);
      return opponent;
    }

    return null;
  }

  private async createMatch(player1: QueueEntry, player2: QueueEntry) {
    // P0 FIX (2026-01-11): Mark players as "in match creation" instead of removing.
    // This prevents the race condition where players disappear from queue during
    // the database call window, and other match attempts can't see them.
    player1.matchCreationInProgress = true;
    player2.matchCreationInProgress = true;

    try {
      const prisma = getDatabaseClient();
      if (!prisma) throw new Error('Database not available');

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

      // Success! Now safe to remove from queue
      this.removeFromQueue(player1.userId);
      this.removeFromQueue(player2.userId);

      // Notify players
      this.wsServer.sendToUser(player1.userId, 'match-found', { gameId: game.id });
      this.wsServer.sendToUser(player2.userId, 'match-found', { gameId: game.id });

      logger.info('Match created', {
        gameId: game.id,
        player1: player1.userId,
        player2: player2.userId,
      });
    } catch (err) {
      logger.error('Failed to create match', {
        error: err instanceof Error ? err.message : String(err),
        player1: player1.userId,
        player2: player2.userId,
      });

      // Clear the in-progress flag so they can be matched again
      player1.matchCreationInProgress = false;
      player2.matchCreationInProgress = false;

      // Notify players of temporary failure (they remain in queue)
      const errorPayload: WebSocketErrorPayload = {
        type: 'error',
        code: 'INTERNAL_ERROR',
        message: 'Match creation failed temporarily. You remain in the queue.',
      };
      this.wsServer.sendToUser(player1.userId, 'error', errorPayload);
      this.wsServer.sendToUser(player2.userId, 'error', errorPayload);

      // Emit updated status so clients know they're still queued
      this.emitStatus(player1);
      this.emitStatus(player2);
    }
  }

  private emitStatus(entry: QueueEntry) {
    const now = Date.now();
    const waitTime = now - entry.joinedAt.getTime();
    const position = this.queue.indexOf(entry) + 1;

    // Simple heuristic: decrease the remaining estimated wait time as the
    // player waits longer, but never drop below a small floor to avoid
    // reporting negative or unrealistically low values.
    const baseEstimate = 30000; // 30s baseline
    const estimatedWaitTime = Math.max(5000, baseEstimate - waitTime);

    const status: MatchmakingStatus = {
      inQueue: true,
      estimatedWaitTime,
      queuePosition: position,
      searchCriteria: entry.preferences,
    };

    this.wsServer.sendToUser(entry.userId, 'matchmaking-status', status);
  }
}
