import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from './GameSession';
import { PythonRulesClient } from '../services/PythonRulesClient';
import { getCacheService } from '../cache/redis';
import { logger } from '../utils/logger';
import { getMetricsService } from '../services/MetricsService';
import type { ClientToServerEvents, ServerToClientEvents } from '../../shared/types/websocket';

export class GameSessionManager {
  private sessions: Map<string, GameSession> = new Map();
  private pythonRulesClient: PythonRulesClient;

  constructor(
    private io: SocketIOServer<ClientToServerEvents, ServerToClientEvents>,
    private userSockets: Map<string, string>
  ) {
    this.pythonRulesClient = new PythonRulesClient();
  }

  public async getOrCreateSession(gameId: string): Promise<GameSession> {
    const existing = this.sessions.get(gameId);
    if (existing) {
      return existing;
    }

    // Create a bound lock function for this specific game
    const withLockForGame = <T>(operation: () => Promise<T>): Promise<T> => {
      return this.withGameLock(gameId, operation);
    };

    const session = new GameSession(
      gameId,
      this.io,
      this.pythonRulesClient,
      this.userSockets,
      withLockForGame
    );

    await session.initialize();
    this.sessions.set(gameId, session);
    return session;
  }

  public getSession(gameId: string): GameSession | undefined {
    return this.sessions.get(gameId);
  }

  public removeSession(gameId: string): void {
    const session = this.sessions.get(gameId);

    // Keep the ringrift_game_session_status_current gauge in sync when a
    // session is explicitly torn down. We decrement the gauge for the
    // session's last known derived status kind, if available.
    if (session) {
      try {
        const snapshot = session.getSessionStatusSnapshot();
        const currentKind = snapshot?.kind ?? null;
        if (currentKind) {
          getMetricsService().updateGameSessionStatusCurrent(currentKind, null);
        }
      } catch (err) {
        logger.warn('Failed to update session status gauge on removeSession', {
          gameId,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    this.sessions.delete(gameId);
  }

  /**
   * Execute an operation with a distributed lock on the gameId.
   * This prevents race conditions where multiple requests (e.g. concurrent moves)
   * attempt to modify the game state simultaneously.
   *
   * P0 FIX (2026-01-11): Increased TTL from 5s to 15s to handle complex operations
   * that involve Python parity checks, database persistence, and broadcasting.
   * Operations typically complete in 100-500ms, but network issues or database
   * latency can occasionally extend this. The 15s TTL provides safety margin
   * while still preventing deadlocks from crashed processes.
   */
  public async withGameLock<T>(gameId: string, operation: () => Promise<T>): Promise<T> {
    const cacheService = getCacheService();

    // If Redis is not available, fall back to executing without a lock.
    // This degrades gracefully but reintroduces race condition risks.
    if (!cacheService) {
      logger.warn('Redis not available for locking, proceeding without lock', { gameId });
      return operation();
    }

    const lockKey = `lock:game:${gameId}`;
    // P0 FIX: Increased from 5s to 15s to prevent lock expiration during
    // complex operations (Python rules calls, DB persistence, broadcasts).
    const ttlSeconds = 15;
    const maxRetries = 8; // More retries with longer wait between
    const retryDelayMs = 250;

    for (let i = 0; i < maxRetries; i++) {
      const acquired = await cacheService.acquireLock(lockKey, ttlSeconds);
      if (acquired) {
        const startTime = Date.now();
        try {
          const result = await operation();
          const elapsed = Date.now() - startTime;
          // Warn if operation took more than 50% of TTL
          if (elapsed > ttlSeconds * 500) {
            logger.warn('Game lock operation took significant time', {
              gameId,
              elapsedMs: elapsed,
              ttlSeconds,
            });
          }
          return result;
        } finally {
          await cacheService.releaseLock(lockKey);
        }
      }

      // Wait before retrying with exponential backoff
      const backoffDelay = retryDelayMs * Math.min(2 ** i, 4);
      await new Promise((resolve) => setTimeout(resolve, backoffDelay));
    }

    throw new Error('Game is busy, please try again');
  }
}
