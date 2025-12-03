/**
 * Tests for connection state machine (src/shared/stateMachines/connection.ts)
 *
 * Structural coverage for PlayerConnectionState as described in
 * P18.3-1 §2.4 (connection sub-states) and §4.3 (disconnect timeouts):
 * - CONNECTED                      – active WebSocket in room.
 * - DISCONNECTED_PENDING_RECONNECT – reconnect window running.
 * - DISCONNECTED_EXPIRED           – reconnect window elapsed.
 *
 * These tests cover pure state transitions; host-level effects (canceling
 * choices, AI requests, or terminating sessions) are exercised in
 * GameSession/WebSocket tests and documented in P18.3-1 §4.3/§6.
 */

import {
  PlayerConnectionState,
  markConnected,
  markDisconnectedPendingReconnect,
  markDisconnectedExpired,
} from '../../../src/shared/stateMachines/connection';

describe('connection state machine', () => {
  const gameId = 'game-123';
  const userId = 'user-456';
  const playerNumber = 1;

  describe('markConnected', () => {
    it('creates connected state for new connection', () => {
      const now = 1000;
      const result = markConnected(gameId, userId, playerNumber, undefined, now);

      expect(result).toEqual({
        kind: 'connected',
        gameId,
        userId,
        playerNumber,
        connectedAt: now,
        lastSeenAt: now,
      });
    });

    it('includes playerNumber if provided', () => {
      const result = markConnected(gameId, userId, 2, undefined, 1000);
      expect(result.playerNumber).toBe(2);
    });

    it('omits playerNumber for spectators (undefined)', () => {
      const result = markConnected(gameId, userId, undefined, undefined, 1000);
      expect(result).not.toHaveProperty('playerNumber');
    });

    it('preserves connectedAt and updates lastSeenAt for same connection', () => {
      const previous: PlayerConnectionState = {
        kind: 'connected',
        gameId,
        userId,
        playerNumber,
        connectedAt: 1000,
        lastSeenAt: 1500,
      };

      const result = markConnected(gameId, userId, playerNumber, previous, 2000);

      expect(result).toEqual({
        kind: 'connected',
        gameId,
        userId,
        playerNumber,
        connectedAt: 1000, // Preserved from previous
        lastSeenAt: 2000, // Updated to now
      });
    });

    it('resets connectedAt when gameId differs', () => {
      const previous: PlayerConnectionState = {
        kind: 'connected',
        gameId: 'different-game',
        userId,
        playerNumber,
        connectedAt: 1000,
        lastSeenAt: 1500,
      };

      const result = markConnected(gameId, userId, playerNumber, previous, 2000);
      expect(result.kind).toBe('connected');
      if (result.kind === 'connected') {
        expect(result.connectedAt).toBe(2000); // Not preserved
        expect(result.lastSeenAt).toBe(2000);
      }
    });

    it('resets connectedAt when userId differs', () => {
      const previous: PlayerConnectionState = {
        kind: 'connected',
        gameId,
        userId: 'different-user',
        playerNumber,
        connectedAt: 1000,
        lastSeenAt: 1500,
      };

      const result = markConnected(gameId, userId, playerNumber, previous, 2000);
      expect(result.kind).toBe('connected');
      if (result.kind === 'connected') {
        expect(result.connectedAt).toBe(2000); // Not preserved
      }
    });

    it('resets connectedAt when previous state is not connected', () => {
      const previous: PlayerConnectionState = {
        kind: 'disconnected_pending_reconnect',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: 1000,
        deadlineAt: 2000,
      };

      const result = markConnected(gameId, userId, playerNumber, previous, 2500);
      expect(result.kind).toBe('connected');
      if (result.kind === 'connected') {
        expect(result.connectedAt).toBe(2500); // Not preserved from non-connected state
      }
    });

    it('uses Date.now when no timestamp provided', () => {
      const before = Date.now();
      const result = markConnected(gameId, userId, playerNumber, undefined);
      const after = Date.now();
      expect(result.kind).toBe('connected');
      if (result.kind === 'connected') {
        expect(result.connectedAt).toBeGreaterThanOrEqual(before);
        expect(result.connectedAt).toBeLessThanOrEqual(after);
      }
    });
  });

  describe('markDisconnectedPendingReconnect', () => {
    const timeoutMs = 30000;

    it('creates pending reconnect state from connected', () => {
      const previous: PlayerConnectionState = {
        kind: 'connected',
        gameId,
        userId,
        playerNumber,
        connectedAt: 1000,
        lastSeenAt: 2000,
      };

      const now = 3000;
      const result = markDisconnectedPendingReconnect(
        previous,
        gameId,
        userId,
        playerNumber,
        timeoutMs,
        now
      );

      expect(result).toEqual({
        kind: 'disconnected_pending_reconnect',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: now, // Fresh disconnect time
        deadlineAt: now + timeoutMs,
      });
    });

    it('preserves disconnectedAt when already pending reconnect for same game/user', () => {
      const previous: PlayerConnectionState = {
        kind: 'disconnected_pending_reconnect',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: 2000,
        deadlineAt: 32000,
      };

      const now = 3000;
      const result = markDisconnectedPendingReconnect(
        previous,
        gameId,
        userId,
        playerNumber,
        timeoutMs,
        now
      );
      expect(result.kind).toBe('disconnected_pending_reconnect');
      if (result.kind === 'disconnected_pending_reconnect') {
        expect(result.disconnectedAt).toBe(2000); // Preserved
        expect(result.deadlineAt).toBe(now + timeoutMs); // Reset with new timeout
      }
    });

    it('resets disconnectedAt for different gameId', () => {
      const previous: PlayerConnectionState = {
        kind: 'disconnected_pending_reconnect',
        gameId: 'other-game',
        userId,
        playerNumber,
        disconnectedAt: 2000,
        deadlineAt: 32000,
      };

      const now = 3000;
      const result = markDisconnectedPendingReconnect(
        previous,
        gameId,
        userId,
        playerNumber,
        timeoutMs,
        now
      );
      expect(result.kind).toBe('disconnected_pending_reconnect');
      if (result.kind === 'disconnected_pending_reconnect') {
        expect(result.disconnectedAt).toBe(now); // Reset for new game
      }
    });

    it('resets disconnectedAt for different userId', () => {
      const previous: PlayerConnectionState = {
        kind: 'connected',
        gameId,
        userId: 'other-user',
        playerNumber,
        connectedAt: 1000,
        lastSeenAt: 2000,
      };

      const now = 3000;
      const result = markDisconnectedPendingReconnect(
        previous,
        gameId,
        userId,
        playerNumber,
        timeoutMs,
        now
      );
      expect(result.kind).toBe('disconnected_pending_reconnect');
      if (result.kind === 'disconnected_pending_reconnect') {
        expect(result.disconnectedAt).toBe(now);
      }
    });

    it('handles undefined previous state', () => {
      const now = 1000;
      const result = markDisconnectedPendingReconnect(
        undefined,
        gameId,
        userId,
        playerNumber,
        timeoutMs,
        now
      );

      expect(result).toEqual({
        kind: 'disconnected_pending_reconnect',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: now,
        deadlineAt: now + timeoutMs,
      });
    });

    it('handles expired previous state', () => {
      const previous: PlayerConnectionState = {
        kind: 'disconnected_expired',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: 1000,
        expiredAt: 2000,
      };

      const now = 3000;
      const result = markDisconnectedPendingReconnect(
        previous,
        gameId,
        userId,
        playerNumber,
        timeoutMs,
        now
      );
      expect(result.kind).toBe('disconnected_pending_reconnect');
      if (result.kind === 'disconnected_pending_reconnect') {
        expect(result.disconnectedAt).toBe(now); // Reset from expired state
      }
    });

    it('uses Date.now when no timestamp provided', () => {
      const before = Date.now();
      const result = markDisconnectedPendingReconnect(
        undefined,
        gameId,
        userId,
        playerNumber,
        timeoutMs
      );
      const after = Date.now();
      expect(result.kind).toBe('disconnected_pending_reconnect');
      if (result.kind === 'disconnected_pending_reconnect') {
        expect(result.disconnectedAt).toBeGreaterThanOrEqual(before);
        expect(result.disconnectedAt).toBeLessThanOrEqual(after);
      }
    });
  });

  describe('markDisconnectedExpired', () => {
    it('creates expired state from pending reconnect', () => {
      const previous: PlayerConnectionState = {
        kind: 'disconnected_pending_reconnect',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: 1000,
        deadlineAt: 31000,
      };

      const now = 35000;
      const result = markDisconnectedExpired(previous, gameId, userId, playerNumber, now);

      expect(result).toEqual({
        kind: 'disconnected_expired',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: 1000, // Preserved from pending
        expiredAt: now,
      });
    });

    it('preserves disconnectedAt from matching pending state', () => {
      const previous: PlayerConnectionState = {
        kind: 'disconnected_pending_reconnect',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: 5000,
        deadlineAt: 35000,
      };

      const result = markDisconnectedExpired(previous, gameId, userId, playerNumber, 40000);
      expect(result.kind).toBe('disconnected_expired');
      if (result.kind === 'disconnected_expired') {
        expect(result.disconnectedAt).toBe(5000);
      }
    });

    it('falls back to now when previous is not pending reconnect', () => {
      const previous: PlayerConnectionState = {
        kind: 'connected',
        gameId,
        userId,
        playerNumber,
        connectedAt: 1000,
        lastSeenAt: 2000,
      };

      const now = 3000;
      const result = markDisconnectedExpired(previous, gameId, userId, playerNumber, now);
      expect(result.kind).toBe('disconnected_expired');
      if (result.kind === 'disconnected_expired') {
        expect(result.disconnectedAt).toBe(now); // Fallback
      }
    });

    it('falls back to now when gameId differs', () => {
      const previous: PlayerConnectionState = {
        kind: 'disconnected_pending_reconnect',
        gameId: 'other-game',
        userId,
        playerNumber,
        disconnectedAt: 1000,
        deadlineAt: 31000,
      };

      const now = 35000;
      const result = markDisconnectedExpired(previous, gameId, userId, playerNumber, now);
      expect(result.kind).toBe('disconnected_expired');
      if (result.kind === 'disconnected_expired') {
        expect(result.disconnectedAt).toBe(now);
      }
    });

    it('falls back to now when userId differs', () => {
      const previous: PlayerConnectionState = {
        kind: 'disconnected_pending_reconnect',
        gameId,
        userId: 'other-user',
        playerNumber,
        disconnectedAt: 1000,
        deadlineAt: 31000,
      };

      const now = 35000;
      const result = markDisconnectedExpired(previous, gameId, userId, playerNumber, now);
      expect(result.kind).toBe('disconnected_expired');
      if (result.kind === 'disconnected_expired') {
        expect(result.disconnectedAt).toBe(now);
      }
    });

    it('handles undefined previous state', () => {
      const now = 5000;
      const result = markDisconnectedExpired(undefined, gameId, userId, playerNumber, now);

      expect(result).toEqual({
        kind: 'disconnected_expired',
        gameId,
        userId,
        playerNumber,
        disconnectedAt: now, // Falls back to now
        expiredAt: now,
      });
    });

    it('uses Date.now when no timestamp provided', () => {
      const before = Date.now();
      const result = markDisconnectedExpired(undefined, gameId, userId, playerNumber);
      const after = Date.now();
      expect(result.kind).toBe('disconnected_expired');
      if (result.kind === 'disconnected_expired') {
        expect(result.expiredAt).toBeGreaterThanOrEqual(before);
        expect(result.expiredAt).toBeLessThanOrEqual(after);
      }
    });
  });

  describe('PlayerConnectionState type discrimination', () => {
    it('connected state has correct shape', () => {
      const state = markConnected(gameId, userId, playerNumber, undefined, 1000);
      expect(state.kind).toBe('connected');
      if (state.kind === 'connected') {
        expect(state.connectedAt).toBeDefined();
        expect(state.lastSeenAt).toBeDefined();
      }
    });

    it('pending reconnect state has correct shape', () => {
      const state = markDisconnectedPendingReconnect(
        undefined,
        gameId,
        userId,
        playerNumber,
        30000,
        1000
      );
      expect(state.kind).toBe('disconnected_pending_reconnect');
      if (state.kind === 'disconnected_pending_reconnect') {
        expect(state.disconnectedAt).toBeDefined();
        expect(state.deadlineAt).toBeDefined();
      }
    });

    it('expired state has correct shape', () => {
      const state = markDisconnectedExpired(undefined, gameId, userId, playerNumber, 1000);
      expect(state.kind).toBe('disconnected_expired');
      if (state.kind === 'disconnected_expired') {
        expect(state.disconnectedAt).toBeDefined();
        expect(state.expiredAt).toBeDefined();
      }
    });
  });

  describe('realistic connection lifecycle', () => {
    it('supports full lifecycle: connect → disconnect → expire', () => {
      // User connects
      let state = markConnected(gameId, userId, playerNumber, undefined, 1000);
      expect(state.kind).toBe('connected');

      // User disconnects unexpectedly
      state = markDisconnectedPendingReconnect(state, gameId, userId, playerNumber, 30000, 2000);
      expect(state.kind).toBe('disconnected_pending_reconnect');
      if (state.kind === 'disconnected_pending_reconnect') {
        expect(state.deadlineAt).toBe(32000);
      }

      // Timeout expires without reconnect
      state = markDisconnectedExpired(state, gameId, userId, playerNumber, 35000);
      expect(state.kind).toBe('disconnected_expired');
      if (state.kind === 'disconnected_expired') {
        expect(state.disconnectedAt).toBe(2000);
        expect(state.expiredAt).toBe(35000);
      }
    });

    it('supports reconnection before expiry', () => {
      // User connects
      let state = markConnected(gameId, userId, playerNumber, undefined, 1000);
      expect(state.kind).toBe('connected');

      // User disconnects
      state = markDisconnectedPendingReconnect(state, gameId, userId, playerNumber, 30000, 2000);
      expect(state.kind).toBe('disconnected_pending_reconnect');

      // User reconnects before deadline
      state = markConnected(gameId, userId, playerNumber, state, 15000);
      expect(state.kind).toBe('connected');
      // Note: connectedAt resets because previous was not 'connected'
      if (state.kind === 'connected') {
        expect(state.connectedAt).toBe(15000);
        expect(state.lastSeenAt).toBe(15000);
      }
    });

    it('supports multiple disconnects with timeout extensions', () => {
      // First disconnect
      let state = markDisconnectedPendingReconnect(
        undefined,
        gameId,
        userId,
        playerNumber,
        30000,
        1000
      );
      expect(state.kind).toBe('disconnected_pending_reconnect');
      if (state.kind === 'disconnected_pending_reconnect') {
        expect(state.disconnectedAt).toBe(1000);
        expect(state.deadlineAt).toBe(31000);
      }

      // Second call extends deadline but preserves disconnectedAt
      state = markDisconnectedPendingReconnect(state, gameId, userId, playerNumber, 30000, 10000);
      if (state.kind === 'disconnected_pending_reconnect') {
        expect(state.disconnectedAt).toBe(1000); // Preserved
        expect(state.deadlineAt).toBe(40000); // Extended
      }
    });

    it('spectator can connect without playerNumber', () => {
      const spectatorState = markConnected(gameId, userId, undefined, undefined, 1000);
      expect(spectatorState.kind).toBe('connected');
      expect(spectatorState).not.toHaveProperty('playerNumber');
    });
  });
});
