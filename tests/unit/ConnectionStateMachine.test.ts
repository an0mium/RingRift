/**
 * Tests for ConnectionStateMachine state machine
 * @module tests/unit/ConnectionStateMachine.test
 */

import {
  // Types
  type ConnectionState,
  type ConnectionIdleState,
  type ConnectionConnectingState,
  type ConnectionConnectedState,
  type ConnectionReconnectingState,
  type ConnectionDisconnectedState,
  type ConnectionErrorState,
  type DisconnectReason,
  // State constructors
  createIdleState,
  createConnectingState,
  createConnectedState,
  createReconnectingState,
  createDisconnectedState,
  createErrorState,
  // State transitions
  startConnecting,
  markConnected,
  updateHeartbeat,
  startReconnecting,
  incrementReconnectAttempt,
  markDisconnected,
  markError,
  resetConnection,
  // Query helpers
  isConnectionActive,
  isConnectionUsable,
  isConnectionFailed,
  getConnectionGameId,
  getTimeSinceHeartbeat,
  isHeartbeatStale,
  toLegacyConnectionStatus,
  // Summary
  getConnectionSummary,
} from '../../src/client/state/ConnectionStateMachine';

describe('ConnectionStateMachine', () => {
  const testGameId = 'test-game-123';
  const testNowMs = 1000000;

  describe('State Constructors', () => {
    describe('createIdleState', () => {
      it('should create an idle state', () => {
        const state = createIdleState();
        expect(state.kind).toBe('idle');
      });
    });

    describe('createConnectingState', () => {
      it('should create a connecting state with game ID and timestamp', () => {
        const state = createConnectingState(testGameId, testNowMs);

        expect(state.kind).toBe('connecting');
        expect(state.gameId).toBe(testGameId);
        expect(state.startedAt).toBe(testNowMs);
      });

      it('should use Date.now() when nowMs is not provided', () => {
        const before = Date.now();
        const state = createConnectingState(testGameId);
        const after = Date.now();

        expect(state.startedAt).toBeGreaterThanOrEqual(before);
        expect(state.startedAt).toBeLessThanOrEqual(after);
      });
    });

    describe('createConnectedState', () => {
      it('should create a connected state with game ID and timestamps', () => {
        const state = createConnectedState(testGameId, testNowMs);

        expect(state.kind).toBe('connected');
        expect(state.gameId).toBe(testGameId);
        expect(state.connectedAt).toBe(testNowMs);
        expect(state.lastHeartbeatAt).toBe(testNowMs);
      });

      it('should use Date.now() when nowMs is not provided', () => {
        const before = Date.now();
        const state = createConnectedState(testGameId);
        const after = Date.now();

        expect(state.connectedAt).toBeGreaterThanOrEqual(before);
        expect(state.connectedAt).toBeLessThanOrEqual(after);
        expect(state.lastHeartbeatAt).toBe(state.connectedAt);
      });
    });

    describe('createReconnectingState', () => {
      it('should create a reconnecting state with all fields', () => {
        const lastHeartbeat = testNowMs - 5000;
        const state = createReconnectingState(testGameId, lastHeartbeat, 3, testNowMs);

        expect(state.kind).toBe('reconnecting');
        expect(state.gameId).toBe(testGameId);
        expect(state.reconnectStartedAt).toBe(testNowMs);
        expect(state.attemptCount).toBe(3);
        expect(state.lastHeartbeatAt).toBe(lastHeartbeat);
      });

      it('should default attemptCount to 1', () => {
        const state = createReconnectingState(testGameId, null, undefined, testNowMs);
        expect(state.attemptCount).toBe(1);
      });

      it('should handle null lastHeartbeatAt', () => {
        const state = createReconnectingState(testGameId, null, 1, testNowMs);
        expect(state.lastHeartbeatAt).toBeNull();
      });
    });

    describe('createDisconnectedState', () => {
      it('should create a disconnected state with reason', () => {
        const state = createDisconnectedState('game_ended', testGameId, testNowMs);

        expect(state.kind).toBe('disconnected');
        expect(state.reason).toBe('game_ended');
        expect(state.previousGameId).toBe(testGameId);
        expect(state.disconnectedAt).toBe(testNowMs);
      });

      it('should not include previousGameId when undefined', () => {
        const state = createDisconnectedState('user_initiated', undefined, testNowMs);

        expect(state.kind).toBe('disconnected');
        expect(state.reason).toBe('user_initiated');
        expect('previousGameId' in state).toBe(false);
      });

      it('should handle all disconnect reasons', () => {
        const reasons: DisconnectReason[] = [
          'user_initiated',
          'game_ended',
          'server_closed',
          'network_error',
          'timeout',
        ];

        for (const reason of reasons) {
          const state = createDisconnectedState(reason, undefined, testNowMs);
          expect(state.reason).toBe(reason);
        }
      });
    });

    describe('createErrorState', () => {
      it('should create an error state with message', () => {
        const state = createErrorState('Connection failed', { nowMs: testNowMs });

        expect(state.kind).toBe('error');
        expect(state.message).toBe('Connection failed');
        expect(state.errorAt).toBe(testNowMs);
        expect(state.canRetry).toBe(true);
      });

      it('should include optional fields when provided', () => {
        const state = createErrorState('Auth error', {
          code: 'AUTH_FAILED',
          gameId: testGameId,
          canRetry: false,
          nowMs: testNowMs,
        });

        expect(state.message).toBe('Auth error');
        expect(state.code).toBe('AUTH_FAILED');
        expect(state.gameId).toBe(testGameId);
        expect(state.canRetry).toBe(false);
      });

      it('should not include optional fields when not provided', () => {
        const state = createErrorState('Error', { nowMs: testNowMs });

        expect('code' in state).toBe(false);
        expect('gameId' in state).toBe(false);
      });

      it('should default canRetry to true', () => {
        const state = createErrorState('Error');
        expect(state.canRetry).toBe(true);
      });
    });
  });

  describe('State Transitions', () => {
    describe('startConnecting', () => {
      it('should transition from idle to connecting', () => {
        const idle = createIdleState();
        const connecting = startConnecting(idle, testGameId, testNowMs);

        expect(connecting.kind).toBe('connecting');
        expect(connecting.gameId).toBe(testGameId);
        expect(connecting.startedAt).toBe(testNowMs);
      });

      it('should transition from disconnected to connecting', () => {
        const disconnected = createDisconnectedState('network_error', undefined, testNowMs);
        const connecting = startConnecting(disconnected, testGameId, testNowMs + 1000);

        expect(connecting.kind).toBe('connecting');
        expect(connecting.gameId).toBe(testGameId);
      });

      it('should transition from error to connecting', () => {
        const error = createErrorState('Previous error', { nowMs: testNowMs });
        const connecting = startConnecting(error, testGameId, testNowMs + 1000);

        expect(connecting.kind).toBe('connecting');
        expect(connecting.gameId).toBe(testGameId);
      });

      it('should warn but still work from invalid states', () => {
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation();

        const connected = createConnectedState(testGameId, testNowMs);
        const connecting = startConnecting(connected, 'new-game', testNowMs + 1000);

        expect(warnSpy).toHaveBeenCalledWith(
          expect.stringContaining('startConnecting called from invalid state')
        );
        expect(connecting.kind).toBe('connecting');

        warnSpy.mockRestore();
      });
    });

    describe('markConnected', () => {
      it('should transition from connecting to connected', () => {
        const connecting = createConnectingState(testGameId, testNowMs);
        const connected = markConnected(connecting, testNowMs + 5000);

        expect(connected.kind).toBe('connected');
        expect(connected.gameId).toBe(testGameId);
        expect(connected.connectedAt).toBe(testNowMs + 5000);
        expect(connected.lastHeartbeatAt).toBe(testNowMs + 5000);
      });

      it('should transition from reconnecting to connected', () => {
        const reconnecting = createReconnectingState(testGameId, testNowMs - 10000, 3, testNowMs);
        const connected = markConnected(reconnecting, testNowMs + 2000);

        expect(connected.kind).toBe('connected');
        expect(connected.gameId).toBe(testGameId);
      });
    });

    describe('updateHeartbeat', () => {
      it('should update lastHeartbeatAt timestamp', () => {
        const connected = createConnectedState(testGameId, testNowMs);
        const updated = updateHeartbeat(connected, testNowMs + 5000);

        expect(updated.kind).toBe('connected');
        expect(updated.gameId).toBe(testGameId);
        expect(updated.connectedAt).toBe(testNowMs);
        expect(updated.lastHeartbeatAt).toBe(testNowMs + 5000);
      });

      it('should use Date.now() when nowMs is not provided', () => {
        const connected = createConnectedState(testGameId, testNowMs);
        const before = Date.now();
        const updated = updateHeartbeat(connected);
        const after = Date.now();

        expect(updated.lastHeartbeatAt).toBeGreaterThanOrEqual(before);
        expect(updated.lastHeartbeatAt).toBeLessThanOrEqual(after);
      });
    });

    describe('startReconnecting', () => {
      it('should transition from connected to reconnecting', () => {
        const connected = createConnectedState(testGameId, testNowMs);
        const reconnecting = startReconnecting(connected, testNowMs + 5000);

        expect(reconnecting.kind).toBe('reconnecting');
        expect(reconnecting.gameId).toBe(testGameId);
        expect(reconnecting.reconnectStartedAt).toBe(testNowMs + 5000);
        expect(reconnecting.attemptCount).toBe(1);
        expect(reconnecting.lastHeartbeatAt).toBe(testNowMs);
      });
    });

    describe('incrementReconnectAttempt', () => {
      it('should increment attempt count', () => {
        const reconnecting = createReconnectingState(testGameId, testNowMs - 5000, 2, testNowMs);
        const incremented = incrementReconnectAttempt(reconnecting, testNowMs + 3000);

        expect(incremented.kind).toBe('reconnecting');
        expect(incremented.attemptCount).toBe(3);
        expect(incremented.reconnectStartedAt).toBe(testNowMs + 3000);
        expect(incremented.lastHeartbeatAt).toBe(testNowMs - 5000);
      });
    });

    describe('markDisconnected', () => {
      it('should transition from connected to disconnected', () => {
        const connected = createConnectedState(testGameId, testNowMs);
        const disconnected = markDisconnected(connected, 'user_initiated', testNowMs + 10000);

        expect(disconnected.kind).toBe('disconnected');
        expect(disconnected.reason).toBe('user_initiated');
        expect(disconnected.previousGameId).toBe(testGameId);
        expect(disconnected.disconnectedAt).toBe(testNowMs + 10000);
      });

      it('should transition from connecting to disconnected', () => {
        const connecting = createConnectingState(testGameId, testNowMs);
        const disconnected = markDisconnected(connecting, 'timeout', testNowMs + 30000);

        expect(disconnected.kind).toBe('disconnected');
        expect(disconnected.previousGameId).toBe(testGameId);
      });

      it('should transition from reconnecting to disconnected', () => {
        const reconnecting = createReconnectingState(testGameId, testNowMs - 5000, 3, testNowMs);
        const disconnected = markDisconnected(reconnecting, 'network_error', testNowMs + 5000);

        expect(disconnected.kind).toBe('disconnected');
        expect(disconnected.previousGameId).toBe(testGameId);
      });

      it('should transition from idle with no previousGameId', () => {
        const idle = createIdleState();
        const disconnected = markDisconnected(idle, 'server_closed', testNowMs);

        expect(disconnected.kind).toBe('disconnected');
        expect(disconnected.previousGameId).toBeUndefined();
      });

      it('should transition from error with optional gameId', () => {
        const error = createErrorState('Error', { gameId: testGameId, nowMs: testNowMs });
        const disconnected = markDisconnected(error, 'game_ended', testNowMs + 1000);

        expect(disconnected.kind).toBe('disconnected');
        // Error state gameId is not copied to previousGameId (only active states)
        expect(disconnected.previousGameId).toBeUndefined();
      });
    });

    describe('markError', () => {
      it('should transition from connecting to error', () => {
        const connecting = createConnectingState(testGameId, testNowMs);
        const error = markError(connecting, 'Connection timeout', { nowMs: testNowMs + 30000 });

        expect(error.kind).toBe('error');
        expect(error.message).toBe('Connection timeout');
        expect(error.gameId).toBe(testGameId);
        expect(error.errorAt).toBe(testNowMs + 30000);
        expect(error.canRetry).toBe(true);
      });

      it('should transition from reconnecting to error', () => {
        const reconnecting = createReconnectingState(testGameId, testNowMs - 5000, 5, testNowMs);
        const error = markError(reconnecting, 'Max retries exceeded', {
          code: 'MAX_RETRIES',
          canRetry: false,
          nowMs: testNowMs + 10000,
        });

        expect(error.kind).toBe('error');
        expect(error.message).toBe('Max retries exceeded');
        expect(error.code).toBe('MAX_RETRIES');
        expect(error.gameId).toBe(testGameId);
        expect(error.canRetry).toBe(false);
      });
    });

    describe('resetConnection', () => {
      it('should return idle state', () => {
        const state = resetConnection();
        expect(state.kind).toBe('idle');
      });
    });
  });

  describe('Query Helpers', () => {
    describe('isConnectionActive', () => {
      it('should return true for connecting state', () => {
        const state = createConnectingState(testGameId, testNowMs);
        expect(isConnectionActive(state)).toBe(true);
      });

      it('should return true for connected state', () => {
        const state = createConnectedState(testGameId, testNowMs);
        expect(isConnectionActive(state)).toBe(true);
      });

      it('should return true for reconnecting state', () => {
        const state = createReconnectingState(testGameId, null, 1, testNowMs);
        expect(isConnectionActive(state)).toBe(true);
      });

      it('should return false for idle state', () => {
        const state = createIdleState();
        expect(isConnectionActive(state)).toBe(false);
      });

      it('should return false for disconnected state', () => {
        const state = createDisconnectedState('user_initiated', undefined, testNowMs);
        expect(isConnectionActive(state)).toBe(false);
      });

      it('should return false for error state', () => {
        const state = createErrorState('Error', { nowMs: testNowMs });
        expect(isConnectionActive(state)).toBe(false);
      });
    });

    describe('isConnectionUsable', () => {
      it('should return true only for connected state', () => {
        expect(isConnectionUsable(createConnectedState(testGameId, testNowMs))).toBe(true);
        expect(isConnectionUsable(createIdleState())).toBe(false);
        expect(isConnectionUsable(createConnectingState(testGameId, testNowMs))).toBe(false);
        expect(isConnectionUsable(createReconnectingState(testGameId, null, 1, testNowMs))).toBe(
          false
        );
        expect(
          isConnectionUsable(createDisconnectedState('user_initiated', undefined, testNowMs))
        ).toBe(false);
        expect(isConnectionUsable(createErrorState('Error', { nowMs: testNowMs }))).toBe(false);
      });
    });

    describe('isConnectionFailed', () => {
      it('should return true for error state', () => {
        const state = createErrorState('Error', { nowMs: testNowMs });
        expect(isConnectionFailed(state)).toBe(true);
      });

      it('should return true for disconnected state', () => {
        const state = createDisconnectedState('network_error', undefined, testNowMs);
        expect(isConnectionFailed(state)).toBe(true);
      });

      it('should return false for active states', () => {
        expect(isConnectionFailed(createIdleState())).toBe(false);
        expect(isConnectionFailed(createConnectingState(testGameId, testNowMs))).toBe(false);
        expect(isConnectionFailed(createConnectedState(testGameId, testNowMs))).toBe(false);
        expect(isConnectionFailed(createReconnectingState(testGameId, null, 1, testNowMs))).toBe(
          false
        );
      });
    });

    describe('getConnectionGameId', () => {
      it('should return gameId for connecting state', () => {
        const state = createConnectingState(testGameId, testNowMs);
        expect(getConnectionGameId(state)).toBe(testGameId);
      });

      it('should return gameId for connected state', () => {
        const state = createConnectedState(testGameId, testNowMs);
        expect(getConnectionGameId(state)).toBe(testGameId);
      });

      it('should return gameId for reconnecting state', () => {
        const state = createReconnectingState(testGameId, null, 1, testNowMs);
        expect(getConnectionGameId(state)).toBe(testGameId);
      });

      it('should return gameId for error state if present', () => {
        const withGameId = createErrorState('Error', { gameId: testGameId, nowMs: testNowMs });
        expect(getConnectionGameId(withGameId)).toBe(testGameId);

        const withoutGameId = createErrorState('Error', { nowMs: testNowMs });
        expect(getConnectionGameId(withoutGameId)).toBeNull();
      });

      it('should return previousGameId for disconnected state if present', () => {
        const withPrevious = createDisconnectedState('user_initiated', testGameId, testNowMs);
        expect(getConnectionGameId(withPrevious)).toBe(testGameId);

        const withoutPrevious = createDisconnectedState('user_initiated', undefined, testNowMs);
        expect(getConnectionGameId(withoutPrevious)).toBeNull();
      });

      it('should return null for idle state', () => {
        const state = createIdleState();
        expect(getConnectionGameId(state)).toBeNull();
      });
    });

    describe('getTimeSinceHeartbeat', () => {
      it('should return time since heartbeat for connected state', () => {
        const connected = createConnectedState(testGameId, testNowMs);
        const timeSince = getTimeSinceHeartbeat(connected, testNowMs + 5000);

        expect(timeSince).toBe(5000);
      });

      it('should return time since heartbeat for reconnecting state', () => {
        const reconnecting = createReconnectingState(testGameId, testNowMs - 10000, 2, testNowMs);
        const timeSince = getTimeSinceHeartbeat(reconnecting, testNowMs + 5000);

        expect(timeSince).toBe(15000);
      });

      it('should return null for reconnecting with null lastHeartbeatAt', () => {
        const reconnecting = createReconnectingState(testGameId, null, 1, testNowMs);
        expect(getTimeSinceHeartbeat(reconnecting, testNowMs + 5000)).toBeNull();
      });

      it('should return null for non-heartbeat states', () => {
        expect(getTimeSinceHeartbeat(createIdleState(), testNowMs)).toBeNull();
        expect(
          getTimeSinceHeartbeat(createConnectingState(testGameId, testNowMs), testNowMs)
        ).toBeNull();
        expect(
          getTimeSinceHeartbeat(
            createDisconnectedState('user_initiated', undefined, testNowMs),
            testNowMs
          )
        ).toBeNull();
        expect(
          getTimeSinceHeartbeat(createErrorState('Error', { nowMs: testNowMs }), testNowMs)
        ).toBeNull();
      });
    });

    describe('isHeartbeatStale', () => {
      it('should return true when time since heartbeat exceeds threshold', () => {
        const connected = createConnectedState(testGameId, testNowMs);
        expect(isHeartbeatStale(connected, 5000, testNowMs + 6000)).toBe(true);
      });

      it('should return false when time since heartbeat is within threshold', () => {
        const connected = createConnectedState(testGameId, testNowMs);
        expect(isHeartbeatStale(connected, 5000, testNowMs + 3000)).toBe(false);
      });

      it('should return false when exactly at threshold', () => {
        const connected = createConnectedState(testGameId, testNowMs);
        expect(isHeartbeatStale(connected, 5000, testNowMs + 5000)).toBe(false);
      });

      it('should return false for non-heartbeat states', () => {
        expect(isHeartbeatStale(createIdleState(), 5000, testNowMs)).toBe(false);
        expect(
          isHeartbeatStale(createConnectingState(testGameId, testNowMs), 5000, testNowMs)
        ).toBe(false);
      });
    });

    describe('toLegacyConnectionStatus', () => {
      it('should map connected to connected', () => {
        const state = createConnectedState(testGameId, testNowMs);
        expect(toLegacyConnectionStatus(state)).toBe('connected');
      });

      it('should map connecting to connecting', () => {
        const state = createConnectingState(testGameId, testNowMs);
        expect(toLegacyConnectionStatus(state)).toBe('connecting');
      });

      it('should map reconnecting to reconnecting', () => {
        const state = createReconnectingState(testGameId, null, 1, testNowMs);
        expect(toLegacyConnectionStatus(state)).toBe('reconnecting');
      });

      it('should map idle to disconnected', () => {
        const state = createIdleState();
        expect(toLegacyConnectionStatus(state)).toBe('disconnected');
      });

      it('should map disconnected to disconnected', () => {
        const state = createDisconnectedState('user_initiated', undefined, testNowMs);
        expect(toLegacyConnectionStatus(state)).toBe('disconnected');
      });

      it('should map error to disconnected', () => {
        const state = createErrorState('Error', { nowMs: testNowMs });
        expect(toLegacyConnectionStatus(state)).toBe('disconnected');
      });
    });
  });

  describe('getConnectionSummary', () => {
    it('should return summary for idle state', () => {
      const state = createIdleState();
      const summary = getConnectionSummary(state);

      expect(summary.kind).toBe('idle');
      expect(summary.isActive).toBe(false);
      expect(summary.isUsable).toBe(false);
      expect(summary.gameId).toBeUndefined();
      expect(summary.error).toBeUndefined();
    });

    it('should return summary for connected state', () => {
      const state = createConnectedState(testGameId, testNowMs);
      const summary = getConnectionSummary(state, testNowMs + 3000);

      expect(summary.kind).toBe('connected');
      expect(summary.isActive).toBe(true);
      expect(summary.isUsable).toBe(true);
      expect(summary.gameId).toBe(testGameId);
      expect(summary.timeSinceHeartbeatMs).toBe(3000);
    });

    it('should return summary for error state', () => {
      const state = createErrorState('Connection failed', { nowMs: testNowMs });
      const summary = getConnectionSummary(state);

      expect(summary.kind).toBe('error');
      expect(summary.isActive).toBe(false);
      expect(summary.isUsable).toBe(false);
      expect(summary.error).toBe('Connection failed');
    });

    it('should return summary for reconnecting state', () => {
      const state = createReconnectingState(testGameId, testNowMs - 5000, 3, testNowMs);
      const summary = getConnectionSummary(state, testNowMs + 2000);

      expect(summary.kind).toBe('reconnecting');
      expect(summary.isActive).toBe(true);
      expect(summary.isUsable).toBe(false);
      expect(summary.gameId).toBe(testGameId);
      expect(summary.reconnectAttempts).toBe(3);
      expect(summary.timeSinceHeartbeatMs).toBe(7000);
    });
  });

  describe('State Machine Flow', () => {
    it('should handle complete happy path: idle -> connecting -> connected', () => {
      let state: ConnectionState = createIdleState();
      expect(state.kind).toBe('idle');

      state = startConnecting(state, testGameId, testNowMs);
      expect(state.kind).toBe('connecting');

      state = markConnected(state as ConnectionConnectingState, testNowMs + 1000);
      expect(state.kind).toBe('connected');
    });

    it('should handle reconnection flow: connected -> reconnecting -> connected', () => {
      let state: ConnectionState = createConnectedState(testGameId, testNowMs);

      state = startReconnecting(state as ConnectionConnectedState, testNowMs + 10000);
      expect(state.kind).toBe('reconnecting');
      expect((state as ConnectionReconnectingState).attemptCount).toBe(1);

      state = incrementReconnectAttempt(state as ConnectionReconnectingState, testNowMs + 13000);
      expect((state as ConnectionReconnectingState).attemptCount).toBe(2);

      state = markConnected(state as ConnectionReconnectingState, testNowMs + 15000);
      expect(state.kind).toBe('connected');
    });

    it('should handle reconnection failure: connected -> reconnecting -> error', () => {
      let state: ConnectionState = createConnectedState(testGameId, testNowMs);

      state = startReconnecting(state as ConnectionConnectedState, testNowMs + 10000);
      expect(state.kind).toBe('reconnecting');

      for (let i = 2; i <= 5; i++) {
        state = incrementReconnectAttempt(
          state as ConnectionReconnectingState,
          testNowMs + 10000 + i * 3000
        );
        expect((state as ConnectionReconnectingState).attemptCount).toBe(i);
      }

      state = markError(state as ConnectionReconnectingState, 'Max retries exceeded', {
        canRetry: false,
        nowMs: testNowMs + 30000,
      });
      expect(state.kind).toBe('error');
      expect((state as ConnectionErrorState).canRetry).toBe(false);
    });

    it('should handle retry from error: error -> connecting -> connected', () => {
      let state: ConnectionState = createErrorState('Previous error', {
        gameId: testGameId,
        canRetry: true,
        nowMs: testNowMs,
      });
      expect(state.kind).toBe('error');

      state = startConnecting(state, testGameId, testNowMs + 5000);
      expect(state.kind).toBe('connecting');

      state = markConnected(state as ConnectionConnectingState, testNowMs + 6000);
      expect(state.kind).toBe('connected');
    });

    it('should handle user-initiated disconnect', () => {
      let state: ConnectionState = createConnectedState(testGameId, testNowMs);

      state = markDisconnected(state, 'user_initiated', testNowMs + 60000);
      expect(state.kind).toBe('disconnected');
      expect((state as ConnectionDisconnectedState).reason).toBe('user_initiated');
      expect((state as ConnectionDisconnectedState).previousGameId).toBe(testGameId);
    });

    it('should handle game ending', () => {
      let state: ConnectionState = createConnectedState(testGameId, testNowMs);

      state = markDisconnected(state, 'game_ended', testNowMs + 300000);
      expect(state.kind).toBe('disconnected');
      expect((state as ConnectionDisconnectedState).reason).toBe('game_ended');
    });

    it('should handle reset from any state', () => {
      // From connected
      let state: ConnectionState = createConnectedState(testGameId, testNowMs);
      state = resetConnection();
      expect(state.kind).toBe('idle');

      // From error
      state = createErrorState('Error', { nowMs: testNowMs });
      state = resetConnection();
      expect(state.kind).toBe('idle');

      // From reconnecting
      state = createReconnectingState(testGameId, testNowMs, 3, testNowMs);
      state = resetConnection();
      expect(state.kind).toBe('idle');
    });
  });

  describe('Edge Cases', () => {
    it('should maintain immutability on state updates', () => {
      const original = createConnectedState(testGameId, testNowMs);
      const updated = updateHeartbeat(original, testNowMs + 5000);

      expect(original.lastHeartbeatAt).toBe(testNowMs);
      expect(updated.lastHeartbeatAt).toBe(testNowMs + 5000);
      expect(original).not.toBe(updated);
    });

    it('should handle transitions with same timestamp', () => {
      const connecting = createConnectingState(testGameId, testNowMs);
      const connected = markConnected(connecting, testNowMs);

      expect(connected.connectedAt).toBe(testNowMs);
    });

    it('should handle very large timestamps', () => {
      const largeTimestamp = Number.MAX_SAFE_INTEGER - 1000;
      const state = createConnectedState(testGameId, largeTimestamp);

      expect(state.connectedAt).toBe(largeTimestamp);
      expect(state.lastHeartbeatAt).toBe(largeTimestamp);
    });

    it('should handle empty string gameId', () => {
      const state = createConnectingState('', testNowMs);
      expect(state.gameId).toBe('');
      expect(getConnectionGameId(state)).toBe('');
    });

    it('should handle special characters in error messages', () => {
      const message = 'Error: <script>alert("xss")</script>';
      const state = createErrorState(message, { nowMs: testNowMs });
      expect(state.message).toBe(message);
    });
  });
});
