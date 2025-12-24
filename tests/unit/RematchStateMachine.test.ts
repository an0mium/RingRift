/**
 * Tests for RematchStateMachine state machine
 * @module tests/unit/RematchStateMachine.test
 */

import {
  // Types
  type RematchState,
  type RematchIdleState,
  type RematchPendingRequestState,
  type RematchPendingResponseState,
  type RematchAcceptedState,
  type RematchDeclinedState,
  type RematchExpiredState,
  // State constructors
  createRematchIdleState,
  createPendingRequestState,
  createPendingResponseState,
  createRematchAcceptedState,
  createRematchDeclinedState,
  createRematchExpiredState,
  // State transitions
  requestRematch,
  receiveRematchRequest,
  acceptRematch,
  declineRematchLocally,
  receiveRematchDecline,
  expireRematch,
  resetRematch,
  // Query helpers
  isRematchActive,
  isAwaitingOpponentResponse,
  isAwaitingLocalResponse,
  hasNewGame,
  getNewGameId,
  getLegacyRematchStatus,
  // Summary
  getRematchSummary,
} from '../../src/client/state/RematchStateMachine';

describe('RematchStateMachine', () => {
  const testGameId = 'test-game-123';
  const testNewGameId = 'new-game-456';
  const testRequestId = 'request-789';
  const testRequesterUsername = 'opponent_player';
  const testNowMs = 1000000;

  describe('State Constructors', () => {
    describe('createRematchIdleState', () => {
      it('should create an idle state', () => {
        const state = createRematchIdleState();
        expect(state.kind).toBe('idle');
      });
    });

    describe('createPendingRequestState', () => {
      it('should create a pending request state with game ID and timestamp', () => {
        const state = createPendingRequestState(testGameId, testNowMs);

        expect(state.kind).toBe('pending_request');
        expect(state.gameId).toBe(testGameId);
        expect(state.requestedAt).toBe(testNowMs);
      });

      it('should use Date.now() when nowMs is not provided', () => {
        const before = Date.now();
        const state = createPendingRequestState(testGameId);
        const after = Date.now();

        expect(state.requestedAt).toBeGreaterThanOrEqual(before);
        expect(state.requestedAt).toBeLessThanOrEqual(after);
      });
    });

    describe('createPendingResponseState', () => {
      it('should create a pending response state with all fields', () => {
        const state = createPendingResponseState(
          testRequestId,
          testGameId,
          testRequesterUsername,
          testNowMs
        );

        expect(state.kind).toBe('pending_response');
        expect(state.requestId).toBe(testRequestId);
        expect(state.gameId).toBe(testGameId);
        expect(state.requesterUsername).toBe(testRequesterUsername);
        expect(state.receivedAt).toBe(testNowMs);
      });

      it('should use Date.now() when nowMs is not provided', () => {
        const before = Date.now();
        const state = createPendingResponseState(testRequestId, testGameId, testRequesterUsername);
        const after = Date.now();

        expect(state.receivedAt).toBeGreaterThanOrEqual(before);
        expect(state.receivedAt).toBeLessThanOrEqual(after);
      });
    });

    describe('createRematchAcceptedState', () => {
      it('should create an accepted state with new and original game IDs', () => {
        const state = createRematchAcceptedState(testNewGameId, testGameId, testNowMs);

        expect(state.kind).toBe('accepted');
        expect(state.newGameId).toBe(testNewGameId);
        expect(state.originalGameId).toBe(testGameId);
        expect(state.acceptedAt).toBe(testNowMs);
      });

      it('should use Date.now() when nowMs is not provided', () => {
        const before = Date.now();
        const state = createRematchAcceptedState(testNewGameId, testGameId);
        const after = Date.now();

        expect(state.acceptedAt).toBeGreaterThanOrEqual(before);
        expect(state.acceptedAt).toBeLessThanOrEqual(after);
      });
    });

    describe('createRematchDeclinedState', () => {
      it('should create a declined state with local decliner', () => {
        const state = createRematchDeclinedState('local', testGameId, testNowMs);

        expect(state.kind).toBe('declined');
        expect(state.declinedBy).toBe('local');
        expect(state.gameId).toBe(testGameId);
        expect(state.declinedAt).toBe(testNowMs);
      });

      it('should create a declined state with opponent decliner', () => {
        const state = createRematchDeclinedState('opponent', testGameId, testNowMs);

        expect(state.kind).toBe('declined');
        expect(state.declinedBy).toBe('opponent');
      });

      it('should use Date.now() when nowMs is not provided', () => {
        const before = Date.now();
        const state = createRematchDeclinedState('local', testGameId);
        const after = Date.now();

        expect(state.declinedAt).toBeGreaterThanOrEqual(before);
        expect(state.declinedAt).toBeLessThanOrEqual(after);
      });
    });

    describe('createRematchExpiredState', () => {
      it('should create an expired state with game ID and timestamp', () => {
        const state = createRematchExpiredState(testGameId, testNowMs);

        expect(state.kind).toBe('expired');
        expect(state.gameId).toBe(testGameId);
        expect(state.expiredAt).toBe(testNowMs);
      });

      it('should use Date.now() when nowMs is not provided', () => {
        const before = Date.now();
        const state = createRematchExpiredState(testGameId);
        const after = Date.now();

        expect(state.expiredAt).toBeGreaterThanOrEqual(before);
        expect(state.expiredAt).toBeLessThanOrEqual(after);
      });
    });
  });

  describe('State Transitions', () => {
    describe('requestRematch', () => {
      it('should transition from idle to pending_request', () => {
        const idle = createRematchIdleState();
        const pending = requestRematch(idle, testGameId, testNowMs);

        expect(pending.kind).toBe('pending_request');
        expect(pending.gameId).toBe(testGameId);
        expect(pending.requestedAt).toBe(testNowMs);
      });

      it('should warn but still work from invalid states', () => {
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation();

        const accepted = createRematchAcceptedState(testNewGameId, testGameId, testNowMs);
        const pending = requestRematch(accepted, 'another-game', testNowMs + 1000);

        expect(warnSpy).toHaveBeenCalledWith(
          expect.stringContaining('requestRematch called from invalid state')
        );
        expect(pending.kind).toBe('pending_request');

        warnSpy.mockRestore();
      });
    });

    describe('receiveRematchRequest', () => {
      it('should transition from idle to pending_response', () => {
        const idle = createRematchIdleState();
        const pending = receiveRematchRequest(
          idle,
          testRequestId,
          testGameId,
          testRequesterUsername,
          testNowMs
        );

        expect(pending.kind).toBe('pending_response');
        expect(pending.requestId).toBe(testRequestId);
        expect(pending.gameId).toBe(testGameId);
        expect(pending.requesterUsername).toBe(testRequesterUsername);
        expect(pending.receivedAt).toBe(testNowMs);
      });

      it('should warn but still work from invalid states', () => {
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation();

        const expired = createRematchExpiredState(testGameId, testNowMs);
        const pending = receiveRematchRequest(
          expired,
          testRequestId,
          'new-game',
          testRequesterUsername,
          testNowMs + 1000
        );

        expect(warnSpy).toHaveBeenCalledWith(
          expect.stringContaining('receiveRematchRequest called from invalid state')
        );
        expect(pending.kind).toBe('pending_response');

        warnSpy.mockRestore();
      });
    });

    describe('acceptRematch', () => {
      it('should transition from pending_request to accepted', () => {
        const pendingRequest = createPendingRequestState(testGameId, testNowMs);
        const accepted = acceptRematch(pendingRequest, testNewGameId, testNowMs + 5000);

        expect(accepted.kind).toBe('accepted');
        expect(accepted.newGameId).toBe(testNewGameId);
        expect(accepted.originalGameId).toBe(testGameId);
        expect(accepted.acceptedAt).toBe(testNowMs + 5000);
      });

      it('should transition from pending_response to accepted', () => {
        const pendingResponse = createPendingResponseState(
          testRequestId,
          testGameId,
          testRequesterUsername,
          testNowMs
        );
        const accepted = acceptRematch(pendingResponse, testNewGameId, testNowMs + 3000);

        expect(accepted.kind).toBe('accepted');
        expect(accepted.newGameId).toBe(testNewGameId);
        expect(accepted.originalGameId).toBe(testGameId);
      });
    });

    describe('declineRematchLocally', () => {
      it('should transition from pending_response to declined with local decliner', () => {
        const pendingResponse = createPendingResponseState(
          testRequestId,
          testGameId,
          testRequesterUsername,
          testNowMs
        );
        const declined = declineRematchLocally(pendingResponse, testNowMs + 2000);

        expect(declined.kind).toBe('declined');
        expect(declined.declinedBy).toBe('local');
        expect(declined.gameId).toBe(testGameId);
        expect(declined.declinedAt).toBe(testNowMs + 2000);
      });
    });

    describe('receiveRematchDecline', () => {
      it('should transition from pending_request to declined with opponent decliner', () => {
        const pendingRequest = createPendingRequestState(testGameId, testNowMs);
        const declined = receiveRematchDecline(pendingRequest, testNowMs + 5000);

        expect(declined.kind).toBe('declined');
        expect(declined.declinedBy).toBe('opponent');
        expect(declined.gameId).toBe(testGameId);
        expect(declined.declinedAt).toBe(testNowMs + 5000);
      });
    });

    describe('expireRematch', () => {
      it('should transition from pending_request to expired', () => {
        const pendingRequest = createPendingRequestState(testGameId, testNowMs);
        const expired = expireRematch(pendingRequest, testNowMs + 30000);

        expect(expired.kind).toBe('expired');
        expect(expired.gameId).toBe(testGameId);
        expect(expired.expiredAt).toBe(testNowMs + 30000);
      });

      it('should transition from pending_response to expired', () => {
        const pendingResponse = createPendingResponseState(
          testRequestId,
          testGameId,
          testRequesterUsername,
          testNowMs
        );
        const expired = expireRematch(pendingResponse, testNowMs + 30000);

        expect(expired.kind).toBe('expired');
        expect(expired.gameId).toBe(testGameId);
        expect(expired.expiredAt).toBe(testNowMs + 30000);
      });
    });

    describe('resetRematch', () => {
      it('should return idle state', () => {
        const state = resetRematch();
        expect(state.kind).toBe('idle');
      });
    });
  });

  describe('Query Helpers', () => {
    describe('isRematchActive', () => {
      it('should return true for pending_request state', () => {
        const state = createPendingRequestState(testGameId, testNowMs);
        expect(isRematchActive(state)).toBe(true);
      });

      it('should return true for pending_response state', () => {
        const state = createPendingResponseState(
          testRequestId,
          testGameId,
          testRequesterUsername,
          testNowMs
        );
        expect(isRematchActive(state)).toBe(true);
      });

      it('should return false for idle state', () => {
        const state = createRematchIdleState();
        expect(isRematchActive(state)).toBe(false);
      });

      it('should return false for accepted state', () => {
        const state = createRematchAcceptedState(testNewGameId, testGameId, testNowMs);
        expect(isRematchActive(state)).toBe(false);
      });

      it('should return false for declined state', () => {
        const state = createRematchDeclinedState('local', testGameId, testNowMs);
        expect(isRematchActive(state)).toBe(false);
      });

      it('should return false for expired state', () => {
        const state = createRematchExpiredState(testGameId, testNowMs);
        expect(isRematchActive(state)).toBe(false);
      });
    });

    describe('isAwaitingOpponentResponse', () => {
      it('should return true only for pending_request state', () => {
        expect(isAwaitingOpponentResponse(createPendingRequestState(testGameId, testNowMs))).toBe(
          true
        );

        expect(isAwaitingOpponentResponse(createRematchIdleState())).toBe(false);
        expect(
          isAwaitingOpponentResponse(
            createPendingResponseState(testRequestId, testGameId, testRequesterUsername, testNowMs)
          )
        ).toBe(false);
        expect(
          isAwaitingOpponentResponse(
            createRematchAcceptedState(testNewGameId, testGameId, testNowMs)
          )
        ).toBe(false);
        expect(
          isAwaitingOpponentResponse(createRematchDeclinedState('local', testGameId, testNowMs))
        ).toBe(false);
        expect(isAwaitingOpponentResponse(createRematchExpiredState(testGameId, testNowMs))).toBe(
          false
        );
      });
    });

    describe('isAwaitingLocalResponse', () => {
      it('should return true only for pending_response state', () => {
        expect(
          isAwaitingLocalResponse(
            createPendingResponseState(testRequestId, testGameId, testRequesterUsername, testNowMs)
          )
        ).toBe(true);

        expect(isAwaitingLocalResponse(createRematchIdleState())).toBe(false);
        expect(isAwaitingLocalResponse(createPendingRequestState(testGameId, testNowMs))).toBe(
          false
        );
        expect(
          isAwaitingLocalResponse(createRematchAcceptedState(testNewGameId, testGameId, testNowMs))
        ).toBe(false);
        expect(
          isAwaitingLocalResponse(createRematchDeclinedState('local', testGameId, testNowMs))
        ).toBe(false);
        expect(isAwaitingLocalResponse(createRematchExpiredState(testGameId, testNowMs))).toBe(
          false
        );
      });
    });

    describe('hasNewGame', () => {
      it('should return true only for accepted state', () => {
        expect(hasNewGame(createRematchAcceptedState(testNewGameId, testGameId, testNowMs))).toBe(
          true
        );

        expect(hasNewGame(createRematchIdleState())).toBe(false);
        expect(hasNewGame(createPendingRequestState(testGameId, testNowMs))).toBe(false);
        expect(
          hasNewGame(
            createPendingResponseState(testRequestId, testGameId, testRequesterUsername, testNowMs)
          )
        ).toBe(false);
        expect(hasNewGame(createRematchDeclinedState('local', testGameId, testNowMs))).toBe(false);
        expect(hasNewGame(createRematchExpiredState(testGameId, testNowMs))).toBe(false);
      });
    });

    describe('getNewGameId', () => {
      it('should return newGameId for accepted state', () => {
        const state = createRematchAcceptedState(testNewGameId, testGameId, testNowMs);
        expect(getNewGameId(state)).toBe(testNewGameId);
      });

      it('should return null for non-accepted states', () => {
        expect(getNewGameId(createRematchIdleState())).toBeNull();
        expect(getNewGameId(createPendingRequestState(testGameId, testNowMs))).toBeNull();
        expect(
          getNewGameId(
            createPendingResponseState(testRequestId, testGameId, testRequesterUsername, testNowMs)
          )
        ).toBeNull();
        expect(getNewGameId(createRematchDeclinedState('local', testGameId, testNowMs))).toBeNull();
        expect(getNewGameId(createRematchExpiredState(testGameId, testNowMs))).toBeNull();
      });
    });

    describe('getLegacyRematchStatus', () => {
      it('should return accepted for accepted state', () => {
        const state = createRematchAcceptedState(testNewGameId, testGameId, testNowMs);
        expect(getLegacyRematchStatus(state)).toBe('accepted');
      });

      it('should return declined for declined state', () => {
        const state = createRematchDeclinedState('opponent', testGameId, testNowMs);
        expect(getLegacyRematchStatus(state)).toBe('declined');
      });

      it('should return expired for expired state', () => {
        const state = createRematchExpiredState(testGameId, testNowMs);
        expect(getLegacyRematchStatus(state)).toBe('expired');
      });

      it('should return null for non-terminal states', () => {
        expect(getLegacyRematchStatus(createRematchIdleState())).toBeNull();
        expect(getLegacyRematchStatus(createPendingRequestState(testGameId, testNowMs))).toBeNull();
        expect(
          getLegacyRematchStatus(
            createPendingResponseState(testRequestId, testGameId, testRequesterUsername, testNowMs)
          )
        ).toBeNull();
      });
    });
  });

  describe('getRematchSummary', () => {
    it('should return summary for idle state', () => {
      const state = createRematchIdleState();
      const summary = getRematchSummary(state);

      expect(summary.kind).toBe('idle');
      expect(summary.isActive).toBe(false);
      expect(summary.message).toBe('No rematch in progress');
    });

    it('should return summary for pending_request state', () => {
      const state = createPendingRequestState(testGameId, testNowMs);
      const summary = getRematchSummary(state);

      expect(summary.kind).toBe('pending_request');
      expect(summary.isActive).toBe(true);
      expect(summary.message).toBe('Waiting for opponent to respond...');
    });

    it('should return summary for pending_response state', () => {
      const state = createPendingResponseState(
        testRequestId,
        testGameId,
        testRequesterUsername,
        testNowMs
      );
      const summary = getRematchSummary(state);

      expect(summary.kind).toBe('pending_response');
      expect(summary.isActive).toBe(true);
      expect(summary.message).toBe(`${testRequesterUsername} wants a rematch`);
      expect(summary.requestId).toBe(testRequestId);
      expect(summary.requesterUsername).toBe(testRequesterUsername);
    });

    it('should return summary for accepted state', () => {
      const state = createRematchAcceptedState(testNewGameId, testGameId, testNowMs);
      const summary = getRematchSummary(state);

      expect(summary.kind).toBe('accepted');
      expect(summary.isActive).toBe(false);
      expect(summary.message).toBe('Rematch accepted! Starting new game...');
      expect(summary.newGameId).toBe(testNewGameId);
    });

    it('should return summary for locally declined state', () => {
      const state = createRematchDeclinedState('local', testGameId, testNowMs);
      const summary = getRematchSummary(state);

      expect(summary.kind).toBe('declined');
      expect(summary.isActive).toBe(false);
      expect(summary.message).toBe('You declined the rematch');
    });

    it('should return summary for opponent declined state', () => {
      const state = createRematchDeclinedState('opponent', testGameId, testNowMs);
      const summary = getRematchSummary(state);

      expect(summary.kind).toBe('declined');
      expect(summary.isActive).toBe(false);
      expect(summary.message).toBe('Opponent declined the rematch');
    });

    it('should return summary for expired state', () => {
      const state = createRematchExpiredState(testGameId, testNowMs);
      const summary = getRematchSummary(state);

      expect(summary.kind).toBe('expired');
      expect(summary.isActive).toBe(false);
      expect(summary.message).toBe('Rematch request expired');
    });
  });

  describe('State Machine Flow', () => {
    it('should handle happy path: local user requests -> opponent accepts', () => {
      let state: RematchState = createRematchIdleState();
      expect(state.kind).toBe('idle');

      state = requestRematch(state, testGameId, testNowMs);
      expect(state.kind).toBe('pending_request');

      state = acceptRematch(state as RematchPendingRequestState, testNewGameId, testNowMs + 5000);
      expect(state.kind).toBe('accepted');
      expect((state as RematchAcceptedState).newGameId).toBe(testNewGameId);
    });

    it('should handle happy path: opponent requests -> local user accepts', () => {
      let state: RematchState = createRematchIdleState();
      expect(state.kind).toBe('idle');

      state = receiveRematchRequest(
        state,
        testRequestId,
        testGameId,
        testRequesterUsername,
        testNowMs
      );
      expect(state.kind).toBe('pending_response');

      state = acceptRematch(state as RematchPendingResponseState, testNewGameId, testNowMs + 3000);
      expect(state.kind).toBe('accepted');
      expect((state as RematchAcceptedState).newGameId).toBe(testNewGameId);
    });

    it('should handle decline path: local user requests -> opponent declines', () => {
      let state: RematchState = createRematchIdleState();

      state = requestRematch(state, testGameId, testNowMs);
      expect(state.kind).toBe('pending_request');

      state = receiveRematchDecline(state as RematchPendingRequestState, testNowMs + 5000);
      expect(state.kind).toBe('declined');
      expect((state as RematchDeclinedState).declinedBy).toBe('opponent');
    });

    it('should handle decline path: opponent requests -> local user declines', () => {
      let state: RematchState = createRematchIdleState();

      state = receiveRematchRequest(
        state,
        testRequestId,
        testGameId,
        testRequesterUsername,
        testNowMs
      );
      expect(state.kind).toBe('pending_response');

      state = declineRematchLocally(state as RematchPendingResponseState, testNowMs + 2000);
      expect(state.kind).toBe('declined');
      expect((state as RematchDeclinedState).declinedBy).toBe('local');
    });

    it('should handle expiry path: local user requests -> timeout', () => {
      let state: RematchState = createRematchIdleState();

      state = requestRematch(state, testGameId, testNowMs);
      expect(state.kind).toBe('pending_request');

      state = expireRematch(state as RematchPendingRequestState, testNowMs + 30000);
      expect(state.kind).toBe('expired');
    });

    it('should handle expiry path: opponent requests -> timeout', () => {
      let state: RematchState = createRematchIdleState();

      state = receiveRematchRequest(
        state,
        testRequestId,
        testGameId,
        testRequesterUsername,
        testNowMs
      );
      expect(state.kind).toBe('pending_response');

      state = expireRematch(state as RematchPendingResponseState, testNowMs + 30000);
      expect(state.kind).toBe('expired');
    });

    it('should handle reset from any terminal state', () => {
      // From accepted
      let state: RematchState = createRematchAcceptedState(testNewGameId, testGameId, testNowMs);
      state = resetRematch();
      expect(state.kind).toBe('idle');

      // From declined
      state = createRematchDeclinedState('opponent', testGameId, testNowMs);
      state = resetRematch();
      expect(state.kind).toBe('idle');

      // From expired
      state = createRematchExpiredState(testGameId, testNowMs);
      state = resetRematch();
      expect(state.kind).toBe('idle');
    });

    it('should handle full cycle: request -> decline -> new request -> accept', () => {
      let state: RematchState = createRematchIdleState();

      // First attempt - declined
      state = requestRematch(state, testGameId, testNowMs);
      state = receiveRematchDecline(state as RematchPendingRequestState, testNowMs + 5000);
      expect(state.kind).toBe('declined');

      // Reset and try again
      state = resetRematch();
      expect(state.kind).toBe('idle');

      // Second attempt - accepted
      state = requestRematch(state, testGameId, testNowMs + 10000);
      state = acceptRematch(state as RematchPendingRequestState, testNewGameId, testNowMs + 15000);
      expect(state.kind).toBe('accepted');
    });
  });

  describe('Edge Cases', () => {
    it('should maintain immutability on state transitions', () => {
      const original = createPendingRequestState(testGameId, testNowMs);
      const accepted = acceptRematch(original, testNewGameId, testNowMs + 5000);

      expect(original.kind).toBe('pending_request');
      expect(accepted.kind).toBe('accepted');
      expect(original).not.toBe(accepted);
    });

    it('should handle empty string gameId', () => {
      const state = createPendingRequestState('', testNowMs);
      expect(state.gameId).toBe('');
    });

    it('should handle empty string requesterUsername', () => {
      const state = createPendingResponseState(testRequestId, testGameId, '', testNowMs);
      expect(state.requesterUsername).toBe('');

      const summary = getRematchSummary(state);
      expect(summary.message).toBe(' wants a rematch');
    });

    it('should handle very large timestamps', () => {
      const largeTimestamp = Number.MAX_SAFE_INTEGER - 1000;
      const state = createPendingRequestState(testGameId, largeTimestamp);

      expect(state.requestedAt).toBe(largeTimestamp);
    });

    it('should handle special characters in requesterUsername', () => {
      const specialUsername = 'player_<script>alert("xss")</script>';
      const state = createPendingResponseState(
        testRequestId,
        testGameId,
        specialUsername,
        testNowMs
      );

      expect(state.requesterUsername).toBe(specialUsername);

      const summary = getRematchSummary(state);
      expect(summary.message).toBe(`${specialUsername} wants a rematch`);
      expect(summary.requesterUsername).toBe(specialUsername);
    });

    it('should handle consecutive same-game requests after reset', () => {
      let state: RematchState = createRematchIdleState();

      for (let i = 0; i < 3; i++) {
        state = requestRematch(state, testGameId, testNowMs + i * 60000);
        expect(state.kind).toBe('pending_request');
        expect((state as RematchPendingRequestState).gameId).toBe(testGameId);

        state = expireRematch(state as RematchPendingRequestState, testNowMs + i * 60000 + 30000);
        expect(state.kind).toBe('expired');

        state = resetRematch();
        expect(state.kind).toBe('idle');
      }
    });

    it('should preserve gameId through accept transition', () => {
      const pendingRequest = createPendingRequestState('original-game-id', testNowMs);
      const accepted = acceptRematch(pendingRequest, testNewGameId, testNowMs + 5000);

      expect(accepted.originalGameId).toBe('original-game-id');
      expect(accepted.newGameId).toBe(testNewGameId);
    });

    it('should preserve gameId through decline transition', () => {
      const pendingResponse = createPendingResponseState(
        testRequestId,
        'original-game-id',
        testRequesterUsername,
        testNowMs
      );
      const declined = declineRematchLocally(pendingResponse, testNowMs + 2000);

      expect(declined.gameId).toBe('original-game-id');
    });
  });

  describe('Type Guards', () => {
    it('should narrow types correctly with isRematchActive', () => {
      const pending = createPendingRequestState(testGameId, testNowMs);

      if (isRematchActive(pending)) {
        // TypeScript should recognize pending as RematchPendingRequestState | RematchPendingResponseState
        expect(pending.gameId).toBe(testGameId);
      }
    });

    it('should narrow types correctly with isAwaitingOpponentResponse', () => {
      const pending = createPendingRequestState(testGameId, testNowMs);

      if (isAwaitingOpponentResponse(pending)) {
        // TypeScript should recognize pending as RematchPendingRequestState
        expect(pending.requestedAt).toBe(testNowMs);
      }
    });

    it('should narrow types correctly with isAwaitingLocalResponse', () => {
      const pending = createPendingResponseState(
        testRequestId,
        testGameId,
        testRequesterUsername,
        testNowMs
      );

      if (isAwaitingLocalResponse(pending)) {
        // TypeScript should recognize pending as RematchPendingResponseState
        expect(pending.requestId).toBe(testRequestId);
        expect(pending.requesterUsername).toBe(testRequesterUsername);
      }
    });

    it('should narrow types correctly with hasNewGame', () => {
      const accepted = createRematchAcceptedState(testNewGameId, testGameId, testNowMs);

      if (hasNewGame(accepted)) {
        // TypeScript should recognize accepted as RematchAcceptedState
        expect(accepted.newGameId).toBe(testNewGameId);
        expect(accepted.originalGameId).toBe(testGameId);
      }
    });
  });
});
