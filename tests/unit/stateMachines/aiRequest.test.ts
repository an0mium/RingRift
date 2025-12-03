/**
 * Tests for AI request state machine (src/shared/stateMachines/aiRequest.ts)
 *
 * Tests the AIRequestState lifecycle transitions: idle → queued → in_flight → completed/failed/timed_out/canceled
 */

import {
  AIRequestState,
  idleAIRequest,
  isTerminalState,
  isCancelable,
  markQueued,
  markInFlight,
  markFallbackLocal,
  markCompleted,
  markTimedOut,
  markFailed,
  markCanceled,
  getTerminalKind,
  isDeadlineExceeded,
} from '../../../src/shared/stateMachines/aiRequest';

describe('aiRequest state machine', () => {
  describe('idleAIRequest constant', () => {
    it('has kind idle', () => {
      expect(idleAIRequest).toEqual({ kind: 'idle' });
    });
  });

  describe('isTerminalState', () => {
    it('returns false for idle', () => {
      expect(isTerminalState({ kind: 'idle' })).toBe(false);
    });

    it('returns false for queued', () => {
      expect(isTerminalState({ kind: 'queued', requestedAt: 1000 })).toBe(false);
    });

    it('returns false for in_flight', () => {
      expect(
        isTerminalState({ kind: 'in_flight', requestedAt: 1000, lastAttemptAt: 1000, attempt: 1 })
      ).toBe(false);
    });

    it('returns false for fallback_local', () => {
      expect(
        isTerminalState({ kind: 'fallback_local', requestedAt: 1000, lastAttemptAt: 1500 })
      ).toBe(false);
    });

    it('returns true for completed', () => {
      expect(isTerminalState({ kind: 'completed', completedAt: 2000 })).toBe(true);
    });

    it('returns true for failed', () => {
      expect(
        isTerminalState({ kind: 'failed', completedAt: 2000, code: 'AI_SERVICE_TIMEOUT' })
      ).toBe(true);
    });

    it('returns true for canceled', () => {
      expect(
        isTerminalState({ kind: 'canceled', completedAt: 2000, reason: 'game_terminated' })
      ).toBe(true);
    });

    it('returns true for timed_out', () => {
      expect(
        isTerminalState({
          kind: 'timed_out',
          requestedAt: 1000,
          completedAt: 2000,
          durationMs: 1000,
          attempt: 1,
        })
      ).toBe(true);
    });
  });

  describe('isCancelable', () => {
    it('returns false for idle', () => {
      expect(isCancelable({ kind: 'idle' })).toBe(false);
    });

    it('returns true for queued', () => {
      expect(isCancelable({ kind: 'queued', requestedAt: 1000 })).toBe(true);
    });

    it('returns true for in_flight', () => {
      expect(
        isCancelable({ kind: 'in_flight', requestedAt: 1000, lastAttemptAt: 1000, attempt: 1 })
      ).toBe(true);
    });

    it('returns true for fallback_local', () => {
      expect(isCancelable({ kind: 'fallback_local', requestedAt: 1000, lastAttemptAt: 1500 })).toBe(
        true
      );
    });

    it('returns false for completed', () => {
      expect(isCancelable({ kind: 'completed', completedAt: 2000 })).toBe(false);
    });

    it('returns false for failed', () => {
      expect(isCancelable({ kind: 'failed', completedAt: 2000, code: 'AI_SERVICE_TIMEOUT' })).toBe(
        false
      );
    });

    it('returns false for canceled', () => {
      expect(isCancelable({ kind: 'canceled', completedAt: 2000, reason: 'game_terminated' })).toBe(
        false
      );
    });
  });

  describe('markQueued', () => {
    it('creates queued state with requestedAt', () => {
      const result = markQueued(1000);
      expect(result).toEqual({ kind: 'queued', requestedAt: 1000 });
    });

    it('includes optional timeoutMs', () => {
      const result = markQueued(1000, 5000);
      expect(result).toEqual({ kind: 'queued', requestedAt: 1000, timeoutMs: 5000 });
    });

    it('uses Date.now when no timestamp provided', () => {
      const before = Date.now();
      const result = markQueued();
      const after = Date.now();
      expect(result.kind).toBe('queued');
      if (result.kind === 'queued') {
        expect(result.requestedAt).toBeGreaterThanOrEqual(before);
        expect(result.requestedAt).toBeLessThanOrEqual(after);
      }
    });
  });

  describe('markInFlight', () => {
    it('creates in_flight state from queued', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 1000 };
      const result = markInFlight(prev, 1500);

      expect(result).toEqual({
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
      });
    });

    it('preserves requestedAt from queued state', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 500 };
      const result = markInFlight(prev, 2000);
      expect(result.kind).toBe('in_flight');
      if (result.kind === 'in_flight') {
        expect(result.requestedAt).toBe(500);
      }
    });

    it('increments attempt on retry from in_flight', () => {
      const prev: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 2,
      };
      const result = markInFlight(prev, 2000);
      expect(result.kind).toBe('in_flight');
      if (result.kind === 'in_flight') {
        expect(result.attempt).toBe(3);
        expect(result.requestedAt).toBe(1000);
        expect(result.lastAttemptAt).toBe(2000);
      }
    });

    it('sets deadlineAt from timeoutMs parameter', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 1000 };
      const result = markInFlight(prev, 2000, 5000);
      expect(result.kind).toBe('in_flight');
      if (result.kind === 'in_flight') {
        expect(result.deadlineAt).toBe(7000); // 2000 + 5000
      }
    });

    it('inherits timeoutMs from queued state to calculate deadlineAt', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 1000, timeoutMs: 3000 };
      const result = markInFlight(prev, 2000);
      expect(result.kind).toBe('in_flight');
      if (result.kind === 'in_flight') {
        expect(result.deadlineAt).toBe(5000); // 2000 + 3000
      }
    });

    it('uses now for requestedAt when previous is not queued or in_flight', () => {
      const prev: AIRequestState = { kind: 'idle' };
      const result = markInFlight(prev, 3000);
      expect(result.kind).toBe('in_flight');
      if (result.kind === 'in_flight') {
        expect(result.requestedAt).toBe(3000);
        expect(result.attempt).toBe(1);
      }
    });
  });

  describe('markFallbackLocal', () => {
    it('creates fallback_local state from in_flight', () => {
      const prev: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 2,
      };
      const result = markFallbackLocal(prev, 2000);

      expect(result).toEqual({
        kind: 'fallback_local',
        requestedAt: 1000,
        lastAttemptAt: 2000,
      });
    });

    it('preserves requestedAt from queued state', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 800 };
      const result = markFallbackLocal(prev, 2500);
      expect(result.kind).toBe('fallback_local');
      if (result.kind === 'fallback_local') {
        expect(result.requestedAt).toBe(800);
      }
    });

    it('uses now for requestedAt when previous is neither in_flight nor queued', () => {
      const prev: AIRequestState = { kind: 'idle' };
      const result = markFallbackLocal(prev, 3000);
      expect(result.kind).toBe('fallback_local');
      if (result.kind === 'fallback_local') {
        expect(result.requestedAt).toBe(3000);
      }
    });
  });

  describe('markCompleted', () => {
    it('creates completed state with latencyMs from in_flight', () => {
      const prev: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
      };
      const result = markCompleted(prev, 2000);

      expect(result).toEqual({
        kind: 'completed',
        completedAt: 2000,
        latencyMs: 1000,
      });
    });

    it('calculates latencyMs from queued state', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 500 };
      const result = markCompleted(prev, 1500);
      expect(result.kind).toBe('completed');
      if (result.kind === 'completed') {
        expect(result.latencyMs).toBe(1000);
      }
    });

    it('calculates latencyMs from fallback_local state', () => {
      const prev: AIRequestState = { kind: 'fallback_local', requestedAt: 100, lastAttemptAt: 500 };
      const result = markCompleted(prev, 600);
      expect(result.kind).toBe('completed');
      if (result.kind === 'completed') {
        expect(result.latencyMs).toBe(500);
      }
    });

    it('handles undefined previous state', () => {
      const result = markCompleted(undefined, 3000);
      expect(result).toEqual({
        kind: 'completed',
        completedAt: 3000,
        latencyMs: 0, // now - now = 0
      });
    });

    it('uses Date.now when no timestamp provided', () => {
      const before = Date.now();
      const result = markCompleted(undefined);
      const after = Date.now();
      expect(result.kind).toBe('completed');
      if (result.kind === 'completed') {
        expect(result.completedAt).toBeGreaterThanOrEqual(before);
        expect(result.completedAt).toBeLessThanOrEqual(after);
      }
    });
  });

  describe('markTimedOut', () => {
    it('creates timed_out state from in_flight', () => {
      const prev: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 3,
      };
      const result = markTimedOut(prev, 5000);

      expect(result).toEqual({
        kind: 'timed_out',
        requestedAt: 1000,
        completedAt: 5000,
        durationMs: 4000,
        attempt: 3,
      });
    });

    it('creates timed_out state from queued', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 2000 };
      const result = markTimedOut(prev, 7000);
      expect(result.kind).toBe('timed_out');
      if (result.kind === 'timed_out') {
        expect(result.durationMs).toBe(5000);
        expect(result.attempt).toBe(1);
      }
    });

    it('uses now for requestedAt when previous is neither in_flight nor queued', () => {
      const prev: AIRequestState = { kind: 'idle' };
      const result = markTimedOut(prev, 3000);
      expect(result.kind).toBe('timed_out');
      if (result.kind === 'timed_out') {
        expect(result.requestedAt).toBe(3000);
        expect(result.durationMs).toBe(0);
      }
    });
  });

  describe('markFailed', () => {
    it('creates failed state with code', () => {
      const prev: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
      };
      const result = markFailed('AI_SERVICE_TIMEOUT', undefined, prev, 2500);

      expect(result).toEqual({
        kind: 'failed',
        completedAt: 2500,
        code: 'AI_SERVICE_TIMEOUT',
        aiErrorType: undefined,
        durationMs: 1500,
      });
    });

    it('includes aiErrorType when provided', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 1000 };
      const result = markFailed('AI_SERVICE_TIMEOUT', 'NetworkError', prev, 3000);
      expect(result.kind).toBe('failed');
      if (result.kind === 'failed') {
        expect(result.aiErrorType).toBe('NetworkError');
        expect(result.durationMs).toBe(2000);
      }
    });

    it('handles undefined previous state', () => {
      const result = markFailed('AI_SERVICE_OVERLOADED', undefined, undefined, 5000);
      expect(result.kind).toBe('failed');
      if (result.kind === 'failed') {
        expect(result.code).toBe('AI_SERVICE_OVERLOADED');
        expect(result.durationMs).toBe(0);
      }
    });

    it('calculates durationMs from fallback_local state', () => {
      const prev: AIRequestState = {
        kind: 'fallback_local',
        requestedAt: 500,
        lastAttemptAt: 1000,
      };
      const result = markFailed('AI_SERVICE_TIMEOUT', undefined, prev, 2000);
      expect(result.kind).toBe('failed');
      if (result.kind === 'failed') {
        expect(result.durationMs).toBe(1500);
      }
    });
  });

  describe('markCanceled', () => {
    it('creates canceled state with reason', () => {
      const prev: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
      };
      const result = markCanceled('game_terminated', prev, 2000);

      expect(result).toEqual({
        kind: 'canceled',
        completedAt: 2000,
        reason: 'game_terminated',
        durationMs: 1000,
      });
    });

    it('handles player_disconnected reason', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 500 };
      const result = markCanceled('player_disconnected', prev, 1500);
      expect(result.kind).toBe('canceled');
      if (result.kind === 'canceled') {
        expect(result.reason).toBe('player_disconnected');
        expect(result.durationMs).toBe(1000);
      }
    });

    it('handles session_cleanup reason', () => {
      const prev: AIRequestState = { kind: 'fallback_local', requestedAt: 200, lastAttemptAt: 500 };
      const result = markCanceled('session_cleanup', prev, 800);
      expect(result.kind).toBe('canceled');
      if (result.kind === 'canceled') {
        expect(result.reason).toBe('session_cleanup');
        expect(result.durationMs).toBe(600);
      }
    });

    it('handles manual reason', () => {
      const result = markCanceled('manual', undefined, 3000);
      expect(result.kind).toBe('canceled');
      if (result.kind === 'canceled') {
        expect(result.reason).toBe('manual');
        expect(result).not.toHaveProperty('durationMs');
      }
    });

    it('omits durationMs when previous state is not cancelable', () => {
      const result = markCanceled('game_terminated', undefined, 5000);
      expect(result.kind).toBe('canceled');
      expect(result).not.toHaveProperty('durationMs');
    });

    it('handles custom string reason', () => {
      const prev: AIRequestState = { kind: 'queued', requestedAt: 1000 };
      const result = markCanceled('custom_reason', prev, 2000);
      expect(result.kind).toBe('canceled');
      if (result.kind === 'canceled') {
        expect(result.reason).toBe('custom_reason');
      }
    });
  });

  describe('getTerminalKind', () => {
    it('returns null for idle', () => {
      expect(getTerminalKind({ kind: 'idle' })).toBe(null);
    });

    it('returns null for queued', () => {
      expect(getTerminalKind({ kind: 'queued', requestedAt: 1000 })).toBe(null);
    });

    it('returns null for in_flight', () => {
      expect(
        getTerminalKind({ kind: 'in_flight', requestedAt: 1000, lastAttemptAt: 1000, attempt: 1 })
      ).toBe(null);
    });

    it('returns null for fallback_local', () => {
      expect(
        getTerminalKind({ kind: 'fallback_local', requestedAt: 1000, lastAttemptAt: 1500 })
      ).toBe(null);
    });

    it('returns "completed" for completed state', () => {
      expect(getTerminalKind({ kind: 'completed', completedAt: 2000 })).toBe('completed');
    });

    it('returns "failed" for failed state', () => {
      expect(
        getTerminalKind({ kind: 'failed', completedAt: 2000, code: 'AI_SERVICE_TIMEOUT' })
      ).toBe('failed');
    });

    it('returns "canceled" for canceled state', () => {
      expect(
        getTerminalKind({ kind: 'canceled', completedAt: 2000, reason: 'game_terminated' })
      ).toBe('canceled');
    });

    it('returns "timed_out" for timed_out state', () => {
      expect(
        getTerminalKind({
          kind: 'timed_out',
          requestedAt: 1000,
          completedAt: 2000,
          durationMs: 1000,
          attempt: 1,
        })
      ).toBe('timed_out');
    });
  });

  describe('isDeadlineExceeded', () => {
    it('returns false for non-in_flight state', () => {
      expect(isDeadlineExceeded({ kind: 'idle' }, 5000)).toBe(false);
      expect(isDeadlineExceeded({ kind: 'queued', requestedAt: 1000 }, 5000)).toBe(false);
      expect(isDeadlineExceeded({ kind: 'completed', completedAt: 2000 }, 5000)).toBe(false);
    });

    it('returns false when in_flight has no deadlineAt', () => {
      const state: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
      };
      expect(isDeadlineExceeded(state, 10000)).toBe(false);
    });

    it('returns false when now is before deadline', () => {
      const state: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
        deadlineAt: 5000,
      };
      expect(isDeadlineExceeded(state, 4000)).toBe(false);
    });

    it('returns true when now equals deadline', () => {
      const state: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
        deadlineAt: 5000,
      };
      expect(isDeadlineExceeded(state, 5000)).toBe(true);
    });

    it('returns true when now is after deadline', () => {
      const state: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
        deadlineAt: 5000,
      };
      expect(isDeadlineExceeded(state, 6000)).toBe(true);
    });

    it('uses Date.now when no timestamp provided', () => {
      const futureDeadline = Date.now() + 100000;
      const state: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
        deadlineAt: futureDeadline,
      };
      expect(isDeadlineExceeded(state)).toBe(false);

      const pastDeadline = Date.now() - 100000;
      const pastState: AIRequestState = {
        kind: 'in_flight',
        requestedAt: 1000,
        lastAttemptAt: 1500,
        attempt: 1,
        deadlineAt: pastDeadline,
      };
      expect(isDeadlineExceeded(pastState)).toBe(true);
    });
  });

  describe('realistic AI request lifecycle', () => {
    it('supports full lifecycle: idle → queued → in_flight → completed', () => {
      let state: AIRequestState = idleAIRequest;
      expect(state.kind).toBe('idle');

      state = markQueued(1000);
      expect(state.kind).toBe('queued');

      state = markInFlight(state, 1100);
      expect(state.kind).toBe('in_flight');

      state = markCompleted(state, 2000);
      expect(state.kind).toBe('completed');
      expect(isTerminalState(state)).toBe(true);
    });

    it('supports retry on failure: queued → in_flight → in_flight (retry) → completed', () => {
      let state: AIRequestState = markQueued(1000);

      // First attempt
      state = markInFlight(state, 1100);
      if (state.kind === 'in_flight') {
        expect(state.attempt).toBe(1);
      }

      // Retry (stays in_flight with incremented attempt)
      state = markInFlight(state, 1500);
      if (state.kind === 'in_flight') {
        expect(state.attempt).toBe(2);
      }

      // Success
      state = markCompleted(state, 2000);
      expect(state.kind).toBe('completed');
    });

    it('supports fallback path: queued → in_flight → fallback_local → completed', () => {
      let state: AIRequestState = markQueued(1000);
      state = markInFlight(state, 1100);
      state = markFallbackLocal(state, 1500);
      expect(state.kind).toBe('fallback_local');
      expect(isCancelable(state)).toBe(true);

      state = markCompleted(state, 2000);
      expect(state.kind).toBe('completed');
    });

    it('supports timeout path: queued → in_flight → timed_out', () => {
      let state: AIRequestState = markQueued(1000, 3000);
      state = markInFlight(state, 1100);

      if (state.kind === 'in_flight') {
        expect(state.deadlineAt).toBe(4100); // 1100 + 3000
      }

      state = markTimedOut(state, 5000);
      expect(state.kind).toBe('timed_out');
      expect(isTerminalState(state)).toBe(true);
    });

    it('supports cancellation at any cancelable state', () => {
      // Cancel from queued
      let state: AIRequestState = markQueued(1000);
      expect(isCancelable(state)).toBe(true);
      state = markCanceled('game_terminated', state, 1500);
      expect(state.kind).toBe('canceled');

      // Cancel from in_flight
      state = markQueued(2000);
      state = markInFlight(state, 2100);
      expect(isCancelable(state)).toBe(true);
      state = markCanceled('player_disconnected', state, 2500);
      expect(state.kind).toBe('canceled');

      // Cancel from fallback_local
      state = markQueued(3000);
      state = markInFlight(state, 3100);
      state = markFallbackLocal(state, 3500);
      expect(isCancelable(state)).toBe(true);
      state = markCanceled('session_cleanup', state, 4000);
      expect(state.kind).toBe('canceled');
    });
  });
});
