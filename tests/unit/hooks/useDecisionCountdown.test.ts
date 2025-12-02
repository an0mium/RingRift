/**
 * Tests for useDecisionCountdown hook.
 */

import { renderHook, act } from '@testing-library/react';
import { useDecisionCountdown } from '../../../src/client/hooks/useDecisionCountdown';
import type { PlayerChoice } from '../../../src/shared/types/game';
import type { DecisionPhaseTimeoutWarningPayload } from '../../../src/shared/types/websocket';

describe('useDecisionCountdown', () => {
  const createMockChoice = (overrides: Partial<PlayerChoice> = {}): PlayerChoice => ({
    id: 'choice-1',
    gameId: 'game-1',
    playerNumber: 1,
    description: 'Test choice',
    options: [],
    allowedMoveIds: [],
    timeoutMs: 30000,
    createdAt: new Date().toISOString(),
    ...overrides,
  });

  const createTimeoutWarning = (
    remainingMs: number,
    playerNumber: number = 1,
    choiceId?: string
  ): DecisionPhaseTimeoutWarningPayload => ({
    type: 'decision_phase_timeout_warning',
    data: {
      remainingMs,
      playerNumber,
      choiceId,
    },
  });

  describe('initial state', () => {
    it('should return null when no pending choice', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: null,
          baseTimeRemainingMs: null,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBeNull();
      expect(result.current.isServerOverrideActive).toBe(false);
      expect(result.current.isServerCapped).toBe(false);
    });

    it('should use base time when no server override', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice(),
          baseTimeRemainingMs: 10000,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(10000);
      expect(result.current.isServerOverrideActive).toBe(false);
      expect(result.current.isServerCapped).toBe(false);
    });

    it('should handle null base time', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice(),
          baseTimeRemainingMs: null,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBeNull();
    });
  });

  describe('server timeout warning', () => {
    it('should use server time when matching player and no base', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ playerNumber: 1 }),
          baseTimeRemainingMs: null,
          timeoutWarning: createTimeoutWarning(5000, 1),
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(5000);
      expect(result.current.isServerOverrideActive).toBe(true);
    });

    it('should use minimum of base and server time when both present', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ playerNumber: 1 }),
          baseTimeRemainingMs: 10000,
          timeoutWarning: createTimeoutWarning(5000, 1),
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(5000);
      expect(result.current.isServerOverrideActive).toBe(true);
      expect(result.current.isServerCapped).toBe(true);
    });

    it('should use base time when server time is higher', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ playerNumber: 1 }),
          baseTimeRemainingMs: 5000,
          timeoutWarning: createTimeoutWarning(10000, 1),
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(5000);
      expect(result.current.isServerOverrideActive).toBe(true);
      expect(result.current.isServerCapped).toBe(false); // server is higher, not capped
    });

    it('should ignore warning for different player', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ playerNumber: 1 }),
          baseTimeRemainingMs: 10000,
          timeoutWarning: createTimeoutWarning(5000, 2), // Player 2 warning
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(10000);
      expect(result.current.isServerOverrideActive).toBe(false);
    });

    it('should match by choiceId when provided', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ id: 'choice-1', playerNumber: 1 }),
          baseTimeRemainingMs: 10000,
          timeoutWarning: createTimeoutWarning(5000, 1, 'choice-1'),
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(5000);
      expect(result.current.isServerOverrideActive).toBe(true);
    });

    it('should ignore warning with non-matching choiceId', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ id: 'choice-1', playerNumber: 1 }),
          baseTimeRemainingMs: 10000,
          timeoutWarning: createTimeoutWarning(5000, 1, 'choice-2'), // Different choice
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(10000);
      expect(result.current.isServerOverrideActive).toBe(false);
    });
  });

  describe('normalization', () => {
    it('should clamp negative values to 0', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice(),
          baseTimeRemainingMs: -100,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(0);
    });

    it('should handle NaN values as null', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice(),
          baseTimeRemainingMs: NaN,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBeNull();
    });

    it('should handle 0 as valid value', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice(),
          baseTimeRemainingMs: 0,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(0);
    });
  });

  describe('state transitions', () => {
    it('should clear override when pending choice is cleared', () => {
      const initialChoice = createMockChoice({ playerNumber: 1 });
      const { result, rerender } = renderHook(
        ({ pendingChoice, baseTimeRemainingMs, timeoutWarning }) =>
          useDecisionCountdown({ pendingChoice, baseTimeRemainingMs, timeoutWarning }),
        {
          initialProps: {
            pendingChoice: initialChoice,
            baseTimeRemainingMs: 10000,
            timeoutWarning: createTimeoutWarning(5000, 1),
          },
        }
      );

      // Override should be active
      expect(result.current.isServerOverrideActive).toBe(true);
      expect(result.current.effectiveTimeRemainingMs).toBe(5000);

      // Clear the pending choice
      rerender({
        pendingChoice: null,
        baseTimeRemainingMs: null,
        timeoutWarning: undefined,
      });

      // Override should be cleared
      expect(result.current.isServerOverrideActive).toBe(false);
      expect(result.current.effectiveTimeRemainingMs).toBeNull();
    });

    it('should update when base time changes', () => {
      const choice = createMockChoice();
      const { result, rerender } = renderHook(
        ({ baseTimeRemainingMs }) =>
          useDecisionCountdown({
            pendingChoice: choice,
            baseTimeRemainingMs,
          }),
        {
          initialProps: { baseTimeRemainingMs: 10000 },
        }
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(10000);

      rerender({ baseTimeRemainingMs: 5000 });

      expect(result.current.effectiveTimeRemainingMs).toBe(5000);
    });

    it('should update when new timeout warning arrives', () => {
      const choice = createMockChoice({ playerNumber: 1 });
      const { result, rerender } = renderHook(
        ({ timeoutWarning }) =>
          useDecisionCountdown({
            pendingChoice: choice,
            baseTimeRemainingMs: 10000,
            timeoutWarning,
          }),
        {
          initialProps: {
            timeoutWarning: undefined as DecisionPhaseTimeoutWarningPayload | undefined,
          },
        }
      );

      expect(result.current.isServerOverrideActive).toBe(false);

      rerender({ timeoutWarning: createTimeoutWarning(3000, 1) });

      expect(result.current.isServerOverrideActive).toBe(true);
      expect(result.current.effectiveTimeRemainingMs).toBe(3000);
    });
  });

  describe('isServerCapped', () => {
    it('should be true when server time is lower than base', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ playerNumber: 1 }),
          baseTimeRemainingMs: 10000,
          timeoutWarning: createTimeoutWarning(5000, 1),
        })
      );

      expect(result.current.isServerCapped).toBe(true);
    });

    it('should be false when server time equals base', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ playerNumber: 1 }),
          baseTimeRemainingMs: 5000,
          timeoutWarning: createTimeoutWarning(5000, 1),
        })
      );

      expect(result.current.isServerCapped).toBe(false);
    });

    it('should be false when no base time', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice({ playerNumber: 1 }),
          baseTimeRemainingMs: null,
          timeoutWarning: createTimeoutWarning(5000, 1),
        })
      );

      expect(result.current.isServerCapped).toBe(false);
    });

    it('should be false when no server override', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createMockChoice(),
          baseTimeRemainingMs: 10000,
        })
      );

      expect(result.current.isServerCapped).toBe(false);
    });
  });
});
