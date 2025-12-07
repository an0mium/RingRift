import { renderHook, act } from '@testing-library/react';
import {
  useDecisionCountdown,
  UseDecisionCountdownArgs,
  DecisionCountdownState,
} from '../../../src/client/hooks/useDecisionCountdown';
import type { PlayerChoice } from '../../../src/shared/types/game';
import type { DecisionPhaseTimeoutWarningPayload } from '../../../src/shared/types/websocket';

describe('useDecisionCountdown', () => {
  // Helper to create a mock PlayerChoice
  const createChoice = (
    playerNumber: number,
    id = 'choice-1'
  ): PlayerChoice => ({
    id,
    playerNumber,
    type: 'capture_decision',
    options: [],
    prompt: 'Test prompt',
    deadline: Date.now() + 30000,
    context: {},
  });

  // Helper to create a timeout warning payload
  const createWarning = (
    playerNumber: number,
    remainingMs: number,
    choiceId?: string
  ): DecisionPhaseTimeoutWarningPayload => ({
    type: 'decision_phase_timeout_warning',
    data: {
      playerNumber,
      remainingMs,
      choiceId,
    },
  });

  describe('effectiveTimeRemainingMs', () => {
    it('returns null when no pending choice', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: null,
          baseTimeRemainingMs: 10000,
          timeoutWarning: null,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(10000);
    });

    it('returns base time when no server override', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createChoice(1),
          baseTimeRemainingMs: 15000,
          timeoutWarning: null,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBe(15000);
      expect(result.current.isServerOverrideActive).toBe(false);
      expect(result.current.isServerCapped).toBe(false);
    });

    it('returns null when both base and override are null', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createChoice(1),
          baseTimeRemainingMs: null,
          timeoutWarning: null,
        })
      );

      expect(result.current.effectiveTimeRemainingMs).toBeNull();
    });

    it('uses server override when it matches pending choice', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(1, 5000, 'choice-abc');

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 10000,
          timeoutWarning: warning,
        })
      );

      // Should use the minimum of base (10000) and override (5000)
      expect(result.current.effectiveTimeRemainingMs).toBe(5000);
      expect(result.current.isServerOverrideActive).toBe(true);
      expect(result.current.isServerCapped).toBe(true);
    });

    it('takes minimum when base < override', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(1, 20000, 'choice-abc');

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 8000,
          timeoutWarning: warning,
        })
      );

      // Base (8000) < override (20000), so use base
      expect(result.current.effectiveTimeRemainingMs).toBe(8000);
      expect(result.current.isServerOverrideActive).toBe(true);
      expect(result.current.isServerCapped).toBe(false); // override > base, not capped
    });

    it('takes minimum when override < base', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(1, 3000, 'choice-abc');

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 12000,
          timeoutWarning: warning,
        })
      );

      // Override (3000) < base (12000), so use override
      expect(result.current.effectiveTimeRemainingMs).toBe(3000);
      expect(result.current.isServerOverrideActive).toBe(true);
      expect(result.current.isServerCapped).toBe(true);
    });

    it('clamps negative values to 0', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(1, -500, 'choice-abc');

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 5000,
          timeoutWarning: warning,
        })
      );

      // Negative override clamped to 0, which is < base (5000)
      expect(result.current.effectiveTimeRemainingMs).toBe(0);
    });
  });

  describe('server override matching', () => {
    it('ignores warnings for different player number', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(2, 1000, 'choice-abc'); // Different player

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 10000,
          timeoutWarning: warning,
        })
      );

      // Should use base time since warning is for different player
      expect(result.current.effectiveTimeRemainingMs).toBe(10000);
      expect(result.current.isServerOverrideActive).toBe(false);
    });

    it('ignores warnings with mismatched choiceId', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(1, 1000, 'choice-xyz'); // Different choiceId

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 10000,
          timeoutWarning: warning,
        })
      );

      // Should use base time since choiceId doesn't match
      expect(result.current.effectiveTimeRemainingMs).toBe(10000);
      expect(result.current.isServerOverrideActive).toBe(false);
    });

    it('accepts warnings without choiceId when player matches', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(1, 2000); // No choiceId

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 10000,
          timeoutWarning: warning,
        })
      );

      // Should accept warning since no choiceId means it applies to any choice
      expect(result.current.effectiveTimeRemainingMs).toBe(2000);
      expect(result.current.isServerOverrideActive).toBe(true);
    });
  });

  describe('override clearing', () => {
    it('clears override when pending choice becomes null', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(1, 5000, 'choice-abc');

      const { result, rerender } = renderHook(
        (props: UseDecisionCountdownArgs) => useDecisionCountdown(props),
        {
          initialProps: {
            pendingChoice: choice,
            baseTimeRemainingMs: 10000,
            timeoutWarning: warning,
          },
        }
      );

      // Initially should have override
      expect(result.current.effectiveTimeRemainingMs).toBe(5000);
      expect(result.current.isServerOverrideActive).toBe(true);

      // Clear the pending choice
      rerender({
        pendingChoice: null,
        baseTimeRemainingMs: 10000,
        timeoutWarning: warning,
      });

      // Override should be cleared, use base
      expect(result.current.effectiveTimeRemainingMs).toBe(10000);
      expect(result.current.isServerOverrideActive).toBe(false);
    });

    it('clears override when pending choice becomes null then new choice', () => {
      const choice1 = createChoice(1, 'choice-1');
      const choice2 = createChoice(1, 'choice-2');
      const warning = createWarning(1, 5000, 'choice-1');

      const { result, rerender } = renderHook(
        (props: UseDecisionCountdownArgs) => useDecisionCountdown(props),
        {
          initialProps: {
            pendingChoice: choice1,
            baseTimeRemainingMs: 10000,
            timeoutWarning: warning,
          },
        }
      );

      // Initially should have override
      expect(result.current.effectiveTimeRemainingMs).toBe(5000);
      expect(result.current.isServerOverrideActive).toBe(true);

      // First clear the choice (simulating decision completion)
      rerender({
        pendingChoice: null,
        baseTimeRemainingMs: null,
        timeoutWarning: null,
      });

      // Override should be cleared
      expect(result.current.isServerOverrideActive).toBe(false);

      // Then set new choice with no warning
      rerender({
        pendingChoice: choice2,
        baseTimeRemainingMs: 8000,
        timeoutWarning: null,
      });

      // Should use new base with no override
      expect(result.current.effectiveTimeRemainingMs).toBe(8000);
      expect(result.current.isServerOverrideActive).toBe(false);
    });
  });

  describe('edge cases', () => {
    it('handles NaN remainingMs in warning', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning: DecisionPhaseTimeoutWarningPayload = {
        type: 'decision_phase_timeout_warning',
        data: {
          playerNumber: 1,
          remainingMs: NaN,
          choiceId: 'choice-abc',
        },
      };

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: 10000,
          timeoutWarning: warning,
        })
      );

      // NaN should be ignored, use base
      expect(result.current.effectiveTimeRemainingMs).toBe(10000);
      expect(result.current.isServerOverrideActive).toBe(false);
    });

    it('handles negative baseTimeRemainingMs', () => {
      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: createChoice(1),
          baseTimeRemainingMs: -1000,
          timeoutWarning: null,
        })
      );

      // Negative base should be clamped to 0
      expect(result.current.effectiveTimeRemainingMs).toBe(0);
    });

    it('uses override alone when base is null', () => {
      const choice = createChoice(1, 'choice-abc');
      const warning = createWarning(1, 7000, 'choice-abc');

      const { result } = renderHook(() =>
        useDecisionCountdown({
          pendingChoice: choice,
          baseTimeRemainingMs: null,
          timeoutWarning: warning,
        })
      );

      // No base, use override
      expect(result.current.effectiveTimeRemainingMs).toBe(7000);
      expect(result.current.isServerOverrideActive).toBe(true);
      // isServerCapped false because base is null
      expect(result.current.isServerCapped).toBe(false);
    });
  });
});
