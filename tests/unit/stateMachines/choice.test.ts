/**
 * Unit tests for choice.ts state machine
 *
 * Structural coverage for PlayerChoice lifecycle transitions as described in
 * P18.3-1 §3.2 (PlayerChoice timeouts) and §2.4 (connection sub-states):
 * - `fulfilled` – player responds before deadline.
 * - `expired`   – decision times out while player remains connected/eligible.
 * - `canceled`  – choice is explicitly canceled (including DISCONNECT).
 */
import {
  makePendingChoiceStatus,
  markChoiceFulfilled,
  markChoiceRejected,
  markChoiceCanceled,
  markChoiceExpired,
  type ChoiceStatus,
} from '../../../src/shared/stateMachines/choice';
import type { PlayerChoice } from '../../../src/shared/types/game';

describe('choice state machine', () => {
  const mockChoice: PlayerChoice = {
    id: 'choice-123',
    gameId: 'game-456',
    playerNumber: 1,
    type: 'region_order',
    options: [
      {
        regionId: 'region-a',
        size: 5,
        representativePosition: { x: 2, y: 3 },
        moveId: 'move-1',
      },
      {
        regionId: 'region-b',
        size: 3,
        representativePosition: { x: 5, y: 6 },
        moveId: 'move-2',
      },
    ],
    prompt: 'Choose region order',
  };

  describe('makePendingChoiceStatus', () => {
    it('should create a pending choice status with correct fields', () => {
      const now = 1000000;
      const timeoutMs = 30000;
      const status = makePendingChoiceStatus(mockChoice, timeoutMs, now);

      expect(status.kind).toBe('pending');
      expect(status.gameId).toBe('game-456');
      expect(status.choiceId).toBe('choice-123');
      expect(status.playerNumber).toBe(1);
      expect(status.choiceType).toBe('region_order');
      if (status.kind === 'pending') {
        expect(status.requestedAt).toBe(now);
        expect(status.deadlineAt).toBe(now + timeoutMs);
      }
    });

    it('should use current time if now is not provided', () => {
      const before = Date.now();
      const status = makePendingChoiceStatus(mockChoice, 30000);
      const after = Date.now();

      if (status.kind === 'pending') {
        expect(status.requestedAt).toBeGreaterThanOrEqual(before);
        expect(status.requestedAt).toBeLessThanOrEqual(after);
      }
    });

    it('should calculate correct deadline', () => {
      const status = makePendingChoiceStatus(mockChoice, 60000, 5000);

      if (status.kind === 'pending') {
        expect(status.deadlineAt).toBe(65000);
      }
    });

    it('should handle choice without type (defensive edge case)', () => {
      // This tests the defensive handling when type is somehow missing at runtime
      // (e.g., malformed data from an external source)
      const choiceWithoutType = { ...mockChoice } as unknown as PlayerChoice;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      delete (choiceWithoutType as any).type;
      const status = makePendingChoiceStatus(choiceWithoutType, 30000, 1000);
      expect(status.choiceType).toBeUndefined();
    });
  });

  describe('markChoiceFulfilled', () => {
    it('should transition pending to fulfilled', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const fulfilledStatus = markChoiceFulfilled(pendingStatus, 2000);

      expect(fulfilledStatus.kind).toBe('fulfilled');
      expect(fulfilledStatus.gameId).toBe('game-456');
      expect(fulfilledStatus.choiceId).toBe('choice-123');
      expect(fulfilledStatus.playerNumber).toBe(1);
      expect(fulfilledStatus.choiceType).toBe('region_order');
      if (fulfilledStatus.kind === 'fulfilled') {
        expect(fulfilledStatus.completedAt).toBe(2000);
      }
    });

    it('should use current time if now is not provided', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const before = Date.now();
      const fulfilledStatus = markChoiceFulfilled(pendingStatus);
      const after = Date.now();

      if (fulfilledStatus.kind === 'fulfilled') {
        expect(fulfilledStatus.completedAt).toBeGreaterThanOrEqual(before);
        expect(fulfilledStatus.completedAt).toBeLessThanOrEqual(after);
      }
    });

    it('should preserve choiceType from previous status', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const fulfilledStatus = markChoiceFulfilled(pendingStatus, 2000);
      expect(fulfilledStatus.choiceType).toBe(pendingStatus.choiceType);
    });
  });

  describe('markChoiceRejected', () => {
    it('should transition to rejected with INVALID_OPTION reason', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const rejectedStatus = markChoiceRejected(pendingStatus, 'INVALID_OPTION', 2500);

      expect(rejectedStatus.kind).toBe('rejected');
      expect(rejectedStatus.gameId).toBe('game-456');
      expect(rejectedStatus.choiceId).toBe('choice-123');
      if (rejectedStatus.kind === 'rejected') {
        expect(rejectedStatus.reason).toBe('INVALID_OPTION');
        expect(rejectedStatus.completedAt).toBe(2500);
      }
    });

    it('should transition to rejected with PLAYER_MISMATCH reason', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const rejectedStatus = markChoiceRejected(pendingStatus, 'PLAYER_MISMATCH', 3000);

      if (rejectedStatus.kind === 'rejected') {
        expect(rejectedStatus.reason).toBe('PLAYER_MISMATCH');
      }
    });

    it('should use current time if now is not provided', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const before = Date.now();
      const rejectedStatus = markChoiceRejected(pendingStatus, 'INVALID_OPTION');
      const after = Date.now();

      if (rejectedStatus.kind === 'rejected') {
        expect(rejectedStatus.completedAt).toBeGreaterThanOrEqual(before);
        expect(rejectedStatus.completedAt).toBeLessThanOrEqual(after);
      }
    });
  });

  describe('markChoiceCanceled', () => {
    it('should transition to canceled with SERVER_CANCEL reason', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const canceledStatus = markChoiceCanceled(pendingStatus, 'SERVER_CANCEL', 4000);

      expect(canceledStatus.kind).toBe('canceled');
      expect(canceledStatus.gameId).toBe('game-456');
      expect(canceledStatus.choiceId).toBe('choice-123');
      if (canceledStatus.kind === 'canceled') {
        expect(canceledStatus.reason).toBe('SERVER_CANCEL');
        expect(canceledStatus.completedAt).toBe(4000);
      }
    });

    it('should transition to canceled with DISCONNECT reason', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const canceledStatus = markChoiceCanceled(pendingStatus, 'DISCONNECT', 5000);

      if (canceledStatus.kind === 'canceled') {
        expect(canceledStatus.reason).toBe('DISCONNECT');
      }
    });

    it('should use current time if now is not provided', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const before = Date.now();
      const canceledStatus = markChoiceCanceled(pendingStatus, 'SERVER_CANCEL');
      const after = Date.now();

      if (canceledStatus.kind === 'canceled') {
        expect(canceledStatus.completedAt).toBeGreaterThanOrEqual(before);
        expect(canceledStatus.completedAt).toBeLessThanOrEqual(after);
      }
    });
  });

  describe('markChoiceExpired', () => {
    it('should transition pending to expired preserving requestedAt and deadlineAt', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const expiredStatus = markChoiceExpired(pendingStatus, 31500);

      expect(expiredStatus.kind).toBe('expired');
      expect(expiredStatus.gameId).toBe('game-456');
      expect(expiredStatus.choiceId).toBe('choice-123');
      expect(expiredStatus.playerNumber).toBe(1);
      if (expiredStatus.kind === 'expired') {
        expect(expiredStatus.requestedAt).toBe(1000);
        expect(expiredStatus.deadlineAt).toBe(31000);
        expect(expiredStatus.completedAt).toBe(31500);
      }
    });

    it('should use current time if now is not provided', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const before = Date.now();
      const expiredStatus = markChoiceExpired(pendingStatus);
      const after = Date.now();

      if (expiredStatus.kind === 'expired') {
        expect(expiredStatus.completedAt).toBeGreaterThanOrEqual(before);
        expect(expiredStatus.completedAt).toBeLessThanOrEqual(after);
      }
    });

    // Branch coverage: when previous status is not 'pending'
    it('should handle non-pending status defensively (fulfilled)', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const fulfilledStatus = markChoiceFulfilled(pendingStatus, 2000);
      const now = 35000;

      const expiredStatus = markChoiceExpired(fulfilledStatus, now);

      expect(expiredStatus.kind).toBe('expired');
      if (expiredStatus.kind === 'expired') {
        // When previous is not pending, both requestedAt and deadlineAt fallback to now
        expect(expiredStatus.requestedAt).toBe(now);
        expect(expiredStatus.deadlineAt).toBe(now);
        expect(expiredStatus.completedAt).toBe(now);
      }
    });

    it('should handle non-pending status defensively (rejected)', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const rejectedStatus = markChoiceRejected(pendingStatus, 'INVALID_OPTION', 2000);
      const now = 40000;

      const expiredStatus = markChoiceExpired(rejectedStatus, now);

      expect(expiredStatus.kind).toBe('expired');
      if (expiredStatus.kind === 'expired') {
        expect(expiredStatus.requestedAt).toBe(now);
        expect(expiredStatus.deadlineAt).toBe(now);
      }
    });

    it('should handle non-pending status defensively (canceled)', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const canceledStatus = markChoiceCanceled(pendingStatus, 'DISCONNECT', 2000);
      const now = 45000;

      const expiredStatus = markChoiceExpired(canceledStatus, now);

      expect(expiredStatus.kind).toBe('expired');
      if (expiredStatus.kind === 'expired') {
        expect(expiredStatus.requestedAt).toBe(now);
        expect(expiredStatus.deadlineAt).toBe(now);
      }
    });

    it('should preserve choiceType in expired status', () => {
      const pendingStatus = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const expiredStatus = markChoiceExpired(pendingStatus, 50000);

      expect(expiredStatus.choiceType).toBe('region_order');
    });
  });

  describe('complete lifecycle transitions', () => {
    it('should support pending -> fulfilled flow', () => {
      const pending = makePendingChoiceStatus(mockChoice, 30000, 1000);
      expect(pending.kind).toBe('pending');

      const fulfilled = markChoiceFulfilled(pending, 5000);
      expect(fulfilled.kind).toBe('fulfilled');
    });

    it('should support pending -> rejected flow', () => {
      const pending = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const rejected = markChoiceRejected(pending, 'INVALID_OPTION', 5000);
      expect(rejected.kind).toBe('rejected');
    });

    it('should support pending -> canceled flow', () => {
      const pending = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const canceled = markChoiceCanceled(pending, 'SERVER_CANCEL', 5000);
      expect(canceled.kind).toBe('canceled');
    });

    it('should support pending -> expired flow', () => {
      const pending = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const expired = markChoiceExpired(pending, 35000);
      expect(expired.kind).toBe('expired');
    });
  });

  describe('disconnect vs timeout vs response semantics', () => {
    it('distinguishes DISCONNECT cancellation from timeout and fulfilled states', () => {
      const pending = makePendingChoiceStatus(mockChoice, 30000, 1000);

      const fulfilled = markChoiceFulfilled(pending, 2000);
      const expired = markChoiceExpired(pending, 31000);
      const canceledOnDisconnect = markChoiceCanceled(pending, 'DISCONNECT', 1500);

      expect(fulfilled.kind).toBe('fulfilled');
      if (fulfilled.kind === 'fulfilled') {
        expect(fulfilled.completedAt).toBe(2000);
      }

      expect(expired.kind).toBe('expired');
      if (expired.kind === 'expired') {
        expect(expired.completedAt).toBe(31000);
      }

      expect(canceledOnDisconnect.kind).toBe('canceled');
      if (canceledOnDisconnect.kind === 'canceled') {
        expect(canceledOnDisconnect.reason).toBe('DISCONNECT');
        expect(canceledOnDisconnect.completedAt).toBe(1500);
      }
    });
  });

  describe('field preservation across transitions', () => {
    it('should preserve gameId across all transitions', () => {
      const pending = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const fulfilled = markChoiceFulfilled(pending, 2000);
      const rejected = markChoiceRejected(pending, 'INVALID_OPTION', 3000);
      const canceled = markChoiceCanceled(pending, 'DISCONNECT', 4000);
      const expired = markChoiceExpired(pending, 5000);

      expect(pending.gameId).toBe('game-456');
      expect(fulfilled.gameId).toBe('game-456');
      expect(rejected.gameId).toBe('game-456');
      expect(canceled.gameId).toBe('game-456');
      expect(expired.gameId).toBe('game-456');
    });

    it('should preserve choiceId across all transitions', () => {
      const pending = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const fulfilled = markChoiceFulfilled(pending, 2000);

      expect(pending.choiceId).toBe('choice-123');
      expect(fulfilled.choiceId).toBe('choice-123');
    });

    it('should preserve playerNumber across all transitions', () => {
      const pending = makePendingChoiceStatus(mockChoice, 30000, 1000);
      const canceled = markChoiceCanceled(pending, 'SERVER_CANCEL', 2000);

      expect(pending.playerNumber).toBe(1);
      expect(canceled.playerNumber).toBe(1);
    });
  });
});
