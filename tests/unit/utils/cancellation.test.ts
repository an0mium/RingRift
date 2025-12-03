import {
  createCancellationSource,
  createLinkedCancellationSource,
  type CancellationToken,
} from '../../../src/shared/utils/cancellation';

describe('shared/utils/cancellation', () => {
  describe('createCancellationSource', () => {
    it('starts non-canceled and becomes canceled with an optional reason', () => {
      const source = createCancellationSource();

      expect(source.token.isCanceled).toBe(false);
      expect(source.token.reason).toBeUndefined();

      const reason = new Error('test cancel');
      source.cancel(reason);

      expect(source.token.isCanceled).toBe(true);
      expect(source.token.reason).toBe(reason);
    });

    it('cancel is idempotent and preserves the first reason', () => {
      const source = createCancellationSource();

      const first = new Error('first');
      const second = new Error('second');

      source.cancel(first);
      source.cancel(second);

      expect(source.token.isCanceled).toBe(true);
      expect(source.token.reason).toBe(first);
    });

    it('throwIfCanceled is a no-op before cancel and throws with enriched error after', () => {
      const source = createCancellationSource();

      // Before cancel, nothing should be thrown.
      expect(() => source.token.throwIfCanceled()).not.toThrow();

      const reason = { code: 'TEST_REASON', meta: { attempt: 1 } };
      source.cancel(reason);

      try {
        source.token.throwIfCanceled('while doing work');

        fail('Expected throwIfCanceled to throw after cancellation');
      } catch (err) {
        const error = err as Error & { cancellationReason: unknown };

        expect(error).toBeInstanceOf(Error);
        expect(error.message).toBe('Operation canceled (while doing work)');
        expect(error.cancellationReason).toBe(reason);
      }
    });

    it('throwIfCanceled without context message produces simpler error message', () => {
      const source = createCancellationSource();
      const reason = new Error('abort');
      source.cancel(reason);

      try {
        source.token.throwIfCanceled(); // No context message
        fail('Expected throwIfCanceled to throw after cancellation');
      } catch (err) {
        const error = err as Error & { cancellationReason: unknown };

        expect(error).toBeInstanceOf(Error);
        expect(error.message).toBe('Operation canceled'); // No parenthetical detail
        expect(error.cancellationReason).toBe(reason);
      }
    });
  });

  describe('createLinkedCancellationSource', () => {
    function createParent(): { token: CancellationToken; cancel: (reason?: unknown) => void } {
      const parent = createCancellationSource();
      return { token: parent.token, cancel: parent.cancel }; // narrow surface for tests
    }

    it('propagates cancellation immediately when parent is already canceled', () => {
      const parent = createCancellationSource();
      const reason = new Error('parent canceled early');
      parent.cancel(reason);

      const child = createLinkedCancellationSource(parent.token);

      expect(child.token.isCanceled).toBe(true);
      expect(child.token.reason).toBe(reason);
    });

    it('propagates cancellation when syncFromParent is invoked later', () => {
      const { token: parentToken, cancel: cancelParent } = createParent();
      const child = createLinkedCancellationSource(parentToken);

      expect(child.token.isCanceled).toBe(false);
      expect(child.token.reason).toBeUndefined();

      const reason = { code: 'PARENT_CANCEL', phase: 'mid-operation' };
      cancelParent(reason);

      // Child only observes the change once syncFromParent is called.
      child.syncFromParent();

      expect(child.token.isCanceled).toBe(true);
      expect(child.token.reason).toBe(reason);
    });

    it('allows canceling the child independently without affecting the parent', () => {
      const { token: parentToken, cancel: cancelParent } = createParent();
      const child = createLinkedCancellationSource(parentToken);

      const childReason = new Error('child only');
      child.cancel(childReason);

      expect(child.token.isCanceled).toBe(true);
      expect(child.token.reason).toBe(childReason);

      // Parent remains unaffected.
      expect(parentToken.isCanceled).toBe(false);
      expect(parentToken.reason).toBeUndefined();

      // Subsequent parent cancel should not overwrite the child reason.
      const parentReason = new Error('parent later');
      cancelParent(parentReason);
      child.syncFromParent();

      expect(parentToken.isCanceled).toBe(true);
      expect(parentToken.reason).toBe(parentReason);
      // Child keeps its original reason because it was already canceled.
      expect(child.token.reason).toBe(childReason);
    });
  });
});
