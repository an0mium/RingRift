import { runWithTimeout } from '../../../src/shared/utils/timeout';
import { createCancellationSource } from '../../../src/shared/utils/cancellation';

describe('shared/utils/timeout', () => {
  beforeEach(() => {
    jest.useRealTimers();
  });

  it('returns canceled immediately when token is already canceled before start', async () => {
    const source = createCancellationSource();
    const reason = { code: 'PRE_CANCELED' };
    source.cancel(reason);

    const operation = jest.fn(async () => 'should-not-run');
    const now = jest.fn().mockReturnValue(1_000);

    const result = await runWithTimeout(operation, {
      timeoutMs: 5_000,
      token: source.token,
      now,
    });

    expect(operation).not.toHaveBeenCalled();
    expect(result).toEqual({
      kind: 'canceled',
      durationMs: 0,
      cancellationReason: reason,
    });
  });

  it('returns ok result and duration when operation completes before timeout', async () => {
    jest.useFakeTimers();

    const operation = jest.fn(async () => 'value');
    const now = jest
      .fn()
      // start
      .mockReturnValueOnce(1_000)
      // completion
      .mockReturnValueOnce(1_250);

    const resultPromise = runWithTimeout(operation, {
      timeoutMs: 10_000,
      now,
    });

    const result = await resultPromise;

    expect(operation).toHaveBeenCalledTimes(1);
    expect(result.kind).toBe('ok');
    expect(result.value).toBe('value');
    expect(result.durationMs).toBe(250);
  });

  it('returns timeout result when operation does not settle before timeout', async () => {
    jest.useFakeTimers();

    const operation = jest.fn(() => new Promise<never>(() => {}));
    const now = jest
      .fn()
      // start
      .mockReturnValueOnce(0)
      // after timeout fires
      .mockReturnValueOnce(1_000);

    const resultPromise = runWithTimeout(operation, {
      timeoutMs: 1_000,
      now,
    });

    // Allow the operation and timeout to be scheduled.
    await Promise.resolve();

    jest.advanceTimersByTime(1_000);

    const result = await resultPromise;

    expect(operation).toHaveBeenCalledTimes(1);
    expect(result.kind).toBe('timeout');
    expect(result.durationMs).toBe(1_000);
  });

  it('treats zero timeoutMs as an immediate timeout when the operation does not win the race', async () => {
    jest.useFakeTimers();

    const operation = jest.fn(() => new Promise<never>(() => {}));
    const now = jest
      .fn()
      // start
      .mockReturnValueOnce(0)
      // after timeout fires immediately
      .mockReturnValueOnce(0);

    const resultPromise = runWithTimeout(operation, {
      timeoutMs: 0,
      now,
    });

    // Allow the timeout to be scheduled.
    await Promise.resolve();
    jest.advanceTimersByTime(0);

    const result = await resultPromise;

    expect(operation).toHaveBeenCalledTimes(1);
    expect(result.kind).toBe('timeout');
    expect(result.durationMs).toBe(0);
  });

  it('maps thrown cancellation errors to a canceled result with reason', async () => {
    const source = createCancellationSource();
    const cancellationReason = { code: 'CANCEL_FROM_OPERATION' };

    const now = jest
      .fn()
      // start
      .mockReturnValueOnce(10)
      // observation after throw
      .mockReturnValueOnce(42);

    const operation = jest.fn(async () => {
      source.cancel(cancellationReason);
      source.token.throwIfCanceled('inside operation');
      return 123 as const;
    });

    const result = await runWithTimeout(operation, {
      timeoutMs: 5_000,
      now,
    });

    expect(operation).toHaveBeenCalledTimes(1);
    expect(result.kind).toBe('canceled');
    expect(result.durationMs).toBe(32);
    expect(result.cancellationReason).toBe(cancellationReason);
  });

  it('rethrows non-timeout, non-cancellation errors', async () => {
    const now = jest.fn().mockReturnValue(0);
    const error = new Error('boom');

    const operation = jest.fn(async () => {
      throw error;
    });

    await expect(
      runWithTimeout(operation, {
        timeoutMs: 5_000,
        now,
      })
    ).rejects.toBe(error);
    expect(operation).toHaveBeenCalledTimes(1);
  });

  it('uses Date.now by default when now function is not provided', async () => {
    const operation = jest.fn(async () => 'result');

    // Don't provide a custom `now` function - uses Date.now by default
    const result = await runWithTimeout(operation, {
      timeoutMs: 10_000,
      // no `now` or `token` provided
    });

    expect(operation).toHaveBeenCalledTimes(1);
    expect(result.kind).toBe('ok');
    expect(result.value).toBe('result');
    // durationMs should be a number (we can't predict exact value since it uses real Date.now)
    expect(typeof result.durationMs).toBe('number');
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('clears the internal timeout handle when operation settles', async () => {
    // Patch setTimeout / clearTimeout to track the handle used by runWithTimeout.
    const originalSetTimeout = global.setTimeout;
    const originalClearTimeout = global.clearTimeout;

    try {
      const fakeHandle = { id: 1 };
      const clearSpy = jest.fn();

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (global as any).setTimeout = (cb: () => void, _ms: number) => {
        // Do not actually schedule the callback; just return our handle.
        // The happy-path operation should win the race before this fires.
        void cb; // suppress unused warning
        return fakeHandle as unknown as ReturnType<typeof setTimeout>;
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (global as any).clearTimeout = (handle: unknown) => {
        clearSpy(handle);
      };

      const operation = jest.fn(async () => 'value');
      const now = jest
        .fn()
        // start
        .mockReturnValueOnce(0)
        // completion
        .mockReturnValueOnce(10);

      const result = await runWithTimeout(operation, {
        timeoutMs: 1_000,
        now,
      });

      expect(result.kind).toBe('ok');
      expect(clearSpy).toHaveBeenCalledWith(fakeHandle);
    } finally {
      global.setTimeout = originalSetTimeout;
      global.clearTimeout = originalClearTimeout;
    }
  });
});
