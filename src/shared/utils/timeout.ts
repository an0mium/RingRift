// Shared timeout helpers for async operations.
//
// These utilities provide a small, typed wrapper around Promise-based
// operations so that callers can:
//   - Enforce explicit time budgets for AI requests, WebSocket flows, etc.
//   - Record durations in milliseconds for observability.
//   - Distinguish between successful completion, timeout, and cancellation.
//
// They are intentionally host-agnostic and designed to be used alongside the
// cancellation primitives in src/shared/utils/cancellation.ts and the
// shared state machines in src/shared/stateMachines/**.

import type { CancellationReason, CancellationToken } from './cancellation';

/** Error type augmented with timeout marker */
interface TimeoutError extends Error {
  isTimeout: true;
}

/** Error type augmented with cancellation info */
interface CancellationError extends Error {
  cancellationReason: CancellationReason;
}

function isTimeoutError(error: unknown): error is TimeoutError {
  return error instanceof Error && (error as TimeoutError).isTimeout === true;
}

function isCancellationError(error: unknown): error is CancellationError {
  return error instanceof Error && (error as CancellationError).cancellationReason !== undefined;
}

export type TimedOperationOutcome = 'ok' | 'timeout' | 'canceled';

export interface TimedOperationResult<T> {
  /** Outcome of the operation. */
  kind: TimedOperationOutcome;
  /**
   * Wall-clock duration in milliseconds between start and resolution
   * (success, timeout, or cancellation observation).
   */
  durationMs: number;
  /** Present when kind === 'ok'. */
  value?: T;
  /** Present when kind === 'canceled'. */
  cancellationReason?: CancellationReason;
}

export interface TimedOperationOptions {
  /** Maximum allowed duration in milliseconds. */
  timeoutMs: number;
  /** Optional cancellation token for cooperative cancellation. */
  token?: CancellationToken;
  /** Clock dependency (overridable for tests). Defaults to Date.now. */
  now?: () => number;
}

/**
 * Run an async operation with an upper time bound, returning a structured
 * result instead of throwing on timeout.
 *
 * Notes on cancellation integration:
 * - If a CancellationToken is provided and is already canceled before the
 *   operation starts, the function returns `kind: 'canceled'` immediately.
 * - If the underlying operation throws an error that carries a
 *   `cancellationReason` property (as produced by `CancellationToken`
 *   throwIfCanceled), the result is mapped to `kind: 'canceled'`.
 * - This helper does **not** forcibly abort in-flight work; callers should
 *   check `token.throwIfCanceled()` at appropriate boundaries inside the
 *   operation to cooperate with cancellation.
 */
export async function runWithTimeout<T>(
  operation: () => Promise<T>,
  options: TimedOperationOptions
): Promise<TimedOperationResult<T>> {
  const { timeoutMs, token, now = Date.now } = options;
  const start = now();

  if (token?.isCanceled) {
    return {
      kind: 'canceled',
      durationMs: 0,
      cancellationReason: token.reason,
    };
  }

  let timeoutHandle: ReturnType<typeof setTimeout> | undefined;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutHandle = setTimeout(() => {
      const error: TimeoutError = Object.assign(new Error('Timed operation exceeded timeoutMs'), {
        isTimeout: true as const,
      });
      reject(error);
    }, timeoutMs);
  });

  try {
    const result = await Promise.race([operation(), timeoutPromise]);
    const durationMs = now() - start;
    return {
      kind: 'ok',
      durationMs,
      value: result as T,
    };
  } catch (error) {
    const durationMs = now() - start;

    // Timeout: marked by our internal isTimeout flag.
    if (isTimeoutError(error)) {
      return {
        kind: 'timeout',
        durationMs,
      };
    }

    // Cooperative cancellation: CancellationToken.throwIfCanceled attaches
    // a cancellationReason property to the thrown error.
    if (isCancellationError(error)) {
      return {
        kind: 'canceled',
        durationMs,
        cancellationReason: error.cancellationReason,
      };
    }

    // For all other errors, rethrow and let callers handle domain-specific
    // failures (for example network errors, rules violations).
    throw error;
  } finally {
    if (timeoutHandle !== undefined) {
      clearTimeout(timeoutHandle);
    }
  }
}
