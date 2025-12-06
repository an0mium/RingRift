export type CountdownSeverity = 'normal' | 'warning' | 'critical';

/**
 * Classify a countdown value into a coarse-grained severity bucket for
 * consistent HUD / dialog styling.
 *
 * Thresholds (in milliseconds):
 *   - normal:   > 10_000
 *   - warning:  3_000 < t <= 10_000
 *   - critical: 0 <= t <= 3_000 (and any negative values)
 */
export function getCountdownSeverity(
  timeRemainingMs: number | null | undefined
): CountdownSeverity | null {
  if (typeof timeRemainingMs !== 'number' || Number.isNaN(timeRemainingMs)) {
    return null;
  }

  const ms = timeRemainingMs;

  if (ms > 10_000) {
    return 'normal';
  }

  if (ms > 3_000) {
    return 'warning';
  }

  // Treat zero or negative remaining time as critical: the UI should
  // still emphasise urgency/expiry even if the timer has just lapsed.
  return 'critical';
}

/**
 * Convert a millisecond countdown value to a whole-number value of seconds
 * suitable for display in the HUD / decision UI.
 *
 * Semantics:
 * - Nullish or non-finite input (NaN, ±Infinity) returns null.
 * - Negative values are clamped to 0.
 * - 0ms returns 0, representing "at or beyond the deadline" for display.
 * - Positive values use Math.ceil(ms / 1000) so we never show 0 while there
 *   is still time remaining (e.g. 1-1000ms -> 1s, 1001-2000ms -> 2s).
 */
export function msToDisplaySeconds(timeRemainingMs: number | null | undefined): number | null {
  if (typeof timeRemainingMs !== 'number' || !Number.isFinite(timeRemainingMs)) {
    return null;
  }

  const clamped = Math.max(0, timeRemainingMs);
  if (clamped === 0) {
    return 0;
  }

  return Math.ceil(clamped / 1000);
}
