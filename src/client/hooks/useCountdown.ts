/**
 * useCountdown - Reusable countdown timer hook
 *
 * This hook consolidates countdown logic that was previously duplicated
 * across GameHUD, decision UI, and other components. It provides:
 * - Configurable tick interval
 * - Pause/resume support
 * - Server time reconciliation
 * - Formatted display values
 *
 * @module hooks/useCountdown
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Options for useCountdown.
 */
export interface UseCountdownOptions {
  /** Initial time in milliseconds */
  initialMs: number;
  /** Whether the countdown is active (ticking) */
  isActive?: boolean;
  /** Tick interval in milliseconds (default: 1000) */
  intervalMs?: number;
  /** Callback when countdown reaches zero */
  onExpire?: () => void;
  /** Callback on each tick with remaining time */
  onTick?: (remainingMs: number) => void;
  /** Minimum value (default: 0) */
  minMs?: number;
  /** Server-provided remaining time for reconciliation */
  serverRemainingMs?: number | null;
}

/**
 * Return type for useCountdown.
 */
export interface CountdownState {
  /** Remaining time in milliseconds */
  remainingMs: number;
  /** Remaining time in seconds (floored) */
  remainingSeconds: number;
  /** Whether countdown has expired (reached minMs) */
  isExpired: boolean;
  /** Whether countdown is currently active */
  isActive: boolean;
  /** Formatted time string (MM:SS or SS) */
  formattedTime: string;
  /** Reset to initial value */
  reset: (newInitialMs?: number) => void;
  /** Pause the countdown */
  pause: () => void;
  /** Resume the countdown */
  resume: () => void;
  /** Set remaining time manually */
  setRemainingMs: (ms: number) => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Reusable countdown timer hook.
 */
export function useCountdown(options: UseCountdownOptions): CountdownState {
  const {
    initialMs,
    isActive: externalIsActive = true,
    intervalMs = 1000,
    onExpire,
    onTick,
    minMs = 0,
    serverRemainingMs,
  } = options;

  const [remainingMs, setRemainingMs] = useState(Math.max(initialMs, minMs));
  const [internalPaused, setInternalPaused] = useState(false);
  const expiredRef = useRef(false);
  const lastServerReconcileRef = useRef<number | null>(null);

  // Effective active state (external control + internal pause)
  const isActive = externalIsActive && !internalPaused && remainingMs > minMs;

  // Reconcile with server time when provided
  useEffect(() => {
    if (serverRemainingMs === null || serverRemainingMs === undefined) {
      return;
    }

    // Only reconcile if server time is different from last reconciled value
    if (lastServerReconcileRef.current === serverRemainingMs) {
      return;
    }

    lastServerReconcileRef.current = serverRemainingMs;

    // Take the minimum of server and client time (never show more time than either)
    setRemainingMs((current) => Math.min(current, Math.max(serverRemainingMs, minMs)));
  }, [serverRemainingMs, minMs]);

  // Reset when initialMs changes
  useEffect(() => {
    setRemainingMs(Math.max(initialMs, minMs));
    expiredRef.current = false;
    lastServerReconcileRef.current = null;
  }, [initialMs, minMs]);

  // Main countdown effect
  useEffect(() => {
    if (!isActive) {
      return;
    }

    const interval = setInterval(() => {
      setRemainingMs((prev) => {
        const next = Math.max(prev - intervalMs, minMs);
        onTick?.(next);

        // Check for expiration
        if (next <= minMs && !expiredRef.current) {
          expiredRef.current = true;
          // Schedule onExpire outside the state update
          setTimeout(() => onExpire?.(), 0);
        }

        return next;
      });
    }, intervalMs);

    return () => clearInterval(interval);
  }, [isActive, intervalMs, minMs, onExpire, onTick]);

  // Derived values
  const remainingSeconds = Math.floor(remainingMs / 1000);
  const isExpired = remainingMs <= minMs;

  // Formatted time string
  const formattedTime = useMemo(() => {
    return formatTime(remainingMs);
  }, [remainingMs]);

  // Control functions
  const reset = useCallback(
    (newInitialMs?: number) => {
      const resetValue = newInitialMs ?? initialMs;
      setRemainingMs(Math.max(resetValue, minMs));
      expiredRef.current = false;
      setInternalPaused(false);
      lastServerReconcileRef.current = null;
    },
    [initialMs, minMs]
  );

  const pause = useCallback(() => {
    setInternalPaused(true);
  }, []);

  const resume = useCallback(() => {
    setInternalPaused(false);
  }, []);

  const setRemaining = useCallback(
    (ms: number) => {
      setRemainingMs(Math.max(ms, minMs));
    },
    [minMs]
  );

  return {
    remainingMs,
    remainingSeconds,
    isExpired,
    isActive,
    formattedTime,
    reset,
    pause,
    resume,
    setRemainingMs: setRemaining,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// FORMATTING UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format milliseconds to MM:SS or SS string.
 */
export function formatTime(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  if (minutes > 0) {
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  return seconds.toString();
}

/**
 * Format milliseconds to MM:SS.mmm string (with milliseconds).
 */
export function formatTimeWithMs(ms: number): string {
  const totalMs = Math.max(0, ms);
  const totalSeconds = Math.floor(totalMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const milliseconds = totalMs % 1000;

  if (minutes > 0) {
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
  }

  return `${seconds}.${milliseconds.toString().padStart(3, '0')}`;
}

/**
 * Format remaining time for display with appropriate precision.
 *
 * - Under 10 seconds: show tenths (9.5)
 * - Under 60 seconds: show whole seconds (45)
 * - Over 60 seconds: show MM:SS (2:30)
 */
export function formatTimeAdaptive(ms: number): string {
  const totalMs = Math.max(0, ms);
  const totalSeconds = totalMs / 1000;

  if (totalSeconds < 10) {
    return totalSeconds.toFixed(1);
  }

  if (totalSeconds < 60) {
    return Math.floor(totalSeconds).toString();
  }

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// ═══════════════════════════════════════════════════════════════════════════
// SPECIALIZED HOOKS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for decision timeout countdown.
 *
 * This is a specialized version of useCountdown for decision phases,
 * with warning threshold support.
 */
export interface UseDecisionTimerOptions {
  /** Deadline timestamp (ms since epoch) */
  deadlineMs: number | null;
  /** Warning threshold (ms before deadline to trigger warning) */
  warningThresholdMs?: number;
  /** Callback when warning threshold is crossed */
  onWarning?: () => void;
  /** Callback when deadline is reached */
  onExpire?: () => void;
  /** Server-provided remaining time for reconciliation */
  serverRemainingMs?: number | null;
}

export interface DecisionTimerState extends Omit<CountdownState, 'reset' | 'setRemainingMs'> {
  /** Whether warning threshold has been crossed */
  isWarning: boolean;
}

/**
 * Hook for decision timeout countdown with warning support.
 */
export function useDecisionTimer(options: UseDecisionTimerOptions): DecisionTimerState {
  const {
    deadlineMs,
    warningThresholdMs = 10000, // 10 seconds default
    onWarning,
    onExpire,
    serverRemainingMs,
  } = options;

  const [hasWarned, setHasWarned] = useState(false);

  // Calculate initial remaining time
  const initialMs = useMemo(() => {
    if (deadlineMs === null) {
      return 0;
    }
    return Math.max(0, deadlineMs - Date.now());
  }, [deadlineMs]);

  const countdown = useCountdown({
    initialMs,
    isActive: deadlineMs !== null && initialMs > 0,
    intervalMs: 100, // More frequent updates for precision
    onExpire,
    serverRemainingMs,
  });

  // Check for warning threshold
  const isWarning = countdown.remainingMs > 0 && countdown.remainingMs <= warningThresholdMs;

  // Trigger warning callback once
  useEffect(() => {
    if (isWarning && !hasWarned) {
      setHasWarned(true);
      onWarning?.();
    }
  }, [isWarning, hasWarned, onWarning]);

  // Reset warning state when deadline changes
  useEffect(() => {
    setHasWarned(false);
  }, [deadlineMs]);

  return {
    remainingMs: countdown.remainingMs,
    remainingSeconds: countdown.remainingSeconds,
    isExpired: countdown.isExpired,
    isActive: countdown.isActive,
    formattedTime: countdown.formattedTime,
    pause: countdown.pause,
    resume: countdown.resume,
    isWarning,
  };
}
