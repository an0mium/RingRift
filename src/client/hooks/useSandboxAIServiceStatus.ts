/**
 * @fileoverview useSandboxAIServiceStatus Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for AI service availability tracking.
 * It monitors when the AI service is unavailable and the sandbox falls back to local heuristics.
 *
 * Canonical SSoT:
 * - AI turn logic: `src/client/sandbox/sandboxAI.ts`
 * - Service availability: `src/client/utils/aiServiceAvailability.ts`
 * - Diagnostics: `src/client/sandbox/sandboxAiDiagnostics.ts`
 *
 * This adapter:
 * - Tracks whether AI service is available or in fallback mode
 * - Polls sandbox AI diagnostics to detect service failures
 * - Provides user-friendly status messaging
 * - Supports retry functionality when service becomes available
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { getSandboxAIServiceAvailable } from '../utils/aiServiceAvailability';
import { getSandboxAiDiagnostics } from '../sandbox/sandboxAiDiagnostics';

/**
 * AI service connection status.
 */
export type AIServiceStatus =
  | 'connected' // AI service is available and responding
  | 'unavailable' // AI service is not configured for this environment
  | 'fallback' // AI service was available but failed, using local heuristics
  | 'unknown'; // Status not yet determined

/**
 * State for AI service status tracking.
 */
export interface AIServiceStatusState {
  /** Current AI service connection status */
  status: AIServiceStatus;
  /** User-friendly message describing the current status */
  message: string | null;
  /** Number of consecutive service failures */
  failureCount: number;
  /** Timestamp of last successful service call */
  lastSuccessAt: number | null;
  /** Timestamp of last failure */
  lastFailureAt: number | null;
  /** Whether the service is configured for this environment */
  isServiceConfigured: boolean;
}

/**
 * Actions for AI service status management.
 */
export interface AIServiceStatusActions {
  /** Record a successful AI service call */
  recordSuccess: () => void;
  /** Record a failed AI service call */
  recordFailure: (reason?: string) => void;
  /** Attempt to reconnect to AI service */
  retryConnection: () => Promise<boolean>;
  /** Dismiss the status message (but keep tracking) */
  dismissMessage: () => void;
}

/**
 * Return type for useSandboxAIServiceStatus hook.
 */
export interface UseSandboxAIServiceStatusReturn {
  /** Current AI service status state */
  state: AIServiceStatusState;
  /** Available actions */
  actions: AIServiceStatusActions;
}

/**
 * Custom hook for tracking AI service availability and fallback status in sandbox mode.
 *
 * This hook monitors the AI service connection status and provides:
 * - Real-time status updates when service fails or recovers
 * - User-friendly messages explaining the current state
 * - Retry functionality for reconnecting to the service
 *
 * @returns AI service status state and action functions
 *
 * @example
 * ```tsx
 * const { state, actions } = useSandboxAIServiceStatus();
 *
 * // Show banner when in fallback mode
 * {state.status === 'fallback' && (
 *   <StatusBanner variant="warning">
 *     {state.message}
 *     <Button onClick={actions.retryConnection}>Retry</Button>
 *   </StatusBanner>
 * )}
 * ```
 */
export function useSandboxAIServiceStatus(): UseSandboxAIServiceStatusReturn {
  const isServiceConfigured = getSandboxAIServiceAvailable();

  const [status, setStatus] = useState<AIServiceStatus>(() => {
    if (!isServiceConfigured) {
      return 'unavailable';
    }
    return 'unknown';
  });

  const [message, setMessage] = useState<string | null>(null);
  const [failureCount, setFailureCount] = useState(0);
  const [lastSuccessAt, setLastSuccessAt] = useState<number | null>(null);
  const [lastFailureAt, setLastFailureAt] = useState<number | null>(null);

  // Update status when configuration changes (e.g., environment)
  useEffect(() => {
    if (!isServiceConfigured && status !== 'unavailable') {
      setStatus('unavailable');
      setMessage(null);
    }
  }, [isServiceConfigured, status]);

  // Track the last seen diagnostic timestamps to detect changes
  const lastDiagnosticTimestampRef = useRef<Record<number, number>>({});

  // Poll sandbox AI diagnostics to detect service failures
  // This picks up on failures reported by sandboxAI.ts
  useEffect(() => {
    if (!isServiceConfigured) {
      return;
    }

    const pollInterval = window.setInterval(() => {
      const diagnostics = getSandboxAiDiagnostics();

      for (const [playerNumStr, entry] of Object.entries(diagnostics)) {
        const playerNum = parseInt(playerNumStr, 10);
        const lastTimestamp = lastDiagnosticTimestampRef.current[playerNum] || 0;

        // Only process new entries
        if (entry.timestamp > lastTimestamp) {
          lastDiagnosticTimestampRef.current[playerNum] = entry.timestamp;

          // Update status based on decision source
          if (entry.source === 'service') {
            // Service call succeeded - update status
            setStatus('connected');
            setFailureCount(0);
            setLastSuccessAt(entry.timestamp);
            if (message !== null) {
              setMessage(null);
            }
          } else if (entry.source === 'unavailable' || entry.source === 'local') {
            // Service was unavailable - record failure
            setFailureCount((prev) => prev + 1);
            setLastFailureAt(entry.timestamp);

            if (status === 'connected' || status === 'unknown') {
              setStatus('fallback');
              const errorDetail = entry.error ? ` (${entry.error})` : '';
              setMessage(
                `AI service unavailable. Using local heuristics for AI moves.${errorDetail}`
              );
            }
          }
        }
      }
    }, 500); // Poll every 500ms

    return () => {
      window.clearInterval(pollInterval);
    };
  }, [isServiceConfigured, status, message]);

  const recordSuccess = useCallback(() => {
    setStatus('connected');
    setFailureCount(0);
    setLastSuccessAt(Date.now());

    // Clear failure message after recovery
    if (message !== null) {
      setMessage(null);
    }
  }, [message]);

  const recordFailure = useCallback(
    (reason?: string) => {
      const newCount = failureCount + 1;
      setFailureCount(newCount);
      setLastFailureAt(Date.now());

      // Only show fallback message after first failure if we were previously connected
      if (status === 'connected' || status === 'unknown') {
        setStatus('fallback');
        const baseMessage = 'AI service unavailable. Using local heuristics for AI moves.';
        setMessage(reason ? `${baseMessage} (${reason})` : baseMessage);
      }
    },
    [failureCount, status]
  );

  const retryConnection = useCallback(async (): Promise<boolean> => {
    if (!isServiceConfigured) {
      return false;
    }

    try {
      // Quick health check to the AI service
      const response = await fetch('/api/games/sandbox/ai/health', {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });

      if (response.ok) {
        recordSuccess();
        return true;
      }
    } catch {
      // Ignore - will return false
    }

    recordFailure('Health check failed');
    return false;
  }, [isServiceConfigured, recordSuccess, recordFailure]);

  const dismissMessage = useCallback(() => {
    setMessage(null);
  }, []);

  return {
    state: {
      status,
      message,
      failureCount,
      lastSuccessAt,
      lastFailureAt,
      isServiceConfigured,
    },
    actions: {
      recordSuccess,
      recordFailure,
      retryConnection,
      dismissMessage,
    },
  };
}

export default useSandboxAIServiceStatus;
