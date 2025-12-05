import type { DifficultyCalibrationEventPayload } from '../../shared/telemetry/difficultyCalibrationEvents';
import api from '../services/api';

/**
 * Lightweight, privacy-aware client helper for emitting AI difficulty
 * calibration telemetry.
 *
 * - Uses the shared DifficultyCalibrationEventPayload schema for client/server
 *   compatibility.
 * - Sends events via the existing axios API client to
 *   /api/telemetry/difficulty-calibration.
 * - Swallows errors; telemetry must never affect UX flow.
 */

function getEnv(): Record<string, string | undefined> {
  // Vite injects client env vars via a synthetic __VITE_ENV__ object on globalThis
  // (see rulesUxTelemetry.ts and errorReporting.ts for the same pattern).
  return ((globalThis as any).__VITE_ENV__ as Record<string, string | undefined> | undefined) ?? {};
}

function isTelemetryEnabled(): boolean {
  const env = getEnv();
  // Default to enabled; allow explicit opt-out via
  // VITE_DIFFICULTY_CALIBRATION_TELEMETRY_ENABLED=false.
  const raw = env.VITE_DIFFICULTY_CALIBRATION_TELEMETRY_ENABLED;
  if (raw === 'false' || raw === '0') return false;
  return true;
}

/**
 * Optionally log a warning in development when telemetry fails.
 */
function logDevWarning(message: string, error: unknown, extra?: Record<string, unknown>): void {
  const env = getEnv();
  if (env.MODE !== 'development') return;

  console.warn('[DifficultyCalibrationTelemetry]', message, {
    error: error instanceof Error ? error.message : String(error),
    ...extra,
  });
}

/**
 * Send a single difficulty-calibration telemetry event to the backend.
 *
 * - POSTs to /api/telemetry/difficulty-calibration via the shared axios
 *   client, so that authentication and CSRF behaviour mirror other API calls.
 * - Returns a resolved Promise even when the underlying request fails.
 * - Does not perform sampling; calibration events are expected to be low
 *   volume and explicitly opt-in.
 */
export async function sendDifficultyCalibrationEvent(
  event: DifficultyCalibrationEventPayload
): Promise<void> {
  if (!isTelemetryEnabled()) return;

  try {
    // Fire-and-forget; callers do not depend on telemetry success.
    await api.post('/telemetry/difficulty-calibration', event);
  } catch (error) {
    logDevWarning('Failed to send difficulty calibration telemetry event', error, {
      type: event.type,
      boardType: event.boardType,
      difficulty: event.difficulty,
      isCalibrationOptIn: event.isCalibrationOptIn,
    });
  }
}
