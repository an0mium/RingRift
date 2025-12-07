import type { DifficultyCalibrationEventPayload } from '../../shared/telemetry/difficultyCalibrationEvents';
import type { BoardType } from '../../shared/types/game';
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

/** Vite injects client env vars via a synthetic __VITE_ENV__ object on globalThis. */
interface ViteEnvWindow {
  __VITE_ENV__?: Record<string, string | undefined>;
}

function getEnv(): Record<string, string | undefined> {
  // Vite injects client env vars via a synthetic __VITE_ENV__ object on globalThis
  // (see rulesUxTelemetry.ts and errorReporting.ts for the same pattern).
  const viteGlobal = globalThis as unknown as ViteEnvWindow;
  return viteGlobal.__VITE_ENV__ ?? {};
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

// ──────────────────────────────────────────────────────────────────────────────
// Per-game calibration session helpers
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Lightweight client-side record of a calibration session for a specific game.
 * This is used to bridge between game creation (Lobby) and game completion
 * (BackendGameHost) so that we can emit a started + completed pair of events
 * without requiring the server to persist calibration flags yet.
 */
export interface DifficultyCalibrationSession {
  boardType: BoardType;
  numPlayers: number;
  difficulty: number;
  isCalibrationOptIn: boolean;
}

const SESSION_STORAGE_KEY_PREFIX = 'rr_difficulty_calibration_game:';

function getSessionStorage(): Storage | null {
  if (typeof window === 'undefined') return null;
  try {
    if (!('sessionStorage' in window)) return null;
    return window.sessionStorage;
  } catch {
    // Access to sessionStorage can throw in some sandboxed environments; fail soft.
    return null;
  }
}

/**
 * Persist a calibration session for the given gameId in sessionStorage.
 * This is best-effort and silently no-ops when storage is unavailable.
 */
export function storeDifficultyCalibrationSession(
  gameId: string,
  session: DifficultyCalibrationSession
): void {
  const storage = getSessionStorage();
  if (!storage) return;

  const key = `${SESSION_STORAGE_KEY_PREFIX}${gameId}`;
  try {
    storage.setItem(
      key,
      JSON.stringify({
        boardType: session.boardType,
        numPlayers: session.numPlayers,
        difficulty: session.difficulty,
        isCalibrationOptIn: !!session.isCalibrationOptIn,
      })
    );
  } catch {
    // Swallow storage errors; calibration is strictly best-effort.
  }
}

/**
 * Look up a previously stored calibration session for the given gameId.
 * Returns null when no valid session is present.
 */
export function getDifficultyCalibrationSession(
  gameId: string
): DifficultyCalibrationSession | null {
  const storage = getSessionStorage();
  if (!storage) return null;

  const key = `${SESSION_STORAGE_KEY_PREFIX}${gameId}`;
  const raw = storage.getItem(key);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Partial<DifficultyCalibrationSession> & {
      boardType?: string;
    };

    if (
      !parsed ||
      typeof parsed.boardType !== 'string' ||
      typeof parsed.numPlayers !== 'number' ||
      !Number.isFinite(parsed.numPlayers) ||
      parsed.numPlayers < 1 ||
      typeof parsed.difficulty !== 'number' ||
      !Number.isFinite(parsed.difficulty)
    ) {
      return null;
    }

    return {
      boardType: parsed.boardType as BoardType,
      numPlayers: parsed.numPlayers,
      difficulty: parsed.difficulty,
      isCalibrationOptIn: !!parsed.isCalibrationOptIn,
    };
  } catch {
    return null;
  }
}

/**
 * Clear any stored calibration session metadata for the given gameId.
 */
export function clearDifficultyCalibrationSession(gameId: string): void {
  const storage = getSessionStorage();
  if (!storage) return;
  const key = `${SESSION_STORAGE_KEY_PREFIX}${gameId}`;
  try {
    storage.removeItem(key);
  } catch {
    // Ignore removal failures.
  }
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
