import type { RulesUxEventPayload, RulesUxEventType } from '../../shared/telemetry/rulesUxEvents';
import api from '../services/api';

/**
 * Lightweight, privacy-aware client helper for emitting rules-UX telemetry.
 *
 * - Uses the shared RulesUxEventPayload schema for client/server compatibility.
 * - Sends events via the existing axios API client to /api/telemetry/rules-ux.
 * - Swallows errors; telemetry must never affect UX flow.
 * - Applies optional sampling for high-frequency help-open events.
 */

function getEnv(): Record<string, string | undefined> {
  // Vite injects client env vars via a synthetic __VITE_ENV__ object on globalThis
  // (see errorReporting.ts for the same pattern).
  return ((globalThis as any).__VITE_ENV__ as Record<string, string | undefined> | undefined) ?? {};
}

function isTelemetryEnabled(): boolean {
  const env = getEnv();
  // Default to enabled; allow explicit opt-out via VITE_RULES_UX_TELEMETRY_ENABLED=false.
  const raw = env.VITE_RULES_UX_TELEMETRY_ENABLED;
  if (raw === 'false' || raw === '0') return false;
  return true;
}

/**
 * Parse the sampling rate for rules_help_open events from env.
 * Expected range: 0.0–1.0. Defaults to 1.0 (no sampling) when unset/invalid.
 */
function getHelpOpenSampleRate(): number {
  const env = getEnv();
  const raw = env.VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE;
  if (!raw) return 1;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return 1;
  if (parsed <= 0) return 0;
  if (parsed >= 1) return 1;
  return parsed;
}

/**
 * Simple deterministic string hash (32-bit, unsigned) for sampling.
 */
function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  // Convert to unsigned 32-bit
  return hash >>> 0;
}

/**
 * Decide whether to emit a given event based on type-specific sampling rules.
 * Currently only rules_help_open is sampled; all other events are always sent
 * (subject to the global enable flag).
 */
function shouldEmitSampled(event: RulesUxEventPayload): boolean {
  const { type } = event;

  if (type !== 'rules_help_open') {
    // Non-help-open events are expected to be relatively low volume.
    return true;
  }

  const rate = getHelpOpenSampleRate();
  if (rate >= 1) return true;
  if (rate <= 0) return false;

  const topic = event.topic ?? 'unknown';
  const key = [topic, event.boardType, String(event.numPlayers)].join('|');
  const hash = hashString(key);
  const normalized = hash / 0xffffffff; // 0.0–1.0
  return normalized < rate;
}

/**
 * Optionally log a warning in development when telemetry fails.
 */
function logDevWarning(message: string, error: unknown, extra?: Record<string, unknown>): void {
  const env = getEnv();
  if (env.MODE !== 'development') return;

  console.warn('[RulesUxTelemetry]', message, {
    error: error instanceof Error ? error.message : String(error),
    ...extra,
  });
}

/**
 * Send a single rules-UX telemetry event to the backend.
 *
 * - POSTs to /api/telemetry/rules-ux via the shared axios client, so that
 *   authentication and CSRF behaviour mirror other API calls.
 * - Returns a resolved Promise even when the underlying request fails.
 * - Applies type-specific sampling for rules_help_open events.
 */
export async function sendRulesUxEvent(event: RulesUxEventPayload): Promise<void> {
  if (!isTelemetryEnabled()) return;
  if (!shouldEmitSampled(event)) return;

  try {
    // Fire-and-forget; callers do not depend on telemetry success.
    await api.post('/telemetry/rules-ux', event);
  } catch (error) {
    logDevWarning('Failed to send rules UX telemetry event', error, {
      type: event.type as RulesUxEventType,
      boardType: event.boardType,
    });
  }
}
