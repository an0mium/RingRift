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

  if (type !== 'rules_help_open' && type !== 'help_open') {
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

let cachedSessionId: string | null = null;

/**
 * Best-effort classification of the current client platform.
 * Used to populate RulesUxEventPayload.clientPlatform.
 */
function getClientPlatform(): 'web' | 'mobile_web' | 'desktop' | string {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return 'desktop';
  }

  const ua = navigator.userAgent || '';
  const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(ua);
  return isMobile ? 'mobile_web' : 'web';
}

/**
 * Best-effort locale detection for telemetry enrichment.
 */
function getLocale(): string | undefined {
  if (typeof navigator !== 'undefined' && typeof navigator.language === 'string') {
    return navigator.language;
  }
  return undefined;
}

/**
 * Lazily generate a per-session identifier for correlating rules-UX events.
 * This is intentionally not tied to user identity and is not persisted
 * beyond the current runtime environment.
 */
function getSessionId(): string {
  if (cachedSessionId) return cachedSessionId;

  try {
    const anyCrypto = (globalThis as any).crypto as Crypto | undefined;
    if (anyCrypto && typeof anyCrypto.randomUUID === 'function') {
      cachedSessionId = anyCrypto.randomUUID();
      return cachedSessionId;
    }
  } catch {
    // Ignore and fall back to Math.random-based id.
  }

  cachedSessionId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  return cachedSessionId;
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

  const enriched: RulesUxEventPayload = {
    ...event,
    ts: event.ts ?? new Date().toISOString(),
    clientBuild:
      event.clientBuild ??
      getEnv().VITE_CLIENT_BUILD ??
      getEnv().VITE_GIT_SHA ??
      getEnv().VITE_APP_VERSION ??
      getEnv().MODE,
    clientPlatform: event.clientPlatform ?? getClientPlatform(),
    locale: event.locale ?? getLocale(),
    sessionId: event.sessionId ?? getSessionId(),
  };

  try {
    // Fire-and-forget; callers do not depend on telemetry success.
    await api.post('/telemetry/rules-ux', enriched);
  } catch (error) {
    logDevWarning('Failed to send rules UX telemetry event', error, {
      type: enriched.type as RulesUxEventType,
      boardType: enriched.boardType,
    });
  }
}

/**
 * High-level helper that enriches a RulesUxEventPayload with common
 * client metadata (timestamp, build, platform, locale, and a
 * per-session identifier) before sending it via {@link sendRulesUxEvent}.
 */
export async function logRulesUxEvent(event: RulesUxEventPayload): Promise<void> {
  await sendRulesUxEvent(event);
}

/**
 * Generate a fresh correlation id for a help session.
 *
 * This is separate from the per-session {@link getSessionId} and is intended
 * to be reused across a single help_open → help_topic_view → help_reopen
 * interaction as described in UX_RULES_TELEMETRY_SPEC.md.
 */
export function newHelpSessionId(): string {
  try {
    const anyCrypto = (globalThis as any).crypto as Crypto | undefined;
    if (anyCrypto && typeof anyCrypto.randomUUID === 'function') {
      return anyCrypto.randomUUID();
    }
  } catch {
    // Ignore and fall back to Math.random-based id.
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Generate a fresh correlation id for a weird-state overlay / banner session.
 *
 * Used to tie together weird_state_banner_impression, weird_state_overlay_shown,
 * weird_state_overlay_dismiss, and resign_after_weird_state events.
 */
export function newOverlaySessionId(): string {
  try {
    const anyCrypto = (globalThis as any).crypto as Crypto | undefined;
    if (anyCrypto && typeof anyCrypto.randomUUID === 'function') {
      return anyCrypto.randomUUID();
    }
  } catch {
    // Ignore and fall back to Math.random-based id.
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Generate a fresh correlation id for a multi-step teaching flow.
 *
 * Intended for teaching_step_started / teaching_step_completed and
 * sandbox_scenario_* events that are part of a single coherent flow.
 */
export function newTeachingFlowId(): string {
  try {
    const anyCrypto = (globalThis as any).crypto as Crypto | undefined;
    if (anyCrypto && typeof anyCrypto.randomUUID === 'function') {
      return anyCrypto.randomUUID();
    }
  } catch {
    // Ignore and fall back to Math.random-based id.
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Options for emitting a spec-aligned help_open event.
 *
 * This is a thin, typed wrapper over {@link sendRulesUxEvent} that fills
 * the envelope fields recommended in UX_RULES_TELEMETRY_SPEC.md §3.1.
 */
export interface HelpOpenEventOptions {
  boardType: RulesUxEventPayload['boardType'];
  numPlayers: RulesUxEventPayload['numPlayers'];
  aiDifficulty?: number;
  difficulty?: string;
  rulesContext?: RulesUxEventPayload['rulesContext'];
  rulesConcept?: RulesUxEventPayload['rulesConcept'];
  topic?: RulesUxEventPayload['topic'];
  scenarioId?: RulesUxEventPayload['scenarioId'];
  source: RulesUxEventPayload['source'];
  /**
   * Low-cardinality identifier for where help was opened from, e.g.:
   * - 'hud_help_chip'
   * - 'mobile_hud_help_chip'
   * - 'victory_modal_help_link'
   * - 'sandbox_toolbar_help'
   * - 'faq_button'
   */
  entrypoint: string;
  gameId?: string;
  isRanked?: boolean;
  isCalibrationGame?: boolean;
  isSandbox?: boolean;
  seatIndex?: number;
  perspectivePlayerCount?: number;
  /**
   * Optional pre-generated help_session_id. When omitted, a new one is
   * created via {@link newHelpSessionId}.
   */
  helpSessionId?: string;
}

/**
 * Emit a spec-aligned help_open event with the given options.
 *
 * This does not replace the legacy rules_help_open event; callers that
 * still need the legacy metrics can emit both. The payload here follows
 * the language-agnostic contract in UX_RULES_TELEMETRY_SPEC.md.
 */
export async function logHelpOpenEvent(options: HelpOpenEventOptions): Promise<void> {
  const {
    boardType,
    numPlayers,
    aiDifficulty,
    difficulty,
    rulesContext,
    rulesConcept,
    topic,
    scenarioId,
    source,
    entrypoint,
    gameId,
    isRanked,
    isCalibrationGame,
    isSandbox,
    seatIndex,
    perspectivePlayerCount,
    helpSessionId,
  } = options;

  const event: RulesUxEventPayload = {
    type: 'help_open',
    boardType,
    numPlayers,
    aiDifficulty,
    difficulty,
    rulesContext,
    rulesConcept,
    topic,
    scenarioId,
    source,
    gameId,
    isRanked,
    isCalibrationGame,
    isSandbox,
    seatIndex,
    perspectivePlayerCount,
    helpSessionId: helpSessionId ?? newHelpSessionId(),
    payload: {
      entrypoint,
    },
  };

  await sendRulesUxEvent(event);
}
