import type { BoardType } from '../types/game';

/**
 * Discriminant for lightweight rules UX telemetry events.
 *
 * These events are emitted by the client when players interact with
 * rules-heavy UX surfaces (teaching overlays, weird‑state banners, undo)
 * so that the backend can aggregate where rules understanding breaks down.
 *
 * NOTE: This schema is intentionally low‑cardinality and must not be
 * extended with user identifiers, raw board positions, or free‑text.
 */
export type RulesUxEventType =
  | 'rules_help_open'
  | 'rules_help_repeat'
  | 'rules_undo_churn'
  | 'rules_weird_state_resign'
  | 'rules_weird_state_help';

/**
 * Coarse classification of weird rules states as surfaced in the HUD.
 * Mirrors the values produced by getWeirdStateBanner on the client.
 */
export type RulesUxWeirdStateType =
  | 'active-no-moves-movement'
  | 'active-no-moves-line'
  | 'active-no-moves-territory'
  | 'forced-elimination'
  | 'structural-stalemate';

/**
 * Payload for a single rules‑UX telemetry event.
 *
 * All fields other than {@link type}, {@link boardType}, and
 * {@link numPlayers} are optional and may be omitted when they do not
 * apply to a particular event.
 */
export interface RulesUxEventPayload {
  /** Event discriminant. */
  type: RulesUxEventType;

  /** Coarse board topology; never includes full positions. */
  boardType: BoardType;

  /** Number of seats in the game (2, 3, or 4). */
  numPlayers: number;

  /**
   * AI difficulty when applicable (for example, sandbox vs AI or online
   * games with AI opponents). 1–10 scale, aligned with AI ladder docs.
   */
  aiDifficulty?: number;

  /** TeachingOverlay topic identifier (e.g. "active_no_moves"). */
  topic?: string;

  /** Curated scenario rulesConcept when applicable. */
  rulesConcept?: string;

  /** Curated scenario identifier when applicable. */
  scenarioId?: string;

  /** Coarse weird‑state classification when event is weird‑state related. */
  weirdStateType?: RulesUxWeirdStateType | string;

  /** Number of undos in the recent streak for undo‑churn events. */
  undoStreak?: number;

  /** Number of times a help topic has been opened during the current game. */
  repeatCount?: number;

  /**
   * Seconds between the first observation of the weird state and a
   * subsequent resign/abandonment event.
   */
  secondsSinceWeirdState?: number;
}

/**
 * Exhaustive list of supported rules‑UX telemetry event types, exposed
 * for runtime validation on the server.
 */
export const RULES_UX_EVENT_TYPES: readonly RulesUxEventType[] = [
  'rules_help_open',
  'rules_help_repeat',
  'rules_undo_churn',
  'rules_weird_state_resign',
  'rules_weird_state_help',
];

/**
 * Runtime guard for validating arbitrary input as a {@link RulesUxEventType}.
 */
export function isRulesUxEventType(value: unknown): value is RulesUxEventType {
  return typeof value === 'string' && (RULES_UX_EVENT_TYPES as readonly string[]).includes(value);
}
