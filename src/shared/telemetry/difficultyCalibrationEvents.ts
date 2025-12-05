import type { BoardType } from '../types/game';

/**
 * Discriminant for lightweight AI difficulty calibration telemetry events.
 *
 * These events are emitted by the client for games where the user has
 * explicitly opted into helping calibrate AI difficulty tiers against
 * human perception. The schema is intentionally low-cardinality and must
 * not be extended with user identifiers, raw board state, or free-text.
 */
export type DifficultyCalibrationEventType =
  | 'difficulty_calibration_game_started'
  | 'difficulty_calibration_game_completed';

/**
 * Payload for a single difficulty-calibration telemetry event.
 *
 * All fields other than {@link type}, {@link boardType}, {@link numPlayers},
 * {@link difficulty}, and {@link isCalibrationOptIn} are optional and may be
 * omitted when they do not apply to a particular event.
 */
export interface DifficultyCalibrationEventPayload {
  /** Event discriminant. */
  type: DifficultyCalibrationEventType;

  /**
   * Coarse board topology; for current calibration experiments this is
   * expected to be 'square8' but the schema allows any BoardType so that
   * future multi-board calibration runs can reuse the same surface.
   */
  boardType: BoardType;

  /** Number of seats in the game (typically 2 for calibration runs). */
  numPlayers: number;

  /**
   * Primary AI difficulty knob (1-10), aligned with the AI ladder and
   * difficulty UX descriptors. For games with multiple AI opponents this
   * should reflect the primary calibration target (usually the strongest AI).
   */
  difficulty: number;

  /**
   * Whether the player explicitly opted into calibration mode when creating
   * the game. This allows analysis pipelines to distinguish organic games
   * from structured calibration experiments.
   */
  isCalibrationOptIn: boolean;

  // ──────────────────────────────────────────────────────────────────────────
  // Coarse outcome data (completed events only)
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Coarse human perspective result for the calibration game.
   *
   * - 'win' / 'loss' / 'draw' model the human's outcome against the AI
   * - 'abandoned' is used when the game did not reach a clean terminal
   *   result (e.g. resignation without clear score, disconnect, timeout)
   */
  result?: 'win' | 'loss' | 'draw' | 'abandoned';

  /** Number of moves played in the game, if easily available. */
  movesPlayed?: number;

  /**
   * Optional self-reported perceived difficulty on a coarse 1–5 scale:
   *   1 – Far too easy
   *   2 – A bit too easy
   *   3 – About right
   *   4 – A bit too hard
   *   5 – Far too hard
   *
   * The client is responsible for enforcing bounds before sending.
   */
  perceivedDifficulty?: number;
}

/**
 * Exhaustive list of supported difficulty calibration telemetry event types,
 * exposed for runtime validation on the server.
 */
export const DIFFICULTY_CALIBRATION_EVENT_TYPES: readonly DifficultyCalibrationEventType[] = [
  'difficulty_calibration_game_started',
  'difficulty_calibration_game_completed',
];

/**
 * Runtime guard for validating arbitrary input as a DifficultyCalibrationEventType.
 */
export function isDifficultyCalibrationEventType(
  value: unknown
): value is DifficultyCalibrationEventType {
  return (
    typeof value === 'string' &&
    (DIFFICULTY_CALIBRATION_EVENT_TYPES as readonly string[]).includes(value)
  );
}
