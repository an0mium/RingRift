/**
 * ---------------------------------------------------------------------------
 * Canonical Record Error
 * ---------------------------------------------------------------------------
 *
 * Custom error class for canonical phase/move validation failures.
 * Provides detailed error information for debugging non-canonical records.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */

import type { GamePhase, MoveType } from '../../types/game';

/**
 * Phase validation error types.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */
export type PhaseValidationErrorType =
  | 'PHASE_MISMATCH'
  | 'PLAYER_MISMATCH'
  | 'MISSING_BOOKKEEPING'
  | 'INVALID_PHASE_TRANSITION'
  | 'LEGACY_COERCION_DETECTED';

/**
 * Detailed validation error information for canonical record violations.
 *
 * Contains all context needed for debugging:
 * - Expected vs actual phase
 * - Move type that caused the mismatch
 * - Game ID and move number for tracking
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */
export interface PhaseValidationError {
  /** The type of validation error */
  type: PhaseValidationErrorType;
  /** The phase the engine expected based on state */
  expectedPhase: GamePhase;
  /** The phase implied by the incoming move or actual state */
  actualPhase: GamePhase;
  /** The move type that caused the mismatch */
  moveType: MoveType;
  /** Optional game ID for debugging */
  gameId?: string;
  /** Optional move number in the game history */
  moveNumber?: number;
  /** Expected player number */
  expectedPlayer?: number;
  /** Actual player number from the move */
  actualPlayer?: number;
  /** Human-readable error message */
  message: string;
}

/**
 * Error thrown when a move violates canonical phase/move recording rules.
 *
 * This error is thrown in strict mode (default) when:
 * - A move type doesn't match the current phase
 * - A player mismatch is detected
 * - Required bookkeeping moves are missing
 *
 * Use this error to:
 * - Fail fast on non-canonical records in training pipelines
 * - Surface detailed debugging information for parity issues
 * - Quarantine legacy records that need migration
 *
 * @example
 * ```typescript
 * throw new CanonicalRecordError({
 *   type: 'PHASE_MISMATCH',
 *   expectedPhase: 'ring_placement',
 *   actualPhase: 'territory_processing',
 *   moveType: 'place_ring',
 *   gameId: 'abc-123',
 *   moveNumber: 42,
 *   message: 'Cannot apply place_ring in territory_processing phase'
 * });
 * ```
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */
export class CanonicalRecordError extends Error {
  /** Structured validation error details */
  public readonly validationError: PhaseValidationError;

  constructor(error: PhaseValidationError) {
    const fullMessage = formatErrorMessage(error);
    super(fullMessage);
    this.name = 'CanonicalRecordError';
    this.validationError = error;

    // Maintain proper stack trace in V8 environments
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, CanonicalRecordError);
    }
  }

  /**
   * Get the error type for programmatic handling.
   */
  get type(): PhaseValidationErrorType {
    return this.validationError.type;
  }

  /**
   * Get the expected phase.
   */
  get expectedPhase(): GamePhase {
    return this.validationError.expectedPhase;
  }

  /**
   * Get the actual phase.
   */
  get actualPhase(): GamePhase {
    return this.validationError.actualPhase;
  }

  /**
   * Get the move type that caused the error.
   */
  get moveType(): MoveType {
    return this.validationError.moveType;
  }

  /**
   * Get the game ID if available.
   */
  get gameId(): string | undefined {
    return this.validationError.gameId;
  }

  /**
   * Get the move number if available.
   */
  get moveNumber(): number | undefined {
    return this.validationError.moveNumber;
  }

  /**
   * Convert to a plain object for JSON serialization.
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      validationError: this.validationError,
    };
  }
}

/**
 * Format a detailed error message from the validation error.
 */
function formatErrorMessage(error: PhaseValidationError): string {
  const parts: string[] = [error.message];

  // Add context if available
  if (error.gameId) {
    parts.push(`Game: ${error.gameId}`);
  }
  if (error.moveNumber !== undefined) {
    parts.push(`Move #${error.moveNumber}`);
  }

  // Add phase/player details
  parts.push(`Expected phase: ${error.expectedPhase}`);
  parts.push(`Actual phase: ${error.actualPhase}`);
  parts.push(`Move type: ${error.moveType}`);

  if (error.expectedPlayer !== undefined && error.actualPlayer !== undefined) {
    parts.push(`Expected player: ${error.expectedPlayer}`);
    parts.push(`Actual player: ${error.actualPlayer}`);
  }

  // Add canonical rule reference
  parts.push('See RULES_CANONICAL_SPEC.md RR-CANON-R073 / RR-CANON-R075');

  return parts.join(' | ');
}

/**
 * Create a phase mismatch error.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 */
export function createPhaseMismatchError(params: {
  expectedPhase: GamePhase;
  actualPhase: GamePhase;
  moveType: MoveType;
  gameId?: string;
  moveNumber?: number;
  expectedPlayer?: number;
  actualPlayer?: number;
}): CanonicalRecordError {
  const error: PhaseValidationError = {
    type: 'PHASE_MISMATCH',
    expectedPhase: params.expectedPhase,
    actualPhase: params.actualPhase,
    moveType: params.moveType,
    message: `Cannot apply ${params.moveType} move in ${params.expectedPhase} phase`,
  };
  if (params.gameId !== undefined) error.gameId = params.gameId;
  if (params.moveNumber !== undefined) error.moveNumber = params.moveNumber;
  if (params.expectedPlayer !== undefined) error.expectedPlayer = params.expectedPlayer;
  if (params.actualPlayer !== undefined) error.actualPlayer = params.actualPlayer;
  return new CanonicalRecordError(error);
}

/**
 * Create a player mismatch error.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */
export function createPlayerMismatchError(params: {
  currentPhase: GamePhase;
  moveType: MoveType;
  expectedPlayer: number;
  actualPlayer: number;
  gameId?: string;
  moveNumber?: number;
}): CanonicalRecordError {
  const error: PhaseValidationError = {
    type: 'PLAYER_MISMATCH',
    expectedPhase: params.currentPhase,
    actualPhase: params.currentPhase,
    moveType: params.moveType,
    expectedPlayer: params.expectedPlayer,
    actualPlayer: params.actualPlayer,
    message: `Move ${params.moveType} from player ${params.actualPlayer} but expected player ${params.expectedPlayer}`,
  };
  if (params.gameId !== undefined) error.gameId = params.gameId;
  if (params.moveNumber !== undefined) error.moveNumber = params.moveNumber;
  return new CanonicalRecordError(error);
}

/**
 * Create a missing bookkeeping error (when phase transition is missing explicit move).
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */
export function createMissingBookkeepingError(params: {
  currentPhase: GamePhase;
  nextPhase: GamePhase;
  moveType: MoveType;
  gameId?: string;
  moveNumber?: number;
}): CanonicalRecordError {
  const error: PhaseValidationError = {
    type: 'MISSING_BOOKKEEPING',
    expectedPhase: params.currentPhase,
    actualPhase: params.nextPhase,
    moveType: params.moveType,
    message: `Transition from ${params.currentPhase} to ${params.nextPhase} requires explicit bookkeeping move, got ${params.moveType}`,
  };
  if (params.gameId !== undefined) error.gameId = params.gameId;
  if (params.moveNumber !== undefined) error.moveNumber = params.moveNumber;
  return new CanonicalRecordError(error);
}

/**
 * Create a legacy coercion detected error.
 *
 * This is thrown when replayCompatibility mode would have coerced the phase,
 * but strict mode is enabled (which is the default).
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */
export function createLegacyCoercionError(params: {
  currentPhase: GamePhase;
  wouldCoerceTo: GamePhase;
  moveType: MoveType;
  gameId?: string;
  moveNumber?: number;
  currentPlayer?: number;
  movePlayer?: number;
}): CanonicalRecordError {
  const error: PhaseValidationError = {
    type: 'LEGACY_COERCION_DETECTED',
    expectedPhase: params.currentPhase,
    actualPhase: params.wouldCoerceTo,
    moveType: params.moveType,
    message:
      `Non-canonical record: ${params.moveType} in ${params.currentPhase} would require ` +
      `coercion to ${params.wouldCoerceTo}. Use check_canonical_phase_history.py to validate ` +
      `and quarantine legacy records, or enable replayCompatibility for migration scripts.`,
  };
  if (params.gameId !== undefined) error.gameId = params.gameId;
  if (params.moveNumber !== undefined) error.moveNumber = params.moveNumber;
  if (params.currentPlayer !== undefined) error.expectedPlayer = params.currentPlayer;
  if (params.movePlayer !== undefined) error.actualPlayer = params.movePlayer;
  return new CanonicalRecordError(error);
}

/**
 * Type guard to check if an error is a CanonicalRecordError.
 */
export function isCanonicalRecordError(error: unknown): error is CanonicalRecordError {
  return error instanceof CanonicalRecordError;
}
