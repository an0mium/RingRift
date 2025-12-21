/**
 * ---------------------------------------------------------------------------
 * Engine Errors Module
 * ---------------------------------------------------------------------------
 *
 * Exports custom error types for the shared engine.
 */

export {
  CanonicalRecordError,
  createPhaseMismatchError,
  createPlayerMismatchError,
  createMissingBookkeepingError,
  createLegacyCoercionError,
  isCanonicalRecordError,
  type PhaseValidationError,
  type PhaseValidationErrorType,
} from './CanonicalRecordError';
