/**
 * Centralized error code catalog for standardized API error responses.
 * All error codes follow the format: CATEGORY_SPECIFIC_ERROR
 *
 * Categories:
 * - AUTH: Authentication and authorization errors
 * - VALIDATION: Input validation errors
 * - RESOURCE: Resource-related errors (not found, already exists, etc.)
 * - SERVER: Internal server errors and service unavailability
 * - GAME: Game-specific errors
 * - RATE_LIMIT: Rate limiting errors
 */

export const ErrorCodes = {
  // ============================================================================
  // Authentication Errors (AUTH_*)
  // ============================================================================
  /** Invalid username/password combination */
  AUTH_INVALID_CREDENTIALS: 'AUTH_INVALID_CREDENTIALS',
  /** JWT token is malformed or signature invalid */
  AUTH_TOKEN_INVALID: 'AUTH_TOKEN_INVALID',
  /** JWT token has expired */
  AUTH_TOKEN_EXPIRED: 'AUTH_TOKEN_EXPIRED',
  /** Refresh token is invalid or not found */
  AUTH_REFRESH_TOKEN_INVALID: 'AUTH_REFRESH_TOKEN_INVALID',
  /** Refresh token has expired */
  AUTH_REFRESH_TOKEN_EXPIRED: 'AUTH_REFRESH_TOKEN_EXPIRED',
  /** Refresh token was reused (potential theft) */
  AUTH_REFRESH_TOKEN_REUSED: 'AUTH_REFRESH_TOKEN_REUSED',
  /** Refresh token is required but missing */
  AUTH_REFRESH_TOKEN_REQUIRED: 'AUTH_REFRESH_TOKEN_REQUIRED',
  /** User account has been deactivated */
  AUTH_ACCOUNT_DEACTIVATED: 'AUTH_ACCOUNT_DEACTIVATED',
  /** Authentication is required for this endpoint */
  AUTH_REQUIRED: 'AUTH_REQUIRED',
  /** Authorization token is required but missing */
  AUTH_TOKEN_REQUIRED: 'AUTH_TOKEN_REQUIRED',
  /** User does not have permission for this action */
  AUTH_FORBIDDEN: 'AUTH_FORBIDDEN',
  /** Too many failed login attempts */
  AUTH_LOGIN_LOCKED_OUT: 'AUTH_LOGIN_LOCKED_OUT',
  /** Email verification failed - invalid or expired token */
  AUTH_VERIFICATION_INVALID: 'AUTH_VERIFICATION_INVALID',
  /** Email verification token required */
  AUTH_VERIFICATION_TOKEN_REQUIRED: 'AUTH_VERIFICATION_TOKEN_REQUIRED',
  /** Password reset token invalid or expired */
  AUTH_RESET_TOKEN_INVALID: 'AUTH_RESET_TOKEN_INVALID',
  /** Password does not meet requirements */
  AUTH_WEAK_PASSWORD: 'AUTH_WEAK_PASSWORD',

  // ============================================================================
  // Validation Errors (VALIDATION_*)
  // ============================================================================
  /** Generic validation failure */
  VALIDATION_FAILED: 'VALIDATION_FAILED',
  /** Request body is malformed or missing required fields */
  VALIDATION_INVALID_REQUEST: 'VALIDATION_INVALID_REQUEST',
  /** Invalid format (e.g., email format, UUID format) */
  VALIDATION_INVALID_FORMAT: 'VALIDATION_INVALID_FORMAT',
  /** Invalid query parameters */
  VALIDATION_INVALID_QUERY_PARAMS: 'VALIDATION_INVALID_QUERY_PARAMS',
  /** Invalid ID format */
  VALIDATION_INVALID_ID: 'VALIDATION_INVALID_ID',
  /** Email address is required */
  VALIDATION_EMAIL_REQUIRED: 'VALIDATION_EMAIL_REQUIRED',
  /** Search query is required */
  VALIDATION_SEARCH_QUERY_REQUIRED: 'VALIDATION_SEARCH_QUERY_REQUIRED',
  /** Invalid profile data */
  VALIDATION_INVALID_PROFILE_DATA: 'VALIDATION_INVALID_PROFILE_DATA',
  /** Invalid AI configuration */
  VALIDATION_INVALID_AI_CONFIG: 'VALIDATION_INVALID_AI_CONFIG',
  /** Invalid difficulty level */
  VALIDATION_INVALID_DIFFICULTY: 'VALIDATION_INVALID_DIFFICULTY',

  // ============================================================================
  // Resource Errors (RESOURCE_*)
  // ============================================================================
  /** Requested resource was not found */
  RESOURCE_NOT_FOUND: 'RESOURCE_NOT_FOUND',
  /** Resource already exists (conflict) */
  RESOURCE_ALREADY_EXISTS: 'RESOURCE_ALREADY_EXISTS',
  /** User not found */
  RESOURCE_USER_NOT_FOUND: 'RESOURCE_USER_NOT_FOUND',
  /** Game not found */
  RESOURCE_GAME_NOT_FOUND: 'RESOURCE_GAME_NOT_FOUND',
  /** Email already registered */
  RESOURCE_EMAIL_EXISTS: 'RESOURCE_EMAIL_EXISTS',
  /** Username already taken */
  RESOURCE_USERNAME_EXISTS: 'RESOURCE_USERNAME_EXISTS',
  /** Route not found */
  RESOURCE_ROUTE_NOT_FOUND: 'RESOURCE_ROUTE_NOT_FOUND',
  /** Access denied to resource */
  RESOURCE_ACCESS_DENIED: 'RESOURCE_ACCESS_DENIED',

  // ============================================================================
  // Server Errors (SERVER_*)
  // ============================================================================
  /** Generic internal server error */
  SERVER_INTERNAL_ERROR: 'SERVER_INTERNAL_ERROR',
  /** Database is unavailable */
  SERVER_DATABASE_UNAVAILABLE: 'SERVER_DATABASE_UNAVAILABLE',
  /** External service is unavailable */
  SERVER_SERVICE_UNAVAILABLE: 'SERVER_SERVICE_UNAVAILABLE',
  /** Failed to send email */
  SERVER_EMAIL_SEND_FAILED: 'SERVER_EMAIL_SEND_FAILED',

  // ============================================================================
  // AI Service Errors (AI_*)
  // ============================================================================
  /** AI service timed out */
  AI_SERVICE_TIMEOUT: 'AI_SERVICE_TIMEOUT',
  /** AI service is unavailable */
  AI_SERVICE_UNAVAILABLE: 'AI_SERVICE_UNAVAILABLE',
  /** AI service returned an error */
  AI_SERVICE_ERROR: 'AI_SERVICE_ERROR',
  /** AI service is overloaded */
  AI_SERVICE_OVERLOADED: 'AI_SERVICE_OVERLOADED',

  // ============================================================================
  // Game Errors (GAME_*)
  // ============================================================================
  /** Game not found */
  GAME_NOT_FOUND: 'GAME_NOT_FOUND',
  /** Invalid game ID format */
  GAME_INVALID_ID: 'GAME_INVALID_ID',
  /** Game is not accepting new players */
  GAME_NOT_JOINABLE: 'GAME_NOT_JOINABLE',
  /** Player already joined this game */
  GAME_ALREADY_JOINED: 'GAME_ALREADY_JOINED',
  /** Game is full, no more players can join */
  GAME_FULL: 'GAME_FULL',
  /** Access denied to game */
  GAME_ACCESS_DENIED: 'GAME_ACCESS_DENIED',
  /** Invalid move attempt */
  GAME_INVALID_MOVE: 'GAME_INVALID_MOVE',
  /** Not the player's turn */
  GAME_NOT_YOUR_TURN: 'GAME_NOT_YOUR_TURN',
  /** Game has already ended */
  GAME_ALREADY_ENDED: 'GAME_ALREADY_ENDED',
  /** AI games cannot be rated */
  GAME_AI_UNRATED: 'GAME_AI_UNRATED',

  // ============================================================================
  // Rate Limiting Errors (RATE_LIMIT_*)
  // ============================================================================
  /** Generic rate limit exceeded */
  RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',
  /** Game creation rate limit exceeded */
  RATE_LIMIT_GAME_CREATE: 'RATE_LIMIT_GAME_CREATE',
  /** Auth endpoint rate limit exceeded */
  RATE_LIMIT_AUTH: 'RATE_LIMIT_AUTH',
} as const;

export type ErrorCode = (typeof ErrorCodes)[keyof typeof ErrorCodes];

/**
 * HTTP status codes for each error category.
 * Used by ApiError class for consistent status code mapping.
 */
export const ErrorCodeToStatus: Record<ErrorCode, number> = {
  // Auth errors
  [ErrorCodes.AUTH_INVALID_CREDENTIALS]: 401,
  [ErrorCodes.AUTH_TOKEN_INVALID]: 401,
  [ErrorCodes.AUTH_TOKEN_EXPIRED]: 401,
  [ErrorCodes.AUTH_REFRESH_TOKEN_INVALID]: 401,
  [ErrorCodes.AUTH_REFRESH_TOKEN_EXPIRED]: 401,
  [ErrorCodes.AUTH_REFRESH_TOKEN_REUSED]: 401,
  [ErrorCodes.AUTH_REFRESH_TOKEN_REQUIRED]: 400,
  [ErrorCodes.AUTH_ACCOUNT_DEACTIVATED]: 401,
  [ErrorCodes.AUTH_REQUIRED]: 401,
  [ErrorCodes.AUTH_TOKEN_REQUIRED]: 401,
  [ErrorCodes.AUTH_FORBIDDEN]: 403,
  [ErrorCodes.AUTH_LOGIN_LOCKED_OUT]: 429,
  [ErrorCodes.AUTH_VERIFICATION_INVALID]: 400,
  [ErrorCodes.AUTH_VERIFICATION_TOKEN_REQUIRED]: 400,
  [ErrorCodes.AUTH_RESET_TOKEN_INVALID]: 400,
  [ErrorCodes.AUTH_WEAK_PASSWORD]: 400,

  // Validation errors
  [ErrorCodes.VALIDATION_FAILED]: 400,
  [ErrorCodes.VALIDATION_INVALID_REQUEST]: 400,
  [ErrorCodes.VALIDATION_INVALID_FORMAT]: 400,
  [ErrorCodes.VALIDATION_INVALID_QUERY_PARAMS]: 400,
  [ErrorCodes.VALIDATION_INVALID_ID]: 400,
  [ErrorCodes.VALIDATION_EMAIL_REQUIRED]: 400,
  [ErrorCodes.VALIDATION_SEARCH_QUERY_REQUIRED]: 400,
  [ErrorCodes.VALIDATION_INVALID_PROFILE_DATA]: 400,
  [ErrorCodes.VALIDATION_INVALID_AI_CONFIG]: 400,
  [ErrorCodes.VALIDATION_INVALID_DIFFICULTY]: 400,

  // Resource errors
  [ErrorCodes.RESOURCE_NOT_FOUND]: 404,
  [ErrorCodes.RESOURCE_ALREADY_EXISTS]: 409,
  [ErrorCodes.RESOURCE_USER_NOT_FOUND]: 404,
  [ErrorCodes.RESOURCE_GAME_NOT_FOUND]: 404,
  [ErrorCodes.RESOURCE_EMAIL_EXISTS]: 409,
  [ErrorCodes.RESOURCE_USERNAME_EXISTS]: 409,
  [ErrorCodes.RESOURCE_ROUTE_NOT_FOUND]: 404,
  [ErrorCodes.RESOURCE_ACCESS_DENIED]: 403,

  // Server errors
  [ErrorCodes.SERVER_INTERNAL_ERROR]: 500,
  [ErrorCodes.SERVER_DATABASE_UNAVAILABLE]: 503,
  [ErrorCodes.SERVER_SERVICE_UNAVAILABLE]: 503,
  [ErrorCodes.SERVER_EMAIL_SEND_FAILED]: 500,

  // AI service errors
  [ErrorCodes.AI_SERVICE_TIMEOUT]: 503,
  [ErrorCodes.AI_SERVICE_UNAVAILABLE]: 503,
  [ErrorCodes.AI_SERVICE_ERROR]: 502,
  [ErrorCodes.AI_SERVICE_OVERLOADED]: 503,

  // Game errors
  [ErrorCodes.GAME_NOT_FOUND]: 404,
  [ErrorCodes.GAME_INVALID_ID]: 400,
  [ErrorCodes.GAME_NOT_JOINABLE]: 400,
  [ErrorCodes.GAME_ALREADY_JOINED]: 400,
  [ErrorCodes.GAME_FULL]: 400,
  [ErrorCodes.GAME_ACCESS_DENIED]: 403,
  [ErrorCodes.GAME_INVALID_MOVE]: 400,
  [ErrorCodes.GAME_NOT_YOUR_TURN]: 400,
  [ErrorCodes.GAME_ALREADY_ENDED]: 400,
  [ErrorCodes.GAME_AI_UNRATED]: 400,

  // Rate limit errors
  [ErrorCodes.RATE_LIMIT_EXCEEDED]: 429,
  [ErrorCodes.RATE_LIMIT_GAME_CREATE]: 429,
  [ErrorCodes.RATE_LIMIT_AUTH]: 429,
};

/**
 * Default human-readable messages for each error code.
 * These are safe to expose to clients.
 */
export const ErrorCodeMessages: Record<ErrorCode, string> = {
  // Auth errors
  [ErrorCodes.AUTH_INVALID_CREDENTIALS]: 'Invalid credentials',
  [ErrorCodes.AUTH_TOKEN_INVALID]: 'Invalid authentication token',
  [ErrorCodes.AUTH_TOKEN_EXPIRED]: 'Authentication token has expired',
  [ErrorCodes.AUTH_REFRESH_TOKEN_INVALID]: 'Invalid refresh token',
  [ErrorCodes.AUTH_REFRESH_TOKEN_EXPIRED]: 'Refresh token has expired',
  [ErrorCodes.AUTH_REFRESH_TOKEN_REUSED]:
    'Refresh token has been revoked due to suspicious activity',
  [ErrorCodes.AUTH_REFRESH_TOKEN_REQUIRED]: 'Refresh token required',
  [ErrorCodes.AUTH_ACCOUNT_DEACTIVATED]: 'Account is deactivated',
  [ErrorCodes.AUTH_REQUIRED]: 'Authentication required',
  [ErrorCodes.AUTH_TOKEN_REQUIRED]: 'Authorization token required',
  [ErrorCodes.AUTH_FORBIDDEN]: 'Access denied',
  [ErrorCodes.AUTH_LOGIN_LOCKED_OUT]: 'Too many failed login attempts. Please try again later.',
  [ErrorCodes.AUTH_VERIFICATION_INVALID]: 'Invalid or expired verification token',
  [ErrorCodes.AUTH_VERIFICATION_TOKEN_REQUIRED]: 'Verification token required',
  [ErrorCodes.AUTH_RESET_TOKEN_INVALID]: 'Invalid or expired password reset token',
  [ErrorCodes.AUTH_WEAK_PASSWORD]: 'Password must be at least 8 characters long',

  // Validation errors
  [ErrorCodes.VALIDATION_FAILED]: 'Validation failed',
  [ErrorCodes.VALIDATION_INVALID_REQUEST]: 'Invalid request',
  [ErrorCodes.VALIDATION_INVALID_FORMAT]: 'Invalid format',
  [ErrorCodes.VALIDATION_INVALID_QUERY_PARAMS]: 'Invalid query parameters',
  [ErrorCodes.VALIDATION_INVALID_ID]: 'Invalid ID format',
  [ErrorCodes.VALIDATION_EMAIL_REQUIRED]: 'Email required',
  [ErrorCodes.VALIDATION_SEARCH_QUERY_REQUIRED]: 'Search query required',
  [ErrorCodes.VALIDATION_INVALID_PROFILE_DATA]: 'Invalid profile data',
  [ErrorCodes.VALIDATION_INVALID_AI_CONFIG]: 'Must provide difficulty for each AI opponent',
  [ErrorCodes.VALIDATION_INVALID_DIFFICULTY]: 'AI difficulty must be between 1 and 10',

  // Resource errors
  [ErrorCodes.RESOURCE_NOT_FOUND]: 'Resource not found',
  [ErrorCodes.RESOURCE_ALREADY_EXISTS]: 'Resource already exists',
  [ErrorCodes.RESOURCE_USER_NOT_FOUND]: 'User not found',
  [ErrorCodes.RESOURCE_GAME_NOT_FOUND]: 'Game not found',
  [ErrorCodes.RESOURCE_EMAIL_EXISTS]: 'Email already registered',
  [ErrorCodes.RESOURCE_USERNAME_EXISTS]: 'Username already taken',
  [ErrorCodes.RESOURCE_ROUTE_NOT_FOUND]: 'Route not found',
  [ErrorCodes.RESOURCE_ACCESS_DENIED]: 'Access denied',

  // Server errors
  [ErrorCodes.SERVER_INTERNAL_ERROR]: 'Internal server error',
  [ErrorCodes.SERVER_DATABASE_UNAVAILABLE]: 'Database not available',
  [ErrorCodes.SERVER_SERVICE_UNAVAILABLE]: 'Service temporarily unavailable',
  [ErrorCodes.SERVER_EMAIL_SEND_FAILED]: 'Failed to send email',

  // AI service errors
  [ErrorCodes.AI_SERVICE_TIMEOUT]: 'AI service timed out',
  [ErrorCodes.AI_SERVICE_UNAVAILABLE]: 'AI service unavailable',
  [ErrorCodes.AI_SERVICE_ERROR]: 'AI service error',
  [ErrorCodes.AI_SERVICE_OVERLOADED]: 'AI service is overloaded',

  // Game errors
  [ErrorCodes.GAME_NOT_FOUND]: 'Game not found',
  [ErrorCodes.GAME_INVALID_ID]: 'Invalid game ID format',
  [ErrorCodes.GAME_NOT_JOINABLE]: 'Game is not accepting players',
  [ErrorCodes.GAME_ALREADY_JOINED]: 'Already joined this game',
  [ErrorCodes.GAME_FULL]: 'Game is full',
  [ErrorCodes.GAME_ACCESS_DENIED]: 'Access denied',
  [ErrorCodes.GAME_INVALID_MOVE]: 'Invalid move',
  [ErrorCodes.GAME_NOT_YOUR_TURN]: "Not your turn",
  [ErrorCodes.GAME_ALREADY_ENDED]: 'Game has already ended',
  [ErrorCodes.GAME_AI_UNRATED]: 'AI games cannot be rated',

  // Rate limit errors
  [ErrorCodes.RATE_LIMIT_EXCEEDED]: 'Rate limit exceeded. Please try again later.',
  [ErrorCodes.RATE_LIMIT_GAME_CREATE]:
    'Too many games created in a short period. Please try again later.',
  [ErrorCodes.RATE_LIMIT_AUTH]: 'Too many requests. Please try again later.',
};

/**
 * Legacy error code mapping for backward compatibility.
 * Maps old error codes to new standardized codes.
 */
export const LegacyCodeMapping: Record<string, ErrorCode> = {
  // Auth
  INVALID_CREDENTIALS: ErrorCodes.AUTH_INVALID_CREDENTIALS,
  INVALID_TOKEN: ErrorCodes.AUTH_TOKEN_INVALID,
  TOKEN_EXPIRED: ErrorCodes.AUTH_TOKEN_EXPIRED,
  INVALID_REFRESH_TOKEN: ErrorCodes.AUTH_REFRESH_TOKEN_INVALID,
  REFRESH_TOKEN_EXPIRED: ErrorCodes.AUTH_REFRESH_TOKEN_EXPIRED,
  REFRESH_TOKEN_REUSED: ErrorCodes.AUTH_REFRESH_TOKEN_REUSED,
  REFRESH_TOKEN_REQUIRED: ErrorCodes.AUTH_REFRESH_TOKEN_REQUIRED,
  ACCOUNT_DEACTIVATED: ErrorCodes.AUTH_ACCOUNT_DEACTIVATED,
  AUTH_REQUIRED: ErrorCodes.AUTH_REQUIRED,
  AUTHENTICATION_ERROR: ErrorCodes.AUTH_TOKEN_INVALID,
  AUTHORIZATION_ERROR: ErrorCodes.AUTH_FORBIDDEN,
  LOGIN_LOCKED_OUT: ErrorCodes.AUTH_LOGIN_LOCKED_OUT,
  TOKEN_REQUIRED: ErrorCodes.AUTH_TOKEN_REQUIRED,
  VERIFICATION_TOKEN_REQUIRED: ErrorCodes.AUTH_VERIFICATION_TOKEN_REQUIRED,
  WEAK_PASSWORD: ErrorCodes.AUTH_WEAK_PASSWORD,

  // Validation
  INVALID_REQUEST: ErrorCodes.VALIDATION_INVALID_REQUEST,
  INVALID_ID: ErrorCodes.VALIDATION_INVALID_ID,
  INVALID_QUERY_PARAMS: ErrorCodes.VALIDATION_INVALID_QUERY_PARAMS,
  EMAIL_REQUIRED: ErrorCodes.VALIDATION_EMAIL_REQUIRED,
  SEARCH_QUERY_REQUIRED: ErrorCodes.VALIDATION_SEARCH_QUERY_REQUIRED,
  INVALID_PROFILE_DATA: ErrorCodes.VALIDATION_INVALID_PROFILE_DATA,
  INVALID_AI_CONFIG: ErrorCodes.VALIDATION_INVALID_AI_CONFIG,
  INVALID_DIFFICULTY: ErrorCodes.VALIDATION_INVALID_DIFFICULTY,

  // Resource
  NOT_FOUND: ErrorCodes.RESOURCE_NOT_FOUND,
  USER_NOT_FOUND: ErrorCodes.RESOURCE_USER_NOT_FOUND,
  GAME_NOT_FOUND: ErrorCodes.RESOURCE_GAME_NOT_FOUND,
  EMAIL_EXISTS: ErrorCodes.RESOURCE_EMAIL_EXISTS,
  USERNAME_EXISTS: ErrorCodes.RESOURCE_USERNAME_EXISTS,
  ACCESS_DENIED: ErrorCodes.RESOURCE_ACCESS_DENIED,
  DUPLICATE_ENTRY: ErrorCodes.RESOURCE_ALREADY_EXISTS,

  // Server
  INTERNAL_ERROR: ErrorCodes.SERVER_INTERNAL_ERROR,
  DATABASE_UNAVAILABLE: ErrorCodes.SERVER_DATABASE_UNAVAILABLE,
  EMAIL_SEND_FAILED: ErrorCodes.SERVER_EMAIL_SEND_FAILED,

  // AI
  AI_SERVICE_TIMEOUT: ErrorCodes.AI_SERVICE_TIMEOUT,
  AI_SERVICE_UNAVAILABLE: ErrorCodes.AI_SERVICE_UNAVAILABLE,
  AI_SERVICE_ERROR: ErrorCodes.AI_SERVICE_ERROR,
  AI_SERVICE_OVERLOADED: ErrorCodes.AI_SERVICE_OVERLOADED,

  // Game
  INVALID_GAME_ID: ErrorCodes.GAME_INVALID_ID,
  GAME_NOT_JOINABLE: ErrorCodes.GAME_NOT_JOINABLE,
  ALREADY_JOINED: ErrorCodes.GAME_ALREADY_JOINED,
  GAME_FULL: ErrorCodes.GAME_FULL,
  AI_GAMES_UNRATED: ErrorCodes.GAME_AI_UNRATED,
  GAME_CREATE_RATE_LIMITED: ErrorCodes.RATE_LIMIT_GAME_CREATE,
};

/**
 * Normalize a legacy error code to a standardized error code.
 * Returns the original code if already standardized or not in mapping.
 */
export function normalizeErrorCode(code: string): ErrorCode {
  // Check if it's already a standardized code
  const standardizedCodes = Object.values(ErrorCodes) as string[];
  if (standardizedCodes.includes(code)) {
    return code as ErrorCode;
  }

  // Check legacy mapping
  if (code in LegacyCodeMapping) {
    return LegacyCodeMapping[code];
  }

  // Default to internal error for unknown codes
  return ErrorCodes.SERVER_INTERNAL_ERROR;
}