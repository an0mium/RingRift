import {
  ErrorCode,
  ErrorCodes,
  ErrorCodeToStatus,
  ErrorCodeMessages,
  normalizeErrorCode,
} from './errorCodes';

/**
 * Standardized API error response structure.
 * This interface defines the exact shape of error responses sent to clients.
 */
export interface ApiErrorResponse {
  success: false;
  error: {
    /** Machine-readable error code (e.g., "AUTH_INVALID_TOKEN") */
    code: ErrorCode;
    /** Human-readable error message safe to display to users */
    message: string;
    /** Optional field-level validation errors */
    details?: ValidationErrorDetail[] | Record<string, unknown>;
    /** Request correlation ID for tracing */
    requestId?: string;
    /** ISO 8601 timestamp when error occurred */
    timestamp: string;
  };
}

/**
 * Validation error detail for field-level errors.
 */
export interface ValidationErrorDetail {
  /** Field path (e.g., "email", "password", "body.items[0].quantity") */
  field: string;
  /** Error message for this field */
  message: string;
  /** Zod error code or custom code */
  code?: string;
}

/**
 * Options for creating an ApiError.
 */
export interface ApiErrorOptions {
  /** Machine-readable error code */
  code: ErrorCode | string;
  /** Human-readable message (optional, defaults to message from errorCodes) */
  message?: string;
  /** HTTP status code (optional, defaults to status from errorCodes) */
  statusCode?: number;
  /** Field-level validation errors */
  details?: ValidationErrorDetail[] | Record<string, unknown>;
  /** Original error for internal logging (never exposed to clients) */
  cause?: Error;
  /** Whether this is an operational error (expected) vs programming error */
  isOperational?: boolean;
}

/**
 * Custom error class for standardized API errors.
 *
 * This class provides a consistent interface for creating and handling
 * errors throughout the application. It automatically maps error codes
 * to HTTP status codes and default messages.
 *
 * @example
 * ```ts
 * // Simple usage with just an error code
 * throw new ApiError({ code: ErrorCodes.AUTH_INVALID_CREDENTIALS });
 *
 * // With custom message
 * throw new ApiError({
 *   code: ErrorCodes.VALIDATION_FAILED,
 *   message: 'Email must be a valid email address',
 *   details: [{ field: 'email', message: 'Invalid email format' }]
 * });
 *
 * // With original error for logging
 * try {
 *   await database.query();
 * } catch (err) {
 *   throw new ApiError({
 *     code: ErrorCodes.SERVER_DATABASE_UNAVAILABLE,
 *     cause: err as Error
 *   });
 * }
 * ```
 */
export class ApiError extends Error {
  /** Machine-readable error code */
  public readonly code: ErrorCode;

  /** HTTP status code */
  public readonly statusCode: number;

  /** Field-level validation errors */
  public readonly details: ValidationErrorDetail[] | Record<string, unknown> | undefined;

  /** Original error (for internal logging only) */
  public readonly cause: Error | undefined;

  /** Whether this is an operational (expected) error */
  public readonly isOperational: boolean;

  /** Timestamp when error was created */
  public readonly timestamp: Date;

  constructor(options: ApiErrorOptions) {
    // Normalize the error code (handles legacy codes)
    const normalizedCode = normalizeErrorCode(options.code);

    // Get default message if not provided
    const message = options.message || ErrorCodeMessages[normalizedCode];

    super(message);

    this.name = 'ApiError';
    this.code = normalizedCode;
    this.statusCode = options.statusCode || ErrorCodeToStatus[normalizedCode];
    this.details = options.details;
    this.cause = options.cause;
    this.isOperational = options.isOperational ?? true;
    this.timestamp = new Date();

    // Maintain proper stack trace in V8 environments
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Convert the error to a client-safe response format.
   * This method is used by the error handler middleware.
   *
   * @param requestId - Optional request correlation ID
   * @returns Standardized API error response
   */
  toResponse(requestId?: string): ApiErrorResponse {
    return {
      success: false,
      error: {
        code: this.code,
        message: this.message,
        ...(this.details && { details: this.details }),
        ...(requestId && { requestId }),
        timestamp: this.timestamp.toISOString(),
      },
    };
  }
  /**
   * Create a copy of this error with an updated request ID.
   * Useful when the request ID becomes available after error creation.
   */
  withRequestId(_requestId: string): ApiError {
    const opts: ApiErrorOptions = {
      code: this.code,
      message: this.message,
      statusCode: this.statusCode,
      isOperational: this.isOperational,
    };
    if (this.details !== undefined) {
      opts.details = this.details;
    }
    if (this.cause !== undefined) {
      opts.cause = this.cause;
    }
    const error = new ApiError(opts);
    // Override timestamp to preserve original
    (error as any).timestamp = this.timestamp;
    return error;
  }

  /**
   * Check if an error is an ApiError instance.
   */
  static isApiError(error: unknown): error is ApiError {
    return error instanceof ApiError;
  }

  /**
   * Create an ApiError from an unknown error.
   * Useful for wrapping errors from external sources.
   *
   * @param error - The original error
   * @param defaultCode - Default error code if not determinable
   * @returns ApiError instance
   */
  static fromError(
    error: unknown,
    defaultCode: ErrorCode = ErrorCodes.SERVER_INTERNAL_ERROR
  ): ApiError {
    // Already an ApiError, return as-is
    if (error instanceof ApiError) {
      return error;
    }

    // Error instance
    if (error instanceof Error) {
      // Check if it has our error properties (from createError legacy function)
      const anyError = error as any;
      if (anyError.code && anyError.statusCode) {
        return new ApiError({
          code: anyError.code,
          message: error.message,
          statusCode: anyError.statusCode,
          cause: error,
          isOperational: anyError.isOperational ?? true,
        });
      }

      return new ApiError({
        code: defaultCode,
        message: error.message,
        cause: error,
        isOperational: false,
      });
    }

    // String error
    if (typeof error === 'string') {
      return new ApiError({
        code: defaultCode,
        message: error,
      });
    }

    // Unknown error type
    return new ApiError({
      code: defaultCode,
      message: 'An unexpected error occurred',
      isOperational: false,
    });
  }
}

/**
 * Type guard for checking if a value is an Error-like object
 * with our custom error properties.
 */
export function isAppError(
  error: unknown
): error is Error & { statusCode?: number; code?: string; isOperational?: boolean } {
  return error instanceof Error;
}

/**
 * Export common error instances for convenience.
 * These can be thrown directly or used as templates.
 */
export const CommonErrors = {
  // Auth
  invalidCredentials: () => new ApiError({ code: ErrorCodes.AUTH_INVALID_CREDENTIALS }),

  tokenInvalid: () => new ApiError({ code: ErrorCodes.AUTH_TOKEN_INVALID }),

  tokenExpired: () => new ApiError({ code: ErrorCodes.AUTH_TOKEN_EXPIRED }),

  refreshTokenInvalid: () => new ApiError({ code: ErrorCodes.AUTH_REFRESH_TOKEN_INVALID }),

  refreshTokenExpired: () => new ApiError({ code: ErrorCodes.AUTH_REFRESH_TOKEN_EXPIRED }),

  refreshTokenReused: () => new ApiError({ code: ErrorCodes.AUTH_REFRESH_TOKEN_REUSED }),

  accountDeactivated: () => new ApiError({ code: ErrorCodes.AUTH_ACCOUNT_DEACTIVATED }),

  authRequired: () => new ApiError({ code: ErrorCodes.AUTH_REQUIRED }),

  forbidden: () => new ApiError({ code: ErrorCodes.AUTH_FORBIDDEN }),

  loginLockedOut: () => new ApiError({ code: ErrorCodes.AUTH_LOGIN_LOCKED_OUT }),

  // Validation
  validationFailed: (details?: ValidationErrorDetail[]) => {
    const opts: ApiErrorOptions = { code: ErrorCodes.VALIDATION_FAILED };
    if (details !== undefined) {
      opts.details = details;
    }
    return new ApiError(opts);
  },

  invalidRequest: (message?: string) => {
    const opts: ApiErrorOptions = { code: ErrorCodes.VALIDATION_INVALID_REQUEST };
    if (message !== undefined) {
      opts.message = message;
    }
    return new ApiError(opts);
  },

  invalidQueryParams: () => new ApiError({ code: ErrorCodes.VALIDATION_INVALID_QUERY_PARAMS }),

  // Resource
  userNotFound: () => new ApiError({ code: ErrorCodes.RESOURCE_USER_NOT_FOUND }),

  gameNotFound: () => new ApiError({ code: ErrorCodes.RESOURCE_GAME_NOT_FOUND }),

  emailExists: () => new ApiError({ code: ErrorCodes.RESOURCE_EMAIL_EXISTS }),

  usernameExists: () => new ApiError({ code: ErrorCodes.RESOURCE_USERNAME_EXISTS }),

  accessDenied: () => new ApiError({ code: ErrorCodes.RESOURCE_ACCESS_DENIED }),

  // Server
  internalError: (cause?: Error) => {
    const opts: ApiErrorOptions = {
      code: ErrorCodes.SERVER_INTERNAL_ERROR,
      isOperational: false,
    };
    if (cause !== undefined) {
      opts.cause = cause;
    }
    return new ApiError(opts);
  },

  databaseUnavailable: () => new ApiError({ code: ErrorCodes.SERVER_DATABASE_UNAVAILABLE }),

  // Game
  gameNotJoinable: () => new ApiError({ code: ErrorCodes.GAME_NOT_JOINABLE }),

  alreadyJoined: () => new ApiError({ code: ErrorCodes.GAME_ALREADY_JOINED }),

  gameFull: () => new ApiError({ code: ErrorCodes.GAME_FULL }),

  invalidMove: (message?: string) => {
    const opts: ApiErrorOptions = { code: ErrorCodes.GAME_INVALID_MOVE };
    if (message !== undefined) {
      opts.message = message;
    }
    return new ApiError(opts);
  },

  notYourTurn: () => new ApiError({ code: ErrorCodes.GAME_NOT_YOUR_TURN }),

  // Rate limiting
  rateLimitExceeded: () => new ApiError({ code: ErrorCodes.RATE_LIMIT_EXCEEDED }),

  gameCreateRateLimited: () => new ApiError({ code: ErrorCodes.RATE_LIMIT_GAME_CREATE }),
} as const;
