/**
 * Tests for standardized error response format.
 *
 * Verifies that all API errors follow the standardized format:
 * {
 *   success: false,
 *   error: {
 *     code: string,     // Machine-readable error code
 *     message: string,  // Human-readable message
 *     details?: [...],  // Optional validation details
 *     requestId?: string, // Request correlation ID
 *     timestamp: string   // ISO 8601 timestamp
 *   }
 * }
 */

import { Request, Response, NextFunction } from 'express';
import { ZodError, z } from 'zod';
import {
  ApiError,
  ErrorCodes,
  CommonErrors,
  normalizeErrorCode,
  ErrorCodeToStatus,
  ErrorCodeMessages,
} from '../../src/server/errors';
import {
  errorHandler,
  createError,
  notFoundHandler,
  asyncHandler,
  AppError,
} from '../../src/server/middleware/errorHandler';
import {
  ValidationError,
  AuthenticationError,
  AuthorizationError,
} from '../../src/shared/validation/schemas';

// Mock config
jest.mock('../../src/server/config', () => ({
  config: {
    isDevelopment: false,
    isProduction: true,
  },
}));

// Mock logger
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    error: jest.fn(),
    warn: jest.fn(),
    info: jest.fn(),
  },
  withRequestContext: jest.fn((req, ctx) => ctx),
}));

describe('Standardized Error Response Format', () => {
  let mockRequest: Partial<Request> & { requestId?: string };
  let mockResponse: Partial<Response>;
  let mockNext: NextFunction;
  let jsonMock: jest.Mock;
  let statusMock: jest.Mock;

  beforeEach(() => {
    jsonMock = jest.fn();
    statusMock = jest.fn().mockReturnValue({ json: jsonMock });

    mockRequest = {
      url: '/api/test',
      method: 'GET',
      ip: '127.0.0.1',
      get: jest.fn().mockReturnValue('test-user-agent'),
      originalUrl: '/api/test',
      requestId: 'test-request-id-123',
    };

    mockResponse = {
      status: statusMock,
      json: jsonMock,
    };

    mockNext = jest.fn();
  });

  describe('ApiError class', () => {
    it('should create error with standardized code', () => {
      const error = new ApiError({ code: ErrorCodes.AUTH_INVALID_CREDENTIALS });

      expect(error.code).toBe(ErrorCodes.AUTH_INVALID_CREDENTIALS);
      expect(error.statusCode).toBe(401);
      expect(error.message).toBe('Invalid credentials');
      expect(error.isOperational).toBe(true);
    });

    it('should allow custom message', () => {
      const error = new ApiError({
        code: ErrorCodes.VALIDATION_FAILED,
        message: 'Custom validation message',
      });

      expect(error.message).toBe('Custom validation message');
    });

    it('should include validation details', () => {
      const details = [
        { field: 'email', message: 'Invalid email format' },
        { field: 'password', message: 'Password too short' },
      ];
      const error = new ApiError({
        code: ErrorCodes.VALIDATION_FAILED,
        details,
      });

      expect(error.details).toEqual(details);
    });

    it('should convert to response format with requestId', () => {
      const error = new ApiError({ code: ErrorCodes.AUTH_INVALID_CREDENTIALS });
      const response = error.toResponse('req-123');

      expect(response).toEqual({
        success: false,
        error: {
          code: 'AUTH_INVALID_CREDENTIALS',
          message: 'Invalid credentials',
          requestId: 'req-123',
          timestamp: expect.any(String),
        },
      });

      // Verify timestamp is valid ISO 8601
      expect(new Date(response.error.timestamp).toISOString()).toBe(response.error.timestamp);
    });

    it('should convert to response format without requestId', () => {
      const error = new ApiError({ code: ErrorCodes.AUTH_INVALID_CREDENTIALS });
      const response = error.toResponse();

      expect(response.error.requestId).toBeUndefined();
    });

    it('should normalize legacy error codes', () => {
      const error = new ApiError({ code: 'INVALID_CREDENTIALS' });

      expect(error.code).toBe(ErrorCodes.AUTH_INVALID_CREDENTIALS);
    });

    it('should wrap unknown errors via fromError', () => {
      const originalError = new Error('Something went wrong');
      const apiError = ApiError.fromError(originalError);

      expect(apiError.code).toBe(ErrorCodes.SERVER_INTERNAL_ERROR);
      expect(apiError.message).toBe('Something went wrong');
      expect(apiError.cause).toBe(originalError);
    });

    it('should pass through ApiError via fromError', () => {
      const original = new ApiError({ code: ErrorCodes.GAME_NOT_FOUND });
      const result = ApiError.fromError(original);

      expect(result).toBe(original);
    });

    it('should create from string error', () => {
      const error = ApiError.fromError('String error message');

      expect(error.code).toBe(ErrorCodes.SERVER_INTERNAL_ERROR);
      expect(error.message).toBe('String error message');
    });
  });

  describe('CommonErrors factory', () => {
    it('should create auth errors', () => {
      expect(CommonErrors.invalidCredentials().code).toBe(ErrorCodes.AUTH_INVALID_CREDENTIALS);
      expect(CommonErrors.tokenInvalid().code).toBe(ErrorCodes.AUTH_TOKEN_INVALID);
      expect(CommonErrors.tokenExpired().code).toBe(ErrorCodes.AUTH_TOKEN_EXPIRED);
      expect(CommonErrors.accountDeactivated().code).toBe(ErrorCodes.AUTH_ACCOUNT_DEACTIVATED);
      expect(CommonErrors.authRequired().code).toBe(ErrorCodes.AUTH_REQUIRED);
      expect(CommonErrors.forbidden().code).toBe(ErrorCodes.AUTH_FORBIDDEN);
    });

    it('should create validation errors with details', () => {
      const details = [{ field: 'email', message: 'Required' }];
      const error = CommonErrors.validationFailed(details);

      expect(error.code).toBe(ErrorCodes.VALIDATION_FAILED);
      expect(error.details).toEqual(details);
    });

    it('should create resource errors', () => {
      expect(CommonErrors.userNotFound().code).toBe(ErrorCodes.RESOURCE_USER_NOT_FOUND);
      expect(CommonErrors.gameNotFound().code).toBe(ErrorCodes.RESOURCE_GAME_NOT_FOUND);
      expect(CommonErrors.emailExists().code).toBe(ErrorCodes.RESOURCE_EMAIL_EXISTS);
      expect(CommonErrors.usernameExists().code).toBe(ErrorCodes.RESOURCE_USERNAME_EXISTS);
    });

    it('should create server errors', () => {
      expect(CommonErrors.databaseUnavailable().code).toBe(ErrorCodes.SERVER_DATABASE_UNAVAILABLE);

      const cause = new Error('DB connection failed');
      const internal = CommonErrors.internalError(cause);
      expect(internal.code).toBe(ErrorCodes.SERVER_INTERNAL_ERROR);
      expect(internal.cause).toBe(cause);
      expect(internal.isOperational).toBe(false);
    });

    it('should create game errors', () => {
      expect(CommonErrors.gameNotJoinable().code).toBe(ErrorCodes.GAME_NOT_JOINABLE);
      expect(CommonErrors.alreadyJoined().code).toBe(ErrorCodes.GAME_ALREADY_JOINED);
      expect(CommonErrors.gameFull().code).toBe(ErrorCodes.GAME_FULL);
      expect(CommonErrors.notYourTurn().code).toBe(ErrorCodes.GAME_NOT_YOUR_TURN);
    });
  });

  describe('Error code normalization', () => {
    it('should map legacy auth codes', () => {
      expect(normalizeErrorCode('INVALID_CREDENTIALS')).toBe(ErrorCodes.AUTH_INVALID_CREDENTIALS);
      expect(normalizeErrorCode('INVALID_TOKEN')).toBe(ErrorCodes.AUTH_TOKEN_INVALID);
      expect(normalizeErrorCode('TOKEN_EXPIRED')).toBe(ErrorCodes.AUTH_TOKEN_EXPIRED);
      expect(normalizeErrorCode('ACCOUNT_DEACTIVATED')).toBe(ErrorCodes.AUTH_ACCOUNT_DEACTIVATED);
    });

    it('should map legacy validation codes', () => {
      expect(normalizeErrorCode('INVALID_REQUEST')).toBe(ErrorCodes.VALIDATION_INVALID_REQUEST);
      expect(normalizeErrorCode('INVALID_ID')).toBe(ErrorCodes.VALIDATION_INVALID_ID);
    });

    it('should map legacy resource codes', () => {
      expect(normalizeErrorCode('NOT_FOUND')).toBe(ErrorCodes.RESOURCE_NOT_FOUND);
      expect(normalizeErrorCode('USER_NOT_FOUND')).toBe(ErrorCodes.RESOURCE_USER_NOT_FOUND);
      expect(normalizeErrorCode('EMAIL_EXISTS')).toBe(ErrorCodes.RESOURCE_EMAIL_EXISTS);
    });

    it('should pass through standardized codes', () => {
      expect(normalizeErrorCode(ErrorCodes.AUTH_INVALID_CREDENTIALS)).toBe(
        ErrorCodes.AUTH_INVALID_CREDENTIALS
      );
      expect(normalizeErrorCode(ErrorCodes.GAME_NOT_FOUND)).toBe(ErrorCodes.GAME_NOT_FOUND);
    });

    it('should default unknown codes to SERVER_INTERNAL_ERROR', () => {
      expect(normalizeErrorCode('UNKNOWN_CODE')).toBe(ErrorCodes.SERVER_INTERNAL_ERROR);
    });
  });

  describe('errorHandler middleware', () => {
    it('should handle ApiError and return standardized format', () => {
      const error = new ApiError({ code: ErrorCodes.AUTH_INVALID_CREDENTIALS });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(401);
      expect(jsonMock).toHaveBeenCalledWith({
        success: false,
        error: {
          code: 'AUTH_INVALID_CREDENTIALS',
          message: 'Invalid credentials',
          requestId: 'test-request-id-123',
          timestamp: expect.any(String),
        },
      });
    });

    it('should handle ZodError and include validation details', () => {
      const schema = z.object({
        email: z.string().email(),
        password: z.string().min(8),
      });

      let zodError: ZodError | null = null;
      try {
        schema.parse({ email: 'invalid', password: '123' });
      } catch (e) {
        zodError = e as ZodError;
      }

      errorHandler(zodError!, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(400);
      expect(jsonMock).toHaveBeenCalledWith({
        success: false,
        error: {
          code: 'VALIDATION_FAILED',
          message: expect.any(String),
          details: expect.arrayContaining([
            expect.objectContaining({
              field: expect.any(String),
              message: expect.any(String),
            }),
          ]),
          requestId: 'test-request-id-123',
          timestamp: expect.any(String),
        },
      });
    });

    it('should handle legacy AppError with code', () => {
      const error: AppError = new Error('Custom error');
      error.statusCode = 400;
      error.code = 'INVALID_REQUEST';
      error.isOperational = true;

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(400);
      expect(jsonMock).toHaveBeenCalledWith({
        success: false,
        error: {
          code: 'VALIDATION_INVALID_REQUEST',
          message: 'Custom error',
          requestId: 'test-request-id-123',
          timestamp: expect.any(String),
        },
      });
    });

    it('should handle ValidationError', () => {
      const error = new ValidationError('Validation failed');

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(400);
      expect(jsonMock).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'VALIDATION_FAILED',
          }),
        })
      );
    });

    it('should handle AuthenticationError', () => {
      const error = new AuthenticationError('Auth failed');

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(401);
      expect(jsonMock).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'AUTH_TOKEN_INVALID',
          }),
        })
      );
    });

    it('should handle AuthorizationError', () => {
      const error = new AuthorizationError('Access denied');

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(403);
      expect(jsonMock).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'AUTH_FORBIDDEN',
          }),
        })
      );
    });

    it('should handle JWT errors', () => {
      const error = new Error('jwt malformed');
      error.name = 'JsonWebTokenError';

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(401);
      expect(jsonMock).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'AUTH_TOKEN_INVALID',
          }),
        })
      );
    });

    it('should handle TokenExpiredError', () => {
      const error = new Error('jwt expired');
      error.name = 'TokenExpiredError';

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(401);
      expect(jsonMock).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'AUTH_TOKEN_EXPIRED',
          }),
        })
      );
    });

    it('should include requestId in response', () => {
      const error = new ApiError({ code: ErrorCodes.SERVER_INTERNAL_ERROR });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(jsonMock).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            requestId: 'test-request-id-123',
          }),
        })
      );
    });

    it('should handle missing requestId gracefully', () => {
      delete mockRequest.requestId;
      const error = new ApiError({ code: ErrorCodes.SERVER_INTERNAL_ERROR });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      const responseArg = jsonMock.mock.calls[0][0];
      expect(responseArg.error.requestId).toBeUndefined();
    });

    it('includes stack trace for 5xx errors in development mode only', () => {
      const { config } = require('../../src/server/config');
      config.isDevelopment = true;

      const error = new ApiError({ code: ErrorCodes.SERVER_INTERNAL_ERROR });
      error.stack = 'STACK_TRACE_EXAMPLE';

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      const responseArg = jsonMock.mock.calls[0][0] as any;
      expect(statusMock).toHaveBeenCalledWith(500);
      expect(responseArg.error.stack).toBeDefined();
      expect(typeof responseArg.error.stack).toBe('string');

      config.isDevelopment = false;
    });

    it('does not include stack trace for 4xx errors even in development mode', () => {
      const { config } = require('../../src/server/config');
      config.isDevelopment = true;

      const error = new ApiError({ code: ErrorCodes.AUTH_INVALID_CREDENTIALS });

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      const responseArg = jsonMock.mock.calls[0][0] as any;
      expect(statusMock).toHaveBeenCalledWith(401);
      expect(responseArg.error.stack).toBeUndefined();

      config.isDevelopment = false;
    });
  });

  describe('notFoundHandler', () => {
    it('should create standardized not found error', () => {
      notFoundHandler(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockNext).toHaveBeenCalledWith(
        expect.objectContaining({
          code: ErrorCodes.RESOURCE_ROUTE_NOT_FOUND,
          statusCode: 404,
        })
      );
    });
  });

  describe('createError backward compatibility', () => {
    it('should return ApiError with standardized code', () => {
      const error = createError('Test error', 400, 'INVALID_REQUEST');

      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).code).toBe('VALIDATION_INVALID_REQUEST');
    });

    it('should create legacy error when no code provided', () => {
      const error = createError('Test error', 500);

      expect(error.message).toBe('Test error');
      expect(error.statusCode).toBe(500);
      expect(error.isOperational).toBe(true);
    });
  });

  describe('asyncHandler', () => {
    it('should pass errors to next', async () => {
      const error = new Error('Async error');
      const handler = asyncHandler(async () => {
        throw error;
      });

      await handler(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockNext).toHaveBeenCalledWith(error);
    });

    it('should not call next on success', async () => {
      const handler = asyncHandler(async (_req, res) => {
        res.json({ success: true });
      });

      await handler(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockNext).not.toHaveBeenCalled();
    });
  });

  describe('MongoDB/Prisma error handling', () => {
    it('should handle CastError (invalid ID format)', () => {
      const error = new Error('Cast to ObjectId failed');
      error.name = 'CastError';

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(400);
      expect(jsonMock).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'VALIDATION_INVALID_ID',
          }),
        })
      );
    });

    // Note: MongoError with code 11000 branch is unreachable in current implementation
    // The explicit code property check runs first, converting 11000 to string
    // and normalizeErrorCode("11000") returns SERVER_INTERNAL_ERROR.
    // This is a known limitation - the branch at line 81 is dead code.
  });

  describe('Generic error fallback', () => {
    it('should handle generic Error without special properties', () => {
      const error = new Error('Unknown internal error');

      errorHandler(error, mockRequest as Request, mockResponse as Response, mockNext);

      expect(statusMock).toHaveBeenCalledWith(500);
      expect(jsonMock).toHaveBeenCalledWith(
        expect.objectContaining({
          success: false,
          error: expect.objectContaining({
            code: 'SERVER_INTERNAL_ERROR',
            message: 'Unknown internal error',
          }),
        })
      );
    });
  });

  describe('Error code consistency', () => {
    it('should have consistent status codes for all error codes', () => {
      // All error codes should have a defined status code
      Object.values(ErrorCodes).forEach((code) => {
        expect(ErrorCodeToStatus[code]).toBeDefined();
        expect(typeof ErrorCodeToStatus[code]).toBe('number');
        expect(ErrorCodeToStatus[code]).toBeGreaterThanOrEqual(400);
        expect(ErrorCodeToStatus[code]).toBeLessThan(600);
      });
    });

    it('should have consistent messages for all error codes', () => {
      // All error codes should have a defined message
      Object.values(ErrorCodes).forEach((code) => {
        expect(ErrorCodeMessages[code]).toBeDefined();
        expect(typeof ErrorCodeMessages[code]).toBe('string');
        expect(ErrorCodeMessages[code].length).toBeGreaterThan(0);
      });
    });

    it('should use proper HTTP status code ranges', () => {
      // 4xx for client errors
      expect(ErrorCodeToStatus[ErrorCodes.AUTH_INVALID_CREDENTIALS]).toBeGreaterThanOrEqual(400);
      expect(ErrorCodeToStatus[ErrorCodes.AUTH_INVALID_CREDENTIALS]).toBeLessThan(500);

      expect(ErrorCodeToStatus[ErrorCodes.VALIDATION_FAILED]).toBeGreaterThanOrEqual(400);
      expect(ErrorCodeToStatus[ErrorCodes.VALIDATION_FAILED]).toBeLessThan(500);

      expect(ErrorCodeToStatus[ErrorCodes.RESOURCE_NOT_FOUND]).toBe(404);

      // 5xx for server errors
      expect(ErrorCodeToStatus[ErrorCodes.SERVER_INTERNAL_ERROR]).toBeGreaterThanOrEqual(500);
      expect(ErrorCodeToStatus[ErrorCodes.SERVER_DATABASE_UNAVAILABLE]).toBeGreaterThanOrEqual(500);
    });
  });
});
