/**
 * RequestLogger middleware branch coverage tests
 * Tests for src/server/middleware/requestLogger.ts
 */

import { Request, Response, NextFunction } from 'express';

// Mock logger module before import
const mockLoggerInfo = jest.fn();
const mockLoggerWarn = jest.fn();
const mockLoggerError = jest.fn();

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: (...args: unknown[]) => mockLoggerInfo(...args),
    warn: (...args: unknown[]) => mockLoggerWarn(...args),
    error: (...args: unknown[]) => mockLoggerError(...args),
  },
  getRequestContext: jest.fn(() => ({ requestId: 'test-req-id' })),
  maskHeaders: jest.fn((h) => h),
  maskSensitiveData: jest.fn((d) => d),
}));

// Mock config
jest.mock('../../../src/server/config', () => ({
  config: {
    isProduction: false,
  },
}));

import {
  getRequestContext,
  maskHeaders,
  maskSensitiveData,
} from '../../../src/server/utils/logger';
import { requestLogger, debugRequestLogger } from '../../../src/server/middleware/requestLogger';

describe('requestLogger middleware', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;
  let finishCallback: (() => void) | null = null;
  let errorCallback: ((err: Error) => void) | null = null;

  beforeEach(() => {
    jest.clearAllMocks();
    finishCallback = null;
    errorCallback = null;

    mockReq = {
      method: 'GET',
      originalUrl: '/api/games',
      path: '/api/games',
      ip: '127.0.0.1',
      get: jest.fn().mockReturnValue('test-user-agent'),
      headers: {},
      body: {},
      query: {},
      socket: { remoteAddress: '127.0.0.1' } as any,
    };

    mockRes = {
      statusCode: 200,
      on: jest.fn((event: string, callback: any) => {
        if (event === 'finish') {
          finishCallback = callback;
        }
        if (event === 'error') {
          errorCallback = callback;
        }
        return mockRes as Response;
      }),
      get: jest.fn().mockReturnValue(undefined),
      end: jest.fn().mockReturnThis() as any,
    };

    mockNext = jest.fn();
  });

  describe('path exclusion', () => {
    it('skips logging for /health path', () => {
      mockReq.path = '/health';

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockLoggerInfo).not.toHaveBeenCalled();
    });

    it('skips logging for /metrics path', () => {
      mockReq.path = '/metrics';

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockLoggerInfo).not.toHaveBeenCalled();
    });

    it('skips logging for /favicon.ico', () => {
      mockReq.path = '/favicon.ico';

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockLoggerInfo).not.toHaveBeenCalled();
    });

    it('skips paths matching custom excludePaths option', () => {
      mockReq.path = '/custom/path';

      const middleware = requestLogger({ excludePaths: ['/custom'] });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockLoggerInfo).not.toHaveBeenCalled();
    });
  });

  describe('debugRequestLogger preset', () => {
    it('logs headers and full response body with increased maxBodySize', () => {
      mockReq.path = '/api/debug';
      mockReq.headers = { 'x-debug': 'on' };
      mockRes.statusCode = 200;

      const middleware = debugRequestLogger;
      middleware(mockReq as Request, mockRes as Response, mockNext);

      // Simulate writing a response body larger than the default 1KB but
      // smaller than the debug maxBodySize (4KB). The debug logger should
      // log the full body without truncation.
      const largeBody = 'x'.repeat(1500);
      (mockRes as any).end(largeBody);

      if (finishCallback) finishCallback();

      // Incoming request should include headers because debugRequestLogger
      // enables logHeaders.
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({
          headers: expect.objectContaining({ 'x-debug': 'on' }),
        })
      );

      // Response log should contain the full body string (no [TRUNCATED] prefix).
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          body: largeBody,
        })
      );
    });

    it('truncates very large response bodies at the debug maxBodySize (4KB)', () => {
      mockReq.path = '/api/debug/truncate';
      mockRes.statusCode = 200;

      const middleware = debugRequestLogger;
      middleware(mockReq as Request, mockRes as Response, mockNext);

      // Simulate a response body larger than the debug maxBodySize (4096 bytes).
      const veryLargeBody = 'y'.repeat(5000);
      (mockRes as any).end(veryLargeBody);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          body: expect.stringMatching(/^\[TRUNCATED: 5000 bytes, showing first 4096\]/),
        })
      );
    });
  });

  describe('request logging', () => {
    it('logs incoming request for non-excluded paths', () => {
      mockReq.path = '/api/games';

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({
          type: 'http_request',
          method: 'GET',
          path: '/api/games',
        })
      );
    });

    it('includes query parameters when present', () => {
      mockReq.path = '/api/games';
      mockReq.query = { limit: '10', offset: '0' };

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({
          query: { limit: '10', offset: '0' },
        })
      );
    });

    it('logs request headers when logHeaders option is true', () => {
      mockReq.path = '/api/games';
      mockReq.headers = { 'x-custom': 'value' };

      const middleware = requestLogger({ logHeaders: true });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({
          headers: { 'x-custom': 'value' },
        })
      );
    });

    it('logs request body when logRequestBody option is true', () => {
      mockReq.path = '/api/games';
      mockReq.body = { name: 'test' };

      const middleware = requestLogger({ logRequestBody: true });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({
          body: { name: 'test' },
        })
      );
    });
  });

  describe('response logging', () => {
    it('logs successful 2xx response at info level', () => {
      mockRes.statusCode = 200;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      // Trigger finish callback
      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          type: 'http_response',
          statusCode: 200,
        })
      );
    });

    it('logs 3xx response at info level', () => {
      mockRes.statusCode = 302;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          statusCode: 302,
        })
      );
    });

    it('logs 4xx client error at warn level', () => {
      mockRes.statusCode = 404;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerWarn).toHaveBeenCalledWith(
        'Request completed with client error',
        expect.objectContaining({
          statusCode: 404,
        })
      );
    });

    it('logs 5xx server error at error level', () => {
      mockRes.statusCode = 500;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerError).toHaveBeenCalledWith(
        'Request completed with error',
        expect.objectContaining({
          statusCode: 500,
        })
      );
    });

    it('includes duration in response log', () => {
      mockRes.statusCode = 200;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          duration: expect.any(Number),
        })
      );
    });

    it('records larger durations for slow responses', () => {
      mockRes.statusCode = 200;

      const nowSpy = jest.spyOn(Date, 'now');
      nowSpy.mockReturnValueOnce(1_000_000).mockReturnValue(1_002_500);

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          duration: expect.any(Number),
        })
      );

      const [, responseLog] = (mockLoggerInfo as jest.Mock).mock.calls.find(
        ([message]) => message === 'Request completed'
      ) as [string, { duration: number }];

      expect(responseLog.duration).toBeGreaterThanOrEqual(2500);

      nowSpy.mockRestore();
    });

    it('includes content-length when available', () => {
      mockRes.statusCode = 200;
      (mockRes.get as jest.Mock).mockReturnValue('1234');

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          contentLength: 1234,
        })
      );
    });
  });

  describe('error handling', () => {
    it('logs response error events', () => {
      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      // Trigger error callback
      if (errorCallback) {
        errorCallback(new Error('Connection reset'));
      }

      expect(mockLoggerError).toHaveBeenCalledWith(
        'Request error',
        expect.objectContaining({
          type: 'http_error',
          error: 'Connection reset',
        })
      );
    });
  });

  describe('HTTP methods', () => {
    it('logs POST requests correctly', () => {
      mockReq.method = 'POST';
      mockRes.statusCode = 201;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('logs DELETE requests correctly', () => {
      mockReq.method = 'DELETE';
      mockRes.statusCode = 204;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          method: 'DELETE',
        })
      );
    });

    it('logs PUT requests correctly', () => {
      mockReq.method = 'PUT';
      mockRes.statusCode = 200;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          method: 'PUT',
        })
      );
    });
  });

  describe('middleware chain', () => {
    it('calls next() to continue middleware chain', () => {
      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalledTimes(1);
    });

    it('attaches finish event listener to response', () => {
      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.on).toHaveBeenCalledWith('finish', expect.any(Function));
    });

    it('attaches error event listener to response', () => {
      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.on).toHaveBeenCalledWith('error', expect.any(Function));
    });
  });

  describe('user context', () => {
    it('includes userId when user is authenticated', () => {
      (mockReq as any).user = { id: 'user-123' };
      mockRes.statusCode = 200;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          userId: 'user-123',
        })
      );
    });
  });

  describe('correlation IDs and masking', () => {
    it('uses requestId from request when present', () => {
      (mockReq as any).requestId = 'explicit-req-id';
      mockRes.statusCode = 200;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({ requestId: 'explicit-req-id' })
      );
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({ requestId: 'explicit-req-id' })
      );
    });

    it('falls back to request context requestId when not set on req', () => {
      // Ensure no explicit requestId on the request object.
      (mockReq as any).requestId = undefined;
      (getRequestContext as jest.Mock).mockReturnValueOnce({ requestId: 'context-id' });
      mockRes.statusCode = 200;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);
      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({ requestId: 'context-id' })
      );
    });

    it('uses "unknown" requestId when neither req nor context provide one', () => {
      (mockReq as any).requestId = undefined;
      (getRequestContext as jest.Mock).mockReturnValueOnce(undefined);
      mockRes.statusCode = 200;

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);
      if (finishCallback) finishCallback();

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({ requestId: 'unknown' })
      );
    });

    it('applies maskHeaders when logging headers', () => {
      mockReq.path = '/api/games';
      mockReq.headers = { authorization: 'secret-token', 'x-custom': 'value' };

      (maskHeaders as jest.Mock).mockImplementationOnce(() => ({
        authorization: '[REDACTED]',
        'x-custom': 'value',
      }));

      const middleware = requestLogger({ logHeaders: true });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(maskHeaders).toHaveBeenCalledWith(
        expect.objectContaining({ authorization: 'secret-token', 'x-custom': 'value' })
      );

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({
          headers: {
            authorization: '[REDACTED]',
            'x-custom': 'value',
          },
        })
      );
    });

    it('applies maskSensitiveData to request bodies before logging', () => {
      mockReq.path = '/api/games';
      mockReq.body = { password: 'super-secret' };

      (maskSensitiveData as jest.Mock).mockImplementationOnce(() => ({
        password: '***',
      }));

      const middleware = requestLogger({ logRequestBody: true, maxBodySize: 1024 });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(maskSensitiveData).toHaveBeenCalledWith({ password: 'super-secret' });

      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({
          body: { password: '***' },
        })
      );
    });

    it('applies maskSensitiveData to response bodies when enabled', () => {
      mockRes.statusCode = 200;

      (maskSensitiveData as jest.Mock).mockImplementationOnce(() => '{"ok":true}');

      const middleware = requestLogger({
        logResponseBody: true,
        maxBodySize: 1024,
      });

      middleware(mockReq as Request, mockRes as Response, mockNext);

      // res.end is overridden by middleware, call it to simulate a body write.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (mockRes as any).end('{"ok":true}');

      if (finishCallback) finishCallback();

      expect(maskSensitiveData).toHaveBeenCalledWith('{"ok":true}');
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          body: '{"ok":true}',
        })
      );
    });
  });

  describe('truncateBody branch coverage', () => {
    it('truncates object body via JSON.stringify when exceeds maxSize', () => {
      mockReq.path = '/api/games';
      // Create an object body that when stringified exceeds the small maxBodySize
      const largeObject = { data: 'x'.repeat(100), nested: { field: 'value' } };
      mockReq.body = largeObject;

      const middleware = requestLogger({ logRequestBody: true, maxBodySize: 50 });
      middleware(mockReq as Request, mockRes as Response, mockNext);

      // The body should be truncated and prefixed with [TRUNCATED: ...]
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Incoming request',
        expect.objectContaining({
          body: expect.stringMatching(/^\[TRUNCATED: \d+ bytes, showing first 50\]/),
        })
      );
    });

    it('truncates string response body when exceeds maxSize', () => {
      mockRes.statusCode = 200;
      // Large string body that exceeds the small maxBodySize
      const largeStringBody = 'a'.repeat(200);

      const middleware = requestLogger({
        logResponseBody: true,
        maxBodySize: 50,
      });

      middleware(mockReq as Request, mockRes as Response, mockNext);

      // Simulate response with a large string body
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (mockRes as any).end(largeStringBody);

      if (finishCallback) finishCallback();

      // The string body should be truncated and prefixed with [TRUNCATED: ...]
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          body: expect.stringMatching(/^\[TRUNCATED: 200 bytes, showing first 50\]/),
        })
      );
    });
  });

  describe('getContentLength branch coverage', () => {
    it('returns undefined when content-length header is not a valid number', () => {
      mockRes.statusCode = 200;
      // Return an invalid (non-numeric) content-length header
      (mockRes.get as jest.Mock).mockImplementation((header: string) => {
        if (header === 'content-length') return 'invalid-not-a-number';
        return undefined;
      });

      const middleware = requestLogger();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      if (finishCallback) finishCallback();

      // contentLength should be undefined due to NaN from parseInt
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          contentLength: undefined,
        })
      );
    });
  });

  describe('res.end binary data handling', () => {
    it('logs [BINARY_DATA] when chunk.toString fails for binary data', () => {
      mockRes.statusCode = 200;

      const middleware = requestLogger({
        logResponseBody: true,
        maxBodySize: 1024,
      });

      middleware(mockReq as Request, mockRes as Response, mockNext);

      // Create a mock Buffer-like object that throws when toString('utf-8') is called
      const binaryChunk = {
        toString: (encoding?: string) => {
          if (encoding === 'utf-8') {
            throw new Error('Cannot convert binary to UTF-8');
          }
          return '[object Buffer]';
        },
      };

      // Call the overridden res.end with the problematic binary chunk
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (mockRes as any).end(binaryChunk);

      if (finishCallback) finishCallback();

      // The catch block should set responseBody to '[BINARY_DATA]'
      expect(mockLoggerInfo).toHaveBeenCalledWith(
        'Request completed',
        expect.objectContaining({
          body: '[BINARY_DATA]',
        })
      );
    });
  });
});
