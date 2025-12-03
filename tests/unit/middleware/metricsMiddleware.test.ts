/**
 * Metrics Middleware Unit Tests
 *
 * Tests for the HTTP request metrics middleware including:
 * - Path skipping (health, metrics, socket.io)
 * - Content length detection from headers and body
 * - Response size tracking via write/end interception
 * - Duration calculation
 * - Safe middleware error handling
 */

import { Request, Response, NextFunction } from 'express';
import {
  metricsMiddleware,
  safeMetricsMiddleware,
} from '../../../src/server/middleware/metricsMiddleware';

const mockRecordHttpRequest = jest.fn();
const mockGetMetricsService = jest.fn(() => ({
  recordHttpRequest: mockRecordHttpRequest,
}));

jest.mock('../../../src/server/services/MetricsService', () => ({
  getMetricsService: () => mockGetMetricsService(),
}));

describe('metricsMiddleware', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: jest.Mock;
  let originalWrite: jest.Mock;
  let originalEnd: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();

    originalWrite = jest.fn().mockReturnValue(true);
    originalEnd = jest.fn().mockReturnThis();

    mockReq = {
      path: '/api/games',
      method: 'GET',
      headers: {},
      body: undefined,
    };

    mockRes = {
      statusCode: 200,
      write: originalWrite,
      end: originalEnd,
    };

    mockNext = jest.fn();
  });

  describe('path skipping', () => {
    it('should skip /health path and not record metrics', () => {
      mockReq.path = '/health';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockRecordHttpRequest).not.toHaveBeenCalled();
    });

    it('should skip /healthz path', () => {
      mockReq.path = '/healthz';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      // Interceptors not set up, so write/end are original
      expect(mockRes.write).toBe(originalWrite);
    });

    it('should skip /ready path', () => {
      mockReq.path = '/ready';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should skip /readyz path', () => {
      mockReq.path = '/readyz';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should skip /metrics path', () => {
      mockReq.path = '/metrics';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should skip paths starting with /socket.io', () => {
      mockReq.path = '/socket.io/polling';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockRes.write).toBe(originalWrite);
    });

    it('should skip exact /socket.io path', () => {
      mockReq.path = '/socket.io';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should NOT skip /api/health path (not in skip set)', () => {
      mockReq.path = '/api/health';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      // Interceptors should be set up (write/end replaced)
      expect(mockRes.write).not.toBe(originalWrite);
    });

    it('should NOT skip /api/metrics path', () => {
      mockReq.path = '/api/metrics';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.write).not.toBe(originalWrite);
    });
  });

  describe('content length from headers', () => {
    it('should use Content-Length header when present', () => {
      mockReq.headers = { 'content-length': '1234' };
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      // Trigger end to record metrics
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        1234,
        undefined
      );
    });

    it('should handle Content-Length as array', () => {
      mockReq.headers = { 'content-length': ['5678'] };
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        5678,
        undefined
      );
    });

    it('should handle empty Content-Length array', () => {
      mockReq.headers = { 'content-length': [] };
      mockReq.body = 'test body';
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      // Falls back to body estimation
      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        9, // Buffer.byteLength('test body')
        undefined
      );
    });

    it('should handle invalid Content-Length value', () => {
      mockReq.headers = { 'content-length': 'not-a-number' };
      mockReq.body = 'body';
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      // Falls back to body estimation
      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        4, // Buffer.byteLength('body')
        undefined
      );
    });
  });

  describe('content length from body', () => {
    it('should estimate size from string body', () => {
      mockReq.body = 'Hello World';
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        11, // 'Hello World'.length
        undefined
      );
    });

    it('should estimate size from Buffer body', () => {
      mockReq.body = Buffer.from('buffer data');
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        11, // 'buffer data'.length
        undefined
      );
    });

    it('should estimate size from object body (JSON)', () => {
      mockReq.body = { key: 'value', num: 123 };
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      const expectedSize = Buffer.byteLength(JSON.stringify({ key: 'value', num: 123 }));
      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        expectedSize,
        undefined
      );
    });

    it('should handle circular object body gracefully', () => {
      const circular: Record<string, unknown> = { key: 'value' };
      circular.self = circular;
      mockReq.body = circular;
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      // Falls back to undefined when JSON.stringify fails
      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined
      );
    });

    it('should handle undefined body', () => {
      mockReq.body = undefined;
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined
      );
    });

    it('should handle null body', () => {
      mockReq.body = null;
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      // null is typeof 'object', JSON.stringify returns 'null' (4 bytes)
      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        4,
        undefined
      );
    });

    it('should handle number body (primitive)', () => {
      mockReq.body = 42;
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      // Number is not string, buffer, or object - returns undefined
      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined
      );
    });

    it('should handle boolean body (primitive)', () => {
      mockReq.body = true;
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined
      );
    });
  });

  describe('response size tracking via write', () => {
    it('should track size from Buffer chunks via write', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      // Write some chunks
      (mockRes.write as jest.Mock)(Buffer.from('first'));
      (mockRes.write as jest.Mock)(Buffer.from('second'));
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        11 // 'first' + 'second'
      );
    });

    it('should track size from string chunks via write', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.write as jest.Mock)('hello');
      (mockRes.write as jest.Mock)(' world');
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        11 // 'hello world'
      );
    });

    it('should handle null chunk in write', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.write as jest.Mock)(null);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined // 0 size = undefined
      );
    });

    it('should handle undefined chunk in write', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.write as jest.Mock)(undefined);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined
      );
    });

    it('should use specified encoding for string chunks', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      // Unicode string with explicit encoding
      (mockRes.write as jest.Mock)('Ü', 'utf8');
      (mockRes.end as jest.Mock)();

      // Ü is 2 bytes in UTF-8
      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        2
      );
    });

    it('should call original write and return its result', () => {
      mockReq.path = '/api/data';
      originalWrite.mockReturnValue(false);

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      const result = (mockRes.write as jest.Mock)('data');

      expect(originalWrite).toHaveBeenCalled();
      expect(result).toBe(false);
    });
  });

  describe('response size tracking via end', () => {
    it('should track size from Buffer chunk in end', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.end as jest.Mock)(Buffer.from('final data'));

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        10 // 'final data'
      );
    });

    it('should track size from string chunk in end', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.end as jest.Mock)('end data');

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        8 // 'end data'
      );
    });

    it('should not count function callback as chunk', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      const callback = jest.fn();
      (mockRes.end as jest.Mock)(callback);

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined // function not counted
      );
    });

    it('should accumulate write and end sizes', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.write as jest.Mock)('prefix-');
      (mockRes.end as jest.Mock)('suffix');

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        13 // 'prefix-suffix'
      );
    });

    it('should call original end and return its result', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      const result = (mockRes.end as jest.Mock)('data');

      expect(originalEnd).toHaveBeenCalled();
      expect(result).toBe(mockRes);
    });

    it('should use encoding parameter for string in end', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.end as jest.Mock)('Ü', 'utf8');

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        2 // Ü in UTF-8
      );
    });
  });

  describe('duration calculation', () => {
    it('should record positive duration in seconds', () => {
      mockReq.path = '/api/slow';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/slow',
        200,
        expect.any(Number),
        undefined,
        undefined
      );

      // Duration should be a small positive number (sub-millisecond)
      const duration = mockRecordHttpRequest.mock.calls[0][3];
      expect(duration).toBeGreaterThanOrEqual(0);
      expect(duration).toBeLessThan(1); // Should be well under a second
    });
  });

  describe('status code and method recording', () => {
    it('should record correct status code', () => {
      mockReq.path = '/api/data';
      mockRes.statusCode = 404;

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'GET',
        '/api/data',
        404,
        expect.any(Number),
        undefined,
        undefined
      );
    });

    it('should record POST method', () => {
      mockReq.path = '/api/data';
      mockReq.method = 'POST';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'POST',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined
      );
    });

    it('should record DELETE method', () => {
      mockReq.path = '/api/data';
      mockReq.method = 'DELETE';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);
      (mockRes.end as jest.Mock)();

      expect(mockRecordHttpRequest).toHaveBeenCalledWith(
        'DELETE',
        '/api/data',
        200,
        expect.any(Number),
        undefined,
        undefined
      );
    });
  });

  describe('middleware chain', () => {
    it('should call next() for tracked paths', () => {
      mockReq.path = '/api/data';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should call next() for skipped paths', () => {
      mockReq.path = '/health';

      metricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });
  });
});

describe('safeMetricsMiddleware', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: jest.Mock;
  let consoleErrorSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.clearAllMocks();

    mockReq = {
      path: '/api/games',
      method: 'GET',
      headers: {},
      body: undefined,
    };

    mockRes = {
      statusCode: 200,
      write: jest.fn().mockReturnValue(true),
      end: jest.fn().mockReturnThis(),
    };

    mockNext = jest.fn();
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
  });

  afterEach(() => {
    consoleErrorSpy.mockRestore();
  });

  it('should call metricsMiddleware normally when no error', () => {
    safeMetricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

    expect(mockNext).toHaveBeenCalled();
    expect(consoleErrorSpy).not.toHaveBeenCalled();
  });

  it('should catch errors and still call next()', () => {
    // Make getMetricsService throw
    mockGetMetricsService.mockImplementationOnce(() => {
      throw new Error('Metrics service unavailable');
    });

    safeMetricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

    expect(consoleErrorSpy).toHaveBeenCalledWith('Metrics middleware error:', expect.any(Error));
    expect(mockNext).toHaveBeenCalled();
  });

  it('should log the actual error thrown', () => {
    const testError = new Error('Test error message');
    mockGetMetricsService.mockImplementationOnce(() => {
      throw testError;
    });

    safeMetricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

    expect(consoleErrorSpy).toHaveBeenCalledWith('Metrics middleware error:', testError);
  });

  it('should skip path checking in safe mode too', () => {
    mockReq.path = '/health';

    safeMetricsMiddleware(mockReq as Request, mockRes as Response, mockNext);

    expect(mockNext).toHaveBeenCalled();
    expect(consoleErrorSpy).not.toHaveBeenCalled();
  });
});
