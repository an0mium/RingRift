/**
 * Request Context Middleware Unit Tests
 *
 * Tests the request context middleware including:
 * - Request ID generation and propagation
 * - X-Request-Id header handling
 * - AsyncLocalStorage context establishment
 */

// Mock the logger module
const mockRunWithContext = jest.fn((context, callback) => callback());
const mockGetRequestContext = jest.fn();

jest.mock('../../../src/server/utils/logger', () => ({
  runWithContext: mockRunWithContext,
  getRequestContext: mockGetRequestContext,
  logger: {
    info: jest.fn(),
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

import { Request, Response, NextFunction } from 'express';
import {
  requestContext,
  updateContextWithUser,
  RequestWithId,
} from '../../../src/server/middleware/requestContext';

describe('requestContext middleware', () => {
  let mockReq: Partial<RequestWithId>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;

  beforeEach(() => {
    jest.clearAllMocks();

    mockReq = {
      header: jest.fn(),
      method: 'GET',
      path: '/api/test',
    };

    mockRes = {
      locals: {},
      setHeader: jest.fn(),
    };

    mockNext = jest.fn();
  });

  describe('request ID handling', () => {
    it('generates UUID when no X-Request-Id header present', () => {
      (mockReq.header as jest.Mock).mockReturnValue('');

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockReq.requestId).toBeDefined();
      expect(mockReq.requestId).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
      );
    });

    it('uses X-Request-Id header when provided (lowercase)', () => {
      (mockReq.header as jest.Mock).mockImplementation((name: string) => {
        if (name === 'x-request-id') return 'custom-request-id-123';
        return '';
      });

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockReq.requestId).toBe('custom-request-id-123');
    });

    it('uses X-Request-Id header when provided (uppercase)', () => {
      (mockReq.header as jest.Mock).mockImplementation((name: string) => {
        if (name === 'X-Request-Id') return 'uppercase-request-id-456';
        if (name === 'x-request-id') return '';
        return '';
      });

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockReq.requestId).toBe('uppercase-request-id-456');
    });

    it('trims whitespace from header value', () => {
      (mockReq.header as jest.Mock).mockImplementation((name: string) => {
        if (name === 'x-request-id') return '  trimmed-id  ';
        return '';
      });

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockReq.requestId).toBe('trimmed-id');
    });

    it('generates UUID when header is whitespace only', () => {
      (mockReq.header as jest.Mock).mockImplementation((name: string) => {
        if (name === 'x-request-id') return '   ';
        return '';
      });

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockReq.requestId).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
      );
    });
  });

  describe('response header propagation', () => {
    it('sets X-Request-Id response header', () => {
      (mockReq.header as jest.Mock).mockReturnValue('my-request-id');

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockRes.setHeader).toHaveBeenCalledWith('X-Request-Id', 'my-request-id');
    });

    it('attaches requestId to res.locals', () => {
      (mockReq.header as jest.Mock).mockReturnValue('local-id');

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockRes.locals!.requestId).toBe('local-id');
    });
  });

  describe('AsyncLocalStorage context', () => {
    it('calls runWithContext with correct context', () => {
      (mockReq.header as jest.Mock).mockReturnValue('context-test-id');

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockRunWithContext).toHaveBeenCalledTimes(1);
      expect(mockRunWithContext).toHaveBeenCalledWith(
        expect.objectContaining({
          requestId: 'context-test-id',
          method: 'GET',
          path: '/api/test',
          startTime: expect.any(Number),
        }),
        expect.any(Function)
      );
    });

    it('calls next() within the context', () => {
      (mockReq.header as jest.Mock).mockReturnValue('');

      requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalledTimes(1);
    });
  });
});

describe('updateContextWithUser', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('updates context with userId when context exists', () => {
    const mockContext = {
      requestId: 'test-123',
      method: 'GET',
      path: '/api',
      startTime: Date.now(),
    };
    mockGetRequestContext.mockReturnValue(mockContext);

    updateContextWithUser('user-456');

    expect(mockContext).toHaveProperty('userId', 'user-456');
  });

  it('does nothing when context is null', () => {
    mockGetRequestContext.mockReturnValue(null);

    // Should not throw
    expect(() => updateContextWithUser('user-789')).not.toThrow();
  });
});
