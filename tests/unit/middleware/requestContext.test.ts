/**
 * RequestContext middleware branch coverage tests
 * Tests for src/server/middleware/requestContext.ts
 */

import { Request, Response, NextFunction } from 'express';

// Mock crypto module
jest.mock('crypto', () => ({
  randomUUID: jest.fn(() => 'generated-uuid-1234'),
}));

// Mock logger utils
const mockRunWithContext = jest.fn((_context, callback) => callback());
const mockGetRequestContext = jest.fn();

jest.mock('../../../src/server/utils/logger', () => ({
  runWithContext: (context: unknown, callback: () => void) => mockRunWithContext(context, callback),
  getRequestContext: () => mockGetRequestContext(),
}));

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
      method: 'GET',
      path: '/api/games',
      header: jest.fn(),
    };

    mockRes = {
      locals: {},
      setHeader: jest.fn(),
    };

    mockNext = jest.fn();
  });

  describe('requestContext', () => {
    describe('request ID extraction', () => {
      it('uses X-Request-Id header (lowercase) when present', () => {
        (mockReq.header as jest.Mock).mockImplementation((name: string) => {
          if (name === 'x-request-id') return 'client-request-id-123';
          return '';
        });

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockReq.requestId).toBe('client-request-id-123');
        expect(mockRes.locals!.requestId).toBe('client-request-id-123');
        expect(mockRes.setHeader).toHaveBeenCalledWith('X-Request-Id', 'client-request-id-123');
      });

      it('uses X-Request-Id header (capitalized) when present', () => {
        (mockReq.header as jest.Mock).mockImplementation((name: string) => {
          if (name === 'x-request-id') return '';
          if (name === 'X-Request-Id') return 'capitalized-request-id';
          return '';
        });

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockReq.requestId).toBe('capitalized-request-id');
        expect(mockRes.locals!.requestId).toBe('capitalized-request-id');
      });

      it('generates UUID when X-Request-Id header is absent', () => {
        (mockReq.header as jest.Mock).mockReturnValue('');

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockReq.requestId).toBe('generated-uuid-1234');
        expect(mockRes.locals!.requestId).toBe('generated-uuid-1234');
        expect(mockRes.setHeader).toHaveBeenCalledWith('X-Request-Id', 'generated-uuid-1234');
      });

      it('generates UUID when X-Request-Id header is undefined', () => {
        (mockReq.header as jest.Mock).mockReturnValue(undefined);

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockReq.requestId).toBe('generated-uuid-1234');
      });

      it('trims whitespace from X-Request-Id header', () => {
        (mockReq.header as jest.Mock).mockImplementation((name: string) => {
          if (name === 'x-request-id') return '  trimmed-id  ';
          return '';
        });

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockReq.requestId).toBe('trimmed-id');
      });

      it('generates UUID when X-Request-Id header is only whitespace', () => {
        (mockReq.header as jest.Mock).mockImplementation((name: string) => {
          if (name === 'x-request-id') return '   ';
          return '';
        });

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        // After trim, length is 0, so UUID is generated
        expect(mockReq.requestId).toBe('generated-uuid-1234');
      });
    });

    describe('response header', () => {
      it('sets X-Request-Id response header for downstream clients', () => {
        (mockReq.header as jest.Mock).mockReturnValue('');

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockRes.setHeader).toHaveBeenCalledWith('X-Request-Id', 'generated-uuid-1234');
      });

      it('echoes client-provided request ID back in response', () => {
        (mockReq.header as jest.Mock).mockImplementation((name: string) => {
          if (name === 'x-request-id') return 'echo-this-id';
          return '';
        });

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockRes.setHeader).toHaveBeenCalledWith('X-Request-Id', 'echo-this-id');
      });
    });

    describe('AsyncLocalStorage context', () => {
      it('runs next() within AsyncLocalStorage context', () => {
        (mockReq.header as jest.Mock).mockReturnValue('');

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockRunWithContext).toHaveBeenCalledWith(
          expect.objectContaining({
            requestId: 'generated-uuid-1234',
            method: 'GET',
            path: '/api/games',
            startTime: expect.any(Number),
          }),
          expect.any(Function)
        );
        expect(mockNext).toHaveBeenCalled();
      });

      it('includes request method and path in context', () => {
        const postReq: Partial<RequestWithId> = {
          method: 'POST',
          path: '/api/auth/login',
          header: jest.fn().mockReturnValue(''),
        };

        requestContext(postReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockRunWithContext).toHaveBeenCalledWith(
          expect.objectContaining({
            method: 'POST',
            path: '/api/auth/login',
          }),
          expect.any(Function)
        );
      });

      it('includes startTime timestamp in context', () => {
        (mockReq.header as jest.Mock).mockReturnValue('');
        const beforeTime = Date.now();

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        const afterTime = Date.now();
        const contextArg = mockRunWithContext.mock.calls[0][0];
        expect(contextArg.startTime).toBeGreaterThanOrEqual(beforeTime);
        expect(contextArg.startTime).toBeLessThanOrEqual(afterTime);
      });
    });

    describe('middleware chain', () => {
      it('calls next() to continue middleware chain', () => {
        (mockReq.header as jest.Mock).mockReturnValue('');

        requestContext(mockReq as RequestWithId, mockRes as Response, mockNext);

        expect(mockNext).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('updateContextWithUser', () => {
    it('updates context with userId when context exists', () => {
      const mockContext: {
        requestId: string;
        method: string;
        path: string;
        startTime: number;
        userId?: string;
      } = {
        requestId: 'test-id',
        method: 'GET',
        path: '/',
        startTime: Date.now(),
      };
      mockGetRequestContext.mockReturnValue(mockContext);

      updateContextWithUser('user-123');

      expect(mockContext.userId).toBe('user-123');
    });

    it('does nothing when context is undefined', () => {
      mockGetRequestContext.mockReturnValue(undefined);

      // Should not throw
      expect(() => updateContextWithUser('user-123')).not.toThrow();
    });

    it('does nothing when context is null', () => {
      mockGetRequestContext.mockReturnValue(null);

      // Should not throw
      expect(() => updateContextWithUser('user-123')).not.toThrow();
    });
  });
});
