/**
 * Degradation Headers Middleware Unit Tests
 *
 * Tests for the degradation headers middleware including:
 * - Adding degradation headers to responses
 * - Blocking requests in offline mode
 * - Wrapping JSON responses with degradation info
 * - Periodic logging middleware
 * - getDegradationStatus helper
 */

import { Request, Response, NextFunction } from 'express';
import {
  degradationHeadersMiddleware,
  offlineModeMiddleware,
  wrapResponseWithDegradationInfo,
  degradationLoggingMiddleware,
  getDegradationStatus,
} from '../../../src/server/middleware/degradationHeaders';
import {
  ServiceStatusManager,
  DegradationLevel,
  resetServiceStatusManager,
  initServiceStatusManager,
} from '../../../src/server/services/ServiceStatusManager';

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

import { logger } from '../../../src/server/utils/logger';

describe('degradationHeaders middleware', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: jest.Mock;
  let manager: ServiceStatusManager;

  beforeEach(() => {
    jest.clearAllMocks();
    resetServiceStatusManager();
    manager = initServiceStatusManager();

    mockReq = {
      path: '/api/games',
      method: 'GET',
      ip: '127.0.0.1',
    };

    mockRes = {
      setHeader: jest.fn(),
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };

    mockNext = jest.fn();
  });

  afterEach(() => {
    resetServiceStatusManager();
  });

  describe('degradationHeadersMiddleware', () => {
    it('should not add headers when system is fully operational', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      degradationHeadersMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.setHeader).not.toHaveBeenCalled();
      expect(mockNext).toHaveBeenCalled();
    });

    it('should add X-Service-Status header when degraded', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'healthy');

      degradationHeadersMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.setHeader).toHaveBeenCalledWith('X-Service-Status', 'degraded');
      expect(mockNext).toHaveBeenCalled();
    });

    it('should add X-Degraded-Services header listing degraded services', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'degraded');

      degradationHeadersMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.setHeader).toHaveBeenCalledWith(
        'X-Degraded-Services',
        expect.stringContaining('redis')
      );
      expect(mockRes.setHeader).toHaveBeenCalledWith(
        'X-Degraded-Services',
        expect.stringContaining('aiService')
      );
      expect(mockNext).toHaveBeenCalled();
    });

    it('should set offline status when database is unhealthy', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      degradationHeadersMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.setHeader).toHaveBeenCalledWith('X-Service-Status', 'offline');
      expect(mockNext).toHaveBeenCalled();
    });
  });

  describe('offlineModeMiddleware', () => {
    it('should allow requests when system is not offline', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');

      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockRes.status).not.toHaveBeenCalled();
    });

    it('should block requests in offline mode with 503', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.status).toHaveBeenCalledWith(503);
      expect(mockRes.json).toHaveBeenCalledWith({
        success: false,
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Service is temporarily unavailable due to database maintenance',
          retryAfter: 60,
        },
      });
      expect(mockRes.setHeader).toHaveBeenCalledWith('Retry-After', '60');
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should allow health endpoints in offline mode', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      mockReq.path = '/health';
      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockRes.status).not.toHaveBeenCalled();
    });

    it('should allow /ready endpoint in offline mode', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      mockReq.path = '/ready';
      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should allow /api/health endpoint in offline mode', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      mockReq.path = '/api/health';
      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should use custom allowed paths', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      mockReq.path = '/custom/status';
      const middleware = offlineModeMiddleware(['/custom/status']);
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should match path prefixes', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      mockReq.path = '/api/health/detailed';
      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should log warning when blocking request', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(logger.warn).toHaveBeenCalledWith(
        'Request blocked due to offline mode',
        expect.objectContaining({
          path: '/api/games',
          method: 'GET',
        })
      );
    });
  });

  describe('wrapResponseWithDegradationInfo', () => {
    it('should not modify response when system is fully operational', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      const originalJson = jest.fn();
      mockRes.json = originalJson;

      wrapResponseWithDegradationInfo(mockReq as Request, mockRes as Response, mockNext);

      // Call the wrapped json method
      (mockRes.json as jest.Mock)({ success: true, data: 'test' });

      expect(originalJson).toHaveBeenCalledWith({ success: true, data: 'test' });
      expect(mockNext).toHaveBeenCalled();
    });

    it('should add _serviceStatus to response when degraded', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'healthy');

      let capturedBody: unknown;
      const originalJson = jest.fn((body) => {
        capturedBody = body;
        return mockRes;
      });
      mockRes.json = originalJson;

      wrapResponseWithDegradationInfo(mockReq as Request, mockRes as Response, mockNext);

      // Call the wrapped json method
      (mockRes.json as jest.Mock)({ success: true, data: 'test' });

      expect(capturedBody).toEqual({
        success: true,
        data: 'test',
        _serviceStatus: {
          degradationLevel: DegradationLevel.DEGRADED,
          degradedServices: expect.arrayContaining(['redis']),
        },
      });
    });

    it('should not modify null body', () => {
      manager.updateServiceStatus('redis', 'unhealthy');

      const originalJson = jest.fn();
      mockRes.json = originalJson;

      wrapResponseWithDegradationInfo(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.json as jest.Mock)(null);

      expect(originalJson).toHaveBeenCalledWith(null);
    });

    it('should not modify primitive body values', () => {
      manager.updateServiceStatus('redis', 'unhealthy');

      const originalJson = jest.fn();
      mockRes.json = originalJson;

      wrapResponseWithDegradationInfo(mockReq as Request, mockRes as Response, mockNext);

      (mockRes.json as jest.Mock)('string value');

      expect(originalJson).toHaveBeenCalledWith('string value');
    });
  });

  describe('degradationLoggingMiddleware', () => {
    it('should not log when system is fully operational', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      const middleware = degradationLoggingMiddleware(1); // Log every request

      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(logger.info).not.toHaveBeenCalledWith(
        'Service degradation status (periodic log)',
        expect.anything()
      );
      expect(mockNext).toHaveBeenCalled();
    });

    it('should log periodically when degraded', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');

      const middleware = degradationLoggingMiddleware(1); // Log every request

      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(logger.info).toHaveBeenCalledWith(
        'Service degradation status (periodic log)',
        expect.objectContaining({
          degradationLevel: DegradationLevel.DEGRADED,
          requestCount: 1,
        })
      );
    });

    it('should only log every N requests', () => {
      manager.updateServiceStatus('redis', 'unhealthy');

      const middleware = degradationLoggingMiddleware(3);

      // First request - no log
      middleware(mockReq as Request, mockRes as Response, mockNext);
      expect(logger.info).not.toHaveBeenCalledWith(
        'Service degradation status (periodic log)',
        expect.anything()
      );

      // Second request - no log
      middleware(mockReq as Request, mockRes as Response, mockNext);
      expect(logger.info).not.toHaveBeenCalledWith(
        'Service degradation status (periodic log)',
        expect.anything()
      );

      // Third request - should log
      middleware(mockReq as Request, mockRes as Response, mockNext);
      expect(logger.info).toHaveBeenCalledWith(
        'Service degradation status (periodic log)',
        expect.objectContaining({
          requestCount: 3,
        })
      );
    });

    it('should use default interval of 100', () => {
      manager.updateServiceStatus('redis', 'unhealthy');

      const middleware = degradationLoggingMiddleware();

      // 99 requests - no log
      for (let i = 0; i < 99; i++) {
        middleware(mockReq as Request, mockRes as Response, mockNext);
      }
      expect(logger.info).not.toHaveBeenCalledWith(
        'Service degradation status (periodic log)',
        expect.anything()
      );

      // 100th request - should log
      middleware(mockReq as Request, mockRes as Response, mockNext);
      expect(logger.info).toHaveBeenCalledWith(
        'Service degradation status (periodic log)',
        expect.objectContaining({
          requestCount: 100,
        })
      );
    });
  });

  describe('getDegradationStatus', () => {
    it('should return null when system is fully operational', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      const status = getDegradationStatus();

      expect(status).toBeNull();
    });

    it('should return degradation info when system is degraded', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'healthy');

      const status = getDegradationStatus();

      expect(status).toEqual({
        isDegraded: true,
        level: DegradationLevel.DEGRADED,
        services: expect.arrayContaining(['redis']),
      });
    });

    it('should return offline info when database is down', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      const status = getDegradationStatus();

      expect(status).toEqual({
        isDegraded: true,
        level: DegradationLevel.OFFLINE,
        services: expect.arrayContaining(['database']),
      });
    });

    it('should list all degraded services', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'degraded');

      const status = getDegradationStatus();

      expect(status?.services).toContain('redis');
      expect(status?.services).toContain('aiService');
      expect(status?.services).not.toContain('database');
    });
  });
});
