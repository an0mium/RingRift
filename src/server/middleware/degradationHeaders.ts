/**
 * Degradation Headers Middleware
 *
 * Adds service status headers to API responses when the system is operating
 * in degraded mode. This allows clients to be aware of service limitations
 * and adjust their behavior accordingly.
 *
 * Headers added when degraded:
 * - X-Service-Status: Current degradation level (degraded, minimal, offline)
 * - X-Degraded-Services: Comma-separated list of unavailable services
 */

import { Request, Response, NextFunction } from 'express';
import {
  getServiceStatusManager,
  DegradationLevel,
} from '../services/ServiceStatusManager';
import { logger } from '../utils/logger';

/**
 * Middleware that adds degradation headers to all responses when
 * the system is not operating at full capacity.
 */
export function degradationHeadersMiddleware(
  _req: Request,
  res: Response,
  next: NextFunction
): void {
  const statusManager = getServiceStatusManager();
  const headers = statusManager.getDegradationHeaders();

  // Add any degradation headers to the response
  for (const [key, value] of Object.entries(headers)) {
    res.setHeader(key, value);
  }

  next();
}

/**
 * Middleware that blocks requests when the system is in OFFLINE mode.
 * Returns 503 Service Unavailable with appropriate retry headers.
 *
 * @param allowedPaths - Paths that are still allowed even in offline mode (e.g., health endpoints)
 */
export function offlineModeMiddleware(allowedPaths: string[] = ['/health', '/ready', '/api/health']) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const statusManager = getServiceStatusManager();
    const degradationLevel = statusManager.getDegradationLevel();

    // Check if path is allowed
    const isAllowedPath = allowedPaths.some((path) => req.path.startsWith(path));

    if (degradationLevel === DegradationLevel.OFFLINE && !isAllowedPath) {
      logger.warn('Request blocked due to offline mode', {
        path: req.path,
        method: req.method,
        ip: req.ip,
      });

      res.status(503).json({
        success: false,
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Service is temporarily unavailable due to database maintenance',
          retryAfter: 60,
        },
      });
      res.setHeader('Retry-After', '60');
      return;
    }

    next();
  };
}

/**
 * Express response wrapper that adds degradation info to JSON responses.
 * This is an alternative to header-based degradation info.
 */
export function wrapResponseWithDegradationInfo(
  _req: Request,
  res: Response,
  next: NextFunction
): void {
  const originalJson = res.json.bind(res);

  res.json = function (body: unknown): Response {
    const statusManager = getServiceStatusManager();
    const systemStatus = statusManager.getSystemStatus();

    // Only add degradation info if we're not at full capacity
    if (systemStatus.degradationLevel !== DegradationLevel.FULL && typeof body === 'object' && body !== null) {
      const bodyWithStatus = {
        ...(body as object),
        _serviceStatus: {
          degradationLevel: systemStatus.degradationLevel,
          degradedServices: systemStatus.degradedServices,
        },
      };
      return originalJson(bodyWithStatus);
    }

    return originalJson(body);
  };

  next();
}

/**
 * Get a middleware that logs degradation status periodically.
 * Useful for debugging in development.
 */
export function degradationLoggingMiddleware(logEveryNRequests: number = 100) {
  let requestCount = 0;

  return (_req: Request, _res: Response, next: NextFunction): void => {
    requestCount++;

    if (requestCount % logEveryNRequests === 0) {
      const statusManager = getServiceStatusManager();
      const systemStatus = statusManager.getSystemStatus();

      if (systemStatus.degradationLevel !== DegradationLevel.FULL) {
        logger.info('Service degradation status (periodic log)', {
          degradationLevel: systemStatus.degradationLevel,
          degradedServices: systemStatus.degradedServices,
          requestCount,
        });
      }
    }

    next();
  };
}

/**
 * Helper to get degradation status for inclusion in API responses.
 * Can be called directly from route handlers.
 */
export function getDegradationStatus(): {
  isDegraded: boolean;
  level: DegradationLevel;
  services: string[];
} | null {
  const statusManager = getServiceStatusManager();
  const systemStatus = statusManager.getSystemStatus();

  if (systemStatus.degradationLevel === DegradationLevel.FULL) {
    return null;
  }

  return {
    isDegraded: true,
    level: systemStatus.degradationLevel,
    services: systemStatus.degradedServices,
  };
}

export default {
  degradationHeadersMiddleware,
  offlineModeMiddleware,
  wrapResponseWithDegradationInfo,
  degradationLoggingMiddleware,
  getDegradationStatus,
};