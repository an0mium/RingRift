import { Request, Response, NextFunction } from 'express';
import { logger, getRequestContext, maskHeaders, maskSensitiveData } from '../utils/logger';
import { config } from '../config';

/**
 * Express Request with requestId and user attached by middleware.
 */
interface RequestWithContext extends Request {
  requestId?: string;
  user?: {
    id: string;
    email?: string;
    username?: string;
  };
}

/**
 * Configuration options for the request logger middleware.
 */
export interface RequestLoggerOptions {
  /**
   * Whether to log request bodies. Default: false in production, true in development.
   * Request bodies may contain sensitive data; enable with caution.
   */
  logRequestBody?: boolean;

  /**
   * Whether to log response bodies. Default: false.
   * This significantly increases log volume and may impact performance.
   */
  logResponseBody?: boolean;

  /**
   * Maximum size of body to log (in bytes). Default: 1024 (1KB).
   * Larger bodies will be truncated.
   */
  maxBodySize?: number;

  /**
   * Paths to exclude from logging. Default: ['/health', '/metrics'].
   */
  excludePaths?: string[];

  /**
   * Whether to log headers. Default: false.
   */
  logHeaders?: boolean;
}

const defaultOptions: Required<RequestLoggerOptions> = {
  logRequestBody: !config.isProduction,
  logResponseBody: false,
  maxBodySize: 1024,
  excludePaths: ['/health', '/metrics', '/favicon.ico'],
  logHeaders: false,
};

/**
 * Truncate a string or object to a maximum size for logging.
 */
const truncateBody = (body: unknown, maxSize: number): unknown => {
  if (body === undefined || body === null) {
    return undefined;
  }

  const str = typeof body === 'string' ? body : JSON.stringify(body);
  if (str.length <= maxSize) {
    return body;
  }

  return `[TRUNCATED: ${str.length} bytes, showing first ${maxSize}] ${str.slice(0, maxSize)}...`;
};

/**
 * Get content length from response headers.
 */
const getContentLength = (res: Response): number | undefined => {
  const contentLength = res.get('content-length');
  if (contentLength) {
    const parsed = parseInt(contentLength, 10);
    return isNaN(parsed) ? undefined : parsed;
  }
  return undefined;
};

/**
 * Request logging middleware.
 *
 * Logs incoming requests and their responses with:
 * - Request ID (for correlation)
 * - User ID (if authenticated)
 * - HTTP method and path
 * - Status code
 * - Response time (duration in ms)
 * - Content length
 *
 * Uses structured JSON format suitable for log aggregation systems
 * (ELK, CloudWatch, Datadog, etc.).
 */
export const requestLogger = (options: RequestLoggerOptions = {}) => {
  const opts: Required<RequestLoggerOptions> = { ...defaultOptions, ...options };

  return (req: RequestWithContext, res: Response, next: NextFunction): void => {
    // Skip excluded paths
    if (opts.excludePaths.some((path) => req.path.startsWith(path))) {
      return next();
    }

    const startTime = Date.now();
    const requestId = req.requestId || getRequestContext()?.requestId || 'unknown';

    // Log incoming request
    const requestLog: Record<string, unknown> = {
      type: 'http_request',
      requestId,
      method: req.method,
      path: req.path,
      query: Object.keys(req.query).length > 0 ? req.query : undefined,
      ip: req.ip || req.socket.remoteAddress,
      userAgent: req.get('user-agent'),
    };

    if (opts.logHeaders) {
      requestLog.headers = maskHeaders(req.headers as Record<string, string | string[] | undefined>);
    }

    if (opts.logRequestBody && req.body && Object.keys(req.body).length > 0) {
      requestLog.body = truncateBody(maskSensitiveData(req.body), opts.maxBodySize);
    }

    logger.info('Incoming request', requestLog);

    // Capture response data
    const originalEnd = res.end;
    let responseBody: unknown;

    // Override res.end to capture response body if needed
    if (opts.logResponseBody) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      res.end = function (chunk?: any, encoding?: any, callback?: any): Response {
        if (chunk) {
          try {
            responseBody = typeof chunk === 'string' ? chunk : chunk.toString('utf-8');
          } catch {
            responseBody = '[BINARY_DATA]';
          }
        }
        return originalEnd.call(this, chunk, encoding, callback);
      };
    }

    // Log response when finished
    res.on('finish', () => {
      const duration = Date.now() - startTime;
      const userId = req.user?.id;

      const responseLog: Record<string, unknown> = {
        type: 'http_response',
        requestId,
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        duration,
        contentLength: getContentLength(res),
        userId,
      };

      if (opts.logResponseBody && responseBody) {
        responseLog.body = truncateBody(maskSensitiveData(responseBody), opts.maxBodySize);
      }

      // Log at appropriate level based on status code
      if (res.statusCode >= 500) {
        logger.error('Request completed with error', responseLog);
      } else if (res.statusCode >= 400) {
        logger.warn('Request completed with client error', responseLog);
      } else {
        logger.info('Request completed', responseLog);
      }
    });

    // Handle errors
    res.on('error', (error) => {
      const duration = Date.now() - startTime;
      logger.error('Request error', {
        type: 'http_error',
        requestId,
        method: req.method,
        path: req.path,
        duration,
        error: error.message,
        stack: error.stack,
      });
    });

    next();
  };
};

/**
 * Pre-configured request logger for API routes.
 * Excludes health/metrics endpoints and logs request bodies in development.
 */
export const apiRequestLogger = requestLogger({
  logRequestBody: !config.isProduction,
  logResponseBody: false,
  excludePaths: ['/health', '/metrics', '/favicon.ico'],
  logHeaders: false,
});

/**
 * Debug request logger with full body logging.
 * Use with caution - may log sensitive data.
 */
export const debugRequestLogger = requestLogger({
  logRequestBody: true,
  logResponseBody: true,
  logHeaders: true,
  maxBodySize: 4096,
});