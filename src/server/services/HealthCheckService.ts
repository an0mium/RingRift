/**
 * Health Check Service for container orchestration and load balancer probes.
 *
 * Provides two types of checks:
 * - Liveness: Is the process alive? (minimal checks, fast response)
 * - Readiness: Can the service serve traffic? (checks critical dependencies)
 *
 * For Kubernetes deployments:
 * - /health -> livenessProbe (restart dead containers)
 * - /ready -> readinessProbe (remove from service endpoints)
 *
 * Integrated with ServiceStatusManager for centralized service health tracking.
 */

import { checkDatabaseHealth, getDatabaseClient } from '../database/connection';
import { getRedisClient } from '../cache/redis';
import { config } from '../config';
import { logger } from '../utils/logger';
import {
  getServiceStatusManager,
  ServiceHealthStatus,
  ServiceName,
  HealthCheckResult,
} from './ServiceStatusManager';

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

export interface DependencyCheckResult {
  status: HealthStatus;
  latency?: number;
  error?: string;
}

export interface HealthCheckResponse {
  status: HealthStatus;
  timestamp: string;
  version: string;
  uptime: number;
  checks?: {
    database?: DependencyCheckResult;
    redis?: DependencyCheckResult;
    aiService?: DependencyCheckResult;
  };
}

/**
 * Default timeout for individual health checks in milliseconds.
 * Configurable but defaults to 5 seconds per check.
 */
const DEFAULT_CHECK_TIMEOUT_MS = 5000;

/**
 * Process start time for uptime calculation.
 */
const processStartTime = Date.now();

/**
 * Run a health check with a timeout wrapper.
 * Returns error result if the check times out.
 */
async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  checkName: string
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(
        () => reject(new Error(`${checkName} health check timed out after ${timeoutMs}ms`)),
        timeoutMs
      )
    ),
  ]);
}

/**
 * Check database connectivity by running a simple query.
 */
async function checkDatabase(timeoutMs: number): Promise<DependencyCheckResult> {
  const start = Date.now();
  try {
    const client = getDatabaseClient();
    if (!client) {
      return {
        status: 'unhealthy',
        error: 'Database client not initialized',
      };
    }

    const isHealthy = await withTimeout(checkDatabaseHealth(), timeoutMs, 'Database');
    const latency = Date.now() - start;

    if (isHealthy) {
      return { status: 'healthy', latency };
    } else {
      return {
        status: 'unhealthy',
        latency,
        error: 'Database query failed',
      };
    }
  } catch (error) {
    const latency = Date.now() - start;
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      status: 'unhealthy',
      latency,
      error: errorMessage,
    };
  }
}

/**
 * Map HealthStatus to ServiceHealthStatus for ServiceStatusManager integration.
 */
function mapHealthToServiceStatus(status: HealthStatus): ServiceHealthStatus {
  switch (status) {
    case 'healthy':
      return 'healthy';
    case 'degraded':
      return 'degraded';
    case 'unhealthy':
      return 'unhealthy';
    default:
      return 'unknown';
  }
}

/**
 * Update the ServiceStatusManager with a health check result.
 */
function updateServiceStatus(
  serviceName: ServiceName,
  result: DependencyCheckResult
): void {
  try {
    const statusManager = getServiceStatusManager();
    statusManager.updateServiceStatus(
      serviceName,
      mapHealthToServiceStatus(result.status),
      result.error,
      result.latency
    );
  } catch (error) {
    // Don't fail health checks if status manager update fails
    logger.debug('Failed to update service status manager', {
      service: serviceName,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

/**
 * Check Redis connectivity with a PING command.
 * Redis is considered optional - degraded status if unavailable.
 */
async function checkRedis(timeoutMs: number): Promise<DependencyCheckResult> {
  const start = Date.now();
  try {
    const client = getRedisClient();
    if (!client) {
      // Redis not connected - this is acceptable in development
      return {
        status: 'degraded',
        error: 'Redis client not connected',
      };
    }

    const pingPromise = client.ping();
    const result = await withTimeout(pingPromise, timeoutMs, 'Redis');
    const latency = Date.now() - start;

    if (result === 'PONG') {
      return { status: 'healthy', latency };
    } else {
      return {
        status: 'degraded',
        latency,
        error: `Unexpected PING response: ${result}`,
      };
    }
  } catch (error) {
    const latency = Date.now() - start;
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      status: 'degraded',
      latency,
      error: errorMessage,
    };
  }
}

/**
 * Check AI service availability.
 * AI service is optional - degraded status if unavailable (game falls back to local heuristics).
 */
async function checkAIService(timeoutMs: number): Promise<DependencyCheckResult> {
  const start = Date.now();
  try {
    const aiServiceUrl = config.aiService.url;
    if (!aiServiceUrl) {
      return {
        status: 'degraded',
        error: 'AI service URL not configured',
      };
    }

    // Use a simple health check endpoint on the AI service
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${aiServiceUrl}/health`, {
        method: 'GET',
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      const latency = Date.now() - start;

      if (response.ok) {
        return { status: 'healthy', latency };
      } else {
        return {
          status: 'degraded',
          latency,
          error: `AI service returned status ${response.status}`,
        };
      }
    } catch (fetchError) {
      clearTimeout(timeoutId);
      throw fetchError;
    }
  } catch (error) {
    const latency = Date.now() - start;
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      status: 'degraded',
      latency,
      error: errorMessage,
    };
  }
}

/**
 * Compute overall health status from individual dependency checks.
 * - unhealthy: if any critical dependency (database) is unhealthy
 * - degraded: if any non-critical dependency (redis, aiService) is unhealthy/degraded
 * - healthy: all checks passed
 */
function computeOverallStatus(checks: {
  database?: DependencyCheckResult;
  redis?: DependencyCheckResult;
  aiService?: DependencyCheckResult;
}): HealthStatus {
  // Database is critical - if it's down, the service is unhealthy
  if (checks.database?.status === 'unhealthy') {
    return 'unhealthy';
  }

  // Check for any degraded status
  const isDegraded =
    checks.database?.status === 'degraded' ||
    checks.redis?.status === 'unhealthy' ||
    checks.redis?.status === 'degraded' ||
    checks.aiService?.status === 'unhealthy' ||
    checks.aiService?.status === 'degraded';

  return isDegraded ? 'degraded' : 'healthy';
}

/**
 * Perform a simple liveness check.
 * This should be fast and only verify the process is responsive.
 * Used by orchestrators to restart dead containers.
 */
export function getLivenessStatus(): HealthCheckResponse {
  return {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: config.app.version,
    uptime: Math.floor((Date.now() - processStartTime) / 1000),
  };
}

/**
 * Perform a full readiness check of all dependencies.
 * Returns detailed status of each dependency for debugging.
 * Used by orchestrators to remove unhealthy instances from rotation.
 */
export async function getReadinessStatus(
  options: { timeoutMs?: number; includeAIService?: boolean } = {}
): Promise<HealthCheckResponse> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_CHECK_TIMEOUT_MS;
  const includeAIService = options.includeAIService ?? true;

  // Run checks in parallel for faster response
  const checkPromises: Promise<[string, DependencyCheckResult]>[] = [
    checkDatabase(timeoutMs).then((result) => ['database', result] as [string, DependencyCheckResult]),
    checkRedis(timeoutMs).then((result) => ['redis', result] as [string, DependencyCheckResult]),
  ];

  if (includeAIService) {
    checkPromises.push(
      checkAIService(timeoutMs).then((result) => ['aiService', result] as [string, DependencyCheckResult])
    );
  }

  const results = await Promise.all(checkPromises);

  const checks: {
    database?: DependencyCheckResult;
    redis?: DependencyCheckResult;
    aiService?: DependencyCheckResult;
  } = {};

  for (const [name, result] of results) {
    if (name === 'database') {
      checks.database = result;
      updateServiceStatus('database', result);
    } else if (name === 'redis') {
      checks.redis = result;
      updateServiceStatus('redis', result);
    } else if (name === 'aiService') {
      checks.aiService = result;
      updateServiceStatus('aiService', result);
    }
  }

  const status = computeOverallStatus(checks);

  // Log health check results at appropriate level (avoid excessive noise)
  if (status === 'unhealthy') {
    logger.warn('Readiness check failed', {
      event: 'health_check_unhealthy',
      checks,
    });
  } else if (status === 'degraded') {
    // Log degraded at debug level to reduce noise - this is expected in some environments
    logger.debug('Readiness check degraded', {
      event: 'health_check_degraded',
      checks,
    });
  }

  return {
    status,
    timestamp: new Date().toISOString(),
    version: config.app.version,
    uptime: Math.floor((Date.now() - processStartTime) / 1000),
    checks,
  };
}

/**
 * Check if the service is ready to serve traffic (for HTTP response code).
 * Returns true if status is 'healthy' or 'degraded' (can still serve with degraded dependencies).
 * Returns false only if critical dependencies (database) are unavailable.
 */
export function isServiceReady(response: HealthCheckResponse): boolean {
  return response.status !== 'unhealthy';
}

/**
 * Register health check callbacks with the ServiceStatusManager.
 * This enables the status manager to poll health checks independently.
 */
export function registerHealthChecksWithStatusManager(timeoutMs: number = DEFAULT_CHECK_TIMEOUT_MS): void {
  try {
    const statusManager = getServiceStatusManager();

    // Register database health check
    statusManager.registerHealthCheck('database', async (): Promise<HealthCheckResult> => {
      const result = await checkDatabase(timeoutMs);
      return {
        status: mapHealthToServiceStatus(result.status),
        error: result.error,
        latencyMs: result.latency,
      };
    });

    // Register Redis health check
    statusManager.registerHealthCheck('redis', async (): Promise<HealthCheckResult> => {
      const result = await checkRedis(timeoutMs);
      return {
        status: mapHealthToServiceStatus(result.status),
        error: result.error,
        latencyMs: result.latency,
      };
    });

    // Register AI service health check
    statusManager.registerHealthCheck('aiService', async (): Promise<HealthCheckResult> => {
      const result = await checkAIService(timeoutMs);
      return {
        status: mapHealthToServiceStatus(result.status),
        error: result.error,
        latencyMs: result.latency,
      };
    });

    logger.info('Health checks registered with ServiceStatusManager');
  } catch (error) {
    logger.warn('Failed to register health checks with ServiceStatusManager', {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

export const HealthCheckService = {
  getLivenessStatus,
  getReadinessStatus,
  isServiceReady,
  registerHealthChecksWithStatusManager,
};