/**
 * ServiceStatusManager Unit Tests
 *
 * Tests for the centralized service status management including:
 * - Service status updates and tracking
 * - Degradation level computation
 * - Event emission (statusChange, degradationLevelChange, serviceRecovered, serviceDown)
 * - Health check registration and polling
 * - HTTP degradation headers
 * - Singleton management
 */

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

import {
  ServiceStatusManager,
  DegradationLevel,
  getServiceStatusManager,
  initServiceStatusManager,
  resetServiceStatusManager,
  type ServiceName,
  type ServiceHealthStatus,
} from '../../../src/server/services/ServiceStatusManager';
import { logger } from '../../../src/server/utils/logger';

describe('ServiceStatusManager', () => {
  let manager: ServiceStatusManager;

  beforeEach(() => {
    jest.clearAllMocks();
    resetServiceStatusManager();
    manager = new ServiceStatusManager();
  });

  afterEach(() => {
    manager.destroy();
  });

  describe('initialization', () => {
    it('should initialize all services with unknown status', () => {
      const status = manager.getSystemStatus();

      expect(status.services.database.status).toBe('unknown');
      expect(status.services.redis.status).toBe('unknown');
      expect(status.services.aiService.status).toBe('unknown');
    });

    it('should start with FULL degradation level', () => {
      expect(manager.getDegradationLevel()).toBe(DegradationLevel.FULL);
    });

    it('should initialize with all services having zero failure count', () => {
      const dbStatus = manager.getServiceStatus('database');
      const redisStatus = manager.getServiceStatus('redis');
      const aiStatus = manager.getServiceStatus('aiService');

      expect(dbStatus?.failureCount).toBe(0);
      expect(redisStatus?.failureCount).toBe(0);
      expect(aiStatus?.failureCount).toBe(0);
    });

    it('should initialize with fallbackActive false for all services', () => {
      const dbStatus = manager.getServiceStatus('database');
      expect(dbStatus?.fallbackActive).toBe(false);
    });
  });

  describe('updateServiceStatus', () => {
    it('should update service status correctly', () => {
      manager.updateServiceStatus('database', 'healthy');

      const status = manager.getServiceStatus('database');
      expect(status?.status).toBe('healthy');
      expect(status?.lastHealthy).toBeDefined();
    });

    it('should clear error when status becomes healthy', () => {
      manager.updateServiceStatus('database', 'unhealthy', 'Connection failed');
      manager.updateServiceStatus('database', 'healthy');

      const status = manager.getServiceStatus('database');
      expect(status?.error).toBeUndefined();
    });

    it('should store error message when unhealthy', () => {
      manager.updateServiceStatus('database', 'unhealthy', 'Connection refused');

      const status = manager.getServiceStatus('database');
      expect(status?.error).toBe('Connection refused');
    });

    it('should store latency when provided', () => {
      manager.updateServiceStatus('database', 'healthy', undefined, 50);

      const status = manager.getServiceStatus('database');
      expect(status?.latencyMs).toBe(50);
    });

    it('should increment failure count when status changes to unhealthy', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('database', 'unhealthy');

      const status = manager.getServiceStatus('database');
      expect(status?.failureCount).toBe(1);
    });

    it('should reset failure count when status becomes healthy', () => {
      manager.updateServiceStatus('database', 'unhealthy');
      manager.updateServiceStatus('database', 'unhealthy');
      manager.updateServiceStatus('database', 'healthy');

      const status = manager.getServiceStatus('database');
      expect(status?.failureCount).toBe(0);
    });

    it('should set fallbackActive when status is not healthy', () => {
      manager.updateServiceStatus('aiService', 'degraded');

      const status = manager.getServiceStatus('aiService');
      expect(status?.fallbackActive).toBe(true);
    });

    it('should clear fallbackActive when status is healthy', () => {
      manager.updateServiceStatus('aiService', 'degraded');
      manager.updateServiceStatus('aiService', 'healthy');

      const status = manager.getServiceStatus('aiService');
      expect(status?.fallbackActive).toBe(false);
    });

    it('should warn and return if service is unknown', () => {
      manager.updateServiceStatus('unknown' as ServiceName, 'healthy');

      expect(logger.warn).toHaveBeenCalledWith(
        'Attempted to update unknown service',
        expect.objectContaining({ service: 'unknown' })
      );
    });
  });

  describe('event emission', () => {
    it('should emit statusChange when status changes', () => {
      const statusChangeHandler = jest.fn();
      manager.on('statusChange', statusChangeHandler);

      manager.updateServiceStatus('database', 'healthy');

      expect(statusChangeHandler).toHaveBeenCalledWith('database', 'unknown', 'healthy');
    });

    it('should emit serviceRecovered when recovering from unhealthy', () => {
      const recoveredHandler = jest.fn();
      manager.on('serviceRecovered', recoveredHandler);

      manager.updateServiceStatus('database', 'unhealthy');
      manager.updateServiceStatus('database', 'healthy');

      expect(recoveredHandler).toHaveBeenCalledWith('database');
    });

    it('should emit serviceRecovered when recovering from degraded', () => {
      const recoveredHandler = jest.fn();
      manager.on('serviceRecovered', recoveredHandler);

      manager.updateServiceStatus('redis', 'degraded');
      manager.updateServiceStatus('redis', 'healthy');

      expect(recoveredHandler).toHaveBeenCalledWith('redis');
    });

    it('should emit serviceDown when service becomes unhealthy', () => {
      const downHandler = jest.fn();
      manager.on('serviceDown', downHandler);

      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('database', 'unhealthy', 'Connection lost');

      expect(downHandler).toHaveBeenCalledWith('database', 'Connection lost');
    });

    it('should emit serviceDown when service becomes degraded from healthy', () => {
      const downHandler = jest.fn();
      manager.on('serviceDown', downHandler);

      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('redis', 'degraded', 'High latency');

      expect(downHandler).toHaveBeenCalledWith('redis', 'High latency');
    });

    it('should emit degradationLevelChange when level changes', () => {
      const levelChangeHandler = jest.fn();
      manager.on('degradationLevelChange', levelChangeHandler);

      manager.updateServiceStatus('database', 'unhealthy');

      expect(levelChangeHandler).toHaveBeenCalledWith(
        DegradationLevel.FULL,
        DegradationLevel.OFFLINE
      );
    });

    it('should not emit statusChange when status is the same', () => {
      const statusChangeHandler = jest.fn();
      manager.updateServiceStatus('database', 'healthy');

      manager.on('statusChange', statusChangeHandler);
      manager.updateServiceStatus('database', 'healthy');

      expect(statusChangeHandler).not.toHaveBeenCalled();
    });
  });

  describe('degradation level computation', () => {
    it('should return OFFLINE when database is unhealthy', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      expect(manager.getDegradationLevel()).toBe(DegradationLevel.OFFLINE);
    });

    it('should return DEGRADED when only redis is unhealthy', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'healthy');

      expect(manager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);
    });

    it('should return DEGRADED when only aiService is unhealthy', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'unhealthy');

      expect(manager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);
    });

    it('should return DEGRADED when database is degraded', () => {
      manager.updateServiceStatus('database', 'degraded');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      expect(manager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);
    });

    it('should return MINIMAL when both non-critical services are down', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'unhealthy');

      expect(manager.getDegradationLevel()).toBe(DegradationLevel.MINIMAL);
    });

    it('should return MINIMAL when database is degraded and a non-critical service is down', () => {
      manager.updateServiceStatus('database', 'degraded');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'healthy');

      expect(manager.getDegradationLevel()).toBe(DegradationLevel.MINIMAL);
    });

    it('should return FULL when all services are healthy', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      expect(manager.getDegradationLevel()).toBe(DegradationLevel.FULL);
    });
  });

  describe('isDegraded', () => {
    it('should return false when level is FULL', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      expect(manager.isDegraded()).toBe(false);
    });

    it('should return true when level is DEGRADED', () => {
      manager.updateServiceStatus('redis', 'unhealthy');

      expect(manager.isDegraded()).toBe(true);
    });

    it('should return true when level is MINIMAL', () => {
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'unhealthy');

      expect(manager.isDegraded()).toBe(true);
    });

    it('should return true when level is OFFLINE', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      expect(manager.isDegraded()).toBe(true);
    });
  });

  describe('isServiceHealthy', () => {
    it('should return true when service is healthy', () => {
      manager.updateServiceStatus('database', 'healthy');

      expect(manager.isServiceHealthy('database')).toBe(true);
    });

    it('should return false when service is unhealthy', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      expect(manager.isServiceHealthy('database')).toBe(false);
    });

    it('should return false when service is degraded', () => {
      manager.updateServiceStatus('redis', 'degraded');

      expect(manager.isServiceHealthy('redis')).toBe(false);
    });

    it('should return false when service status is unknown', () => {
      expect(manager.isServiceHealthy('database')).toBe(false);
    });
  });

  describe('getDegradedServices', () => {
    it('should return empty array when all services are healthy', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      expect(manager.getDegradedServices()).toEqual([]);
    });

    it('should return services that are not healthy', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'degraded');

      const degraded = manager.getDegradedServices();
      expect(degraded).toContain('redis');
      expect(degraded).toContain('aiService');
      expect(degraded).not.toContain('database');
    });

    it('should include unknown status as degraded', () => {
      manager.updateServiceStatus('database', 'healthy');
      // redis and aiService remain 'unknown'

      const degraded = manager.getDegradedServices();
      expect(degraded).toContain('redis');
      expect(degraded).toContain('aiService');
    });
  });

  describe('getSystemStatus', () => {
    it('should return comprehensive system status', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');

      const status = manager.getSystemStatus();

      expect(status.degradationLevel).toBe(DegradationLevel.DEGRADED);
      expect(status.services.database.status).toBe('healthy');
      expect(status.services.redis.status).toBe('unhealthy');
      expect(status.degradedServices).toContain('redis');
      expect(status.timestamp).toBeInstanceOf(Date);
    });
  });

  describe('setFallbackActive', () => {
    it('should set fallback active status', () => {
      manager.setFallbackActive('aiService', true);

      const status = manager.getServiceStatus('aiService');
      expect(status?.fallbackActive).toBe(true);
    });

    it('should clear fallback active status', () => {
      manager.setFallbackActive('aiService', true);
      manager.setFallbackActive('aiService', false);

      const status = manager.getServiceStatus('aiService');
      expect(status?.fallbackActive).toBe(false);
    });

    it('should do nothing for unknown service', () => {
      // Should not throw
      manager.setFallbackActive('unknown' as ServiceName, true);
    });
  });

  describe('getDegradationHeaders', () => {
    it('should return empty object when not degraded', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'healthy');
      manager.updateServiceStatus('aiService', 'healthy');

      const headers = manager.getDegradationHeaders();
      expect(headers).toEqual({});
    });

    it('should return X-Service-Status header when degraded', () => {
      manager.updateServiceStatus('redis', 'unhealthy');

      const headers = manager.getDegradationHeaders();
      expect(headers['X-Service-Status']).toBe('degraded');
    });

    it('should return X-Degraded-Services header listing degraded services', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');
      manager.updateServiceStatus('aiService', 'degraded');

      const headers = manager.getDegradationHeaders();
      expect(headers['X-Degraded-Services']).toContain('redis');
      expect(headers['X-Degraded-Services']).toContain('aiService');
    });

    it('should return offline status for OFFLINE level', () => {
      manager.updateServiceStatus('database', 'unhealthy');

      const headers = manager.getDegradationHeaders();
      expect(headers['X-Service-Status']).toBe('offline');
    });
  });

  describe('health check registration and polling', () => {
    it('should register health check callback', () => {
      const callback = jest.fn().mockResolvedValue({ status: 'healthy' as ServiceHealthStatus });
      manager.registerHealthCheck('database', callback);

      expect(logger.debug).toHaveBeenCalledWith(
        'Health check callback registered',
        expect.objectContaining({ service: 'database' })
      );
    });

    it('should run health checks and update status', async () => {
      const callback = jest.fn().mockResolvedValue({
        status: 'healthy' as ServiceHealthStatus,
        latencyMs: 25,
      });
      manager.registerHealthCheck('database', callback);

      await manager.runHealthChecks();

      expect(callback).toHaveBeenCalled();
      expect(manager.getServiceStatus('database')?.status).toBe('healthy');
      expect(manager.getServiceStatus('database')?.latencyMs).toBe(25);
    });

    it('should handle health check errors', async () => {
      const callback = jest.fn().mockRejectedValue(new Error('Connection refused'));
      manager.registerHealthCheck('database', callback);

      await manager.runHealthChecks();

      expect(manager.getServiceStatus('database')?.status).toBe('unhealthy');
      expect(manager.getServiceStatus('database')?.error).toBe('Connection refused');
    });

    it('should handle non-Error thrown values', async () => {
      const callback = jest.fn().mockRejectedValue('String error');
      manager.registerHealthCheck('database', callback);

      await manager.runHealthChecks();

      expect(manager.getServiceStatus('database')?.status).toBe('unhealthy');
      expect(manager.getServiceStatus('database')?.error).toBe('String error');
    });

    it('should skip services without registered callbacks', async () => {
      // Only register for database
      const callback = jest.fn().mockResolvedValue({ status: 'healthy' as ServiceHealthStatus });
      manager.registerHealthCheck('database', callback);

      await manager.runHealthChecks();

      // Database should be updated, others should remain unknown
      expect(manager.getServiceStatus('database')?.status).toBe('healthy');
      expect(manager.getServiceStatus('redis')?.status).toBe('unknown');
    });
  });

  describe('polling', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('should start polling when enabled', () => {
      const pollingManager = new ServiceStatusManager({
        enablePolling: true,
        pollingIntervalMs: 1000,
      });

      pollingManager.startPolling();

      expect(logger.info).toHaveBeenCalledWith(
        'Service status polling started',
        expect.objectContaining({ intervalMs: 1000 })
      );

      pollingManager.destroy();
    });

    it('should not start polling when disabled', () => {
      const pollingManager = new ServiceStatusManager({
        enablePolling: false,
      });

      pollingManager.startPolling();

      expect(logger.info).not.toHaveBeenCalledWith(
        'Service status polling started',
        expect.anything()
      );

      pollingManager.destroy();
    });

    it('should not start polling twice', () => {
      const pollingManager = new ServiceStatusManager({
        enablePolling: true,
        pollingIntervalMs: 1000,
      });

      pollingManager.startPolling();
      pollingManager.startPolling();

      // Should only log once
      expect(logger.info).toHaveBeenCalledTimes(1);

      pollingManager.destroy();
    });

    it('should stop polling', () => {
      const pollingManager = new ServiceStatusManager({
        enablePolling: true,
        pollingIntervalMs: 1000,
      });

      pollingManager.startPolling();
      pollingManager.stopPolling();

      expect(logger.info).toHaveBeenCalledWith('Service status polling stopped');

      pollingManager.destroy();
    });

    it('stopPolling should be idempotent', () => {
      const pollingManager = new ServiceStatusManager({
        enablePolling: true,
        pollingIntervalMs: 1000,
      });

      pollingManager.stopPolling(); // Should not throw
      pollingManager.stopPolling();

      pollingManager.destroy();
    });
  });

  describe('refresh', () => {
    it('should run health checks and return system status', async () => {
      const callback = jest.fn().mockResolvedValue({ status: 'healthy' as ServiceHealthStatus });
      manager.registerHealthCheck('database', callback);

      const status = await manager.refresh();

      expect(callback).toHaveBeenCalled();
      expect(status.services.database.status).toBe('healthy');
    });
  });

  describe('reset', () => {
    it('should reset all services to unknown status', () => {
      manager.updateServiceStatus('database', 'healthy');
      manager.updateServiceStatus('redis', 'unhealthy');

      manager.reset();

      expect(manager.getServiceStatus('database')?.status).toBe('unknown');
      expect(manager.getServiceStatus('redis')?.status).toBe('unknown');
    });

    it('should reset degradation level to FULL', () => {
      manager.updateServiceStatus('database', 'unhealthy');
      expect(manager.getDegradationLevel()).toBe(DegradationLevel.OFFLINE);

      manager.reset();

      expect(manager.getDegradationLevel()).toBe(DegradationLevel.FULL);
    });

    it('should clear health check callbacks', async () => {
      const callback = jest.fn().mockResolvedValue({ status: 'healthy' as ServiceHealthStatus });
      manager.registerHealthCheck('database', callback);

      manager.reset();
      await manager.runHealthChecks();

      // Callback should not be called after reset
      expect(callback).not.toHaveBeenCalled();
    });
  });

  describe('destroy', () => {
    it('should stop polling', () => {
      const pollingManager = new ServiceStatusManager({
        enablePolling: true,
        pollingIntervalMs: 1000,
      });

      pollingManager.startPolling();
      pollingManager.destroy();

      expect(logger.info).toHaveBeenCalledWith('Service status polling stopped');
    });

    it('should remove all listeners', () => {
      const handler = jest.fn();
      manager.on('statusChange', handler);

      manager.destroy();
      // Manually emit - handler should not be called
      manager.emit('statusChange', 'database', 'unknown', 'healthy');

      expect(handler).not.toHaveBeenCalled();
    });
  });
});

describe('ServiceStatusManager singleton', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    resetServiceStatusManager();
  });

  afterEach(() => {
    resetServiceStatusManager();
  });

  it('should return same instance from getServiceStatusManager', () => {
    const instance1 = getServiceStatusManager();
    const instance2 = getServiceStatusManager();

    expect(instance1).toBe(instance2);
  });

  it('should create new instance with initServiceStatusManager', () => {
    const instance1 = getServiceStatusManager();
    const instance2 = initServiceStatusManager({ pollingIntervalMs: 5000 });

    expect(instance1).not.toBe(instance2);
  });

  it('should destroy old instance when initializing new one', () => {
    const instance1 = getServiceStatusManager();
    const destroySpy = jest.spyOn(instance1, 'destroy');

    initServiceStatusManager();

    expect(destroySpy).toHaveBeenCalled();
  });

  it('should reset singleton instance', () => {
    const instance1 = getServiceStatusManager();
    resetServiceStatusManager();
    const instance2 = getServiceStatusManager();

    expect(instance1).not.toBe(instance2);
  });

  it('resetServiceStatusManager should handle null instance', () => {
    // Call twice - second call should handle null gracefully
    resetServiceStatusManager();
    resetServiceStatusManager();
    // Should not throw
  });
});
